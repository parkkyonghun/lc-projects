import asyncio
import time
from typing import Dict, List, Callable, Optional
from enum import Enum
import httpx
from datetime import datetime, timedelta
import redis.asyncio as redis
from config.settings import settings


class ConnectionQuality(str, Enum):
    EXCELLENT = "excellent"  # < 100ms, > 10 Mbps
    GOOD = "good"           # < 300ms, > 5 Mbps  
    FAIR = "fair"           # < 500ms, > 1 Mbps
    POOR = "poor"           # < 1000ms, > 0.5 Mbps
    OFFLINE = "offline"     # No connection


class NetworkStatus:
    def __init__(self):
        self.is_connected: bool = False
        self.quality: ConnectionQuality = ConnectionQuality.OFFLINE
        self.latency: float = 0.0  # milliseconds
        self.bandwidth: float = 0.0  # Mbps estimate
        self.last_check: datetime = datetime.utcnow()
        self.consecutive_failures: int = 0


class ConnectivityMonitor:
    """Monitors network connectivity and quality for sync optimization"""
    
    def __init__(self):
        self.status = NetworkStatus()
        self.listeners: List[Callable] = []
        self.redis_client: Optional[redis.Redis] = None
        self.monitoring = False
        self.check_interval = 30  # seconds
        
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True
            )
    
    def add_listener(self, callback: Callable[[NetworkStatus], None]):
        """Add a callback to be notified of connectivity changes"""
        self.listeners.append(callback)
    
    def remove_listener(self, callback: Callable):
        """Remove a connectivity change listener"""
        if callback in self.listeners:
            self.listeners.remove(callback)
    
    async def _notify_listeners(self):
        """Notify all listeners of connectivity changes"""
        for listener in self.listeners:
            try:
                if asyncio.iscoroutinefunction(listener):
                    await listener(self.status)
                else:
                    listener(self.status)
            except Exception as e:
                print(f"Error notifying connectivity listener: {e}")
    
    async def check_connectivity(self) -> NetworkStatus:
        """Check current network connectivity and quality"""
        previous_status = self.status.is_connected
        previous_quality = self.status.quality
        
        try:
            # Test connectivity with a simple HTTP request
            start_time = time.time()
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get("https://httpbin.org/get")
                
            end_time = time.time()
            latency = (end_time - start_time) * 1000  # Convert to milliseconds
            
            if response.status_code == 200:
                self.status.is_connected = True
                self.status.latency = latency
                self.status.consecutive_failures = 0
                
                # Estimate quality based on latency
                if latency < 100:
                    self.status.quality = ConnectionQuality.EXCELLENT
                elif latency < 300:
                    self.status.quality = ConnectionQuality.GOOD
                elif latency < 500:
                    self.status.quality = ConnectionQuality.FAIR
                else:
                    self.status.quality = ConnectionQuality.POOR
            else:
                raise Exception(f"HTTP {response.status_code}")
                
        except Exception as e:
            self.status.is_connected = False
            self.status.quality = ConnectionQuality.OFFLINE
            self.status.latency = 0.0
            self.status.consecutive_failures += 1
            print(f"Connectivity check failed: {e}")
        
        self.status.last_check = datetime.utcnow()
        
        # Store status in Redis for other services
        await self._store_status()
        
        # Notify listeners if status changed
        if (previous_status != self.status.is_connected or 
            previous_quality != self.status.quality):
            await self._notify_listeners()
        
        return self.status
    
    async def _store_status(self):
        """Store current status in Redis"""
        await self.init_redis()
        
        status_data = {
            "is_connected": self.status.is_connected,
            "quality": self.status.quality.value,
            "latency": self.status.latency,
            "last_check": self.status.last_check.isoformat(),
            "consecutive_failures": self.status.consecutive_failures
        }
        
        await self.redis_client.setex(
            "network_status",
            timedelta(minutes=5),  # Expire after 5 minutes
            str(status_data)
        )
    
    async def get_stored_status(self) -> Optional[NetworkStatus]:
        """Get last stored network status from Redis"""
        await self.init_redis()
        
        try:
            status_str = await self.redis_client.get("network_status")
            if status_str:
                import ast
                status_data = ast.literal_eval(status_str)
                
                status = NetworkStatus()
                status.is_connected = status_data["is_connected"]
                status.quality = ConnectionQuality(status_data["quality"])
                status.latency = status_data["latency"]
                status.last_check = datetime.fromisoformat(status_data["last_check"])
                status.consecutive_failures = status_data["consecutive_failures"]
                
                return status
        except Exception as e:
            print(f"Error retrieving stored network status: {e}")
        
        return None
    
    async def start_monitoring(self):
        """Start continuous connectivity monitoring as a background task"""
        if self.monitoring:
            return
        
        self.monitoring = True
        
        # Create a background task instead of blocking
        asyncio.create_task(self._monitoring_loop())
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while self.monitoring:
            try:
                await self.check_connectivity()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                print(f"Error in connectivity monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)
    
    async def stop_monitoring(self):
        """Stop connectivity monitoring"""
        self.monitoring = False
    
    def should_sync(self) -> bool:
        """Determine if sync operations should proceed based on connectivity"""
        if not self.status.is_connected:
            return False
        
        # Don't sync if connection quality is poor and we have many failures
        if (self.status.quality == ConnectionQuality.POOR and 
            self.status.consecutive_failures > 3):
            return False
        
        return True
    
    def get_sync_batch_size(self) -> int:
        """Get recommended batch size based on connection quality"""
        if not self.status.is_connected:
            return 0
        
        quality_batch_sizes = {
            ConnectionQuality.EXCELLENT: settings.sync_batch_size,
            ConnectionQuality.GOOD: settings.sync_batch_size // 2,
            ConnectionQuality.FAIR: settings.sync_batch_size // 4,
            ConnectionQuality.POOR: 1,
            ConnectionQuality.OFFLINE: 0
        }
        
        return quality_batch_sizes.get(self.status.quality, 1)
    
    def get_sync_timeout(self) -> int:
        """Get recommended timeout based on connection quality"""
        if not self.status.is_connected:
            return settings.sync_timeout
        
        quality_timeouts = {
            ConnectionQuality.EXCELLENT: settings.sync_timeout,
            ConnectionQuality.GOOD: settings.sync_timeout * 2,
            ConnectionQuality.FAIR: settings.sync_timeout * 3,
            ConnectionQuality.POOR: settings.sync_timeout * 5,
            ConnectionQuality.OFFLINE: settings.sync_timeout
        }
        
        return quality_timeouts.get(self.status.quality, settings.sync_timeout)


# Global connectivity monitor instance
connectivity_monitor = ConnectivityMonitor()