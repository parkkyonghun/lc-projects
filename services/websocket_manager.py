import asyncio
import json
from typing import Dict, List, Set, Optional, Any
from datetime import datetime
import uuid
from fastapi import WebSocket, WebSocketDisconnect
import redis.asyncio as redis
from config.settings import settings


class ConnectionManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        # Store active connections by user_id
        self.active_connections: Dict[str, Set[WebSocket]] = {}
        # Store connection metadata
        self.connection_metadata: Dict[WebSocket, Dict[str, Any]] = {}
        self.redis_client: Optional[redis.Redis] = None
        
    async def init_redis(self):
        """Initialize Redis connection for pub/sub"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True
            )
    
    async def connect(self, websocket: WebSocket, user_id: str, client_info: Dict[str, Any] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        # Add to active connections
        if user_id not in self.active_connections:
            self.active_connections[user_id] = set()
        self.active_connections[user_id].add(websocket)
        
        # Store metadata
        self.connection_metadata[websocket] = {
            "user_id": user_id,
            "connected_at": datetime.utcnow(),
            "client_info": client_info or {},
            "connection_id": str(uuid.uuid4())
        }
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "message": "Connected to real-time updates",
            "connection_id": self.connection_metadata[websocket]["connection_id"],
            "timestamp": datetime.utcnow().isoformat()
        }, websocket)
        
        print(f"User {user_id} connected via WebSocket")
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        if websocket in self.connection_metadata:
            user_id = self.connection_metadata[websocket]["user_id"]
            
            # Remove from active connections
            if user_id in self.active_connections:
                self.active_connections[user_id].discard(websocket)
                if not self.active_connections[user_id]:
                    del self.active_connections[user_id]
            
            # Remove metadata
            del self.connection_metadata[websocket]
            
            print(f"User {user_id} disconnected from WebSocket")
    
    async def send_personal_message(self, message: Dict[str, Any], websocket: WebSocket):
        """Send a message to a specific WebSocket connection"""
        try:
            await websocket.send_text(json.dumps(message))
        except Exception as e:
            print(f"Error sending message to WebSocket: {e}")
            self.disconnect(websocket)
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send a message to all connections of a specific user"""
        if user_id in self.active_connections:
            disconnected_connections = []
            
            for websocket in self.active_connections[user_id].copy():
                try:
                    await websocket.send_text(json.dumps(message))
                except Exception as e:
                    print(f"Error sending message to user {user_id}: {e}")
                    disconnected_connections.append(websocket)
            
            # Clean up disconnected connections
            for websocket in disconnected_connections:
                self.disconnect(websocket)
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast a message to all connected users"""
        for user_id in list(self.active_connections.keys()):
            await self.send_to_user(user_id, message)
    
    async def send_to_multiple_users(self, user_ids: List[str], message: Dict[str, Any]):
        """Send a message to multiple specific users"""
        for user_id in user_ids:
            await self.send_to_user(user_id, message)
    
    def get_connected_users(self) -> List[str]:
        """Get list of currently connected user IDs"""
        return list(self.active_connections.keys())
    
    def get_connection_count(self) -> int:
        """Get total number of active connections"""
        return sum(len(connections) for connections in self.active_connections.values())
    
    def get_user_connection_count(self, user_id: str) -> int:
        """Get number of connections for a specific user"""
        return len(self.active_connections.get(user_id, set()))


class NotificationManager:
    """Manages real-time notifications and updates"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.redis_client: Optional[redis.Redis] = None
        
    async def init_redis(self):
        """Initialize Redis connection"""
        if not self.redis_client:
            self.redis_client = redis.from_url(
                settings.redis_url,
                password=settings.redis_password,
                decode_responses=True
            )
    
    async def notify_loan_status_change(self, loan_id: str, old_status: str, new_status: str, 
                                       customer_id: str, updated_by: str):
        """Notify about loan status changes"""
        message = {
            "type": "loan_status_change",
            "loan_id": loan_id,
            "old_status": old_status,
            "new_status": new_status,
            "customer_id": customer_id,
            "updated_by": updated_by,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to customer
        await self.connection_manager.send_to_user(customer_id, message)
        
        # Send to all managers/officers (broadcast to all for now)
        await self.connection_manager.broadcast_to_all({
            **message,
            "type": "loan_status_update_notification"
        })
    
    async def notify_new_application(self, loan_id: str, customer_id: str, customer_name: str):
        """Notify about new loan applications"""
        message = {
            "type": "new_loan_application",
            "loan_id": loan_id,
            "customer_id": customer_id,
            "customer_name": customer_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Broadcast to all managers/officers
        await self.connection_manager.broadcast_to_all(message)
    
    async def notify_sync_status(self, user_id: str, entity_type: str, entity_id: str, 
                                sync_status: str, error_message: Optional[str] = None):
        """Notify about sync status changes"""
        message = {
            "type": "sync_status_update",
            "entity_type": entity_type,
            "entity_id": entity_id,
            "sync_status": sync_status,
            "error_message": error_message,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.send_to_user(user_id, message)
    
    async def notify_comment_added(self, loan_id: str, comment_id: str, author_id: str, 
                                  author_name: str, customer_id: str):
        """Notify about new comments on loans"""
        message = {
            "type": "comment_added",
            "loan_id": loan_id,
            "comment_id": comment_id,
            "author_id": author_id,
            "author_name": author_name,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        # Send to customer and other officers viewing the loan
        await self.connection_manager.send_to_user(customer_id, message)
        
        # For now, broadcast to all (in production, you'd track who's viewing what)
        await self.connection_manager.broadcast_to_all({
            **message,
            "type": "loan_comment_notification"
        })
    
    async def notify_system_maintenance(self, message: str, start_time: datetime, 
                                       end_time: Optional[datetime] = None):
        """Notify about system maintenance"""
        notification = {
            "type": "system_maintenance",
            "message": message,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat() if end_time else None,
            "timestamp": datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.broadcast_to_all(notification)
    
    async def send_heartbeat(self):
        """Send heartbeat to all connections"""
        message = {
            "type": "heartbeat",
            "timestamp": datetime.utcnow().isoformat(),
            "server_time": datetime.utcnow().isoformat()
        }
        
        await self.connection_manager.broadcast_to_all(message)


# Global instances
connection_manager = ConnectionManager()
notification_manager = NotificationManager(connection_manager)


async def websocket_heartbeat():
    """Background task to send periodic heartbeats"""
    while True:
        await asyncio.sleep(settings.websocket_heartbeat_interval)
        await notification_manager.send_heartbeat()