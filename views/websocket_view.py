from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends
from typing import Dict, Any
import json
import asyncio

from services.websocket_manager import connection_manager, notification_manager
from services.auth_manager import auth_manager
from models.user import User


router = APIRouter(prefix="/ws", tags=["websocket"])


@router.websocket("/connect/{user_id}")
async def websocket_endpoint(websocket: WebSocket, user_id: str):
    """WebSocket endpoint for real-time communication"""
    await connection_manager.connect(websocket, user_id)
    
    try:
        # Send initial connection confirmation
        await connection_manager.send_personal_message(
            user_id,
            {
                "type": "connection_established",
                "message": "Connected to real-time updates",
                "user_id": user_id
            }
        )
        
        # Start heartbeat
        heartbeat_task = asyncio.create_task(
            _send_heartbeat(user_id)
        )
        
        while True:
            # Listen for messages from client
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Handle different message types
            await _handle_websocket_message(user_id, message)
            
    except WebSocketDisconnect:
        connection_manager.disconnect(user_id)
        heartbeat_task.cancel()
    except Exception as e:
        print(f"WebSocket error for user {user_id}: {e}")
        connection_manager.disconnect(user_id)
        if 'heartbeat_task' in locals():
            heartbeat_task.cancel()


async def _send_heartbeat(user_id: str):
    """Send periodic heartbeat to keep connection alive"""
    while True:
        try:
            await asyncio.sleep(30)  # Send heartbeat every 30 seconds
            await notification_manager.send_heartbeat(user_id)
        except Exception as e:
            print(f"Heartbeat error for user {user_id}: {e}")
            break


async def _handle_websocket_message(user_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages from client"""
    message_type = message.get("type")
    
    if message_type == "ping":
        # Respond to ping with pong
        await connection_manager.send_personal_message(
            user_id,
            {"type": "pong", "timestamp": message.get("timestamp")}
        )
    
    elif message_type == "subscribe":
        # Handle subscription to specific events
        event_types = message.get("events", [])
        await connection_manager.send_personal_message(
            user_id,
            {
                "type": "subscription_confirmed",
                "events": event_types,
                "message": f"Subscribed to {len(event_types)} event types"
            }
        )
    
    elif message_type == "unsubscribe":
        # Handle unsubscription from events
        event_types = message.get("events", [])
        await connection_manager.send_personal_message(
            user_id,
            {
                "type": "unsubscription_confirmed",
                "events": event_types,
                "message": f"Unsubscribed from {len(event_types)} event types"
            }
        )
    
    elif message_type == "status_request":
        # Send current status information
        active_connections = len(connection_manager.active_connections)
        await connection_manager.send_personal_message(
            user_id,
            {
                "type": "status_response",
                "active_connections": active_connections,
                "user_id": user_id,
                "server_time": asyncio.get_event_loop().time()
            }
        )
    
    else:
        # Unknown message type
        await connection_manager.send_personal_message(
            user_id,
            {
                "type": "error",
                "message": f"Unknown message type: {message_type}"
            }
        )


@router.get("/active-connections")
async def get_active_connections():
    """Get count of active WebSocket connections"""
    return {
        "active_connections": len(connection_manager.active_connections),
        "connected_users": list(connection_manager.active_connections.keys())
    }