# Loan Management API with Real-time Sync

A comprehensive FastAPI-based loan management system with real-time synchronization, WebSocket notifications, and robust offline capabilities.

## 🚀 Features

### Core Functionality
- **User Management**: Complete user lifecycle with authentication
- **Loan Processing**: Full loan management with status tracking
- **Payment Tracking**: Payment history and management
- **Dashboard Analytics**: Real-time statistics and insights

### API Integration Features
- **Real-time Synchronization**: Automatic sync with remote servers
- **Offline Support**: Queue operations when offline, sync when online
- **Conflict Resolution**: Handle data conflicts intelligently
- **WebSocket Notifications**: Real-time updates for all clients
- **Network Monitoring**: Adaptive sync based on connection quality
- **Retry Mechanisms**: Robust error handling and retry logic

## 📋 Requirements

- Python 3.8+
- PostgreSQL 12+
- Redis 6+
- FastAPI
- SQLAlchemy (async)
- WebSockets support

## 🛠 Installation

### Quick Setup
```bash
# Clone the repository
git clone <repository-url>
cd lc-projects

# Run the setup script
python setup.py
```

### Manual Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Create environment file
cp .env.example .env
# Edit .env with your configuration

# Run database migrations
python migrations/add_sync_fields.py

# Start the application
uvicorn main:app --reload
```

## ⚙️ Configuration

### Environment Variables

Create a `.env` file with the following configuration:

```env
# Database Configuration
DATABASE_URL=postgresql+asyncpg://username:password@localhost:5432/loan_management

# JWT Configuration
JWT_SECRET_KEY=your-super-secret-jwt-key-change-this-in-production
JWT_ALGORITHM=HS256
JWT_ACCESS_TOKEN_EXPIRE_MINUTES=30
JWT_REFRESH_TOKEN_EXPIRE_DAYS=7

# Redis Configuration
REDIS_URL=redis://localhost:6379/0

# API Configuration
API_TITLE=Loan Management API
API_DESCRIPTION=API for managing loans with real-time sync capabilities
API_VERSION=1.0.0

# Sync Configuration
SYNC_BATCH_SIZE=50
SYNC_RETRY_ATTEMPTS=3
SYNC_RETRY_DELAY=5
SYNC_TIMEOUT=30

# File Upload Configuration
MAX_FILE_SIZE=10485760
ALLOWED_FILE_TYPES=pdf,doc,docx,jpg,jpeg,png

# WebSocket Configuration
WS_HEARTBEAT_INTERVAL=30
WS_MAX_CONNECTIONS=1000

# Security Configuration
BCRYPT_ROUNDS=12
SESSION_TIMEOUT=3600
```

## 🏗 Architecture

### Project Structure
```
lc-projects/
├── config/                 # Configuration management
│   ├── __init__.py
│   └── settings.py
├── controllers/            # Business logic controllers
├── models/                 # Database models
│   ├── user.py            # User model with sync fields
│   └── loan.py            # Loan model with sync fields
├── schemas/               # Pydantic schemas
│   └── dto.py             # Data Transfer Objects
├── services/              # Core services
│   ├── auth_manager.py    # Authentication service
│   ├── sync_manager.py    # Synchronization service
│   ├── connectivity_monitor.py  # Network monitoring
│   ├── websocket_manager.py     # WebSocket management
│   └── application_repository.py # Data repository
├── views/                 # API endpoints
│   ├── auth_view.py       # Authentication endpoints
│   ├── sync_view.py       # Sync management endpoints
│   ├── websocket_view.py  # WebSocket endpoints
│   ├── user_view.py       # User management endpoints
│   ├── loan_view.py       # Loan management endpoints
│   ├── payment_view.py    # Payment endpoints
│   └── dashboard_view.py  # Dashboard endpoints
├── migrations/            # Database migrations
├── docs/                  # Documentation
├── main.py               # FastAPI application
├── requirements.txt      # Python dependencies
└── setup.py             # Setup script
```

### Key Components

#### 1. Authentication Manager (`services/auth_manager.py`)
- JWT token management (access & refresh tokens)
- Password hashing and verification
- User session management with Redis
- Token blacklisting for security

#### 2. Sync Manager (`services/sync_manager.py`)
- Orchestrates data synchronization
- Handles batch operations
- Manages retry logic
- Conflict resolution

#### 3. Connectivity Monitor (`services/connectivity_monitor.py`)
- Network status monitoring
- Connection quality assessment
- Adaptive sync recommendations
- Bandwidth and latency tracking

#### 4. WebSocket Manager (`services/websocket_manager.py`)
- Real-time connection management
- Notification broadcasting
- User presence tracking
- Message routing

#### 5. Application Repository (`services/application_repository.py`)
- Unified data access layer
- Automatic sync queuing
- CRUD operations with sync support
- Transaction management

## 🔌 API Endpoints

### Authentication
- `POST /auth/login` - User login
- `POST /auth/refresh` - Refresh access token
- `POST /auth/logout` - User logout
- `GET /auth/me` - Get current user

### User Management
- `GET /users/` - List all users
- `GET /users/me` - Get current user details
- `GET /users/{user_id}` - Get user by ID
- `PUT /users/{user_id}` - Update user
- `DELETE /users/{user_id}` - Delete user

### Loan Management
- `POST /loans/` - Create new loan
- `GET /loans/` - List all loans
- `GET /loans/{loan_id}` - Get loan by ID
- `PUT /loans/{loan_id}` - Update loan
- `DELETE /loans/{loan_id}` - Delete loan
- `GET /loans/customer/{customer_id}` - Get loans by customer
- `GET /loans/status/{status}` - Get loans by status

### Payment Management
- `POST /payments/` - Create payment
- `GET /payments/` - List all payments
- `GET /payments/{payment_id}` - Get payment by ID
- `PUT /payments/{payment_id}` - Update payment
- `DELETE /payments/{payment_id}` - Delete payment
- `GET /payments/loan/{loan_id}` - Get payments by loan

### Synchronization
- `GET /sync/status/{entity_type}/{entity_id}` - Get sync status
- `GET /sync/pending-count` - Get pending sync count
- `POST /sync/force/{entity_type}/{entity_id}` - Force sync
- `POST /sync/batch` - Batch synchronization
- `GET /sync/network-status` - Network connectivity status
- `GET /sync/failed-items` - Get failed sync items
- `POST /sync/retry-failed` - Retry failed syncs

### Dashboard
- `GET /dashboard/stats` - Dashboard statistics
- `GET /dashboard/network-status` - Network status
- `GET /dashboard/sync-summary` - Sync summary

### WebSocket
- `WS /ws/connect/{user_id}` - WebSocket connection
- `GET /ws/active-connections` - Active connections count

### System
- `GET /` - API information
- `GET /health` - Health check
- `GET /docs` - API documentation

## 🔄 Synchronization Features

### Sync Status Types
- `pending` - Waiting to be synchronized
- `syncing` - Currently being synchronized
- `synced` - Successfully synchronized
- `failed` - Synchronization failed
- `conflict` - Conflict detected, needs resolution

### Conflict Resolution
The system handles conflicts using:
- Version numbers for optimistic locking
- Last-modified timestamps
- User-defined resolution strategies
- Manual conflict resolution interface

### Offline Support
- Operations are queued when offline
- Automatic sync when connection is restored
- Intelligent batching based on connection quality
- Retry mechanisms with exponential backoff

## 📡 WebSocket Notifications

### Notification Types
- `loan_status_change` - Loan status updates
- `new_application` - New loan applications
- `sync_status` - Synchronization updates
- `comment_added` - New comments
- `system_maintenance` - System notifications
- `heartbeat` - Connection health checks

### Client Integration
```javascript
const ws = new WebSocket('ws://localhost:8000/ws/connect/user123');

ws.onmessage = function(event) {
    const notification = JSON.parse(event.data);
    console.log('Notification:', notification);
};

// Subscribe to specific notification types
ws.send(JSON.stringify({
    type: 'subscribe',
    notification_types: ['loan_status_change', 'new_application']
}));
```

## 🧪 Testing

### Run Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio httpx

# Run all tests
pytest

# Run with coverage
pytest --cov=.

# Run specific test file
pytest tests/test_sync_manager.py
```

### Test WebSocket Connection
```bash
# Using wscat (install with: npm install -g wscat)
wscat -c ws://localhost:8000/ws/connect/test-user
```

## 🚀 Deployment

### Development
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Production
```bash
# Using Gunicorn with Uvicorn workers
gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Using Docker
docker build -t loan-management-api .
docker run -p 8000:8000 loan-management-api
```

### Environment Setup
1. Set up PostgreSQL database
2. Configure Redis server
3. Set environment variables
4. Run database migrations
5. Start the application

## 📊 Monitoring

### Health Checks
- `GET /health` - Overall system health
- Redis connectivity
- Database connectivity
- WebSocket connection count
- Network status

### Logging
The application logs important events:
- Authentication attempts
- Sync operations
- WebSocket connections
- Error conditions
- Performance metrics

## 🔒 Security

### Authentication
- JWT-based authentication
- Refresh token rotation
- Token blacklisting
- Session management

### Data Protection
- Password hashing with bcrypt
- SQL injection prevention
- Input validation
- CORS configuration

### API Security
- Rate limiting (recommended)
- Request validation
- Error handling
- Secure headers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings
- Write tests

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

For support and questions:
1. Check the documentation in `docs/`
2. Review the API documentation at `/docs`
3. Check existing issues
4. Create a new issue with detailed information

## 🔄 Migration from Previous Version

If upgrading from a previous version:

1. Backup your database
2. Run the migration script: `python migrations/add_sync_fields.py`
3. Update your environment configuration
4. Test the new features
5. Update client applications to use new endpoints

## 📈 Performance Considerations

- Use connection pooling for database
- Configure Redis for optimal performance
- Monitor WebSocket connection limits
- Implement caching strategies
- Use async operations throughout
- Monitor sync queue sizes
- Optimize batch sizes based on network conditions
