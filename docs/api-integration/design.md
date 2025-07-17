# API Integration Design Document

## Overview

This design document outlines the architecture and implementation details for integrating the LC Work Flow application with a backend API. The integration will enable data synchronization between the local SQLite database and a remote server, allowing for seamless offline and online operation while ensuring data consistency and security.

The backend will be implemented using the following technology stack:

1. **FastAPI**: A modern, high-performance web framework for building APIs with Python based on standard Python type hints
2. **PostgreSQL**: A powerful, open-source object-relational database system for storing application data
3. **DragonFly DB**: An in-memory data store used for caching and real-time data processing
4. **WebSockets**: For real-time communication between clients and server
5. **Pydantic**: For data validation and settings management
6. **SQLAlchemy**: For ORM and database operations

## Architecture

The API integration will follow a layered architecture with clear separation of concerns:

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Presentation   │     │    Domain       │     │      Data       │
│     Layer       │◄────┤     Layer       │◄────┤     Layer       │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
                                                        │
                                                        │
                                                        ▼
                                            ┌─────────────────────────┐
                                            │                         │
                                            │  Local Storage          │
                                            │  (SQLite + SharedPrefs) │
                                            │                         │
                                            └───────────┬─────────────┘
                                                        │
                                                        │
                                                        ▼
                                            ┌─────────────────────────┐
                                            │                         │
                                            │  Remote API             │
                                            │  (REST/GraphQL)         │
                                            │                         │
                                            └─────────────────────────┘
```

### Key Components:

1. **API Service Layer**: Handles communication with the backend server
2. **Sync Manager**: Orchestrates data synchronization between local and remote storage
3. **Repository Layer**: Abstracts data access for the application
4. **Auth Manager**: Handles authentication and token management
5. **Connectivity Monitor**: Tracks network status and triggers sync when appropriate
6. **Conflict Resolution Service**: Handles data conflicts during synchronization

## Components and Interfaces

### 1. API Service

The API Service will handle all HTTP communication with the backend server.

```dart
abstract class ApiService {
  // Authentication
  Future<AuthResponse> login(String username, String password);
  Future<AuthResponse> refreshToken(String refreshToken);
  Future<void> logout();
  
  // Application CRUD
  Future<List<ApplicationDto>> getApplications({Map<String, dynamic>? filters});
  Future<ApplicationDto> getApplication(String id);
  Future<ApplicationDto> createApplication(ApplicationDto application);
  Future<ApplicationDto> updateApplication(String id, ApplicationDto application);
  Future<void> deleteApplication(String id);
  
  // File uploads
  Future<String> uploadFile(File file, String applicationId, String fileType);
  Future<List<String>> uploadFiles(List<File> files, String applicationId, String fileType);
  
  // Sync status
  Future<SyncStatusDto> getSyncStatus(String applicationId);
}
```

Implementation:

```dart
class ApiServiceImpl implements ApiService {
  final Dio _dio;
  final AuthManager _authManager;
  
  ApiServiceImpl(this._dio, this._authManager) {
    _setupInterceptors();
  }
  
  void _setupInterceptors() {
    _dio.interceptors.add(AuthInterceptor(_authManager));
    _dio.interceptors.add(LoggingInterceptor());
    _dio.interceptors.add(ConnectivityInterceptor());
  }
  
  // Implementation of interface methods...
}
```

### 2. Sync Manager

The Sync Manager will coordinate data synchronization between local and remote storage.

```dart
class SyncManager {
  final ApiService _apiService;
  final LocalStorageService _localStorageService;
  final ConnectivityService _connectivityService;
  
  // Queue for pending sync operations
  final Queue<SyncOperation> _syncQueue = Queue();
  
  // Sync all applications
  Future<SyncResult> syncAll() async {
    // Implementation...
  }
  
  // Sync specific application
  Future<SyncResult> syncApplication(String applicationId) async {
    // Implementation...
  }
  
  // Handle sync conflicts
  Future<void> resolveConflict(SyncConflict conflict) async {
    // Implementation...
  }
  
  // Background sync process
  Future<void> startBackgroundSync() async {
    // Implementation...
  }
}
```

### 3. Repository Layer

The Repository Layer will abstract data access for the application.

```dart
abstract class ApplicationRepository {
  // Local operations
  Future<List<Customer>> getLocalApplications({ApplicationFilter? filter});
  Future<Customer?> getLocalApplication(String id);
  Future<String> saveLocalApplication(Customer application);
  Future<bool> updateLocalApplication(String id, Customer application);
  Future<bool> deleteLocalApplication(String id);
  
  // Remote operations
  Future<List<Customer>> getRemoteApplications({ApplicationFilter? filter});
  Future<Customer?> getRemoteApplication(String id);
  Future<String> saveRemoteApplication(Customer application);
  Future<bool> updateRemoteApplication(String id, Customer application);
  Future<bool> deleteRemoteApplication(String id);
  
  // Sync operations
  Future<SyncResult> syncApplication(String id);
  Future<SyncResult> syncAllApplications();
  Future<SyncStatus> getApplicationSyncStatus(String id);
}
```

Implementation:

```dart
class ApplicationRepositoryImpl implements ApplicationRepository {
  final LocalStorageService _localStorageService;
  final ApiService _apiService;
  final SyncManager _syncManager;
  
  ApplicationRepositoryImpl(
    this._localStorageService,
    this._apiService,
    this._syncManager,
  );
  
  // Implementation of interface methods...
}
```

### 4. Auth Manager

The Auth Manager will handle authentication and token management.

```dart
class AuthManager {
  final SharedPreferences _prefs;
  final StreamController<AuthState> _authStateController = StreamController<AuthState>.broadcast();
  
  Stream<AuthState> get authStateChanges => _authStateController.stream;
  
  // Get current auth token
  Future<String?> getAuthToken() async {
    // Implementation...
  }
  
  // Refresh token if needed
  Future<String?> refreshTokenIfNeeded() async {
    // Implementation...
  }
  
  // Login
  Future<bool> login(String username, String password) async {
    // Implementation...
  }
  
  // Logout
  Future<void> logout() async {
    // Implementation...
  }
}
```

### 5. Connectivity Monitor

The Connectivity Monitor will track network status and trigger sync when appropriate.

```dart
class ConnectivityMonitor {
  final Connectivity _connectivity = Connectivity();
  final StreamController<ConnectivityStatus> _controller = StreamController<ConnectivityStatus>.broadcast();
  
  Stream<ConnectivityStatus> get status => _controller.stream;
  
  ConnectivityMonitor() {
    _connectivity.onConnectivityChanged.listen((result) {
      _controller.add(_getStatusFromResult(result));
    });
  }
  
  Future<ConnectivityStatus> checkConnectivity() async {
    final result = await _connectivity.checkConnectivity();
    return _getStatusFromResult(result);
  }
  
  ConnectivityStatus _getStatusFromResult(ConnectivityResult result) {
    switch (result) {
      case ConnectivityResult.mobile:
        return ConnectivityStatus.cellular;
      case ConnectivityResult.wifi:
        return ConnectivityStatus.wifi;
      case ConnectivityResult.ethernet:
        return ConnectivityStatus.ethernet;
      default:
        return ConnectivityStatus.offline;
    }
  }
}
```

## Data Models

### 1. API Data Transfer Objects (DTOs)

```dart
class ApplicationDto {
  final String? id;
  final String status;
  final String? idCardType;
  final String? idNumber;
  final String? fullNameKhmer;
  final String? fullNameLatin;
  final String? phone;
  final String? dateOfBirth;
  final String? portfolioOfficerName;
  final double? requestedAmount;
  final List<String>? loanPurposes;
  final String? purposeDetails;
  final String? productType;
  final String? desiredLoanTerm;
  final String? requestedDisbursementDate;
  final String? guarantorName;
  final String? guarantorPhone;
  final List<String>? idCardImages;
  final String? borrowerNidPhoto;
  final String? borrowerHomePhoto;
  final String? borrowerBusinessPhoto;
  final String? guarantorNidPhoto;
  final String? guarantorHomePhoto;
  final String? guarantorBusinessPhoto;
  final String? profilePhoto;
  final List<String>? selectedCollateralTypes;
  final String? createdAt;
  final String? updatedAt;
  final String? syncStatus;
  final String? lastSyncedAt;
  
  // Constructor, toJson, fromJson methods...
}

class SyncStatusDto {
  final String applicationId;
  final String status; // 'synced', 'pending', 'failed', 'conflict'
  final String? lastSyncedAt;
  final String? errorMessage;
  
  // Constructor, toJson, fromJson methods...
}

class AuthResponse {
  final String accessToken;
  final String refreshToken;
  final int expiresIn;
  final String tokenType;
  
  // Constructor, toJson, fromJson methods...
}
```

### 2. Local Database Schema Updates

The existing SQLite schema will need to be updated to include sync-related fields:

```sql
ALTER TABLE customer_applications ADD COLUMN sync_status TEXT DEFAULT 'not_synced';
ALTER TABLE customer_applications ADD COLUMN server_id TEXT;
ALTER TABLE customer_applications ADD COLUMN last_synced_at TEXT;
ALTER TABLE customer_applications ADD COLUMN sync_error TEXT;
ALTER TABLE customer_applications ADD COLUMN version INTEGER DEFAULT 1;
```

### 3. Sync Operation Models

```dart
enum SyncStatus {
  notSynced,
  pending,
  synced,
  failed,
  conflict
}

class SyncOperation {
  final String id;
  final String applicationId;
  final SyncOperationType type;
  final DateTime createdAt;
  int retryCount = 0;
  
  // Constructor and methods...
}

enum SyncOperationType {
  create,
  update,
  delete
}

class SyncResult {
  final bool success;
  final int syncedCount;
  final int failedCount;
  final int conflictCount;
  final List<String> failedIds;
  final List<SyncConflict> conflicts;
  
  // Constructor...
}

class SyncConflict {
  final String applicationId;
  final ApplicationDto localVersion;
  final ApplicationDto remoteVersion;
  
  // Constructor and resolution methods...
}
```

## Error Handling

The API integration will implement a comprehensive error handling strategy:

1. **Network Errors**: Detect and handle network connectivity issues
2. **Authentication Errors**: Handle token expiration and authentication failures
3. **API Errors**: Parse and handle server-side error responses
4. **Sync Conflicts**: Detect and resolve data conflicts during synchronization

Error handling will follow these principles:

- Graceful degradation to offline mode when network is unavailable
- Automatic retry with exponential backoff for transient errors
- Clear user feedback for persistent errors
- Detailed logging for debugging purposes

## Testing Strategy

The API integration will be tested at multiple levels:

1. **Unit Tests**:
   - Test individual components in isolation
   - Mock external dependencies
   - Test error handling and edge cases

2. **Integration Tests**:
   - Test interaction between components
   - Test data flow through the system
   - Test synchronization logic

3. **End-to-End Tests**:
   - Test complete workflows
   - Test offline/online transitions
   - Test conflict resolution

4. **Performance Tests**:
   - Test sync performance with large datasets
   - Test behavior under poor network conditions
   - Test battery and data usage

## Security Considerations

1. **Authentication**: Use OAuth 2.0 or JWT for secure authentication
2. **Data Encryption**: Use HTTPS for all API communication
3. **Token Storage**: Store authentication tokens securely
4. **Sensitive Data**: Handle PII according to data protection regulations
5. **Input Validation**: Validate all data before sending to the server
6. **Output Sanitization**: Sanitize all data received from the server

## Migration Strategy

To migrate from the current local-only storage to the new API-integrated system:

1. **Database Migration**: Update local database schema to include sync fields
2. **Initial Sync**: Upload existing local data to the server
3. **Conflict Resolution**: Handle any conflicts during initial sync
4. **Feature Flags**: Use feature flags to gradually roll out API integration

## Dependencies

The API integration will require the following additional dependencies:

```yaml
dependencies:
  dio: ^5.0.0            # HTTP client
  connectivity_plus: ^4.0.0  # Network connectivity monitoring
  jwt_decoder: ^2.0.1    # JWT token handling
  flutter_secure_storage: ^8.0.0  # Secure storage for tokens
  retry: ^3.1.1          # Retry logic for failed requests
  synchronized: ^3.1.0   # Thread synchronization
  workmanager: ^0.5.1    # Background task scheduling
```