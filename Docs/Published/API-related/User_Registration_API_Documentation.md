# User Registration & Authentication API Documentation

**Version**: 1.0.0
**Base URL**: `http://localhost:8000/api/v1`
**Authentication**: JWT Bearer tokens (except for public endpoints)

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [User Management](#user-management)
4. [Admin Operations](#admin-operations)
5. [Health Monitoring](#health-monitoring)
6. [Error Responses](#error-responses)
7. [Rate Limiting](#rate-limiting)
8. [Migration Guide](#migration-guide)

---

## Overview

The tldw User Registration system provides a complete authentication and user management solution supporting both single-user and multi-user deployments. The API follows RESTful principles and uses JWT tokens for authentication.

### Key Features
- JWT-based authentication with refresh tokens
- User registration with optional registration codes
- Role-based access control (RBAC)
- Storage quota management
- Audit logging for security events
- Rate limiting for abuse prevention
- Health monitoring endpoints for Kubernetes

### Authentication Modes
- **Single-User Mode**: Simplified authentication for personal deployments
- **Multi-User Mode**: Full authentication with PostgreSQL backend for teams

---

## Authentication

### POST /auth/login
**Description**: Authenticate user and receive JWT tokens

**Request Body** (application/x-www-form-urlencoded):
```
username: string (required) - Username or email
password: string (required) - User password
```

**Response** (200 OK):
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Error Responses**:
- `401 Unauthorized`: Invalid credentials
- `403 Forbidden`: Account inactive or locked
- `429 Too Many Requests`: Rate limit exceeded

**Example**:
```bash
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=john&password=SecurePass123!"
```

---

### POST /auth/logout
**Description**: Logout current user and invalidate session

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "message": "Successfully logged out",
  "details": {
    "user_id": 123
  }
}
```

---

### POST /auth/refresh
**Description**: Refresh access token using refresh token

**Request Body**:
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Response** (200 OK):
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 3600
}
```

**Error Responses**:
- `401 Unauthorized`: Invalid or expired refresh token

---

### POST /auth/register
**Description**: Register a new user account

**Request Body**:
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "SecurePass123!",
  "registration_code": "INVITE-ABC123" // Optional, if required
}
```

**Response** (201 Created):
```json
{
  "message": "Registration successful",
  "user_id": 123,
  "username": "john_doe",
  "email": "john@example.com",
  "requires_verification": false
}
```

**Error Responses**:
- `400 Bad Request`: Validation error (weak password, invalid email)
- `409 Conflict`: Username or email already exists

**Password Requirements**:
- Minimum 8 characters
- At least one uppercase letter
- At least one lowercase letter
- At least one number
- At least one special character

---

### GET /auth/me
**Description**: Get current authenticated user information

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "id": 123,
  "uuid": "550e8400-e29b-41d4-a716-446655440000",
  "username": "john_doe",
  "email": "john@example.com",
  "role": "user",
  "is_active": true,
  "is_verified": true,
  "created_at": "2025-01-14T10:00:00Z",
  "last_login": "2025-01-14T15:30:00Z",
  "storage_quota_mb": 5120,
  "storage_used_mb": 1024
}
```

---

## User Management

### GET /users/me
**Description**: Get detailed user profile

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "id": 123,
  "username": "john_doe",
  "email": "john@example.com",
  "role": "user",
  "created_at": "2025-01-14T10:00:00Z",
  "storage_quota_mb": 5120,
  "storage_used_mb": 1024,
  "media_count": 42,
  "notes_count": 15,
  "prompts_count": 8,
  "last_activity": "2025-01-14T15:30:00Z"
}
```

---

### PUT /users/me
**Description**: Update current user profile

**Headers**:
```
Authorization: Bearer <access_token>
```

**Request Body**:
```json
{
  "email": "newemail@example.com",
  "full_name": "John Doe",
  "preferences": {
    "theme": "dark",
    "language": "en"
  }
}
```

**Response** (200 OK):
```json
{
  "message": "Profile updated successfully",
  "user": {
    "id": 123,
    "email": "newemail@example.com",
    "full_name": "John Doe"
  }
}
```

---

### POST /users/change-password
**Description**: Change user password

**Headers**:
```
Authorization: Bearer <access_token>
```

**Request Body**:
```json
{
  "current_password": "OldPassword123!",
  "new_password": "NewPassword456!"
}
```

**Response** (200 OK):
```json
{
  "message": "Password changed successfully"
}
```

**Error Responses**:
- `400 Bad Request`: Current password incorrect or new password weak
- `401 Unauthorized`: Not authenticated

---

### GET /users/sessions
**Description**: Get all active sessions for current user

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
[
  {
    "id": 1,
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "created_at": "2025-01-14T10:00:00Z",
    "last_activity": "2025-01-14T15:30:00Z",
    "expires_at": "2025-01-14T16:00:00Z"
  }
]
```

---

### DELETE /users/sessions/{session_id}
**Description**: Revoke a specific session

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "message": "Session revoked successfully"
}
```

---

### GET /users/storage
**Description**: Get storage usage details

**Headers**:
```
Authorization: Bearer <access_token>
```

**Response** (200 OK):
```json
{
  "user_id": 123,
  "storage_used_mb": 1024.5,
  "storage_quota_mb": 5120,
  "available_mb": 4095.5,
  "usage_percentage": 20.0
}
```

---

## Admin Operations

### GET /admin/users
**Description**: List all users (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Query Parameters**:
- `page` (int): Page number (default: 1)
- `limit` (int): Items per page (default: 50)
- `search` (string): Search by username or email
- `role` (string): Filter by role (user, moderator, admin)
- `is_active` (boolean): Filter by active status

**Response** (200 OK):
```json
{
  "users": [
    {
      "id": 123,
      "username": "john_doe",
      "email": "john@example.com",
      "role": "user",
      "is_active": true,
      "created_at": "2025-01-14T10:00:00Z",
      "storage_used_mb": 1024
    }
  ],
  "total": 150,
  "page": 1,
  "pages": 3
}
```

---

### GET /admin/users/{user_id}
**Description**: Get detailed user information (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "id": 123,
  "username": "john_doe",
  "email": "john@example.com",
  "role": "user",
  "is_active": true,
  "is_verified": true,
  "created_at": "2025-01-14T10:00:00Z",
  "last_login": "2025-01-14T15:30:00Z",
  "storage_quota_mb": 5120,
  "storage_used_mb": 1024,
  "media_count": 42,
  "notes_count": 15,
  "login_count": 25,
  "failed_login_count": 3
}
```

---

### PUT /admin/users/{user_id}
**Description**: Update user account (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Request Body**:
```json
{
  "role": "moderator",
  "is_active": true,
  "is_verified": true,
  "storage_quota_mb": 10240
}
```

**Response** (200 OK):
```json
{
  "message": "User updated successfully",
  "user": {
    "id": 123,
    "username": "john_doe",
    "role": "moderator",
    "storage_quota_mb": 10240
  }
}
```

---

### DELETE /admin/users/{user_id}
**Description**: Deactivate user account (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "message": "User deactivated successfully"
}
```

---

### POST /admin/users/{user_id}/reset-password
**Description**: Reset user password (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Request Body**:
```json
{
  "new_password": "TempPassword123!"
}
```

**Response** (200 OK):
```json
{
  "message": "Password reset successfully",
  "temporary": true
}
```

---

### POST /admin/registration-codes
**Description**: Create registration code (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Request Body**:
```json
{
  "max_uses": 5,
  "expiry_days": 7,
  "role_to_grant": "user",
  "storage_quota_mb": 5120
}
```

**Response** (200 OK):
```json
{
  "code": "INVITE-XYZ789",
  "max_uses": 5,
  "uses_remaining": 5,
  "expires_at": "2025-01-21T16:00:00Z",
  "role_to_grant": "user",
  "storage_quota_mb": 5120
}
```

---

### GET /admin/registration-codes
**Description**: List all registration codes (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "codes": [
    {
      "id": 1,
      "code": "INVITE-XYZ789",
      "max_uses": 5,
      "uses": 2,
      "is_active": true,
      "expires_at": "2025-01-21T16:00:00Z",
      "created_by": "admin",
      "created_at": "2025-01-14T16:00:00Z"
    }
  ]
}
```

---

### DELETE /admin/registration-codes/{code_id}
**Description**: Deactivate registration code (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "message": "Registration code deactivated"
}
```

---

### GET /admin/stats
**Description**: Get system statistics (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Response** (200 OK):
```json
{
  "users": {
    "total": 150,
    "active": 145,
    "verified": 140,
    "new_today": 5,
    "new_this_week": 23,
    "new_this_month": 89
  },
  "storage": {
    "total_gb": 500,
    "used_gb": 127.5,
    "available_gb": 372.5,
    "percentage_used": 25.5
  },
  "sessions": {
    "active": 42,
    "expired_today": 15
  },
  "activity": {
    "logins_today": 234,
    "api_calls_today": 5678,
    "media_processed_today": 45
  }
}
```

---

### GET /admin/audit-log
**Description**: View audit log (admin only)

**Headers**:
```
Authorization: Bearer <admin_token>
```

**Query Parameters**:
- `user_id` (int): Filter by user
- `action` (string): Filter by action type
- `days` (int): Number of days to look back (default: 30)
- `limit` (int): Maximum entries (default: 100)

**Response** (200 OK):
```json
{
  "entries": [
    {
      "id": 1,
      "user_id": 123,
      "username": "john_doe",
      "action": "user_login",
      "details": {
        "success": true
      },
      "ip_address": "192.168.1.100",
      "user_agent": "Mozilla/5.0...",
      "created_at": "2025-01-14T15:30:00Z"
    }
  ],
  "total": 523
}
```

**Audit Action Types**:
- Authentication: `user_login`, `user_logout`, `login_failed`, `token_refresh`
- User Management: `user_registered`, `user_updated`, `user_deleted`, `password_changed`
- Admin Actions: `admin_user_update`, `admin_code_created`, `admin_quota_changed`
- Security: `rate_limit_exceeded`, `invalid_token`, `unauthorized_access`

---

## Health Monitoring

### GET /health
**Description**: Comprehensive health check

**Response** (200 OK):
```json
{
  "status": "healthy",
  "checks": {
    "database": {
      "status": "healthy",
      "response_time_ms": 5
    },
    "redis": {
      "status": "healthy",
      "response_time_ms": 2
    },
    "storage": {
      "status": "healthy",
      "available_gb": 372.5
    }
  },
  "version": "1.0.0",
  "timestamp": "2025-01-14T16:00:00Z"
}
```

**Status Codes**:
- `200 OK`: All systems healthy
- `206 Partial Content`: Some systems degraded
- `503 Service Unavailable`: Critical systems down

---

### GET /health/live
**Description**: Kubernetes liveness probe

**Response** (200 OK):
```json
{
  "status": "alive",
  "timestamp": "2025-01-14T16:00:00Z"
}
```

---

### GET /health/ready
**Description**: Kubernetes readiness probe

**Response** (200 OK):
```json
{
  "status": "ready",
  "timestamp": "2025-01-14T16:00:00Z"
}
```

**Status Codes**:
- `200 OK`: Ready to serve traffic
- `503 Service Unavailable`: Not ready (starting up or unhealthy)

---

### GET /health/metrics
**Description**: System metrics for monitoring

**Response** (200 OK):
```json
{
  "cpu": {
    "usage_percent": 45.2,
    "cores": 4
  },
  "memory": {
    "usage_percent": 62.5,
    "used_mb": 2560,
    "available_mb": 1536
  },
  "disk": {
    "usage_percent": 25.5,
    "used_gb": 127.5,
    "available_gb": 372.5
  },
  "requests": {
    "total": 123456,
    "rate_per_second": 45.6,
    "error_rate": 0.02
  },
  "uptime_seconds": 864000
}
```

---

## Error Responses

All error responses follow a consistent format:

```json
{
  "detail": "Human-readable error message",
  "error_code": "SPECIFIC_ERROR_CODE",
  "timestamp": "2025-01-14T16:00:00Z",
  "path": "/api/v1/auth/login"
}
```

### Common HTTP Status Codes

- `400 Bad Request`: Invalid request parameters or body
- `401 Unauthorized`: Missing or invalid authentication
- `403 Forbidden`: Insufficient permissions
- `404 Not Found`: Resource not found
- `409 Conflict`: Resource already exists
- `422 Unprocessable Entity`: Validation error
- `429 Too Many Requests`: Rate limit exceeded
- `500 Internal Server Error`: Server error
- `503 Service Unavailable`: Service temporarily unavailable

### Error Codes

- `INVALID_CREDENTIALS`: Username or password incorrect
- `ACCOUNT_INACTIVE`: Account has been deactivated
- `ACCOUNT_LOCKED`: Account locked due to too many failed attempts
- `TOKEN_EXPIRED`: JWT token has expired
- `TOKEN_INVALID`: JWT token is invalid or malformed
- `REFRESH_TOKEN_REVOKED`: Refresh token has been revoked
- `WEAK_PASSWORD`: Password doesn't meet requirements
- `DUPLICATE_USER`: Username or email already exists
- `REGISTRATION_DISABLED`: Registration is currently disabled
- `INVALID_REGISTRATION_CODE`: Registration code invalid or expired
- `QUOTA_EXCEEDED`: Storage quota exceeded
- `RATE_LIMIT_EXCEEDED`: Too many requests

---

## Rate Limiting

The API implements rate limiting to prevent abuse:

### Default Limits

- **Authentication Endpoints**: 5 requests per minute
- **Registration**: 3 requests per hour
- **Password Reset**: 3 requests per hour
- **General API**: 100 requests per minute
- **Admin Endpoints**: 1000 requests per minute

### Rate Limit Headers

- Authentication endpoints return HTTP 429 with a `Retry-After` header (seconds) when rate limited.
- Some non-auth modules may include `X-RateLimit-*` headers (e.g., Evaluations, Chat, RAG) for client visibility.

### Rate Limit Response

When rate limit is exceeded:

```json
{
  "detail": "Too many requests. Please try again later.",
  "retry_after": 60
}
```

---

## Migration Guide

### Migrating from Single-User to Multi-User

1. **Install PostgreSQL**:
```bash
sudo apt install postgresql postgresql-contrib
```

2. **Create Database**:
```sql
CREATE DATABASE tldw_multiuser;
CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'secure_password';
GRANT ALL PRIVILEGES ON DATABASE tldw_multiuser TO tldw_user;
```

3. **Run Schema**:
```bash
psql -U tldw_user -d tldw_multiuser -f tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql
```

4. **Update Environment**:
```bash
# .env file
AUTH_MODE=multi_user
DATABASE_URL=postgresql://tldw_user:secure_password@localhost/tldw_multiuser
JWT_SECRET_KEY=<auto-generated-or-set-manually>
ENABLE_REGISTRATION=true
REQUIRE_REGISTRATION_CODE=false
```

5. **Run Migration Script**:
```bash
python tldw_Server_API/scripts/migrate_to_multiuser.py \
  --admin-email admin@example.com \
  --admin-password SecureAdminPass123!
```

6. **Verify Migration**:
```bash
# Start server
python -m uvicorn tldw_Server_API.app.main:app --reload

# Test login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=SecureAdminPass123!"
```

### Environment Variables

Required environment variables for multi-user mode:

```bash
# Authentication
AUTH_MODE=multi_user
DATABASE_URL=postgresql://user:pass@localhost/dbname
REDIS_URL=redis://localhost:6379  # Optional, for caching
JWT_SECRET_KEY=your-secret-key-here
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# Registration
ENABLE_REGISTRATION=true
REQUIRE_REGISTRATION_CODE=false
DEFAULT_USER_ROLE=user
DEFAULT_STORAGE_QUOTA_MB=5120

# Rate Limiting
AUTH_RATE_LIMIT=5/minute
API_RATE_LIMIT=100/minute

# Security
ALLOWED_ORIGINS=http://localhost:3000,https://yourdomain.com
CORS_ALLOW_CREDENTIALS=true
```

---

## Examples

### Complete Authentication Flow

```python
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

def request_json(method: str, url: str, payload=None, headers=None, params=None):
    if params:
        url = f"{url}?{urlencode(params)}"
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method=method)
    with urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        try:
            return resp.status, json.loads(body)
        except json.JSONDecodeError:
            return resp.status, body

def request_form(method: str, url: str, form, headers=None):
    data = urlencode(form).encode("utf-8")
    hdrs = {"Content-Type": "application/x-www-form-urlencoded"}
    if headers:
        hdrs.update(headers)
    req = Request(url, data=data, headers=hdrs, method=method)
    with urlopen(req) as resp:
        body = resp.read().decode("utf-8")
        try:
            return resp.status, json.loads(body)
        except json.JSONDecodeError:
            return resp.status, body

# Base URL
BASE_URL = "http://localhost:8000/api/v1"

# 1. Register new user
status, _ = request_json(
    "POST",
    f"{BASE_URL}/auth/register",
    {
        "username": "alice",
        "email": "alice@example.com",
        "password": "AlicePass123!"
    },
)
print(f"Registration: {status}")

# 2. Login
_, tokens = request_form(
    "POST",
    f"{BASE_URL}/auth/login",
    {
        "username": "alice",
        "password": "AlicePass123!"
    },
)
access_token = tokens["access_token"]
refresh_token = tokens["refresh_token"]

# 3. Access protected endpoint
headers = {"Authorization": f"Bearer {access_token}"}
_, profile = request_json("GET", f"{BASE_URL}/users/me", headers=headers)
print(f"Profile: {profile}")

# 4. Refresh token
_, new_tokens = request_json(
    "POST",
    f"{BASE_URL}/auth/refresh",
    {"refresh_token": refresh_token},
)
new_access_token = new_tokens["access_token"]

# 5. Logout
_, logout = request_json(
    "POST",
    f"{BASE_URL}/auth/logout",
    headers={"Authorization": f"Bearer {new_access_token}"},
)
print(f"Logout: {logout}")
```

### Admin User Management

```python
# Admin login
_, admin_tokens = request_form(
    "POST",
    f"{BASE_URL}/auth/login",
    {
        "username": "admin",
        "password": "AdminPass123!"
    },
)
admin_token = admin_tokens["access_token"]
admin_headers = {"Authorization": f"Bearer {admin_token}"}

# List users
_, users = request_json(
    "GET",
    f"{BASE_URL}/admin/users",
    headers=admin_headers,
    params={"page": 1, "limit": 10},
)

# Update user quota
request_json(
    "PUT",
    f"{BASE_URL}/admin/users/123",
    {"storage_quota_mb": 10240},
    headers=admin_headers,
)

# Create registration code
_, code_payload = request_json(
    "POST",
    f"{BASE_URL}/admin/registration-codes",
    {
        "max_uses": 10,
        "expiry_days": 30,
        "role_to_grant": "user"
    },
    headers=admin_headers,
)
registration_code = code_payload["code"]
print(f"Registration code: {registration_code}")
```

---

## Support

For issues, questions, or contributions:
- GitHub Issues: [tldw_server/issues](https://github.com/your-repo/issues)
- Documentation: [/Docs](https://github.com/your-repo/tree/main/Docs)
- API Explorer: http://localhost:8000/docs (when server is running)

---

*Last Updated: January 14, 2025*
