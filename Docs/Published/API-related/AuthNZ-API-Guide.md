# AuthNZ API Guide

## Table of Contents
- [Overview](#overview)
- [Authentication Modes](#authentication-modes)
- [API Endpoints](#api-endpoints)
- [Request Examples](#request-examples)
- [Response Formats](#response-formats)
- [Error Handling](#error-handling)
- [Rate Limiting](#rate-limiting)
- [Best Practices](#best-practices)

## Overview

The AuthNZ module provides authentication and authorization for the tldw_server API. It supports both single-user (API key) and multi-user (JWT) modes with robust security features (JWT sessions, rate-limited auth endpoints, audit events). Some features (e.g., public API key CRUD endpoints) are planned but not yet exposed.

### Base URL
```
http://localhost:8000/api/v1
```

### Authentication Headers

#### Single-User Mode
```http
X-API-KEY: your-api-key-here
```

#### Multi-User Mode
```http
Authorization: Bearer your-jwt-token-here
```

## Authentication Modes

### Single-User Mode

Simple API key authentication for personal deployments:

- **Configuration**: Follow the canonical single-user setup profile: `Docs/Getting_Started/Profile_Local_Single_User.md` (or Docker single-user: `Docs/Getting_Started/Profile_Docker_Single_User.md`)
- **API Key**: Auto-generated on first run, displayed in console
- **Header**: `X-API-KEY`
- **Use Case**: Personal installations, development

### Multi-User Mode

JWT-based authentication with user management:

- **Configuration**: Follow the canonical multi-user setup profile: `Docs/Getting_Started/Profile_Docker_Multi_User_Postgres.md`
- **Authentication**: Username/password login returns JWT tokens
- **Header**: `Authorization: Bearer <token>`
- **Features**: User registration, roles, permissions; API key CRUD endpoints are not yet public
- **Use Case**: Team deployments, production environments

Signing algorithms:
- Default HS256 with `JWT_SECRET_KEY`.
- Recommended RS256 with `JWT_PRIVATE_KEY`/`JWT_PUBLIC_KEY` for multi-service deployments.

## API Endpoints

### Authentication Endpoints

#### Login (Multi-User Mode)
```http
POST /api/v1/auth/login
```

**Request (form-encoded):**
Content-Type: `application/x-www-form-urlencoded`

Fields:
- `username`: string (username or email)
- `password`: string

Example curl:
```bash
curl -X POST \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=secure_password" \
  http://localhost:8000/api/v1/auth/login
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

#### Refresh Token
```http
POST /api/v1/auth/refresh
```

**Request Body:**
```json
{
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc..."
}
```

**Response:**
```json
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "refresh_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

Notes:
- In multi-user mode, refresh ties to the server-side session. If the session is missing or revoked, refresh fails with 401.
- When `ROTATE_REFRESH_TOKENS=true` (default), the endpoint returns a new refresh token and invalidates the previous one for the session. Clients must persist the returned refresh token for subsequent refreshes.
- Tokens optionally include `iss` and `aud` claims when configured. See Authentication Setup for `JWT_ISSUER` and `JWT_AUDIENCE`.

#### Logout
```http
POST /api/v1/auth/logout
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "message": "Successfully logged out"
}
```

### User Management Endpoints

#### Register New User (If Enabled)
```http
POST /api/v1/auth/register
```

**Request Body:**
```json
{
  "username": "newuser",
  "email": "newuser@example.com",
  "password": "SecurePassword123!",
  "registration_code": "INVITE-CODE-123"
}
```

**Response:**
```json
{
  "message": "Registration successful",
  "user_id": 2,
  "username": "newuser",
  "email": "newuser@example.com",
  "requires_verification": true
}
```

#### Get Current User
```http
GET /api/v1/auth/me
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "id": 1,
  "uuid": "123e4567-e89b-12d3-a456-426614174000",
  "username": "user@example.com",
  "email": "user@example.com",
  "role": "user",
  "is_active": true,
  "is_verified": true,
  "storage_quota_mb": 5120,
  "storage_used_mb": 1024,
  "created_at": "2024-01-01T00:00:00Z",
  "last_login": "2024-01-15T09:00:00Z"
}
```

`storage_quota_mb` is `null` when no storage quota is set for the user, which means unlimited. The `5120` above is an example of a quota that was set (via `limits.storage_quota_mb`), not a default. `DEFAULT_STORAGE_QUOTA_MB` is deprecated and sets no quota.

#### Update Password
```http
POST /api/v1/users/change-password
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Request Body:**
```json
{
  "current_password": "OldPassword123!",
  "new_password": "NewSecurePassword456!"
}
```

**Response:**
```json
{
  "message": "Password changed successfully"
}
```

### API Key Management (Status)

Public API endpoints for API key CRUD (list/create/rotate/revoke) are not yet exposed. In single-user mode, use the `X-API-KEY` header (key printed at startup). In multi-user mode, authenticate via JWT using the login and refresh endpoints above.

### Session Management

#### List Active Sessions
```http
GET /api/v1/users/sessions
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
[
  {
    "id": 123,
    "ip_address": "192.168.1.100",
    "user_agent": "Mozilla/5.0...",
    "created_at": "2024-01-15T08:00:00Z",
    "last_activity": "2024-01-15T10:00:00Z",
    "expires_at": "2024-01-15T12:00:00Z",
    "device_label": "Firefox on macOS",
    "is_current": true,
    "same_install": true
  }
]
```

- `is_current` is true only for the session making the request. The server derives it from the verified credential (the `session_id` claim of the access token, or the validated single-user cookie) and checks that the session is one of the caller's live sessions. A client cannot claim it.
- `device_label` is a short browser and OS label computed on read from the stored user agent. It is client-reported and not a proof of identity.
- `same_install` is true when a session was signed in from the same browser install (the `X-TLDW-Device-Id` header) as the caller's. The install id itself is never returned.

#### Revoke Session
```http
DELETE /api/v1/users/sessions/{session_id}
```

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "message": "Session revoked successfully"
}
```

#### Revoke All Sessions
```http
POST /api/v1/users/sessions/revoke-all?keep_current=true
```

`keep_current=true` keeps the caller's own session and ends the others. The count in the response covers live sessions only; sessions that were already revoked or expired are neither counted nor changed (their `revoke_reason`, for example `session_cap`, is kept).

**Headers:**
```http
Authorization: Bearer your-jwt-token
```

**Response:**
```json
{
  "message": "Successfully revoked 2 sessions",
  "details": {"sessions_revoked": 2}
}
```

#### Interactive session required

Revoke one session, revoke all sessions, and `POST /auth/logout` with `all_devices: true` (or with a token that carries no `session_id`) need an interactive browser session. A browser session is an access token that carries `session_id`, or a single-user cookie session. API keys, virtual keys, service tokens and impersonation tokens are refused with:

```json
{"detail": {"code": "interactive_session_required", "message": "..."}}
```
(HTTP 403.) Before this change an API key sent as a Bearer token to `logout` could sign every session out; it can no longer. Listing sessions is not restricted. If the session store cannot be read, the endpoints answer 503 rather than treating the caller as a non-browser credential.

#### New sign-in alert

When a user signs in while another live session exists, or while the sign-in evicted a session through the session cap, the server writes a notification to that user's inbox, off the request path and best effort (a failure never fails the sign-in). It is not written when the same browser install signs in again, and a user's first sign-in with no other session gets none.

| Field | Value |
| --- | --- |
| `kind` | `new_sign_in` |
| `title` | `New sign-in to your account` |
| `severity` | `warning` |
| `link_url` | `/settings/sessions` |
| `dedupe_key` | `new_sign_in:<hash>`, bucketed in 10-minute windows (a sign-in storm is one alert; the key is a hash so the install id never leaves the server) |
| `source_job_id` | the id of the new session. The server uses it for ordering; clients also match it against the sessions list to hide the alert on the browser that signed in |

The message names the device label "as the browser reported it", the IP the server saw and the time. It carries no token, cookie or install id. `new_sign_in` notifications are pruned after 30 days.

Who may silence it (mark read, dismiss or snooze):

- Only an interactive session older than the alert (a lower `sessions.id` than the alert's `source_job_id`). A newer session cannot hide the alert of its own sign-in.
- After 24 hours any interactive browser session may, including the one the alert is about. An owner who is signed out by the intruder and does not sign in again for a day may never see the alert.
- API keys and other non-browser credentials never may.

Refusals: dismiss and snooze answer 403 `new_sign_in_protected`. `POST /notifications/mark-read` answers 200 with `refused_ids` listing the alerts it kept unread when other ids were allowed, and 403 `new_sign_in_protected` when every id was refused.

`GET /notifications` accepts an optional `kind` filter (for example `?kind=new_sign_in`); the sign-in banner uses it so other notification kinds cannot bury the alert.

#### Caveats

- The IP in the alert and in the listing is the address the server saw. Behind the bundled Next.js proxy with no reverse proxy in front, that is the proxy's address, because Next's rewrite proxy cannot forward the browser's IP. A reverse proxy that sets `X-Forwarded-For` is honoured only when `AUTH_TRUST_X_FORWARDED_FOR=true` and the physical peer is listed in `AUTH_TRUSTED_PROXY_IPS`.
- Single-user mode: the protection is the alert plus the session cap, not isolation from a holder of the API key. `POST /auth/single-user/session` mints a cookie session for anyone with the key, and that session counts as interactive. "Sign out all other sessions" ends other browsers, not a script that holds the key. The remedy is to rotate the API key.
- Sign-ins without an install id (scripts, OIDC) are grouped for the 10-minute dedupe window by IP address and user agent. A repeat from the same address with the same user agent can therefore fold into an alert the owner already dismissed, and write no new one.
- Cancelling a snooze ("remind me later") is not covered by the silence rule. Snoozing dismisses the alert, and only an allowed session can snooze, so the owner has already seen it. The snooze's reminder task can be changed or deleted with any credential through `PATCH` and `DELETE /api/v1/reminders/tasks/{id}`, and when it fires it is an ordinary `reminder_due` notification that anyone may dismiss. Gating cancel-snooze alone would close nothing; protecting "remind me later" would mean gating reminder tasks derived from an alert.
- PostgreSQL session ids are not commit-ordered, so two sign-ins racing within milliseconds can invert by one id in the silence rule.

## Request Examples

### JavaScript/TypeScript

#### Single-User Mode
```javascript
// Simple API call with API key
const response = await fetch('http://localhost:8000/api/v1/media/search', {
  headers: {
    'X-API-KEY': 'your-api-key-here',
    'Content-Type': 'application/json'
  }
});
const data = await response.json();
```

#### Multi-User Mode
```javascript
// Login and get token
async function login(username, password) {
  const response = await fetch('http://localhost:8000/api/v1/auth/login', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/x-www-form-urlencoded'
    },
    body: new URLSearchParams({ username, password })
  });

  if (!response.ok) {
    throw new Error('Login failed');
  }

  const data = await response.json();
  // Store tokens securely
  localStorage.setItem('access_token', data.access_token);
  localStorage.setItem('refresh_token', data.refresh_token);

  return data;
}

// Make authenticated request
async function makeAuthenticatedRequest(endpoint) {
  const token = localStorage.getItem('access_token');

  const response = await fetch(`http://localhost:8000/api/v1${endpoint}`, {
    headers: {
      'Authorization': `Bearer ${token}`,
      'Content-Type': 'application/json'
    }
  });

  if (response.status === 401) {
    // Token expired, try refresh
    await refreshToken();
    return makeAuthenticatedRequest(endpoint);
  }

  return response.json();
}

// Refresh token
async function refreshToken() {
  const refreshToken = localStorage.getItem('refresh_token');

  const response = await fetch('http://localhost:8000/api/v1/auth/refresh', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ refresh_token: refreshToken })
  });

  if (!response.ok) {
    // Refresh failed, redirect to login
    window.location.href = '/login';
    return;
  }

  const data = await response.json();
  localStorage.setItem('access_token', data.access_token);
}
```

### Python

#### Single-User Mode
```python
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

# Set up headers with API key
headers = {'X-API-KEY': 'your-api-key-here'}
timeout_seconds = 10  # seconds

# Make request
params = urlencode({'query': 'test'})
req = Request(f'http://localhost:8000/api/v1/media/search?{params}', headers=headers, method='GET')
with urlopen(req, timeout=timeout_seconds) as resp:
    data = json.loads(resp.read().decode('utf-8'))
```

#### Multi-User Mode
```python
import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError
from datetime import datetime, timedelta
import socket

class TLDWClient:
    def __init__(self, base_url='http://localhost:8000', timeout=10):
        self.base_url = base_url
        self.access_token = None
        self.refresh_token = None
        self.token_expires = None
        self.timeout = timeout  # seconds

    def _parse_auth_response(self, body, required_fields):
        try:
            data = json.loads(body)
        except json.JSONDecodeError as err:
            raise RuntimeError("Auth response was not valid JSON.") from err

        missing = [field for field in required_fields if field not in data]
        if missing:
            raise RuntimeError(f"Auth response missing fields: {', '.join(missing)}")
        return data

    def login(self, username, password):
        """Login and store tokens"""
        form = urlencode({'username': username, 'password': password}).encode('utf-8')
        req = Request(
            f'{self.base_url}/api/v1/auth/login',
            data=form,
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            method='POST',
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode('utf-8')
        except HTTPError as err:
            detail = err.read().decode('utf-8')
            raise RuntimeError(f'Login failed ({err.code}): {detail}') from err
        except (URLError, socket.timeout) as err:
            raise RuntimeError(f'Login request failed: {err}') from err

        data = self._parse_auth_response(body, ('access_token', 'refresh_token', 'expires_in'))
        self.access_token = data['access_token']
        self.refresh_token = data['refresh_token']
        self.token_expires = datetime.now() + timedelta(seconds=data['expires_in'])

        return data

    def refresh_access_token(self):
        """Refresh the access token"""
        if not self.refresh_token:
            raise RuntimeError('No refresh token available; please login again.')
        payload = json.dumps({'refresh_token': self.refresh_token}).encode('utf-8')
        req = Request(
            f'{self.base_url}/api/v1/auth/refresh',
            data=payload,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                body = resp.read().decode('utf-8')
        except HTTPError as err:
            detail = err.read().decode('utf-8')
            raise RuntimeError(f'Refresh failed ({err.code}): {detail}') from err
        except (URLError, socket.timeout) as err:
            raise RuntimeError(f'Refresh request failed: {err}') from err

        data = self._parse_auth_response(body, ('access_token', 'expires_in'))
        self.access_token = data['access_token']
        self.token_expires = datetime.now() + timedelta(seconds=data['expires_in'])

    def request(self, method, endpoint, **kwargs):
        """Make authenticated request with automatic token refresh"""
        # Check if token needs refresh
        if self.token_expires and datetime.now() >= self.token_expires:
            try:
                self.refresh_access_token()
            except HTTPError as refresh_err:
                if refresh_err.code == 401:
                    raise RuntimeError(
                        'Refresh token expired; please login again.'
                    ) from refresh_err
                raise

        url = f'{self.base_url}/api/v1{endpoint}'
        params = kwargs.pop('params', None)
        json_body = kwargs.pop('json', None)
        if params:
            url = f"{url}?{urlencode(params)}"

        headers = kwargs.pop('headers', {})
        headers['Authorization'] = f'Bearer {self.access_token}'
        data = None
        if json_body is not None:
            headers['Content-Type'] = 'application/json'
            data = json.dumps(json_body).encode('utf-8')

        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=self.timeout) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except HTTPError as err:
            if err.code == 401:
                try:
                    self.refresh_access_token()
                except HTTPError as refresh_err:
                    if refresh_err.code == 401:
                        raise RuntimeError(
                            'Refresh token expired; please login again.'
                        ) from refresh_err
                    raise
                headers['Authorization'] = f'Bearer {self.access_token}'
                req = Request(url, data=data, headers=headers, method=method)
                with urlopen(req, timeout=self.timeout) as resp:
                    return json.loads(resp.read().decode('utf-8'))
            raise

# Usage
client = TLDWClient()
client.login('user@example.com', 'password')

# Make authenticated requests
media = client.request('GET', '/media/search', params={'query': 'machine learning'})
```

### cURL

#### Single-User Mode
```bash
# Simple request with API key
curl -H "X-API-KEY: your-api-key-here" \
     http://localhost:8000/api/v1/media/search?query=test

# POST request with API key
curl -X POST \
     -H "X-API-KEY: your-api-key-here" \
     -H "Content-Type: application/json" \
     -d '{"url": "https://example.com/video"}' \
     http://localhost:8000/api/v1/media/process
```

#### Multi-User Mode
```bash
# Login and save tokens
response=$(curl -X POST \
     -H "Content-Type: application/x-www-form-urlencoded" \
     -d 'username=user@example.com&password=password' \
     http://localhost:8000/api/v1/auth/login)

# Extract token (requires jq)
token=$(echo $response | jq -r '.access_token')

# Make authenticated request
curl -H "Authorization: Bearer $token" \
     http://localhost:8000/api/v1/media/search?query=test

# Create API key
curl -X POST \
     -H "Authorization: Bearer $token" \
     -H "Content-Type: application/json" \
     -d '{"name": "My API Key", "scope": "read"}' \
     http://localhost:8000/api/v1/auth/api-keys
```

## Response Formats

### Success Response
```json
{
  "status": "success",
  "data": {
    // Response data here
  },
  "message": "Operation completed successfully"
}
```

### Error Response
```json
{
  "status": "error",
  "error": {
    "code": "UNAUTHORIZED",
    "message": "Invalid or expired token",
    "details": "Token expired at 2024-01-15T10:00:00Z"
  },
  "request_id": "req_123456"
}
```

### Paginated Response
```json
{
  "status": "success",
  "data": {
    "items": [...],
    "pagination": {
      "page": 1,
      "per_page": 20,
      "total": 100,
      "total_pages": 5,
      "has_next": true,
      "has_prev": false
    }
  }
}
```

## Error Handling

### HTTP Status Codes

| Code | Meaning | Description |
|------|---------|-------------|
| 200 | OK | Request succeeded |
| 201 | Created | Resource created successfully |
| 204 | No Content | Request succeeded with no response body |
| 400 | Bad Request | Invalid request parameters |
| 401 | Unauthorized | Missing or invalid authentication |
| 403 | Forbidden | Authenticated but not authorized |
| 404 | Not Found | Resource not found |
| 409 | Conflict | Resource conflict (e.g., duplicate username) |
| 422 | Unprocessable Entity | Validation error |
| 429 | Too Many Requests | Rate limit exceeded |
| 500 | Internal Server Error | Server error |

### Error Codes

| Code | Description | Resolution |
|------|-------------|------------|
| `INVALID_CREDENTIALS` | Wrong username/password | Check credentials |
| `TOKEN_EXPIRED` | JWT token has expired | Refresh token |
| `TOKEN_INVALID` | JWT token is malformed | Re-authenticate |
| `API_KEY_INVALID` | API key not found or revoked | Check API key |
| `RATE_LIMIT_EXCEEDED` | Too many requests | Wait and retry |
| `INSUFFICIENT_PERMISSIONS` | User lacks required permissions | Check user role |
| `REGISTRATION_DISABLED` | Registration is not enabled | Contact admin |
| `PASSWORD_TOO_WEAK` | Password doesn't meet requirements | Use stronger password |
| `SESSION_EXPIRED` | Session has expired | Login again |

### Error Handling Best Practices

1. **Always check HTTP status codes**
2. **Parse error responses for details**
3. **Implement exponential backoff for rate limits**
4. **Handle token expiration gracefully**
5. **Log errors for debugging**

## Rate Limiting

### Default Limits

- **Anonymous**: 10 requests/minute
- **Authenticated**: 60 requests/minute
- **Service Accounts**: 1000 requests/minute

### Rate Limiting

Auth endpoints enforce strict rate limits. When limits are exceeded, responses include HTTP 429 with a `Retry-After` header (in seconds). Some non-auth modules may additionally return `X-RateLimit-*` headers, but AuthNZ endpoints use `Retry-After`.

### Handling Rate Limits

```javascript
async function makeRequestWithRetry(url, options, maxRetries = 3) {
  for (let i = 0; i < maxRetries; i++) {
    const response = await fetch(url, options);

    if (response.status === 429) {
      // Get retry delay from standard header
      const retryAfter = response.headers.get('Retry-After');
      const delay = retryAfter ? parseInt(retryAfter) * 1000 : (i + 1) * 1000;

      console.log(`Rate limited. Retrying after ${delay}ms...`);
      await new Promise(resolve => setTimeout(resolve, delay));
      continue;
    }

    return response;
  }

  throw new Error('Max retries exceeded');
}
```

## Best Practices

### Security

1. **Never expose tokens in URLs** - Use headers or request body
2. **Store tokens securely** - Use secure storage, not localStorage for sensitive apps
3. **Implement token refresh** - Don't wait for expiration
4. **Use HTTPS in production** - Required for secure cookies
5. **Rotate API keys regularly** - Set up scheduled rotation
6. **Monitor failed authentication** - Watch for attacks

### Performance

1. **Cache tokens** - Don't login for every request
2. **Batch requests** - Use bulk endpoints when available
3. **Handle rate limits gracefully** - Implement backoff
4. **Use connection pooling** - Reuse HTTP connections
5. **Implement request timeouts** - Prevent hanging requests

### Error Recovery

1. **Implement retry logic** - For transient failures
2. **Handle token refresh** - Automatically refresh expired tokens
3. **Fallback strategies** - Have backup plans for failures
4. **User feedback** - Show meaningful error messages
5. **Logging** - Log errors for debugging

### Code Organization

```javascript
// Good: Centralized auth handling
class AuthService {
  constructor() {
    this.token = null;
    this.refreshToken = null;
  }

  async authenticate() {
    // Handle login, token storage, refresh
  }

  async makeAuthenticatedRequest(endpoint, options) {
    // Handle auth headers, refresh, retry
  }
}

// Bad: Scattered auth logic
async function getMedia() {
  const token = localStorage.getItem('token');
  // Auth logic repeated everywhere
}
```

## Migration from Gradio

If migrating from the old Gradio interface:

1. **API Keys**: Generate new API keys using the AuthNZ system
2. **Endpoints**: Update to new `/api/v1` endpoints
3. **Authentication**: Switch from session-based to token-based
4. **Headers**: Use standard authentication headers
5. **Response Format**: Handle new standardized JSON responses

## Testing Authentication

### Test Endpoints

```bash
# Test single-user mode (use a protected API with X-API-KEY)
curl -H "X-API-KEY: your-key" http://localhost:8000/api/v1/media/search

# Test multi-user mode (after login)
curl -H "Authorization: Bearer your-token" http://localhost:8000/api/v1/auth/me
```

### Postman Collection

Import this collection for testing:

```json
{
  "info": {
    "name": "TLDW AuthNZ",
    "schema": "https://schema.getpostman.com/json/collection/v2.1.0/collection.json"
  },
  "auth": {
    "type": "bearer",
    "bearer": [
      {
        "key": "token",
        "value": "{{access_token}}",
        "type": "string"
      }
    ]
  },
  "variable": [
    {
      "key": "base_url",
      "value": "http://localhost:8000/api/v1"
    },
    {
      "key": "access_token",
      "value": ""
    }
  ],
  "item": [
    {
      "name": "Login",
      "request": {
        "method": "POST",
        "header": [
          {"key": "Content-Type", "value": "application/x-www-form-urlencoded"}
        ],
        "body": {
          "mode": "urlencoded",
          "urlencoded": [
            {"key": "username", "value": "admin", "type": "text"},
            {"key": "password", "value": "password", "type": "text"}
          ]
        },
        "url": {
          "raw": "{{base_url}}/auth/login",
          "host": ["{{base_url}}"],
          "path": ["auth", "login"]
        }
      },
      "event": [
        {
          "listen": "test",
          "script": {
            "exec": [
              "const response = pm.response.json();",
              "pm.collectionVariables.set('access_token', response.access_token);"
            ]
          }
        }
      ]
    }
  ]
}
```

## Support

- **Documentation**: See [AuthNZ Developer Guide](../Code_Documentation/AuthNZ-Developer-Guide.md)
- **Issues**: Report at [GitHub Issues](https://github.com/rmusser01/tldw_server/issues)
- **Security**: Report security issues privately to the maintainers
