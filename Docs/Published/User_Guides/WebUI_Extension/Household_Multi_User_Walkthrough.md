# Household Multi-User Walkthrough

A complete, step-by-step guide to setting up tldw_server for a household of 3 users (1 admin + 2 regular users). Follow this from top to bottom to go from a fresh install to a working multi-user setup.

> **Companion guide**: For guardrails and parental controls after setup, see [Family_Guardian_Setup.md](../WebUI_Extension/Family_Guardian_Setup.md).

---

## 1. What You'll End Up With

By the end of this guide you will have:

- **1 admin account** (parent/household manager) with full control
- **2 regular user accounts** (family members) with isolated data
- Each user gets their own media library, notes, chats, RAG collections, and prompts
- JWT-based authentication (login with username + password, get a token)
- Registration locked down so no one else can create accounts

### How Multi-User Mode Works

| Concept | What It Means |
|---------|---------------|
| **AuthNZ DB** | Central database storing users, sessions, roles, permissions, and API keys. Shared by all users. |
| **Per-user data** | Each user gets a private directory under `Databases/user_databases/<user_id>/` containing their media DB, notes, prompts, evaluations, vector store, etc. |
| **JWT** | JSON Web Token. After logging in, users include this token in API requests. The WebUI handles this automatically. |
| **RBAC** | Role-Based Access Control. Admins can manage users and settings; regular users can only access their own data. |
| **Virtual Key** | A per-user API key with optional provider/model allowlists and usage budgets. |

---

## 2. Prerequisites

### Hardware

Any modern machine works. For a household of 3:
- **CPU**: Any x86_64 or ARM64 processor
- **RAM**: 4 GB minimum (8 GB recommended if using local transcription)
- **Disk**: 2 GB for the application + space for your media/data

### Software

| Requirement | How to Check | Install |
|-------------|-------------|---------|
| Python 3.11+ | `python3 --version` | [python.org](https://www.python.org/downloads/) |
| pip | `pip --version` | Included with Python |
| ffmpeg | `ffmpeg -version` | `brew install ffmpeg` (macOS) / `sudo apt install ffmpeg` (Ubuntu) |
| Git | `git --version` | `brew install git` / `sudo apt install git` |

### Optional

| Requirement | When You Need It |
|-------------|-----------------|
| Docker + Docker Compose | If you prefer containerized deployment |
| PostgreSQL 13+ | If you want a more robust AuthNZ backend (not needed for a small household) |
| CUDA-capable GPU | For accelerated audio transcription |

---

## 3. Choose Your Path

For a household of 3 users on a home LAN, the simplest path is **SQLite + bare metal + LAN-only**. Use this decision matrix if your situation differs:

| Decision | Option A (Simple) | Option B (Robust) | Recommendation for 3-User Household |
|----------|-------------------|-------------------|--------------------------------------|
| **AuthNZ DB** | SQLite | PostgreSQL | SQLite is fine for <10 users |
| **Runtime** | Bare metal (pip) | Docker Compose | Bare metal is simpler to debug |
| **Access** | LAN-only (http) | Remote (HTTPS + reverse proxy) | Start LAN-only, add remote later |

> **Note**: SQLite multi-user is not supported when `tldw_production=true`. For a home setup this is fine since you won't set that flag. If you later want production mode, switch to Postgres (see [Multi-User_Postgres_Setup.md](../Server/Multi-User_Postgres_Setup.md)).

This guide covers all paths. Follow the sections marked with your choice.

---

## 4. Initial Setup

### 4a. Install tldw_server

#### Bare Metal

```bash
# Clone the repository
git clone https://github.com/rmusser01/tldw_server.git tldw_server2
cd tldw_server2

# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install with multi-user extras
pip install -e ".[multiplayer]"
```

#### Docker

```bash
git clone https://github.com/rmusser01/tldw_server.git tldw_server2
cd tldw_server2

# Build and start (Postgres included in compose file)
docker compose -f Dockerfiles/docker-compose.yml up -d --build
```

#### Docker WebUI: Separate Container vs Same Container

| Approach | How it works | Pros | Cons | Recommendation |
|----------|---------------|------|------|----------------|
| **Separate WebUI container** | API runs in `app`; Next.js runs in a dedicated `webui` service via `docker-compose.webui.yml`. | Clear separation of concerns, easier upgrades/rollbacks, independent scaling and restarts, matches repo-provided compose overlays. | One extra container and one extra public port. | **Recommended** for Docker deployments. |
| **Same container (custom image)** | Run Python API + Next.js in one container using a custom entrypoint/process manager. | Single container artifact. | More complex image/runtime, harder health checks and restarts, coupled release lifecycle, not the default in this repo. | Use only if you have strict single-container constraints. |

Use the supported separate-container setup:

```bash
export NEXT_PUBLIC_API_URL=http://<server-ip>:8000
docker compose -f Dockerfiles/docker-compose.yml -f Dockerfiles/docker-compose.webui.yml up -d --build
```

Then open the WebUI at `http://<server-ip>:8080`.

### 4b. Configure Environment

Copy the template:

```bash
cp tldw_Server_API/Config_Files/.env.example tldw_Server_API/Config_Files/.env
```

Now generate secrets. Run each command and paste the output into your `.env` file:

```bash
# JWT secret (required)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"

# Session encryption key (required)
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

# MCP secrets (optional for LAN-only, required for production)
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
python3 -c "import secrets; print(secrets.token_urlsafe(32))"
```

Edit `.env` with the generated values. Here is the complete set of settings you need:

#### SQLite Path (Recommended for Households)

```bash
# --- Core Auth ---
AUTH_MODE=multi_user
DATABASE_URL=sqlite:///./Databases/users.db
JWT_SECRET_KEY=<paste your JWT secret here>
SESSION_ENCRYPTION_KEY=<paste your Fernet key here>

# --- MCP Secrets (paste generated values) ---
MCP_JWT_SECRET=<paste here>
MCP_API_KEY_SALT=<paste here>

# --- User Data ---
USER_DB_BASE_DIR=./Databases/user_databases
TLDW_SQLITE_WAL_MODE=true

# --- Registration (locked down) ---
ENABLE_REGISTRATION=false
REQUIRE_REGISTRATION_CODE=true

# --- BYOK (disable unless family members bring their own API keys) ---
BYOK_ENABLED=false

# --- Resource Governor Rate Limits ---
RG_ENABLED=true
# Tune requests.rpm / requests.burst in Config_Files/resource_governor_policies.yaml

# --- Provider Keys (add the ones you use) ---
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
```

#### Postgres Path (Alternative)

If you chose Postgres, replace the `DATABASE_URL` line:

```bash
DATABASE_URL=postgresql://tldw_user:YourStrongPassword@localhost:5432/tldw_users
```

And ensure Postgres is running (via Docker or system install). See [Multi-User_Postgres_Setup.md](../Server/Multi-User_Postgres_Setup.md) for details.

### 4c. Initialize AuthNZ and Create Admin Account

```bash
python -m tldw_Server_API.app.core.AuthNZ.initialize
```

You'll be prompted to create the admin account:

```text
Enter admin username: householder
Enter admin email: <your-admin-email>
Enter admin password: ********
Confirm password: ********
```

> **Tip**: Choose a strong password. This account has full control over all users and settings.

#### Docker Alternative

```bash
docker compose -f Dockerfiles/docker-compose.yml exec app \
  python -m tldw_Server_API.app.core.AuthNZ.initialize
```

### 4d. Start the Server

#### Bare Metal

```bash
python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000 --reload
```

> **`--host 0.0.0.0`** makes the server accessible from other devices on your LAN (not just localhost). Omit this if you only want access from the machine itself.

#### Docker

The server starts automatically with `docker compose up`.

### 4e. Verify Admin Login

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=householder" \
  --data-urlencode "password=YOUR_ADMIN_PASSWORD" | python3 -m json.tool
```

Expected response (truncated):

```json
{
    "access_token": "eyJ...",
    "refresh_token": "eyJ...",
    "token_type": "bearer",
    "expires_in": 1800
}
```

Save the `access_token` value for the next steps:

```bash
export ADMIN_JWT="eyJ..."
```

---

## 5. Create Family Member Accounts

### 5a. Option A: Admin Creates Accounts Directly (Recommended)

This is the simplest approach for a household. The admin creates accounts for everyone.

**Create User 2 (e.g., "alex")**:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/admin/users \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alex",
    "email": "alex@example.com",
    "password": "AlexSecurePass123!",
    "is_active": true
  }' | python3 -m json.tool
```

**Create User 3 (e.g., "jordan")**:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/admin/users \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "username": "jordan",
    "email": "jordan@example.com",
    "password": "JordanSecurePass123!",
    "is_active": true
  }' | python3 -m json.tool
```

Each response returns the created user's details including their `id`.

### 5b. Option B: Registration Codes (Self-Service)

If you'd rather let family members register themselves:

1. **Generate registration codes** (one per person):

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/admin/registration-codes \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{
    "max_uses": 1,
    "expiry_days": 7,
    "role_to_grant": "user"
  }' | python3 -m json.tool
```

The response includes a `code` value to share with the user.

2. **Temporarily enable registration** in `.env`:

```bash
ENABLE_REGISTRATION=true
REQUIRE_REGISTRATION_CODE=true
```

Restart the server for the change to take effect.

3. **Share the code** with each family member. They register via:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "alex",
    "email": "alex@example.com",
    "password": "AlexSecurePass123!",
    "registration_code": "THE_CODE_YOU_GENERATED"
  }' | python3 -m json.tool
```

4. **Lock registration again** after everyone has signed up:

```bash
ENABLE_REGISTRATION=false
```

Restart the server.

### 5c. Verification Checklist

Run these checks to confirm everything works.

**Each user can log in**:

```bash
# Alex logs in
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=alex" \
  --data-urlencode "password=AlexSecurePass123!" | python3 -m json.tool

# Jordan logs in
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=jordan" \
  --data-urlencode "password=JordanSecurePass123!" | python3 -m json.tool
```

**Each user can see their own profile**:

```bash
export ALEX_JWT="<paste alex's access_token>"
curl -s http://127.0.0.1:8000/api/v1/users/me/profile \
  -H "Authorization: Bearer $ALEX_JWT" | python3 -m json.tool
```

Expected: returns profile data with `user.username = "alex"` and `user.role = "user"`.

**Per-user directories exist**:

```bash
# Bare metal
ls -la Databases/user_databases/

# Docker
docker compose -f Dockerfiles/docker-compose.yml exec app ls -la /app/Databases/user_databases/
```

You should see directories for each user ID (created on first access to any data endpoint).

> **Note**: Per-user directories are created lazily. If a user just logged in but hasn't accessed any data endpoints yet, their directory may not exist yet. That's normal.

**Regular users cannot access admin endpoints**:

```bash
curl -s -X GET http://127.0.0.1:8000/api/v1/admin/users \
  -H "Authorization: Bearer $ALEX_JWT"
```

Expected: `403 Forbidden` response.

---

## 6. Configure Guardrails

With accounts created, apply sensible defaults for a household.

### Close Registration

If you haven't already, ensure registration is locked:

```bash
# In .env
ENABLE_REGISTRATION=false
REQUIRE_REGISTRATION_CODE=true
```

### Rate Limits

These defaults prevent any single user from overwhelming the server:

```bash
# In .env
RG_ENABLED=true
# Tune requests.rpm / requests.burst in Config_Files/resource_governor_policies.yaml
```

### Disable BYOK

Unless family members need to add their own API keys:

```bash
# In .env
BYOK_ENABLED=false
```

If you do want BYOK, you can restrict which providers are allowed:

```bash
BYOK_ENABLED=true
BYOK_ALLOWED_PROVIDERS=openai,anthropic
```

See [BYOK_User_Guide.md](../Server/BYOK_User_Guide.md) for details.

### Roles in This Guide

The built-in user record role field supports `user`, `admin`, and `service`. For a household setup, keep family members as `user`.

If you need to promote a trusted person to admin:

```bash
curl -s -X PUT http://127.0.0.1:8000/api/v1/admin/users/<USER_ID> \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"role": "admin"}'
```

### Virtual Keys (Optional)

Virtual keys let you set per-user budgets and model restrictions. See [Family_Guardian_Setup.md](../WebUI_Extension/Family_Guardian_Setup.md) for detailed virtual key setup.

---

## 7. Daily Use

### How Users Log In

**WebUI**:
- Bare metal frontend dev server: `http://<server-ip>:3000`
- Docker with `docker-compose.webui.yml`: `http://<server-ip>:8080`
- API docs only: `http://<server-ip>:8000/docs`

The default `Dockerfiles/docker-compose.yml` stack is API-only (no WebUI container). Add `Dockerfiles/docker-compose.webui.yml` if you want browser UI in Docker.

**API**: POST to `/api/v1/auth/login` using `application/x-www-form-urlencoded` fields (`username`, `password`), receive JWTs, include access token as `Authorization: Bearer <token>` in subsequent requests.

### Token Refresh

JWTs expire after a configurable period (default: 30 minutes via `ACCESS_TOKEN_EXPIRE_MINUTES=30`). When a token expires:
- **WebUI**: Automatically refreshes or prompts for re-login
- **API**: Call `/api/v1/auth/refresh` with the refresh token to get a new token pair

### What Each User Can Do

| Feature | Regular User | Admin |
|---------|-------------|-------|
| Media ingestion (video, audio, PDF, etc.) | Own library only | Own library |
| Chat with LLMs | Yes | Yes |
| RAG search | Own collections | Own collections |
| Notes and prompts | Own data | Own data |
| Character chat | Own sessions | Own sessions |
| Manage other users | No | Yes |
| View all sessions | No | Yes |
| Adjust server settings | No | Yes |
| Create/delete accounts | No | Yes |

### What the Admin Can Do

- Create, deactivate, and delete user accounts
- Revoke sessions for specific users
- Create/list/revoke registration codes
- Configure virtual keys and permission overrides
- Access server health and metrics endpoints

---

## 8. Optional: Remote Access

If you want to access your household server from outside your home network.

> **Warning**: Exposing your server to the internet requires proper security. Do not skip TLS.

### Reverse Proxy with nginx (HTTPS)

Install nginx and create a site config:

```nginx
server {
    listen 443 ssl http2;
    server_name tldw.yourdomain.com;

    ssl_certificate     /etc/letsencrypt/live/tldw.yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/tldw.yourdomain.com/privkey.pem;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }

    # WebSocket support (for streaming transcription and MCP)
    location /api/v1/audio/stream/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 3600s;
    }

    location /api/v1/mcp/ {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
    }
}

# Redirect HTTP to HTTPS
server {
    listen 80;
    server_name tldw.yourdomain.com;
    return 301 https://$host$request_uri;
}
```

### TLS with Let's Encrypt

```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d tldw.yourdomain.com
```

### CORS Configuration

Add your domain to the allowed origins in `.env`:

```bash
ALLOWED_ORIGINS=https://tldw.yourdomain.com
```

For multiple origins (e.g., if the WebUI is on a different port):

```bash
ALLOWED_ORIGINS='["https://tldw.yourdomain.com", "https://tldw.yourdomain.com:8080"]'
```

See [Production_Hardening_Checklist.md](../Server/Production_Hardening_Checklist.md) for more details.

---

## 9. Backups

### SQLite Backups

The simplest approach is to copy the database files while the server is stopped (or use SQLite's `.backup` command):

```bash
# Bare metal: stop the server first (or use WAL mode for hot backups)
cp -r Databases/ Databases_backup_$(date +%Y%m%d)/

# Docker: stop app, copy from container, restart
docker compose -f Dockerfiles/docker-compose.yml stop app
docker cp tldw-app:/app/Databases ./Databases_backup_$(date +%Y%m%d)
docker compose -f Dockerfiles/docker-compose.yml start app
```

For automated continuous backups, see [Backups_Using_Litestream.md](../Server/Backups_Using_Litestream.md).

### Postgres Backups

```bash
# Bare metal
pg_dump -U tldw_user -h localhost tldw_users > authnz_backup_$(date +%Y%m%d).sql

# Docker
docker compose -f Dockerfiles/docker-compose.yml exec -T postgres \
  pg_dump -U tldw_user tldw_users > authnz_backup_$(date +%Y%m%d).sql
```

### Per-User Data

Always include the per-user directory in backups:

```bash
# Bare metal: this contains all user media DBs, notes, prompts, vector stores, etc.
cp -r Databases/user_databases/ user_data_backup_$(date +%Y%m%d)/

# Docker
docker cp tldw-app:/app/Databases/user_databases ./user_data_backup_$(date +%Y%m%d)
```

### Backup Schedule Recommendation

For a household:
- **Weekly**: Full backup of `Databases/` directory
- **Before upgrades**: Always back up before updating tldw_server

---

## 10. Troubleshooting

### "JWT_SECRET_KEY not set" or server won't start

Your `.env` file is missing required secrets or isn't being loaded.

**Fix**: Verify `.env` exists in the repo root and contains `JWT_SECRET_KEY`, `SESSION_ENCRYPTION_KEY`, and `AUTH_MODE=multi_user`. Ensure the values are not the template defaults.

### "Database is locked" errors

SQLite can't handle many concurrent writes.

**Fix**: Ensure `TLDW_SQLITE_WAL_MODE=true` is set in `.env`. For persistent issues with >5 concurrent users, switch to Postgres.

### Login returns 401 Unauthorized

Wrong username or password.

**Fix**: Double-check credentials. If you've forgotten the admin password, reset it:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/forgot-password \
  -H "Content-Type: application/json" \
  -d '{"email":"<your-admin-email>"}'
```

Then submit the token from the reset email:

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/reset-password \
  -H "Content-Type: application/json" \
  -d '{
    "token":"<RESET_TOKEN_FROM_EMAIL>",
    "new_password":"NewSecurePass123!"
  }'
```

If email delivery is not configured, `forgot-password` still returns a generic success message but no reset email will be sent. Configure SMTP/email first, or use `/api/v1/users/change-password` for users who still know their current password.

### Regular user gets 403 on an endpoint

The user's role doesn't have permission for that endpoint.

**Fix**: This is expected for admin endpoints. If a regular user needs access to a specific feature, check available permissions:

```bash
curl -H "Authorization: Bearer $ADMIN_JWT" \
  http://127.0.0.1:8000/api/v1/admin/permissions
```

Then grant an override if appropriate:

```bash
curl -X POST http://127.0.0.1:8000/api/v1/admin/users/<USER_ID>/overrides \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"permission_name":"<permission>","effect":"allow"}'
```

### Per-user directories not created

Directories under `Databases/user_databases/` are created lazily on first data access.

**Fix**: Have the user make any data request (e.g., search media, create a note). The directory will be created automatically.

### Token expired / "Token is invalid"

JWTs expire after a set period.

**Fix**: Log in again to get a fresh token. For the WebUI, refresh the page or log out and back in.

### Can't connect from another device on LAN

The server may only be listening on `127.0.0.1`.

**Fix**: Start uvicorn with `--host 0.0.0.0`:

```bash
python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000
```

Then access via `http://<server-ip>:8000` from other devices. Find your server's IP with `hostname -I` (Linux) or `ipconfig getifaddr en0` (macOS).

### How to check active sessions

```bash
curl -s http://127.0.0.1:8000/api/v1/admin/users/<USER_ID>/sessions \
  -H "Authorization: Bearer $ADMIN_JWT" | python3 -m json.tool
```

### How to deactivate a user

```bash
curl -s -X PUT http://127.0.0.1:8000/api/v1/admin/users/<USER_ID> \
  -H "Authorization: Bearer $ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"is_active": false}'
```

---

## 11. Quick Reference Card

### Commands Used in This Guide

| Step | Command |
|------|---------|
| Install | `pip install -e ".[multiplayer]"` |
| Generate JWT secret | `python3 -c "import secrets; print(secrets.token_urlsafe(32))"` |
| Generate Fernet key | `python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"` |
| Initialize AuthNZ | `python -m tldw_Server_API.app.core.AuthNZ.initialize` |
| Start server (LAN) | `python -m uvicorn tldw_Server_API.app.main:app --host 0.0.0.0 --port 8000` |
| Start server (local only) | `python -m uvicorn tldw_Server_API.app.main:app --reload` |

### Key API Endpoints

| Action | Method | Endpoint |
|--------|--------|----------|
| Log in | POST | `/api/v1/auth/login` |
| Register (if enabled) | POST | `/api/v1/auth/register` |
| View own profile | GET | `/api/v1/users/me/profile` |
| Refresh token | POST | `/api/v1/auth/refresh` |
| Change own password | POST | `/api/v1/users/change-password` |
| Admin: create user | POST | `/api/v1/admin/users` |
| Admin: list users | GET | `/api/v1/admin/users` |
| Admin: update user | PUT | `/api/v1/admin/users/<id>` |
| Admin: create registration code | POST | `/api/v1/admin/registration-codes` |
| Admin: list permissions | GET | `/api/v1/admin/permissions` |
| Admin: set permission override | POST | `/api/v1/admin/users/<id>/overrides` |
| Admin: create virtual key | POST | `/api/v1/admin/users/<id>/virtual-keys` |
| Admin: view user sessions | GET | `/api/v1/admin/users/<id>/sessions` |
| Admin: revoke all user sessions | POST | `/api/v1/admin/users/<id>/sessions/revoke-all` |

### Key Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `AUTH_MODE` | Authentication mode | `multi_user` |
| `DATABASE_URL` | AuthNZ database location | `sqlite:///./Databases/users.db` |
| `JWT_SECRET_KEY` | Signs JWT tokens | `<32+ char random string>` |
| `SESSION_ENCRYPTION_KEY` | Encrypts session data | `<Fernet key>` |
| `USER_DB_BASE_DIR` | Per-user data root | `./Databases/user_databases` |
| `ENABLE_REGISTRATION` | Allow new signups | `false` |
| `REQUIRE_REGISTRATION_CODE` | Require code to register | `true` |
| `BYOK_ENABLED` | Allow user-provided API keys | `false` |
| `RG_ENABLED` | Enable Resource Governor rate limiting | `true` |
| `RG_POLICY_PATH` | Policy file containing requests.rpm / requests.burst limits | `tldw_Server_API/Config_Files/resource_governor_policies.yaml` |
| `ALLOWED_ORIGINS` | CORS allowed origins | `https://tldw.yourdomain.com` |

### Validated Commands (AuthNZ API-Guide Style)

#### Login (Multi-User Mode)
`POST /api/v1/auth/login`

Request content type: `application/x-www-form-urlencoded`

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=householder" \
  --data-urlencode "password=YOUR_ADMIN_PASSWORD" | python3 -m json.tool
```

#### Refresh Token
`POST /api/v1/auth/refresh`

Request body: JSON

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/auth/refresh \
  -H "Content-Type: application/json" \
  -d '{"refresh_token":"<REFRESH_TOKEN>"}' | python3 -m json.tool
```

#### Create User (Admin)
`POST /api/v1/admin/users`

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/admin/users \
  -H "Authorization: Bearer <ADMIN_JWT>" \
  -H "Content-Type: application/json" \
  -d '{
    "username":"alex",
    "email":"alex@example.com",
    "password":"AlexSecurePass123!",
    "role":"user",
    "is_active": true
  }' | python3 -m json.tool
```

#### Create Registration Code (Admin)
`POST /api/v1/admin/registration-codes`

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/admin/registration-codes \
  -H "Authorization: Bearer <ADMIN_JWT>" \
  -H "Content-Type: application/json" \
  -d '{"max_uses":1,"expiry_days":7,"role_to_grant":"user"}' | python3 -m json.tool
```

#### Revoke All Sessions for a User (Admin)
`POST /api/v1/admin/users/<USER_ID>/sessions/revoke-all`

```bash
curl -s -X POST http://127.0.0.1:8000/api/v1/admin/users/<USER_ID>/sessions/revoke-all \
  -H "Authorization: Bearer <ADMIN_JWT>" | python3 -m json.tool
```

---

## Related Documentation

- [Family_Guardian_Setup.md](../WebUI_Extension/Family_Guardian_Setup.md) - Guardrails, virtual keys, and parental controls
- [Authentication_Setup.md](../Server/Authentication_Setup.md) - Single-user and multi-user auth overview
- [Multi-User_SQLite_Setup.md](../Server/Multi-User_SQLite_Setup.md) - SQLite multi-user technical details
- [Multi-User_Postgres_Setup.md](../Server/Multi-User_Postgres_Setup.md) - PostgreSQL AuthNZ setup
- [Multi-User_Deployment_Guide.md](../Server/Multi-User_Deployment_Guide.md) - Production multi-user deployment
- [Production_Hardening_Checklist.md](../Server/Production_Hardening_Checklist.md) - Security hardening for internet-facing deployments
- [BYOK_User_Guide.md](../Server/BYOK_User_Guide.md) - Bring Your Own Key configuration
- [Backups_Using_Litestream.md](../Server/Backups_Using_Litestream.md) - Continuous SQLite backup with Litestream
