# Multi-User Deployment Guide for tldw_server

**Version**: 1.0.0
**Last Updated**: January 14, 2025

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [PostgreSQL Setup](#postgresql-setup)
4. [Redis Setup (Optional)](#redis-setup-optional)
5. [Application Configuration](#application-configuration)
6. [Migration from Single-User](#migration-from-single-user)
7. [Production Deployment](#production-deployment)
8. [Security Hardening](#security-hardening)
9. [Monitoring & Maintenance](#monitoring--maintenance)
10. [Troubleshooting](#troubleshooting)

---

## Overview

This guide covers deploying tldw_server in multi-user mode for teams and organizations. Multi-user mode provides:

- **User isolation**: Each user's data is completely separated
- **Role-based access**: Admin, moderator, and user roles
- **Quota management**: Storage limits per user
- **Audit logging**: Track all security-relevant events
- **Session management**: Control active sessions
- **Registration control**: Open, closed, or invite-only registration

### Deployment Options

1. **Docker Deployment** (Recommended for most users)
2. **Bare Metal Deployment** (For advanced users)
3. **Kubernetes Deployment** (For enterprise scale)
4. **Cloud Deployment** (AWS, GCP, Azure)

> Reverse Proxy Samples
>
> - See `../Deployment/Reverse_Proxy_Examples.md` for Nginx/Traefik guidance.
> - Sample configs included in this repo:
>   - Nginx: `Samples/Nginx/nginx.conf`
>   - Traefik: `Samples/Traefik/traefik-dynamic.yml`
> - Ensure WebSocket upgrade is enabled for `/api/v1/audio/stream/transcribe` and `/api/v1/mcp/*`.

---

## Prerequisites

### System Requirements

**Minimum**:
- CPU: 2 cores
- RAM: 4GB
- Storage: 20GB + user data
- OS: Ubuntu 20.04+ / Debian 11+ / RHEL 8+

**Recommended**:
- CPU: 4+ cores
- RAM: 8GB+
- Storage: 100GB+ SSD
- OS: Ubuntu 22.04 LTS

### Software Requirements

```bash
# Core requirements
- Python 3.9+
- PostgreSQL 13+
- Redis 6+ (optional but recommended)
- nginx or Apache (reverse proxy)
- FFmpeg (for media processing)

# Python packages (installed automatically)
- FastAPI
- asyncpg
- aioredis
- python-jose[cryptography]
- argon2-cffi
```

---

## PostgreSQL Setup

### 1. Install PostgreSQL

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install postgresql postgresql-contrib
```

**RHEL/CentOS**:
```bash
sudo dnf install postgresql postgresql-server postgresql-contrib
sudo postgresql-setup --initdb
sudo systemctl enable postgresql
sudo systemctl start postgresql
```

### 2. Create Database and User

```bash
# Switch to postgres user
sudo -u postgres psql

# Create database
CREATE DATABASE tldw_multiuser;

# Create user with password
CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'SecurePassword123!';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE tldw_multiuser TO tldw_user;

# Enable required extensions
\c tldw_multiuser
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

# Exit
\q
```

### 3. Configure PostgreSQL

Edit PostgreSQL configuration:

```bash
sudo nano /etc/postgresql/14/main/postgresql.conf
```

Recommended settings:
```ini
# Connection settings
max_connections = 200
shared_buffers = 256MB

# Performance
effective_cache_size = 1GB
maintenance_work_mem = 64MB
checkpoint_completion_target = 0.9
wal_buffers = 16MB
default_statistics_target = 100
random_page_cost = 1.1
```

Edit authentication:
```bash
sudo nano /etc/postgresql/14/main/pg_hba.conf
```

Add:
```
# Allow tldw_user to connect
host    tldw_multiuser    tldw_user    127.0.0.1/32    md5
host    tldw_multiuser    tldw_user    ::1/128         md5
```

Restart PostgreSQL:
```bash
sudo systemctl restart postgresql
```

### 4. Initialize AuthNZ Schema (after `.env` is configured)

AuthNZ schema creation and RBAC seeding are handled by the initializer:

```bash
python -m tldw_Server_API.app.core.AuthNZ.initialize
```

This bootstraps the Postgres tables for users, sessions, RBAC, orgs/teams, API keys, and usage.
For a focused walkthrough, see `Docs/User_Guides/Server/Multi-User_Postgres_Setup.md`.

---

## Redis Setup (Optional)

Redis improves performance by caching sessions and rate limit data.

### Install Redis

```bash
# Ubuntu/Debian
sudo apt install redis-server

# RHEL/CentOS
sudo dnf install redis
```

### Configure Redis

```bash
sudo nano /etc/redis/redis.conf
```

Key settings:
```ini
# Security
requirepass YourRedisPassword123!
bind 127.0.0.1 ::1

# Persistence
save 900 1
save 300 10
save 60 10000

# Memory management
maxmemory 256mb
maxmemory-policy allkeys-lru
```

Start Redis:
```bash
sudo systemctl enable redis
sudo systemctl start redis
```

---

## Application Configuration

### 1. Clone Repository

```bash
git clone https://github.com/your-repo/tldw_server.git
cd tldw_server
```

### 2. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -e .[multiplayer]
```

### 4. Configure Environment

Create `.env` file:
```bash
nano .env
```

Add configuration:
```ini
# Database
AUTH_MODE=multi_user
DATABASE_URL=postgresql://tldw_user:SecurePassword123!@localhost/tldw_multiuser
REDIS_URL=redis://:YourRedisPassword123!@localhost:6379/0

# JWT Security
JWT_SECRET_KEY=generate-a-secure-random-key-here-use-openssl-rand-base64-64
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# Registration
ENABLE_REGISTRATION=false  # Set to true for open registration
REQUIRE_REGISTRATION_CODE=true
DEFAULT_USER_ROLE=user
# DEFAULT_STORAGE_QUOTA_MB is deprecated: it sets no one's quota (setting it logs a warning).
# Per-user storage quotas are limits.storage_quota_mb values; see Docs/Operations/Usage_Quotas.md

# Rate Limiting
AUTH_RATE_LIMIT=5/minute
API_RATE_LIMIT=100/minute

# CORS (adjust for your domain)
ALLOWED_ORIGINS=https://yourdomain.com,https://www.yourdomain.com
CORS_ALLOW_CREDENTIALS=true

# Storage
MEDIA_STORAGE_PATH=/var/lib/tldw/media
TEMP_PATH=/var/lib/tldw/temp
MAX_UPLOAD_SIZE_MB=500

# Logging
LOG_LEVEL=INFO
LOG_FILE=/var/log/tldw/app.log
AUDIT_LOG_FILE=/var/log/tldw/audit.log
```

Generate secure JWT secret:
```bash
openssl rand -base64 64
```

### 5. Create Required Directories

```bash
sudo mkdir -p /var/lib/tldw/{media,temp,backups}
sudo mkdir -p /var/log/tldw
sudo chown -R $USER:$USER /var/lib/tldw /var/log/tldw
```

---

## Migration from Single-User

If you have an existing single-user deployment:

### 1. Backup Existing Data

```bash
# Create backup directory
mkdir -p backups/$(date +%Y%m%d)

# Backup databases
cp Databases/*.db backups/$(date +%Y%m%d)/

# Backup configuration
cp config.txt .env backups/$(date +%Y%m%d)/
```

### 2. Run Migration Script

```bash
python tldw_Server_API/scripts/migrate_to_multiuser.py \
  --admin-email admin@yourdomain.com \
  --admin-password YourAdminPassword123! \
  --sqlite-db ./Databases/media_summary.db
```

The script will:
- Create admin account
- Migrate existing data to PostgreSQL
- Update configuration files
- Generate nginx configuration template

### 3. Verify Migration

```bash
# Start server
python -m uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000

# Test admin login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -d "username=admin&password=YourAdminPassword123!"
```

---

## Production Deployment

### 1. Create System Service

Create systemd service file:
```bash
sudo nano /etc/systemd/system/tldw.service
```

Add:
```ini
[Unit]
Description=tldw Server API
After=network.target postgresql.service redis.service

[Service]
Type=simple
User=tldw
Group=tldw
WorkingDirectory=/opt/tldw_server
Environment="PATH=/opt/tldw_server/venv/bin"
ExecStart=/opt/tldw_server/venv/bin/uvicorn tldw_Server_API.app.main:app --host 127.0.0.1 --port 8000 --workers 4
Restart=always
RestartSec=10
StandardOutput=append:/var/log/tldw/service.log
StandardError=append:/var/log/tldw/error.log

[Install]
WantedBy=multi-user.target
```

### 2. Create Dedicated User

```bash
# Create system user
sudo useradd -r -s /bin/bash -d /opt/tldw_server tldw

# Set ownership
sudo chown -R tldw:tldw /opt/tldw_server
sudo chown -R tldw:tldw /var/lib/tldw
sudo chown -R tldw:tldw /var/log/tldw
```

### 3. Configure nginx

Install nginx:
```bash
sudo apt install nginx
```

Create site configuration:
```bash
sudo nano /etc/nginx/sites-available/tldw
```

Add:
```nginx
upstream tldw_backend {
    server 127.0.0.1:8000;
}

server {
    listen 80;
    server_name yourdomain.com;

    # Redirect to HTTPS
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/letsencrypt/live/yourdomain.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/yourdomain.com/privkey.pem;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security Headers
    add_header X-Content-Type-Options nosniff;
    add_header X-Frame-Options DENY;
    add_header X-XSS-Protection "1; mode=block";
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Request size limits
    client_max_body_size 500M;
    client_body_buffer_size 128k;

    # Timeouts
    proxy_connect_timeout 600;
    proxy_send_timeout 600;
    proxy_read_timeout 600;
    send_timeout 600;

    # Main application
    location / {
        proxy_pass http://tldw_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
        proxy_buffering off;
    }

    # WebSocket support for real-time features
    location /ws {
        proxy_pass http://tldw_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }

    # Static files (if applicable)
    location /static {
        alias /opt/tldw_server/static;
        expires 30d;
        add_header Cache-Control "public, immutable";
    }

    # Media files
    location /media {
        alias /var/lib/tldw/media;
        expires 7d;
        add_header Cache-Control "private";
        # Require authentication for media files
        auth_request /auth/verify;
    }

    # Internal auth verification endpoint
    location = /auth/verify {
        internal;
        proxy_pass http://tldw_backend/api/v1/auth/verify;
        proxy_pass_request_body off;
        proxy_set_header Content-Length "";
        proxy_set_header X-Original-URI $request_uri;
    }
}
```

Enable site:
```bash
sudo ln -s /etc/nginx/sites-available/tldw /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl reload nginx
```

### 4. SSL Certificate Setup

Using Let's Encrypt:
```bash
sudo apt install certbot python3-certbot-nginx
sudo certbot --nginx -d yourdomain.com
```

### 5. Start Services

```bash
# Enable and start tldw service
sudo systemctl enable tldw
sudo systemctl start tldw

# Check status
sudo systemctl status tldw
```

---

## Security Hardening

### 1. Firewall Configuration

```bash
# Install UFW
sudo apt install ufw

# Configure firewall
sudo ufw default deny incoming
sudo ufw default allow outgoing
sudo ufw allow ssh
sudo ufw allow 80/tcp
sudo ufw allow 443/tcp
sudo ufw enable
```

### 2. Fail2ban Setup

```bash
# Install fail2ban
sudo apt install fail2ban

# Create jail configuration
sudo nano /etc/fail2ban/jail.local
```

Add:
```ini
[tldw-auth]
enabled = true
port = http,https
filter = tldw-auth
logpath = /var/log/tldw/audit.log
maxretry = 5
findtime = 3600
bantime = 86400
```

Create filter:
```bash
sudo nano /etc/fail2ban/filter.d/tldw-auth.conf
```

Add:
```ini
[Definition]
failregex = ^.*LOGIN_FAILED.*IP:\s+<HOST>.*$
ignoreregex =
```

Restart fail2ban:
```bash
sudo systemctl restart fail2ban
```

### 3. Database Security

```sql
-- Limit connections per user
ALTER USER tldw_user CONNECTION LIMIT 50;

-- Create read-only user for backups
CREATE USER tldw_backup WITH ENCRYPTED PASSWORD 'BackupPassword123!';
GRANT CONNECT ON DATABASE tldw_multiuser TO tldw_backup;
GRANT USAGE ON SCHEMA public TO tldw_backup;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO tldw_backup;
```

### 4. Application Security

Update `.env`:
```ini
# Security settings
SECURE_COOKIES=true
SESSION_COOKIE_HTTPONLY=true
SESSION_COOKIE_SAMESITE=strict
TRUSTED_HOSTS=yourdomain.com,www.yourdomain.com

# Content Security Policy
CSP_DIRECTIVES="default-src 'self'; script-src 'self' 'unsafe-inline'"
```

---

## Monitoring & Maintenance

### 1. Health Monitoring

Set up monitoring endpoints:
```bash
# Kubernetes liveness probe
curl http://localhost:8000/api/v1/health/live

# Readiness probe
curl http://localhost:8000/api/v1/health/ready

# Metrics
curl http://localhost:8000/api/v1/health/metrics
```

### 2. Log Rotation

Create logrotate configuration:
```bash
sudo nano /etc/logrotate.d/tldw
```

Add:
```
/var/log/tldw/*.log {
    daily
    missingok
    rotate 30
    compress
    delaycompress
    notifempty
    create 640 tldw tldw
    sharedscripts
    postrotate
        systemctl reload tldw >/dev/null 2>&1 || true
    endscript
}
```

### 3. Database Backups

Create backup script:
```bash
sudo nano /usr/local/bin/tldw-backup.sh
```

Add:
```bash
#!/bin/bash
BACKUP_DIR="/var/lib/tldw/backups"
DB_NAME="tldw_multiuser"
DB_USER="tldw_backup"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
pg_dump -U $DB_USER -h localhost $DB_NAME | gzip > $BACKUP_DIR/backup_$DATE.sql.gz

# Keep only last 30 days
find $BACKUP_DIR -name "backup_*.sql.gz" -mtime +30 -delete
```

Make executable and schedule:
```bash
sudo chmod +x /usr/local/bin/tldw-backup.sh

# Add to crontab
sudo crontab -e
# Add: 0 2 * * * /usr/local/bin/tldw-backup.sh
```

### 4. Monitoring Stack

**Option 1: Prometheus + Grafana**

```yaml
# docker-compose.monitoring.yml
version: '3.8'
services:
  prometheus:
    image: prom/prometheus
    volumes:
      - ./prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    ports:
      - "9090:9090"

  grafana:
    image: grafana/grafana
    volumes:
      - grafana_data:/var/lib/grafana
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin

volumes:
  prometheus_data:
  grafana_data:
```

**Option 2: Application Performance Monitoring**

```python
# Add to .env
SENTRY_DSN=your-sentry-dsn
NEW_RELIC_LICENSE_KEY=your-license-key
```

### 5. Audit Log Analysis

Query audit logs:
```bash
# View recent login failures
curl -H "Authorization: Bearer $ADMIN_TOKEN" \
  "http://localhost:8000/api/v1/admin/audit-log?action=login_failed&days=7"

# Export audit logs
psql -U tldw_user -d tldw_multiuser -c \
  "COPY (SELECT * FROM audit_log WHERE created_at > NOW() - INTERVAL '30 days')
   TO '/tmp/audit_export.csv' CSV HEADER;"
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors

**Error**: `asyncpg.exceptions.InvalidPasswordError`

**Solution**:
```bash
# Verify credentials
psql -U tldw_user -h localhost -d tldw_multiuser

# Check pg_hba.conf
sudo nano /etc/postgresql/14/main/pg_hba.conf
```

#### 2. JWT Token Errors

**Error**: `JWT signature verification failed`

**Solution**:
```bash
# Regenerate JWT secret
openssl rand -base64 64

# Update .env
JWT_SECRET_KEY=new-secret-key

# Restart service
sudo systemctl restart tldw
```

#### 3. Permission Errors

**Error**: `Permission denied: '/var/lib/tldw/media'`

**Solution**:
```bash
# Fix ownership
sudo chown -R tldw:tldw /var/lib/tldw
sudo chmod -R 755 /var/lib/tldw
```

#### 4. High Memory Usage

**Solution**:
```bash
# Adjust worker count in service file
ExecStart=/opt/tldw_server/venv/bin/uvicorn ... --workers 2

# Tune PostgreSQL
shared_buffers = 128MB
effective_cache_size = 512MB
```

#### 5. Slow API Response

**Diagnostics**:
```bash
# Check database queries
psql -U tldw_user -d tldw_multiuser
\x
SELECT * FROM pg_stat_activity WHERE state != 'idle';

# Check indexes
\di

# Analyze tables
ANALYZE;
```

### Debug Mode

Enable debug logging:
```bash
# .env
LOG_LEVEL=DEBUG

# View logs
tail -f /var/log/tldw/app.log
```

### Performance Tuning

1. **Database Indexes**:
```sql
-- Add custom indexes for common queries
CREATE INDEX idx_media_user_created ON media_entries(user_id, created_at DESC);
CREATE INDEX idx_sessions_token ON sessions(token_hash);
```

2. **Connection Pooling**:
```python
# .env
DATABASE_POOL_SIZE=20
DATABASE_MAX_OVERFLOW=10
```

3. **Redis Caching**:
```python
# .env
CACHE_TTL_SECONDS=300
SESSION_CACHE_TTL=3600
```

---

## Support & Resources

### Documentation
- API Documentation: http://yourdomain.com/docs
- GitHub: https://github.com/your-repo/tldw_server
- Wiki: https://github.com/your-repo/tldw_server/wiki

### Getting Help
- GitHub Issues: Report bugs and request features
- Discussions: Community support
- Email: support@yourdomain.com

### Useful Commands

```bash
# Check service status
sudo systemctl status tldw

# View logs
sudo journalctl -u tldw -f

# Restart service
sudo systemctl restart tldw

# Database console
psql -U tldw_user -d tldw_multiuser

# Redis console
redis-cli -a YourRedisPassword123!

# Test endpoints
curl http://localhost:8000/api/v1/health
```

---

## Appendix: Docker Deployment

### Docker Compose Setup

```yaml
# docker-compose.yml
version: '3.8'

services:
  postgres:
    image: postgres:18-bookworm
    environment:
      POSTGRES_DB: tldw_multiuser
      POSTGRES_USER: tldw_user
      POSTGRES_PASSWORD: SecurePassword123!
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql:/docker-entrypoint-initdb.d/init.sql
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U tldw_user"]
      interval: 10s
      timeout: 5s
      retries: 5

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass YourRedisPassword123!
    volumes:
      - redis_data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5

  tldw:
    build: .
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    environment:
      DATABASE_URL: postgresql://tldw_user:SecurePassword123!@postgres/tldw_multiuser
      REDIS_URL: redis://:YourRedisPassword123!@redis:6379/0
      AUTH_MODE: multi_user
      JWT_SECRET_KEY: ${JWT_SECRET_KEY}
    volumes:
      - ./media:/app/media
      - ./logs:/app/logs
    ports:
      - "8000:8000"
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    depends_on:
      - tldw
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf:ro
      - ./certs:/etc/nginx/certs:ro
    ports:
      - "80:80"
      - "443:443"
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:
```

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    postgresql-client \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install from pyproject
COPY pyproject.toml .
RUN pip install --no-cache-dir -e .[multiplayer]

# Copy application
COPY tldw_Server_API/ ./tldw_Server_API/
COPY schema/ ./schema/

# Create non-root user
RUN useradd -m -s /bin/bash tldw && \
    chown -R tldw:tldw /app

USER tldw

# Run application
CMD ["uvicorn", "tldw_Server_API.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]
```

### Deploy with Docker

```bash
# Build and start services
docker-compose up -d

# View logs
docker-compose logs -f tldw

# Scale workers
docker-compose up -d --scale tldw=3

# Backup database
docker-compose exec postgres pg_dump -U tldw_user tldw_multiuser > backup.sql
```

---

*This deployment guide is maintained by the tldw_server team. Last updated: January 14, 2025*
