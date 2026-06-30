# MCP Unified - System Administrator Guide

> Part of the MCP Unified documentation set. See `Docs/MCP/Unified/README.md` for the full guide index.

## Table of Contents
1. [Deployment Overview](#deployment-overview)
2. [Installation](#installation)
3. [Configuration](#configuration)
4. [Security Setup](#security-setup)
5. [Performance Tuning](#performance-tuning)
6. [Monitoring & Metrics](#monitoring--metrics)
7. [Backup & Recovery](#backup--recovery)
8. [Troubleshooting](#troubleshooting)
9. [Maintenance](#maintenance)
10. [Security Checklist](#security-checklist)

## Deployment Overview

> **Current state:** MCP Unified is operated as part of TLDW Server at
> `/api/v1/mcp`. The package-local gateway boundary is internal/experimental;
> this guide does not describe a separately shipped standalone gateway process.

The MCP Unified module is a critical component of the TLDW server that provides:
- Secure API access to media processing capabilities
- WebSocket and HTTP endpoints for client connections
- Module-based architecture for extensibility
- Enterprise-grade security and monitoring

### System Requirements
- **Python**: 3.9+ (3.11+ recommended)
- **Memory**: Minimum 2GB, 4GB+ recommended
- **Storage**: Depends on media volume (10GB minimum for system)
- **Network**: Open ports 8000 (HTTP), 8001 (WebSocket, optional)
- **Database**: SQLite (default) or PostgreSQL (production)

## Installation

### 1. Install Dependencies
```bash
# Core dependencies
pip install -e .

# Production dependencies
pip install gunicorn uvloop httptools redis aioredis prometheus-client

# Optional: PostgreSQL support
pip install asyncpg "psycopg[binary]"
```

### 2. Set Up Directory Structure
```bash
# Create required directories
mkdir -p /var/log/tldw/mcp
mkdir -p /var/lib/tldw/databases
mkdir -p /etc/tldw/config

# Set permissions
chown -R tldw:tldw /var/log/tldw
chown -R tldw:tldw /var/lib/tldw
chmod 750 /var/log/tldw
chmod 750 /var/lib/tldw
```

### 3. Install as Service (systemd)
```ini
# /etc/systemd/system/tldw-mcp.service
[Unit]
Description=TLDW MCP Unified Server
After=network.target postgresql.service redis.service

[Service]
Type=exec
User=tldw
Group=tldw
WorkingDirectory=/opt/tldw_server
Environment="PATH=/opt/tldw_server/venv/bin"
Environment="MCP_CONFIG_PATH=/etc/tldw/config/mcp.env"
EnvironmentFile=/etc/tldw/config/mcp.env
ExecStart=/opt/tldw_server/venv/bin/gunicorn \
    tldw_Server_API.app.main:app \
    --worker-class uvicorn.workers.UvicornWorker \
    --workers 4 \
    --bind 0.0.0.0:8000 \
    --access-logfile /var/log/tldw/mcp/access.log \
    --error-logfile /var/log/tldw/mcp/error.log \
    --log-level info
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

## Configuration

### Environment Variables

Create `/etc/tldw/config/mcp.env`:
```bash
# CRITICAL SECURITY SETTINGS - MUST CHANGE IN PRODUCTION
AUTH_MODE=multi_user
SINGLE_USER_API_KEY=""  # single_user only; leave unset for multi_user
MCP_JWT_SECRET="$(openssl rand -base64 32)"
MCP_API_KEY_SALT="$(openssl rand -base64 32)"

# TLDW Server process binding is controlled by gunicorn/uvicorn and systemd.
MCP_LOG_LEVEL=INFO
MCP_LOG_FILE=/var/log/tldw/mcp/mcp.log

# Database
MCP_DATABASE_URL=postgresql+asyncpg://tldw:password@localhost/tldw_mcp
MCP_DATABASE_POOL_SIZE=20
MCP_DATABASE_POOL_TIMEOUT=30

# Redis (for distributed deployments)
MCP_REDIS_URL=redis://localhost:6379/0
MCP_REDIS_POOL_SIZE=10

# Rate Limiting
MCP_RATE_LIMIT_ENABLED=true
MCP_RATE_LIMIT_RPM=60
MCP_RATE_LIMIT_BURST=100
RG_BACKEND=redis  # or memory
REDIS_URL=redis://localhost:6379/0

# Security
MCP_CORS_ORIGINS=["https://app.example.com"]
MCP_TRUST_X_FORWARDED=true
MCP_TRUSTED_PROXY_IPS=["127.0.0.1", "10.0.0.0/8"]
MCP_TRUSTED_PROXY_DEPTH=1
MCP_HTTP_MAX_BODY_BYTES=10485760  # 10MB

# Authentication
MCP_JWT_ALGORITHM=HS256
MCP_JWT_ACCESS_EXPIRE=30
MCP_JWT_REFRESH_EXPIRE=7
MCP_WS_AUTH_REQUIRED=true
MCP_WS_ALLOW_QUERY_AUTH=false

# Monitoring
MCP_METRICS_ENABLED=true
MCP_METRICS_PORT=9090
MCP_HEALTH_PATH=/api/v1/mcp/health

# Module Configuration
MCP_MODULES_CONFIG=/etc/tldw/config/mcp_modules.yaml
MCP_MODULE_TIMEOUT=60
MCP_MODULE_MAX_RETRIES=3
MCP_ENABLE_MEDIA_MODULE=true
```

### Nginx Configuration

```nginx
# /etc/nginx/sites-available/tldw-mcp
upstream mcp_backend {
    least_conn;
    server 127.0.0.1:8000 max_fails=3 fail_timeout=30s;
    server 127.0.0.1:8001 max_fails=3 fail_timeout=30s backup;
}

server {
    listen 443 ssl http2;
    server_name mcp.example.com;

    ssl_certificate /etc/ssl/certs/mcp.example.com.crt;
    ssl_certificate_key /etc/ssl/private/mcp.example.com.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;
    add_header Strict-Transport-Security "max-age=31536000; includeSubDomains" always;

    # Rate limiting
    limit_req_zone $binary_remote_addr zone=mcp_limit:10m rate=10r/s;
    limit_req zone=mcp_limit burst=20 nodelay;

    # WebSocket support
    location /api/v1/mcp/ws {
        proxy_pass http://mcp_backend;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_read_timeout 86400;
        proxy_send_timeout 86400;
    }

    # HTTP API
    location /api/v1/mcp {
        proxy_pass http://mcp_backend;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # Timeouts
        proxy_connect_timeout 60s;
        proxy_send_timeout 60s;
        proxy_read_timeout 60s;

        # Buffering
        proxy_buffering off;
        proxy_request_buffering off;

        # Max body size
        client_max_body_size 10M;
    }

    # Metrics endpoint (internal only)
    location /metrics {
        allow 10.0.0.0/8;
        deny all;
        proxy_pass http://127.0.0.1:9090/metrics;
    }
}
```

## Security Setup

### 1. Generate Secure Secrets
```bash
#!/bin/bash
# generate-secrets.sh

# Generate JWT secret
JWT_SECRET=$(openssl rand -base64 64 | tr -d '\n')
echo "MCP_JWT_SECRET='$JWT_SECRET'" >> /etc/tldw/config/mcp.env

# Generate API key salt
API_SALT=$(openssl rand -base64 32 | tr -d '\n')
echo "MCP_API_KEY_SALT='$API_SALT'" >> /etc/tldw/config/mcp.env

# Generate database password
DB_PASS=$(openssl rand -base64 24 | tr -d '\n')
echo "MCP_DATABASE_PASSWORD='$DB_PASS'" >> /etc/tldw/config/mcp.env

# Secure the file
chmod 600 /etc/tldw/config/mcp.env
chown tldw:tldw /etc/tldw/config/mcp.env
```

### 2. Set Up SSL/TLS
```bash
# Generate self-signed certificate (development)
openssl req -x509 -nodes -days 365 -newkey rsa:2048 \
    -keyout /etc/ssl/private/mcp.key \
    -out /etc/ssl/certs/mcp.crt

# Or use Let's Encrypt (production)
certbot certonly --nginx -d mcp.example.com
```

### 3. Configure Firewall
```bash
# UFW example
ufw allow 22/tcp        # SSH
ufw allow 443/tcp       # HTTPS
ufw allow 8000/tcp      # MCP API (if direct access needed)
ufw allow from 10.0.0.0/8 to any port 9090  # Metrics (internal only)
ufw enable
```

### 4. Enforce IP Allowlists / Denylists
- Allow only trusted networks by setting environment variables:
  ```bash
  MCP_ALLOWED_IPS="10.0.0.0/24,203.0.113.42"
  MCP_BLOCKED_IPS="0.0.0.0/0"
  MCP_TRUST_X_FORWARDED=true   # only when running behind a trusted proxy
  MCP_TRUSTED_PROXY_DEPTH=1    # number of proxies that append to X-Forwarded-For
  ```
- The server will return `403` for HTTP requests and close WebSockets with code `1008` when a client IP is outside the allowlist.
- Keep complex network policy (security groups, NGINX allow/deny blocks) at the edge. The in-app guard is a defence-in-depth layer and ideal for local installs.

### 5. Require Client Certificates (mTLS)
- Enable client-certificate enforcement when fronting the service with an mTLS-capable proxy:
  ```bash
  MCP_CLIENT_CERT_REQUIRED=true
  MCP_CLIENT_CERT_HEADER="X-SSL-Client-Verify"   # or X-Client-Cert
  MCP_CLIENT_CERT_HEADER_VALUE="SUCCESS"         # expected value from the proxy
  MCP_CLIENT_CA_BUNDLE="/etc/tldw/config/client-ca.pem"  # optional hint for process managers
  ```
- The application verifies the configured header on both HTTP and WebSocket requests and will reject connections lacking a valid certificate assertion.
- Configure your proxy (Nginx, Traefik, Envoy) to terminate TLS, validate client certificates against the CA bundle, and forward the verification headers.

### 6. Tighten WebSocket Authentication
- Query-string authentication is disabled by default (`MCP_WS_ALLOW_QUERY_AUTH=false`) and the server now requires an `Authorization` header or `X-API-Key`.
- Production deployments should keep `MCP_WS_AUTH_REQUIRED=true` (default) to block anonymous WebSocket sessions.
- Configure heartbeat behaviour if necessary:
  ```bash
  MCP_WS_PING_INTERVAL=30
  MCP_WS_PING_TIMEOUT=60
  ```

### 7. Bound HTTP Payload Size
- Large ingestion payloads can exhaust resources. Set a ceiling that matches your deployment profile:
  ```bash
  MCP_HTTP_MAX_BODY_BYTES=1048576   # 1 MiB
  ```
- Requests exceeding the limit result in `413 Payload Too Large`, preventing oversized tool invocations from reaching the business logic.

### 8. Set Up User Management
```python
# create_admin.py
import asyncio
from tldw_Server_API.app.core.MCP_unified.auth import JWTManager

async def create_admin():
    jwt_manager = JWTManager()

    # Create admin user
    admin_token = jwt_manager.create_access_token(
        subject="admin",
        username="admin",
        roles=["admin"],
        permissions=["*"]
    )

    print(f"Admin token: {admin_token}")

    # Save to secure location
    with open("/etc/tldw/config/admin_token", "w") as f:
        f.write(admin_token)

    os.chmod("/etc/tldw/config/admin_token", 0o600)

asyncio.run(create_admin())
```

## Performance Tuning

### 1. Database Optimization
```sql
-- PostgreSQL optimizations
ALTER SYSTEM SET shared_buffers = '256MB';
ALTER SYSTEM SET effective_cache_size = '1GB';
ALTER SYSTEM SET maintenance_work_mem = '64MB';
ALTER SYSTEM SET checkpoint_completion_target = 0.9;
ALTER SYSTEM SET wal_buffers = '16MB';
ALTER SYSTEM SET default_statistics_target = 100;
ALTER SYSTEM SET random_page_cost = 1.1;

-- Create indexes
CREATE INDEX idx_media_created ON media(created_at DESC);
CREATE INDEX idx_media_type ON media(media_type);
CREATE INDEX idx_sessions_user ON sessions(user_id);
CREATE INDEX idx_sessions_expires ON sessions(expires_at);
```

### 2. Python Optimization
```bash
# Use uvloop for better async performance
pip install uvloop

# In your code
import uvloop
uvloop.install()
```

### 3. System Tuning
```bash
# /etc/sysctl.d/99-tldw.conf
# Network optimizations
net.core.somaxconn = 65535
net.ipv4.tcp_max_syn_backlog = 8192
net.ipv4.tcp_syncookies = 1
net.ipv4.tcp_tw_reuse = 1
net.ipv4.tcp_fin_timeout = 30
net.ipv4.ip_local_port_range = 10000 65000

# File descriptors
fs.file-max = 100000

# Apply settings
sysctl -p /etc/sysctl.d/99-tldw.conf
```

### 4. Connection Pooling
```python
# Configure in environment
MCP_DATABASE_POOL_SIZE=20
MCP_DATABASE_POOL_TIMEOUT=30
MCP_DATABASE_POOL_RECYCLE=1800
MCP_REDIS_POOL_SIZE=10
```

## Monitoring & Metrics

### 1. Prometheus Configuration
```yaml
# /etc/prometheus/prometheus.yml
scrape_configs:
  - job_name: 'mcp_unified'
    static_configs:
      - targets: ['localhost:9090']
    metrics_path: '/metrics'
    scrape_interval: 30s
```

### 2. Grafana Dashboard
Import the provided dashboard JSONs from `Docs/Deployment/Monitoring/` (or the curated samples in `Samples/Grafana/`) to bootstrap observability.

Key metrics to monitor:
- Request rate and latency
- WebSocket connections
- Authentication failures
- Rate limit violations
- Module health status
- Database connection pool usage
- Memory and CPU usage

### 3. Alerting Rules
```yaml
# /etc/prometheus/alerts/mcp.yml
groups:
  - name: mcp_alerts
    rules:
      - alert: HighErrorRate
        expr: rate(mcp_requests_failed_total[5m]) > 0.05
        for: 5m
        annotations:
          summary: "High error rate detected"

      - alert: AuthenticationFailures
        expr: rate(mcp_auth_failures_total[5m]) > 10
        for: 2m
        annotations:
          summary: "High authentication failure rate"

      - alert: DatabasePoolExhaustion
        expr: mcp_database_pool_available < 2
        for: 5m
        annotations:
          summary: "Database connection pool nearly exhausted"
```

### 4. Log Aggregation
```bash
# Configure rsyslog to forward logs
# /etc/rsyslog.d/tldw-mcp.conf
module(load="imfile")
input(type="imfile"
      File="/var/log/tldw/mcp/*.log"
      Tag="tldw-mcp"
      Severity="info"
      Facility="local0")

*.* @@logserver.example.com:514
```

## Backup & Recovery

### 1. Database Backup
```bash
#!/bin/bash
# /usr/local/bin/backup-mcp.sh

BACKUP_DIR="/var/backups/tldw/mcp"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup PostgreSQL
pg_dump -h localhost -U tldw tldw_mcp | gzip > $BACKUP_DIR/db_$DATE.sql.gz

# Backup configuration
tar czf $BACKUP_DIR/config_$DATE.tar.gz /etc/tldw/config/

# Backup JWT keys and secrets (encrypted)
tar czf - /etc/tldw/config/mcp.env | \
    openssl enc -aes-256-cbc -salt -pass file:/etc/tldw/.backup-pass > \
    $BACKUP_DIR/secrets_$DATE.tar.gz.enc

# Rotate old backups (keep 30 days)
find $BACKUP_DIR -type f -mtime +30 -delete

# Sync to remote storage (optional)
# aws s3 sync $BACKUP_DIR s3://backup-bucket/tldw/mcp/
```

### 2. Recovery Procedure
```bash
#!/bin/bash
# /usr/local/bin/restore-mcp.sh

BACKUP_DATE=$1
BACKUP_DIR="/var/backups/tldw/mcp"

# Stop services
systemctl stop tldw-mcp

# Restore database
gunzip < $BACKUP_DIR/db_$BACKUP_DATE.sql.gz | psql -h localhost -U tldw tldw_mcp

# Restore configuration
tar xzf $BACKUP_DIR/config_$BACKUP_DATE.tar.gz -C /

# Restore secrets
openssl enc -d -aes-256-cbc -pass file:/etc/tldw/.backup-pass \
    -in $BACKUP_DIR/secrets_$BACKUP_DATE.tar.gz.enc | tar xzf - -C /

# Start services
systemctl start tldw-mcp
```

## Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory usage
ps aux | grep -E "gunicorn|uvicorn" | awk '{sum+=$6} END {print sum/1024 " MB"}'

# Adjust worker count
MCP_WORKERS=2  # Reduce if memory constrained

# Enable memory profiling
MCP_MEMORY_PROFILING=true
```

#### Database Connection Errors
```bash
# Check connection pool status
curl http://localhost:8000/api/v1/mcp/metrics | grep database_pool

# Increase pool size
MCP_DATABASE_POOL_SIZE=50
MCP_DATABASE_POOL_TIMEOUT=45

# Check PostgreSQL connections
psql -U postgres -c "SELECT count(*) FROM pg_stat_activity WHERE datname='tldw_mcp';"
```

#### WebSocket Disconnections
```bash
# Check nginx timeout settings
proxy_read_timeout 86400;
proxy_send_timeout 86400;

# Check system limits
ulimit -n  # Should be > 65536

# Monitor WebSocket connections
ss -tan | grep :8000 | wc -l
```

#### Authentication Failures
```bash
# Check JWT secret is set
echo $MCP_JWT_SECRET | wc -c  # Should be > 32

# Verify time sync (for JWT validation)
timedatectl status

# Check audit logs
grep "auth_failure" /var/log/tldw/mcp/audit.log | tail -20
```

### Debug Mode
```bash
# Enable debug logging
export MCP_LOG_LEVEL=DEBUG
export MCP_SQL_ECHO=true
export MCP_TRACE_REQUESTS=true

# Run with detailed output
python -m tldw_Server_API.app.main --log-level debug
```

### Performance Profiling
```python
# Enable profiling
export MCP_PROFILING=true
export MCP_PROFILE_DIR=/tmp/mcp_profiles

# Analyze with snakeviz
pip install snakeviz
snakeviz /tmp/mcp_profiles/profile_*.prof
```

## Maintenance

### Daily Tasks
- Monitor error logs for anomalies
- Check disk space usage
- Verify backup completion
- Review authentication failures

### Weekly Tasks
- Analyze performance metrics
- Update rate limiting rules if needed
- Review and rotate logs
- Test backup restoration (staging)

### Monthly Tasks
- Security updates and patches
- Certificate renewal check
- Database maintenance (VACUUM, ANALYZE)
- Review and update firewall rules
- Capacity planning review

### Update Procedure
```bash
#!/bin/bash
# update-mcp.sh

# Backup current version
tar czf /var/backups/tldw/mcp_$(date +%Y%m%d).tar.gz /opt/tldw_server

# Pull latest code
cd /opt/tldw_server
git pull origin main

# Update dependencies
source venv/bin/activate
pip install -e . --upgrade

# Run migrations
python -m tldw_Server_API.app.core.MCP_unified.migrations

# Run tests
python -m pytest tldw_Server_API/app/core/MCP_unified/tests/

# Reload service (zero-downtime)
systemctl reload tldw-mcp
```

## Security Checklist

### Initial Deployment
- [ ] Changed all default passwords and secrets
- [ ] Generated secure JWT secret (min 256 bits)
- [ ] Configured SSL/TLS certificates
- [ ] Set up firewall rules
- [ ] Enabled rate limiting
- [ ] Configured CORS properly
- [ ] Set up audit logging
- [ ] Restricted metrics endpoint access
- [ ] Created separate database user with limited privileges
- [ ] Enabled database SSL connections

### Ongoing Security
- [ ] Regular security updates (weekly)
- [ ] Monitor authentication failures (daily)
- [ ] Review audit logs (weekly)
- [ ] Rotate secrets (quarterly)
- [ ] Security scan dependencies (monthly)
- [ ] Penetration testing (annually)
- [ ] Review user permissions (monthly)
- [ ] Update TLS certificates before expiry
- [ ] Backup encryption keys secured
- [ ] Incident response plan documented

### Compliance
- [ ] Data retention policies implemented
- [ ] GDPR compliance (if applicable)
- [ ] Audit trail for all admin actions
- [ ] Data encryption at rest and in transit
- [ ] Regular compliance audits
- [ ] User data export/deletion procedures
- [ ] Privacy policy alignment
- [ ] Security documentation maintained

## Support Resources

### Logs Location
- Application logs: `/var/log/tldw/mcp/`
- Audit logs: `/var/log/tldw/mcp/audit.log`
- Access logs: `/var/log/tldw/mcp/access.log`
- Error logs: `/var/log/tldw/mcp/error.log`

### Configuration Files
- Main config: `/etc/tldw/config/mcp.env`
- Nginx: `/etc/nginx/sites-available/tldw-mcp`
- Systemd: `/etc/systemd/system/tldw-mcp.service`
- Prometheus: `/etc/prometheus/prometheus.yml`

### Monitoring URLs
- Health check: `https://mcp.example.com/api/v1/mcp/health`
- Metrics: `http://localhost:9090/metrics` (internal only)
- Status: `https://mcp.example.com/api/v1/mcp/status`

### Emergency Procedures
1. **Service Down**: Check systemd status, review error logs
2. **Security Breach**: Rotate all secrets, review audit logs, notify stakeholders
3. **Data Loss**: Initiate recovery procedure, verify backup integrity
4. **Performance Crisis**: Scale horizontally, increase resources, enable caching
