# RAG Module Production Configuration Guide

## Overview

This guide covers the configuration and security setup required to deploy the RAG (Retrieval-Augmented Generation) module in a production environment. The RAG module is fully production-ready with enterprise features including rate limiting, audit logging, and comprehensive metrics collection.

## Table of Contents

1. [Security Configuration](#security-configuration)
2. [Database Configuration](#database-configuration)
3. [Performance Tuning](#performance-tuning)
4. [Rate Limiting](#rate-limiting)
5. [Audit Logging](#audit-logging)
6. [Monitoring & Metrics](#monitoring-metrics)
7. [Authentication Setup](#authentication-setup)
8. [SSL/TLS Configuration](#ssltls-configuration)
9. [Backup Strategy](#backup-strategy)
10. [Troubleshooting](#troubleshooting)

## Security Configuration

### 1. API Key Configuration

**CRITICAL**: The default API key must be replaced before production deployment.

```bash
# Set environment variable (recommended)
export API_KEY="your-secure-random-key-here"

# Or update config.txt
[Settings]
api_key = your-secure-random-key-here
```

Generate a secure API key:
```bash
openssl rand -hex 32
```

### 2. JWT Configuration

For multi-user deployments, configure JWT settings:

```bash
# Environment variables
export JWT_SECRET="your-jwt-secret-key"
export JWT_ALGORITHM="HS256"
export JWT_EXPIRATION_HOURS=24
```

### 3. Session Encryption

Configure session encryption for secure token storage:

```bash
export SESSION_ENCRYPTION_KEY="your-32-byte-key-here"
```

## Database Configuration

### 1. SQLite Configuration (Default)

The RAG module uses SQLite by default with optimized settings:

```ini
# config.txt
[Database]
database_path = /path/to/secure/location/Databases/
media_db_name = user_databases/<user_id>/<content-db>.db
rag_audit_db = rag_audit.db

# Connection pool settings
[RAG]
connection_pool_min = 2
connection_pool_max = 10
connection_timeout = 5.0
```

Replace `<content-db>.db` with your configured per-user content DB filename.

### 2. PostgreSQL Configuration (Optional)

For high-load production environments, PostgreSQL is recommended:

```ini
# config.txt
[Database]
database_type = postgresql
database_url = postgresql://user:password@localhost/tldw_production
pool_size = 20
max_overflow = 10
```

### 3. Database Paths

Ensure database directories have proper permissions:

```bash
# Create secure database directories (per-user)
sudo mkdir -p /var/lib/tldw/user_databases/<user_id>
sudo chown -R tldw:tldw /var/lib/tldw
sudo chmod 750 /var/lib/tldw/user_databases
```

## Performance Tuning

### 1. RAG Service Configuration

Optimize RAG service settings in `config.txt`:

```ini
[RAG]
# Retrieval settings
fts_top_k = 20
vector_top_k = 15
hybrid_alpha = 0.5

# Processing
batch_size = 32
num_workers = 4
use_gpu = true

# Caching
enable_cache = true
cache_ttl = 3600
max_cache_size = 1000

# Chunking
chunk_size = 512
chunk_overlap = 128
enable_smart_chunking = true
```

### 2. Connection Pooling

Configure connection pools for optimal performance:

```ini
[ConnectionPool]
# Per-database pool sizes
media_db_pool_size = 5
notes_db_pool_size = 3
chat_db_pool_size = 3

# Connection lifecycle
max_connection_age = 3600
idle_timeout = 300
connection_timeout = 5.0
```

### 3. Worker Configuration

For production deployments, use multiple workers:

```bash
# Using Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 120 \
  --graceful-timeout 30 \
  tldw_Server_API.app.main:app
```

## Rate Limiting

### 1. User Tier Configuration

The RAG module includes built-in rate limiting with user tiers:

```python
# User tier limits (requests per hour)
FREE_TIER = 100
BASIC_TIER = 500
PREMIUM_TIER = 2000
ENTERPRISE_TIER = 10000
```

### 2. Endpoint-Specific Limits

Configure per-endpoint rate limits:

```ini
[RateLimits]
# Searches per minute
search_rate_limit = 30

# Agent requests per minute
agent_rate_limit = 10

# Embeddings per hour
embeddings_rate_limit = 100
```

### 3. Custom Rate Limiting

Implement custom rate limiting rules:

```python
# In your configuration
RATE_LIMIT_CONFIG = {
    "/api/v1/rag/search": "30/minute",
    "/api/v1/rag/agent": "10/minute",
    "/api/v1/rag/embed": "100/hour"
}
```

## Audit Logging

### 1. Enable Audit Logging

Audit logging is enabled by default. Configure the path:

```ini
[Logging]
audit_log_path = /var/log/tldw/audit/
audit_log_enabled = true
audit_log_level = INFO
```

### 2. Audit Log Rotation

Configure log rotation to manage disk space:

```bash
# /etc/logrotate.d/tldw-audit
/var/log/tldw/audit/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 tldw tldw
}
```

### 3. Security Event Monitoring

Monitor security events in real-time:

```sql
-- Query suspicious activity
SELECT * FROM rag_security_events
WHERE severity IN ('HIGH', 'CRITICAL')
AND resolved = FALSE
ORDER BY timestamp DESC;
```

## Monitoring & Metrics

### 1. Enable Metrics Collection

Metrics are collected automatically. Configure storage:

```ini
[Metrics]
enable_metrics = true
metrics_db_path = /var/lib/tldw/metrics/
metrics_retention_days = 90
```

### 2. Prometheus Integration (Optional)

Export metrics to Prometheus:

```python
# Add to main.py
from prometheus_client import make_asgi_app

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)
```

### 3. Key Metrics to Monitor

- **Response Time**: p50, p95, p99 latencies
- **Error Rate**: 4xx and 5xx responses
- **Token Usage**: Total tokens consumed
- **Cache Hit Rate**: Percentage of cached responses
- **Database Pool Usage**: Active connections

## Authentication Setup

### 1. Single-User Mode

For simple deployments:

```ini
[Auth]
auth_mode = single_user
api_key = your-secure-api-key
```

### 2. Multi-User Mode

For enterprise deployments:

```ini
[Auth]
auth_mode = multi_user
jwt_enabled = true
jwt_secret = your-jwt-secret
session_timeout = 3600
```

### 3. OAuth2 Integration (Optional)

Configure OAuth2 providers:

```ini
[OAuth2]
enabled = true
providers = google,github
google_client_id = your-client-id
google_client_secret = your-client-secret
```

## SSL/TLS Configuration

### 1. Enable HTTPS

Always use HTTPS in production:

```nginx
# nginx.conf
server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    ssl_certificate /etc/ssl/certs/tldw.crt;
    ssl_certificate_key /etc/ssl/private/tldw.key;

    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;

    location / {
        proxy_pass http://localhost:8000;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

### 2. CORS Configuration

Configure CORS for your domains:

```python
# In main.py
ALLOWED_ORIGINS = [
    "https://yourdomain.com",
    "https://app.yourdomain.com"
]
```

## Backup Strategy

### 1. Database Backups

Implement automated backups:

```bash
#!/bin/bash
# backup.sh
BACKUP_DIR="/backup/tldw"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup SQLite databases
sqlite3 /var/lib/tldw/user_databases/<user_id>/<content-db>.db ".backup $BACKUP_DIR/media_$DATE.db"
sqlite3 /var/lib/tldw/databases/rag_audit.db ".backup $BACKUP_DIR/audit_$DATE.db"

# Compress and encrypt
tar -czf - $BACKUP_DIR/*_$DATE.db | openssl enc -aes-256-cbc -salt -out $BACKUP_DIR/backup_$DATE.tar.gz.enc

# Clean up old backups (keep 30 days)
find $BACKUP_DIR -name "*.enc" -mtime +30 -delete
```

Replace `<content-db>.db` with your configured per-user content DB filename.

### 2. Configuration Backup

Version control your configuration:

```bash
git init /etc/tldw
git add config.txt
git commit -m "Production configuration"
git remote add origin git@github.com:yourorg/tldw-config.git
```

## Troubleshooting

### Common Issues and Solutions

#### 1. High Memory Usage

**Solution**: Adjust cache and pool sizes
```ini
[Cache]
max_cache_size = 500  # Reduce from 1000
[ConnectionPool]
media_db_pool_size = 3  # Reduce from 5
```

#### 2. Slow Response Times

**Solution**: Enable caching and optimize queries
```ini
[RAG]
enable_cache = true
cache_ttl = 7200  # Increase cache duration
fts_top_k = 10  # Reduce search results
```

#### 3. Database Lock Errors

**Solution**: Enable WAL mode for SQLite
```sql
PRAGMA journal_mode=WAL;
PRAGMA busy_timeout=5000;
```

#### 4. Rate Limiting Issues

**Solution**: Adjust tier limits or upgrade users
```python
# Temporarily increase limits
user_rate_limiter.update_user_tier(user_id, UserTier.PREMIUM)
```

### Health Checks

Monitor service health:

```bash
# Check API health
curl https://api.yourdomain.com/api/v1/rag/health

# Check database connections
curl https://api.yourdomain.com/api/v1/rag/health/db

# Check metrics
curl https://api.yourdomain.com/metrics
```

### Log Analysis

Analyze logs for issues:

```bash
# Check error logs
grep ERROR /var/log/tldw/app.log | tail -50

# Monitor audit logs
tail -f /var/log/tldw/audit/rag_audit.log

# Check slow queries
grep "duration>" /var/log/tldw/app.log | awk '{print $NF}' | sort -n
```

## Environment Variables Reference

Complete list of environment variables for production:

```bash
# Core Configuration
API_KEY=your-secure-api-key
DATABASE_PATH=/var/lib/tldw/databases

# Authentication
JWT_SECRET=your-jwt-secret
JWT_ALGORITHM=HS256
JWT_EXPIRATION_HOURS=24
SESSION_ENCRYPTION_KEY=your-session-key

# Rate Limiting
RG_ENABLED=true
# Tune requests.rpm / requests.burst in Config_Files/resource_governor_policies.yaml

# Audit Logging
AUDIT_LOG_PATH=/var/log/tldw/audit
AUDIT_LOG_ENABLED=true

# Performance
USE_GPU=true
NUM_WORKERS=4
CACHE_ENABLED=true
CACHE_TTL=3600

# Monitoring
METRICS_ENABLED=true
METRICS_PORT=9090
```

## Security Checklist

Before deploying to production, ensure:

- [ ] Default API key replaced
- [ ] JWT secret configured
- [ ] HTTPS enabled
- [ ] Database permissions secured
- [ ] Audit logging enabled
- [ ] Rate limiting configured
- [ ] Backups automated
- [ ] Monitoring enabled
- [ ] Log rotation configured
- [ ] Firewall rules configured
- [ ] CORS properly configured
- [ ] Session encryption enabled

## Support

For production support and enterprise features, contact:
- GitHub Issues: https://github.com/tldw/tldw_server/issues
- Documentation: https://docs.tldw.ai
- Enterprise Support: enterprise@tldw.ai

---

*Last Updated: 2025-08-19*
*Version: 1.0.0*
