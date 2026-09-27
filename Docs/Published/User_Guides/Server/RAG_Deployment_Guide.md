# RAG Module Deployment Guide

## Overview

This guide provides step-by-step instructions for deploying the tldw RAG module to production. The module is production-ready and can be deployed in 2-3 days with proper configuration.

## Prerequisites

### System Requirements

- **OS**: Ubuntu 20.04+ / CentOS 8+ / macOS 12+
- **Python**: 3.10 or higher
- **RAM**: Minimum 8GB (16GB recommended)
- **Storage**: 50GB+ for databases and media
- **CPU**: 4+ cores recommended
- **GPU**: Optional (for faster embeddings)

### Software Dependencies

```bash
# Required
python3.11+
pip
git
sqlite3
nginx or apache (for reverse proxy)

# Optional
postgresql (for high-load environments)
redis (for distributed caching)
docker (for containerized deployment)
```

## Deployment Options

### Option 1: Direct Installation (Recommended for Single Server)

Best for: Small to medium deployments, single server setups

### Option 2: Docker Deployment

Best for: Containerized environments, microservices architecture

### Option 3: Kubernetes Deployment

Best for: Large scale, high availability requirements

## Step-by-Step Deployment

### Step 1: Server Preparation

#### 1.1 Create deployment user

```bash
# Create dedicated user
sudo useradd -m -s /bin/bash tldw
sudo usermod -aG sudo tldw

# Set up directories
sudo mkdir -p /opt/tldw
sudo mkdir -p /var/lib/tldw/databases
sudo mkdir -p /var/log/tldw
sudo mkdir -p /etc/tldw

# Set permissions
sudo chown -R tldw:tldw /opt/tldw
sudo chown -R tldw:tldw /var/lib/tldw
sudo chown -R tldw:tldw /var/log/tldw
sudo chown -R tldw:tldw /etc/tldw
```

#### 1.2 Install system dependencies

```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install -y \
    python3-pip python3-venv \
    build-essential libssl-dev libffi-dev \
    nginx sqlite3 git curl \
    ffmpeg  # For media processing

# CentOS/RHEL
sudo yum install -y \
    python3-pip python3-devel \
    gcc openssl-devel \
    nginx sqlite git curl \
    ffmpeg
```

### Step 2: Application Installation

#### 2.1 Clone repository

```bash
sudo su - tldw
cd /opt/tldw
git clone https://github.com/tldw/tldw_server.git .
git checkout stable  # Use stable branch for production
```

#### 2.2 Create virtual environment

```bash
python3 -m venv venv
source venv/bin/activate
```

#### 2.3 Install Python dependencies

```bash
pip install --upgrade pip
pip install -e .

# Install production server
pip install gunicorn uvicorn[standard]
```

#### 2.4 Install optional ML dependencies

```bash
# For GPU support (CUDA 11.8+)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For better embeddings
pip install sentence-transformers
```

### Step 3: Configuration

#### 3.1 Copy configuration template

```bash
cp tldw_Server_API/Config_Files/config.production.template.txt /etc/tldw/config.txt
```

#### 3.2 Edit production configuration

```bash
vim /etc/tldw/config.txt
```

Key settings to update:

```ini
[Settings]
# CHANGE THIS - Generate with: openssl rand -hex 32
api_key = your-secure-random-key-here

[Database]
database_path = /var/lib/tldw/user_databases/
media_db_name = <user_id>/<content-db>.db

[API]
# Add your API keys for LLM providers
openai_api_key = sk-...
anthropic_api_key = sk-ant-...

[Logging]
log_path = /var/log/tldw/
log_level = INFO

[RAG]
# Performance settings
batch_size = 32
num_workers = 4
use_gpu = false  # Set to true if GPU available
enable_cache = true
cache_ttl = 3600
```

Replace `<content-db>.db` with your configured per-user content DB filename.

#### 3.3 Set environment variables

Create `/etc/tldw/environment`:

```bash
# Security
API_KEY=your-secure-api-key
JWT_SECRET=your-jwt-secret
SESSION_ENCRYPTION_KEY=your-32-byte-key

# Database (per-user root)
DATABASE_PATH=/var/lib/tldw/user_databases

# Logging
LOG_PATH=/var/log/tldw
AUDIT_LOG_PATH=/var/log/tldw/audit

# Performance
USE_GPU=false
NUM_WORKERS=4
```

### Step 4: Database Setup

#### 4.1 Initialize databases

```bash
cd /opt/tldw
source venv/bin/activate

# Run database migrations
python -c "
from tldw_Server_API.app.core.DB_Management import migrations
migrations.apply_all_migrations('/var/lib/tldw/databases/')
"

# Verify databases created
ls -la /var/lib/tldw/databases/
```

#### 4.2 Set database permissions

```bash
# Secure database files
chmod 640 /var/lib/tldw/databases/*.db
```

### Step 5: Service Configuration

#### 5.1 Create systemd service

Create `/etc/systemd/system/tldw.service`:

```ini
[Unit]
Description=tldw RAG Service
After=network.target

[Service]
Type=notify
User=tldw
Group=tldw
WorkingDirectory=/opt/tldw
Environment="PATH=/opt/tldw/venv/bin"
EnvironmentFile=/etc/tldw/environment
ExecStart=/opt/tldw/venv/bin/gunicorn \
    -w 4 \
    -k uvicorn.workers.UvicornWorker \
    --bind 127.0.0.1:8000 \
    --timeout 120 \
    --graceful-timeout 30 \
    --access-logfile /var/log/tldw/access.log \
    --error-logfile /var/log/tldw/error.log \
    tldw_Server_API.app.main:app
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

#### 5.2 Enable and start service

```bash
sudo systemctl daemon-reload
sudo systemctl enable tldw
sudo systemctl start tldw
sudo systemctl status tldw
```

### Step 6: Reverse Proxy Setup

#### 6.1 Configure Nginx

Create `/etc/nginx/sites-available/tldw`:

```nginx
upstream tldw_backend {
    server 127.0.0.1:8000;
    keepalive 32;
}

server {
    listen 80;
    server_name api.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name api.yourdomain.com;

    # SSL Configuration
    ssl_certificate /etc/ssl/certs/tldw.crt;
    ssl_certificate_key /etc/ssl/private/tldw.key;
    ssl_protocols TLSv1.2 TLSv1.3;
    ssl_ciphers HIGH:!aNULL:!MD5;
    ssl_prefer_server_ciphers on;

    # Security headers
    add_header X-Frame-Options "SAMEORIGIN" always;
    add_header X-Content-Type-Options "nosniff" always;
    add_header X-XSS-Protection "1; mode=block" always;

    # Proxy settings
    client_max_body_size 100M;
    proxy_read_timeout 300s;
    proxy_connect_timeout 75s;

    location / {
        proxy_pass http://tldw_backend;
        proxy_http_version 1.1;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        # WebSocket support for streaming
        proxy_buffering off;
    }

    # Health check endpoint
    location /health {
        proxy_pass http://tldw_backend/api/v1/rag/health;
        access_log off;
    }
}
```

#### 6.2 Enable site and restart Nginx

```bash
sudo ln -s /etc/nginx/sites-available/tldw /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### Step 7: SSL Certificate Setup

#### 7.1 Using Let's Encrypt (Free)

```bash
sudo apt-get install certbot python3-certbot-nginx
sudo certbot --nginx -d api.yourdomain.com
```

#### 7.2 Using commercial certificate

```bash
# Copy certificate files
sudo cp your-cert.crt /etc/ssl/certs/tldw.crt
sudo cp your-cert.key /etc/ssl/private/tldw.key
sudo chmod 600 /etc/ssl/private/tldw.key
```

### Step 8: Firewall Configuration

```bash
# Allow HTTPS
sudo ufw allow 443/tcp

# Allow SSH (if needed)
sudo ufw allow 22/tcp

# Enable firewall
sudo ufw enable
```

### Step 9: Monitoring Setup

#### 9.1 Install monitoring tools

```bash
# Prometheus node exporter
wget https://github.com/prometheus/node_exporter/releases/download/v1.5.0/node_exporter-1.5.0.linux-amd64.tar.gz
tar xvf node_exporter-1.5.0.linux-amd64.tar.gz
sudo mv node_exporter-1.5.0.linux-amd64/node_exporter /usr/local/bin/
```

#### 9.2 Configure log rotation

Create `/etc/logrotate.d/tldw`:

```
/var/log/tldw/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0640 tldw tldw
    sharedscripts
    postrotate
        systemctl reload tldw
    endscript
}
```

### Step 10: Performance Testing

#### 10.1 Basic health check

```bash
curl https://api.yourdomain.com/api/v1/rag/health
```

#### 10.2 Load testing

```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Test search endpoint
ab -n 100 -c 10 -H "Authorization: Bearer your-api-key" \
   https://api.yourdomain.com/api/v1/rag/search?query=test
```

#### 10.3 Verify all endpoints

```python
import requests

base_url = "https://api.yourdomain.com"
headers = {"Authorization": "Bearer your-api-key"}

# Test endpoints
endpoints = [
    "/api/v1/rag/health",
    "/api/v1/rag/search",
    "/api/v1/rag/agent",
]

for endpoint in endpoints:
    response = requests.get(f"{base_url}{endpoint}", headers=headers)
    print(f"{endpoint}: {response.status_code}")
```

## Docker Deployment (Alternative)

### Build Docker image

Create `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc g++ \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install from pyproject (editable)
COPY pyproject.toml .
RUN pip install --no-cache-dir -e . && \
    pip install --no-cache-dir gunicorn uvicorn[standard]

# Copy application
COPY tldw_Server_API/ ./tldw_Server_API/
COPY Docs/ ./Docs/

# Create directories
RUN mkdir -p /data/databases /data/logs

# Environment
ENV DATABASE_PATH=/data/databases
ENV LOG_PATH=/data/logs

EXPOSE 8000

CMD ["gunicorn", "-w", "4", "-k", "uvicorn.workers.UvicornWorker", \
     "--bind", "0.0.0.0:8000", "tldw_Server_API.app.main:app"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  tldw:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/data
      - ./config:/etc/tldw
    environment:
      - API_KEY=${API_KEY}
      - JWT_SECRET=${JWT_SECRET}
      - DATABASE_PATH=/data/databases
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
      - "443:443"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
      - ./ssl:/etc/ssl
    depends_on:
      - tldw
    restart: unless-stopped
```

### Run with Docker Compose

```bash
docker-compose up -d
docker-compose logs -f
```

## Post-Deployment Checklist

### Security Verification

- [ ] API key changed from default
- [ ] HTTPS enabled and working
- [ ] Firewall configured
- [ ] Database permissions secured
- [ ] Audit logging enabled
- [ ] Rate limiting active

### Functionality Testing

- [ ] Search endpoint working
- [ ] Agent endpoint responding
- [ ] Authentication working
- [ ] Rate limiting enforced
- [ ] Audit logs being written
- [ ] Metrics being collected

### Performance Validation

- [ ] Response time < 2 seconds
- [ ] Can handle 100+ concurrent users
- [ ] Memory usage stable
- [ ] CPU usage reasonable
- [ ] Database connections pooled

### Backup Verification

- [ ] Database backups scheduled
- [ ] Configuration backed up
- [ ] Backup restoration tested

## Troubleshooting

### Service won't start

```bash
# Check logs
sudo journalctl -u tldw -n 50

# Check permissions
ls -la /var/lib/tldw/user_databases/
ls -la /var/log/tldw/

# Test configuration
cd /opt/tldw
source venv/bin/activate
python -m tldw_Server_API.app.main
```

### Database errors

```bash
# Check database integrity
sqlite3 /var/lib/tldw/user_databases/<user_id>/<content-db>.db "PRAGMA integrity_check;"

# Reset database (WARNING: Data loss)
rm /var/lib/tldw/databases/*.db
python -c "from tldw_Server_API.app.core.DB_Management import migrations; migrations.apply_all_migrations('/var/lib/tldw/databases/')"
```

### High memory usage

```bash
# Restart service
sudo systemctl restart tldw

# Reduce workers
# Edit /etc/systemd/system/tldw.service
# Change -w 4 to -w 2
sudo systemctl daemon-reload
sudo systemctl restart tldw
```

### SSL issues

```bash
# Test SSL
openssl s_client -connect api.yourdomain.com:443

# Renew Let's Encrypt
sudo certbot renew
```

## Maintenance

### Regular tasks

```bash
# Daily: Check logs for errors
grep ERROR /var/log/tldw/app.log | tail -20

# Weekly: Check disk usage
df -h /var/lib/tldw

# Monthly: Update dependencies
cd /opt/tldw
source venv/bin/activate
pip list --outdated

# Quarterly: Security updates
sudo apt-get update && sudo apt-get upgrade
```

### Backup script

Create `/opt/tldw/backup.sh`:

```bash
#!/bin/bash
BACKUP_DIR="/backup/tldw"
DATE=$(date +%Y%m%d_%H%M%S)

# Create backup
mkdir -p $BACKUP_DIR
sqlite3 /var/lib/tldw/user_databases/<user_id>/<content-db>.db ".backup $BACKUP_DIR/media_$DATE.db"
tar -czf $BACKUP_DIR/config_$DATE.tar.gz /etc/tldw/

# Clean old backups
find $BACKUP_DIR -name "*.db" -mtime +30 -delete
find $BACKUP_DIR -name "*.tar.gz" -mtime +30 -delete
```

Add to crontab:

```bash
0 2 * * * /opt/tldw/backup.sh
```

## Scaling Considerations

### Horizontal Scaling

For high load, deploy multiple instances:

1. Set up load balancer (HAProxy/Nginx)
2. Use shared database (PostgreSQL)
3. Use Redis for distributed caching
4. Share media storage (NFS/S3)

### Database Migration to PostgreSQL

```python
# Migrate from SQLite to PostgreSQL
python scripts/migrate_to_postgres.py \
    --source /var/lib/tldw/user_databases/<user_id>/<content-db>.db \
    --dest postgresql://user:pass@localhost/tldw
```

Replace `<content-db>.db` with your configured per-user content DB filename in each command above.

## Support

- Documentation: https://docs.tldw.ai
- GitHub Issues: https://github.com/tldw/tldw_server/issues
- Community Forum: https://forum.tldw.ai
- Enterprise Support: enterprise@tldw.ai

---

*Last Updated: 2025-08-19*
*Version: 1.0.0*
