# AuthNZ Database Configuration Guide

## Overview

The tldw_server AuthNZ system now supports both **SQLite** and **PostgreSQL** backends through a unified interface. This allows seamless switching between development (SQLite) and production (PostgreSQL) environments without code changes.

## Quick Start

### SQLite (Default/Development)
```bash
# No configuration needed - SQLite is the default
export TLDW_USER_DB_BACKEND=sqlite
export DATABASE_URL=sqlite:///../Databases/Users.db
```

### PostgreSQL (Production)
```bash
export TLDW_USER_DB_BACKEND=postgresql
export DATABASE_URL=postgresql://user:password@localhost:5432/tldw_users
```

## Configuration Methods

### 1. Environment Variables (Recommended)

The system automatically detects the backend from environment variables:

```bash
# Choose backend
export TLDW_USER_DB_BACKEND=postgresql  # or "sqlite"

# For PostgreSQL, provide connection details
export DATABASE_URL=postgresql://user:password@host:5432/dbname

# Optional: Connection pool settings
export TLDW_DB_POOL_SIZE=20
export TLDW_DB_POOL_TIMEOUT=30
```

### 2. Configuration File

Update `tldw_Server_API/Config_Files/config.txt`:

```ini
# For SQLite
DATABASE_URL = sqlite:///../Databases/Users.db

# For PostgreSQL
DATABASE_URL = postgresql://tldw_user:secure_password@localhost:5432/tldw_users
```

### 3. Programmatic Configuration

```python
from tldw_Server_API.app.core.AuthNZ.db_config import get_configured_user_database

# Automatically uses environment/config settings
user_db = get_configured_user_database()

# Or explicitly configure
from tldw_Server_API.app.core.DB_Management.backends.base import DatabaseConfig, BackendType
from tldw_Server_API.app.core.DB_Management.UserDatabase_v2 import UserDatabase

# PostgreSQL
config = DatabaseConfig(
    backend_type=BackendType.POSTGRESQL,
    connection_string="postgresql://user:pass@localhost/tldw",
    pool_size=20
)
user_db = UserDatabase(config=config)
```

## Environment Variables Reference

### Common Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `TLDW_USER_DB_BACKEND` | Backend type: "sqlite" or "postgresql" | sqlite |
| `DATABASE_URL` | Full database connection URL | sqlite:///../Databases/Users.db |
| `TLDW_DB_ECHO` | Enable SQL query logging | false |

### SQLite-Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `TLDW_SQLITE_WAL_MODE` | Enable Write-Ahead Logging | true |
| `TLDW_SQLITE_FOREIGN_KEYS` | Enable foreign key constraints | true |

### PostgreSQL-Specific

| Variable | Description | Default |
|----------|-------------|---------|
| `TLDW_PG_HOST` | Database host | localhost |
| `TLDW_PG_PORT` | Database port | 5432 |
| `TLDW_PG_DATABASE` | Database name | tldw_users |
| `TLDW_PG_USER` | Database username | - |
| `TLDW_PG_PASSWORD` | Database password | - |
| `TLDW_PG_SSLMODE` | SSL mode (prefer/require/disable) | prefer |
| `TLDW_DB_POOL_SIZE` | Connection pool size | 10 |
| `TLDW_DB_MAX_OVERFLOW` | Max overflow connections | 20 |
| `TLDW_DB_POOL_TIMEOUT` | Pool timeout (seconds) | 30 |
| `TLDW_DB_POOL_RECYCLE` | Connection recycle time (seconds) | 3600 |

## Database Setup

### SQLite Setup

SQLite requires no additional setup. The database file is created automatically:

```bash
# Database will be created at ../Databases/Users.db
python -m tldw_Server_API.app.core.AuthNZ.migrate_to_multiuser
```

### PostgreSQL Setup

1. **Install PostgreSQL**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install postgresql postgresql-contrib

   # macOS
   brew install postgresql
   ```

2. **Create Database and User**
   ```sql
   -- Connect as postgres superuser
   sudo -u postgres psql

   -- Create database
   CREATE DATABASE tldw_users;

   -- Create user
   CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'secure_password';

   -- Grant privileges
   GRANT ALL PRIVILEGES ON DATABASE tldw_users TO tldw_user;

   -- Exit
   \q
   ```

3. **Initialize Schema**
   ```bash
   # Set PostgreSQL environment
   export TLDW_USER_DB_BACKEND=postgresql
   export DATABASE_URL=postgresql://tldw_user:secure_password@localhost:5432/tldw_users

   # Run migration
   python -m tldw_Server_API.app.core.AuthNZ.migrate_to_multiuser
   ```

## Migration Between Backends

### SQLite to PostgreSQL Migration

1. **Export from SQLite**
   ```bash
   # Backup SQLite database
   sqlite3 ../Databases/Users.db .dump > users_backup.sql
   ```

2. **Convert and Import to PostgreSQL**
   ```bash
   # Use migration tool (recommended)
   python -m tldw_Server_API.app.core.AuthNZ.migrate_db \
     --from sqlite:///../Databases/Users.db \
     --to postgresql://user:pass@localhost/tldw_users
   ```

### PostgreSQL to SQLite Migration

```bash
# Export from PostgreSQL
pg_dump -U tldw_user -d tldw_users > users_backup.sql

# Convert and import to SQLite
python -m tldw_Server_API.app.core.AuthNZ.migrate_db \
  --from postgresql://user:pass@localhost/tldw_users \
  --to sqlite:///../Databases/Users_new.db
```

## Docker Deployment

### Docker Compose with PostgreSQL

```yaml
version: '3.8'

services:
  postgres:
    image: postgres:18
    environment:
      POSTGRES_DB: tldw_users
      POSTGRES_USER: tldw_user
      POSTGRES_PASSWORD: ${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"

  tldw_server:
    build: .
    environment:
      TLDW_USER_DB_BACKEND: postgresql
      DATABASE_URL: postgresql://tldw_user:${DB_PASSWORD}@postgres:5432/tldw_users
      TLDW_DB_POOL_SIZE: 20
    depends_on:
      - postgres
    ports:
      - "8000:8000"

volumes:
  postgres_data:
```

### Docker with SQLite

```yaml
version: '3.8'

services:
  tldw_server:
    build: .
    environment:
      TLDW_USER_DB_BACKEND: sqlite
      DATABASE_URL: sqlite:///app/data/Users.db
    volumes:
      - ./data:/app/data
    ports:
      - "8000:8000"
```

## Performance Considerations

### SQLite
- **Best for**: Development, single-user, small teams (<10 users)
- **Pros**: Zero configuration, portable, low resource usage
- **Cons**: Limited concurrency, no network access
- **Max concurrent writes**: 1
- **Recommended settings**:
  ```bash
  export TLDW_SQLITE_WAL_MODE=true  # Better concurrency
  export TLDW_SQLITE_FOREIGN_KEYS=true  # Data integrity
  ```

### PostgreSQL
- **Best for**: Production, multi-user, large teams (10+ users)
- **Pros**: High concurrency, network access, advanced features
- **Cons**: Requires setup, more resources
- **Max concurrent connections**: Configurable (default 100)
- **Recommended settings**:
  ```bash
  export TLDW_DB_POOL_SIZE=20  # Adjust based on load
  export TLDW_DB_POOL_RECYCLE=3600  # Refresh connections hourly
  export TLDW_PG_SSLMODE=require  # For production
  ```

## Troubleshooting

### Common Issues

#### 1. "Database backend not detected"
```bash
# Explicitly set backend
export TLDW_USER_DB_BACKEND=postgresql
# or
export TLDW_USER_DB_BACKEND=sqlite
```

#### 2. "Connection refused" (PostgreSQL)
```bash
# Check PostgreSQL is running
sudo systemctl status postgresql

# Check connection details
psql -U tldw_user -d tldw_users -h localhost
```

#### 3. "Database is locked" (SQLite)
```bash
# Enable WAL mode for better concurrency
export TLDW_SQLITE_WAL_MODE=true
```

#### 4. "Too many connections" (PostgreSQL)
```bash
# Increase pool size
export TLDW_DB_POOL_SIZE=50

# Or increase PostgreSQL max_connections
sudo -u postgres psql -c "ALTER SYSTEM SET max_connections = 200;"
sudo systemctl restart postgresql
```

### Debug Mode

Enable SQL query logging for debugging:

```bash
export TLDW_DB_ECHO=true
```

### Check Configuration

```python
from tldw_Server_API.app.core.AuthNZ.db_config import AuthDatabaseConfig

# Print current configuration
AuthDatabaseConfig.print_config()
```

Output:
```
============================================================
AuthNZ Database Configuration
============================================================
  backend_type.................. postgresql
  auth_mode.................... multi_user
  registration_enabled......... True
  require_registration_code.... True
  database_host................ localhost
  database_name................ tldw_users
============================================================
```

## Best Practices

### Development
1. Use SQLite for local development
2. Keep database in version control (schema only)
3. Use migrations for schema changes
4. Regular backups

### Production
1. Use PostgreSQL for production
2. Enable SSL for database connections
3. Use connection pooling
4. Monitor connection usage
5. Regular automated backups
6. Set up replication for high availability

### Security
1. Never commit credentials to version control
2. Use environment variables for sensitive data
3. Enable SSL for PostgreSQL connections
4. Regularly rotate database passwords
5. Limit database user permissions
6. Enable audit logging

## API Compatibility

Both backends support the exact same API:

```python
# This code works with both SQLite and PostgreSQL
from tldw_Server_API.app.core.AuthNZ.db_config import get_configured_user_database

user_db = get_configured_user_database()

# All operations are identical
user_id = user_db.create_user("john", "john@example.com", password_hash)
roles = user_db.get_user_roles(user_id)
perms = user_db.get_user_permissions(user_id)
```

## Monitoring

### SQLite Monitoring
```bash
# Check database size
du -h ../Databases/Users.db

# Check integrity
sqlite3 ../Databases/Users.db "PRAGMA integrity_check;"
```

### PostgreSQL Monitoring
```sql
-- Connection count
SELECT count(*) FROM pg_stat_activity;

-- Database size
SELECT pg_database_size('tldw_users');

-- Active queries
SELECT query, state FROM pg_stat_activity WHERE state != 'idle';
```

## Backup and Recovery

### SQLite Backup
```bash
# Simple file copy
cp ../Databases/Users.db ../Databases/Users.db.backup

# Or use SQLite backup command
sqlite3 ../Databases/Users.db ".backup ../Databases/Users.db.backup"
```

### PostgreSQL Backup
```bash
# Backup
pg_dump -U tldw_user -d tldw_users > tldw_users_backup.sql

# Restore
psql -U tldw_user -d tldw_users < tldw_users_backup.sql
```

## Content Database Modes (Development)

The media/content store can run on either SQLite (default) or a shared PostgreSQL
instance. Switching modes is controlled via environment variables and a light-weight
validation CLI.

### Enable PostgreSQL for Content Storage

1. Configure the Postgres connection (reusing or separate from AuthNZ):

   ```bash
   export CONTENT_DB_MODE=postgres            # same as TLDW_CONTENT_DB_BACKEND=postgresql
   export TLDW_CONTENT_PG_HOST=localhost
   export TLDW_CONTENT_PG_PORT=5432
   export TLDW_CONTENT_PG_DATABASE=tldw_content
   export TLDW_CONTENT_PG_USER=tldw_user
   export TLDW_CONTENT_PG_PASSWORD="your_password"
   ```

2. Apply migrations and ensure row-level security:

   ```bash
   python -m tldw_Server_API.app.core.DB_Management.content_migrate
   ```

   The command exits non-zero if migrations are pending or required policies are missing.

3. Start the API server. Startup now fails fast when the Postgres backend is misconfigured
   or missing required policies, preventing partial initialization.

### Reverting to SQLite

1. Unset the Postgres-specific environment variables:

   ```bash
   unset CONTENT_DB_MODE TLDW_CONTENT_DB_BACKEND TLDW_CONTENT_PG_HOST \
         TLDW_CONTENT_PG_PORT TLDW_CONTENT_PG_DATABASE TLDW_CONTENT_PG_USER \
         TLDW_CONTENT_PG_PASSWORD
   ```

2. Clear any cached per-user database handles (optional but recommended when switching modes):

   ```bash
   python - <<'PY'
from tldw_Server_API.app.api.v1.API_Deps.DB_Deps import reset_media_db_cache
reset_media_db_cache()
PY
   ```

3. Restart the server; it will automatically fall back to the per-user SQLite databases
   defined by `USER_DB_BASE_DIR`.
   `USER_DB_BASE_DIR` is defined in `tldw_Server_API.app.core.config` (defaults to
   `Databases/user_databases/` under the project root). A non-blank
   `USER_DB_BASE_DIR` environment variable takes precedence over the value in
   `Config_Files/config.txt`; a blank or whitespace-only environment value is ignored.

---

*Last Updated: 2024*
*AuthNZ Database Backend Version: 2.0*
