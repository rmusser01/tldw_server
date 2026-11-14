# PostgreSQL Setup Guide for tldw_server Testing

This guide will help you set up PostgreSQL to test the multi-user authentication system.

## Quick Start Options

### Option 1: Docker (Fastest - Recommended)

If you have Docker installed, this is the quickest way:

```bash
# 1. Start PostgreSQL container
docker run --name tldw-postgres \
  -e POSTGRES_USER=tldw_user \
  -e POSTGRES_PASSWORD=TestPassword123! \
  -e POSTGRES_DB=tldw_multiuser \
  -p 5432:5432 \
  -d postgres:18-bookworm

# 2. Wait a few seconds for it to start, then apply the schema
sleep 5
docker exec -i tldw-postgres psql -U tldw_user -d tldw_multiuser < tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql

# 3. Test the connection
python test_postgres_connection.py

# 4. When done testing, stop and remove the container
docker stop tldw-postgres
docker rm tldw-postgres
```

### Option 2: macOS with Homebrew

```bash
# 1. Install PostgreSQL
brew install postgresql@14

# 2. Start PostgreSQL
brew services start postgresql@14

# 3. Create database and user
psql postgres <<EOF
CREATE DATABASE tldw_multiuser;
CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'TestPassword123!';
GRANT ALL PRIVILEGES ON DATABASE tldw_multiuser TO tldw_user;
\c tldw_multiuser
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
GRANT ALL ON SCHEMA public TO tldw_user;
EOF

# 4. Apply schema
psql -U tldw_user -h localhost -d tldw_multiuser -f tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql
# Password: TestPassword123!

# 5. Test connection
python test_postgres_connection.py
```

### Option 3: Postgres.app (macOS GUI)

1. Download from https://postgresapp.com/
2. Install and start Postgres.app
3. Click "Initialize" to create a new server
4. Open psql from the app and run:

```sql
CREATE DATABASE tldw_multiuser;
CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'TestPassword123!';
GRANT ALL PRIVILEGES ON DATABASE tldw_multiuser TO tldw_user;
\c tldw_multiuser
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
GRANT ALL ON SCHEMA public TO tldw_user;
\q
```

5. Apply schema:
```bash
psql -U tldw_user -h localhost -d tldw_multiuser -f tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql
```

### Option 4: Ubuntu/Debian

```bash
# 1. Install PostgreSQL
sudo apt update
sudo apt install postgresql postgresql-contrib

# 2. Switch to postgres user and create database
sudo -u postgres psql <<EOF
CREATE DATABASE tldw_multiuser;
CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'TestPassword123!';
GRANT ALL PRIVILEGES ON DATABASE tldw_multiuser TO tldw_user;
\c tldw_multiuser
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
GRANT ALL ON SCHEMA public TO tldw_user;
EOF

# 3. Configure PostgreSQL to allow password authentication
sudo nano /etc/postgresql/14/main/pg_hba.conf
# Change the line for local connections from 'peer' to 'md5':
# local   all             all                                     md5

# 4. Restart PostgreSQL
sudo systemctl restart postgresql

# 5. Apply schema
psql -U tldw_user -h localhost -d tldw_multiuser -f tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql

# 6. Test connection
python test_postgres_connection.py
```

## Environment Configuration

Create a `.env` file in the project root:

```bash
cat > .env << 'EOF'
# Database Configuration
AUTH_MODE=multi_user
DATABASE_URL=postgresql://tldw_user:TestPassword123!@localhost/tldw_multiuser

# JWT Configuration
JWT_SECRET_KEY=test-secret-key-for-testing-only
JWT_ALGORITHM=HS256
ACCESS_TOKEN_EXPIRE_MINUTES=60
REFRESH_TOKEN_EXPIRE_DAYS=30

# Registration Settings
ENABLE_REGISTRATION=true
REQUIRE_REGISTRATION_CODE=false
DEFAULT_USER_ROLE=user
# DEFAULT_STORAGE_QUOTA_MB is deprecated: it sets no one's quota (setting it logs a warning).
# Per-user storage quotas are limits.storage_quota_mb values; see Docs/Operations/Usage_Quotas.md

# Rate Limiting
EOF
```

## Testing the Setup

### 1. Verify PostgreSQL Connection

```bash
# Run the test script
python test_postgres_connection.py
```

Expected output:
```
✅ Connected to PostgreSQL successfully!
📊 PostgreSQL version: PostgreSQL 14.x
📋 Found 7 tables:
  - audit_log
  - rate_limits
  - registration_codes
  - sessions
  - user_preferences
  - user_storage
  - users
```

### 2. Start the Server in Multi-User Mode

```bash
# Stop the current server if running (Ctrl+C)
# Start with multi-user configuration
python -m uvicorn tldw_Server_API.app.main:app --reload
```

### 3. Create Admin User

```bash
# Using the migration script
python tldw_Server_API/scripts/migrate_to_multiuser.py \
  --admin-email admin@example.com \
  --admin-password AdminPass123! \
  --no-preserve-data  # Since this is a fresh install
```

### 4. Test Authentication Endpoints

```bash
# Test registration (should work since ENABLE_REGISTRATION=true)
curl -X POST http://localhost:8000/api/v1/auth/register \
  -H "Content-Type: application/json" \
  -d '{
    "username": "testuser",
    "email": "test@example.com",
    "password": "TestPass123!"
  }'

# Test login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=testuser&password=TestPass123!"

# Save the token from the response
TOKEN="<access_token_from_response>"

# Test authenticated endpoint
curl http://localhost:8000/api/v1/auth/me \
  -H "Authorization: Bearer $TOKEN"

# Test admin login
curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=admin&password=AdminPass123!"
```

### 5. Run Comprehensive Tests

```bash
# Run the full test suite
python test_auth_endpoints.py

# Or run the pytest suite
python -m pytest tldw_Server_API/tests/AuthNZ/test_auth_comprehensive.py -v
```

## Troubleshooting

### Connection Refused Error
```
Connect call failed ('127.0.0.1', 5432)
```
**Solution**: PostgreSQL is not running. Start it:
- Docker: `docker start tldw-postgres`
- macOS: `brew services start postgresql@14`
- Linux: `sudo systemctl start postgresql`

### Authentication Failed
```
FATAL: password authentication failed for user "tldw_user"
```
**Solution**: User doesn't exist or wrong password. Recreate user:
```sql
DROP USER IF EXISTS tldw_user;
CREATE USER tldw_user WITH ENCRYPTED PASSWORD 'TestPassword123!';
```

### Database Does Not Exist
```
FATAL: database "tldw_multiuser" does not exist
```
**Solution**: Create the database:
```sql
CREATE DATABASE tldw_multiuser;
```

### Permission Denied
```
ERROR: permission denied for schema public
```
**Solution**: Grant permissions:
```sql
GRANT ALL ON SCHEMA public TO tldw_user;
GRANT ALL ON ALL TABLES IN SCHEMA public TO tldw_user;
```

### Tables Not Found
```
ERROR: relation "users" does not exist
```
**Solution**: Apply the schema:
```bash
psql -U tldw_user -h localhost -d tldw_multiuser -f tldw_Server_API/Databases/Postgres/Schema/postgresql_users.sql
```

## Cleanup

When you're done testing:

### Docker
```bash
docker stop tldw-postgres
docker rm tldw-postgres
```

### Local Installation
```bash
# Drop the test database
psql postgres -c "DROP DATABASE IF EXISTS tldw_multiuser;"
psql postgres -c "DROP USER IF EXISTS tldw_user;"

# Stop PostgreSQL (macOS)
brew services stop postgresql@14

# Stop PostgreSQL (Linux)
sudo systemctl stop postgresql
```

## Next Steps

Once PostgreSQL is working:

1. **Test all endpoints**: Use the Swagger UI at http://localhost:8000/docs
2. **Check audit logs**: Query the audit_log table to see all events
3. **Test rate limiting**: Enable it in .env and test limits
4. **Test concurrent users**: Create multiple users and test simultaneous access
5. **Performance testing**: Use a tool like Apache Bench or locust

## Summary

The multi-user mode requires PostgreSQL to be running and configured. The easiest way to test is using Docker, which provides an isolated environment without affecting your system. Once testing is complete, you can deploy to production using the deployment guide.

Remember to:
- Never use test passwords in production
- Always use SSL/TLS in production
- Set strong JWT secrets
- Enable rate limiting
- Configure proper backups

For production deployment, refer to: `/Docs/User_Guides/Server/Multi-User_Deployment_Guide.md`
