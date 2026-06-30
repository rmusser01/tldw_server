# AuthNZ Permission Matrix Documentation

## Overview

The tldw_server AuthNZ module implements a comprehensive Role-Based Access Control (RBAC) system with granular permissions. The system supports both single-user and multi-user modes, with automatic permission grants in single-user mode.

## Default Roles

### 1. **Admin** (`admin`)
- **Description**: Full system administrator access
- **Permissions**: ALL permissions granted by default
- **Use Case**: System administrators, server owners

### 2. **User** (`user`)
- **Description**: Standard user access
- **Default Permissions**:
  - `media.create` - Create and upload new media
  - `media.read` - View and access media content
  - `media.update` - Edit media metadata and content
  - `media.transcribe` - Transcribe audio/video content
  - `media.export` - Export media and transcriptions
  - `users.read` - View own user information
  - `api.generate_keys` - Generate personal API keys
- **Use Case**: Regular users of the system

### 3. **Viewer** (`viewer`)
- **Description**: Read-only access
- **Default Permissions**:
  - `media.read` - View and access media content
  - `users.read` - View own user information
- **Use Case**: Users who should only consume content, not create/modify

### 4. **Custom** (`custom`)
- **Description**: Custom role with no default permissions
- **Default Permissions**: NONE (must be explicitly configured)
- **Use Case**: Special roles with specific permission sets

## Permission Categories

### Media Permissions
| Permission | Resource | Action | Description |
|------------|----------|--------|-------------|
| `media.create` | media | create | Create and upload new media files |
| `media.read` | media | read | View and access media content |
| `media.update` | media | update | Edit media metadata and content |
| `media.delete` | media | delete | Delete media items permanently |
| `media.transcribe` | media | transcribe | Transcribe audio/video content |
| `media.export` | media | export | Export media and transcriptions |

### User Management Permissions
| Permission | Resource | Action | Description |
|------------|----------|--------|-------------|
| `users.create` | users | create | Create new user accounts |
| `users.read` | users | read | View user information |
| `users.update` | users | update | Modify user accounts |
| `users.delete` | users | delete | Delete user accounts |
| `users.manage_roles` | users | manage_roles | Assign and revoke user roles |
| `users.invite` | users | invite | Generate registration codes |

### System Permissions
| Permission | Resource | Action | Description |
|------------|----------|--------|-------------|
| `system.configure` | system | configure | Modify system configuration |
| `system.backup` | system | backup | Create and restore backups |
| `system.export` | system | export | Export system data |
| `system.logs` | system | logs | View system and audit logs |
| `system.maintenance` | system | maintenance | Perform system maintenance |

### API Permissions
| Permission | Resource | Action | Description |
|------------|----------|--------|-------------|
| `api.generate_keys` | api | generate_keys | Generate API keys |
| `api.manage_webhooks` | api | manage_webhooks | Configure webhooks |
| `api.rate_limit_override` | api | rate_limit_override | Bypass rate limits |

## Usage in Code

### Claim-First Dependencies (Preferred)

For new FastAPI endpoints, use the unified `AuthPrincipal` dependency stack from `auth_deps`. This keeps authorization claim-first and aligned with the AuthNZ refactor PRDs and the AuthNZ Code Guide.

```python
from fastapi import APIRouter, Depends
from tldw_Server_API.app.api.v1.API_Deps.auth_deps import (
    RequirePermission,
    RequireRole,
    get_auth_principal,
)
from tldw_Server_API.app.core.AuthNZ.principal_model import AuthPrincipal

router = APIRouter()

@router.get("/secure")
async def secure(principal: AuthPrincipal = Depends(get_auth_principal)):
    return {"principal_id": principal.principal_id, "roles": principal.roles}

@router.post("/media/{media_id}", dependencies=[Depends(RequirePermission("media.update"))])
async def update_media(
    media_id: int,
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    return {"ok": True, "media_id": media_id, "by": principal.principal_id}

@router.get("/admin/dashboard", dependencies=[Depends(RequireRole("admin"))])
async def admin_dashboard(
    principal: AuthPrincipal = Depends(get_auth_principal),
):
    return {"ok": True, "admin_id": principal.principal_id}
```

### Legacy Permission Decorators (Historical)

Earlier versions exposed decorator-style FastAPI helpers (`PermissionChecker`, `RoleChecker`, `AnyPermissionChecker`, `AllPermissionsChecker`) from `permissions.py`. These have been removed in favor of the claim-first dependency pattern; new and existing routes should rely on `get_auth_principal` together with `RequirePermission` / `RequireRole` instead.

### Using Permission Functions

```python
from tldw_Server_API.app.core.AuthNZ.permissions import (
    check_permission,
    check_role,
    check_any_permission,
    check_all_permissions
)

# Check single permission
if check_permission(user, "media.delete"):
    # User has permission
    delete_media_item(media_id)

# Check role
if check_role(user, "admin"):
    # User is admin
    show_admin_options()

# Check any permission
if check_any_permission(user, ["media.read", "media.update"]):
    # User has at least one permission
    allow_access()

# Check all permissions
if check_all_permissions(user, ["system.configure", "system.backup"]):
    # User has all required permissions
    perform_system_operation()
```

### Using Decorators (Non-FastAPI)

Note: decorators from `app.core.AuthNZ.permissions` are non-FastAPI helpers for direct Python call sites, not the removed API-dependency shims from `app/api/v1/API_Deps/auth_deps.py`.

```python
from tldw_Server_API.app.core.AuthNZ.permissions import (
    require_permission,
    require_any_permission,
    require_all_permissions
)

@require_permission("media.delete")
def delete_media_item(user: User, media_id: int):
    # Function only executes if user has permission
    pass

@require_any_permission(["media.read", "media.update"])
def access_media(user: User, media_id: int):
    # Function executes if user has any listed permission
    pass

@require_all_permissions(["system.configure", "system.maintenance"])
def critical_operation(user: User):
    # Function executes only if user has all permissions
    pass
```

## Database Schema

### Key Tables

1. **users** - User accounts
2. **roles** - Role definitions
3. **permissions** - Permission definitions
4. **role_permissions** - Maps permissions to roles
5. **user_roles** - Maps roles to users
6. **user_permissions** - Direct permission grants/revokes
7. **registration_codes** - Registration code management
8. **auth_audit_log** - Security audit trail

## Migration Path

## Admin Endpoints (RBAC helpers)

### Role Effective Permissions

- GET `/api/v1/admin/roles/{role_id}/permissions/effective`
  - Returns a convenience combined view of a role’s permissions:
    - `permissions`: non-tool permissions (e.g., `media.read`)
    - `tool_permissions`: tool execution permissions (e.g., `tools.execute:my_tool` or `tools.execute:*`)
    - `all_permissions`: union of both lists

Example (single-user mode):
```bash
curl -s -H "X-API-KEY: $SINGLE_USER_API_KEY" \
  http://127.0.0.1:8000/api/v1/admin/roles/2/permissions/effective | jq
```

OpenAPI example

```yaml
paths:
  /api/v1/admin/roles/{role_id}/permissions/effective:
    get:
      summary: Get role effective permissions
      tags: [admin]
      parameters:
        - name: role_id
          in: path
          required: true
          schema:
            type: integer
      responses:
        '200':
          description: Effective permissions for the role
          content:
            application/json:
              schema:
                type: object
                properties:
                  role_id:
                    type: integer
                  role_name:
                    type: string
                  permissions:
                    type: array
                    items: { type: string }
                  tool_permissions:
                    type: array
                    items: { type: string }
                  all_permissions:
                    type: array
                    items: { type: string }
              examples:
                sample:
                  value:
                    role_id: 2
                    role_name: user
                    permissions:
                      - media.create
                      - media.read
                      - users.read
                    tool_permissions:
                      - tools.execute:*
                    all_permissions:
                      - media.create
                      - media.read
                      - users.read
                      - tools.execute:*
```

### Single-User to Multi-User

1. Run the migration script:
```bash
python tldw_Server_API/app/core/AuthNZ/migrate_to_multiuser.py
```

2. Update configuration:
```ini
AUTH_MODE = multi_user
ENABLE_REGISTRATION = true
REQUIRE_REGISTRATION_CODE = true
JWT_SECRET_KEY = <secure-random-key>
```

3. Restart the server

### Creating Custom Roles

```python
from tldw_Server_API.app.core.DB_Management.UserDatabase import UserDatabase

# Initialize database
user_db = UserDatabase("../Databases/Users.db", "admin_client")

# Create custom role (would need to add this method)
conn = user_db.get_connection()
# Cross-backend placeholder guidance:
# - Prefer Postgres-style placeholders ($1,$2,...) in examples; SQLite adapters in this project translate $N→? automatically.
# - If using raw sqlite3 instead of the project adapters, change $N placeholders below to '?'.
cursor = conn.execute("""
    INSERT INTO roles (name, description, is_system)
    VALUES ($1, $2, $3)
""", ("content_moderator", "Can moderate user content", 0))
role_id = cursor.lastrowid

# Assign specific permissions
permissions = ["media.read", "media.update", "media.delete", "users.read"]
for perm in permissions:
    cursor = conn.execute("""
        INSERT INTO role_permissions (role_id, permission_id)
        SELECT $1, id FROM permissions WHERE name = $2
    """, (role_id, perm))

conn.commit()
```

## Security Considerations

1. **Default Deny**: Users have no permissions unless explicitly granted
2. **Role Inheritance**: No automatic inheritance - permissions must be explicitly assigned
3. **Custom Role**: Starts with zero permissions for maximum security
4. **Audit Trail**: All permission changes are logged in `auth_audit_log`
5. **Session Management**: JWT tokens with configurable expiration
6. **Rate Limiting**: Built-in rate limiting with override permission

## Best Practices

1. **Principle of Least Privilege**: Grant minimum permissions needed
2. **Use Roles**: Assign permissions via roles rather than direct grants
3. **Regular Audits**: Review `auth_audit_log` regularly
4. **Custom Roles**: Create specific roles for specialized users
5. **Permission Grouping**: Group related permissions in custom roles
6. **Documentation**: Document custom roles and their purposes

## Troubleshooting

### Common Issues

1. **"Permission Denied" errors**
   - Check user's roles: `user_db.get_user_roles(user_id)`
   - Check user's permissions: `user_db.get_user_permissions(user_id)`
   - Verify permission exists in database

2. **Registration codes not working**
   - Check code validity: `user_db.validate_registration_code(code)`
   - Verify registration is enabled in settings
   - Check code hasn't expired or been exhausted

3. **User can't login**
   - Check account is active: `user.is_active`
   - Check account isn't locked: `user_db.is_account_locked(user_id)`
   - Verify password is correct

## Future Enhancements

- [ ] Permission inheritance/hierarchy
- [ ] Time-based permissions
- [ ] IP-based restrictions
- [ ] Resource-specific permissions (per media item)
- [ ] Permission delegation
- [ ] Role templates
- [ ] Permission groups/categories

---

*Last Updated: 2024*
*AuthNZ Module Version: 1.0*
