# Organization Administration Guide

This guide covers organization management for owners, administrators, team leads, and platform operators.

## Introduction

### Organization Hierarchy

```
Organization (e.g., "Acme Corp")
├── Members (owner, admins, leads, members)
├── Teams
│   ├── Team A (e.g., "Engineering")
│   │   └── Team Members
│   └── Team B (e.g., "Research")
│       └── Team Members
├── Invites
└── Subscription (billing plan)
```

### Role-Based Access Control

Organizations use a hierarchical role system:
- **Owner**: Full control, including deletion
- **Admin**: Can manage members, teams, and invites
- **Lead**: Team leadership with limited org-level permissions
- **Member**: Basic access to view org and shared content

## Creating and Managing Organizations

### Creating a New Organization

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/orgs \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "my-company",
    "display_name": "My Company Inc."
  }'
```

The creator automatically becomes the organization owner.

### Updating Organization Settings

**API Request:**
```bash
curl -X PATCH http://localhost:8000/api/v1/orgs/1 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "display_name": "My Company International"
  }'
```

Requires: owner or admin role.

### Transferring Ownership

Transfer ownership to another existing member:

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/orgs/1/transfer \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"new_owner_id": 42}'
```

Requires: owner role. The previous owner is demoted to admin.

### Deleting an Organization

**API Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/orgs/1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Prerequisites:**
- Must be the owner
- All invites will be automatically revoked

## Managing Members

### Viewing Organization Members

**API Request:**
```bash
curl http://localhost:8000/api/v1/orgs/1/members \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "members": [
    {
      "user_id": 1,
      "username": "alice",
      "email": "alice@example.com",
      "role": "owner",
      "added_at": "2024-01-15T10:30:00Z"
    },
    {
      "user_id": 2,
      "username": "bob",
      "email": "bob@example.com",
      "role": "member",
      "added_at": "2024-01-20T14:00:00Z"
    }
  ],
  "count": 2
}
```

### Adding Members Directly

Add a user by their user ID (alternative to invite codes):

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/orgs/1/members \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"user_id": 42, "role": "member"}'
```

Requires: owner or admin role.

### Updating Member Roles

**API Request:**
```bash
curl -X PATCH http://localhost:8000/api/v1/orgs/1/members/42 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"role": "admin"}'
```

**Valid roles:** `member`, `lead`, `admin`

Note: You cannot grant the `owner` role via this endpoint; use ownership transfer instead.

### Removing Members

**API Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/orgs/1/members/42 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Note: You cannot remove the owner. Transfer ownership first if needed.

## Teams

### Creating Teams

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/orgs/1/teams \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "name": "engineering",
    "description": "Engineering team"
  }'
```

Requires: owner or admin role.

### Listing Teams

**API Request:**
```bash
curl http://localhost:8000/api/v1/orgs/1/teams \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Managing Team Settings

**API Request:**
```bash
curl -X PATCH http://localhost:8000/api/v1/orgs/1/teams/5 \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"description": "Core Engineering team"}'
```

### Deleting Teams

**API Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/orgs/1/teams/5 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Requires: owner or admin role.

## Invite Codes

Invite codes allow you to onboard new members without knowing their user IDs.

### Creating Invite Codes

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/orgs/1/invites \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "max_uses": 10,
    "expiry_days": 7,
    "role_to_grant": "member",
    "team_id": null,
    "description": "Q1 onboarding invites"
  }'
```

**Parameters:**

| Parameter | Required | Default | Description |
|-----------|----------|---------|-------------|
| `max_uses` | No | 1 | Maximum redemptions (1-1000) |
| `expiry_days` | No | 7 | Days until expiration (1-365) |
| `role_to_grant` | No | member | Role to assign: `member`, `lead`, or `admin` |
| `team_id` | No | null | Also add to this team |
| `description` | No | null | Internal note about this invite |

**Response:**
```json
{
  "id": 1,
  "code": "ABC123XYZ",
  "org_id": 1,
  "role_to_grant": "member",
  "max_uses": 10,
  "uses_count": 0,
  "expires_at": "2024-01-22T10:30:00Z",
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Team-Specific Invites

To add users to both the org and a specific team:

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/orgs/1/invites \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "max_uses": 5,
    "expiry_days": 14,
    "role_to_grant": "member",
    "team_id": 5
  }'
```

### Listing Active Invites

**API Request:**
```bash
curl "http://localhost:8000/api/v1/orgs/1/invites?include_expired=false" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Monitoring Invite Usage

The response includes `uses_count` showing how many times the invite has been redeemed.

### Revoking Invites

**API Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/orgs/1/invites/1 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

Revoked invites cannot be redeemed even if they haven't expired.

## Role Permissions Matrix

### Organization Roles

| Action | owner | admin | lead | member |
|--------|:-----:|:-----:|:----:|:------:|
| View org | Y | Y | Y | Y |
| Update org settings | Y | Y | - | - |
| Delete org | Y | - | - | - |
| Create/delete teams | Y | Y | - | - |
| Manage org members | Y | Y | - | - |
| Create invite codes | Y | Y | - | - |
| Transfer ownership | Y | - | - | - |

### Team Roles

| Action | org_owner | org_admin | team_lead | team_member |
|--------|:---------:|:---------:|:---------:|:-----------:|
| View team | Y | Y | Y | Y |
| Update team settings | Y | Y | Y | - |
| Delete team | Y | Y | - | - |
| Manage team members | Y | Y | Y | - |

## Billing, Quotas and Limits

### What the open-source server ships

The open-source server has no billing or payment runtime:

- `is_billing_enabled()` returns `False` (`core/Billing/runtime_flags.py`), and no setting named `BILLING_ENABLED` is read anywhere.
- There are no checkout, billing portal, usage or cancel routes. The only billing route is the admin-only `GET /api/v1/billing/subscriptions`, which lists subscriptions with lifecycle and at-risk fields.
- Organization roles carry no billing permissions.

### Billing plan limits (hosted product only)

Plan limits (`storage_mb`, `api_calls_day`, `llm_tokens_month`, `team_members`, `transcription_minutes_month`, `rag_queries_day`, `concurrent_jobs`) are checked only when both of these are true:

1. a billing repository is wired into the subscription service. Nothing in this repository wires one; a hosted deployment adds it;
2. usage quotas are on (`USAGE_QUOTAS_ENABLED`).

With no billing repository, billing checks never run: org resolution and usage aggregation are skipped and an account with no active org is not refused. If a repository is wired while usage quotas are off, startup logs a warning once that plan limits are not enforced.

When plan limits do run:

- A request that reaches the soft limit (80%) is allowed and the response carries an `X-Billing-Warning` header.
- A request that would exceed a limit is refused with HTTP 402, or 429 for a hard block, with a `limit_exceeded` body and a `Retry-After` header.
- `team_members` and `concurrent_jobs` are defined, but no endpoint checks them today.

### Per-user, team and org quotas

On every install, quotas are per-user `limits.*` values, resolved from the user's own value, then the most generous team value, then the most generous org value. A team or org value is each member's allowance, not a shared pool. Nothing is set by default, so nothing is limited until a platform admin assigns a value and turns `USAGE_QUOTAS_ENABLED` on. Org admins cannot change these values; only platform admins can.

See `Docs/Operations/Usage_Quotas.md` for the keys, the routes that set them, and what each refusal looks like.

## Platform Admin Features

Platform administrators (separate from org admins) have additional capabilities.

### Admin Endpoints

Platform admin endpoints at `/api/v1/admin/orgs/*`:
- Create and list organizations
- Create and list an organization's teams, and add, list, update and remove its members
- Set `limits.*` allowances for an org or team (see Usage Quotas)

### Managing All Organizations

**API Request:**
```bash
curl http://localhost:8000/api/v1/admin/orgs \
  -H "Authorization: Bearer ADMIN_TOKEN"
```

### Setting Usage Limits

Platform admins set a team's or org's allowance with `limits.*` overrides:

```bash
curl -X PUT http://localhost:8000/api/v1/admin/orgs/1/profile/overrides/limits.rag_queries_per_day \
  -H "Authorization: Bearer ADMIN_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"value": 500}'
```

`DELETE` the same path to remove the value. Teams use `/api/v1/admin/teams/{team_id}/profile/overrides/{key}`. See `Docs/Operations/Usage_Quotas.md`.

## Best Practices

### Invite Management

1. **Use descriptive names**: Add descriptions to invites for tracking
2. **Set reasonable expiry**: 7-14 days is typical for onboarding
3. **Monitor usage**: Check `uses_count` to track adoption
4. **Revoke unused invites**: Clean up old invites regularly

### Team Structure

1. **Logical grouping**: Organize teams by function or project
2. **Appropriate roles**: Grant minimum necessary permissions
3. **Regular audits**: Review memberships periodically

### Usage Limits

1. **Leave limits unset unless you need them**: a quota applies only where a value was assigned
2. **Check what users see**: `GET /api/v1/users/storage`, `GET /api/v1/audio/stream/limits` and the profile `quotas` section report the limit in force; `null` means unlimited
3. **Prefer a team or org value for groups**: it is each member's allowance, so adding a member never needs a new value

## Related Documentation

- [Organizations and Sharing Guide](../Server/Organizations_and_Sharing.md) - For end users
- `Docs/Operations/Usage_Quotas.md` - Operator guide to per-user, team and org limits
- [Admin Orgs and Teams API Reference](../../API-related/Admin_Orgs_Teams.md) - Full API documentation
- [Production Hardening Checklist](../Server/Production_Hardening_Checklist.md) - Security best practices
