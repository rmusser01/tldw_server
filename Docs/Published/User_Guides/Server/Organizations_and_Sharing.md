# Organizations and Content Sharing Guide

This guide explains how to join organizations, collaborate with teams, and share content with other users.

## Introduction

### What are Organizations and Teams?

**Organizations** are the top-level grouping for collaboration. An organization can represent a company, department, research group, or any other collective that needs to share resources.

**Teams** exist within organizations and provide finer-grained collaboration. For example, a company organization might have separate teams for Engineering, Marketing, and Research.

### Benefits of Team Collaboration

- **Shared Knowledge Base**: Access media, transcripts, and notes shared by team members
- **Unified RAG Search**: Search across all content your team has shared
- **Role-Based Access**: Different permission levels ensure appropriate access control
- **Invite-Based Onboarding**: Simple invite codes for adding new members

## Joining an Organization

### Receiving an Invite Code

Organization administrators create invite codes that can be shared with you. These codes typically look like: `ABC123XYZ`

Invite codes have:
- An expiration date (1-365 days from creation)
- A maximum number of uses (1-1000)
- A role assignment (member, lead, or admin)

### Previewing an Invite

Before redeeming an invite, you can preview what you'll be joining:

**API Request:**
```bash
curl "http://localhost:8000/api/v1/invites/preview?code=ABC123XYZ"
```

**Response:**
```json
{
  "org_name": "Acme Research",
  "org_slug": "acme-research",
  "team_name": null,
  "role_to_grant": "member",
  "is_valid": true,
  "status": "valid"
}
```

If `team_name` is present, you'll also be added to that specific team.

### Redeeming the Invite

To join the organization, redeem the invite code while authenticated:

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/invites/redeem \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"code": "ABC123XYZ"}'
```

**Response:**
```json
{
  "success": true,
  "org_id": 1,
  "org_name": "Acme Research",
  "team_id": null,
  "role": "member",
  "was_already_member": false
}
```

### What Happens After You Join

Once you redeem an invite:
1. You become a member of the organization with the specified role
2. If the invite was team-specific, you're also added to that team
3. You gain access to content shared at the org or team level
4. You can search across shared content in RAG queries

## Viewing Your Memberships

### Listing Your Organizations

**API Request:**
```bash
curl http://localhost:8000/api/v1/orgs \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "organizations": [
    {
      "id": 1,
      "name": "acme-research",
      "display_name": "Acme Research",
      "role": "member"
    }
  ],
  "count": 1
}
```

### Understanding Your Role

| Role | Description |
|------|-------------|
| **owner** | Full control over the organization, including billing and deletion |
| **admin** | Can manage members, teams, and invites |
| **lead** | Team leadership role with limited org-level permissions |
| **member** | Basic access to view org and access shared content |

## Content Visibility

Content in tldw_server has three visibility levels:

### Personal (Default)

- Only you can see this content
- This is the default for all new content
- Works with both SQLite and PostgreSQL backends

### Team

- Visible to all members of a specific team
- Requires PostgreSQL backend with RLS enabled
- You must be a member of the team to see team content

### Organization

- Visible to all members of the organization
- Requires PostgreSQL backend with RLS enabled
- Any org member can access org-level content

## Sharing Your Content

### How to Share Media with Your Team

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/media/123/share \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"visibility": "team", "team_id": 77}'
```

### How to Share Media with Your Organization

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/media/123/share \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"visibility": "org"}'
```

### Changing Visibility Back to Personal

**API Request:**
```bash
curl -X DELETE http://localhost:8000/api/v1/media/123/share \
  -H "Authorization: Bearer YOUR_TOKEN"
```

### Who Can See Your Shared Content?

| Visibility | Who Can Access |
|------------|----------------|
| Personal | Only you (the owner) |
| Team | All members of the specified team |
| Org | All members of your organization |

Note: You always retain ownership of your content. Sharing changes who can view it, not who owns it.

## Recipient Shared Research Workspaces

A recipient opens a shared research workspace at
`/research-workspace?shared={share_id}`. The `/research` route is a separate
experience and does not accept or redirect this recipient flow. If the share is
unavailable, the recipient view fails closed instead of falling back to a local
workspace.

Recipient access requires `sharing.read` and an active, authoritative team
membership. A missing permission returns `403`; a missing, revoked, or
unauthorized share returns the same neutral `404` so callers cannot enumerate
shares.

### Recipient Data Ownership and Scope

- Workspace metadata, shared source metadata, source previews, and source
  content remain authoritative in the owner's databases.
- Recipient chat transcripts, citations, and request receipts are stored in the
  recipient's database. Owner notes and chats are never exposed or used as a
  recipient persistence target.
- Shared mode cannot read a recipient's local workspaces or use Studio, notes,
  artifacts, MCP, ACP, sandbox, or extension writable destinations.
- Recipients have read and chat access only. They cannot mutate shared sources.
- Chat accepts an explicit frozen source scope: `all` for every source in the
  share, or `include` with specific shared source IDs.

The API may disclose the provider and model selected for the shared chat. It
never returns owner API keys or base URLs. Generation is limited to the exact
share scope and uses the recipient's credentials.

Authorization is rechecked before owner-resource loading, before generation,
and before persistence. Revocation prevents later access and prevents an
in-flight response from being saved, but it cannot recall content that was
already delivered or saved before revocation.

### Canonical Recipient API

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/api/v1/sharing/shared-with-me` | List shares available to the recipient |
| `GET` | `/api/v1/sharing/shared-with-me/{share_id}/workspace` | Read shared workspace metadata |
| `GET` | `/api/v1/sharing/shared-with-me/{share_id}/sources` | List and filter shared sources |
| `GET` | `/api/v1/sharing/shared-with-me/{share_id}/sources/{source_id}/preview` | Preview a shared source |
| `GET` | `/api/v1/sharing/shared-with-me/{share_id}/chat/messages` | Read the recipient's shared chat transcript |
| `POST` | `/api/v1/sharing/shared-with-me/{share_id}/chat` | Ask against the frozen shared source scope |

These are the only recipient workspace operations. There is no redirect,
alias, local fallback, or raw full-media recipient operation.

## Searching Shared Content

### Scope Filters

When searching, you can filter by content scope:

**API Request:**
```bash
curl -X POST http://localhost:8000/api/v1/media/search \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "scope": "team"}'
```

**Available Scopes:**
| Scope | Description |
|-------|-------------|
| `personal` | Only your personal content |
| `team` | Content shared with your teams |
| `org` | Content shared with your organization |
| `all` | All content you have access to |

### RAG Search Across Shared Content

The RAG pipeline respects visibility settings. When you perform a RAG search, it automatically includes:
- Your personal content
- Team content (if you're a team member)
- Org content (if you're an org member)

This means search results and LLM context will include relevant shared content from your collaborators.

## Tips and Best Practices

### PostgreSQL Required for Sharing

Team and organization content sharing requires PostgreSQL with Row-Level Security (RLS). SQLite deployments are limited to personal content.

If you need team sharing features, ensure your deployment uses PostgreSQL.

### Content Ownership

- You always own content you create
- Sharing only changes visibility, not ownership
- You can change visibility at any time
- Only you (or platform admins) can delete your content

### Visibility Changes Are Immediate

When you change content visibility:
- Team members can immediately access team-visible content
- Org members can immediately access org-visible content
- Reverting to personal immediately restricts access

## Troubleshooting

### "Invite code not found" Error

This error occurs when:
- The invite code was typed incorrectly
- The invite was revoked by an administrator
- The organization was deleted

**Solution:** Request a new invite code from your organization administrator.

### "Invite has expired" Error

Invite codes have an expiration date set when created.

**Solution:** Request a fresh invite code from your organization administrator.

### "Invite limit reached" Error

Invite codes have a maximum number of uses.

**Solution:** Request a new invite code or ask the administrator to create one with more uses.

### Cannot See Shared Content

If you can't see content that should be shared with you:

1. **Check your membership:** Ensure you're a member of the relevant org/team
2. **Check the backend:** Team/org sharing requires PostgreSQL
3. **Check the visibility:** Confirm the content is shared at the right level
4. **Check the scope filter:** Ensure your search scope includes the content type

### "PostgreSQL required" Error

This error appears when trying to share content on a SQLite deployment.

**Solution:** Either:
- Continue using personal visibility (works on SQLite)
- Ask your administrator to migrate to PostgreSQL

## Related Documentation

- [Organization Administration Guide](../Server/Organization_Administration.md) - For org owners and admins
- [API Design](../../API-related/API_Design.md) - General API documentation
- [User Guide](../WebUI_Extension/User_Guide.md) - Overall platform guide
