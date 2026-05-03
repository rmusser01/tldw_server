# API Versioning Strategy

## Current Versioning

All API endpoints are served under the `/api/v1/` prefix. This path-based versioning provides clear, explicit version identification in every request.

```
https://tldw.example.com/api/v1/chat/completions
https://tldw.example.com/api/v1/media/search
```

## Versioning Rules

### What Constitutes a Breaking Change

The following changes require a major version bump (e.g., v1 to v2):

- **Removing** an endpoint or field from a response
- **Renaming** an endpoint path or response field
- **Changing the type** of an existing response field (e.g., string to integer)
- **Changing required/optional status** of a request field (optional to required)
- **Changing authentication requirements** for an endpoint
- **Changing error response format** or status codes for existing error conditions

### Non-Breaking Changes (No Version Bump)

- Adding new endpoints
- Adding new optional fields to request bodies
- Adding new fields to response bodies
- Adding new query parameters (optional)
- Adding new enum values to existing fields
- Relaxing validation (e.g., increasing max length)

## Phase 3 / Phase 4 Compatibility Policy

During the Phase 3 and Phase 4 contract cleanup work, `/api/v1/` remains
legacy-default.

This means:

- existing default response bodies remain unchanged unless an additive change is
  non-breaking under the rules above;
- existing default error shapes and status behavior remain unchanged;
- existing accepted pagination aliases remain valid in `v1` unless a future
  versioned migration explicitly changes them.

Standard response-envelope behavior inside `v1` is transitional and additive.
If a first-party JSON route supports envelope-style opt-in behavior, that opt-in
does not create a new major-version mechanism. Major default-breaking contract
changes should move to `/api/v2/`, not to a public header-only version.

### Transitional Opt-In In `v1`

For first-party JSON routes, additive opt-in behavior may be exposed with:

```http
X-TLDW-Response-Envelope: v1
```

This header is a compatibility tool for additive behavior inside `v1`. It does
not replace path-based major versioning.

### Exempt Route Families

The following route families are exempt from standard envelopes by default
unless they explicitly document otherwise:

- streaming responses
- file downloads and binary exports
- webhooks
- WebSockets
- `204 No Content` routes
- OpenAI-compatible and provider-compatible payloads

### Default-Breaking Migration Triggers

The following changes should move to a sibling route or `/api/v2/` instead of
changing the default `v1` contract in place:

- wrapping the default response body under `data`
- changing the default error body shape
- removing or renaming existing response fields
- removing accepted pagination aliases
- changing auth requirements or auth/status behavior
- converting provider-compatible defaults into tldw-specific normalized shapes

## Deprecation Policy

The deprecation policy below applies when maintainers approve an actual
deprecation window. It should not be interpreted to mean that additive `v1`
pilot behavior automatically deprecates the existing default `v1` shape.

### Timeline

1. **Announcement** (Day 0): The deprecation is documented in the changelog, API docs, and response headers.
2. **Warning Period** (6 months): The deprecated endpoint/field continues to work but returns a `Deprecation` header and optionally a `Sunset` header:
   ```
   Deprecation: true
   Sunset: Sat, 01 Jan 2028 00:00:00 GMT
   ```
3. **Removal** (after 6 months): The endpoint/field is removed in the next major version.

### Deprecation Headers

Deprecated endpoints include standard headers:

```http
HTTP/1.1 200 OK
Deprecation: true
Sunset: 2028-01-01T00:00:00Z
Link: <https://docs.tldw.example.com/migration/v2>; rel="successor-version"
```

## Version Coexistence

When v2 is introduced:

- `/api/v1/` endpoints continue to operate during the deprecation period.
- `/api/v2/` endpoints are available immediately.
- Both versions share the same database and authentication system.
- Internal business logic is version-agnostic; versioning is handled at the endpoint/schema layer.

## Migration Guide Format

Each major version bump includes a migration guide in `Docs/API/migrations/`:

```
Docs/API/migrations/
  v1-to-v2.md
```

A migration guide contains:

1. **Summary** of changes
2. **Endpoint mapping** table (old path to new path)
3. **Field changes** table (old field to new field, with type changes)
4. **Code examples** showing before/after for common operations
5. **Timeline** with key dates (deprecation start, sunset)

## Changelog Format

The project changelog (`CHANGELOG.md` at repo root, when created) follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) conventions:

```markdown
## [Unreleased]

### Added
- New endpoint: `POST /api/v1/foo/bar`

### Changed
- `GET /api/v1/baz` now returns `total_count` field in response

### Deprecated
- `GET /api/v1/old-endpoint` -- use `GET /api/v1/new-endpoint` instead (sunset: 2028-01-01)

### Removed
- `GET /api/v1/removed-endpoint` (deprecated since v0.9)

### Fixed
- Fixed 500 error on `POST /api/v1/media/process` with empty URL list
```

## Client Guidance

- Always specify the version prefix in your base URL.
- Subscribe to the project changelog for deprecation notices.
- Test against the `/docs` (Swagger UI) endpoint after upgrades.
- Use the `Deprecation` and `Sunset` response headers for automated migration tracking.
