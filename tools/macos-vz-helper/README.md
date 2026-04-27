# macOS VZ Helper

This directory is the planned home for the first-party native macOS helper used by the
`vz_linux` sandbox runtime.

## Scope

The helper is intentionally narrow:

- `vz_linux` only
- local Unix-socket daemon
- owns `Virtualization.framework` lifecycle
- owns host readiness and runnable-template truth
- owns VM create, exec, status, list, and terminate operations

## Non-Goals

The first helper slice is not a generic macOS sandbox backend:

- no `vz_macos` support yet
- no `seatbelt` support here
- no second persistence layer for sandbox sessions
- no APFS clone manager in the first transport slice

Python remains authoritative for sandbox admission, session identity, artifacts, and ACP
integration. The helper only owns runtime VM facts and control-plane operations.
