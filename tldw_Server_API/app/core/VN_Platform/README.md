# VN_Platform

VN_Platform contains shared visual novel platform helpers that are reused across VN assets, scripts, policy, and play APIs. It derives route capabilities, standardizes VN error payloads, and builds canonical idempotency hashes for JSON, stream, and multipart request bodies.

## Start Here

- `capabilities.py` derives platform capability output from registered VN routes.
- `errors.py` formats VN-specific API error details.
- `idempotency.py` builds canonical request hashes for idempotent VN operations.
- Related API surface: `app/api/v1/endpoints/vn_capabilities.py`.
- Related tests: `tests/VN_Platform/`.

## Responsibilities

- Build the VN capabilities response from available API routes.
- Provide consistent VN error detail payloads for endpoint modules.
- Hash canonical JSON payloads and byte streams for idempotency checks.
- Hash multipart payloads in a stable way across repeated requests.
- Keep cross-VN support logic outside individual feature modules.

## Module Map

- `capabilities.py` - route-derived capability builder.
- `errors.py` - VN error response helper.
- `idempotency.py` - canonical JSON, stream, and multipart hashing helpers.

## How It Connects

- `app/api/v1/endpoints/vn_capabilities.py` exposes platform capabilities.
- VN feature endpoints such as `vn_assets.py`, `vn_play.py`, `vn_policy.py`, and `vn_scripts.py` use platform error and idempotency helpers.
- `app/core/VN_Assets/`, `app/core/VN_Play/`, `app/core/VN_Policy/`, and `app/core/VN_Scripts/` rely on these shared helpers for common API behavior.

## Extension Points

- For a new VN API namespace, update `capabilities.py` and capability endpoint tests.
- For new error fields, update `errors.py` and the platform error tests.
- For a new idempotency payload type, extend `idempotency.py` and add canonicalization tests.

## Testing

- `tests/VN_Platform/test_vn_capabilities_api.py`
- `tests/VN_Platform/test_vn_platform_errors.py`
- `tests/VN_Platform/test_vn_platform_idempotency.py`
- `tests/VN_Platform/test_vn_route_namespace.py`

## Gotchas

- Capabilities are route-derived, so route renames can change capability output.
- Idempotency hashes depend on canonical payload serialization; avoid ad hoc hashing in endpoint modules.
