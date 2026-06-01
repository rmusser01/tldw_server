# Security

The Security module centralizes outbound network policy, safe serialization,
secret lookup, request IDs, setup access controls, CSP handling, HTTP hardening,
drain-gate middleware, and URL validation helpers. It is used by API startup,
web scraping, workflows, setup, storage, and any code path that reaches
untrusted URLs or sensitive configuration.

## Start Here

- Egress/SSRF controls: `egress.py` and `url_validation.py`.
- HTTP and request middlewares: `middleware.py`, `request_id_middleware.py`,
  `setup_access_guard.py`, `setup_csp.py`, and `drain_gate_middleware.py`.
- Secrets and crypto: `secret_manager.py`, `crypto.py`, and `safe_pickle.py`.
- Startup wiring: `app/main.py`.
- Tests: `tests/Security/` plus downstream WebScraping and Text2SQL security
  tests.

## Responsibilities

- Reject unsafe outbound URLs before network calls.
- Set security headers, CSP variants, HSTS behavior, and request IDs.
- Guard remote Setup UI access and setup-specific CSP nonces.
- Load and validate secrets without hard-coded defaults or log leakage.
- Provide safe pickle and encrypted JSON helpers for callers that must handle
  serialized or sensitive blobs.

## Module Map

- `egress.py` evaluates URL allow/deny/private-IP/port policies and tenant
  webhook rules.
- `url_validation.py` exposes endpoint-friendly safe-URL assertions.
- `middleware.py` adds response hardening headers.
- `request_id_middleware.py` sanitizes or creates `X-Request-ID`.
- `setup_access_guard.py` and `setup_csp.py` enforce Setup UI access/CSP policy.
- `secret_manager.py` resolves JWT, OAuth, and single-user API-key secrets.
- `crypto.py` encrypts/decrypts JSON blobs for Jobs and related persistence.
- `safe_pickle.py` restricts pickle loading to approved classes.

## How It Connects

- `app/main.py` installs security middlewares during normal startup.
- Web scraping, Watchlists, WebSearch, Workflows, Text2SQL, and third-party
  providers should call egress helpers before outbound work.
- AuthNZ and setup flows read secret and setup guard behavior from this module.

## Extension Points

- Add outbound policy knobs in `egress.py` and cover global, workflow, and
  tenant-specific behavior in tests.
- Add middleware behavior with path-specific tests so `/setup`, `/docs`, and API
  routes keep their intended CSP/header differences.
- Add secret types through `secret_manager.py` with explicit source precedence
  and validation rules.

## Testing

- Egress: `tests/Security/test_egress.py`,
  `tests/Security/test_egress_global_env.py`, and
  `tests/Security/test_websearch_egress_guard.py`.
- Headers and request IDs: `tests/Security/test_security_headers_middleware.py`
  and `tests/Security/test_request_id_middleware.py`.
- Setup guards/CSP: `tests/Security/test_setup_access_guard.py` and
  `tests/Security/test_setup_csp_eval_policy.py`.
- Crypto/serialization: `tests/Security/test_crypto.py` and
  `tests/Security/test_zip_safe_extract.py`.

## Gotchas

- Do not add per-feature URL validators. Central policy prevents inconsistent
  SSRF behavior.
- HSTS should be coordinated with proxies/ingress; middleware respects HTTPS and
  `X-Forwarded-Proto`.
- Secret redaction is not a substitute for avoiding secret logging.
