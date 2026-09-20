# Candidate backend OpenAPI drift

Tracking TASK-13263. backend-required run35538062528 failed the OpenAPI drift gate. The earlier mypy NumPy-stub Python-target diagnostic is advisory (continue-on-error); it was not the failing step.

Reproduced the exact CI fingerprint021fb14cdce07f28cc623e2f1f2efa8d7d4463e914913bf019f4bbec065a8fa4 using FastAPI0.136.3/Pydantic2.13.5 and pydantic-settings2.15.0. The shared local venv has Pydantic2.11.7, which generates a different schema; CI versions were installed into an isolated temporary directory, leaving the shared environment unchanged. The local checkout's vendored tldw_profile_core source was explicitly included.

The complete drift is the /api/v1/health/ready GET description changed by the readiness compatibility repair. Replacing only that description with its old text in the exported JSON yields exactly the checked-in old digest d98e55a945e85c81b84c29eb0ab88321300a99dd9dbb5f30a9cc00dec1452887. There are no other schema changes:2097 paths,3207 schemas.

Regenerated the canonical fingerprint and ignored full OpenAPI/type artifacts using export_openapi_schema.py and installed openapi-typescript7.13.0. Fresh --check passes and full frontend tsc --noEmit passes. The committed change is only the fingerprint SHA; generated source artifacts remain ignored per existing build policy. This changes a protected source file, so the candidate source manifest must be refreshed after this commit.

## Agentic-review refresh

The setup privacy fix adds only the GET `/api/v1/setup/first-run/state` description. A recursive comparison with the prior generated schema identifies this as the entire additional OpenAPI delta; path/schema counts remain 2097/3207. The Request-only OSCE limiter wrapper adds no internal query parameter. Regenerated fingerprint/types with the same isolated CI dependencies; fingerprint check and full frontend TypeScript pass again. Current fingerprint: `4d8718a9387567b3278a9034acb13e39bf75375714632e90c49ffc3dbda58e22`.
