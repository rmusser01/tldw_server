# Security Boundaries Specialist Review

## Scope

- Baseline: `origin/dev` at `669092178b0ba0fa1e840a37250b0deb55acd5a3`
- Report owner: Security boundaries
- In scope: auth/tenant isolation, file/path handling, SSRF/network egress, command execution, sandbox/tool execution, secret handling, admin/debug surfaces, and security-relevant domain findings.
- Out of scope: remediation implementation and new security architecture.
- Review mode: report-only specialist pass using local static inspection and existing audit evidence. No production code, tests, configs, index files, Backlog task files, or other reports were edited.

## Findings Table

| ID | Evidence Tier | Evidence Strength | Severity | Confidence | Category | Title | Status | Validation Status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |

No new specialist-specific finding rows were added. The security-boundary issues found in this pass are already represented in `findings-index.json`; this report cross-links those IDs, records confirmations, and recommends follow-up where evidence is still missing.

## Index Mapping

No new `AUDIT-2026-06-27-SEC-NNN` IDs are requested from this pass.

Existing normalized findings confirmed or recommended for security-boundary follow-up:

- `AUDIT-2026-06-27-AUTH-001`
- `AUDIT-2026-06-27-AUTH-002`
- `AUDIT-2026-06-27-AUTH-003`
- `AUDIT-2026-06-27-MEDIA-001`
- `AUDIT-2026-06-27-MEDIA-002`
- `AUDIT-2026-06-27-MEDIA-003`
- `AUDIT-2026-06-27-MEDIA-004`
- `AUDIT-2026-06-27-CHAT-001`
- `AUDIT-2026-06-27-CHAT-002`
- `AUDIT-2026-06-27-INTEGRATIONS-001`
- `AUDIT-2026-06-27-INTEGRATIONS-002`
- `AUDIT-2026-06-27-INTEGRATIONS-003`
- `AUDIT-2026-06-27-MCP-001`
- `AUDIT-2026-06-27-MCP-002`
- `AUDIT-2026-06-27-WEBUI-002`
- `AUDIT-2026-06-27-OPS-002`
- `AUDIT-2026-06-27-OPS-005`
- `AUDIT-2026-06-27-OPS-006`

## Confirmed Issues

No new confirmed issue was added by this specialist pass.

Confirmed existing security-boundary findings:

- `AUDIT-2026-06-27-AUTH-001`: Confirmed. The impersonation endpoint advertises a 15 minute lifetime, but calls the normal access-token minting path without a lifetime override. Security impact is bounded to authenticated admin impersonation, but the issued authority can outlive the endpoint contract.
- `AUDIT-2026-06-27-AUTH-002`: Confirmed and security-significant. The endpoint creates `impersonated_by` and `impersonation` token claims, but the normal AuthNZ resolver builds downstream `AuthContext` as the target user and does not preserve impersonation actor metadata for durable downstream audit attribution.
- `AUDIT-2026-06-27-MEDIA-001`: Confirmed. Several processing-only media endpoints authenticate through user-scoped dependencies but do not declare the `MEDIA_CREATE`, `media.create` rate-limit, and daily API-call dependency bundle used by `/media/add` and `/process-videos`. This is an authorization boundary mismatch for expensive upload, parsing, transcription, and remote-input work.
- `AUDIT-2026-06-27-MEDIA-002`: Confirmed and high priority for tenant isolation. MediaWiki ingest does not take the request user or request-scoped media DB, and its core importer falls back to shared single-user Media DB and vector user identifiers. This is the clearest cross-tenant data-boundary issue in the reviewed findings.
- `AUDIT-2026-06-27-CHAT-002`: Confirmed. `rag_unified.py` logs raw RAG query text in main and advanced search paths while a safer hash/length pattern already exists in simple search. This is a secret and private-data handling issue at the log boundary.
- `AUDIT-2026-06-27-MCP-001`: Confirmed and related to `AUDIT-2026-06-27-CHAT-001`. Normal HTTP AuthNZ rejects scoped JWTs on routes without scope enforcement, while ACP and sandbox WebSocket helpers directly verify bearer JWTs and return user IDs without applying `scope`, endpoint, method, path, quota, or schedule restrictions. Session/run ownership checks reduce cross-user impact, but the scoped-token authority boundary is bypassed for owned ACP control channels, ACP SSH, sandbox streaming, and sandbox stdin.
- `AUDIT-2026-06-27-WEBUI-002`: Confirmed from the domain report evidence. The client uses query-token WebSocket auth for TTS streaming, while the backend default rejects that legacy mode because URLs leak through browser and log surfaces. This is primarily a contract break, with a security-relevant temptation to re-enable weaker query-token auth.
- `AUDIT-2026-06-27-OPS-006`: Confirmed. The Kubernetes sample Secret contains a concrete default database password and a literal `${POSTGRES_PASSWORD}` inside `DATABASE_URL`. This is an operations security issue because example material can be copied into live manifests and also produces a non-working DB auth boundary.

Positive boundary checks that were not promoted:

- Admin endpoints are mounted under a parent router with `RequireRole("admin")`.
- AuthNZ debug endpoints require single-user compatibility or `super_admin`/`owner` roles.
- Setup write endpoints use local setup access checks and setup-state checks before mutating configuration.
- Upload helpers strip path components, reject dangerous extensions, and sanitize stored filenames.
- Shared download helpers validate target-path containment, enforce outbound policy, and cap streamed size.
- MCP Unified WebSocket auth uses the stronger AuthNZ request projection path, unlike the ACP/sandbox helpers covered by `AUDIT-2026-06-27-MCP-001`.

## Likely Risks

No new likely-risk finding was added by this specialist pass.

Likely risks or follow-up items confirmed from existing normalized findings:

- `AUDIT-2026-06-27-CHAT-001`: Confirmed as a systemic endpoint-enforcement consistency risk. The main RAG search path declares `TokenScopeGuard` with an endpoint ID and call counting, but adjacent RAG, character completion, chat document generation, and embeddings paths are inconsistent. This should be remediated with `AUDIT-2026-06-27-MCP-001` because both findings point to missing shared token-scope enforcement on alternate resource-spending routes.
- `AUDIT-2026-06-27-INTEGRATIONS-001`: Confirmed. Workflow research adapters use direct `httpx.AsyncClient()` calls for user or workflow-derived research URLs and provider lookups instead of the centralized HTTP client and egress policy. This remains a likely SSRF/network-egress risk pending runtime reproduction.
- `AUDIT-2026-06-27-INTEGRATIONS-002`: Confirmed. Tokenizer/counting requests use direct `requests.post()` through `_http_post()` and can carry API keys to configured provider URLs without the central egress/proxy defaults.
- `AUDIT-2026-06-27-AUTH-003`: Confirmed as source-linked but not runtime reproduced. In PostgreSQL AuthNZ mode, the impersonation endpoint uses raw acquired connections with SQLite-style parameter markers, bypassing the pool helpers that normalize parameter style. This is reliability first, but it affects a privileged admin security surface.
- `AUDIT-2026-06-27-MEDIA-003`: Confirmed as source-linked. Permanent original-file storage happens before MediaFiles row insertion, and the failure path does not delete the stored file even though the filesystem backend supports deletion. This can leave untracked user files outside normal metadata cleanup.
- `AUDIT-2026-06-27-MCP-002`: Confirmed. ACP reconnect replay can leak broadcaster tasks/subscriptions. This is reliability first, but remotely reachable authenticated resource accumulation sits on a security-sensitive agent control channel.
- `AUDIT-2026-06-27-OPS-002`: Confirmed. Published worker images run without a non-root user directive and retain build/network tooling in final layers, unlike the production app image. If an ingestion, worker, model, or queue path is compromised, the container blast radius is larger than necessary.

## Improvement Opportunities

No new improvement-opportunity finding was added by this specialist pass.

Existing opportunities with security-boundary relevance:

- `AUDIT-2026-06-27-INTEGRATIONS-003`: Confirmed. Weather provider calls use raw `httpx.Client` for an API-key-bearing fixed OpenWeather request. SSRF risk is narrow because the URL is fixed, but proxy and central-policy consistency should be improved.
- `AUDIT-2026-06-27-MEDIA-004`: Confirmed as a test gap. The no-op oversized-audio header test weakens coverage for a file-size boundary that otherwise exists in the implementation.
- `AUDIT-2026-06-27-OPS-005`: Security-adjacent but not specialist-escalated. Missing dependency update automation for nested package roots increases exposure window for separately shipped JS, Python, and Go surfaces.
- MCP external process policy note from the MCP domain report remains worth tracking: standalone `apps/mcp-unified` has a stdio process policy, while the in-server compatibility adapter uses `ACPStdioClient` for configured external servers. I did not find evidence that ordinary users can write those command definitions, so this remains an architecture watch item rather than a new finding.

## Coverage And Evidence

### Files Inspected

Audit artifacts:

- `Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/inventory.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/authnz-admin.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/chat-rag-llm.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/ci-deployment-operations-release.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/db-migrations-data-durability.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/integrations-providers.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/jobs-scheduler-workflows.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/mcp-sandbox-agent-protocol.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/media-ingestion-storage.md`
- `Docs/superpowers/reviews/2026-06-27-repo-audit/domains/webui-extension-api-contracts.md`
- Existing evidence summaries under `Docs/superpowers/reviews/2026-06-27-repo-audit/evidence/`, including endpoint, backend-test, dependency-manifest, DB-migration, CI/deploy, frontend-client, integration-provider, WebUI-contract, and Bandit summaries.

Security-relevant source paths sampled or line-checked:

- `tldw_Server_API/app/api/v1/API_Deps/auth_deps.py`
- `tldw_Server_API/app/api/v1/API_Deps/setup_deps.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/__init__.py`
- `tldw_Server_API/app/api/v1/endpoints/admin/admin_impersonation.py`
- `tldw_Server_API/app/api/v1/endpoints/authnz_debug.py`
- `tldw_Server_API/app/api/v1/endpoints/setup.py`
- `tldw_Server_API/app/core/AuthNZ/User_DB_Handling.py`
- `tldw_Server_API/app/core/AuthNZ/jwt_service.py`
- `tldw_Server_API/app/api/v1/endpoints/media/add.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_audios.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_documents.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_pdfs.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_ebooks.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_code.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_emails.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_mediawiki.py`
- `tldw_Server_API/app/api/v1/endpoints/media/process_videos.py`
- `tldw_Server_API/app/api/v1/endpoints/media/file.py`
- `tldw_Server_API/app/core/Ingestion_Media_Processing/input_sourcing.py`
- `tldw_Server_API/app/core/Ingestion_Media_Processing/Download_Utils.py`
- `tldw_Server_API/app/core/Ingestion_Media_Processing/persistence.py`
- `tldw_Server_API/app/core/Ingestion_Media_Processing/MediaWiki/Media_Wiki.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/defaults.py`
- `tldw_Server_API/app/core/DB_Management/media_db/runtime/factory.py`
- `tldw_Server_API/app/core/Storage/filesystem_storage.py`
- `tldw_Server_API/app/api/v1/endpoints/rag_unified.py`
- `tldw_Server_API/app/api/v1/endpoints/chat_documents.py`
- `tldw_Server_API/app/api/v1/endpoints/agent_client_protocol.py`
- `tldw_Server_API/app/api/v1/endpoints/sandbox.py`
- `tldw_Server_API/app/core/MCP_unified/server.py`
- `tldw_Server_API/app/core/MCP_unified/adapters/tldw_runtime.py`
- `tldw_Server_API/app/core/MCP_unified/protocol.py`
- `tldw_Server_API/app/core/http_client.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/search.py`
- `tldw_Server_API/app/core/Workflows/adapters/research/bibliography.py`
- `tldw_Server_API/app/core/LLM_Calls/tokenizer_resolver.py`
- `tldw_Server_API/app/core/Integrations/weather_providers.py`
- `Dockerfiles/Dockerfile.worker`
- `Dockerfiles/Dockerfile.audio_gpu_worker`
- `Dockerfiles/Dockerfile.prod`
- `Helper_Scripts/Samples/Kubernetes/app-secret.yaml`

### Tests Or Scans Run

No runtime application tests were run for this specialist pass. The pass was static/report-only and reused existing domain test evidence.

Local inspection commands run included:

- `git status --short`
- `find Docs/superpowers/reviews/2026-06-27-repo-audit -maxdepth 3 -type f | sort`
- `python3 -m json.tool Docs/superpowers/reviews/2026-06-27-repo-audit/findings-index.json`
- `python3` summary of normalized finding IDs, severities, categories, and affected paths.
- `sed -n ...` over all domain reports and audit inventory.
- Targeted `rg` searches for route dependencies, permission guards, token-scope enforcement, WebSocket auth, egress/client calls, and secret-like deployment content.
- Targeted `nl -ba ... | sed -n ...` reads over the source files listed above.

### Blocked Or Unverified Areas

- No network access, dependency installation, Docker, service startup, browser automation, or live provider/API calls were used.
- No new Bandit run was performed because no production/source files were changed; the existing audit Bandit summary was read.
- PostgreSQL behavior for `AUDIT-2026-06-27-AUTH-003` was not runtime reproduced.
- Scoped JWT/API-key bypass behavior for `AUDIT-2026-06-27-CHAT-001` and `AUDIT-2026-06-27-MCP-001` was statically confirmed from source, but broader end-to-end tests should still reproduce exact behavior under supported AuthNZ backends.
- Multi-user MediaWiki cross-tenant visibility for `AUDIT-2026-06-27-MEDIA-002` was statically confirmed from request/user and DB/vector construction paths, but not dynamically reproduced.
- Full source coverage was not exhaustive beyond the audit artifacts and security-relevant source paths named or implicated by existing findings.
- No Backlog tasks were created or updated, following the coordinator instruction for this specialist pass.

### Evidence Notes

- The strongest cross-domain theme is inconsistent application of shared security boundary helpers on alternate routes. HTTP scoped-token enforcement, media write permissions, request-scoped tenant DB/vector handles, and centralized egress helpers exist, but selected adjacent routes bypass or omit them.
- `AUDIT-2026-06-27-CHAT-001` and `AUDIT-2026-06-27-MCP-001` should be handled as one AuthNZ boundary remediation theme: introduce a single way to declare endpoint IDs, action counters, path/method constraints, and WebSocket handshake scope checks for all resource-spending HTTP and WebSocket routes.
- `AUDIT-2026-06-27-MEDIA-001` and `AUDIT-2026-06-27-MEDIA-002` should be handled as one media boundary remediation theme: make processing/ingest route dependency bundles explicit, and thread request-scoped user, media DB, and vector identity into every persistence-capable importer.
- `AUDIT-2026-06-27-INTEGRATIONS-001`, `AUDIT-2026-06-27-INTEGRATIONS-002`, and `AUDIT-2026-06-27-INTEGRATIONS-003` should be handled as one outbound-network remediation theme: direct `httpx`/`requests` use is acceptable only when equivalent egress, proxy, redirect, timeout, logging, and secret-handling policy is deliberately applied and tested.
- No existing normalized finding was refuted in this specialist pass.
