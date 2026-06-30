# Roadmap Phase Closeout Tracker

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans or superpowers:finishing-a-development-branch for each branch closeout. Verify before squash/push. Keep this checklist updated as phases move.

**Goal:** Track remaining Phase 2 and Phase 3 roadmap closeout work and keep PR update order explicit.

**Architecture:** Treat each phase branch as an independent closeout unit. Prefer clean redux branches first, then the large dirty Phase 3.3 branch, and defer Phase 3.1/3.2 until Phase 2 work is merged or PR-stable.

**Tech Stack:** Git worktrees, GitHub PRs, pytest, Bandit

---

## Stage 1: Phase 2 PR Closeout
**Goal**: Bring open Phase 2 PR branches up to date with local verified work.
**Success Criteria**: Each branch is clean, freshly verified, squashed if needed, pushed to its existing PR, and the PR still targets `dev`.
**Tests**: Phase-specific focused pytest suites plus touched-scope Bandit.
**Status**: Complete

- [x] Phase 2.3 ChaChaNotes: squashed and pushed to PR #1115.
- [x] Phase 2.2 Router groups: verified, squashed, and pushed to PR #1122.
- [x] Phase 2.4 Config sections: verified, squashed, and pushed to PR #1121.
- [x] Phase 2.1 Lifespan extraction: verified, fully squashed to one PR commit, pushed to PR #1123, and refreshed against current `dev` as single head `ac5726771730` after clearing the 2026-04-25 `main.py` conflict. Local verification: focused startup/lifecycle suite `62 passed`, Bandit `0` findings. GitHub checks are queued/in progress (`mergeStateStatus=UNSTABLE`).
- [x] Phase 2.5 Unified errors: verified, fully squashed to one PR commit, pushed to PR #1120, and refreshed against current `dev` as single head `6779bd72105a` after clearing the 2026-04-25 `http_errors.py` / `test_api_v1_utils.py` conflicts. Local verification: utility suite `26 passed`, Bandit `0` findings, raw logger interpolation scan clean. GitHub checks are queued/in progress (`mergeStateStatus=UNSTABLE`).

## Stage 2: Phase 3.3 Draft PR Stabilization
**Goal**: Convert the dirty Phase 3.3 error-handler adoption worktree into a reviewable PR update.
**Success Criteria**: Dirty worktree is understood, focused tests and Bandit are rerun, changes are committed/squashed, and draft PR #1125 is updated or explicitly left draft with blockers documented.
**Tests**: Tranche-specific endpoint suites listed in `Docs/superpowers/plans/2026-04-21-phase3-3-error-handler-adoption.md`, plus source-scope Bandit.
**Status**: In Progress

- [x] Inventory dirty files and untracked tests in `phase3.3-error-handler-adoption`.
- [x] Identify the latest completed tranche in the plan file.
- [x] Run a minimal focused verification set for the dirty tranche.
- [x] Commit/squash/push to PR #1125 if verification passes.
- [x] Keep PR draft if blockers remain; document blockers in the plan.
- [x] Fixed the Collections output artifact row-hydration blocker, stabilized adjacent output tests, and pushed commit `e7664b5d4` to PR #1125.
- [x] Sanitized embeddings and health generic failure details and pushed commit `7abfad8b5` to PR #1125.
- [x] Sanitized writing tokenizer/wordcloud and data-table generation failure details and pushed commit `66b5af7d0` to PR #1125.
- [x] Sanitized embedding model warmup/download failure details and pushed commit `04f7d7ae7` to PR #1125.
- [x] Sanitized embedding provider-batch failure details and pushed commit `143a0830c` to PR #1125.
- [x] Sanitized embeddings MLX adapter runtime-error details and pushed commit `9b75a4136` to PR #1125.
- [x] Sanitized Cohere/Google embedding provider HTTP error bodies and pushed commit `8160dd228` to PR #1125.
- [x] Sanitized character chat provider failure details and pushed commit `8e292141f` to PR #1125.
- [x] Sanitized MCP health status and XML processing failure details and pushed commit `03ca8f938` to PR #1125.
- [x] Sanitized legacy web scraping fallback failure details and pushed commit `fda5e1e35` to PR #1125.
- [x] Sanitized enhanced web scraping fallback failure details and pushed commit `a89dd6fe0` to PR #1125.
- [x] Sanitized ChaCha shutdown-abort and Claims service fallback details and pushed commit `707e224ef` to PR #1125.
- [x] Sanitized admin bundle restore/rollback failure details and pushed commit `02bccfac6` to PR #1125.
- [x] Sanitized Qwen3 tokenizer missing-package failure detail and pushed commit `be509e26e` to PR #1125.
- [x] Sanitized media reprocess embedding failure state and pushed commit `a4bdfeab2` to PR #1125.
- [x] Sanitized DSR erasure handler failure details and pushed commit `f5a19255b` to PR #1125.
- [x] Sanitized process-code read failure details and pushed commit `923af515b` to PR #1125.
- [x] Sanitized process-emails processor failure details and pushed commit `60e432b67` to PR #1125.
- [x] Sanitized process-ebooks worker failure details and pushed commit `6e56eb2aa` to PR #1125.
- [x] Sanitized process-documents worker failure details and pushed commit `683456a1b` to PR #1125.
- [x] Sanitized process-pdfs processor failure details in local commit `88287aaa1` (not pushed yet).
- [x] Sanitized speech-chat TTS configuration fallback details in local commit `a62373368` (not pushed yet).
- [x] Sanitized process-code URL download failure details in local commit `d4a3aefab` (not pushed yet).
- [x] Sanitized AuthNZ user validation fallback details in local commit `db73b78d2` (not pushed yet).
- [x] Sanitized process-ebooks URL download failure details in local commit `b0f347a87` (not pushed yet).
- [x] Sanitized document/PDF/ebook preparation fallback batch details in local commit `a14173ccc` (not pushed yet).
- [x] Sanitized audio batch execution and ingest-job upload staging fallback details in local commit `eaffe9f33` (not pushed yet).
- [x] Sanitized video batch execution and upload write-failure fallback details in local commit `74165693d` (not pushed yet).
- [x] Sanitized upload validation fallback details in local commit `5689d33f9` (not pushed yet).
- [x] Sanitized outer upload-processing fallback details in local commit `045a75d57` (not pushed yet).
- [x] Sanitized audio item processing fallback details in local commit `6664d49d9` (not pushed yet).
- [x] Sanitized video item processing fallback details in local commit `78aad47aa` (not pushed yet).
- [x] Sanitized web scraping persistence fallback details in local commit `4183a7c2a` (not pushed yet).
- [x] Sanitized XML processing and evaluation cleanup fallback details in local commit `708920672` (not pushed yet).
- [x] Sanitized audio setup, fatal batch, and podcast fallback details in local commit `e5545159a` (not pushed yet).
- [x] Sanitized plaintext document processing fallback details in local commit `12355314b` (not pushed yet).
- [x] Sanitized video and EPUB processing fallback details in local commit `713a5a545` (not pushed yet).
- [x] Sanitized non-EPUB book processing and legacy text-ingest fallback details in local commit `640c9d38a` (not pushed yet).
- [x] Sanitized email parse and chunking fallback details in local commit `b6c16a4b1` (not pushed yet).
- [x] Sanitized OCR preload failure details in local commit `b1b53766d` (not pushed yet).
- [x] Sanitized legacy arXiv ingest helper failure details in local commit `e4342d1bc` (not pushed yet).
- [x] Sanitized vector-store batch failure status details in local commit `98e994515` (not pushed yet).
- [x] Sanitized media embedding generation failure details in local commit `c5c7bff6d` (not pushed yet).
- [x] Sanitized batch evaluation item failure details in local commit `9c281c32a` (not pushed yet).
- [x] Sanitized audio streaming model initialization/fallback error frame details in local commit `5ca63ffbf` (not pushed yet).
- [x] Sanitized audio streaming runtime provider/internal error frame details in local commit `b302e1e2f` (not pushed yet).
- [x] Sanitized audio streaming diarization/live-insights warning details in local commit `2b4197f7a` (not pushed yet).
- [x] Sanitized legacy Parakeet audio chunk error frame details in local commit `9334b2752` (not pushed yet).
- [x] Sanitized Parakeet core websocket outer error frame details in local commit `90145d951` (not pushed yet).
- [x] Sanitized live-insights LLM failure error frame details in local commit `361f1c5df` (not pushed yet).
- [x] Sanitized audio chat realtime TTS overlap warning frame details in local commit `6d72ff5c0` (not pushed yet).
- [x] Sanitized MediaWiki import stream failure details in local commit `7c216e25a` (not pushed yet).
- [x] Sanitized MediaWiki per-page item processing failure details in local commit `da2cf610e` (not pushed yet).
- [x] Sanitized NeMo Canary/Parakeet runtime transcription failure details in local commit `7b33dfc69` (not pushed yet).
- [x] Sanitized Parakeet ONNX loader failure details in local commit `f50164139` (not pushed yet).
- [x] Sanitized external transcription provider operational failure details in local commit `38eb9b646` (not pushed yet).
- [x] Sanitized Parakeet streaming generator failure details in local commit `0f0552421` (not pushed yet).
- [x] Sanitized Parakeet MLX streaming generator failure details in local commit `e10f9e105` (not pushed yet).
- [x] Sanitized Prompt Studio evaluation-job persisted failure details in local commit `ca0ed8206` (not pushed yet).
- [x] Sanitized XML malformed-parse response details in local commit `ab13f912a` (not pushed yet).
- [x] Sanitized chat audit error metadata details in local commit `7eac397a1` (not pushed yet).
- [x] Sanitized fatal websearch provider failure details in local commit `bb592dad1` (not pushed yet).
- [x] Sanitized Parakeet MLX non-streaming transcription failure details in local commit `bb592dad1` (not pushed yet).
- [x] Sanitized `transcribe_audio` provider-wrapper fallback details in local commit `c5d7cc46a` (not pushed yet).
- [x] Sanitized Parakeet ONNX audio-load/runtime transcription failure details in local commit `5fabfb959` (not pushed yet).
- [x] Sanitized Parakeet MLX missing-file/dependency failure details in local commit `4b4a412ab` (not pushed yet).
- [x] Sanitized org invite redemption membership failure details in local commit `a4af7fba1` (not pushed yet).
- [x] Sanitized Tavily websearch fetch failure details in local commit `c287c1769` (not pushed yet).
- [x] Sanitized Searx websearch fetch failure details in local commit `4ddd7fbf3` (not pushed yet).
- [x] Sanitized external transcription provider config/API failure details in local commit `56f49b775` (not pushed yet).
- [x] Sanitized external transcription provider self-test failure details in local commit `7dccf5fa3` (not pushed yet).
- [x] Sanitized XML processing service parse failure details in local commit `0e4071f52` (not pushed yet).
- [x] Sanitized Searx URL configuration failure details in local commit `6bf337bf9` (not pushed yet).
- [x] Sanitized generic `perform_websearch` provider failure details in local commit `a87ab31ea` (not pushed yet).
- [x] Sanitized RAG service health-check failure details in local commit `de340a3ba` (not pushed yet).
- [x] Sanitized RAG embedding-service health-check failure details in local commit `734e2c65a` (not pushed yet).
- [x] Sanitized embeddings service health-check failure details in local commit `941034e88` (not pushed yet).
- [x] Sanitized legacy WebSearch fallback failure details in local commit `9fe5180a7` (not pushed yet).
- [x] Sanitized MCP module health-message failure details in local commit `5518c7802` (not pushed yet).
- [x] Sanitized media DB legacy backup failure details in local commit `317e5163a` (not pushed yet).
- [x] Sanitized PostgreSQL backup runtime fallback details locally (`8bfe8d901`; not pushed yet).
- [x] Sanitized Ollama stop-server failure details in local commit `4b1808913` (not pushed yet).
- [x] Sanitized admin diagnostic fallback failure details in local commit `16952c074` (not pushed yet).
- [x] Sanitized moderation override persistence failure details in local commit `9d1abf5c0` (not pushed yet).
- [x] Sanitized scraper analyzer defensive fallback details in local commit `8d7bdc49c` (not pushed yet).
- [x] Sanitized Semantic Scholar fetch fallback details in local commit `a88f62b0a` (not pushed yet).
- [x] Sanitized arXiv helper fallback details in local commit `15bcce3aa` (not pushed yet).
- [x] Sanitized Crossref provider fallback details in local commit `0d6b04c7a` (not pushed yet).
- [x] Sanitized Unpaywall provider fallback details in local commit `58d4da664` (not pushed yet).
- [x] Sanitized IACR provider fallback details locally (`d1b84ce8b`; not pushed yet).
- [x] Sanitized OpenAlex provider fallback details locally (`e94d20ee1`; not pushed yet).
- [x] Sanitized Scopus provider fallback details locally (`09be1f665`; not pushed yet).
- [x] Sanitized Springer provider fallback details locally (`9883a82a6`; not pushed yet).
- [x] Sanitized EarthRxiv provider fallback details locally (`c908693cd`; not pushed yet).
- [x] Sanitized RePEc/CitEc provider fallback details locally (`e5242c0c5`; not pushed yet).
- [x] Sanitized HAL provider fallback details locally (`47a13b3c9`; not pushed yet).
- [x] Sanitized IEEE Xplore provider fallback details locally (`96e992ac6`; not pushed yet).
- [x] Sanitized viXra provider fallback details locally (`ef0382538`; not pushed yet).
- [x] Sanitized PubMed provider fallback details locally (`ebf458c00`; not pushed yet).
- [x] Sanitized Zenodo provider fallback details locally (`3d533296d`; not pushed yet).
- [x] Sanitized PMC OA provider fallback details locally (`95e78af4d`; not pushed yet).
- [x] Sanitized Figshare provider fallback details locally (`dab9c111e`; not pushed yet).
- [x] Sanitized OSF provider fallback details locally (`79dd53dd8`; not pushed yet).
- [x] Sanitized ChemRxiv provider fallback details locally (`79dd53dd8`; not pushed yet).
- [x] Sanitized PMC OAI provider fallback details locally (`318a7429f`; not pushed yet).
- [x] Sanitized BioRxiv provider fallback details locally (`0eafdb84d`; not pushed yet).
- [x] Cleaned final arXiv provider residual scan matches locally (`9da93b44a`; not pushed yet).
- [x] Sanitized Workflows webhook DLQ delivery error details locally (`69ce9dd92`; not pushed yet).
- [x] Sanitized admin registration settings error details locally (`ba36afb46`; not pushed yet).
- [x] Sanitized admin LLM provider error details locally (`6a719606f`; not pushed yet).
- [x] Sanitized admin BYOK service error details locally (`f768e271f`; not pushed yet).
- [x] Sanitized admin user creation error details locally (`5f467eb60`; not pushed yet).
- [x] Sanitized meetings webhook DLQ delivery error details locally (`1964a19c8`; not pushed yet).
- [x] Sanitized scoped shared BYOK route error details locally (`58062243f`; not pushed yet).
- [x] Sanitized workflow webhook adapter error payloads locally (`207da8300`; not pushed yet).
- [x] Sanitized workflow storage adapter error payloads locally (`e38e3b58b`; S3 selection verified; full broad adapter file still has unrelated ACP duplicate-session failures; not pushed yet).
- [x] Sanitized user BYOK/OAuth route error details locally (`4c6be3cd9`; not pushed yet).
- [x] Sanitized workflow email adapter error payloads locally (`c40fbeb30`; email selection verified; full broad adapter file still has unrelated ACP duplicate-session failures; not pushed yet).
- [x] Sanitized workflow GitHub adapter error payloads locally (`fb982cd21`; GitHub selection verified; not pushed yet).
- [x] Sanitized workflow Kanban adapter error payloads locally (`2d94e7250`; Kanban selection verified; not pushed yet).
- [x] Sanitized workflow STT adapter error payloads locally (`a54c4fb27`; STT selection verified; not pushed yet).
- [x] Sanitized workflow Chatbooks/Character Chat adapter error payloads locally (`c54dea355`; Chatbooks/Character selection verified; not pushed yet).
- [x] Sanitized workflow TTS adapter error payloads locally (`7baeaeb69`; TTS selection verified; not pushed yet).
- [x] Sanitized workflow document media adapter error payloads locally (`c1c983f80`; full media adapter file verified; not pushed yet).
- [x] Sanitized workflow audio processing adapter error payloads locally (`3172cd68e`; full audio adapter file verified; not pushed yet).
- [x] Sanitized workflow utility adapter error payloads locally (`cef544b36`; full utility adapter file verified; not pushed yet).
- [x] Sanitized workflow text conversion adapter error payloads locally (`7ca4ffe03`; full text adapter file verified; not pushed yet).
- [x] Sanitized workflow text transform adapter error payloads locally (`2e901f155`; full text adapter file verified; not pushed yet).
- [x] Sanitized workflow text NLP adapter error payloads locally (`d0ba8c038`; full text adapter file verified; not pushed yet).
- [x] Sanitized workflow media ingest adapter error payloads locally (`cbabf04d1`; full media adapter file verified; not pushed yet).
- [x] Sanitized workflow video processing/subtitle adapter error payloads locally (`859941019`; full video adapter file verified; not pushed yet).
- [x] Sanitized workflow audio diarize adapter error payloads locally (`4fd6c276d`; full audio adapter file verified; not pushed yet).
- [x] Sanitized workflow flashcard/quiz content generation error payloads locally (`99c5bb0bb`; full content adapter file verified; not pushed yet).
- [x] Sanitized workflow content generation fallback error payloads locally (`62f1c479f`; full content adapter file verified; not pushed yet).
- [x] Sanitized workflow image/summarize/citations/rerank content adapter error payloads locally (`3b810115c`; full content adapter file verified; not pushed yet).
- [x] Sanitized workflow audio briefing content adapter error payloads locally (`505dd63fe`; full content adapter file verified; not pushed yet).
- [x] Sanitized workflow notes/prompts/chunking knowledge adapter error payloads locally (`5ba97d14d`; focused knowledge regressions verified, with two unrelated full-file failures still tracked; not pushed yet).
- [x] Sanitized workflow cache/retry/checkpoint control-state error payloads locally (`737faf937`; full control adapter file verified; not pushed yet).
- [x] Sanitized workflow query rewrite/expand/HyDE/semantic-cache RAG query error payloads locally (`75606492e`; focused regressions verified, with four unrelated full-file RAG failures still tracked; not pushed yet).
- [x] Sanitized workflow call control-orchestration error payloads locally (`291d77891`; full control adapter file verified; not pushed yet).
- [x] Sanitized workflow prompt fallback and parallel substep control-flow error payloads locally (`8f867b63f`; full control adapter file verified; not pushed yet).
- [x] Sanitized workflow LLM tool/compare/critique/moderation error payloads locally (`b18d8be8b`; full LLM adapter file verified; not pushed yet).
- [x] Sanitized workflow LLM template-render and stream-dispatch debug fallbacks locally (`a1d6f4f2a`; full LLM adapter file verified; not pushed yet).
- [x] Sanitized workflow evaluations service error payloads locally (`6d0eb45e3`; full evaluation adapter file verified; not pushed yet).
- [x] Sanitized workflow S3 upload file-read error payloads locally (`054f84787`; S3 upload/download subset verified, with six unrelated full integration ACP session DB failures still tracked; not pushed yet).
- [x] Sanitized workflow DOI/reference/literature-review bibliography error payloads locally (`96235cec7`; full research adapter file verified; not pushed yet).
- [x] Sanitized workflow academic search adapter error payloads locally (`e838cf114`; full research adapter file verified; not pushed yet).
- [x] Sanitized workflow multi-voice TTS warning logs locally (`4cd7875a0`; full audio adapter file verified; not pushed yet).
- [x] Sanitized workflow RAG web/RSS search error payloads locally (`4fcf2b131`; web/RSS RAG adapter selection verified; not pushed yet).
- [x] Sanitized workflow collections/claims/voice-intent knowledge error payloads locally (`955b60797`; focused regressions verified; affected adapter selection still has two unrelated stale failures; not pushed yet).
- [x] Sanitized workflow regex invalid-pattern payloads locally (`c28f1c1cd`; regex extraction selection verified; not pushed yet).
- [x] Sanitized RAG observability memory-usage error payload locally (`c37e51987`; focused RAG observability test verified; broader observability residuals remain; not pushed yet).
- [x] Sanitized workflow podcast RSS write-failure logs locally (`5e6bf9a62`; podcast RSS selection verified; not pushed yet).
- [x] Sanitized workflow path-helper/MCP/moderation/query debug logs locally (`ce8ad5a64`; path-security and MCP policy files plus moderation/query selections verified; not pushed yet).
- [x] Sanitized remaining RAG observability debug/span error details locally (`82d08609c`; RAG observability sanitizer file verified; not pushed yet).
- [x] Sanitized self-monitoring pattern/backend fallback logs and cooldown deactivation payload locally (`634ad2788`; full self-monitoring test file verified; not pushed yet).
- [x] Sanitized supervised-policy pattern/dispatch/proxy fallback logs locally (`2b48f4ac4`; full supervised policy test file verified; not pushed yet).
- [x] Fixed stale workflow knowledge adapter blockers locally (`079dafccd`; full knowledge adapter test file verified; not pushed yet).
- [x] Sanitized semantic matcher fallback logs locally (`cfc3698dc`; full semantic matcher test file verified; not pushed yet).
- [x] Fixed stale workflow RAG adapter blockers locally (`0b22b5e81`; full RAG adapter test file verified; not pushed yet).
- [x] Sanitized governance schedule fallback logs locally (`f14f40858`; full governance utils test file verified; not pushed yet).
- [x] Sanitized governance import fallback logs and stabilized ACP integration adapter tests locally (`530304ea5`; full governance IO and integration adapter test files verified; not pushed yet).
- [x] Sanitized Family Wizard materialization failure persistence/logs locally (`d9f9fc6d8`; full family wizard materialization test file verified; not pushed yet).
- [x] Sanitized moderation runtime override persistence warning locally (`d4bfb2944`; full runtime override test file verified; not pushed yet).
- [x] Sanitized moderation runtime load/save and blocklist warning logs locally (`4aa78884a`; runtime override and blocklist parser test files verified; not pushed yet).
- [x] Sanitized moderation blocklist load/read/write and built-in PII fallback logs locally (`5217c81e3`; blocklist parser/runtime override tests verified; not pushed yet).
- [x] Sanitized moderation user-override load failure log locally (`156cc0265`; user override validation test file verified; not pushed yet).
- [x] Ran combined focused Phase 3.3 closeout sweep across recently stabilized Guardian/RAG/workflow files: `485 passed` locally (not pushed yet).
- [x] Re-ran the earlier strict raw-exception interpolation scan across endpoints/API deps/core/services: no matches; broader residual scans continue tranche-by-tranche.
- [x] Re-ran strict direct 5xx raw-exception interpolation scan across endpoints/API deps/core/services: `TOTAL=0`.
- [x] Rechecked the previously documented reading API blocker; `test_reading_save_returns_archive_requested_field` and the full current reading API file now pass locally.
- [x] Phase 3.3 commits squashed into single head `e697951d2c7d`, merged current `dev` to clear PR #1125 conflicts, reran the focused closeout suite (`729 passed`), source Bandit clean, test Bandit only `B101`, raw error-log scan clean, and force-pushed PR #1125. PR #1125 remains draft with GitHub checks queued/in progress (`mergeStateStatus=UNSTABLE`).
- [x] Triaged PR #1125 GitHub check failures on 2026-04-25: `build-sbom` fails because the SBOM workflow only accepts legacy requirements files while the branch has `pyproject.toml`; `Wizard Tests (Coverage Gate)` fails in `test_db_multi_user_postgres_connectivity` teardown with a SQLite `disk I/O error` while initializing `Collections_DB`; `Full Suite (Ubuntu / Python 3.11)` root artifact fails `test_mediawiki_security.py::test_import_rejects_outside_allowed_base` because the Phase 3.3 sanitized MediaWiki import message lost the safe `"outside allowed directory"` category; `UX Smoke Gate` fails because the mobile chat composer has unlabeled visible icon buttons and no visible role button named `"Attach image"`.
- [ ] Implement PR #1125 CI fixes after explicit approval: create a clean PR-head worktree to avoid existing local dirty Phase 3.3 files, fix the MediaWiki safe error category, update SBOM fallback for `pyproject.toml`, restore/label a visible mobile attach-image control, and reproduce/fix the wizard teardown isolation issue.

## Stage 3: Deferred Phase 3 Starts
**Goal**: Avoid starting broad API contract changes before prerequisites are stable.
**Success Criteria**: Phase 3.1/3.2 are only started after Phase 2 PRs are merged or explicitly accepted as stable bases.
**Tests**: New implementation plans required before code changes.
**Status**: Planning Started

- [x] Created Phase 3.1 implementation plan: `Docs/superpowers/plans/2026-04-25-phase3-1-standard-response-envelope-implementation-plan.md`.
- [x] Created Phase 3.2 implementation plan: `Docs/superpowers/plans/2026-04-25-phase3-2-pagination-standardization-implementation-plan.md`.
- [x] Created Phase 3.4 implementation plan: `Docs/superpowers/plans/2026-04-25-phase3-4-auth-dependency-standardization-implementation-plan.md`.
- [x] Created Phase 3.1 response-shape inventory: `Docs/superpowers/reviews/api-response-envelope/2026-04-25-response-shape-inventory.md`.
- [x] Created Phase 3.2 pagination inventory: `Docs/superpowers/reviews/api-pagination/2026-04-25-pagination-inventory.md`.
- [x] Created Phase 3.4 auth dependency inventory: `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-dependency-inventory.md`.
- [x] Created `skills` pilot-readiness map across response, pagination, auth, frontend callers, exemptions, and verification targets: `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-readiness.md`.
- [x] Created corrected pagination route-family catalogue and candidate frontend caller notes for `skills`, `slides`, and `data_tables`: `Docs/superpowers/reviews/api-pagination/2026-04-25-route-family-catalogue.md`.
- [x] Created Phase 3.4 auth risk scan for raw-user, manual-admin, legacy-dependency, and ordering-sensitive families: `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-risk-scan.md`.
- [x] Created Phase 3.1 mechanical response-envelope migration recipe: `Docs/superpowers/reviews/api-response-envelope/2026-04-25-envelope-migration-recipe.md`.
- [x] Created Phase 3.4 special-route and admin-check triage: `Docs/superpowers/reviews/auth-dependencies/2026-04-25-auth-special-route-and-admin-triage.md`.
- [x] Created Phase 3.1 response-envelope helper contract spec: `Docs/superpowers/reviews/api-response-envelope/2026-04-25-helper-contract-spec.md`.
- [x] Created Phase 3.2 pagination helper contract spec: `Docs/superpowers/reviews/api-pagination/2026-04-25-helper-contract-spec.md`.
- [x] Created Phase 3.4 auth dependency helper contract spec: `Docs/superpowers/reviews/auth-dependencies/2026-04-25-helper-contract-spec.md`.
- [x] Created `skills` Phase 3 execution packet with maintainer decisions, frontend coordination points, sequencing, and verification gates: `Docs/superpowers/reviews/phase3-pilots/2026-04-25-skills-pilot-execution-packet.md`.
- [x] Created Phase 3 readiness gate with refreshed read-only PR status and explicit start criteria: `Docs/superpowers/reviews/phase3-pilots/2026-04-25-phase3-readiness-gate.md`.
- [x] Created consolidated Phase 3 and Phase 4 remaining-work handoff: `Docs/superpowers/reviews/phase3-pilots/2026-04-25-phase3-phase4-remaining-work-handoff.md`.
- [x] Created maintainer decision checklist for Phase 3 and Phase 4 blockers: `Docs/superpowers/reviews/phase3-pilots/2026-04-25-maintainer-decision-checklist.md`.
- [ ] Confirm Phase 2 PRs and PR #1125 are merged, or explicitly accepted as stable bases, before starting broad Phase 3 contract code changes. Read-only GitHub check from this workspace succeeded on 2026-04-25: Phase 2 PRs #1115/#1120/#1121/#1122/#1123 are still open with unstable signals or cancelled full-suite jobs; PR #1125 is still open, draft, `mergeStateStatus=UNSTABLE`, with failed `onboarding-docs-gate`, `run-pre-commit`, `UX Smoke Gate`, and `Jobs (PostgreSQL)` checks, plus full-suite jobs still in progress. `build-sbom` is now green on PR #1125 but still failing on older Phase 2 PRs.
- [ ] Phase 3.1 Standard response envelope: response-shape inventory, migration recipe, helper contract spec, and `skills` pilot decision packet complete; maintainer acceptance, frontend implementation, and backend implementation pending.
- [ ] Phase 3.2 Standardize pagination: backend route/schema inventory, route-family catalogue, first-candidate frontend caller inventory, helper contract spec, and `skills` offset pilot packet complete; helper implementation pending.
- [ ] Phase 3.4 Auth dependency standardization: static dependency inventory, `skills` route map, raw-user scan, ordering-sensitive family scan, special-route classification, admin-check triage, helper contract spec, and `skills` auth pilot packet complete; implementation pending.
- [x] Decided the preliminary first Phase 3 pilot slice after inventories: `skills` is the lowest-risk cross-phase candidate. `slides` and `data_tables` remain follow-up candidates after helper contracts are proven.

## Stage 4: Phase 4 Parking Lot
**Goal**: Keep later roadmap work visible without letting it preempt current closeout.
**Success Criteria**: Phase 4 remains deferred until Phase 2/3 closeout is stable.
**Tests**: TBD per future implementation plans.
**Status**: Triage Started

- [ ] Keep Phase 4 deferred until Phase 2/3 closeout is stable.
- [x] Created Phase 4 parking-lot triage with priority order, first-pass DB/endpoint hotspot signals, and recommended first artifacts: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-phase4-triage.md`.
- [x] Created Phase 4.1 coverage ratchet baseline plan: `Docs/superpowers/plans/2026-04-25-phase4-1-coverage-ratchet-baseline-plan.md`.
- [x] Created Phase 4.1 coverage ratchet measurement packet: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-coverage-ratchet-measurement-packet.md`.
- [ ] Phase 4.1 Coverage ratchet to 25%: baseline plan and measurement packet exist; fresh clean-base coverage measurement and CI ratchet implementation pending.
- [x] Created Phase 4.2 deployment docs inventory: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-deployment-docs-inventory.md`.
- [x] Created Phase 4.2 deployment docs refresh plan with publishing flow and docs-gate commands: `Docs/superpowers/plans/2026-04-25-phase4-2-deployment-docs-refresh-plan.md`.
- [ ] Phase 4.2 Deployment docs: deployment-mode inventory and refresh plan exist; docs owner review, source-doc slice acceptance, and edits pending.
- [x] Created Phase 4.3 DB hotspot inventory with first-target recommendation: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-db-hotspot-inventory.md`.
- [x] Created Phase 4.3 `Prompts_DB.py` first-target decomposition plan: `Docs/superpowers/plans/2026-04-25-phase4-3-prompts-db-decomposition-plan.md`.
- [ ] Phase 4.3 Decompose remaining large DB files: hotspot inventory and draft `Prompts_DB.py` plan exist; first-file acceptance, baseline test run, and implementation pending.
- [x] Created Phase 4.4 endpoint hotspot inventory with first route-family recommendation: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-endpoint-hotspot-inventory.md`.
- [x] Created Phase 4.4 `storage.py` first-target route-family decomposition plan: `Docs/superpowers/plans/2026-04-25-phase4-4-storage-endpoint-decomposition-plan.md`.
- [ ] Phase 4.4 Decompose large endpoint files: endpoint inventory and draft `storage.py` route-family plan exist; target acceptance, OpenAPI baseline, tests, and implementation pending.
- [x] Created Phase 4.5 API versioning / Phase 3 alignment artifact: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-phase3-alignment.md`.
- [x] Created Phase 4.5 API versioning policy decision packet: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-api-versioning-policy-decision-packet.md`.
- [ ] Phase 4.5 API versioning: alignment artifact and decision packet exist; maintainer acceptance, `Docs/API/api-versioning-strategy.md` update, and any migration-guide implementation pending.
- [x] Created Phase 4.6 OpenAPI contract testing plan: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-openapi-contract-testing-plan.md`.
- [ ] Phase 4.6 OpenAPI contract testing: plan exists; implementation should wait for Phase 3.1/3.2 helper schema stability.
- [x] Created Phase 4 readiness gate summary: `Docs/superpowers/reviews/phase4-parking-lot/2026-04-25-phase4-readiness-gate.md`.
