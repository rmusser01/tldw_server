# Buddy cookie model discovery and UAT follow-up

A healthy quickstart WebUI cookie session showed an empty model selector because the shared catalog readiness guard required a key or token. The service now reuses the existing exact-origin active-cookie-session guard and gives cookie caches their own scope. Multi-user token requirements and browser authentication boundaries are unchanged.

## Verification

The original same-origin regression failed before the repair (35 passed, 1 failed). On current dev, 36 model-service and 7 browser-networking cases pass in 1.90 seconds. Negative cases cover cross-origin, advanced deployment, multi-user without a token and missing authSource. Scoped ESLint has zero errors with an existing unused inputMods warning and a Next pages-directory notice. Diff checks pass. No full suite ran; local Node 26 evidence does not replace required hosted Node 20 checks. Bandit is inapplicable to the two TypeScript files.

Real disposable fresh and schema65-to73 upgraded WebUI sessions selected DeepSeek models and received the expected replies. This browser evidence used reviewed head eaebb194b717d3adc335dbca8961bc9ef5884ac5 plus the semantic catalog repair; subsequent current-dev code in the affected catalog/auth paths is unchanged. Workspace inheritance/clear, attachment ownership, navigation, unsent drafts, unread acknowledgement and Static/Dynamic reload persistence were observed.

## Remaining acceptance

The actual legacy Persona Buddy rendered its 128x128 artwork, accepted browser pointer dragging and Home reset, and remained available for 6 minutes 25 seconds with two pack-list, two pack-detail and one session-list requests, zero failures and zero HTTP429 responses. TASK-13211 AC3 is checked; its historical initiating trigger remains unreproduced and AC1 remains open.

A separately authorized synthetic Chatbook realtime request completed with 182400 output-audio bytes and closed its session. That probe did not open a microphone or play audio. Local Kokoro assets were detected after correcting only private runtime configuration. TASK-13202 remains open for intentional human speech, audible output and correlated floating states. Native Terminal Computer Use is explicitly unavailable; headless native input checks do not establish physical desktop acceptance. Extension installation was removed from scope by the requester.

All 164 distinct earlier targeted assertions and broader UAT source attribution remain in the private handoff. Raw logs, credentials, profiles and Next builds remain local. Owned UAT listeners were stopped; the prior production extension build is preserved. The completed PR #3056 and #2910 merges are unchanged.

## Receipt

```json
{
  "base": "f3f1b4fdbe3fe461b371ece30887c5fff8476d9d",
  "file_sha256": {
    "apps/packages/ui/src/services/tldw/TldwModels.ts": "b93bc7227717bdb677d08c6c01e01aa15d7c6a6d14424babc154a32be6f60fc5",
    "apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts": "c0a06d50d7cbc5d119b468a4e29853cb92b3e4d34c3a433877dce717036bceb5"
  },
  "targeted_tests": {
    "model_service": 36,
    "browser_networking": 7,
    "total": 43,
    "duration_seconds": 1.9,
    "node": "26.0.0",
    "vitest": "4.0.18"
  },
  "lint": {
    "errors": 0,
    "existing_warning": "inputMods unused",
    "pages_directory_notice": true
  },
  "full_suite": false,
  "bandit": "Not applicable: two TypeScript source/test files; no Python change",
  "original_live_source": "eaebb194b717d3adc335dbca8961bc9ef5884ac5 plus semantic two-file repair",
  "live_profiles": [
    "fresh schema73",
    "actual predecessor schema65 upgraded to73"
  ],
  "remaining_acceptance": [
    "intentional human speech, audible output and correlated floating states",
    "physical native desktop interaction",
    "original historical reload trigger"
  ],
  "raw_logs_published": false
}
```

## Qodo cookie-expiry correction

Qodo found that persisted cookie metadata can outlive the server session. Five shared-service regression cases reproduced stale results from fresh/expired caches, the forced-refresh cooldown and cache-only hydration. Auditing callers found the chat selector has a second cache; its three expiry/fallback cases also failed before repair.

A real disposable API probe then disproved the assumption that metadata itself verifies authentication: model metadata returns HTTP 200 after the exact session expires because the catalog is public. The existing authenticated profile route returns 200 with the live cookie and 401 after expiry. The first probe retained its incorrect metadata 401 assertion; another probe omitted the Secure cookie because HTTPX excludes it on loopback HTTP. Explicitly sending only that owned opaque cookie completed the real profile 200-to-401 check. No server authentication or cookie policy was changed.

Cookie catalogs now first require the existing getCurrentUserProfile method, then fetch model metadata. Neither shared nor outer chat-selector caches provide cookie fallback or cache-only hydration. A failed fetch clears only its current shared-cache generation. Key/token caching and concurrent request sharing remain intact; a late cookie failure cannot erase a newer key catalog within the same service instance. No new endpoint, credential authority or timeout was added.

The final five-file targeted scope passes 66 cases in 2.32s, including a regression that first failed when public metadata succeeded without live authentication. Scoped ESLint has zero errors and 12 pre-existing warnings across the expanded scope; diff checks pass. Earlier 43/65-case and browser receipts retain their original source attribution. Current-head hosted Node 20 checks and Qodo re-review remain pending. Browser expiry observation is unverified: CDP navigation/focus timed out after the local WebUI compiled. Bandit is inapplicable: only TypeScript and documentation changed.

Local speech follow-up: production Kokoro ONNX CPU generated a 1.408-second WAV and released its model. Existing cached Parakeet MLX weights were found, but loading failed with `No Metal device available` in the ordinary process, an approved native process and with MLX's CPU default. No human speech, microphone capture or audible-output confirmation is claimed. Raw logs and audio stay private.

```json
{
  "qodo_finding_reviewed_source": "c4cd55403d4482368c299b3a7f243989b4ac8081",
  "tested_source": "effca2541f6732c83886eead5216641d79d99654",
  "base": "f3f1b4fdbe3fe461b371ece30887c5fff8476d9d",
  "file_sha256": {
    "apps/packages/ui/src/services/tldw/TldwModels.ts": "4a2fa5c6a9800c2aebc1f6bac7f76a58b7e381160924c6bb89b9e7d92f13b628",
    "apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts": "33afa916b12ca1bfce2308bc58f6811233a87e631c8f586fd1de4e2e88878deb",
    "apps/packages/ui/src/services/tldw-server.ts": "5b5c6db7288b53c7c2363c2a12d42f4bf8bf1a8804a981b5b7a2931f89cc280c",
    "apps/packages/ui/src/services/__tests__/tldw-server.fetch-chat-models.test.ts": "9bafb2986b694354312d01e49e7df02538a3cfa821bb1a2231b16a04230dca9f",
    "apps/packages/ui/src/services/__tests__/tldw-server.chat-models.test.ts": "f9de4d245b781b8e3c005edb98996f5711ff7aae094740578b573c61addc7486"
  },
  "regression_red": {
    "shared_cookie_expiry_cases_failed": 5,
    "outer_cookie_cache_cases_failed": 3,
    "public_metadata_without_live_auth_failed": 1
  },
  "targeted_green": {
    "model_service": 44,
    "browser_networking": 7,
    "normalization": 4,
    "chat_catalog_wrapper": 10,
    "chat_catalog_provider_identity": 1,
    "total": 66,
    "duration_seconds": 2.32
  },
  "lint": {
    "errors": 0,
    "pre_existing_warnings": 12,
    "pages_directory_notice": true
  },
  "key_token_cache_preserved": true,
  "cookie_catalog_requires_authenticated_fetch": true,
  "cookie_cache_only_hydration_withheld": true,
  "concurrent_fetch_deduplication_preserved": true,
  "old_cookie_failure_cannot_clear_new_key_catalog_within_one_instance": true,
  "no_new_endpoint": true,
  "full_suite": false,
  "raw_logs_published": false,
  "cookie_authentication": "Existing getCurrentUserProfile request must succeed before public model metadata is fetched",
  "real_api_probe": {
    "scenario": "Opaque cookie retained in client, exact disposable server session expiry moved into past",
    "mint_status": 200,
    "public_catalog_live_status": 200,
    "public_catalog_expired_status": 200,
    "authenticated_profile_live_status": 200,
    "authenticated_profile_expired_status": 401,
    "only_owned_session_modified": true,
    "client_cookie_retained": true,
    "static_key_used_only_to_mint": true,
    "opaque_cookie_sent_explicitly_on_loopback_http": true,
    "no_provider_request": true,
    "no_microphone": true,
    "cookie_token_recorded": false,
    "expiry_restored": false
  },
  "browser_expiry_observation": "Not established: native browser CDP focus/navigation timed out after local compilation; automated and real API evidence are separate",
  "earlier_mock_only_green": {
    "total": 65,
    "duration_seconds": 3.03,
    "limit": "Catalog metadata alone does not validate authentication"
  }
}
```

## Reviewed cache follow-up and live expiry UAT

Cubic identified two requests crossing cookie/key authentication modes and a cookie failure overwriting another service instance's newer persisted key catalog; regressions reproduced both. Separate AbortError/network-error probes showed that clearing all cookie state also loses the last catalog after live profile authentication succeeds. Five semantic regressions failed before this repair; four additional assertions confirmed unwanted cookie persistence.

The chat wrapper now shares pending requests only within the same cookie/key mode. Cookie catalogs stay in memory and do not write the shared persistent cache, so their success or failure cannot overwrite another context's key/token record. HTTP 401/403 clears current cookie memory; a failed profile check withholds models. If the profile check succeeds and only metadata encounters a transient failure, the same generation may reuse its memory catalog. Existing key/token persistence and same-mode deduplication remain intact. The unforced wrapper expiry case now explicitly rejects with HTTP 401. No endpoint, credential rule, schema, dependency or timeout changed.

Tested source 71f9f93ce5293c7c8cd6f6951863542197a670b1 passes 72 targeted cases in 2.52 seconds. Scoped ESLint reports zero errors and 12 pre-existing warnings. The first lint invocation ignored files because its working directory put them outside the config base; it is not passing evidence. The root-level invocation with the existing config checked all five files. An initial expanded test command named a nonexistent normalization file and ran only 68 cases; the corrected five-file run supplies the 72-case evidence. Diff checks pass. Hosted Node 20 checks remain pending.

Real browser expiry/recovery is now established. On earlier head effca2541f, full reload legitimately minted a fresh cookie, so that first reload did not test expired-session withholding. Expiring the mounted session and opening Manage Models instead showed sign-in required and zero usable providers. Restoring its original expiry values and using Refresh restored three usable providers; the unsent draft survived. On the final tested source, Refresh on an already mounted Manage Models page again withheld usable models after expiry, then recovered three usable providers after all five exact owned fixture expiry values were restored. No new provider request or microphone capture occurred. The public catalog reference remains visible as non-readiness documentation. Returning to Chat preserved its two original messages, unsent draft and independent Buddy artwork. The owned disposable API/WebUI listeners were stopped afterward; both ports were verified closed, and the production extension build remains available.

The earlier c4cd5540 source identifies Qodo's finding; the 66-case file hashes and tests belong to effca2541f. That distinction is corrected in the earlier receipt, and its late-failure guarantee is explicitly limited to one instance. The receipt below covers both independent service instances. Earlier browser, voice and targeted-test receipts retain their original attribution. TASK-13398 stays In Progress for current-head review and hosted gates; human speech/audibility, physical native interaction and the historical reload trigger remain open.

```json
{
  "tested_source": "71f9f93ce5293c7c8cd6f6951863542197a670b1",
  "base": "f3f1b4fdbe3fe461b371ece30887c5fff8476d9d",
  "file_sha256": {
    "apps/packages/ui/src/services/tldw/TldwModels.ts": "6de558a8bb24d9752f6c1d90224c8b7699d831e162da102488bc781d497f674a",
    "apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts": "ffe49510038476ee16833abb68bae9f123ad4a122f86cbd361c67a4b03d38ded",
    "apps/packages/ui/src/services/tldw-server.ts": "4f4f52ee8aafec54cb5d42f1b188afa6804f1ebfb5e0701acbe6514c60de91d5",
    "apps/packages/ui/src/services/__tests__/tldw-server.fetch-chat-models.test.ts": "c915ed0fd338cbbdc53d32ae3eceeaa2a46b282aecffa8608d6cead510432d03",
    "apps/packages/ui/src/services/__tests__/tldw-server.chat-models.test.ts": "f9de4d245b781b8e3c005edb98996f5711ff7aae094740578b573c61addc7486"
  },
  "red": {
    "new_semantic_regressions_failed": 5,
    "cookie_persistence_assertions_failed": 4,
    "total_failed": 9,
    "passed": 50
  },
  "targeted_green": {
    "model_service": 47,
    "browser_networking": 7,
    "normalization": 4,
    "chat_catalog_wrapper": 13,
    "chat_catalog_provider_identity": 1,
    "total": 72,
    "duration_seconds": 2.52,
    "node": "26.0.0",
    "vitest": "4.0.18"
  },
  "lint": {
    "errors": 0,
    "pre_existing_warnings": 12,
    "pages_directory_notice": true
  },
  "cookie_same_auth_mode_deduplication": true,
  "cross_auth_mode_promises_isolated": true,
  "cookie_cache_persisted": false,
  "profile_failure_fallback": false,
  "authenticated_transient_catalog_fallback": true,
  "cross_instance_key_cache_preserved": true,
  "live_browser": {
    "source": "71f9f93ce5293c7c8cd6f6951863542197a670b1",
    "scenario": "Existing Manage Models Refresh, expired exact private browser sessions, then restored original expiries and Refresh",
    "usable_providers_expired": 0,
    "usable_providers_recovered": 3,
    "original_expiry_values_restored": 5,
    "provider_requests": 0,
    "microphone_capture": false,
    "screenshot_sha256": {
      "cookie-review-ui-withheld.jpg": "ca781a8f514414dfbb1c2ea827f7a57a0e85d0142f3b47496aede9d1b03e0247",
      "cookie-review-ui-recovered.jpg": "88f0189ca32a642ce2baf2eb674bf82b093fe10b3f9ba3b5882d620b413e63fd",
      "cookie-review-ui-chat-preserved.jpg": "a804ad49e61ed6ee701e85d49b2779859b71bed8ca1d1e38d5d438afaeef49cd"
    },
    "chat_messages_preserved": 2,
    "unsent_draft_preserved": true,
    "independent_buddy_artwork_present": true
  },
  "full_suite": false,
  "raw_logs_published": false,
  "bandit": "Not applicable: TypeScript/test/documentation changes only",
  "cleanup": {
    "owned_api_and_web_listeners_stopped": true,
    "ports_closed": [
      18280,
      18281
    ],
    "production_extension_build_preserved": true
  }
}
```

## Current-dev qualification — October 1

Actual dev advanced to b365af1827b607fc221f9bf31ca76dde881edb4f with strict Workspace startup and resource-governance/auth changes. Rebased all four prior commits without conflicts; range-diff confirms all four patches are identical. The incoming frontend change is only the OpenAPI fingerprint; the catalog, cookie client, frontend smoke tests and UX workflow are unchanged. Earlier receipts keep their original tested sources.

At tested source a7e14881f1d9b92100ce6ff2fbf31e473259f0ba, the existing five-file frontend scope passes 72 cases in 2.42 seconds. Seven incoming backend files pass 124 cases in 88.89 seconds, with 32 PostgreSQL-backed variants skipped and 25 warnings reported. These cover cookie authentication, RG ownership/replay and same-entity charging, Buddy model handoff, strict Workspace startup and migration. Scoped ESLint reports zero errors and the same 12 pre-existing warnings; diff checks pass. An initial lint command pointed to a nonexistent legacy config and is excluded; the corrected command used the repository's existing eslint.config.mjs. No full suite ran.

A separate copy of the previous private upgraded profile migrated from SQLite schema 73 to 74 during current-dev startup. The real protected profile request returned 200 with a live opaque cookie, 401 after its exact owned session expiry was moved into the past, and 200 after that same expiry was restored. Public metadata remained 200 throughout. The existing chat read returned 200; hashes of the conversation and all three stored message rows remain unchanged. This is API/storage evidence; the earlier two-message mounted browser observation retains its original source. The original schema-73 private profile is untouched. No provider request or microphone capture occurred. The initial sandbox process could not bind its loopback socket; the approved native process supplied the live evidence. Only that owned API was stopped, both ports are closed, and existing private builds remain preserved.

On previous published head 3f2625f4c04d9565dc0e40081b560669a7c6b1dd, all seven actual dev ruleset gates passed, Qodo reported zero findings, and all eight review threads are resolved. Those are previous-head results after this rebase. The auxiliary UX Smoke Gate failed before all-pages navigation because 31 existing exception ownership records expired on September 30. Earlier route, cockpit and audio stages passed; the expired guard supplies no all-pages route evidence. The expiry guard and exception dates remain unchanged. Matching-head reviews and hosted checks are required after publication; TASK-13398 stays In Progress.

```json
{
  "tested_source": "a7e14881f1d9b92100ce6ff2fbf31e473259f0ba",
  "base": "b365af1827b607fc221f9bf31ca76dde881edb4f",
  "previous_published_head": "3f2625f4c04d9565dc0e40081b560669a7c6b1dd",
  "all_four_prior_patches_identical": true,
  "file_sha256": {
    "apps/packages/ui/src/services/tldw/TldwModels.ts": "6de558a8bb24d9752f6c1d90224c8b7699d831e162da102488bc781d497f674a",
    "apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts": "ffe49510038476ee16833abb68bae9f123ad4a122f86cbd361c67a4b03d38ded",
    "apps/packages/ui/src/services/tldw-server.ts": "4f4f52ee8aafec54cb5d42f1b188afa6804f1ebfb5e0701acbe6514c60de91d5",
    "apps/packages/ui/src/services/__tests__/tldw-server.fetch-chat-models.test.ts": "c915ed0fd338cbbdc53d32ae3eceeaa2a46b282aecffa8608d6cead510432d03",
    "apps/packages/ui/src/services/__tests__/tldw-server.chat-models.test.ts": "f9de4d245b781b8e3c005edb98996f5711ff7aae094740578b573c61addc7486"
  },
  "frontend_tests": {
    "passed": 72,
    "duration_seconds": 2.42,
    "node": "26.0.0",
    "vitest": "4.0.18"
  },
  "incoming_backend_tests": {
    "passed": 124,
    "skipped": 32,
    "duration_seconds": 88.89,
    "warnings_reported": 25,
    "skip_scope": "PostgreSQL-backed fixture variants; no local PostgreSQL acceptance claimed",
    "scope": "single-user cookie auth, RG cookie owner and WebUI replay, same-entity charging, Buddy/workspace model handoff, strict Workspace startup API and migration"
  },
  "lint": {
    "errors": 0,
    "pre_existing_warnings": 12,
    "initial_missing_config_attempt_excluded": true
  },
  "real_api_probe": {
    "tested_source": "a7e14881f1d9b92100ce6ff2fbf31e473259f0ba",
    "base": "b365af1827b607fc221f9bf31ca76dde881edb4f",
    "mint_status": 200,
    "public_catalog_live_status": 200,
    "public_catalog_expired_status": 200,
    "authenticated_profile_live_status": 200,
    "authenticated_profile_expired_status": 401,
    "authenticated_profile_recovered_status": 200,
    "exact_owned_session_expiry_restored": true,
    "opaque_cookie_retained": true,
    "static_key_used_only_to_mint": true,
    "chat_read_status": 200,
    "sqlite_schema_before": 74,
    "sqlite_schema_after": 74,
    "conversation_rows_retained": 1,
    "message_rows_retained": 3,
    "visible_messages_retained": 3,
    "chat_row_hashes_unchanged": true,
    "original_private_profile_untouched": true,
    "provider_requests": 0,
    "microphone_capture": false,
    "browser_observation": false,
    "credentials_recorded": false,
    "original_and_initial_clone_schema": 73,
    "migration_at_startup_before_api_probe": true,
    "original_profile_row_hashes_unchanged": true
  },
  "cleanup": {
    "owned_api_pid": 14630,
    "signal": "SIGTERM",
    "exit_code": 143,
    "ports_closed": [
      18280,
      18281
    ],
    "original_profile_preserved": true,
    "private_next_builds_preserved": true
  },
  "hosted_evidence": {
    "head": "3f2625f4c04d9565dc0e40081b560669a7c6b1dd",
    "required_gates_passed": 7,
    "qodo_findings": 0,
    "review_threads_resolved": 8,
    "current_rebased_head_gates_and_review": "Pending publication and actual matching-head results",
    "auxiliary_ux_smoke_failure": "31 all-pages allowlist ownership entries expired 2026-09-30; expiry guard retained",
    "failure_url": "https://github.com/rmusser01/tldw_server/actions/runs/36799310033/job/110179389812"
  },
  "full_suite": false,
  "paid_provider_requests": 0,
  "microphone_capture": false,
  "raw_logs_published": false,
  "bandit": "Inapplicable: TypeScript/test/docs-only follow-up; incoming backend tests run against unchanged dev code",
  "remaining_acceptance": [
    "intentional human speech and heard audio with correlated floating states",
    "physical native desktop interaction",
    "historical reload initiating trigger"
  ]
}
```

## October 1 later-dev capability and authentication integration

Actual dev 5f3ed81e88ec44750a5baa7838f7c69672e32694 brings FastAPI 0.142.1, cookie-account admission, the single RG switch, audio authentication fixes and fresh image-capability Retry. The two shared catalog patches needed context/guard reconciliation; the other three prior patches remained identical. The existing fresh-read generation and concurrency controls are retained. Authenticated cookie discovery now also honors requireFresh: transient metadata failure returns no stale capability while ordinary authenticated refresh may reuse current-generation memory. Cookie 401/403 still clears memory first. Two regressions failed before the one-line guard; the incoming caller mock needed the real canonical cookie helper and storage serializer exports. All 13 real image-retry caller cases pass in the final 223-case scope.

At tested source 0b41be14d71d77e831e1b1f4a19d3df59f9f3c3c, 223 frontend cases pass in 4.21s, and 150 backend cases pass in 102.23s with 32 unreachable-PostgreSQL variants skipped and 25 warnings. Six-file ESLint reports zero errors and 12 existing warnings; diff checks pass. Incoming dev already retired 31 expired UX exceptions with owner evidence. Its four remaining records validate at the real clock, and expired/invalid calendar negatives still reject; four existing classification tests pass in 0.897s. This does not claim an all-pages browser run. Failed mock/config/empty-profile setup attempts remain private and excluded. No full suite, provider request, microphone or guard relaxation occurred.

A fresh copy of the populated schema-73 UAT profile migrated to74 during startup using existing FastAPI0.142.1 bytes. The protected opaque-cookie profile returned200→401→200 after exact expiry restoration; public metadata stayed200. The chat read returned200 and hashes of its one conversation and three stored messages remain unchanged. The original profile remains73 with unchanged hashes. Only the owned API was stopped; ports18280/18281 are closed and private builds remain preserved. Earlier mounted-browser, artwork/drag, provider-reply and voice observations keep their original tested sources.

Dev then advanced to 3304f6cf0c2cab8836c6a73308d7210534a759d9 through documentation-only PR#3073: only unrelated TASK-13405 changed. All seven preceding branch patches remain identical after that rebase, and runtime/tests/dependencies/workflows are byte-identical to the tested source; no redundant rerun was needed. Incoming FastAPI independently owns TASK-13398. The cookie task is now unique TASK-13408; only its filename/frontmatter identity was mechanically repaired because the Python CLI has no renumber operation. All subsequent sections use that CLI; canonical-marker/parser/append/summary roundtrip retains four AC and six DoD items, status In Progress. Historical TASK-13398 mentions keep their original cookie attribution. The incoming FastAPI task is byte-identical to dev.

All seven required gates and Qodo zero findings/eight resolved threads on72623044 retain old-head attribution. New-head hosted gates/review remain pending. Human speech/audibility, physical native interaction and the historical reload initiating trigger remain open.

Before publication, dev advanced once more to d81c13fddd1dac1948b30401af0388932a0af8f2 via documentation-only PR#3072, adding seven unrelated Backlog records. All eight branch patches remain identical after this second documentation rebase; runtime, tests, dependencies and workflows remain byte-identical to the tested source.

```json
{
  "tested_source": "0b41be14d71d77e831e1b1f4a19d3df59f9f3c3c",
  "tested_base": "5f3ed81e88ec44750a5baa7838f7c69672e32694",
  "publication_base": "d81c13fddd1dac1948b30401af0388932a0af8f2",
  "previous_published_head": "72623044da3e2e14453f440e3b64d700a853b8cd",
  "initial_rebase": {
    "prior_commits": 5,
    "patches_identical": 3,
    "two_catalog_patches_adjusted_for_incoming_fresh_controls": true
  },
  "documentation_only_rebase": {
    "all_seven_patches_identical": true,
    "source_tests_dependencies_workflows_byte_identical": true,
    "incoming_change": "backlog/tasks/task-13405 - RG-ingress-safety-net-spec-1-of-2.md"
  },
  "file_sha256": {
    "apps/packages/ui/src/services/tldw/TldwModels.ts": "36cb2954e38538bfc441452b08c46539af19b188dd9aea3d8df89448793ddf94",
    "apps/packages/ui/src/services/tldw/__tests__/TldwModels.test.ts": "0fd8f01cc6f8864ffd73f3b3848c9c69c1f8669db0059722aa55296ba7a972a1",
    "apps/packages/ui/src/services/tldw-server.ts": "4f4f52ee8aafec54cb5d42f1b188afa6804f1ebfb5e0701acbe6514c60de91d5",
    "apps/packages/ui/src/services/__tests__/tldw-server.fetch-chat-models.test.ts": "c915ed0fd338cbbdc53d32ae3eceeaa2a46b282aecffa8608d6cead510432d03",
    "apps/packages/ui/src/services/__tests__/tldw-server.chat-models.test.ts": "f9de4d245b781b8e3c005edb98996f5711ff7aae094740578b573c61addc7486",
    "apps/packages/ui/src/models/__tests__/image-retry-capability.test.ts": "bac8a10adf8b18835c0c8ea0ac63514853d5a1b7f73a2c709500a8c9dbabd271"
  },
  "red": {
    "fresh_cookie_transient_failures": 2,
    "caller_mock_missing_export_failures": 13,
    "setup_attempts_excluded": true
  },
  "frontend_tests": {
    "passed": 223,
    "files": 9,
    "duration_seconds": 4.21,
    "node": "26.0.0",
    "vitest": "4.0.18"
  },
  "backend_tests": {
    "passed": 150,
    "skipped": 32,
    "warnings_reported": 25,
    "duration_seconds": 102.23,
    "skip_scope": "PostgreSQL not reachable; no PostgreSQL acceptance claimed",
    "fastapi": "0.142.1",
    "starlette": "1.2.1",
    "pydantic": "2.11.7",
    "scope": "cookie auth/RG owner/replay/single-charge/single-switch, audio health auth, Buddy handoff, strict Workspace startup/migration"
  },
  "lint": {
    "files": 6,
    "errors": 0,
    "pre_existing_warnings": 12,
    "initial_missing_config_attempt_excluded": true
  },
  "incoming_smoke_repair": {
    "current_rules": 4,
    "validation_errors": [],
    "real_clock_used": true,
    "expired_date_rejected": true,
    "invalid_calendar_rejected": true,
    "full_pages_run": false
  },
  "smoke_classification_tests": {
    "passed": 4,
    "duration_seconds": 0.897,
    "all_pages_navigation": false
  },
  "real_api_probe": {
    "tested_source": "0b41be14d71d77e831e1b1f4a19d3df59f9f3c3c",
    "base": "5f3ed81e88ec44750a5baa7838f7c69672e32694",
    "mint_status": 200,
    "public_catalog_live_status": 200,
    "public_catalog_expired_status": 200,
    "authenticated_profile_live_status": 200,
    "authenticated_profile_expired_status": 401,
    "authenticated_profile_recovered_status": 200,
    "exact_owned_session_expiry_restored": true,
    "opaque_cookie_retained": true,
    "static_key_used_only_to_mint": true,
    "chat_read_status": 200,
    "sqlite_schema_before": 74,
    "sqlite_schema_after": 74,
    "conversation_rows_retained": 1,
    "message_rows_retained": 3,
    "visible_messages_retained": 3,
    "chat_row_hashes_unchanged": true,
    "original_private_profile_untouched": true,
    "provider_requests": 0,
    "microphone_capture": false,
    "browser_observation": false,
    "credentials_recorded": false,
    "original_and_initial_clone_schema": 73,
    "migration_at_startup_before_api_probe": true,
    "original_profile_row_hashes_unchanged": true,
    "initial_empty_profile_excluded": true,
    "python39_readonly_probe_excluded": true
  },
  "backlog_roundtrip": {
    "task": "TASK-13408",
    "canonical_markers_retained": true,
    "ac_unchanged": 4,
    "dod_unchanged": 6,
    "append_notes_and_summary_roundtrip": true,
    "incoming_fastapi_task_unchanged": true,
    "status": "In Progress"
  },
  "cleanup": {
    "owned_api_pid": 28665,
    "signal": "SIGTERM",
    "exit_code": 143,
    "ports_closed": [
      18280,
      18281
    ],
    "private_profiles_builds_preserved": true
  },
  "hosted_evidence": {
    "head": "72623044da3e2e14453f440e3b64d700a853b8cd",
    "required_gates_passed": 7,
    "qodo_findings": 0,
    "review_threads_resolved": 8,
    "auxiliary_ux_failure": "Old expired exceptions; incoming dev retires31 with owner evidence and keeps4 validated narrow records",
    "new_head_checks_and_review": "Pending publication and matching-head results"
  },
  "canonical_cookie_task": "TASK-13408",
  "full_suite": false,
  "paid_provider_requests": 0,
  "microphone_capture": false,
  "raw_logs_published": false,
  "adr_required": false,
  "bandit": "Inapplicable to TypeScript/test/docs-only changes; no backend source changed",
  "remaining_acceptance": [
    "intentional human speech and heard audio with correlated floating states",
    "physical native desktop interaction",
    "historical reload initiating trigger"
  ],
  "second_documentation_only_rebase": {
    "all_eight_patches_identical": true,
    "incoming_backlog_records": 7,
    "runtime_tests_dependencies_workflows_byte_identical": true
  }
}
```
