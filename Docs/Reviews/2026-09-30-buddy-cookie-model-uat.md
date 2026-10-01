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
