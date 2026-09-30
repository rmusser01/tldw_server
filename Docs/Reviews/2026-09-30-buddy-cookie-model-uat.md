# Buddy cookie model discovery and UAT follow-up

A healthy quickstart WebUI cookie session showed an empty model selector because the shared catalog readiness guard required a key or token. The service now reuses the existing exact-origin active-cookie-session guard and gives cookie caches their own scope. Multi-user token requirements and browser authentication boundaries are unchanged.

## Verification

The original same-origin regression failed before the repair (35 passed, 1 failed). On current dev, 36 model-service and 7 browser-networking cases pass in 1.90 seconds. Negative cases cover cross-origin, advanced deployment, multi-user without a token and missing authSource. Scoped ESLint has zero errors with an existing unused inputMods warning and a Next pages-directory notice. Diff checks pass. No full suite ran; local Node26 evidence does not replace required hosted Node20 checks. Bandit is inapplicable to the two TypeScript files.

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
