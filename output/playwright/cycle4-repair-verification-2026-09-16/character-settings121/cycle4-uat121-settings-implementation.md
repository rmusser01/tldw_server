# UAT121 Character current-chat controls

Ready for independent review. One production file, two test files, task13260.61. No backend production, store persistence, browser, inference, staging or commit changes.

The Character action omitted model controls that the current-chat dialog saves in the existing scoped in-memory model store. The action now captures numPredict, temperature, topP and repeatPenalty before asynchronous turn work and maps them to existing complete-v2 max_tokens, temperature, top_p and repetition_penalty fields. Unset fields are absent; valid zero values survive. Other settings without fields in the current endpoint contract are not forwarded. Provider/model/default resolution remains unchanged.

Current-chat controls deliberately have session lifetime. Reload blank is expected for this store, not a second persistence defect. Native117 report was clarified accordingly; the actual missing max_tokens request remains the121 failure. Native117 reasoning-only negative remains unverified; its single real completion was a positive answer control only.

## Validation

Permanent actual scoped store → real Character action → real delegated complete-v2 transport controls reproduced RED2 failed/20 passed. Final UI35 passed across3 suites: configured nonzero/zero values, absent controls on unconfigured model, pending save across model+scope switch retaining captured original values, invalidated owner recovery preserving new settings and no stale local writes, existing Character behavior/reasoning, scoped store and numeric form normalization.

From apps/packages/ui:

    ./node_modules/.bin/vitest run src/hooks/chat/__tests__/useChatActions.character.integration.test.tsx src/store/__tests__/model.scoped-settings.test.ts src/components/Common/Settings/__tests__/current-chat-model-settings-values.test.ts --maxWorkers=1 --no-file-parallelism

Logs /private/tmp/cycle4-uat121-settings-{red,green}.log.

Existing actual endpoint generation-override integration now also sends max_tokens16 and verifies16 at the provider boundary; existing omitted/default control passes. Backend2 passed/40 deselected/2 warnings:

    source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat_NEW/integration/test_role_normalization_and_search.py -k 'request_generation_fields_override_character_defaults or applies_character_generation_defaults_when_request_omits_fields' -q

Log /private/tmp/cycle4-uat121-settings-endpoint-green.log. Endpoint test extends an already-supported contract; no backend bug or new backend RED claimed.

Scoped ESLint0 errors/39 exact unchanged warnings on2 UI files, script /private/tmp/cycle4-uat121-settings-static-check.mjs and comparison.json under same prefix. Ruff1 existing I001 on backend test, unchanged against HEAD. Bandit full backend test:358 B101 test assertions (357 baseline plus required max_tokens assertion),3 existing B105 synthetic test-key literals unchanged. Skipping B101 confirms only those3 existing test-fixture findings; no new production security issue. Exact logs/JSON under same prefix. Parent owns combined compiler.

No new native pass claimed. Review and actual request-body verification must precede reasoning-only117 retest. Manifest /private/tmp/cycle4-uat121-settings-owned-manifest.json holds exact owned paths and hashes.

Native117/121 safe evidence: /private/tmp/cycle4-uat117-new-native-multi-final-report.md and evidence-manifest.json;18 artifacts,17 text artifacts rescanned against8 known credential values with zero matches. No headers or secrets retained.
