# Quarantined Test Suites — burn-down tracker

> **CLOSED 2026-07-03 — all three suites unquarantined.** Full unrestricted
> re-runs on current `dev` came back green: Character_Chat_NEW 476P/4S/0F,
> TTS_NEW 626P/8S/0F, Embeddings 421P/18S/0F (after fixing one real
> cross-test bug: stale app-lifecycle drain state; see the autouse
> `_reset_app_lifecycle_state` fixture in `tests/Embeddings/conftest.py`).
> The quarantine hooks, the shared `tests/_plugins/quarantine.py` helper,
> and ci.yml's `RUN_QUARANTINED` env vars are all removed. This document is
> retained as a historical record of the 2026-07-02 measurements.

Tracking issue: https://github.com/rmusser01/tldw_server/issues/2581

These suites were hidden from **default, argument-less `pytest` collection and
local full-suite runs** via `norecursedirs` until 2026-07-02; they are now
collected and skipped-with-reason by default (opt in: `RUN_QUARANTINED=1`).

> **Correction (2026-07-02, remediation pass):** the opening claim above
> originally read "hidden via `norecursedirs` until 2026-07-02" without
> qualification, implying these suites never ran anywhere. That was wrong.
> `norecursedirs` never applies to explicit CLI paths, and
> `.github/workflows/ci.yml`'s gating shard jobs
> (`full-suite-linux-312-shards`, `full-suite-linux-313-shards`,
> `full-suite-macos-312-shards`, `full-suite-windows-312-shards`,
> `full-suite-os-313-release-shards`, `character-chat-rate-limits`) pass
> explicit file lists from all three directories and have run those curated
> subsets on every gating build the whole time — those CI jobs now set
> `RUN_QUARANTINED: "1"` in their `env:` block so the quarantine hooks don't
> turn that CI-gated coverage into silent skips. The local failure counts
> below (68 / 325 / 192) were measured with unrestricted, un-curated
> directory-wide runs and include failures specific to the local dev
> environment (missing models/services, e.g. real TTS engines and HF
> embedding downloads) that do not reproduce in CI's curated subsets, which
> stay green. **The burn-down below should start from the CI shard file
> lists as the known-green inventory** (see `ci.yml` shard `paths:` for
> `chat-character-unit-*`/`chat-character-integration-*`/`chat-character-property`,
> the `media-audio` shard's `tests/TTS_NEW`, and the `ai-embeddings-*`
> shards' `tests/Embeddings/test_*.py` patterns) rather than treating every
> local failure as a regression to fix before un-quarantining.

Measured 2026-07-02 (60s per-test timeout, unrestricted directory-wide local run):

| Suite | Failed | Passed | Skipped/xfail | Runtime | Status |
|---|---|---|---|---|---|
| tests/Character_Chat_NEW | 68 | 408 | 4 | 6m45s | **UNQUARANTINED 2026-07-03** — full re-run on dev: 476 passed / 4 skipped / 0 failed (1h14m); hook removed |
| tests/TTS_NEW | 325 | 309 | 2 xfail | 10m24s | **UNQUARANTINED 2026-07-03** — re-run on dev: 626 passed / 8 skipped / 0 failed (7m06s) |
| tests/Embeddings | 192 | 230 | 17 | 2m54s | **UNQUARANTINED 2026-07-03** — re-run on dev: 421 passed / 18 skipped / 0 failed (2m37s) after lifecycle-drain fixture fix |

> **2026-07-03 update:** the Character_Chat_NEW failures did not reproduce on
> current `dev` — the suite went green via intervening Character_Chat work
> (the 2026-07-02 numbers were measured against a main-based checkout under
> back-to-back-suite load). Its quarantine hook is removed. One watch item:
> `property/test_ambiguous_sender_heuristics.py` showed a transient failure
> in a partial run but passed the full run (possible hypothesis flake).

Exit criteria per suite: starting from the curated `ci.yml` gating shard file
lists as the known-green inventory (see the correction above), 0 failures
with `RUN_QUARANTINED=1` across the full, un-curated directory — not just the
shard subset — then delete the quarantine hook from its conftest and close
issue #2581 once all three suites clear. Reproduce:

    RUN_QUARANTINED=1 PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true \
    DISABLE_HEAVY_STARTUP=1 .venv/bin/python -m pytest tldw_Server_API/tests/<suite> -q \
    -p no:cacheprovider -p timeout --timeout=60
