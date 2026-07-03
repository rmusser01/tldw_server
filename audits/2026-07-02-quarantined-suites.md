# Quarantined Test Suites — burn-down tracker

These suites were hidden via `norecursedirs` until 2026-07-02; they are now
collected and skipped-with-reason by default (opt in: `RUN_QUARANTINED=1`).
Measured 2026-07-02 (60s per-test timeout):

| Suite | Failed | Passed | Skipped/xfail | Runtime |
|---|---|---|---|---|
| tests/Character_Chat_NEW | 68 | 408 | 4 | 6m45s |
| tests/TTS_NEW | 325 | 309 | 2 xfail | 10m24s |
| tests/Embeddings | 192 | 230 | 17 | 2m54s |

Exit criteria per suite: 0 failures with `RUN_QUARANTINED=1`, then delete the
quarantine hook from its conftest. Reproduce:

    RUN_QUARANTINED=1 PYTHONPATH=. PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 TEST_MODE=true \
    DISABLE_HEAVY_STARTUP=1 .venv/bin/python -m pytest tldw_Server_API/tests/<suite> -q \
    -p no:cacheprovider -p timeout --timeout=60
