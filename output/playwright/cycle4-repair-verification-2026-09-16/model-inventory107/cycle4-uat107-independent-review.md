# Independent review: UAT107 current model inventory

## Verdict
Review-clear within the requested inventory/import/invalidation scope. No actionable correctness or regression finding. Unrelated Chat acknowledgement/metadata diffs were excluded.

## Inspected behavior
- `chat_service.py:959` reads `load_comprehensive_config()` instead of its import-time `_config`. Actual `save_first_run_provider` writes through `setup_manager.update_config`, then clears the central loader cache through `_refresh_runtime_config_cache`; the next strict inventory check therefore sees the saved configuration without a process restart.
- Both independent inventory LRUs are removed. The compatibility-named merged helper keeps no stale result, and invalidation no longer calls removed cache methods. No remaining application/test references to those removed cache-clear methods were found in the searched scope.
- Configuration still uses the central `lru_cache(maxsize=1)` loader. Pricing models come from the existing loaded `PricingCatalog` singleton. This patch adds no network discovery, provider inference, or file reload on every validation.
- The shared discovery resolver preserves custom OpenAI environment precedence. Numbered provider mappings remain intact and scoped; slot 2 has a behavioral test, with slots 3+ retained through the existing generated mapping and common resolver.
- Existing alias and provider normalization logic is unchanged. Strict flat-provider membership still rejects an unknown model when inventory exists. The pre-existing `None` result when no inventory exists remains unchanged; this is not a new universal deny-all policy.

## Independent verification
Ran from the project virtual environment:

```
python -m pytest tldw_Server_API/tests/Setup/test_setup_chat_model_refresh.py tldw_Server_API/tests/Setup/test_setup_provider_validation.py tldw_Server_API/tests/Chat/unit/test_chat_service_normalization.py -q --tb=short
```

Result: **57 passed, 4 warnings in 7.74s**, exit 0.
Log: `/private/tmp/cycle4-uat107-independent-tests.log`.
Pytest also reported cleanup warnings for an unrelated prior temporary `test_kokoro_constructor_direct3` directory; no test failure.

The six new tests exercise actual temporary config writes, the actual setup endpoint function/cache-refresh boundary, ordinary request normalization and real inventory validation. Four parameter cases cover cold/warm inventories for custom OpenAI and Ollama and two successive saves. Positive catalog controls, old startup-model rejection, unrelated-model rejection, numbered-slot isolation and environment-over-file precedence are meaningful. Catalog enumeration alone is controlled to keep the test independent of external/provider inventory. Reported historical RED6 and Bandit/Ruff results were read, not rerun/reclaimed as independent checks.

## Limits
The endpoint function is invoked directly with its authorization dependency bypassed for this isolated configuration test; this does not verify HTTP authentication or the browser. No native setup-to-Chat acceptance or live inference was performed. No source/test edits, runtime changes or commits were made. A native no-restart setup-to-ordinary-Chat check remains for the parent acceptance pass.
