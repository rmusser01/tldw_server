# UAT261 boundary harness run

## Scope

Only the temporary harness and evidence files under this directory were retained. `fixture_runner.py` preserves the exact one-off pytest module used for the fixture run. It was copied under `tldw_Server_API/tests/Character_Chat/` only to discover the existing Character_Chat fixtures, then removed after the successful run. No maintained source or test change remains.

## Startup reassessment

Three standalone startup attempts stopped before the route or provider seam. Their safe classifications are retained in `REASSESSMENT.md`; no startup logs or response bodies were retained. The successful run instead used the existing Character_Chat fixture environment and its `character_provider_adapter_boundary` seam.

## Command and result

```sh
source .venv/bin/activate && python -m pytest tldw_Server_API/tests/Character_Chat/test_uat261_boundary_harness_tmp.py -q --tb=short
```

Exit: `0`. The raw pytest output was intentionally not retained; the only retained original command result is the safe `fixture_harness_exit=0` receipt. The preserved runner defines one test, and pytest therefore completed that test successfully. No additional test-result detail is claimed.

The temporary runner called `run_fixture_harness`, which set disposable test-process AuthNZ and user-data paths before importing the route, patched only `character_chat_sessions.perform_chat_api_call`, and consumed two fresh `complete-v2` streams. The provider adapter was never called.

A follow-up command passed:

```sh
source .venv/bin/activate && python -m py_compile .tmp/uat-repairs-231-246/model261-boundary-harness/boundary_harness.py
```

Exit: `0`.

Scoped static check:

```sh
source .venv/bin/activate && python -m bandit -r .tmp/uat-repairs-231-246/model261-boundary-harness/boundary_harness.py -f json -o .tmp/uat-repairs-231-246/model261-boundary-harness/bandit.json
```

Exit: `0`. Result: 0 findings and 0 scanner errors.

## Safe evidence checks

- Calls A and B have equal assembled-message and generation-settings fingerprints.
- Synthetic terminal reasons are `stop` and `length`.
- Call B has `usage: null` and no final answer; synthetic reasoning was not converted into a final answer.
- The projection contains no supplied user text, synthetic reasoning or answer text, test credential marker, credential field name, or configuration field name.

No raw logs, prompt bodies, streams, provider reasoning, credentials, headers, IDs, or configuration values are retained.

## Independent replay procedure

1. Copy `fixture_runner.py` without edits to `tldw_Server_API/tests/Character_Chat/test_uat261_boundary_harness_tmp.py`.
2. Run the command above from the repository root with `.venv` active.
3. Remove only that copied temporary test module after the run. The retained harness and evidence remain under this directory.

The shim must stay in the Character_Chat test subtree so pytest discovers the existing `healthy_absent_provider_override_snapshot`, `character_provider_adapter_boundary`, and autouse rate-limit fixture. It also requires pytest's `monkeypatch`, `tmp_path`, and asyncio fixtures. `run_fixture_harness` establishes disposable AuthNZ and user-data paths before it imports the route; the fake provider boundary does not delegate to a provider adapter.
