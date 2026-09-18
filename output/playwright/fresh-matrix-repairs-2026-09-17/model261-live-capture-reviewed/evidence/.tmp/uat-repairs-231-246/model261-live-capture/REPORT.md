# UAT261 live-capture wrapper report

## Scope and result

This diagnostic-only implementation consists solely of:

- `uat261_capture.py`, exporting `create_app()` for the root-owned launcher;
- `test_live_capture.py`, isolated behavioral coverage; and
- this directory's run/report evidence.

No application source, maintained test, runtime, browser, database, provider
setting, or model setting was changed. No live call was made.

The first frozen wrapper source hash was
`2fc4903f398201fd16b652c33465628a688b566f0770a6da9d45d933a11224b0`.
Independent review reopened it after a retained control showed that closing an
unstarted wrapper generator did not close its original provider stream.

`create_app()` returns the already-loaded application after installing a
temporary wrapper at the exact module-level `complete-v2` provider seam. The
wrapper observes at most two calls, delegates exactly once per observed call,
and restores the original binding after the second call or shutdown.

## Safe projection and limits

The record uses the existing final-message envelope helper and a hash of only
the effective non-secret outbound controls. It retains terminal finish reason,
numeric usage, and a hash of a provider system fingerprint when present.
Credentials, headers, user IDs, config objects, messages, stream frames,
reasoning, provider text, and final text are excluded.

`final_answer` is deliberately `null`: a provider's `delta.content` can contain
inline think tags and `reasoning_content` is adapter-specific. A passive seam
observer cannot declare an exact final answer from either field. The native
protocol performs the visible exact-output comparison separately.

The correction replaces generator wrappers with explicit sync/async iterator
delegates. Their `close`/`aclose` calls the original provider source even if no
frame was requested, or after partial consumption, and preserves an original
close exception. `stream_observation_state` remains `incomplete` if malformed
frame parsing occurred, even when a later terminal frame permits a terminal
summary. This avoids hiding a partial-observation limitation behind
`capture_state: terminal_observed`.

A second review control found that the first explicit delegates eagerly called
`iter(source)`/`source.__aiter__()` in their constructors, moving acquisition
outside the product helper's bounded/cancellation-managed path. The corrected
delegates retain only the source at construction and acquire on
`__iter__`/`__aiter__` (or direct-next fallback). They close an acquired
iterator first and a distinct source second, matching
`_close_character_provider_stream`; if either close fails, both are attempted
and the first original error is re-raised. This is a diagnostic-wrapper-only
correction, not a product defect.

The pre-deferred-acquisition wrapper source hash was
`fb7061f8874e989cfd9532db5091c94957afcefd64eae4da9160895ce6914159`.

This can provide a bounded current boundary record. It cannot reconstruct the
historical UAT261 outbound fingerprint or prove historical/provider causality.

## Test-driven receipts

1. The initial contract test was intentionally red because `uat261_capture.py`
   did not exist (`FileNotFoundError` during collection).
2. The first implementation run found only an incorrect expected test fixture
   hash; the expected hash was corrected and the behavioral test passed.
3. A real `PromptCostEnvelope` shape control was added after source inspection.
   It was red because the wrapper expected a nonexistent `role_counts` member;
   the wrapper now derives only known role counts from the existing message
   list.
4. A dangling-output-symlink preflight control was red because `Path.exists()`
   ignores dangling symlinks; preflight now rejects either existence or a
   symlink before app import.

Correction red/green command:

```text
source .venv/bin/activate && python -m pytest \
  .tmp/uat-repairs-231-246/model261-live-capture/test_live_capture.py \
  .tmp/uat-repairs-231-246/model261-live-capture-review/test_cleanup.py -q --tb=short
```

The retained review red controls initially showed the original provider
`closed` counter at zero and eager iterator acquisition before consumption.
Final result: **18 passed**. This command exercised synthetic local iterators
only; it made no network or provider call.

## Static receipts

```text
source .venv/bin/activate && python -m py_compile \
  .tmp/uat-repairs-231-246/model261-live-capture/uat261_capture.py \
  .tmp/uat-repairs-231-246/model261-live-capture/test_live_capture.py
```

Exit 0.

```text
source .venv/bin/activate && python -m ruff check \
  .tmp/uat-repairs-231-246/model261-live-capture/uat261_capture.py \
  .tmp/uat-repairs-231-246/model261-live-capture/test_live_capture.py
```

Exit 0: `All checks passed!`

```text
source .venv/bin/activate && python -m bandit -r \
  .tmp/uat-repairs-231-246/model261-live-capture/uat261_capture.py -f json \
  -o .tmp/uat-repairs-231-246/model261-live-capture/bandit-source.json
```

Exit 0: 0 results and 0 errors.

A separate full diagnostic-directory Bandit receipt is retained in
`bandit-all.json`: 34 B101 assertions, all in `test_live_capture.py`. These are
test assertions, not source findings. The source-only result above is the
security result for the executable diagnostic wrapper. The full command exits
0 under this Bandit configuration despite those test findings, so this report
does not represent the whole directory as Bandit-clean.
