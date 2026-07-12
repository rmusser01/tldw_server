# Parakeet ONNX Bundle Metadata Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent Parakeet TDT v3 ONNX bundles from reaching transcription with an 80-feature preprocessor when their encoder requires 128 features.

**Architecture:** Keep `onnx-asr` responsible for preprocessing and decoding. Make its required `config.json` metadata part of bundle completeness, refresh incomplete remote caches, and validate the configured feature count against the selected encoder graph before caching the runtime.

**Tech Stack:** Python 3.11+, pathlib/json, onnxruntime, onnx-asr, pytest/unittest.mock, Loguru.

---

## File Structure

- Modify `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py` for sidecar completeness and feature-dimension validation.
- Modify `tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py` for cache, local-bundle, metadata, and graph-mismatch regressions.
- Update `backlog/tasks/task-12949 - Fix-Parakeet-ONNX-80-128-feature-mismatch-for-incomplete-model-bundles.md` through Backlog MCP with verification and final status.
- Remove this plan file after all stages are complete, per repository instructions.

### Task 1: Reproduce incomplete-cache behavior

**Files:**
- Test: `tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py`

- [ ] **Step 1: Add failing remote-cache test**

Create an existing cache directory with graphs and `vocab.txt`, omit
`config.json`, invoke the configured Hugging Face model path, and assert
`snapshot_download` is called with `config.json` in `allow_patterns`.

- [ ] **Step 2: Add failing local-bundle test**

Create a local graph bundle with `vocab.txt` but no `config.json`; assert the
loader returns `(None, None)`, never calls `onnx_asr.load_model`, and logs the
missing sidecar plus the local model directory.

- [ ] **Step 3: Verify RED**

Run:

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
PYTHONDONTWRITEBYTECODE=1 python -m pytest \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py \
  -q -p no:cacheprovider
```

Expected: the new tests fail because cache freshness and bundle completeness
only check `vocab.txt`.

### Task 2: Reproduce invalid metadata and encoder mismatch

**Files:**
- Test: `tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py`

- [ ] **Step 1: Add invalid-config parameterized test**

Cover malformed JSON and `features_size` values that are missing, boolean,
string/float, zero, or negative. Assert fail-closed behavior before
`onnx_asr.load_model`, plus an actionable log containing the model directory.

- [ ] **Step 2: Add parameterized selected-encoder mismatch test**

Provide `features_size: 80` and an `audio_signal` input shaped
`[batch, 128, time]` on the encoder session selected by quantization. Cover
both paired INT8 graphs and paired unquantized graphs, and assert the exact
selected encoder path. Assert the loader returns `(None, None)` and does not
call `onnx_asr.load_model`.

- [ ] **Step 3: Update the valid bundle fixture**

Add the real v3 config shape:

```json
{"model_type": "nemo-conformer-tdt", "features_size": 128, "subsampling_factor": 8}
```

Expose an encoder input metadata double with name `audio_signal` and shape
`[None, 128, None]`.

- [ ] **Step 4: Verify RED remains diagnostic**

Run the focused command from Task 1. Expected: new invalid/mismatch assertions
fail because the loader does not validate metadata yet; existing tests remain
otherwise healthy.

### Task 3: Implement the minimal loader fix

**Files:**
- Modify: `tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py`

- [ ] **Step 1: Make sidecar completeness explicit**

Add a required-sidecar tuple containing `vocab.txt` and `config.json`. Use it
both for remote-cache refresh and bundle resolution. Preserve the rule that
explicit local paths are never treated as Hugging Face repository IDs, and log
the missing sidecars plus model directory when such a local bundle fails.

- [ ] **Step 2: Parse and validate `features_size`**

Load `config.json`, reject malformed/non-object content, and accept only a
positive integer that is not a boolean. Log the model directory and validation
reason without logging configuration contents.

- [ ] **Step 3: Validate the selected encoder**

Resolve quantization first, choose the matching encoder path, inspect the
`audio_signal` input metadata with onnxruntime, and reject a declared static
feature axis that differs from `features_size`. Log the model directory plus
expected and actual feature counts. Dynamic axes remain accepted.

- [ ] **Step 4: Verify GREEN**

Run the focused command from Task 1. Expected: all focused tests pass with the
existing expected skips.

- [ ] **Step 5: Refactor only if needed**

Keep helpers private and small; do not add dependencies, fallback behavior, or
new abstractions.

### Task 4: Verify and finalize

**Files:**
- Update through MCP: `backlog/tasks/task-12949 - Fix-Parakeet-ONNX-80-128-feature-mismatch-for-incomplete-model-bundles.md`

- [ ] **Step 1: Run focused tests again**

Run the Task 1 command and record totals.

- [ ] **Step 2: Run Ruff on touched Python files**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m ruff check \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py \
  tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py
```

- [ ] **Step 3: Run Bandit on touched production scope**

```bash
source /Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py \
  -f json -o /tmp/bandit_task_12949.json
```

- [ ] **Step 4: Review the final diff**

Run `git diff --check`, inspect the branch diff against `origin/dev`, and
confirm no unrelated files are included.

- [ ] **Step 5: Finalize task and branch**

Update Backlog acceptance criteria, verification notes, final summary, and
status. Remove only this plan file, commit working changes, push the branch,
and open a PR against `dev` with the required human-authored Change summary
merge-gate reminder.
