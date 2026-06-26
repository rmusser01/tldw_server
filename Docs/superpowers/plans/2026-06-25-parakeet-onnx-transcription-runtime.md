## Stage 1: Reproduce And Trace
**Goal**: Prove the Parakeet ONNX transcription error comes from loading the wrong ONNX graph and feeding it the wrong inputs.
**Success Criteria**: Direct execution shows the current loader picks `decoder_joint-model.onnx` and raises the missing `encoder_outputs` / decoder-state input error.
**Tests**: Direct local ONNX Runtime probe against the cached Parakeet ONNX export.
**Status**: Complete

## Stage 2: Red Tests
**Goal**: Capture the broken contract in focused automated tests.
**Success Criteria**: Tests fail on current code because multi-graph Parakeet exports are not loaded or dispatched as a bundle.
**Tests**: Unit tests for sidecar download patterns, multi-graph loader selection, and transcription dispatch to a multi-graph runner.
**Status**: Complete

## Stage 3: Multi-Graph Runner
**Goal**: Replace first-file ONNX loading for Parakeet TDT exports with a real preprocessor + encoder + decoder/joint runner.
**Success Criteria**: Loader detects the `nemo128.onnx`, `encoder-model*.onnx`, `decoder_joint-model*.onnx`, and `vocab.txt` layout; transcription no longer sends waveform data to decoder `targets`.
**Tests**: Red tests from Stage 2 pass; existing Parakeet ONNX tests remain green.
**Status**: Complete

## Stage 4: Verification
**Goal**: Verify the fix with automated tests and a no-mock local ONNX Runtime smoke.
**Success Criteria**: Targeted tests pass, `git diff --check` passes, Bandit has no new findings, and a direct cached-graph smoke returns a normal transcript string or `[No speech detected]` instead of a runtime input-feed error.
**Tests**: Pytest, Bandit, direct ONNX Runtime smoke.
**Status**: Complete

Verification recorded:
- `python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py -q`
- `python -m pytest tldw_Server_API/tests/Audio/test_parakeet_onnx_failfast.py tldw_Server_API/tests/Audio/test_transcription_model_parsing.py tldw_Server_API/tests/Audio/test_model_variant_normalization.py -q`
- `git diff --check`
- `python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py -f json -o /tmp/bandit_parakeet_onnx_runtime.json`
- No-mock upstream smoke against local cached Parakeet graphs loaded through `onnx-asr`; silence returned `[No speech detected]`, and an artificial sine tone returned `Oh.` without the decoder missing-input error.
- `python -m pip check` still reports the existing local venv conflict: `typer-slim 0.24.0` requires `typer>=0.24.0`, but the venv has `typer 0.16.1`.
- `python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py -f json -o /tmp/bandit_parakeet_onnx_runtime.json`
- No-mock ONNX Runtime smoke against local cached Parakeet graphs using silence and sine inputs; both returned `[No speech detected]` without the decoder missing-input error.

## Stage 5: Upstream Runtime Adapter
**Goal**: Replace the local greedy TDT decoder with the upstream `onnx-asr` implementation for Parakeet TDT graph bundles.
**Success Criteria**: Bundle detection routes to an `onnx_asr.load_model("nemo-conformer-tdt", path=...)` adapter; missing `onnx-asr` fails with a clear dependency error; generic single-session ONNX fallback remains available only for non-bundle exports.
**Tests**: Red tests for upstream adapter selection, adapter transcription, and missing-dependency failure.
**Status**: Complete

Verification recorded:
- `python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py -q`
- `python -m pytest tldw_Server_API/tests/Audio/test_parakeet_onnx_failfast.py tldw_Server_API/tests/Audio/test_transcription_model_parsing.py tldw_Server_API/tests/Audio/test_model_variant_normalization.py -q`
- `git diff --check`
- `python -m bandit -r tldw_Server_API/app/core/Ingestion_Media_Processing/Audio/Audio_Transcription_Parakeet_ONNX.py -f json -o /tmp/bandit_parakeet_onnx_runtime.json`
- Installed `onnx-asr[hub]` 0.11.0 into the venv and ran a no-mock upstream smoke against local cached Parakeet graphs; silence and sine returned `[No speech detected]`.
- `python -m pip check` reports an existing local venv conflict: `typer-slim 0.24.0` requires `typer>=0.24.0`, but the venv has `typer 0.16.1`.

## Stage 6: PR Review Rebase
**Goal**: Rebase PR #2524 onto current `dev` and address active review findings without reintroducing local TDT decoding.
**Success Criteria**: Existing local Parakeet artifact directories are never treated as Hugging Face repo ids, missing bundle `vocab.txt` fails closed with a clear log, upstream bundle chunking respects `merge_algo="middle"`, and new tests have explicit type hints.
**Tests**: Red/green pytest coverage for local-path download prevention and upstream bundle middle merging; focused Parakeet and adjacent audio tests.
**Status**: Complete

Verification recorded:
- `git rebase --autostash origin/dev`
- `python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py -q -k "local_parakeet_bundle_missing_vocab or upstream_bundle_chunking_respects_middle_merge"` failed before the fix and passed after it.
- `python -m pytest tldw_Server_API/tests/Media_Ingestion_Modification/test_parakeet_onnx.py -q`
- `python -m pytest tldw_Server_API/tests/Audio/test_parakeet_onnx_failfast.py tldw_Server_API/tests/Audio/test_transcription_model_parsing.py tldw_Server_API/tests/Audio/test_model_variant_normalization.py -q`
- `git diff --check`
