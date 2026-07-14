# Parakeet ONNX Bundle Metadata Validation Design

## Problem

The Parakeet TDT v3 encoder accepts 128 mel features. `onnx-asr` selects its
NumPy preprocessor from `config.json`, but defaults to 80 features when that
file is absent. The current loader considers graphs plus `vocab.txt` a complete
bundle and refreshes remote caches only when `vocab.txt` is missing. An
incomplete cache can therefore load successfully and fail only when audio is
sent to the encoder.

## Approaches Considered

1. Require the upstream bundle metadata and validate it before model load.
   This is the smallest fix and keeps `onnx-asr` as the source of preprocessing
   behavior.
2. Infer the feature count from `nemo128.onnx` or encoder input metadata and
   synthesize configuration. This adds a second configuration path and risks
   masking other missing metadata.
3. Bypass `onnx-asr` preprocessing and run the bundled preprocessor graph
   directly. This duplicates upstream orchestration and expands the change far
   beyond the cache-completeness bug.

Approach 1 is selected.

## Design

- Treat `config.json` as a required Parakeet TDT bundle sidecar alongside the
  graphs and `vocab.txt`.
- For Hugging Face model IDs, rerun `snapshot_download` when either required
  sidecar is missing so existing stale caches self-heal.
- For explicit local directories, fail closed with an actionable log message
  when required metadata is missing.
- Parse `config.json` before calling `onnx-asr` and require a positive integer
  (excluding booleans) `features_size`. Compare it with the `audio_signal`
  input's feature axis in the encoder graph selected by the same quantization
  decision passed to `onnx-asr` when that dimension is statically declared.
- Keep dynamic or unavailable encoder metadata compatible by limiting the
  graph check to declared positive integer dimensions.

## Error Handling

Invalid JSON, missing or invalid `features_size`, and a static encoder/config
dimension mismatch prevent the model from entering the runtime cache. The
loader logs the model directory and the incompatible values, then returns its
existing `(None, None)` failure result.

## Tests

- A remote cache containing vocab but no config triggers a sidecar refresh.
- A local graph bundle without config fails closed before `onnx-asr` is called.
- A valid 128-feature config is passed through to `onnx-asr` successfully.
- Malformed JSON and missing, boolean, non-integer, zero, or negative
  `features_size` values fail closed before `onnx-asr` is called.
- A config/selected-encoder `audio_signal` feature mismatch fails closed before
  `onnx-asr` is called for both quantized and unquantized graph selection.

## Scope

No provider fallback, decoder changes, new dependencies, or unrelated audio
refactoring are included.
