# Stage 2 — The OCR backend family

## Scope

`core/Ingestion_Media_Processing/OCR/` — nine sibling backends behind one abstract interface, plus
the registry and the PDF page loop that drives them. This sub-package is the sharpest instance in the
module of the repo-wide pattern "N siblings, each rolling its own copy of the same five-line helper,
already drifted". It is 4,314 LOC — 8% of the module — and produces a disproportionate share of this
ledger's findings because the drift is measurable rather than stylistic.

Explicitly checked against `Docs/ADR/024-deepseek-ocr-local-transformers-backend.md` before
asserting: nothing below proposes changing the `deepseek` backend's identity, Transformers-only
ownership, `trust_remote_code=True` posture, availability gates, registry priority, or opt-in result
persistence.

## Code Paths Reviewed

Interface and shared infrastructure:
- `OCR/base.py:OCRBackend (11-54)` — holds `name`, `available()`, `ocr_image()`, and one default
  `ocr_image_structured()` (32-54). Carries **no shared helper behaviour at all**.
- `OCR/registry.py:_env_flag (77-81)`, `:_is_auto_eligible (84-93)`, `:get_backend (95-187)`.
- `OCR/runtime_support.py:_parse_bool (17-25)`, `:_parse_int (28-37)` — the file's own docstring (:1)
  calls it "Shared helpers for OCR runtime configuration"; no backend imports from it.
- `OCR/types.py` (`OCRResult`, `OCRBlock`, `OCRTable`, `normalize_ocr_format`).

Backends (nine): `chatllm_ocr.py` (560), `llamacpp_ocr.py` (636), `dolphin_ocr.py` (558),
`nemotron_parse.py` (494), `hunyuan_ocr.py` (453), `points_reader.py`, `dots_ocr.py`,
`deepseek_ocr.py`, `tesseract_cli.py`.

Per-backend helpers read line by line:
- `chatllm_ocr.py:_env_bool (51-55)`, `:_env_int (58-65)`, `:_resolve_mode (89)`,
  `:_resolve_prompt (106)`, managed-start gate `:263`, `:347`, describe payload `:461-468`.
- `llamacpp_ocr.py:_env_bool (60-64)`, `:_env_int (67-74)`, `:_resolve_prompt (145)`,
  remote call `:250-289` (temp-file lifetime), describe payload `:517-518`, `:590-591`.
- `deepseek_ocr.py:_env_bool (42-46)`, `:_env_int (49-56)`, `:_resolve_device (59)`,
  `:_resolve_prompt (97)`, `:_load_transformers (241-290)`.
- `nemotron_parse.py:_resolve_mode (55-59)`, `:_env_bool (62-66)`,
  `:_resolve_skip_special_tokens (69-78)`, `:_try_parse_json (314-323)`,
  `:_ocr_via_vllm (339-395)` incl. nested `_getf (356-360)` and inline truthy lambda `:378`,
  `:_load_transformers (398-438)`, `:_ocr_via_transformers (440-…)` incl. `_getf (455)`, `:467`.
- `dolphin_ocr.py:ocr_image (145-147)`, `:ocr_image_structured (149-210)`,
  `:_resolve_prompt (214-229)`, `:_resolve_json_prompt (231-247)`, `:_run_prompt (250-…)`,
  `:_getf (463-467)`, `:_bool_env (480-484)`, `:_try_parse_json (496-525)`,
  `:_extract_text_from_parsed (527-…)`, `:_ocr_via_transformers (402-…)` incl. `:430`.
- `hunyuan_ocr.py:_resolve_mode (36)`, `:_resolve_prompt (160)`,
  `:_ocr_via_vllm (177-228)` incl. `_getf (198-202)` and inline truthy `:185`,
  `:_load_transformers (232-…)`, `:_ocr_via_transformers (266-…)` incl. `:291`, `:303`,
  `:_build_result_from_output (…-350)`, `:_try_parse_json (353-383)`,
  `:_extract_text_from_parsed (386-414)`, `:_fill_blocks_tables_from_parsed (416-453)`.
- `dots_ocr.py:_ocr_via_vllm (181-229)` incl. `_getf (201-205)` and inline truthy `:187`, `:223`.
- `points_reader.py:_resolve_mode (34)`, `:_ocr_via_sglang (148-198)` incl. `_getf (151-155)`,
  `:_load_transformers (197-…)`, `:_ocr_via_transformers (233-…)` incl. `_getf (246)`.
- `tesseract_cli.py:ocr_image (24-42)`.

Consumer:
- `PDF/PDF_Processing_Lib.py:_ocr_pdf_pages (1552-1675)` — the page loop that decides
  `ocr_image` vs `ocr_image_structured` (1575-1581) and dispatches into a thread pool (1608-1648).

## Tests Reviewed

Found by import-grep (`grep -rl "Ingestion_Media_Processing\.OCR" tldw_Server_API/tests`) — 16 files:

| Test file | What it protects | Downgrades risk? |
| --- | --- | --- |
| `tests/unit/test_ocr_types.py` | `normalize_ocr_format`, `OCRResult.as_dict`, base-class structured fallback, **and one real behavioural test of hunyuan's JSON path** (`test_hunyuan_json_parse_builds_blocks`, :52, via `_build_result_from_output`) | Partly — it pins the happy path of one of the three JSON parsers |
| `tests/Media_Ingestion_Modification/test_ocr_adapter.py` | Registry name resolution for each backend, incl. `nemotron_parse` (:50-54) | No — resolution only, never `ocr_image` |
| `tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py` | `auto`/`auto_high_quality` ordering; monkeypatches `NemotronParseBackend.available`, `HunyuanOCRBackend.available`, `DolphinOCRBackend.available` at ~30 sites | No — `available()` is always stubbed; no backend body runs |
| `test_ocr_runtime_support.py`, `test_ocr_runtime_discovery.py`, `test_ocr_runtime_managed.py` | `runtime_support.py` profiles, managed-process registry, readiness probes | Yes, for `runtime_support.py` specifically |
| `test_chatllm_ocr_backend.py`, `test_llamacpp_ocr_backend.py`, `test_ocr_llamacpp_chatllm_pdf_pipeline.py` | chatllm + llamacpp end-to-end incl. managed lifecycle | Yes, for those two backends |
| `tests/MediaIngestion_NEW/test_ocr_backend_deepseek.py`, `_dots.py`, `_points.py` | deepseek / dots / points construction and availability | Partly |
| `test_ocr_structured_output.py` | PDF→structured-page attachment through `PDF_Processing_Lib` | Yes, for the page loop |
| `test_ocr_endpoint_error_mapping.py`, `tests/Workflows/adapters/test_media_adapters.py`, `tests/Evaluations/integration/test_ocr_pdf_deepseek_backend_local_integration.py` | endpoint error mapping; adapter wiring; gated live deepseek integration | Partly |

**Gap that matters:** `dolphin`, `hunyuan` and `nemotron_parse` have **no test that executes any of
their bodies**. They appear in tests only as `available()` monkeypatch targets, with the single
exception of `test_hunyuan_json_parse_builds_blocks`. Every finding below against those three files
is therefore un-ratcheted: a fix could regress silently. This is import-grep reachability, not
measured coverage.

## Validation Commands

```
$ python -m pytest tldw_Server_API/tests/unit/test_ocr_types.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_adapter.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_structured_output.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py --collect-only -q
========================= 19 tests collected in 0.99s ==========================

$ python -m pytest tldw_Server_API/tests/unit/test_ocr_types.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_adapter.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py -q
======================== 24 passed, 7 warnings in 0.93s ========================

$ grep -rn "def _getf" tldw_Server_API/app/core/Ingestion_Media_Processing/OCR/backends
dots_ocr.py:201, nemotron_parse.py:356, nemotron_parse.py:455, hunyuan_ocr.py:198,
dolphin_ocr.py:463, points_reader.py:151, points_reader.py:246          (7 definitions)

$ grep -rn "def _load_transformers" .../OCR/backends
dolphin_ocr.py:356, nemotron_parse.py:398, hunyuan_ocr.py:232, deepseek_ocr.py:241,
points_reader.py:197                                                    (5 definitions)

# ast-extract + difflib over the named function bodies (2026-09-21):
try_parse_json   hunyuan == dolphin : False — differs ONLY by 2 comment lines (926 vs 838 chars)
extract_text     hunyuan == dolphin : False — 4 behavioural differences (diff below)
_env_bool        chatllm == llamacpp: True  — byte-identical
_env_int         chatllm == llamacpp: True  — byte-identical

$ grep -rn "delete=True" .../OCR/backends
dots_ocr.py:196, hunyuan_ocr.py:193, tesseract_cli.py:27
$ grep -rn "delete=False" .../OCR/backends
nemotron_parse.py:350, llamacpp_ocr.py:259, llamacpp_ocr.py:392
```

## Findings

### FINDING ingestion-media-processing-1 — three mutually incompatible definitions of "true" across nine sibling OCR backends

```
axis:        duplication
class:       divergent-copies
severity:    High
sites:       SET A {1,true,yes,y,on}, strip+lower — delegating to core/testing.py:30 is_truthy:
               OCR/backends/deepseek_ocr.py:_env_bool (42-46)
               OCR/backends/nemotron_parse.py:_env_bool (62-66)
               Audio/Audio_Transcription_Qwen3ASR.py:_as_bool (67-79)
               Audio/Audio_Transcription_VibeVoice.py:_as_bool (75-86)
               Audio/Audio_Custom_Vocabulary.py:_as_bool (41-50)
               Audio/Audio_Transcription_Lib.py:260
               Audio/Audio_Streaming_Unified.py:2671, :2738, :2957, :2967
               Audio/Audio_Files.py:_as_bool (215-220)  [Set A written inline, not via is_truthy]
             SET B {1,true,yes,on}, strip+lower — hand-written, `y` rejected:
               OCR/backends/chatllm_ocr.py:_env_bool (51-55)
               OCR/backends/llamacpp_ocr.py:_env_bool (60-64)   [byte-identical to chatllm]
               OCR/registry.py:_env_flag (77-81)
               OCR/runtime_support.py:_parse_bool (17-25)
               PDF/mineru_adapter.py:54
               chunking_options.py:93
               Audio/Audio_Streaming_Unified.py:1578
             SET C ("1","true","yes"), lower() ONLY — no strip, `on` and `y` both rejected:
               OCR/backends/dolphin_ocr.py:_bool_env (480-484)  and inline at :430
               OCR/backends/nemotron_parse.py:378, :467          [in the SAME FILE that imports is_truthy]
               OCR/backends/hunyuan_ocr.py:185, :291, :303
               OCR/backends/dots_ocr.py:187, :223
               OCR/backends/points_reader.py:149, :182, :258
             Full inventory: 2026-09-21-stage2-truthy-divergence.txt
canonical:   core/testing.py:30 `is_truthy` — ALREADY IMPORTED by six production files in this module
             (deepseek_ocr, nemotron_parse, Audio_Transcription_Lib, Audio_Transcription_Qwen3ASR,
             Audio_Transcription_VibeVoice, Audio_Custom_Vocabulary, Audio_Streaming_Unified).
             core/MCP_unified/environment.py:17 is a deliberate copy for the standalone MCP package
             boundary (its own docstring says so) — justified-divergence, leave it.
destination: The canonical function exists but lives in a module named `testing.py` whose docstring
             opens "Lightweight helpers for test-mode detection". That is why nine sibling files did
             not find it. Proposal: create `app/core/Utils/coercion.py` with ONE responsibility —
             scalar/environment coercion — exporting `coerce_bool(value, default)`,
             `coerce_int(value, default)`, `coerce_float(value, default)`, `env_bool(name, default)`;
             re-export from `core/testing.py` for the existing importers.
             EXPLICITLY NOT `Utils/Utils.py` and NOT `http_client.py`.
knowledge:   "which spellings of an environment variable mean yes." Three answers today.
scenario:    (a) An operator sets `DOTS_VLLM_USE_DATA_URL=true ` — one trailing space, the normal
             result of a `docker-compose` `environment:` list or a hand-edited `.env`. Set C does not
             strip, so `"true "` fails the membership test and the flag reads False. dots_ocr falls
             to the else branch at :195-198, writes the page PNG to a temp file, and sends the
             *server-local filesystem path* as the `image_url` to a remote vLLM endpoint that cannot
             read it. OCR returns empty text for every page, with no error.
             (b) `CHATLLM_OCR_AUTO_ELIGIBLE=y` is False (Set B), while the same spelling in
             `DEEPSEEK_OCR_SAVE_RESULTS=y` is True (Set A) — the same operator convention silently
             enables one backend's feature and not its sibling's.
             (c) `DOLPHIN_DISABLE_JSON=on` is False (Set C), while `CHATLLM_OCR_ALLOW_MANAGED_START=on`
             is True (Set B). See finding -2: that specific mis-read costs a doubled inference bill.
impact:      High. These are not cosmetic: two of the three sets gate remote-vs-local image transport
             and managed-process startup, and Set C — the weakest — guards eleven of them, including
             the only opt-out from finding -2. The failures are all silent; nothing logs a rejected
             spelling.
tests:       tests/Media_Ingestion_Modification/test_ocr_runtime_support.py covers
             runtime_support._parse_bool indirectly via profile loading. No test exercises any
             backend's `_env_bool`/`_bool_env` directly. dolphin/hunyuan/nemotron/dots/points have no
             behavioural tests at all (see Tests Reviewed).
effort:      cheap per call site (mechanical substitution), moderate overall because ~10 of the 30
             sites are in files with no behavioural test, so each needs a characterisation test first.
             Sequence the well-covered backends (chatllm, llamacpp, deepseek) first.
owner-only:  no
confidence:  confirmed — every site read and the three sets verified by reading the literal.
```

### FINDING ingestion-media-processing-2 — `dolphin` runs two model inferences per page and the only way to turn the second one off is guarded by the weakest coercer in the module

```
axis:        efficiency
class:       n/a
severity:    High
sites:       OCR/backends/dolphin_ocr.py:ocr_image_structured (149-210) — two `_run_prompt` calls at
               :165 and :170.
             OCR/backends/dolphin_ocr.py:ocr_image (145-147) — delegates to ocr_image_structured, so
               even the plain-text entry point pays both.
             OCR/backends/dolphin_ocr.py:_resolve_json_prompt (231-247) — returns
               `_PROMPT_PRESETS["json"]` on EVERY path; the `preset == "json"` branch (:240-241) and
               the `fmt == "json"` branch (:243-244) return exactly what the unconditional
               fallthrough at :246 returns, so neither can change the result.
             OCR/backends/dolphin_ocr.py:_bool_env (480-484) — the `DOLPHIN_DISABLE_JSON` gate at
               :232, using truthy Set C from finding -1.
             PDF/PDF_Processing_Lib.py:_ocr_pdf_pages (1575-1581) — prefers `ocr_image_structured`
               whenever a backend overrides it, which dolphin does, so the PDF path always takes the
               2× route.
canonical:   NONE
destination: n/a — this is a local logic fix, not a consolidation.
knowledge:   n/a
cost-driver: model inferences = **2 × page_count** instead of `page_count`, for every output format.
             Scales linearly with PDF page count and is the dominant cost in the pipeline: a
             Transformers-local dolphin run (`_ocr_via_transformers`, :402) is a full VLM generate
             per call; a remote run (`_ocr_via_openai` / `_ocr_via_vllm`) is a full round trip plus
             a second base64 image upload of the same pixmap. The second result is written only to
             `result.raw` (:192-196) unless `fmt == "json"` (:197-199), so on the default
             `output_format="markdown"` path 50% of the inference budget produces a field most
             callers discard. On a 200-page scanned PDF that is 200 wasted VLM generates.
scenario:    An operator who notices the doubled cost and sets `DOLPHIN_DISABLE_JSON=on` — the
             spelling that works for `CHATLLM_OCR_ALLOW_MANAGED_START` — gets no effect at all,
             because `_bool_env` (:484) accepts only `("1","true","yes")`. The doubling silently
             persists.
tests:       none. dolphin has no test that executes any of its methods.
effort:      cheap — three changes: make `_resolve_json_prompt` return "" unless JSON output is
             actually requested (collapsing its three equal branches), route `_bool_env` through the
             shared coercer from finding -1, and add a characterisation test asserting one
             `_run_prompt` call for `output_format="markdown"`.
owner-only:  no
confidence:  confirmed (both `_run_prompt` calls, the three-equal-branch function, and the Set C
             gate were read at those lines); probable-risk (the exact per-page cost depends on the
             deployment mode, which I did not execute).
```

### FINDING ingestion-media-processing-3 — `dots` and `hunyuan` delete the page image before sending its path to the model

```
axis:        correctness
class:       divergent-copies
severity:    High
sites:       BROKEN — file closed and unlinked by the `with` block before the request is built:
               OCR/backends/dots_ocr.py:_ocr_via_vllm (195-198), request sent at :225-226
               OCR/backends/hunyuan_ocr.py:_ocr_via_vllm (192-195), request sent at :222-224
             CORRECT — `delete=False` plus a `finally: os.unlink`:
               OCR/backends/nemotron_parse.py:_ocr_via_vllm (350-354) / unlink at :384-387
               OCR/backends/llamacpp_ocr.py (259-262) / unlink at :287-289
               OCR/backends/llamacpp_ocr.py (392-…) — same shape
             CORRECT — consumer runs inside the `with`:
               OCR/backends/tesseract_cli.py:ocr_image (27-42)
             NOT AFFECTED — reads an existing path rather than writing one:
               OCR/backends/points_reader.py:_ocr_via_sglang (148-165)
canonical:   NONE. Five siblings solve the same "hand a local image path to a model server" problem
             five ways; `nemotron_parse.py:350-387` and `llamacpp_ocr.py:259-289` are the correct
             copies and should be promoted.
destination: `OCR/runtime_support.py` — already declared "Shared helpers for OCR runtime
             configuration, managed processes, and readiness probes" and already imported by the
             managed backends. Add one context manager, `image_payload(image_bytes, use_data_url)`,
             yielding either a `data:` URL or a path whose lifetime spans the request.
knowledge:   "how long the temp image must outlive the block that creates it." Encoded five times.
scenario:    Operator sets `DOTS_VLLM_USE_DATA_URL=0` (a documented knob — the code comment at
             dots_ocr.py:193 anticipates exactly this deployment: a vLLM server on the same host
             that can read local paths). `NamedTemporaryFile(delete=True)` closes and unlinks at the
             `with` exit on :198; `content_image` still holds `f.name`. The POST at :225 therefore
             names a path that no longer exists. The server returns an error or an empty completion;
             `_ocr_via_vllm` returns `"" ` via the `or ""` at :229, and the page is recorded as
             "OCR produced no text" rather than as a failure. Identical for hunyuan at :195/:222.
             Every page of every document silently OCRs to empty text for as long as the flag is set.
impact:      High. Silent total data loss on a supported configuration, with no error surfaced —
             `_ocr_pdf_pages` (PDF_Processing_Lib.py:1652-1655) only counts a page as OCR'd if the
             text is non-empty, so the run completes "successfully" with zero content.
tests:       tests/MediaIngestion_NEW/test_ocr_backend_dots.py exists but does not execute
             `_ocr_via_vllm`. hunyuan has no behavioural test beyond
             tests/unit/test_ocr_types.py:52, which calls `_build_result_from_output` directly.
effort:      cheap — two files, two lines each, following the shape already present in
             `nemotron_parse.py`. Add one test per backend asserting the file exists at request time.
owner-only:  no
confidence:  confirmed — `delete=True` and the `with` scope read at both sites; the consuming
             `fetch_json` call is demonstrably outside the block in both files.
```

### FINDING ingestion-media-processing-4 — three copies of `_try_parse_json`, and the one that runs against the noisiest model output is the weakest

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       OCR/backends/hunyuan_ocr.py:_try_parse_json (353-383) — full-parse, then salvage the
               outermost {...}, then salvage the outermost [...]
             OCR/backends/dolphin_ocr.py:_try_parse_json (496-525) — identical to hunyuan apart from
               two comment lines (verified by ast-extract + difflib: 926 vs 838 characters, the
               only diff hunks being the removal of `# Try full string first` and
               `# Try to extract a JSON object or array from the output`)
             OCR/backends/nemotron_parse.py:_try_parse_json (314-323) — same name, WEAKER: it
               returns None immediately unless the stripped text literally starts with `{` or `[`
               (:318-319), and it performs no substring salvage at all.
             Call sites: hunyuan_ocr.py:341, dolphin_ocr.py:171, nemotron_parse.py:248.
canonical:   NONE. hunyuan/dolphin is the correct copy — it is the one that survives the output shape
             VLMs actually emit.
destination: `OCR/types.py`, which already owns `OCRResult`/`OCRBlock`/`normalize_ocr_format` and is
             imported by every backend that needs it. Add `parse_model_json(raw: str) -> Any | None`
             there. Do NOT put it in `Utils/Utils.py`.
knowledge:   "how to recover a JSON document from a chat model's free-text answer." The three copies
             disagree on whether prose or a code fence around the JSON is recoverable.
scenario:    `nemotron_parse.ocr_image` requests structured output; the vLLM server returns
             "```json\n{\"text\": \"Invoice 1041\", \"blocks\": [...]}\n```" — the single most
             common shape for an instruction-tuned model asked for JSON, and the shape the other two
             copies were specifically extended to handle. `_try_parse_json` at :318 sees the leading
             backtick, fails `startswith(("{","["))`, and returns None without attempting the parse.
             The same output fed to hunyuan's or dolphin's copy is recovered by the `txt.find("{")`
             / `txt.rfind("}")` salvage at :371 / :513. Result: nemotron drops the structured payload
             and falls through to `_strip_tags` at :326, producing plain text with no blocks and no
             bboxes, while its two siblings return a populated OCRResult from byte-identical input.
             A leading "Here is the JSON:" produces the same divergence.
impact:      Medium — degraded output rather than a crash, and confined to nemotron's structured
             path. Raised above Low because the whole point of selecting `nemotron_parse` over
             `tesseract` is the structured bbox output the weak parser discards.
tests:       tests/unit/test_ocr_types.py:52 covers the hunyuan copy's happy path only (clean JSON,
             no fence). No test for dolphin's or nemotron's copy.
effort:      cheap — promote one function to `OCR/types.py`, delete three; add table-driven tests for
             fenced, prose-prefixed, and bare inputs. Well worth doing before finding -5.
owner-only:  no
confidence:  confirmed — all three bodies extracted by AST and diffed.
```

### FINDING ingestion-media-processing-5 — `_extract_text_from_parsed` has already drifted four ways between the two backends that share it

```
axis:        duplication
class:       divergent-copies
severity:    Medium
sites:       OCR/backends/hunyuan_ocr.py:_extract_text_from_parsed (386-414)
             OCR/backends/dolphin_ocr.py:_extract_text_from_parsed (527-…)
             Related, single-copy-but-parallel: hunyuan_ocr.py:_fill_blocks_tables_from_parsed
               (416-453) — dolphin has no equivalent, so dolphin never populates
               `OCRResult.blocks` or `.tables` even when the model returns them.
canonical:   NONE; neither copy is strictly better. hunyuan handles bare-string blocks and a scalar
             root; dolphin handles the `elements` key and a `content` text key. The union is correct.
destination: same as finding -4 — `OCR/types.py`, alongside `parse_model_json`.
knowledge:   "which keys in a model's JSON answer hold the page text." Four disagreements already:
             1. dolphin also accepts `parsed["elements"]` as a block list; hunyuan does not.
             2. hunyuan accepts a list of bare strings inside `blocks`; dolphin skips non-dict items,
                so `{"blocks": ["line one", "line two"]}` yields "" from dolphin and
                "line one\nline two" from hunyuan.
             3. dolphin falls back to `item["content"]` when `item["text"]` is absent; hunyuan does not.
             4. Root fallback: hunyuan `return str(parsed)`; dolphin `return ""`. hunyuan also filters
                empty parts (`[p for p in parts if p]`); dolphin joins them raw, so dolphin can emit
                leading/trailing blank lines.
scenario:    Model returns `{"blocks": ["Invoice 1041", "Total 91.20"]}`. hunyuan yields
             "Invoice 1041\nTotal 91.20". dolphin's copy yields "" — mitigated, not fixed, by the
             `or raw_json_text or primary_text` fallback chain at dolphin_ocr.py:199, which
             substitutes the *raw markdown* from the first inference. So the caller gets text, but
             not the structured text it asked for, and no warning says so.
impact:      Medium. Lower than finding -4 because dolphin's fallback chain hides the empty result,
             but the drift is live and undetected: two functions with the same name in sibling files
             now answer the same question differently, and a future fix applied to one will not
             reach the other.
tests:       none for either copy.
effort:      cheap once finding -4 lands — the two belong in the same move.
owner-only:  no
confidence:  confirmed — both bodies extracted by AST and diffed; the four differences are the
             literal diff hunks.
```

### FINDING ingestion-media-processing-6 — the same seven-line env reader is written seven times, five of them as closures rebuilt on every OCR call

```
axis:        duplication
class:       true-duplication
severity:    Low
sites:       `_getf(env, cast, default)` — body byte-identical at all seven (only `dolphin_ocr.py`
             and `hunyuan_ocr.py` annotate `env: str`):
               dolphin_ocr.py:_getf (463-467)        [module level]
               hunyuan_ocr.py:_ocr_via_vllm._getf (198-202)
               nemotron_parse.py:_ocr_via_vllm._getf (356-360)
               nemotron_parse.py:_ocr_via_transformers._getf (455-459)
               dots_ocr.py:_ocr_via_vllm._getf (201-205)
               points_reader.py:_ocr_via_sglang._getf (151-155)
               points_reader.py:_ocr_via_transformers._getf (246-250)
             `_load_transformers()` — five structurally identical double-checked-locking singletons,
             differing only in which module globals they populate and which HF class they call:
               deepseek_ocr.py:241-290, dolphin_ocr.py:356-…, hunyuan_ocr.py:232-…,
               nemotron_parse.py:398-438, points_reader.py:197-…
               (all five correctly use a module `_TF_LOCK` with a re-check inside — no bug here)
             `_resolve_mode()` 5x: dolphin_ocr.py:37, nemotron_parse.py:55, chatllm_ocr.py:89,
               hunyuan_ocr.py:36, points_reader.py:34
             `_resolve_prompt()` 5x: dolphin_ocr.py:214, deepseek_ocr.py:97, hunyuan_ocr.py:160,
               chatllm_ocr.py:106, llamacpp_ocr.py:145
             `_env_int()` 3x, byte-identical between chatllm_ocr.py:58 and llamacpp_ocr.py:67:
               deepseek_ocr.py:49, chatllm_ocr.py:58, llamacpp_ocr.py:67
canonical:   `OCR/base.py:OCRBackend (11-54)` is the designated shared home and holds only the
             interface. `OCR/runtime_support.py` is the designated shared *helper* home
             (its docstring, :1) and no backend imports from it.
destination: `OCR/runtime_support.py` for `env_str/env_int/env_float` and a
             `cached_transformers(key, factory)` helper; `OCR/base.py` gains a small
             `PromptPresetMixin` for the `_resolve_mode`/`_resolve_prompt` pair. The env coercers
             should delegate to the coercion module from finding -1 rather than re-deriving.
knowledge:   "read an env var, cast it, fall back to the default on a bad value" and "load a heavy
             HF model once per process under a lock." The second is the expensive one to get wrong.
impact:      Low on its own — the copies currently agree. Raised to a finding rather than dropped
             because it is the *delivery mechanism* for finding -1: five of the seven `_getf` copies
             exist only to be handed the inline truthy lambda that constitutes truthy Set C. Remove
             `_getf` and Set C has nowhere to live. Also a real, if small, cost driver: five of the
             seven are nested `def`s, so a new function object is compiled-in and bound on every
             single `ocr_image` call — that is once per PDF page.
tests:       covered indirectly for chatllm/llamacpp/deepseek/dots/points; not at all for
             dolphin/hunyuan/nemotron.
effort:      cheap for `_getf`/`_env_int`; moderate for `_load_transformers` (five distinct model
             classes and global sets — do it only if the loaders start to drift, which they have
             not; ADR-024 also pins deepseek's loader behaviour, so that one must keep its
             `use_safetensors` fallback).
owner-only:  no
confidence:  confirmed for every count and for byte-identity where claimed.
```

### FINDING ingestion-media-processing-7 — auto-eligibility for two backends is computed twice, by two different functions

```
axis:        duplication
class:       divergent-copies
severity:    Low
sites:       Routing decision: OCR/registry.py:_env_flag (77-81) used by
               OCR/registry.py:_is_auto_eligible (84-93) for LLAMACPP_OCR_AUTO_ELIGIBLE,
               LLAMACPP_OCR_AUTO_HIGH_QUALITY_ELIGIBLE, CHATLLM_OCR_AUTO_ELIGIBLE,
               CHATLLM_OCR_AUTO_HIGH_QUALITY_ELIGIBLE.
             Advertised metadata for the same four variables, computed independently:
               OCR/backends/chatllm_ocr.py:461-469 via chatllm_ocr.py:_env_bool (51-55)
               OCR/backends/llamacpp_ocr.py:517-519 and :590-592 via llamacpp_ocr.py:_env_bool (60-64)
canonical:   NONE — three functions, all currently truthy Set B, so they agree today.
destination: `OCR/registry.py` should own the answer and the backends' `describe()` should call it,
             rather than each side reading the env independently.
knowledge:   "is this backend eligible for `auto` selection." Two sources of truth for one question.
impact:      Low today because all three coercers happen to be Set B. It becomes a real defect the
             moment finding -1 is fixed backend-by-backend: migrate `chatllm_ocr._env_bool` to the
             shared Set A coercer and leave `registry._env_flag` alone, and
             `GET /api/v1/ocr/backends` starts reporting `auto_eligible: true` for
             `CHATLLM_OCR_AUTO_ELIGIBLE=y` while the registry still refuses to route to it. That is
             a diagnostics surface that lies, which is worse than the original inconsistency.
tests:       tests/Media_Ingestion_Modification/test_ocr_runtime_auto_selection.py covers registry
             ordering; tests/Media_Ingestion_Modification/test_chatllm_ocr_backend.py and
             test_llamacpp_ocr_backend.py cover the describe payloads. Neither asserts the two agree.
effort:      cheap — one call-direction change plus one test asserting
             `registry._is_auto_eligible(name) == backend.describe()["auto_eligible"]`.
owner-only:  no
confidence:  confirmed (the two computation paths); probable-risk (the divergence, which is
             contingent on finding -1 being fixed piecemeal — which is exactly how it will be fixed).
```

## Suggested Refactor/Actions

Ordered so that each step de-risks the next. Steps 1–3 are small enough not to need a design doc;
step 4 needs one.

1. **Characterisation tests first.** `dolphin`, `hunyuan` and `nemotron_parse` currently have no test
   that runs their code. Before touching any of findings -2 through -5, add one test file per backend
   that stubs `fetch_json` and asserts the request payload and the parsed result. Without this, every
   fix below is unratcheted. Cheap, and it is the gate that makes the rest cheap.
2. **Fix the two temp-file bugs (finding -3)** — two lines each, copying the `delete=False` +
   `finally: os.unlink` shape already correct in `nemotron_parse.py:350-387`. Highest
   severity-to-effort ratio in this ledger.
3. **Collapse dolphin's second inference (finding -2)** — make `_resolve_json_prompt` return `""`
   unless JSON output was actually requested. Its three current branches all return the same value,
   so there is no behaviour to preserve on the JSON path.
4. **Coercion consolidation (finding -1)** needs
   `Docs/Design/2026-MM-DD-scalar-coercion-consolidation-design.md`, an ADR entry (it fixes a
   cross-module policy: what an env flag means), a Backlog task linking both, and
   `IMPLEMENTATION_PLAN_scalar-coercion.md` with staged goals. This finding is one module's slice of
   a repo-wide cluster (~270 re-implementations); the design must be written once at repo scope, not
   per module. Destination must be a new cohesive `app/core/Utils/coercion.py`, never
   `Utils/Utils.py` and never `http_client.py`. Sequence within this module: chatllm → llamacpp →
   deepseek (all three have behavioural tests) before the untested three.
5. **JSON salvage (findings -4, -5) moves as one change** into `OCR/types.py`, taking the union of
   the hunyuan and dolphin key handling and hunyuan's `_fill_blocks_tables_from_parsed`. Gate on
   step 1.
6. Findings -6 and -7 are opportunistic. Do `_getf` as part of step 4 (it has no independent value)
   and `_is_auto_eligible` as part of step 4's chatllm/llamacpp slice, because that is exactly when
   it would otherwise break.
