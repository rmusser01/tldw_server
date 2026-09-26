# Stage 1 — Inventory and architecture survey

## Scope

Establish the shape of `tldw_Server_API/app/core/Ingestion_Media_Processing/` before reviewing it:
size, churn, package boundaries, where its tests actually live, and which ADRs bind it. No findings
are raised in this stage other than the two structural ones that only the inventory can support.

## Code Paths Reviewed

Package layout (82 `.py` files, 55,842 LOC). Top-level modules and seven sub-packages:

- Sub-packages: `Audio/` (33 files, ~22k LOC), `OCR/` (13 files, ~4.3k LOC), `PDF/` (2 files, 2,220),
  `Books/` (1,673), `Email/` (1,234), `MediaWiki/` (1,148), `Video/` (2 files, ~1.8k),
  `Plaintext/` (631), `VLM/` (5 files, 430).
- Top-level orchestration: `persistence.py` (6,390), `Upload_Sink.py` (1,273),
  `chunking_options.py` (686), `download_utils.py` (625), `input_sourcing.py` (580),
  `pipeline.py`, `result_normalization.py`, `audio_batch.py`, `video_batch.py`,
  `Media_Update_lib.py`, `visual_ingestion.py`, `document_upload_drafts.py`,
  `document_upload_preflight.py`, `research_discovery_handoff.py`, `transcription_models.py`,
  `code_utils.py`, `path_utils.py`, `logging_safety.py`, `XML_Ingestion_Lib.py`, `yt_dlp_support.py`.

Size × churn hot set (full data in `2026-09-21-stage1-source-inventory.txt` and
`-churn-baseline.txt`):

| File | LOC | commits / 12mo |
| --- | ---: | ---: |
| `persistence.py` | 6,390 | 98 |
| `Audio/Audio_Transcription_Lib.py` | 5,027 | 82 |
| `Audio/Audio_Streaming_Unified.py` | 3,466 | 70 |
| `Audio/stt_provider_adapter.py` | 3,159 | 46 |
| `Audio/Audio_Files.py` | 2,002 | 49 |
| `Audio/Diarization_Lib.py` | 2,018 | 13 |
| `Video/Video_DL_Ingestion_Lib.py` | 1,733 | 30 |
| `PDF/PDF_Processing_Lib.py` | 1,699 | 28 |
| `Books/Book_Processing_Lib.py` | 1,673 | 15 |

Interface surfaces read in full:
`OCR/base.py:OCRBackend (11-54)`, `OCR/registry.py:get_backend (95-187)`,
`VLM/base.py:VLMBackend (33-68)`, `VLM/registry.py:get_backend (33-52)`,
`result_normalization.py:MediaItemProcessResponse (16-40)`,
`input_sourcing.py:TempDirManager (35-80)`.

## Tests Reviewed

**Located by import-grep, never by path** — `grep -rl "core\.Ingestion_Media_Processing"
tldw_Server_API/tests` returns **192 files** (full list in `2026-09-21-stage1-test-inventory.txt`).
This module is heavily tested; nothing in this ledger should be read as "untested".

Distribution across test trees:

| Tree | files |
| --- | ---: |
| `tests/MediaIngestion_NEW/` | 61 |
| `tests/Audio/` | 46 |
| `tests/Media_Ingestion_Modification/` | 30 |
| `tests/Media/` | 14 |
| `tests/Research/`, `tests/DB_Management/` | 5 each |
| `tests/VLM/`, `tests/Persona/` | 3 each |
| 21 further trees | 1–2 each |

Reverse index (which trees import a given source module) — the partition has no rule:

| Source module | distinct test trees importing it |
| --- | --- |
| `Audio/Audio_Transcription_Lib.py` | **11** — Audio, AudioJobs, DB_Management, MediaIngestion_NEW, Media_Ingestion_Modification, Resource_Governance, STT, Setup, TTS_NEW, VoiceAssistant, Workflows |
| `Audio/Audio_Files.py` | 7 |
| `PDF/PDF_Processing_Lib.py` | 5 |
| `OCR/backends/*` | 4 — Evaluations, MediaIngestion_NEW, Media_Ingestion_Modification, unit |
| `persistence.py` | 4 — Collections, DB_Management, MediaIngestion_NEW, Media_Ingestion_Modification |

No basename collides across the four main trees, so these are not duplicate suites — they are one
suite split along no discoverable axis.

## Validation Commands

```
$ find tldw_Server_API/app/core/Ingestion_Media_Processing -name '*.py' | wc -l
      82
$ find tldw_Server_API/app/core/Ingestion_Media_Processing -name '*.py' -exec wc -l {} + | tail -1
   55842 total
$ grep -rl "core\.Ingestion_Media_Processing" tldw_Server_API/tests | wc -l
     192
$ grep -rl "core\.Ingestion_Media_Processing" tldw_Server_API/tests \
    | sed 's|tldw_Server_API/tests/||' | cut -d/ -f1 | sort | uniq -c | sort -rn | head -4
  61 MediaIngestion_NEW
  46 Audio
  30 Media_Ingestion_Modification
  14 Media
$ grep -rn "pytest.mark.media_processing" tldw_Server_API/tests --include="*.py"
tldw_Server_API/tests/e2e/test_full_user_workflow.py:277:    @pytest.mark.media_processing
$ grep -rn "pytest.mark.pipeline" tldw_Server_API/tests --include="*.py" | wc -l
       0
$ grep -rn "from tldw_Server_API.app.api" tldw_Server_API/app/core/Ingestion_Media_Processing
persistence.py:1888, persistence.py:2509, persistence.py:2700, persistence.py:2710,
persistence.py:4880, document_upload_preflight.py:8      (6 hits, 2 files)
$ python -m pytest tldw_Server_API/tests/unit/test_ocr_types.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_adapter.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_support.py \
    tldw_Server_API/tests/Media_Ingestion_Modification/test_ocr_runtime_discovery.py -q
======================== 24 passed, 7 warnings in 0.93s ========================
```

## ADR constraints checked before asserting anything

- `Docs/ADR/024-deepseek-ocr-local-transformers-backend.md` (Accepted) binds the `deepseek` backend:
  local-Transformers-only, `trust_remote_code=True` accepted, `DEEPSEEK_OCR_SAVE_RESULTS` defaults
  false with temporary output paths, availability gates explicit, registry priority is current
  implementation behaviour. **Nothing in this ledger proposes changing any of those.** The findings
  against `deepseek_ocr.py` are confined to the shape of its private `_env_bool`/`_env_int` helpers,
  which the ADR does not address.
- `Docs/ADR/022-embeddings-api-and-media-pipeline.md` (Accepted) binds media-embedding pipeline
  ownership: core Jobs owns the durable root `embeddings_pipeline` record, Redis Streams owns stage
  delivery. `persistence.py:2509` imports `api/v1/endpoints/media_embeddings` inside that pipeline;
  the finding in stage 3 is about the *import direction*, not the ownership split the ADR fixes.
- `Docs/ADR/026-security-outbound-egress-and-ssrf-policy.md` — `persistence.py:4915` calls
  `Security/url_validation.assert_url_safe` on each item URL before download, consistent with the
  policy. No finding.

## Findings

### FINDING ingestion-media-processing-11 — test suite is partitioned along no axis a maintainer can follow

```
axis:        duplication
class:       true-duplication (of the suite's organising decision, not of code)
severity:    Medium
sites:       tests/MediaIngestion_NEW/ (61 files importing this module),
             tests/Audio/ (46), tests/Media_Ingestion_Modification/ (30), tests/Media/ (14),
             plus 41 files across 25 further trees — full list in
             2026-09-21-stage1-test-inventory.txt.
             Worst case: Audio/Audio_Transcription_Lib.py is imported from 11 distinct trees.
             Declared-but-unused organising mechanism: pyproject.toml:632 marker
             "media_processing" (1 use, tests/e2e/test_full_user_workflow.py:277) and
             pyproject.toml:631 marker "pipeline" (0 uses).
canonical:   pyproject.toml:585-635 already declares the `media_processing` marker.
destination: n/a — the fix is adoption of the existing marker plus a location ratchet, not a new module.
knowledge:   "where the tests for this source file live." There is no naming rule, no conftest
             boundary that implies one, and no marker applied. The knowledge exists only in the
             heads of people who wrote each tree.
impact:      Medium, and it is change-amplification not coverage. An engineer editing
             Audio_Transcription_Lib.py who runs `pytest tldw_Server_API/tests/Audio` runs 46 of the
             ~60 files that import it and silently skips the other ~14 across 10 trees. The repo's
             blocking mypy gate is changed-files-only (backend-required.yml:183), so nothing else
             catches the gap either. This is the mechanism by which a regression in a hot file
             (98 commits/12mo for persistence.py) reaches main with a green local run.
tests:       n/a — this finding is about the tests.
effort:      cheap for the marker (mechanical: add `pytestmark = pytest.mark.media_processing` to the
             192 files, then `pytest -m media_processing` is the complete set); moderate for a
             ratchet test. Do not attempt to physically merge the trees.
owner-only:  no
confidence:  confirmed (the scatter and the unused markers); assumption (that the scatter is
             historical accretion rather than a deliberate split — no doc explains it).
```

### FINDING ingestion-media-processing-12 — `result_normalization.MediaItemProcessResponse` is a dead type shadowed by a second, differently-shaped dead type

```
axis:        encapsulation
class:       divergent-copies
severity:    Medium
sites:       core/Ingestion_Media_Processing/result_normalization.py:MediaItemProcessResponse (16-40)
               — dataclass, ZERO importers repo-wide.
             api/v1/schemas/media_request_models.py:MediaItemProcessResponse (611-629)
               — Pydantic BaseModel, ZERO uses as a `response_model` and zero importers.
             Docstrings that claim conformance without any type enforcing it:
               Plaintext/Plaintext_Files.py:351, :356; Video/Video_DL_Ingestion_Lib.py:1319.
             Producers that hand-build the envelope as a raw dict (representative, in-module):
               PDF/PDF_Processing_Lib.py:process_pdf (478-1443) and :process_pdf_task (1445-1543);
               persistence.py:1476, :2148, :2157, :2393, :4184, :4415, :4457, :4637, :4681,
               :4824, :4888, :5505; audio_batch.py:88, :189, :295.
canonical:   NONE is authoritative. Two candidate types exist and neither is wired to anything.
destination: `core/Ingestion_Media_Processing/result_normalization.py` is the right home and already
             has the single responsibility ("normalizing process-only media results"); the fix is to
             make its dataclass the one producers construct and to derive the API schema from it,
             not to add a third type.
knowledge:   the process-result envelope: which keys exist, which are optional, and which
             `media_type` values are legal.
impact:      Medium. The two declarations have already drifted in four ways and nothing detects it:
             the Pydantic model carries `segments`, `claims`, `claims_details` that the dataclass
             lacks; the dataclass carries `db_id`, `db_message` that the Pydantic model lacks *and
             forbids* (`extra="forbid"`, :629); the Pydantic model constrains
             `media_type: Literal["video","audio","document","pdf","ebook","email"]` while core
             emits `"zip"` at 10 sites; `content: str` is required in the Pydantic model and
             `Any | None` in the dataclass. If anyone ever attaches the Pydantic model as a
             `response_model` — the obvious next step for anyone tidying the API — every `zip`
             result and every persisted result carrying `db_id` starts failing validation.
tests:       tests/Media_Ingestion_Modification/test_ingestion_helpers_stage3.py imports
             `result_normalization` (the functions, not the dataclass).
effort:      moderate — mechanical but wide; needs a Docs/Design note because it changes a public
             response contract.
owner-only:  yes (touches `tldw_Server_API/app/api/v1/schemas/`).
confidence:  confirmed (both types dead, and the four field-level divergences);
             probable-risk (the "someone attaches the response_model" consequence).
```

## Suggested Refactor/Actions

1. **Marker adoption (cheap, do first).** Add `pytestmark = pytest.mark.media_processing` to the 192
   files in `2026-09-21-stage1-test-inventory.txt`. `pytest -m media_processing` then becomes the
   answer to "what covers this module", with no directory moves and no import churn. Follow with a
   `tests/lint/` ratchet in the shape of the existing
   `tests/lint/test_endpoint_auth_deps_import_boundary.py` (AST-based, seeded at the current number
   so it can only go down) asserting that a source module is imported from at most N test trees.
2. **Envelope consolidation** needs `Docs/Design/2026-MM-DD-media-process-envelope-design.md` plus a
   Backlog task, because it crosses the owner-only `api/v1/schemas/` boundary. Sequence: make
   `result_normalization.MediaItemProcessResponse` the producer-side type; add `"zip"` to the
   Pydantic `Literal` or stop emitting it; then attach the response model. Do not add a third type.
3. Do **not** merge the four test trees. The cost is high (192 file moves, conftest collisions) and
   the marker gets the same benefit.
