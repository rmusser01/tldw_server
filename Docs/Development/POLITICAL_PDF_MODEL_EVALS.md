# Repeatable political PDF model evaluations

Use `Helper_Scripts/Evals/run_political_pdf_eval.py` to compare models against the same PDF, frozen questions and saved retrieval evidence. It calls the existing Media, RAG, Chat and Workspace Jobs APIs; the report is generated offline. It reuses ingestion and never marks an output as human-approved.

Tracked by TASK-15505.11. See the [design](../Design/POLITICAL_PDF_MODEL_EVALS_2026_11_15.md) for scope and architecture.

## Prepare a local configuration

Run commands from the repository root with the project's installed environment:

```bash
source .venv/bin/activate
mkdir -p .tmp/political-evals
cp Helper_Scripts/Evals/configs/political_pdf_eval.v1.json .tmp/political-evals/config.json
```

Edit the copied [sample configuration](../../Helper_Scripts/Evals/configs/political_pdf_eval.v1.json). Its example Media ID is a placeholder, not an instruction to adopt that item.

| Setting | What to supply |
| --- | --- |
| `api.base_url` | Your local application origin; default `http://127.0.0.1:8000`. Only credential-free loopback HTTP(S) origins are accepted. |
| `api.api_key_env` or `api.jwt_env` | The **name** of an environment variable containing application authentication. Populate it through your shell or CI secret store; never put its value in JSON. |
| `source.path`, `source.sha256`, `source.pdf_pages` | The original PDF and its frozen identity. The sample matches the supplied 922-page fixture. |
| `source.media_id` | An existing Media item owned by the authenticated account and containing that exact original PDF. The runner does not upload or guess an item. |
| `cases` | The frozen [case fixture](../../tldw_Server_API/tests/Research_Workspace/fixtures/political_pdf_claim_cases.json). Optional `case_ids` selects a nonempty subset. |
| `models` | A list of unique `key`, exact `provider` and `model` IDs, and `max_tokens` values. Optional controls are `temperature`, `top_p` and QA-only `reasoning_effort`. |
| `retrieval` | `modes`: `text`, `vector`, and/or `hybrid`; `top_k`; and `hybrid_alpha`. Required embeddings/indexes must already be available on the server. |
| `qa.contexts` | `reference_pages`, `retrieved`, or both. `qa.retrieval_mode` must be in the configured retrieval modes. |

Paths in the config resolve from the current working directory, not the config file's directory. Model and case keys use letters, digits, `_`, `-` and `.`. Add one model entry per comparison candidate; each receives the selected question/context matrix. Use exact IDs supported by your configured server. Provider credentials stay in server configuration.

The fixture holds questions, physical PDF pages, extracted-page SHA-256 fingerprints and expected source facts. Planning validates the PDF hash, page count and selected page fingerprints. Expected facts stay in the review evidence; they are not included in generation prompts. QA uses the production section-extraction prompt and saves its request/response locally with `save_to_db: false`.

This runner currently requires a Unix-like environment because it uses `fcntl` file locking and `O_NOFOLLOW`. Use Linux, macOS or a suitable WSL environment; native Windows is not supported.

## Plan, run and report

Planning is offline: it validates local inputs and prints the resolved identity and potential QA/job counts. It does not verify credentials or model access.

```bash
python Helper_Scripts/Evals/run_political_pdf_eval.py plan \
  --config .tmp/political-evals/config.json
```

The default execution stages are `source,retrieval`. QA and summary generation require explicit stage selection. Review the plan before adding generation stages:

```bash
python Helper_Scripts/Evals/run_political_pdf_eval.py run \
  --config .tmp/political-evals/config.json \
  --run-dir .tmp/political-evals/run-a \
  --stages source,retrieval,qa --execute
```

`run` requires a new run identity and `--execute`; it refuses an existing manifest. Every non-pack workflow verifies the retained server PDF and navigation identity before generation. Retrieval disables generation and model-driven query enhancements. Retrieved QA reuses the saved selected retrieval mode; reference-page QA reads the frozen pages directly. Results remain separate by model, case and context.

Retrieval uses chunk-level FTS with late chunking disabled. QA rejects empty context or context over the production limit of 24,000 characters before making a paid request. To reduce retrieved context, lower `retrieval.top_k` in a new configuration/run; evidence is not silently truncated.

Regenerate reports without a server, credentials or the original checkout revision:

```bash
python Helper_Scripts/Evals/run_political_pdf_eval.py report \
  --run-dir .tmp/political-evals/run-a
```

Use a fixed, preferably clean checkout for comparable runs. `manifest.json` freezes resolved configuration, source/fixture hashes, selected cases and Git `HEAD`; the Git SHA does not fingerprint uncommitted edits. `resume` refuses changed configuration, paths, source, fixture or revision. Use a new run directory for a changed matrix or code revision, preserving the old run.

## Summaries and existing jobs

For `--stages summary`, set `workspace_id` and explicit `summary.sections`. Each section requires the workspace `source_id`, navigation `node_id` and `navigation_version` from the owned source. Do not infer these identities from titles. The sample leaves the workspace and selected sections empty so it cannot guess a summary scope.

`summary.artifact_type` defaults to `summary`. Configure the instruction and generation settings before the run; model-level generation settings override matching summary defaults. The public Jobs interface cannot represent explicit `reasoning_effort`, so a summary stage rejects a model entry containing it before submission. Use a separate default-reasoning summary config if QA needs an explicit effort.

```bash
python Helper_Scripts/Evals/run_political_pdf_eval.py resume \
  --config .tmp/political-evals/config.json \
  --run-dir .tmp/political-evals/run-a \
  --stages summary --execute
```

Set summary configuration **before** creating `run-a`, even if the first invocation only runs QA. Otherwise, use a new run directory for the changed config.

Without adoption, each configured model submits a distinct output job. To inspect an existing job instead, set `summary.job_ids` to a mapping from a configured model key to its positive existing job ID before planning. The runner verifies the artifact type and configured provider/model/settings/section scope, then polls and reads back the artifact without a new submission. An incompatible job is rejected.

`summary.wait_seconds` defaults to `3600` and accepts integers from `1` to `86400`. Polls wait up to 15 seconds, bounded by the remaining wait time. Reaching the wait deadline records `pending` and returns a nonzero exit; it does not cancel the server job. Resume with the same config to continue polling its saved ID without another POST.

Every poll retains its full response, so large artifacts and long jobs can consume substantial disk space even at this interval. Allow room for the evidence; interrupting CLI polling does not cancel the server job.

Terminal summary results, including failures and partial evidence, are retained on resume. The runner rejects `--retry-operation` for a known summary job, whether running or terminal. Resume its recorded job or use a new run directory for a new intentional generation.

To review or export CLI-created outputs in an already-open browser, choose **Workspaces → Manage server Workspaces → Open** for the same workspace. This reopens its canonical server state. A reload of the generic Research Workspace page can retain its older local artifact list, and Complete Pack exports that local list. Resolve any unsaved-work warning before reopening; verify that the new output is visible before exporting.

## Resume and explicit retries

```bash
python Helper_Scripts/Evals/run_political_pdf_eval.py resume \
  --config .tmp/political-evals/config.json \
  --run-dir .tmp/political-evals/run-a \
  --stages source,retrieval,qa --execute
```

Completed operations reuse saved evidence. Recorded failures also remain saved until an explicit supported retry is requested. A started attempt without a finish, or an uncertain POST outcome, blocks automatic resubmission. Inspect `attempts.jsonl` and reconcile the server's state first: a timeout does not prove that paid work never ran. If a summary job exists, its verified ID can be adopted in a separately configured run while retaining the uncertain run as evidence.

After deciding that another request is appropriate, use the exact ordinary operation ID from the ledger. For example, if the copied config has model key `candidate-a`:

```bash
python Helper_Scripts/Evals/run_political_pdf_eval.py resume \
  --config .tmp/political-evals/config.json \
  --run-dir .tmp/political-evals/run-a \
  --stages qa --execute \
  --retry-operation 'qa:candidate-a:R01:reference_pages'
```

The flag creates another immutable attempt and can incur another provider call. It never erases the first failure. Repeat the flag for additional reviewed operation IDs; do not edit the ledger to manufacture success. One process holds the run-directory lock at a time.

CLI exit codes are `0` when requested operations are recorded as completed, `1` for returned non-completed outcomes such as failure, unknown or pending, and `2` for validation/unsafe-resume errors. A completed HTTP operation is not a semantic pass; inspect its structural metrics and human review packet. `report` returning `0` means report generation succeeded.

## Validate a supplied research pack

Add `pack.path` and its whole-file `pack.sha256` to the config before creating the run. Then explicitly select `--stages pack` with `run` or `resume` and `--execute`. This stage inspects a local supplied archive; it does not create, upload or restore one.

Validation checks bounded ZIP inventory, required entries, manifest byte counts and hashes, and an included original PDF matching the frozen source hash. These checks do not certify browser export behavior, the offline reader or a cross-server UI restore.

## Read the evidence

| Output | Contents |
| --- | --- |
| `manifest.json` | Frozen resolved configuration and source/fixture/revision identity. |
| `attempts.jsonl` | Append-only started/finished records, operation IDs, all attempts, outcomes, timings and checks. |
| `evidence/` | Immutable requests, response envelopes and retained binary artifacts. |
| `report.json` | Every attempt plus first/latest outcomes, per-model operation counts, structural metrics and available usage/identity. |
| `report.md` | Compact comparison with failures, pending outcomes, citation/page checks and summary counts. |
| `semantic-review.json` | Frozen expected facts, physical page references, evidence paths and initially unfilled human labels. |

Retrieval coverage uses the frozen expected physical-page set as its denominator. Source scope, pipeline errors and overbroad returned ranges are separate diagnostics. QA checks output shape, supplied passage IDs, finish reason/truncation and returned model identity; token counts are reported only when present and valid. Missing usage is **unavailable**, not zero.

Summary checks retain section/portion completion and failure counts, selected-scope and exact evidence-span validation, and artifact readback. Jobs do not expose provider-call token usage through this interface. New-job elapsed time is submission-to-observed-terminal wall time, including resume downtime; adopted-job latency is unavailable. Per-operation elapsed time and its sum are not whole-run wall time.

Faithfulness, author attribution, disagreement and completeness remain human judgments. Review the source and saved response, then fill the packet's labels/status and optional reviewer/notes. Regenerating reports preserves those decisions only for the same evidence fingerprint; changed evidence is rejected. H05 is a tuned regression, not an untouched held-out result. No automatic judge or accuracy score is supplied, and no frontend UI evaluation is automated here.

CLI QA uses the production extraction prompt; the UI uses its chat pipeline. Passing CLI checks does not establish UI behavior or semantic correctness.

## Local and CI verification

The deterministic tests use temporary PDFs, saved files and fake HTTP responses. They do not require provider credentials or a running application:

```bash
source .venv/bin/activate
python -m pytest \
  tldw_Server_API/tests/Evaluations/test_political_pdf_eval_workflow.py \
  tldw_Server_API/tests/Evaluations/test_political_pdf_eval_report.py -q
```

Use these tests as the default CI check. A live CI run is an explicit separate job with a reachable loopback application, pre-existing owned source/workspace, provider configuration and application authentication supplied through CI secrets. Preserve the run directory as access-controlled evidence: it contains source passages and generated text even though authentication headers and known secrets are excluded or redacted. Keep prior runs and human labels when comparing another model or revision.
