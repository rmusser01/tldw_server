# Research

Research implements deep research run orchestration and provider-backed collection helpers. It creates per-user research sessions, advances planning, collection, synthesis, and packaging phases through Jobs, records artifacts and stream events, supports checkpoint review, and also sits near legacy research endpoints for arXiv, Semantic Scholar, and web search.

## Start Here

- `service.py` is the deep research session service used by `/research/runs`.
- `jobs.py` enqueues and handles phase jobs for planning, collecting, synthesizing, and packaging.
- `jobs_worker.py` runs the deep research Jobs worker.
- `artifact_store.py` writes versioned artifacts and registers them in `ResearchSessionsDB`.
- `broker.py` selects local, academic, and web collection lanes and normalizes source and evidence records.
- `synthesizer.py` builds deterministic or provider-backed synthesis outputs.
- Related API surface: `tldw_Server_API/app/api/v1/endpoints/research_runs.py` and `tldw_Server_API/app/api/v1/endpoints/research.py`.
- Related schemas: `tldw_Server_API/app/api/v1/schemas/research_runs_schemas.py` and `tldw_Server_API/app/api/v1/schemas/research_schemas.py`.
- Related tests: `tldw_Server_API/tests/Research/` and `tldw_Server_API/tests/e2e/test_deep_research_runs.py`.

## Responsibilities

- Create, list, read, delete, pause, resume, and cancel deep research sessions.
- Enqueue phase jobs with stable idempotency keys in the `research` Jobs domain.
- Draft initial plans and optionally pause for human checkpoint review.
- Collect sources through local corpus, academic, and web lanes according to source policy.
- Write source registries, evidence notes, summaries, outlines, claims, reports, and bundles as versioned artifacts.
- Synthesize claims with citation, trust, contradiction, and support metadata.
- Stream reconnect-safe run events through persisted research run events.
- Package final research output and optionally deliver chat handoff content.

## Module Map

- `models.py`: dataclasses for plans, sources, evidence, outlines, claims, synthesis, and artifacts.
- `service.py`: session lifecycle, checkpoint approval, artifact reads, stream snapshots, and run controls.
- `jobs.py`: phase job enqueueing and phase handlers.
- `jobs_worker.py`: WorkerSDK loop for research Jobs.
- `artifact_store.py`: artifact file writes, reads, checksums, versions, and DB manifest records.
- `broker.py`: source-policy lane selection, dedupe, and evidence note normalization.
- `providers/`: local corpus, academic, web, synthesis provider, and provider config resolution.
- `planner.py`: initial plan construction.
- `synthesizer.py`: deterministic and provider-backed synthesis.
- `exporter.py`: final package validation and assembly.
- `streaming.py`: stream state and event conversion helpers.
- `checkpoint_service.py`, `chat_handoff.py`, `limits.py`: checkpoint patching, chat return handoff, and bounded input helpers.

## How It Connects

- `research_runs.py` exposes `/research/runs`, run controls, `/events/stream`, `/bundle`, artifact reads, and checkpoint patch-and-approve routes.
- `research.py` exposes deprecated arXiv and Semantic Scholar search routes plus `/research/websearch`; those routes use Third_Party and Web_Scraping helpers rather than the deep research phase pipeline.
- `ResearchSessionsDB` stores sessions, checkpoints, artifacts, run events, and chat handoff links.
- `DatabasePaths` resolves per-user research session DB paths and output directories.
- Jobs integration uses domain `research`, job type `research_phase`, and queue `default`.
- Local collection uses the RAG multi-database retriever over media, notes, prompts, and kanban sources.
- Academic and web providers call Third_Party search helpers and WebSearch APIs outside test mode.
- Workflows can resume when a research checkpoint is approved through `Workflows.research_wait_bridge`.

## Extension Points

- Add a phase by extending the executable phase set, job dispatch in `jobs.py`, service transition logic, and schemas or endpoint responses.
- Add a provider lane by extending `providers/`, `broker.py`, and provider config validation.
- Add an artifact by updating `ResearchService._ALLOWED_ARTIFACT_NAMES`, phase writes, and artifact response tests.
- Change checkpoint patch behavior in `checkpoint_service.py`.
- Change synthesis behavior in `synthesizer.py` and provider-backed synthesis tests.
- Change final package shape in `exporter.py` and research package adapter tests.

## Testing

- Direct research coverage lives under `tldw_Server_API/tests/Research/`.
- End-to-end deep research coverage lives in `tldw_Server_API/tests/e2e/test_deep_research_runs.py`.
- Related workflow and chat integration coverage includes `tldw_Server_API/tests/Workflows/adapters/test_research_adapters.py`, `tldw_Server_API/tests/AuthNZ/integration/test_chat_research_runs_endpoint.py`, and `tldw_Server_API/tests/Chat_NEW/unit/test_research_chat_context.py`.

## Gotchas

- The deep research API and legacy research search endpoints share the `/research` tag area but use different implementation paths.
- Artifact reads are allowlisted by name in `ResearchService`.
- Checkpoint approval is blocked while a run is paused, pause-requested, cancel-requested, or cancelled.
- Provider-backed collection and synthesis fall back to test-mode simulations only when test mode is active.
