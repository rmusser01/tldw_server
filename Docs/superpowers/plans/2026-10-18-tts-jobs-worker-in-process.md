# Long-Form TTS Jobs Worker Runs By Default Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A default deployment processes the long-form TTS jobs that `POST /api/v1/audio/speech/jobs` queues and the WebUI submits. The TTS jobs worker runs in-process wherever the audio routes run, and as a sidecar worker in sidecar deployments (TASK-13849).

**Architecture:**
- **In-process.** `app/services/startup_content_jobs_pollers.py` gains a `tts_jobs_task` worker spec next to `audio_jobs_task`.
  - Its service function lazily imports and runs `core/TTS/tts_jobs_worker.run_tts_jobs_worker(stop_event)`.
  - It is enabled by the same route-policy predicate the media ingest workers use (`media_ingest_worker_predicate`, which wraps `should_start_inprocess_worker`), with `TTS_JOBS_WORKER_ENABLED` and the `audio` route.
  - So it starts by default wherever the `audio` route (which serves the speech endpoints) is enabled. It is off in sidecar mode and in test mode, and `TTS_JOBS_WORKER_ENABLED=false` turns it off.
  - The server's startup already loads and refreshes the LLM provider overrides (`app/services/startup_auth.py`), which the standalone entry point does itself.
- **Sidecar.**
  - `Docs/Deployment/sidecar_workers_manifest.json` gains a `tts` worker (module `tldw_Server_API.app.core.TTS.tts_jobs_worker`) in `default_workers`.
  - The `tts-only` sidecar profile (`Helper_Scripts/common/sidecar_profiles.py`) gains `tts`.
  - `Helper_Scripts/Deployment/generate_sidecar_files.py` then regenerates:
    - `Dockerfiles/docker-compose.workers.yml`;
    - the sidecar docs;
    - systemd and launchd units, with their `Docs/Published` copies.

**Tech Stack:** Python, the lifecycle worker specs (`app/services/lifecycle_worker_specs.py`, `worker_startup_policy.py`), the sidecar generator and profiles, pytest.

**Spec:** `backlog/tasks/task-13849 - Long-form-TTS-jobs-have-no-worker-in-any-deployment-but-the-WebUI-offers-them.md`:
- AC #1: "A default deployment either runs the TTS jobs worker (for example a lifespan worker behind a flag, like the other Jobs workers) or tells the caller that long-form TTS needs it, instead of leaving jobs queued silently".
- AC #2: "The deployment docs say how to run the TTS jobs worker".

## What was verified (on `origin/dev` `42176e5bd4`)

- **Nothing starts the TTS jobs worker.**
  - There is no worker spec in `startup_content_jobs_pollers.py` or any other startup module.
  - It has no entry in `sidecar_workers_manifest.json`, so there is no compose service and no unit file.
  - The `tts-only` sidecar profile selects only `audio` and `audio_jobs`, which are the speech-to-text pipeline workers.
  - `Dockerfile.prod` runs only the API server.
  - The worker runs only as `python -m tldw_Server_API.app.core.TTS.tts_jobs_worker`, and the WebUI calls `POST /api/v1/audio/speech/jobs` (`apps/packages/ui/src/services/tldw/domains/models-audio.ts`, `TldwApiClient.ts`).
- **`route_enabled_predicate`, which most job workers use, needs its env flag set explicitly.**
  - `should_start_inprocess_worker`, used by the media ingest workers, instead defaults the flag to the route's own policy (`worker_path_enabled`).
  - `Env_Vars.md` describes the media ingest flags as following route policy. It wrongly says the same of `AUDIO_JOBS_WORKER_ENABLED`, which uses `route_enabled_predicate`. This plan corrects that line.
- **The speech endpoints are served under the `audio` route key** (`app/api/v1/router_groups/content.py`).

## Global Constraints

- **Branch:** `fix/tts-jobs-worker-in-process` from `origin/dev`. The PR targets `dev`.
- **Owner decision:** this plan runs the TTS worker by default, by route policy. The alternative AC #1 allows is to refuse long-form jobs when no worker runs. It is not taken, because the endpoint cannot tell whether a standalone or sidecar worker exists elsewhere. The plan's approval is the owner's call on this.
- **Spec, verbatim:**
  - `stop_event_worker_spec(name="tts_jobs_task", worker_service=_run_tts_jobs_worker_service, category="jobs", phase=ShutdownPhase.JOB_POLLER_QUIESCE, enabled=media_ingest_worker_predicate("TTS_JOBS_WORKER_ENABLED", "audio", default_stable=True))`, placed right after `audio_jobs_task`.
  - `media_ingest_worker_predicate` is reused as is. Its name is media-specific but its body is generic; renaming it would touch unrelated workers.
- **Sidecar manifest:** add the entry `{"key": "tts", "slug": "tts", "label": "Long-form TTS jobs worker", "module": "tldw_Server_API.app.core.TTS.tts_jobs_worker", "optional": false}` after `audio`, and add `"tts"` to `default_workers` after `"audio"`. Then run the generator and commit its outputs. Never hand-edit the generated files.
- **Sidecar profile:** `"tts-only": {"audio", "audio_jobs", "tts"}`.
- **Docs:**
  - `Docs/Operations/Env_Vars.md`:
    - gains `TTS_JOBS_WORKER_ENABLED`, with two operator notes. A server restart during a synthesis fails that job (`tts_replay_blocked`). With large local models on a GPU and several server processes, run one dedicated worker instead.
    - gains `TTS_JOBS_QUEUE`;
    - corrects `AUDIO_JOBS_WORKER_ENABLED` to "default `false`; it also needs the `audio-jobs` route enabled".
  - `Docs/Architecture/audio.md` item 8 says where the worker runs.
  - `Docs/Architecture/jobs-background.md` lists `tts` among the sidecar workers.
  - `Docs/Deployment/Long_Term_Admin_Guide.md` says the TTS jobs worker follows the `audio` route policy.
  - Mirrors via `bash Helper_Scripts/refresh_docs_published.sh`. The generator already writes the `Docs/Published` copies of its outputs.
- **Upgrade note:** sidecar deployments with hand-installed systemd/launchd units, or an explicit `TLDW_SIDECAR_WORKERS` list, get no TTS worker until they add `tts`. The commit body and the PR's Change summary say so.
- **Tests:**
  - Three files pin the worker spec names (`tests/Services/test_startup_content_jobs_pollers.py`, `test_lifecycle_worker_catalog.py`, `test_startup_worker_groups.py`), and all three are updated with the new spec.
  - `tests/Utils/test_sidecar_profiles.py` and `tests/Utils/test_sidecar_manifest_defaults.py` each gain a test for the sidecar half.
- **Test first:** the RED run happens before the source change.
- **Backlog:** `tools/backlog-py` only, `--append-notes`, explicit id `TASK-13849`, `git add -A backlog/tasks`.
- **Commits** end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`. Never `--no-verify`, never `git stash`.
- **Out of scope:**
  - The `tts_jobs_worker` module itself, which is unchanged. That includes the restart behaviour (`tts_replay_blocked`), which is by design.
  - Lease handling for very long syntheses. `WorkerSDK` already renews leases.
  - The audio jobs worker's own default, which stays explicit-flag. Only its doc line is corrected.
  - The pre-existing ruff `I001` in `tests/Utils/test_sidecar_manifest_defaults.py`.
  - The example `TLDW_SIDECAR_WORKERS` list in `README.md`, which is already an incomplete example.

## Review Focus

1. **A deployment that already runs a standalone TTS worker** now also runs one in each server process. Both take only `tts_longform` jobs from the same queue, and leases plus the durable dispatch marker keep them from double-processing, so the only cost is extra pollers. The docs tell such an operator to set `TTS_JOBS_WORKER_ENABLED=false`. No test: the plan relies on the Jobs lease contract.
2. **Sidecar mode.** The in-process worker does not start, and the `tts` sidecar worker runs instead.
   - `test_tts_jobs_worker_runs_in_process_by_default` asserts the spec is off with `sidecar_mode=True`.
   - `test_sidecar_manifest_runs_the_long_form_tts_worker_by_default` asserts `tts` is a default sidecar worker with the right module.
   - `test_resolve_workers_tts_only_profile_includes_the_long_form_tts_worker` asserts the `tts-only` profile selects it.
   - No test checks that the generated compose and unit files are current, because the generator has no check mode. Step 6 checks the compose service by its diff.
3. **A server restart during a long synthesis** fails that job permanently (`tts_replay_blocked`). The dispatch marker is already set, and it blocks a second synthesis by design. In-process, every deploy, crash or `uvicorn --reload` restart now does this to in-flight jobs. The `Env_Vars.md` note tells operators who restart often to run the sidecar or a standalone worker. No test: the module is unchanged.
4. **Several server processes** (`Dockerfile.prod` defaults `UVICORN_WORKERS=4`) each run a TTS worker.
   - They share the queue through leases, and `WorkerSDK` renews leases during synthesis.
   - Each process lazily loads its own copy of a local TTS model, so up to four long-form jobs can hold four copies on a GPU. The synchronous `/audio/speech` endpoint already loads one per process.
   - The `Env_Vars.md` note tells GPU operators to set the flag to `false` and run one dedicated worker. No new test.
5. **Test mode and the audio route off.** The worker does not start in test runs, so existing suites that build the app see no new background task, and it does not start where the `audio` route is off.
   - The new test asserts both with the flag unset.
   - `test_content_jobs_worker_spec_predicates_use_route_enabled_arguments` covers route off with the flag explicitly on (spec on, as for media ingest).

---

### Task 1: The TTS jobs worker runs in-process by default and as a sidecar worker

**Files:**
- Modify: `tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py`. Update the name lists and add `test_tts_jobs_worker_runs_in_process_by_default` and a `replace` import.
- Modify: `tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py` (`TASK5_JOB_POLLER_SPEC_NAMES`).
- Modify: `tldw_Server_API/tests/Services/test_startup_worker_groups.py` (expected spec names).
- Modify: `tldw_Server_API/tests/Utils/test_sidecar_profiles.py`. Add `test_resolve_workers_tts_only_profile_includes_the_long_form_tts_worker`.
- Modify: `tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py`. Add `test_sidecar_manifest_runs_the_long_form_tts_worker_by_default`.
- Modify: `tldw_Server_API/app/services/startup_content_jobs_pollers.py`. Add the spec and `_run_tts_jobs_worker_service`.
- Modify: `Helper_Scripts/common/sidecar_profiles.py` (`tts-only`).
- Modify: `Docs/Deployment/sidecar_workers_manifest.json`.
- Regenerate:
  - `Dockerfiles/docker-compose.workers.yml`;
  - `Docs/Deployment/Sidecar_Workers.md` and `Docs/Published/Deployment/Sidecar_Workers.md`;
  - `Docs/Published/Deployment/sidecar_workers_manifest.json` (via the docs refresh);
  - new files `Docs/{,Published/}Deployment/systemd/tldw-worker-tts.{service,timer}` and `Docs/{,Published/}Deployment/launchd/com.tldw.worker.tts.plist`.
- Modify: the source docs below, plus their mirrors `Docs/Published/Env_Vars.md` and `Docs/Published/Deployment/Long_Term_Admin_Guide.md`:
  - `Docs/Operations/Env_Vars.md`;
  - `Docs/Architecture/audio.md`;
  - `Docs/Architecture/jobs-background.md`;
  - `Docs/Deployment/Long_Term_Admin_Guide.md`.
- Modify: `backlog/tasks/task-13849*.md` (backlog-py only).
- Add: this plan, `Docs/superpowers/plans/2026-10-18-tts-jobs-worker-in-process.md`.

**Interfaces:**
- Produces:
  - the worker spec `tts_jobs_task`;
  - `startup_content_jobs_pollers._run_tts_jobs_worker_service(stop_event)`;
  - the sidecar worker key `tts`, which the `tts-only` profile selects.

Set once for every step: `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`, `export TLDW_TEST_NO_DOCKER=1`, and:

```bash
T1="tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py tldw_Server_API/tests/Services/test_startup_worker_groups.py tldw_Server_API/tests/Utils/test_sidecar_profiles.py tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py"
```

- [ ] **Step 1: Branch**

```bash
git fetch -q origin dev
git switch -c fix/tts-jobs-worker-in-process origin/dev
```

Copy this plan to `Docs/superpowers/plans/2026-10-18-tts-jobs-worker-in-process.md`.

- [ ] **Step 2: Write the failing tests**

Apply exactly this diff with `git apply`:

```diff
diff --git a/tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py b/tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py
index be3d3bc42c..83a910dc50 100644
--- a/tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py
+++ b/tldw_Server_API/tests/Services/test_lifecycle_worker_catalog.py
@@ -105,6 +105,7 @@ TASK5_JOB_POLLER_SPEC_NAMES = {
     "study_suggestions_jobs_task",
     "privilege_snapshot_task",
     "audio_jobs_task",
+    "tts_jobs_task",
     "audiobook_jobs_task",
     "chat_macros_jobs_task",
     "presentation_render_jobs_task",
diff --git a/tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py b/tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py
index a07f2dc8b5..6d7729980f 100644
--- a/tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py
+++ b/tldw_Server_API/tests/Services/test_startup_content_jobs_pollers.py
@@ -5,6 +5,7 @@ import importlib
 import sys
 import threading
 from collections.abc import Callable
+from dataclasses import replace
 from types import SimpleNamespace
 from typing import Any

@@ -127,6 +128,7 @@ def _standalone_config(
     "spec_name",
     [
         "audio_jobs_task",
+        "tts_jobs_task",
         "audiobook_jobs_task",
         "presentation_render_jobs_task",
         "standalone_html_generation_jobs_task",
@@ -164,6 +166,7 @@ def test_content_jobs_worker_specs_use_expected_names() -> None:

     assert [spec.name for spec in startup_pollers.provide_content_jobs_worker_specs()] == [
         "audio_jobs_task",
+        "tts_jobs_task",
         "audiobook_jobs_task",
         "presentation_render_jobs_task",
         "standalone_html_generation_jobs_task",
@@ -189,6 +192,7 @@ def test_content_jobs_worker_spec_factories_delegate_to_existing_worker_services

     for spec_name, factory_name in [
         ("audio_jobs_task", "_run_audio_jobs_worker_service"),
+        ("tts_jobs_task", "_run_tts_jobs_worker_service"),
         ("audiobook_jobs_task", "_run_audiobook_jobs_worker_service"),
         ("presentation_render_jobs_task", "_run_presentation_render_jobs_worker_service"),
         ("standalone_html_generation_jobs_task", "_run_standalone_html_generation_jobs_service"),
@@ -221,6 +225,7 @@ def test_content_jobs_worker_spec_factories_delegate_to_existing_worker_services

     assert calls == [
         ("audio_jobs_task", "audio_jobs_task-stop"),
+        ("tts_jobs_task", "tts_jobs_task-stop"),
         ("audiobook_jobs_task", "audiobook_jobs_task-stop"),
         ("presentation_render_jobs_task", "presentation_render_jobs_task-stop"),
         ("standalone_html_generation_jobs_task", "standalone_html_generation_jobs_task-stop"),
@@ -254,12 +259,14 @@ def test_content_jobs_worker_spec_predicates_use_route_enabled_arguments(
     context = _context(route_enabled=_route_enabled)
     specs = _specs_by_name(startup_pollers)
     media_ingest_explicit_flag_specs = {
+        "tts_jobs_task",
         "media_ingest_jobs_task",
         "media_ingest_heavy_jobs_task",
     }

     for spec_name in [
         "audio_jobs_task",
+        "tts_jobs_task",
         "audiobook_jobs_task",
         "presentation_render_jobs_task",
         "research_workspace_output_jobs_task",
@@ -277,6 +284,7 @@ def test_content_jobs_worker_spec_predicates_use_route_enabled_arguments(
         monkeypatch.setenv(
             {
                 "audio_jobs_task": "AUDIO_JOBS_WORKER_ENABLED",
+                "tts_jobs_task": "TTS_JOBS_WORKER_ENABLED",
                 "audiobook_jobs_task": "AUDIOBOOK_JOBS_WORKER_ENABLED",
                 "presentation_render_jobs_task": "PRESENTATION_RENDER_JOBS_WORKER_ENABLED",
                 "research_workspace_output_jobs_task": "RESEARCH_WORKSPACE_OUTPUT_JOBS_WORKER_ENABLED",
@@ -316,6 +324,30 @@ def test_content_jobs_worker_spec_predicates_use_route_enabled_arguments(
     ]


+def test_tts_jobs_worker_runs_in_process_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
+    """Long-form TTS jobs (POST /audio/speech/jobs) have no other worker in a default deployment (TASK-13849)."""
+    startup_pollers = _import_startup_content_jobs_pollers()
+    spec = _specs_by_name(startup_pollers)["tts_jobs_task"]
+    monkeypatch.delenv("TTS_JOBS_WORKER_ENABLED", raising=False)
+    routes: list[object] = []
+
+    def _route_enabled(route: object, *_args: object, **_kwargs: object) -> bool:
+        routes.append(route)
+        return True
+
+    live = replace(_context(route_enabled=_route_enabled), test_mode=False)
+    assert spec.enabled(live) is True
+    assert routes == ["audio"]
+    # It follows the audio route's answer, not just its presence.
+    assert spec.enabled(replace(live, route_enabled=lambda *_a, **_k: False)) is False
+    # Test runs do not start it unless asked.
+    assert spec.enabled(_context(route_enabled=_route_enabled)) is False
+    # A sidecar deployment runs it as its own service instead.
+    assert spec.enabled(replace(live, sidecar_mode=True)) is False
+    monkeypatch.setenv("TTS_JOBS_WORKER_ENABLED", "false")
+    assert spec.enabled(live) is False
+
+
 def test_standalone_html_composite_worker_uses_slides_route_without_a_second_flag() -> None:
     startup_pollers = _import_startup_content_jobs_pollers()
     calls: list[tuple[tuple[object, ...], dict[str, object]]] = []
diff --git a/tldw_Server_API/tests/Services/test_startup_worker_groups.py b/tldw_Server_API/tests/Services/test_startup_worker_groups.py
index 305cc17d7a..09790cd08c 100644
--- a/tldw_Server_API/tests/Services/test_startup_worker_groups.py
+++ b/tldw_Server_API/tests/Services/test_startup_worker_groups.py
@@ -163,6 +163,7 @@ def test_collect_startup_worker_specs_accepts_real_provider_graph() -> None:
         "study_pack_jobs_task",
         "study_suggestions_jobs_task",
         "tts_history_cleanup_task",
+        "tts_jobs_task",
         "usage_aggregator",
         "visual_identity_jobs_task",
         "vn_asset_generation_jobs_task",
diff --git a/tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py b/tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py
index d2a4457c37..7a47a7efab 100644
--- a/tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py
+++ b/tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py
@@ -26,3 +26,11 @@ def test_sidecar_manifest_exists_and_has_expected_shape() -> None:

     for key in default_workers:
         assert key in by_key, f"default worker '{key}' must be present in workers"
+
+
+def test_sidecar_manifest_runs_the_long_form_tts_worker_by_default() -> None:
+    # In sidecar mode the in-process TTS worker is off, so the sidecar must run it (TASK-13849).
+    data = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
+    assert "tts" in data["default_workers"]
+    by_key = {worker["key"]: worker for worker in data["workers"]}
+    assert by_key["tts"]["module"] == "tldw_Server_API.app.core.TTS.tts_jobs_worker"
diff --git a/tldw_Server_API/tests/Utils/test_sidecar_profiles.py b/tldw_Server_API/tests/Utils/test_sidecar_profiles.py
index 37a926a4bc..9a7f9e71e5 100644
--- a/tldw_Server_API/tests/Utils/test_sidecar_profiles.py
+++ b/tldw_Server_API/tests/Utils/test_sidecar_profiles.py
@@ -17,3 +17,13 @@ def test_resolve_workers_tts_only_profile_with_audio_key():
         default_workers_csv="media_ingest,audio,embeddings",
     )
     assert workers == ["audio"]
+
+
+def test_resolve_workers_tts_only_profile_includes_the_long_form_tts_worker():
+    # The tts-only profile must run the worker that processes POST /audio/speech/jobs (TASK-13849).
+    workers = resolve_workers(
+        profile="tts-only",
+        explicit_workers_csv=None,
+        default_workers_csv="media_ingest,audio,tts,embeddings",
+    )
+    assert workers == ["audio", "tts"]
```

- [ ] **Step 3: Run them to see them fail**

Run: `$PY -m pytest -p no:cacheprovider -p no:randomly -q --tb=line $T1`

Expected: `9 failed, 72 passed`, every failure for the missing worker. Measured on `origin/dev` `42176e5bd4`:
- `KeyError: 'tts_jobs_task'`:
  - the legacy-contract case `[tts_jobs_task]`;
  - the predicate test;
  - `test_tts_jobs_worker_runs_in_process_by_default`.
- The expected-names list does not match.
- `AttributeError: ... has no attribute '_run_tts_jobs_worker_service'` (the factory test).
- The catalog set and the worker-groups set do not match (`'tts_jobs_task'` is missing from the actual set).
- The profile test: `assert ['audio'] == ['audio', 'tts']`.
- The manifest test: `assert 'tts' in ['chatbooks', ...]`.

- [ ] **Step 4: Apply the source change**

Apply exactly this diff (the worker spec, the `tts-only` profile and the sidecar manifest):

```diff
diff --git a/Docs/Deployment/sidecar_workers_manifest.json b/Docs/Deployment/sidecar_workers_manifest.json
index d6742b1749..495aecfa31 100644
--- a/Docs/Deployment/sidecar_workers_manifest.json
+++ b/Docs/Deployment/sidecar_workers_manifest.json
@@ -6,6 +6,7 @@
     "prompt_studio",
     "privilege_snapshots",
     "audio",
+    "tts",
     "media_ingest",
     "reading_import",
     "reading_digest",
@@ -54,6 +55,13 @@
       "module": "tldw_Server_API.app.services.audio_jobs_worker",
       "optional": false
     },
+    {
+      "key": "tts",
+      "slug": "tts",
+      "label": "Long-form TTS jobs worker",
+      "module": "tldw_Server_API.app.core.TTS.tts_jobs_worker",
+      "optional": false
+    },
     {
       "key": "media_ingest",
       "slug": "media-ingest",
diff --git a/Helper_Scripts/common/sidecar_profiles.py b/Helper_Scripts/common/sidecar_profiles.py
index e1d6b5ad68..8766a78e3f 100644
--- a/Helper_Scripts/common/sidecar_profiles.py
+++ b/Helper_Scripts/common/sidecar_profiles.py
@@ -3,7 +3,7 @@ from __future__ import annotations
 PROFILE_MAP: dict[str, set[str] | None] = {
     "full": None,
     "llm-only": {"embeddings", "media_ingest"},
-    "tts-only": {"audio", "audio_jobs"},
+    "tts-only": {"audio", "audio_jobs", "tts"},
     "ingest-only": {"media_ingest"},
 }

diff --git a/tldw_Server_API/app/services/startup_content_jobs_pollers.py b/tldw_Server_API/app/services/startup_content_jobs_pollers.py
index 84ef2a6e5e..61bc266403 100644
--- a/tldw_Server_API/app/services/startup_content_jobs_pollers.py
+++ b/tldw_Server_API/app/services/startup_content_jobs_pollers.py
@@ -148,6 +148,15 @@ def provide_content_jobs_worker_specs(
             phase=ShutdownPhase.JOB_POLLER_QUIESCE,
             enabled=route_enabled_predicate("AUDIO_JOBS_WORKER_ENABLED", "audio-jobs"),
         ),
+        stop_event_worker_spec(
+            name="tts_jobs_task",
+            worker_service=_run_tts_jobs_worker_service,
+            category="jobs",
+            phase=ShutdownPhase.JOB_POLLER_QUIESCE,
+            # Long-form TTS (POST /audio/speech/jobs) has no other worker in a default deployment,
+            # so it runs wherever the audio routes do, unless TTS_JOBS_WORKER_ENABLED turns it off.
+            enabled=media_ingest_worker_predicate("TTS_JOBS_WORKER_ENABLED", "audio", default_stable=True),
+        ),
         stop_event_worker_spec(
             name="audiobook_jobs_task",
             worker_service=_run_audiobook_jobs_worker_service,
@@ -1100,6 +1109,14 @@ def _run_audio_jobs_worker_service(stop_event: Any) -> Any:
     return _run_audio_jobs_worker(stop_event)


+def _run_tts_jobs_worker_service(stop_event: Any) -> Any:
+    # The server's startup already loads and refreshes LLM provider overrides, which the standalone
+    # entry point (python -m ...tts_jobs_worker) has to do itself.
+    from tldw_Server_API.app.core.TTS.tts_jobs_worker import run_tts_jobs_worker
+
+    return run_tts_jobs_worker(stop_event)
+
+
 def _run_audiobook_jobs_worker_service(stop_event: Any) -> Any:
     from tldw_Server_API.app.services.audiobook_jobs_worker import (
         run_audiobook_jobs_worker as _run_audiobook_jobs_worker,
```

- [ ] **Step 5: Run the tests to see them pass**

Run the Step 3 command. Expected: `81 passed`.

- [ ] **Step 6: Document the worker, and regenerate the sidecar files**

Apply this diff (the four source docs):

```diff
diff --git a/Docs/Architecture/audio.md b/Docs/Architecture/audio.md
index 41e7b3abda..26e8d047e9 100644
--- a/Docs/Architecture/audio.md
+++ b/Docs/Architecture/audio.md
@@ -27,7 +27,7 @@ All paths under `tldw_Server_API/` unless noted.
 5. All provider egress goes through the shared HTTP client's egress validation (SSRF allow/deny lists).
 6. Errors map through the TTS exception taxonomy (auth, rate-limit, quota, model-not-found/load, …) to HTTP.
 7. Response: audio bytes with optional sample-rate and alignment headers; metrics record totals, durations, TTFB.
-8. Long-form: `POST /speech/jobs` enqueues an audio-domain `tts_longform` job forcing non-streaming; artifacts are listed per job. The job goes on the `tts` queue. `TTS_JOBS_QUEUE` overrides it: set the same value for the API server and the TTS worker, and register a custom name in `JOBS_ALLOWED_QUEUES_AUDIO`. `default` and `transcribe` are refused, because the audio pipeline workers read them without a job-type filter. The TTS jobs worker (`python -m tldw_Server_API.app.core.TTS.tts_jobs_worker`) reads that queue and takes only `tts_longform` jobs; nothing starts it automatically.
+8. Long-form: `POST /speech/jobs` enqueues an audio-domain `tts_longform` job forcing non-streaming; artifacts are listed per job. The job goes on the `tts` queue. `TTS_JOBS_QUEUE` overrides it: set the same value for the API server and the TTS worker, and register a custom name in `JOBS_ALLOWED_QUEUES_AUDIO`. `default` and `transcribe` are refused, because the audio pipeline workers read them without a job-type filter. The TTS jobs worker reads that queue and takes only `tts_longform` jobs. It runs in-process wherever the audio routes do (`TTS_JOBS_WORKER_ENABLED`, default on by route policy); in sidecar mode it runs as the `tts` sidecar worker (`Dockerfiles/docker-compose.workers.yml`), and it can also run standalone as `python -m tldw_Server_API.app.core.TTS.tts_jobs_worker`.

 ## STT (condensed)

diff --git a/Docs/Architecture/jobs-background.md b/Docs/Architecture/jobs-background.md
index c1dbcfd270..4fbc3094b6 100644
--- a/Docs/Architecture/jobs-background.md
+++ b/Docs/Architecture/jobs-background.md
@@ -30,7 +30,7 @@ One generic `jobs` table (SQLite or PG — PG iff `JOBS_DB_URL` starts with `pos
 ## Execution models

 - **In-process pollers**: asyncio loops acquire jobs (poll ~1 s, lease default 120 s), process, and ack with worker+lease+completion-token. Managed by the lifecycle registry; app shutdown closes the acquire gate first so acquisitions return None, then quiesces pollers.
-- **Sidecar mode**: `TLDW_WORKERS_SIDECAR_MODE=true` makes the API skip in-process pollers; `start-sidecars.sh` launches worker processes from `Docs/Deployment/sidecar_workers_manifest.json` (chatbooks, files, data_tables, prompt_studio, privilege_snapshots, audio, media_ingest, reading_import/digest, evals_abtest, optional connectors).
+- **Sidecar mode**: `TLDW_WORKERS_SIDECAR_MODE=true` makes the API skip in-process pollers; `start-sidecars.sh` launches worker processes from `Docs/Deployment/sidecar_workers_manifest.json` (chatbooks, files, data_tables, prompt_studio, privilege_snapshots, audio, tts, media_ingest, reading_import/digest, evals_abtest, optional connectors).
 - **PG concurrency**: single-UPDATE `SKIP LOCKED` acquisition when enabled, else two-step pick+update; expired-lease recovery batched with `FOR UPDATE SKIP LOCKED` and a per-job policy (`consume_retry` vs `requeue_no_attempt`); advisory locks guard the dependency graph. Lease enforcement on ack is on by default (`JOBS_DISABLE_LEASE_ENFORCEMENT` to disable).

 ## Jobs vs Scheduler (the decision rule)
diff --git a/Docs/Deployment/Long_Term_Admin_Guide.md b/Docs/Deployment/Long_Term_Admin_Guide.md
index c0f4470c8f..4c0094ac36 100644
--- a/Docs/Deployment/Long_Term_Admin_Guide.md
+++ b/Docs/Deployment/Long_Term_Admin_Guide.md
@@ -139,7 +139,7 @@ Circuit breaker observability and tuning
 Application
 - CPU workers: set `UVICORN_WORKERS` (default 4 in Dockerfile.prod). Monitor latency and CPU to tune.
 - Caching: ensure embeddings/model caches reside on fast disk; configure per module where available.
-- Background jobs: Chatbooks worker enabled by default (core backend). Control via `CHATBOOKS_CORE_WORKER_ENABLED`. Media ingest jobs worker follows the `media` route policy by default; control via `MEDIA_INGEST_JOBS_WORKER_ENABLED`, or use sidecar workers for multi-worker deployments.
+- Background jobs: Chatbooks worker enabled by default (core backend). Control via `CHATBOOKS_CORE_WORKER_ENABLED`. Media ingest jobs worker follows the `media` route policy by default; control via `MEDIA_INGEST_JOBS_WORKER_ENABLED`, or use sidecar workers for multi-worker deployments. The long-form TTS jobs worker (`POST /api/v1/audio/speech/jobs`) follows the `audio` route policy the same way; control via `TTS_JOBS_WORKER_ENABLED`, and in sidecar mode run the `tts` sidecar worker.
 - SQLite deployments: avoid multiple Uvicorn workers with in-process jobs. Use sidecar workers or Postgres for higher concurrency.

 Database
diff --git a/Docs/Operations/Env_Vars.md b/Docs/Operations/Env_Vars.md
index 2f8b812983..5f0295c6af 100644
--- a/Docs/Operations/Env_Vars.md
+++ b/Docs/Operations/Env_Vars.md
@@ -287,9 +287,11 @@ Notes:
 - `CHATBOOKS_IMPORT_DICT_STRICT`: When true, skip dictionaries with fatal validation errors instead of importing with warnings.

 ## Audio Jobs
-- `AUDIO_JOBS_WORKER_ENABLED`: Enable the in-process Audio Jobs worker (`true|false`, default follows route policy for `audio-jobs`). When true, the worker starts at app startup and polls the Jobs backend for the `audio` domain pipeline stages.
+- `AUDIO_JOBS_WORKER_ENABLED`: Enable the in-process Audio Jobs worker (`true|false`, default `false`; it also needs the `audio-jobs` route enabled). When true, the worker starts at app startup and polls the Jobs backend for the `audio` domain pipeline stages.
 - `AUDIO_JOBS_OWNER_STRICT`: Enable owner-aware acquisition for fairness across users (`true|false`, default `false`). When enabled, the worker preferentially acquires jobs for owners under their concurrent-job caps.
 - `AUDIO_JOBS_GPU_TRANSCRIBE`: Send the `audio_transcribe` stage to the GPU transcription worker's `transcribe` queue instead of transcribing in the audio jobs worker (`true|false`, default `false`). Set it where the audio jobs worker runs, and only when a GPU worker (`Dockerfiles/Dockerfile.audio_gpu_worker`) reads the same Jobs database.
+- `TTS_JOBS_WORKER_ENABLED`: Run the long-form TTS jobs worker in-process, for `POST /api/v1/audio/speech/jobs` (`true|false`, default follows route policy for `audio`, so it runs wherever the audio routes do). It does not start in sidecar mode (`TLDW_WORKERS_SIDECAR_MODE`), where the `tts` sidecar worker runs instead. Set `false` when a standalone worker (`python -m tldw_Server_API.app.core.TTS.tts_jobs_worker`) handles these jobs. A server restart during a synthesis fails that job (`tts_replay_blocked`), so run the `tts` sidecar or a standalone worker where restarts are frequent. With large local TTS models on a GPU and several server processes (`UVICORN_WORKERS`), set it to `false` and run one dedicated worker, since each process loads its own model.
+- `TTS_JOBS_QUEUE`: Jobs queue for long-form TTS jobs (default `tts`). Set the same value for the API server and any standalone TTS worker, and register a custom name in `JOBS_ALLOWED_QUEUES_AUDIO`; `default` and `transcribe` are refused.
 - `REDIS_URL`: Optional Redis URL used by Resource Governor when `RG_BACKEND=redis`.

 ## Media Ingest Jobs
```

Then regenerate and mirror:

```bash
$PY Helper_Scripts/Deployment/generate_sidecar_files.py
bash Helper_Scripts/refresh_docs_published.sh
git status --short Dockerfiles Docs
```

Expected:
- Modified (11):
  - `Dockerfiles/docker-compose.workers.yml`, with a new `worker-tts` service whose command is `python -m tldw_Server_API.app.core.TTS.tts_jobs_worker` (check with `git diff Dockerfiles`);
  - `Docs/Architecture/audio.md` and `Docs/Architecture/jobs-background.md`;
  - `Docs/Deployment/Long_Term_Admin_Guide.md` and `Docs/Published/Deployment/Long_Term_Admin_Guide.md`;
  - `Docs/Deployment/Sidecar_Workers.md` and `Docs/Published/Deployment/Sidecar_Workers.md`;
  - `Docs/Deployment/sidecar_workers_manifest.json` and `Docs/Published/Deployment/sidecar_workers_manifest.json`;
  - `Docs/Operations/Env_Vars.md` and `Docs/Published/Env_Vars.md`.
- Untracked:
  - the six new `tldw-worker-tts.service`, `tldw-worker-tts.timer` and `com.tldw.worker.tts.plist` files under `Docs/Deployment/` and `Docs/Published/Deployment/`;
  - this plan.

- [ ] **Step 7: Run the Services, sidecar, startup, Docs, TTS and lint tests**

```bash
git add -N Docs/Deployment/systemd/tldw-worker-tts.service Docs/Deployment/systemd/tldw-worker-tts.timer Docs/Deployment/launchd/com.tldw.worker.tts.plist Docs/Published/Deployment/systemd/tldw-worker-tts.service Docs/Published/Deployment/systemd/tldw-worker-tts.timer Docs/Published/Deployment/launchd/com.tldw.worker.tts.plist
RUN_JOBS=1 $PY -m pytest -p no:cacheprovider -q tldw_Server_API/tests/Services tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py tldw_Server_API/tests/Utils/test_sidecar_profiles.py tldw_Server_API/tests/Utils/test_start_sidecars_dry_run.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_startup.py tldw_Server_API/tests/Research_Workspace/test_output_jobs_startup.py tldw_Server_API/tests/Slides/test_standalone_html_api.py tldw_Server_API/tests/DB_Management/test_media_db_api_imports.py tldw_Server_API/tests/Docs tldw_Server_API/tests/TTS_NEW tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py tldw_Server_API/tests/Audio/test_tts_job_dispatch_marker.py tldw_Server_API/tests/lint
```

The `git add -N` registers the new generated files. `tests/Docs/test_docs_published_refresh.py` compares the refresh output against tracked files, and fails while they are untracked.

The run covers every test that references the changed modules or the sidecar manifest. Run it serially; it takes about 22 minutes.

Expected: `1 failed, 3411 passed, 8 skipped, 1 xfailed, 1 xpassed`, measured on the prototype. The one failure does not come from this change: `tests/TTS_NEW/unit/test_tts_service.py::TestTextGeneration::test_chunked_retry_metadata` fails on unchanged `origin/dev` too (`ModuleNotFoundError: No module named 'av'`, an optional dependency missing locally). `tests/lint/test_rg_route_map_lint.py::test_lint_sees_the_routes_a_fresh_app_serves` has once aborted natively in a long run (`libc++abi: ... recursive_mutex lock failed`, exit -6), and passes on its own on both `origin/dev` and the change. If anything else fails, compare with `origin/dev` before going on.

- [ ] **Step 8: Lint and security**

```bash
$PY -m ruff check --output-format concise tldw_Server_API/app/services/startup_content_jobs_pollers.py Helper_Scripts/common/sidecar_profiles.py $T1
uvx bandit -q -ll tldw_Server_API/app/services/startup_content_jobs_pollers.py Helper_Scripts/common/sidecar_profiles.py
```

Expected:
- **Ruff** prints exactly one finding, `tldw_Server_API/tests/Utils/test_sidecar_manifest_defaults.py:1:1: I001`, and exits 1. `origin/dev` has the same finding; this change does not touch that import block, and fixing it is out of scope.
- **Bandit** reports no findings and exits 0. On its first run, `uvx` prints package download lines first.

- [ ] **Step 9: Commit the change**

```bash
git add tldw_Server_API/app/services/startup_content_jobs_pollers.py Helper_Scripts/common/sidecar_profiles.py $T1 Dockerfiles/docker-compose.workers.yml Docs/Deployment Docs/Published/Deployment Docs/Operations/Env_Vars.md Docs/Published/Env_Vars.md Docs/Architecture/audio.md Docs/Architecture/jobs-background.md
git status --short
git commit -m "feat(tts): the long-form TTS jobs worker runs in-process by default and as a sidecar worker (TASK-13849)

POST /api/v1/audio/speech/jobs queues long-form TTS jobs and the WebUI
offers it, but nothing started the TTS jobs worker: no lifespan spec, no
sidecar entry, no service unit. Jobs stayed queued with nothing to say why.

The worker now runs in-process wherever the audio routes do (tts_jobs_task,
default by route policy like the media ingest workers; off in sidecar and
test mode; TTS_JOBS_WORKER_ENABLED=false turns it off), and as the tts
sidecar worker, with its compose service and systemd/launchd units
generated from the sidecar manifest. The tts-only sidecar profile now
selects it.

Env_Vars.md documents TTS_JOBS_WORKER_ENABLED and TTS_JOBS_QUEUE, and
corrects AUDIO_JOBS_WORKER_ENABLED, which needs the flag set explicitly.

Upgrade note: sidecar deployments with hand-installed systemd/launchd
units, or an explicit TLDW_SIDECAR_WORKERS list, must add the tts worker
themselves.

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git status --short` before the commit shows no unstaged change under `Docs`, `Dockerfiles`, `Helper_Scripts` or `tldw_Server_API`, other than this plan.

- [ ] **Step 10: Close the task and commit the plan**

The notes below quote the measured results. If Steps 3, 5, 7 or 8 printed different lines, put the actual ones in the notes.

```bash
PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd . task edit TASK-13849 --append-notes "Fixed: app/services/startup_content_jobs_pollers.py gains the tts_jobs_task worker spec (route policy on the audio route via media_ingest_worker_predicate / should_start_inprocess_worker: on by default wherever the audio routes run, off in sidecar and test mode, TTS_JOBS_WORKER_ENABLED=false turns it off), running core/TTS/tts_jobs_worker.run_tts_jobs_worker; the server's startup already refreshes LLM provider overrides. The sidecar manifest gains the tts worker (default) and the tts-only profile selects it; the generator added the worker-tts compose service and tldw-worker-tts systemd/launchd units. Docs/Operations/Env_Vars.md documents TTS_JOBS_WORKER_ENABLED (with the restart and GPU notes) and TTS_JOBS_QUEUE and corrects AUDIO_JOBS_WORKER_ENABLED; Docs/Architecture/audio.md, jobs-background.md and Docs/Deployment/Long_Term_Admin_Guide.md say where the worker runs. Chosen over refusing long-form jobs when no worker runs, because the endpoint cannot see a standalone or sidecar worker elsewhere. Upgrade note: sidecar deployments with hand-installed units or an explicit TLDW_SIDECAR_WORKERS list must add tts. Tests: RED 9 failed, 72 passed on dev (no tts_jobs_task spec, no tts sidecar worker), GREEN 81 passed; Services, sidecar, startup, Docs, TTS_NEW, the TTS Audio tests and lint 1 failed, 3411 passed, 8 skipped, 1 xfailed, 1 xpassed. Ruff clean on the changed files apart from the I001 dev already has; bandit clean." --check-ac 1 --check-ac 2 --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 --final-summary "The long-form TTS jobs worker runs in-process wherever the audio routes do (on by default by route policy, off in sidecar and test mode, TTS_JOBS_WORKER_ENABLED=false turns it off) and as the tts sidecar worker, so jobs from POST /api/v1/audio/speech/jobs and the WebUI are processed in a default deployment." --status Done
git add -A backlog/tasks
git add Docs/superpowers/plans/2026-10-18-tts-jobs-worker-in-process.md
git commit -m "docs(tts): plan for running the long-form TTS jobs worker by default; close TASK-13849

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>"
```

Expected: `git show --stat HEAD` lists the task file and the plan only.
