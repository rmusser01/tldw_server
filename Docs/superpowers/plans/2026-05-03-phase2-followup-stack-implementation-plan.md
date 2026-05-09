# Phase 2 Follow-Up Stack Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete the remaining #1116 Phase 2 follow-up extraction debt as conservative, test-covered PRs after the Phase 4 roadmap stack has closed.

**Architecture:** Treat this as a stacked work program, not one large refactor. Each implementation branch starts from current `origin/dev`, owns one subsystem tranche, preserves public behavior, runs focused verification, and updates #1116 after merge. PR #1237/OpenAPI raw response contracts stay separate and must not be included in this stack.

**Tech Stack:** FastAPI startup/lifespan services, router group `RouterSpec` helpers, ChaChaNotes store/facade delegation, pytest, Bandit, Git worktrees, Backlog.md, GitHub PRs against `dev`.

---

## Source Spec

- `Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md`
- Backlog planning task: `TASK-8`
- Roadmap tracker: https://github.com/rmusser01/tldw_server/issues/1116

## Scope Check

This design spans three independent implementation surfaces plus one optional enabling lane:

- Phase 2.1: lifecycle and startup cleanup
- Phase 2.2: conditional router group cleanup
- Phase 2.3: ChaChaNotes PersonaStateStore/facade delegation
- Phase 2.4: typed config follow-up only if it unblocks a concrete Phase 2.1 or 2.2 tranche

Do not execute this as one PR. Each task below should be its own reviewable branch/PR unless the task says it is docs-only.

## Shared Rules For Every Implementation Task

- Start from a clean worktree based on current `origin/dev`.
- Create or reuse a Backlog.md task before editing repo files.
- Keep one behavioral invariant per PR.
- Write or strengthen tests before moving behavior when practical.
- Preserve public route paths, payloads, method signatures, and compatibility re-exports.
- Run `source .venv/bin/activate` before `python`, `pytest`, or `bandit`.
- Run `git diff --check` before every commit.
- Run Bandit on touched Python source before PR completion.
- Add the required human-authored PR `Change summary` before merge.
- After merge, update #1116 with the PR link, scope, verification, and remaining debt.

## File Structure

### Phase 2.1 Lifecycle/Init Cleanup

- Modify likely: `tldw_Server_API/app/services/lifecycle_workers.py`
  - Shared `WorkerRegistry`, `ManagedWorker`, `ShutdownPhase`, and stop behavior.
- Modify likely: `tldw_Server_API/app/services/shutdown_legacy_adapters.py`
  - Legacy direct-stop plan and compatibility stop paths.
- Modify likely: `tldw_Server_API/app/services/lifespan_shutdown_sequence.py`
  - Shutdown orchestration if legacy inputs can be reduced.
- Modify likely: `tldw_Server_API/app/services/startup_cleanup_workers.py`
  - First candidate overlap: `chatbooks_cleanup` registry ownership plus legacy handle return.
- Modify likely: `tldw_Server_API/app/services/startup_worker_groups.py`
  - Bridge object currently carries cleanup worker handles through startup.
- Modify likely: `tldw_Server_API/app/main.py`
  - Only if shutdown context inputs or aliases need reduction.
- Test likely: `tldw_Server_API/tests/Services/test_lifecycle_workers.py`
- Test likely: `tldw_Server_API/tests/Services/test_startup_cleanup_workers.py`
- Test likely: `tldw_Server_API/tests/Services/test_shutdown_coordinated_legacy_components.py`
- Test likely: `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`

### Phase 2.2 Router Conditional Groups

- Modify likely: `tldw_Server_API/app/api/v1/router_groups/spec.py`
  - Only for small helper types if needed.
- Modify likely: `tldw_Server_API/app/api/v1/router_groups/admin.py`
  - First router target: sandbox conditional registration.
- Modify likely: `tldw_Server_API/app/api/v1/router_groups/core.py`
  - First/second router target: ACP route-family conditional registration.
- Modify likely: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
  - Preserve minimal-test-app equivalents after helper extraction.
- Create maybe: `tldw_Server_API/app/api/v1/router_groups/conditional.py`
  - Shared lazy-router helper only if it removes repeated conditional import logic.
- Test: `tldw_Server_API/tests/Services/test_router_groups_contract.py`
- Test: `tldw_Server_API/tests/Services/test_main_router_contract.py`
- Test: `tldw_Server_API/tests/Services/test_openapi_contracts.py`

### Phase 2.3 ChaChaNotes Persona Delegation

- Modify likely: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
  - Compatibility facade and method delegation list.
- Modify likely: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
  - Focused store for persona state methods.
- Create maybe: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_helpers.py`
  - Pure helpers only if new coverage identifies duplicated non-SQL logic.
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py`
- Test: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_profiles_api.py`
- Test: `tldw_Server_API/tests/Persona/test_persona_sessions.py`

### Optional Phase 2.4 Config Follow-Up

- Modify likely: `tldw_Server_API/app/core/config_sections/`
  - Only the section needed by a Phase 2.1/2.2 refactor.
- Test: `tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py`
- Test: `tldw_Server_API/tests/Services/test_startup_validation.py`
- Test: relevant router or startup test from the tranche that needs the typed section.

## Task 1: Create The First Phase 2.1 Worktree And Backlog Task

**Files:**
- Create worktree: `.worktrees/phase2-1-worker-lifecycle-cleanup-a`
- Create branch: `codex/phase2-1-worker-lifecycle-cleanup-a`
- Create Backlog task: title `Phase 2.1 worker lifecycle cleanup A`

- [ ] **Step 1: Refresh remote state**

Run:

```bash
git fetch origin
```

Expected: fetch succeeds. If network fails, stop and report that current `origin/dev` could not be refreshed.

- [ ] **Step 2: Create a clean worktree**

Run:

```bash
git worktree add .worktrees/phase2-1-worker-lifecycle-cleanup-a -b codex/phase2-1-worker-lifecycle-cleanup-a origin/dev
```

Expected: worktree is created and `git status --short --branch` inside it is clean.

- [ ] **Step 3: Create the Backlog task**

Use MCP `task_create` from the new worktree context.

Required fields:

- Title: `Phase 2.1 worker lifecycle cleanup A`
- References: `https://github.com/rmusser01/tldw_server/issues/1116`
- Documentation: `Docs/superpowers/specs/2026-05-03-phase2-followup-stack-design.md`, this plan file, and `Docs/superpowers/specs/2026-05-03-worker-lifecycle-deprecated-code-removal-design.md` if that design is committed/available on the branch.
- Acceptance criteria:
  - A focused lifecycle ownership test proves the targeted worker is registry-owned in the expected shutdown phase.
  - The targeted worker no longer has an unguarded duplicate legacy direct-stop path.
  - Startup/shutdown behavior and app-state inventory semantics are preserved.
  - Focused lifecycle/startup/shutdown tests, Bandit touched-source scope, and `git diff --check` pass.

- [ ] **Step 4: Record the starting state**

Run:

```bash
git status --short --branch
git log --oneline -5
```

Expected: branch is based on `origin/dev`, clean, and has no unrelated local changes.

## Task 2: Phase 2.1 Ownership Test Before Deletion

**Files:**
- Modify: `tldw_Server_API/tests/Services/test_startup_cleanup_workers.py`
- Modify: `tldw_Server_API/tests/Services/test_shutdown_coordinated_legacy_components.py`
- Modify maybe: `tldw_Server_API/tests/Services/test_main_lifecycle_contract.py`

- [ ] **Step 1: Inspect current cleanup worker ownership**

Run:

```bash
rg -n "chatbooks_cleanup|stopped_background_worker_names|shutdown_legacy|WorkerRegistry" tldw_Server_API/app/services tldw_Server_API/tests/Services
```

Expected: output shows `chatbooks_cleanup` is registered through `WorkerRegistry` but still appears in legacy shutdown compatibility paths.

- [ ] **Step 2: Write a failing ownership regression test**

Add a test near the existing `chatbooks_cleanup` tests in `tldw_Server_API/tests/Services/test_startup_cleanup_workers.py`.

The test should prove:

- `_start_chatbooks_cleanup_worker(worker_inventory=WorkerRegistry(app))` registers `chatbooks_cleanup`.
- The worker is in `ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN`.
- `app.state._tldw_shutdown_job_poller_inventory` does not include it.
- `app.state._tldw_shutdown_worker_inventory` includes it exactly once.

Skeleton:

```python
async def test_chatbooks_cleanup_has_single_background_registry_owner(monkeypatch):
    from types import SimpleNamespace

    from tldw_Server_API.app.services.lifecycle_workers import ShutdownPhase, WorkerRegistry
    from tldw_Server_API.app.services import startup_cleanup_workers as startup_cleanup

    monkeypatch.setenv("CHATBOOKS_CLEANUP_INTERVAL_SEC", "60")

    async def _fake_runner(stop_event):
        await stop_event.wait()

    monkeypatch.setattr(startup_cleanup, "_run_chatbooks_cleanup_loop", _fake_runner)

    app = SimpleNamespace(state=SimpleNamespace())
    worker_inventory = WorkerRegistry(app)
    task, stop_event = await startup_cleanup._start_chatbooks_cleanup_worker(
        worker_inventory=worker_inventory
    )
    try:
        handles = worker_inventory.handles_for_phase(ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN)
        assert [handle.name for handle in handles].count("chatbooks_cleanup") == 1
        assert app.state._tldw_shutdown_job_poller_inventory == []
        assert [
            item for item in app.state._tldw_shutdown_worker_inventory
            if item["name"] == "chatbooks_cleanup"
        ][0]["shutdown_phase"] == ShutdownPhase.BACKGROUND_WORKER_SHUTDOWN.value
    finally:
        stop_event.set()
        await task
```

Adjust to match existing test utilities and cleanup style in the file.

- [ ] **Step 3: Run the focused red test**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_startup_cleanup_workers.py -k "chatbooks_cleanup_has_single_background_registry_owner" -q
```

Expected: fail if current coverage does not assert the single-owner invariant. If it passes because equivalent coverage already exists, record that and continue to the legacy direct-stop test.

- [ ] **Step 4: Write a failing legacy-suppression test**

In `tldw_Server_API/tests/Services/test_shutdown_coordinated_legacy_components.py`, add or strengthen a test proving the legacy coordinator does not directly stop `chatbooks_cleanup` when `stopped_background_worker_names` already contains `chatbooks_cleanup`.

Expected assertion shape:

```python
assert summary.components["chatbooks_cleanup"].result == "skipped"
```

If the current behavior already passes, add a narrower test that fails only after removing the wrong branch would regress double-stop prevention.

- [ ] **Step 5: Run the focused red/guard tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_startup_cleanup_workers.py -k "chatbooks_cleanup" -q
python -m pytest tldw_Server_API/tests/Services/test_shutdown_coordinated_legacy_components.py -k "chatbooks_cleanup" -q
```

Expected: tests either expose the missing invariant or establish the guard before deletion.

## Task 3: Phase 2.1 Remove One Deprecated Direct-Stop Path

**Files:**
- Modify likely: `tldw_Server_API/app/services/shutdown_legacy_adapters.py`
- Modify likely: `tldw_Server_API/app/services/startup_cleanup_workers.py`
- Modify likely: `tldw_Server_API/app/services/startup_worker_groups.py`
- Modify maybe: `tldw_Server_API/app/main.py`
- Modify tests from Task 2 as needed.

- [ ] **Step 1: Identify the smallest safe deletion**

Run:

```bash
rg -n "chatbooks_cleanup_task|chatbooks_cleanup_stop_event|chatbooks_cleanup" tldw_Server_API/app/services tldw_Server_API/app/main.py tldw_Server_API/tests/Services
```

Expected: direct handle flow is visible from startup cleanup into shutdown legacy context.

- [ ] **Step 2: Remove only the duplicate direct-stop behavior**

Implementation target:

- Keep `WorkerRegistry` registration for `chatbooks_cleanup`.
- Keep app-state diagnostics.
- Remove or narrow legacy direct stop so `chatbooks_cleanup` is not stopped by both registry shutdown and legacy direct shutdown.
- If removing the handle fields is too broad, first change the legacy adapter so registry-owned `chatbooks_cleanup` is represented as skipped, then defer field removal to a second PR.

Do not change:

- `CHATBOOKS_CLEANUP_INTERVAL_SEC` semantics
- cleanup loop behavior
- shutdown timeout behavior
- unrelated cleanup workers

- [ ] **Step 3: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_startup_cleanup_workers.py -k "chatbooks_cleanup" -q
python -m pytest tldw_Server_API/tests/Services/test_shutdown_coordinated_legacy_components.py -q
python -m pytest tldw_Server_API/tests/Services/test_lifecycle_workers.py -q
```

Expected: all pass.

- [ ] **Step 4: Run broader lifecycle contract tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_main_lifecycle_contract.py -k "chatbooks_cleanup or legacy or worker_inventory" -q
python -m pytest tldw_Server_API/tests/Services/test_lifespan_shutdown_sequence.py -q
```

Expected: all pass or only unrelated baseline skips.

- [ ] **Step 5: Run Bandit and diff checks**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/services/lifecycle_workers.py tldw_Server_API/app/services/shutdown_legacy_adapters.py tldw_Server_API/app/services/startup_cleanup_workers.py tldw_Server_API/app/services/startup_worker_groups.py -f json -o /tmp/bandit_phase2_1_lifecycle_cleanup_a.json
git diff --check
git status --short --branch
```

Expected: Bandit reports no new findings in touched source; diff check is clean.

- [ ] **Step 6: Commit**

Run:

```bash
git add tldw_Server_API/app/services tldw_Server_API/tests/Services backlog/tasks
git commit -m "Phase 2.1: remove duplicate chatbooks cleanup shutdown path"
```

Expected: commit includes only the focused lifecycle cleanup and its Backlog task update.

- [ ] **Step 7: Push and open PR**

Run:

```bash
git push -u origin codex/phase2-1-worker-lifecycle-cleanup-a
gh pr create --base dev --head codex/phase2-1-worker-lifecycle-cleanup-a --title "Phase 2.1: clean up worker lifecycle ownership" --body-file /tmp/phase2_1_lifecycle_cleanup_a_pr.md
```

The PR body must include:

- `Change summary` placeholder for human-authored rationale if the author cannot write it personally.
- Scope and behavior-preservation notes.
- Test plan with exact commands.
- Bandit touched-scope result.

## Task 4: Phase 2.2 Router Conditional Helper Tranche

**Files:**
- Create maybe: `tldw_Server_API/app/api/v1/router_groups/conditional.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/admin.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/core.py`
- Modify: `tldw_Server_API/app/api/v1/router_groups/minimal.py`
- Modify: `tldw_Server_API/tests/Services/test_router_groups_contract.py`

- [ ] **Step 1: Create worktree and Backlog task**

Run:

```bash
git fetch origin
git worktree add .worktrees/phase2-2-router-conditionals-a -b codex/phase2-2-router-conditionals-a origin/dev
```

Create Backlog task:

- Title: `Phase 2.2 router conditional cleanup A`
- Acceptance criteria:
  - Sandbox and/or ACP router conditional specs preserve prefix, tags, route keys, default stability, and lazy import behavior.
  - Repeated conditional import logic is extracted only where tests cover the result.
  - Minimal test app behavior remains unchanged.
  - Focused router contract tests, OpenAPI route contract tests if applicable, Bandit, and `git diff --check` pass.

- [ ] **Step 2: Add characterization tests before extraction**

In `tldw_Server_API/tests/Services/test_router_groups_contract.py`, add or strengthen tests for the selected first target.

For sandbox:

- `iter_admin_router_specs()` returns sandbox with prefix `/api/v1`, tags `("sandbox",)`, route key `"sandbox"`, and `default_stable is False`.
- `iter_minimal_test_router_specs()` returns sandbox with prefix `/api/v1`, tags `("sandbox",)`, and route key `""`.
- lazy import is still deferred through factory if the target code already uses a factory.

For ACP:

- `iter_core_router_specs()` or `iter_minimal_test_router_specs()` returns ACP route families with the same tags and route keys as before.
- no direct import of ACP endpoint modules is reintroduced into `main.py`.

- [ ] **Step 3: Run characterization tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -k "sandbox or acp or ACP" -q
```

Expected: tests pass before refactor, proving the current contract.

- [ ] **Step 4: Extract the smallest helper**

If duplication justifies it, create `tldw_Server_API/app/api/v1/router_groups/conditional.py`.

Helper constraints:

- Return `RouterSpec` instances only.
- Do not call `app.include_router`.
- Do not read runtime route policy directly.
- Keep optional import failures scoped to the existing router group behavior.
- Preserve logging message content unless tests intentionally update it.

If a helper would be larger than the duplicated code it removes, do not create it. Instead, make the smallest local cleanup in the router group file and stop.

- [ ] **Step 5: Run focused router verification**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_router_groups_contract.py -q
python -m pytest tldw_Server_API/tests/Services/test_main_router_contract.py -q
```

Expected: all pass.

- [ ] **Step 6: Run generated contract verification if route output changed**

Run when route registration code changed:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Services/test_openapi_contracts.py -q
```

Expected: pass. If failures show intended route metadata changes, stop and reassess because this tranche should preserve behavior.

- [ ] **Step 7: Run Bandit and commit**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/api/v1/router_groups -f json -o /tmp/bandit_phase2_2_router_conditionals_a.json
git diff --check
git status --short --branch
```

Expected: no new Bandit findings in touched source; diff check clean.

Commit:

```bash
git add tldw_Server_API/app/api/v1/router_groups tldw_Server_API/tests/Services/test_router_groups_contract.py backlog/tasks
git commit -m "Phase 2.2: extract covered router conditionals"
```

## Task 5: Phase 2.3 ChaChaNotes Persona Delegation Tranche

**Files:**
- Modify: `tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py`
- Modify: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py`
- Create maybe: `tldw_Server_API/app/core/DB_Management/chacha/persona_state_helpers.py`
- Modify: `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py`
- Modify maybe: `tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py`

- [ ] **Step 1: Create worktree and Backlog task**

Run:

```bash
git fetch origin
git worktree add .worktrees/phase2-3-chacha-persona-delegation-a -b codex/phase2-3-chacha-persona-delegation-a origin/dev
```

Create Backlog task:

- Title: `Phase 2.3 ChaChaNotes persona delegation A`
- Acceptance criteria:
  - Selected persona method family has public `CharactersRAGDB` facade coverage before movement.
  - Implementation moves only covered behavior into `PersonaStateStore` or a pure helper behind it.
  - Public method names, signatures, sync logging, and schema behavior remain compatible.
  - Focused ChaChaNotes/persona tests, Bandit, and `git diff --check` pass.

- [ ] **Step 2: Inventory remaining inline persona-related code**

Run:

```bash
rg -n "persona|PersonaStateStore|_delegate_store_method|for _persona_state_store_method" tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py
```

Expected: bottom delegation list exists and many persona methods are already delegated.

- [ ] **Step 3: Select one method family**

Start with a small method family that is already mostly delegated and has direct tests. Prefer:

- row-to-dict normalization helpers
- exemplar tag/tone normalization helpers
- persona memory archive/delete/list helpers

Do not start with:

- schema migration code
- sync log code
- broad profile/session CRUD rewrites
- code that requires changing public method signatures

- [ ] **Step 4: Write or strengthen facade tests**

In `tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py` or `test_persona_persistence_db.py`, add tests that call the public `CharactersRAGDB` method, not only `PersonaStateStore`.

Example shape:

```python
def test_characters_rag_db_persona_memory_facade_matches_store(tmp_path):
    db = CharactersRAGDB(str(tmp_path / "ChaChaNotes.db"), client_id="persona-delegation-test")
    # create profile/session/memory through public facade
    # assert public facade returns the same normalized shape expected from PersonaStateStore
```

Use existing fixtures and helper factories in the file instead of inventing new DB setup when possible.

- [ ] **Step 5: Run red/guard tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -k "persona" -q
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py -q
```

Expected: new tests fail if they assert new coverage, or pass as guard tests if they characterize existing behavior.

- [ ] **Step 6: Move only covered implementation**

Implementation constraints:

- If code is pure normalization, move it to `persona_state_helpers.py` only if reuse or file-size reduction justifies the extra file.
- If code needs DB access, keep it on `PersonaStateStore`.
- Update the bottom `_persona_state_store_method` delegation list only for methods fully owned by the store.
- Leave uncovered monolith methods in `ChaChaNotes_DB.py`.

- [ ] **Step 7: Run focused tests**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_chacha_persona_state_store.py -q
python -m pytest tldw_Server_API/tests/ChaChaNotesDB/test_persona_persistence_db.py -q
python -m pytest tldw_Server_API/tests/Persona/test_persona_profiles_api.py tldw_Server_API/tests/Persona/test_persona_sessions.py -q
```

Expected: all pass.

- [ ] **Step 8: Run Bandit and commit**

Run:

```bash
source .venv/bin/activate
python -m bandit -r tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha/persona_state_store.py -f json -o /tmp/bandit_phase2_3_chacha_persona_a.json
git diff --check
git status --short --branch
```

Expected: no new Bandit findings in touched source; diff check clean.

Commit:

```bash
git add tldw_Server_API/app/core/DB_Management/ChaChaNotes_DB.py tldw_Server_API/app/core/DB_Management/chacha tldw_Server_API/tests/ChaChaNotesDB tldw_Server_API/tests/Persona backlog/tasks
git commit -m "Phase 2.3: delegate covered persona state behavior"
```

## Task 6: Optional Phase 2.4 Enabling Config Tranche

Only run this task if a Phase 2.1 or Phase 2.2 tranche identifies repeated config parsing that blocks safe extraction.

**Files:**
- Modify likely: `tldw_Server_API/app/core/config_sections/__init__.py`
- Modify likely: one section file under `tldw_Server_API/app/core/config_sections/`
- Modify: `tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py`
- Modify: relevant startup/router tests from the blocking tranche

- [ ] **Step 1: Prove the enabling dependency**

Before creating the branch, write down which tranche is blocked and why typed config is the smallest safe fix.

Examples:

- A startup helper repeats the same env/config parsing in multiple places.
- Router gating needs a single typed default source to avoid conditional drift.
- A test exposes mismatch between documented config and runtime parsing.

If there is no concrete blocker, skip Phase 2.4.

- [ ] **Step 2: Create worktree and task**

Run:

```bash
git fetch origin
git worktree add .worktrees/phase2-4-config-followup-a -b codex/phase2-4-config-followup-a origin/dev
```

Create Backlog task:

- Title: `Phase 2.4 config follow-up A`
- Acceptance criteria must name the tranche it unblocks.

- [ ] **Step 3: Add typed-loader tests first**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py -q
```

Expected: baseline passes before changes.

Add tests for the new typed section or option. Run them red if the helper does not exist yet.

- [ ] **Step 4: Implement minimal typed section support**

Constraints:

- Do not change global config precedence.
- Do not rewrite unrelated config sections.
- Do not move startup behavior in the same PR unless it is the named unblocker.

- [ ] **Step 5: Verify and commit**

Run:

```bash
source .venv/bin/activate
python -m pytest tldw_Server_API/tests/Config/test_config_sections_typed_loaders.py -q
python -m pytest tldw_Server_API/tests/Services/test_startup_validation.py -q
python -m bandit -r tldw_Server_API/app/core/config_sections -f json -o /tmp/bandit_phase2_4_config_followup_a.json
git diff --check
```

Commit:

```bash
git add tldw_Server_API/app/core/config_sections tldw_Server_API/tests/Config tldw_Server_API/tests/Services backlog/tasks
git commit -m "Phase 2.4: add typed config follow-up"
```

## Task 7: Per-PR Closeout And #1116 Update

**Files:**
- Modify: Backlog task for the implemented tranche
- GitHub: update #1116 after merge

- [ ] **Step 1: Before PR review, verify branch hygiene**

Run:

```bash
git status --short --branch
git log --oneline origin/dev..HEAD
```

Expected: only intentional commits are ahead of `origin/dev`.

- [ ] **Step 2: Ensure PR body has required sections**

PR body must contain:

```markdown
## Change summary
Human-authored summary required before merge.

## What changed
- ...

## Why this shape
- ...

## Test plan
- ...

## Bandit
- ...
```

If a human-written summary has already been provided, replace the placeholder with it before merge.

- [ ] **Step 3: After merge, comment on #1116**

Use this template:

```text
Phase <phase> follow-up update (<date>):

- Merged PR #<number>: <url>
- Scope: <one sentence>
- Behavior boundary: <what did not change>
- Verification: <focused tests>, Bandit touched scope, git diff --check
- Remaining #1116 follow-up: <next tranche or none for this phase>
```

- [ ] **Step 4: Finalize Backlog task**

Use MCP `task_edit`:

- check acceptance criteria
- check Definition of Done
- add final summary
- document skipped tests or baseline blockers
- set status `Done`

## Recommended First Execution Order

1. Task 1: create `codex/phase2-1-worker-lifecycle-cleanup-a`.
2. Task 2: add ownership tests for `chatbooks_cleanup`.
3. Task 3: remove one duplicate direct-stop path if tests prove it is safe.
4. Open PR and update #1116 after merge.
5. Start Task 4 for sandbox/ACP router conditionals.
6. Start Task 5 for ChaChaNotes persona delegation.
7. Only run Task 6 if a concrete config blocker is found.

## Stop Conditions

Stop and ask for direction if:

- the first lifecycle deletion requires changing startup ordering
- route helper extraction changes generated OpenAPI output unexpectedly
- ChaChaNotes persona movement requires schema migration changes
- Bandit finds a new issue in touched source
- three attempts fail on the same test or implementation issue
- #1237 changes overlap with the same files and creates rebase risk

## Verification Summary Before Declaring Any Tranche Complete

Minimum:

```bash
source .venv/bin/activate
python -m pytest <focused tests> -q
python -m bandit -r <touched source paths> -f json -o /tmp/<task>.json
git diff --check
git status --short --branch
```

For router work, add generated OpenAPI/client verification if route output can change.

For docs-only planning commits, Bandit is skipped with a note because no production Python source changed.
