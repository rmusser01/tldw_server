# #3472 final-review minors Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Act on the three Minor findings of #3472's final review (TASK-15500): one definition of a fresh root's call-context overrides; the embeddings-task test also pins the task's context user; and the self-referential sentence in the committed plan copy.

**Architecture:** One read-only module constant `FRESH_ROOT` in `llm_call_context.py`, used by `llm_call_root` and by the media ingest worker's embeddings `create_task`. No behaviour change: both sites already reset exactly `prechecked` and `provider_credentials`.

**Tech Stack:** Python 3, pytest, contextvars.

**Spec:** #3472's final review, section "Minor (Nice to Have)": `<plans dir>/final_review_15500.md` (lines 51-72).

**Base:** `origin/dev` (at or after `211fe5682c`, #3472's merge). Branch `fix/own-key-minors-round2-15503`.

## Global Constraints

- PRs target `dev`.
- Tests make no real outbound network request. Every pytest run uses the probe: `PYTHONPATH=<plans dir> NETPROBE_LOG=<file> python -m pytest -p netprobe_15427 ...`; 0 netprobe lines.
- Backlog edits only with backlog-py (`PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . ...`), `--append-notes` never `--notes`; stage with `git add -A backlog/tasks`.
- Never `git stash`, `--no-verify`, force-push or `--admin`.
- Commit trailer: `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>`.
- No trailing whitespace in any committed file, this plan's copy included.
- No behaviour change: `FRESH_ROOT` holds exactly the two fields both sites reset today. Whether a root should also reset `credential_source` is out of scope (see Review Focus 2).

## Review Focus

1. **Same overrides, same effect:** `llm_call_root` still sets `user_id`/`operation` on a copy and never mutates the constant; the worker still resets both fields around `create_task`.
2. **`credential_source`:** the final review noted `LLMCallContext.credential_source` is carried by chat paths. Neither site resets it today. The reviewer of this plan says whether a root inheriting it is a defect a person would hit (if so, it is a finding for a separate task, not this PR).
3. **The new assertion** checks what a quota check inside the task would see (`current_llm_user_id()`), and fails if the task lost the owner.

---

### Task 0: File the task and copy the plan

- [ ] **Step 1: Take the id.** `mkdir <scratchpad>/id_lock` (fails if held; wait), then `zsh <plans dir>/task_next.sh` prints the highest id (it scans `origin/dev`, `origin/main` and open PRs); `15503` = that + 1. Release the lock (`rmdir`) once the draft PR's file list shows the task.

- [ ] **Step 2: File it.**

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task create \
  "Act on #3472's final-review minors: one fresh-root definition, the embeddings test's context user" \
  --id TASK-15503 -l byok,usage-quotas,tests \
  -d "Follow-up of TASK-15500 (#3472). Its final review left three Minor findings: (1) the overrides that make a fresh root ({prechecked: False, provider_credentials: None}) are written in llm_roots.llm_call_root and again in the media ingest worker's embeddings create_task, so a field added to one could be missed in the other; (2) test_the_embeddings_task_carries_no_key_and_no_admission pins the user_id argument but not current_llm_user_id() inside the task; (3) the committed plan copy Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md says 'replacing 15500 with the id' after its own substitution." \
  --ac "One read-only constant in llm_call_context.py defines a fresh root's overrides, and llm_call_root and the worker's embeddings create_task both use it" \
  --ac "The embeddings-task test asserts current_llm_user_id() inside the task is the job owner" \
  --ac "The plan copy's Task 0 Step 3 sentence no longer refers to the id it was substituted with"
```

- [ ] **Step 3: Copy the plan and commit.** Copy this file to `Docs/superpowers/plans/2026-11-15-own-key-minors-round2.md`, with every `<FN>` replaced by the id except the one in this sentence.

```bash
git add -A backlog/tasks Docs/superpowers/plans/2026-11-15-own-key-minors-round2.md
git commit -m "chore(backlog): file TASK-15503 for #3472's final-review minors, with its plan"
```

---

### Task 1: One fresh-root definition; the test pins the context user (Minors 1, 2)

**Files:**
- Modify: `tldw_Server_API/app/core/Usage/llm_call_context.py` (imports; constant after `_IN_ACCOUNTED_CALL`)
- Modify: `tldw_Server_API/app/core/Usage/llm_roots.py` (import block at `:19`; `llm_call_root` at `:64`)
- Modify: `tldw_Server_API/app/services/media_ingest_jobs_worker.py` (import block at `:44-49`; `:713-714`)
- Modify: `tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py` (import at `:18`; `test_the_embeddings_task_carries_no_key_and_no_admission`)

- [ ] **Step 1: Strengthen the test first (Minor 2).** Import `current_llm_user_id` next to `current_llm_call_context`. In `test_the_embeddings_task_carries_no_key_and_no_admission`, record it inside the stub and assert it:

```python
    async def _embeddings(**kwargs: Any) -> None:
        seen.append((current_llm_call_context(), current_llm_user_id(), kwargs["user_id"]))
        started.set()
```

```python
    [(context, context_user_id, user_id)] = seen
    assert context.provider_credentials is None
    assert context.prechecked is False
    assert context_user_id == 1
    assert str(user_id) == "1"
```

- [ ] **Step 2: Run it, then sabotage it (the RED for a coverage assertion).**

Run: `python -m pytest -p netprobe_15427 tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py -k embeddings_task -v`
Expected: PASS (the owner is carried today). Then sabotage twice, re-running after each and reverting each (`git diff --stat` shows only the test file at the end):
- change the worker's override to `llm_call_context(prechecked=False, provider_credentials=None, user_id=2)`: expected FAIL `assert 2 == 1`;
- in `_handle_job`, drop `scoped_context(user_id=int(user_id), is_admin=False)` from the `with` (keep `llm_call_context()`): expected FAIL `assert None == 1` (only the new assertion catches this one).

(`user_id=None` in the override is not a sabotage: the content scope still supplies the user, so the assertion pins the scope's user.)

- [ ] **Step 3: Add the constant (Minor 1).** In `llm_call_context.py`, add `from types import MappingProxyType` to the stdlib imports, and after the `_IN_ACCOUNTED_CALL` ContextVar line:

```python
# What a unit of background work never inherits from whatever started it: the request's admission and the
# user's own-key handle (llm_call_root; the media ingest job's embeddings task, TASK-15500).
FRESH_ROOT: Mapping[str, Any] = MappingProxyType({"prechecked": False, "provider_credentials": None})
```

- [ ] **Step 4: Use it in `llm_call_root`.** Add `FRESH_ROOT` to the `llm_call_context` import in `llm_roots.py`, and replace

```python
    overrides: dict[str, Any] = {"prechecked": False, "provider_credentials": None}  # nor its own-key handle
```

with

```python
    overrides: dict[str, Any] = dict(FRESH_ROOT)
```

- [ ] **Step 5: Use it in the worker.** Add `FRESH_ROOT` to the worker's `llm_call_context` import (keep the block's order: `FRESH_ROOT` first, uppercase sorts before lowercase), and replace

```python
            with llm_call_context(prechecked=False, provider_credentials=None):
```

with

```python
            with llm_call_context(**FRESH_ROOT):
```

- [ ] **Step 6: Run the affected tests.**

Run: `git grep -l -E "llm_call_root|llm_roots|media_ingest_jobs_worker|FRESH_ROOT" -- tldw_Server_API/tests`, then those files with `-n 4 --dist loadfile` under the probe.
Expected: 0 failed, 0 netprobe lines. Also `python -m ruff check` on the four touched files: no new errors (the worker has two existing ones, I001 and F401, on the base; Step 5 must not add an I001 of its own).

- [ ] **Step 7: Commit.**

```bash
git add tldw_Server_API/app/core/Usage/llm_call_context.py tldw_Server_API/app/core/Usage/llm_roots.py tldw_Server_API/app/services/media_ingest_jobs_worker.py tldw_Server_API/tests/MediaIngestion_NEW/unit/test_media_ingest_job_llm_quota_refusal.py
git commit -m "refactor(usage): one definition of a fresh root's call-context overrides (TASK-15503)"
```

---

### Task 2: The plan copy's sentence, checks, close (Minor 3)

**Files:**
- Modify: `Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md:67`
- Modify: `backlog/tasks/task-15503 - ....md` (backlog-py)

- [ ] **Step 1: Fix the sentence.** Replace `Copy this file to \`Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md\`, replacing \`15500\` with the id.` with `Copy this file to \`Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md\`, with its task-id placeholder replaced by the id.`

- [ ] **Step 2: Docs and lint.**

Run: `python -m pytest -p netprobe_15427 tldw_Server_API/tests/Docs tldw_Server_API/tests/lint -q -n 4`
Expected: no failure beyond the base's environment-only error (`tests/lint/test_runtime_package_discovery.py`, no `setuptools` locally).

- [ ] **Step 3: Close the task and commit.** Check `task edit --help` for the flags first.

```bash
PYTHONPATH=tools/backlog-py/src python -m backlog_py --cwd . task edit TASK-15503 --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --final-summary "<what changed, the counts, the sabotage result>" -s Done
git add -A backlog/tasks Docs/superpowers/plans/2026-11-14-own-key-media-analysis-minors.md
git commit -m "docs(plans): the TASK-15500 plan copy no longer names its own substituted id; close TASK-15503"
```
