# The deferred final-review minors of #3544 and #3554 (TASK-15615 to TASK-15618) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Chat refuses an off-list model on the server's key before input moderation again; three test gaps of #3544 are closed; #3544's eight E302 sites and one over-long docstring line are fixed; and #3554's mixed provider listing is pinned.

**Architecture:** Four small tasks, one per backlog task, each its own commit. TASK-15615 changes one call in `create_chat_completion`: chat's first admin-override check moves into a module-level helper, `_override_error_before_the_key`, that checks with the allowlist and, only when the allowlist alone would refuse, asks `users_own_key_serves` (the pre-resolution own-key test the quota exemption already uses, ADR-083) whether the user's own key serves the provider. TASK-15616 and TASK-15618 add or fix tests only, each proven by a named source mutation. TASK-15617 is whitespace and one docstring rewrap. TASK-15618 first merges `origin/dev`, which must hold #3554.

**Tech Stack:** Python 3, FastAPI, pytest (pytest-asyncio auto mode, xdist), ruff 0.16 (`--preview` for E302), backlog-py.

**Spec:** TASK-15615, TASK-15616, TASK-15617 and TASK-15618 (descriptions and ACs, under `backlog/tasks/`); the final whole-branch reviews of #3544 (Minors M2, M3, M4) and #3554 (Minor M1), as the four tasks quote them; the owner's rules below; the controller's planning brief. Context: `Docs/superpowers/plans/2026-11-21-model-denylist.md` (TASK-15582, on `dev`) and `Docs/superpowers/plans/2026-11-21-webui-byok-provider.md` (TASK-15583, on #3554's branch).

**Base:** `origin/fix/llm-policy-review-minors-15615` at `10c8b30c8d` (`dev` at `7f0d18bedd`, #3544 merged, plus the commit filing the four tasks). Every line number below is at that commit; Task 4 merges `origin/dev` and then cites #3554's head `6428b61fc3`. Draft PR #3555 targets `dev`.

**Rulings (review round 1, 2026-11-22):** the independent dry run found 0 Blocker, 0 Important and 7 Minor. Every `Expected:` it ran matched. Its rulings are at the end of this plan, under "Rulings (review round 1, 2026-11-22)", and they override the task text where they differ. #3554 merged at 2026-11-22T16:22Z (`332e8ddd90`), so Task 4's check of `origin/dev` finds it.

## The rules the code implements (owner decisions, settled)

- A provider the admin disables (`is_enabled=False`), or a model on `denied_models`, is refused on every key and path.
- A model off `allowed_models` is "not offered on the server key": refused on the server's key, works on the user's own key (credential source `user`). Team and org keys, and `user_unmetered`, are held to the allowlist (TASK-15582 R1).
- "No server key or credits" is not a disable: BYOK works. A store outage fails closed (503 `credential_store_unavailable`).

## Global Constraints

- Tests make no real outbound network requests. Every pytest run loads the network probe plugin (`-p netprobe_15427`, from `<plans dir>`); a run ends with 0 probe lines for the tests this plan adds or changes.
- Stub every adapter and every model discovery. A test that drives `/chat/completions` replaces `chat_service.perform_chat_api_call` (a commercial provider otherwise calls the real API); a listing test replaces `provider_model_inventory._http_fetch` (the root conftest's default raises).
- Backlog edits use backlog-py only (`BACKLOG task edit ...`), with `--append-notes`, never `--notes`. Stage with `git add -A backlog/tasks`. Never hand-edit a task file. TASK-15615 to TASK-15618 exist; create no other task. Each task's commit checks its ACs and DoD and sets it Done.
- Branch `fix/llm-policy-review-minors-15615`, draft PR #3555 against `dev`. The implementer commits on it and never pushes; the controller pushes.
  - Commit trailer: `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
  - The PR body ends with `🤖 Generated with [Claude Code](https://claude.com/claude-code)`.
  - ADR-004: the owner waived the Change summary for this PR. The body has a full `## Change summary` section reading "ADR-004: the owner waived a user-written Change summary for this PR on 2026-11-22."
- Never `--no-verify`, never force-push, never `git stash`, never `git prune`, never rebase. Task 4 updates the branch with a merge of `origin/dev`.
- PostgreSQL only through the repo fixtures, with `TLDW_TEST_NO_DOCKER=1` and a session-unique `TEST_DB_NAME`; never start Docker. (This plan runs no PostgreSQL test.) Redis: never FLUSHALL or SCRIPT FLUSH.
- A mirrored doc (ADRs, the BYOK guide, everything under `Docs/Published`) needs `bash Helper_Scripts/refresh_docs_published.sh`, with the `Docs/Published` diff in the same commit.
- The ratchets, measured at the base:
  - Function size (`tests/lint/function_size_baseline.json`): `chat:create_chat_completion` is 2904 lines against a ceiling of 2908. Task 2 replaces three lines with three, so it stays 2904; no baseline edit.
  - Endpoint monkeypatch (`tests/lint/endpoint_monkeypatch_baseline.json`): `api/v1/endpoints/chat.py` measures `{"monkeypatch": 84, "other": 206}` and `llm_providers.py` `{"monkeypatch": 71, "other": 5}`, each exactly at its ceiling. No new test patches an attribute of an endpoint module, by object or string target. New tests patch core modules by string target, or reuse a fixture whose existing patch site already counts.
  - Scope predicate (`Helper_Scripts/ci/scope_predicate_baseline.txt`): no file this plan touches is listed, and no SQL changes.
  - Endpoint getter calls (`tests/lint/test_service_singleton_ratchet.py`): no zero-argument `get_*()` call is added to `chat.py`.
  - Test-quality triage (`Helper_Scripts/ci/test_quality_triage.py --enforce`): every new or changed test asserts a real output read through an attribute (`response.json()`, `.denied_models`), never only a list its own nested function fills or only `.status_code`.
  - OpenAPI contract: no route, schema or parameter changes, so the fingerprint does not change.
- No new test file: every test goes into an existing file, so shard coverage is unchanged.
- Disk rules override everything; the owner is angry about the disk filling.
  - Every pytest command runs through `RUN` (below), which is `mkdir -p <T> && TMPDIR=<T> zsh <scratchpad>/disk_guard.zsh <T> -- $PY -m pytest --basetemp=<T>/bt -p no:cacheprovider ...` from a fresh `<T>` under `<scratchpad>`, with at most `-n 2`.
  - Measure RED in place: write the test before the source, or (after a commit) change one source hunk and restore it with `git restore <file>`. Never `git archive`, never repo copies, never extra worktrees.
  - No log files: output goes through `tail`/`grep`. The sweep's id lists are deleted in Task 5.
- `$PY` = `/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python`; every command that runs Python puts `$(dirname $PY)` first on `PATH`.
- The plan stays free of trailing whitespace (it is committed under `Docs/superpowers/plans/`), and the committed copy names no private planning file.

**Abbreviations.** `<scratchpad>` = `/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad`; `<plans dir>` = `/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans`. Every command runs from the worktree root. Shell state does not persist between tool calls, so every call that uses `$PY` starts with `PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python;`. `RUN <name> <pytest args>` = `zsh <scratchpad>/run15615.zsh <name> <pytest args>` (Task 0 writes it). `BACKLOG` = `PATH="$(dirname $PY):$PATH" PYTHONPATH=tools/backlog-py/src $PY -m backlog_py --cwd .`. In prose only: `CHAT` = `tldw_Server_API/app/api/v1/endpoints/chat.py`, `BOUNDARY` = `tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py` (commands spell the paths out).

## Review Focus

The five inputs likeliest to bite that the tasks' AC tests alone would not exercise, each pinned by the test named:

1. **A chat macro (`/command`) for an off-list model from a user with their own key.** The macro runs later as a job on the server's key, so it is refused `model_not_allowed` before the job is created, and the own-key lookup is not asked (pinned at the helper; the call site's macro flag is unpinned, as on `dev`). Task 2, `test_chats_first_check_asks_for_the_users_own_key_only_for_an_off_list_model[off-list-macro]`.
2. **Every ordinary chat request: an on-list model, an override with no lists, a denied model, a disabled provider.** No own-key lookup runs (no extra credential read per request), and the denied model and disabled provider are refused without it. Task 2, the same test's `on-list`, `no-lists`, `denied` and `disabled` rows.
3. **The override store is down when chat's first check runs.** 503 `credential_store_unavailable`, before moderation and before any own-key lookup (the helper reads the store first). The existing `BOUNDARY::test_public_chat_fails_closed_before_static_fallback_or_provider_dispatch` pins the 503 and runs in Task 2's file run.
4. **A user with an own key for one provider, and a server-key provider whose override has only `allowed_models`.** That user's entry for the server-key provider lists only the allowlisted models, as for a user with no own key. Task 4, `test_a_users_own_key_leaves_the_override_on_their_server_key_entries[allowlist]`.
5. **An admin denylist with padding and blanks, then an empty list.** `[" gpt-4o ", "", "o1"]` is stored and read back as `["gpt-4o", "o1"]`; `[]` clears it to `None`. Task 3, `test_an_admin_sets_and_clears_denied_models_through_the_real_repo_sqlite`.

## Rulings

| # | Ruling | Why |
|---|---|---|
| R1 | **TASK-15615's seam: chat's first check asks `users_own_key_serves` only when the allowlist alone would refuse.** `_override_error_before_the_key` calls `validate_provider_override(provider, model)` (allowlist on); when the answer is `model_not_allowed`, the request is no macro, and the user's own key serves the provider, it returns None. | `policy_error` checks disabled, then denied, then the allowlist, so a `model_not_allowed` answer means neither of the others holds: dropping it is exactly the `allowlist=False` answer. Costs weighed: (a) **this seam**: no extra read for an on-list, no-override, denied or disabled request (Review Focus 2); one secret-row read and one server-config snapshot for an off-list request; no other test's ordering changes (table in Task 2); failover untouched. (b) **Resolve credentials before moderation**: moves `credential_runtime.resolve` and the local-llm autoswitch about 120 lines up a function 4 lines under its size cap; a macro would resolve a key it never uses; credential errors (503 missing key, an OAuth refresh at the vendor) would come before moderation for every request, changing the order existing tests and clients see. (c) **Moderation after resolution**: the same ordering change from the other side. Neither (b) nor (c) is a minor. |
| R2 | **The lookup is for `runtime_user_id`**, the user `ProviderCredentialRuntime` resolves for (`_trusted_credential_runtime_scope`, `chat.py:3732`). | The pre-check must ask about the key the runtime will pick. `users_own_key_serves` is "the same first test credential resolution makes" (its docstring): BYOK on, the provider allowlisted, an active stored key, the vendor's host. |
| R3 | **The billing pre-check's `own_key` is not reused.** | It is None unless the hosted plan checks the request (`_own_key_before_the_hosted_plan`), so reuse saves one read only on hosted deployments and only for an off-list request, at the cost of a parameter. |
| R4 | **The helper lives at module scope in `chat.py`**, after `_own_key_before_the_hosted_plan`. | `create_chat_completion` stays at 2904 lines; no ratchet bump. |
| R5 | **Docs: one ADR-086 amendment; no CHANGELOG bullet; the BYOK guide, Usage_Quotas and llm-providers unchanged.** | ADR-086's TASK-15582 amendment says "a check made before the call's credential is known refuses only what every credential refuses", which chat's first check no longer does. No doc states chat's order against moderation. The ordering regression came with #3544, which is still under `## [Unreleased]`, so no release shipped it; the TASK-15582 bullet stays true. |
| R6 | **Task order: TASK-15617, TASK-15615, TASK-15616, TASK-15618.** | The E302 scan maps #3544's added lines onto the files as #3544 left them, so it runs before any other edit. TASK-15616's first RED is the TASK-15615 hunk reverted. TASK-15618 needs the merge, so it goes last. |
| R7 | **TASK-15616 (1): the double takes `*, allowlist: bool = True`**, the real signature. | A new keyword then fails loudly instead of being swallowed by `**_kwargs`. |
| R8 | **TASK-15616 (2) goes in `tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py`**, through the service functions the admin routes call (`upsert_override`, `get_override`). The stubbed test in `tests/Admin/` stays. | That file's `_open_sqlite_override_repo` gives a real SQLite repo and pool. The route itself needs a patch of `admin_llm_providers._get_ensure_sqlite_authnz_ready_if_test_mode`, an endpoint module the ratchet forbids new patches of; the existing Admin test calls the same service layer and calls it "the admin API". The stubbed test still pins the exact JSON written to the column. |
| R9 | **TASK-15616 (3) reuses the `own_key_failover` fixture** in `tests/Chat/integration/test_chat_llm_tokens_quota_402.py`. | Failover is on only through `chat.ENABLE_PROVIDER_FALLBACK`, read once at import; the fixture's patch of it is an existing site, so the chat ceiling (206 "other", 84 "monkeypatch") holds. |
| R10 | **TASK-15617 by `ruff check --preview --select E302 --fix` on exactly the eight files.** | Measured: E302 and E305 in those eight files hit only #3544's added lines, so the fix touches nothing else. The scan uses `git diff 7f0d18bedd^1 7f0d18bedd` (#3544's merge against its first parent), not `0d5e9c7228..7f0d18bedd`, which also holds #3551, #3552 and earlier merges. |
| R11 | **TASK-15618 gives the server a Groq key through `config_llm_providers.get_api_keys`** (core, string target) and reads the Groq entry of `/llm/providers`. Three rows: a denied model, an allowlist, a disabled provider. | `get_configured_providers` reads `_config_llm_providers.get_api_keys()` at call time; Groq lists models with a Bearer key at `https://api.groq.com/openai/v1`, so the existing fake vendor answers it. The test also asserts user 7's own-key OpenAI models, proving the listing took the mixed branch. |
| R12 | **The disabled row is pinned through `enabled`.** | For a disabled provider the builder already sets `provider_enabled: false` and lists no models for a provider with a model listing; the per-entry override adds `enabled: false`, which a plain append leaves out (`None`). |

## Decisions for the owner

None. The owner's rules settle the product behaviour; R1 to R12 are implementation rulings.

## Known limits

- **A key the own-key lookup counts as the user's own but that resolves off the vendor's host** (a base URL stored on a trusted caller's credential, or a trusted caller's per-request base URL) passes chat's first check and is refused `model_not_allowed` after moderation, by the credential runtime or the adapter guard, as on `dev`. The lookup reads only the server config's endpoint for the provider (R2).
- **A trusted caller whose stored key carries the vendor's base URL, while the server config points that provider at a gateway,** is counted as not own, so an off-list model is refused before moderation; the ADR-083 quota exemption already counts it the same way.
- **No test combines a chat macro with a provider override,** so the call site's macro flag is unpinned (as on `dev`); the macro row is pinned at the helper.
- **An own-key lookup that fails** (an AuthNZ database error) counts as "not own", so an off-list model is refused `model_not_allowed` before moderation although the user's key might have served it. It fails closed; the runtime would read the same store next.
- **An off-list request makes one own-key lookup before moderation** (one secret-row read and one server-config snapshot), then the runtime's own resolution. Other requests make none (Review Focus 2).
- **The lookup's failure warning still reads "Own-key lookup for the LLM quota failed"**, now also for chat's first check (wording only).
- **The failover test's server-key row is refused by the credential runtime** before chat's failover check runs, so only the own-key row pins that check: the `allowlist=True` mutation fails it, while an `allowlist=False` mutation would pass unnoticed behind the runtime's own check.
- **A disabled no-listing provider** (Bedrock, Z.AI, MiniMax) keeps its configured and pricing-catalog model names in every listing, own key or not, with `provider_enabled: false` and `enabled: false` (pre-existing; not changed here).
- **E302 is a ruff preview rule** the project's config does not select, so new sites can recur; only #3544's eight are fixed (a pre-existing one, `tests/Chunking/test_chunking_endpoint.py:189`, is not #3544's).
- **The admin round trip is tested on SQLite only**; the PostgreSQL repo round trip is #3544's `tests/AuthNZ_Postgres/test_provider_credentials_storage_pg.py`.

---

### Task 0: Branch, run wrapper, plan copy, baseline

**Files:**
- Create: `<scratchpad>/run15615.zsh`, `<scratchpad>/sweep_files_15615.txt` (not committed)
- Create: `Docs/superpowers/plans/2026-11-22-llm-policy-review-minors.md` (this plan, verbatim)

- [ ] **Step 1: The branch.** Work in the worktree the controller gives you.

```bash
git fetch origin
git rev-parse --abbrev-ref HEAD
git status --short
git log --oneline -1
```

Expected: `fix/llm-policy-review-minors-15615`, a clean tree, and `10c8b30c8d chore(backlog): file TASK-15615 to TASK-15618, the deferred review minors of #3544 and #3554` (or a later commit the controller pushed). If the branch is not checked out, run `git switch fix/llm-policy-review-minors-15615` then `git merge --ff-only origin/fix/llm-policy-review-minors-15615`; if git answers that the branch is checked out in another worktree, stop and ask the controller (never add a worktree).

- [ ] **Step 2: The run wrapper.** Write `<scratchpad>/run15615.zsh`:

```zsh
#!/bin/zsh
# run15615.zsh <name> <pytest args...>: one disk-guarded pytest run from the current directory (the worktree root).
# Prints failures and the summary, then the network-probe line count; leaves no file behind.
# With FAILED_TO=<file> it also writes the sorted FAILED/ERROR ids there (the sweeps; delete it after use).
S=/private/tmp/claude-501/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/scratchpad
PL=/Users/macbook-dev/.claude/projects/-Users-macbook-dev-Documents-GitHub-tldw-server--claude-worktrees-core-review-fixes/a0583666-bebe-4c56-a0f9-cfe2e049d483/plans
PY=/Users/macbook-dev/Documents/GitHub/tldw_server/.venv/bin/python
n=$1; shift
T=$S/r15615_$n
rm -rf "${T:?}"; mkdir -p "$T"
export PATH="$(dirname $PY):$PATH" PYTHONPATH=$PL:$PL/pyext NETPROBE_LOG=$S/r15615_$n.net
rm -f "$NETPROBE_LOG"
out=$(TMPDIR=$T zsh $S/disk_guard.zsh $T -- $PY -m pytest --basetemp=$T/bt -p no:cacheprovider -p netprobe_15427 -q "$@" 2>&1)
[[ -n $FAILED_TO ]] && print -r -- "$out" | grep -E '^(FAILED|ERROR) ' | sed -E 's/ - .*//' | sort -u > "$FAILED_TO"
print -r -- "$out" | grep -E '^(FAILED|ERROR) |^E  |[0-9]+ (passed|failed|skipped|errors?)|DISK_GUARD' | tail -40
print -r -- "netprobe lines: $([[ -f $NETPROBE_LOG ]] && wc -l < $NETPROBE_LOG || print 0)"
rm -f "$NETPROBE_LOG"
rm -rf "${T:?}"
```

Expected: `zsh -n <scratchpad>/run15615.zsh` prints nothing, and `ls <scratchpad>/disk_guard.zsh <plans dir>/netprobe_15427.py` lists both.

- [ ] **Step 3: Copy the plan, check it, commit.** Copy this plan to `Docs/superpowers/plans/2026-11-22-llm-policy-review-minors.md`.

```bash
grep -n -E '(brief|plan|final_review|impl_report|design_review|review|pr_body)_[0-9a-z_]*\.md' Docs/superpowers/plans/2026-11-22-llm-policy-review-minors.md
grep -n -E ' +$' Docs/superpowers/plans/2026-11-22-llm-policy-review-minors.md
git add Docs/superpowers/plans/2026-11-22-llm-policy-review-minors.md
git commit -m "docs(plans): the plan for TASK-15615 to TASK-15618

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: both greps print nothing; the commit lists 1 file.

- [ ] **Step 4: Baseline.** The sweep set is every test file this plan edits plus every chat, chat-NEW and Usage test that names the own-key lookup, the first check, moderation or an override list (23 files):

```bash
git grep -l -E 'users_own_key_serves|users_own_key\(|validate_provider_override|moderate_input_messages|allowed_models|denied_models' \
  -- 'tldw_Server_API/tests/Chat/*.py' 'tldw_Server_API/tests/Chat_NEW/*.py' 'tldw_Server_API/tests/Usage/*.py' \
  | grep -v conftest.py > <scratchpad>/sweep_files_15615.txt
printf '%s\n' \
  tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py \
  tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py \
  tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py \
  tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py \
  tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_denied_models.py >> <scratchpad>/sweep_files_15615.txt
sort -u -o <scratchpad>/sweep_files_15615.txt <scratchpad>/sweep_files_15615.txt
wc -l < <scratchpad>/sweep_files_15615.txt
FAILED_TO=<scratchpad>/sweep0_15615.txt RUN sweep0 -n 2 $(cat <scratchpad>/sweep_files_15615.txt)
wc -l < <scratchpad>/sweep0_15615.txt
```

Expected: `23`. The run: predicted 0 failed and `netprobe lines: 0`; record the passed, skipped and probe counts (Task 5 compares against them). Any failure here is pre-existing: keep `sweep0_15615.txt` and `sweep_files_15615.txt` until Task 5.

---

### Task 1: TASK-15617, PEP 8 spacing and the docstring line

**Files:**
- Modify (one blank line each): `tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py:1465`, `tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py:2708`, `tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py:2082`, `tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py:564`, `tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py:372`, `tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py:1330`, `tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py:1034`, `tldw_Server_API/tests/Usage/test_llm_quota_decision.py:440` (ruff reports the first line of each definition, its decorator where it has one)
- Modify: `tldw_Server_API/cli/utils/llm_store.py:14-15`
- Create: `<scratchpad>/e302_scan_15617.py` (not committed)
- Modify: `backlog/tasks/task-15617 - PEP-8-spacing-in-the-tests-PR-3544-appended-and-an-over-long-docstring-line.md` (backlog-py)

**Interfaces:** none.

- [ ] **Step 1: The scan.** Write `<scratchpad>/e302_scan_15617.py`:

```python
"""E302/E305 hits on lines a diff range added. Usage: python -I e302_scan_15617.py <repo> <base> <head> <ruff>"""
import re
import subprocess
import sys

repo, base, head, ruff = sys.argv[1:5]
files = subprocess.run(
    ["git", "-C", repo, "diff", "--name-only", base, head, "--", "*.py"],
    capture_output=True, text=True, check=True,
).stdout.split()
for path in files:
    diff = subprocess.run(
        ["git", "-C", repo, "diff", "-U0", base, head, "--", path],
        capture_output=True, text=True, check=True,
    ).stdout
    added = set()
    for m in re.finditer(r"^@@ -\S+ \+(\d+)(?:,(\d+))? @@", diff, re.M):
        start, count = int(m.group(1)), int(m.group(2) or 1)
        added.update(range(start, start + count))
    out = subprocess.run(
        [ruff, "check", "--preview", "--select", "E302,E305", "--output-format", "concise", "--no-cache",
         f"{repo}/{path}"],
        capture_output=True, text=True,
    ).stdout
    for line in out.splitlines():
        m = re.match(r".*?:(\d+):\d+: (.*)", line)
        if m:
            tag = "ADDED" if int(m.group(1)) in added else "pre-existing"
            print(f"{tag}\t{path}:{m.group(1)}\t{m.group(2)}")
```

```bash
PATH="$(dirname $PY):$PATH" $PY -I <scratchpad>/e302_scan_15617.py "$PWD" 7f0d18bedd^1 7f0d18bedd "$(dirname $PY)/ruff" | grep '^ADDED' | cut -f2
```

Expected, exactly these 8 lines (measured at the base):

```text
tldw_Server_API/app/core/MCP_unified/tests/test_model_completion_integration.py:1465
tldw_Server_API/tests/Audio/test_tts_credential_runtime_boundary.py:2708
tldw_Server_API/tests/AuthNZ_Unit/test_provider_credential_runtime.py:2082
tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py:564
tldw_Server_API/tests/Flashcards/test_study_assistant_adapter_contract.py:372
tldw_Server_API/tests/LLM_Calls/test_provider_adapter_runtime_boundary.py:1330
tldw_Server_API/tests/Usage/test_llm_quota_byok_calls.py:1034
tldw_Server_API/tests/Usage/test_llm_quota_decision.py:440
```

All 8 are in test files; every other hit the scan prints is `pre-existing` and in a file this task does not touch.

- [ ] **Step 2: Fix the eight sites.** Write the 8 paths (the Step 1 output without `:<line>`) to a list, then fix:

```bash
PATH="$(dirname $PY):$PATH" $PY -I <scratchpad>/e302_scan_15617.py "$PWD" 7f0d18bedd^1 7f0d18bedd "$(dirname $PY)/ruff" \
  | grep '^ADDED' | cut -f2 | sed 's/:[0-9]*$//' | sort -u > <scratchpad>/e302_files_15617.txt
wc -l < <scratchpad>/e302_files_15617.txt
"$(dirname $PY)/ruff" check --preview --select E302 --fix --no-cache $(cat <scratchpad>/e302_files_15617.txt)
"$(dirname $PY)/ruff" check --preview --select E302,E305 --no-cache $(cat <scratchpad>/e302_files_15617.txt)
git diff --numstat -- $(cat <scratchpad>/e302_files_15617.txt)
git diff -U0 -- $(cat <scratchpad>/e302_files_15617.txt) | grep -E '^[+-][^+-]'
```

Expected: `8`; `Found 8 errors (8 fixed, 0 remaining).`; then `All checks passed!`; `numstat` prints 8 lines, each `1`, `0`, `<path>`; the last grep prints nothing (only blank lines were added).

- [ ] **Step 3: The docstring.** In `tldw_Server_API/cli/utils/llm_store.py`, replace lines 14-15 of the `LLMStoreGroup` docstring:

```text
    Every LLM call refuses while the store is unloaded, and nothing else loads it in a CLI process. A request for help (an option, or an LLM group
    given no subcommand) and shell completion skip the load, so they never need the AuthNZ database.
```

with (118, 118 and 13 characters):

```text
    Every LLM call refuses while the store is unloaded, and nothing else loads it in a CLI process. A request for help
    (an option, or an LLM group given no subcommand) and shell completion skip the load, so they never need the AuthNZ
    database.
```

```bash
awk 'length > 120 {print FILENAME":"FNR}' tldw_Server_API/cli/utils/llm_store.py
git diff --numstat -- tldw_Server_API/cli/utils/llm_store.py
PATH="$(dirname $PY):$PATH" $PY -I -c "import ast, sys; [ast.parse(open(f).read(), f) for f in sys.argv[1:]]" \
  tldw_Server_API/cli/utils/llm_store.py $(cat <scratchpad>/e302_files_15617.txt)
```

Expected: `awk` prints nothing (before the edit it printed `tldw_Server_API/cli/utils/llm_store.py:14`); `numstat` prints `3`, `2`, `tldw_Server_API/cli/utils/llm_store.py`; the parse check prints nothing (it writes no `.pyc`).

- [ ] **Step 4: Close TASK-15617 and commit.**

```bash
BACKLOG task edit TASK-15617 --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "Scan (ruff 0.16 --preview E302/E305 on the lines #3544's merge 7f0d18bedd added against its first parent): <the 8 sites> before; ruff --fix '8 fixed, 0 remaining', then 'All checks passed!' on the 8 files; numstat 1/0 each and no non-blank line changed. llm_store.py: no line over 120 (was :14, 146 chars). Docs: none relevant. Bandit: skipped, blank lines and a docstring only, no code. Known skip: E302 is a ruff preview rule the project does not select; tests/Chunking/test_chunking_endpoint.py:189 is pre-existing, not #3544's." \
  --final-summary "The eight top-level test definitions PR #3544 appended one blank line after the previous definition now have two (E302): in the MCP model-completion integration tests, the TTS credential runtime boundary tests, the provider credential runtime tests, the chat provider-override store boundary tests, the study assistant adapter contract tests, the provider adapter runtime boundary tests, and the two Usage quota test files. The LLMStoreGroup docstring in cli/utils/llm_store.py is rewrapped to 120 characters. Whitespace and docstring only; no behaviour change." \
  -s Done
git add -A backlog/tasks
git add tldw_Server_API/cli/utils/llm_store.py $(cat <scratchpad>/e302_files_15617.txt)
git commit -m "style(tests): two blank lines before the tests #3544 appended; wrap the LLMStoreGroup docstring (TASK-15617)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
rm -f <scratchpad>/e302_files_15617.txt <scratchpad>/e302_scan_15617.py
```

Expected: `TASK-15617 - PEP 8 spacing in the tests PR #3544 appended, and an over-long docstring line (Done)`; the commit lists 10 files.

---

### Task 2: TASK-15615, chat refuses an off-list model on the server's key before moderation

**Files:**
- Modify: `CHAT` (new helper after `_own_key_before_the_hosted_plan`, `:387-394`; the first check's call, `:4475-4477`)
- Test: `BOUNDARY` (rewrite `test_chat_runs_a_model_off_the_allowlist_only_on_the_users_own_key`, `:564-601` before Task 1's blank line; append one test)
- Modify: `Docs/ADR/086-authoritative-provider-credential-scope.md` (a paragraph after `:37`); generated: `Docs/Published/ADR/086-authoritative-provider-credential-scope.md`
- Modify: `backlog/tasks/task-15615 - Chat-refuses-an-off-list-model-on-the-server-s-key-before-input-moderation-runs.md` (backlog-py)

**Interfaces:**
- Produces: `async def _override_error_before_the_key(provider: str, model: str | None, user_id: Any, *, macro: bool) -> dict[str, str] | None` in `CHAT`. None means "go on"; otherwise the `policy_error` dict (`error_code`, `message`). Raises `ByokResolutionError("credential_store_unavailable")` when the store cannot be read.
- Consumes: `_llm_provider_overrides.validate_provider_override(provider, model, *, allowlist=True)` and `_usage_quota_deps.users_own_key_serves(user_id, provider) -> bool` (async; never raises), both looked up on their modules at call time so string-target patches reach them.

**Which tests pin chat's current ordering**, found with `git grep -n -E 'validate_provider_override|moderate_input_messages|users_own_key_serves|allowed_models' -- 'tldw_Server_API/tests/Chat' 'tldw_Server_API/tests/Chat_NEW'`:

| Test | Changes? |
|---|---|
| `BOUNDARY::test_chat_runs_a_model_off_the_allowlist_only_on_the_users_own_key` | **Yes**: rewritten below (renamed, moderation recorded, two rows added). Its old tuples pass before and after this task, so they do not pin the order. |
| `test_chat_endpoint_simplified.py::test_chat_completion_revalidates_default_model` (`:995`) | No: its double ignores `allowlist`, so it passes either way. Task 3 makes the double honest. |
| `BOUNDARY::test_late_chat_override_policy_change_blocks_provider_dispatch` | No: its first check sees an on-list model (no lookup); the late policy is refused by the runtime as before. |
| `BOUNDARY::test_chat_autoswitch_cannot_select_disabled_openai_override`, `::test_public_chat_fails_closed_before_static_fallback_or_provider_dispatch` | No: no allowlist refusal at the first check; a store outage raises in `validate_provider_override` before any lookup. |
| `Chat_NEW/integration/test_history_selection_api.py:1714` (patches moderation) | No: no override. |
| Own-key lookup counters: `test_chat_llm_tokens_quota_402.py`, `Chat_NEW/integration/test_roleplay_resume_completion.py`, `test_document_generation_endpoints.py`, `test_messages_endpoint.py`, `unit/test_messages_native_usage_recording.py` | No: none sets `allowed_models`, so the first check never asks the lookup. |

- [ ] **Step 1: Write the failing tests.** In `BOUNDARY`, replace the whole `test_chat_runs_a_model_off_the_allowlist_only_on_the_users_own_key` (its decorator through its last assert) with:

```python
@pytest.mark.parametrize(
    ("own_key", "override", "expected"),
    [
        (True, {"allowed_models": ["gpt-4.1"]}, (200, None, ["openai"], ["input"])),
        (False, {"allowed_models": ["gpt-4.1"]}, (403, "model_not_allowed", [], [])),
        (True, {"is_enabled": False}, (403, "provider_disabled", [], [])),
        (True, {"denied_models": ["gpt-4o-mini"]}, (403, "model_denied", [], [])),
    ],
    ids=["own-key", "server-key", "own-key-provider-disabled", "own-key-model-denied"],
)
def test_chat_refuses_before_moderation_what_the_key_may_not_run(
    authenticated_client,
    monkeypatch: pytest.MonkeyPatch,
    own_key: bool,
    override: dict[str, Any],
    expected: tuple[int, str | None, list[str], list[str]],
) -> None:
    """TASK-15582, TASK-15615: a model off allowed_models runs on the user's own key; on the server's key it is refused
    model_not_allowed before input moderation (and its topic monitoring and audit) runs. A disabled provider and a
    denied model are refused before moderation on every key."""
    from tldw_Server_API.tests.helpers.llm_quota import users_own_key

    dispatch_calls: list[str] = []
    moderated: list[str] = []

    def dispatch(**kwargs: Any):
        dispatch_calls.append(str(kwargs.get("api_endpoint")))
        return {
            "id": "chatcmpl-own-key", "object": "chat.completion", "created": 1, "model": "gpt-4o-mini",
            "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }

    async def moderate(**_kwargs: Any) -> None:
        moderated.append("input")

    _configure_no_dispatch_boundary(monkeypatch, static_calls=[], dispatch_calls=[])
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.perform_chat_api_call", dispatch)
    monkeypatch.setattr("tldw_Server_API.app.core.Chat.chat_service.moderate_input_messages", moderate)
    if own_key:
        users_own_key(monkeypatch, "openai")
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", api_key="server-key", **override)}
    )

    response = authenticated_client.post("/api/v1/chat/completions", json=_chat_request())

    # The refusal's code is read from the body, so the test is not status-only (review round 1, B4).
    assert (
        response.status_code,
        (response.json().get("detail") or {}).get("error_code"),
        dispatch_calls,
        moderated,
    ) == expected
```

and append at the end of `BOUNDARY` (two blank lines before it):

```python
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("override", "macro", "expected"),
    [
        ({"allowed_models": ["gpt-4.1"]}, False, (None, ["openai"])),
        ({"allowed_models": ["gpt-4.1"]}, True, ("model_not_allowed", [])),
        ({"allowed_models": ["gpt-4o-mini"]}, False, (None, [])),
        ({}, False, (None, [])),
        ({"denied_models": ["gpt-4o-mini"]}, False, ("model_denied", [])),
        ({"is_enabled": False}, False, ("provider_disabled", [])),
    ],
    ids=["off-list-own-key", "off-list-macro", "on-list", "no-lists", "denied", "disabled"],
)
async def test_chats_first_check_asks_for_the_users_own_key_only_for_an_off_list_model(
    monkeypatch: pytest.MonkeyPatch,
    override: dict[str, Any],
    macro: bool,
    expected: tuple[str | None, list[str]],
) -> None:
    """TASK-15615 (Review Focus 1, 2): chat's first check asks whether the user's own key serves the provider only
    when the allowlist alone would refuse, so an ordinary request makes no extra credential read; a macro, which runs
    later on the server's key, is held to the allowlist without asking."""
    from tldw_Server_API.tests.helpers.llm_quota import users_own_key

    asked = users_own_key(monkeypatch, "openai")
    overrides_module.set_llm_provider_overrides_cache_for_tests(
        {"openai": LLMProviderOverride(provider="openai", **override)}
    )

    error = await chat_endpoint._override_error_before_the_key("openai", "gpt-4o-mini", 7, macro=macro)

    assert ((error or {}).get("error_code"), asked) == expected
```

- [ ] **Step 2: Run them to see them fail.**

```bash
RUN t2red -p no:xdist tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py -k "before_moderation or first_check"
```

Expected: `7 failed, 3 passed`, netprobe lines 0. The 7: all six `test_chats_first_check_asks_for_the_users_own_key_only_for_an_off_list_model[...]` (`AttributeError: module 'tldw_Server_API.app.api.v1.endpoints.chat' has no attribute '_override_error_before_the_key'`) and `test_chat_refuses_before_moderation_what_the_key_may_not_run[server-key]` (the tuples differ only in the last item, `['input']` against `[]`: moderation runs before the runtime's refusal). The other three rows of that test pass: the own key already runs the model, and a disabled provider or denied model is already refused before moderation.

- [ ] **Step 3: Implement.** In `CHAT`, after `_own_key_before_the_hosted_plan` (two blank lines before and after), add:

```python
async def _override_error_before_the_key(
    provider: str, model: str | None, user_id: Any, *, macro: bool
) -> dict[str, str] | None:
    """Chat's first admin-override check, made before input moderation and before the call's key is resolved.

    A disabled provider and a denied model are refused on every key. A model off ``allowed_models`` is refused unless
    the user's own key serves the provider (``users_own_key_serves``, asked only then), so a request the server's key
    would refuse is refused before moderation runs. A macro runs later as a job on the server's key, so it is always
    held to the list (TASK-15582, TASK-15615). ``credential_runtime.resolve`` checks the list again on the key it
    resolves. Raises ``ByokResolutionError("credential_store_unavailable")`` when the store cannot be read.
    """
    error = _llm_provider_overrides.validate_provider_override(provider, model)
    if (
        error is not None
        and error.get("error_code") == "model_not_allowed"
        and not macro
        and await _usage_quota_deps.users_own_key_serves(user_id, provider)
    ):
        return None
    return error
```

and replace the first check's three lines (`:4475-4477`):

```text
                # Only what every key refuses: credential_runtime.resolve checks the allowlist below, once it knows the
                # key; a macro runs later as a job on the server's key, so it is held to the allowlist now (TASK-15582).
                override_error = _llm_provider_overrides.validate_provider_override(provider, model, allowlist=chat_macro_intent is not None)
```

with:

```text
                override_error = await _override_error_before_the_key(
                    provider, model, runtime_user_id, macro=chat_macro_intent is not None
                )
```

It stays inside the same `try`, whose `except ByokResolutionError` maps a store outage to 503.

- [ ] **Step 4: Run them to see them pass, then the size ratchet.**

```bash
RUN t2green -p no:xdist tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py -k "before_moderation or first_check"
PATH="$(dirname $PY):$PATH" $PY -I -c "import ast; t = ast.parse(open('tldw_Server_API/app/api/v1/endpoints/chat.py').read()); print([n.end_lineno - n.lineno + 1 for n in ast.walk(t) if isinstance(n, ast.AsyncFunctionDef) and n.name == 'create_chat_completion'])"
```

Expected: `10 passed`, netprobe lines 0; `[2904]` (unchanged; the ceiling is 2908).

- [ ] **Step 5: ADR-086 and the mirror.** In `Docs/ADR/086-authoritative-provider-credential-scope.md`, after the TASK-15582 amendment (`:37`) and before `## Follow-up`, add one blank line and this paragraph (without the `> ` marker):

> Amendment (TASK-15615, 2026-11-22): chat's first check (`/chat/completions`, before input moderation) no longer leaves a model off `allowed_models` to the credential runtime. For such a model only, it asks whether the user's own credential serves the provider, by the test ADR-083's quota exemption makes before resolution (BYOK on, the provider allowlisted, an active stored credential, the vendor's host). When it does not, or when the request is a chat macro (which runs later on the server's credential), the request is refused with `model_not_allowed` before moderation, topic monitoring and their audit records run. The credential runtime still checks `allowed_models` on the credential it resolves, so a user's credential that resolves to an endpoint the operator runs is still refused there.

```bash
bash Helper_Scripts/refresh_docs_published.sh
git status --short Docs
RUN docs -n 2 tldw_Server_API/tests/Docs
```

Expected: `git status` lists exactly ` M Docs/ADR/086-authoritative-provider-credential-scope.md` and ` M Docs/Published/ADR/086-authoritative-provider-credential-scope.md`; the docs tests report 0 failed, netprobe lines 0.

- [ ] **Step 6: The chat files, against the baseline.**

```bash
FAILED_TO=<scratchpad>/t2_fail_15615.txt RUN t2files -n 2 \
  tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py \
  tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py \
  tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py \
  tldw_Server_API/tests/Chat/integration/test_auto_routing_override_snapshot_atomicity.py \
  tldw_Server_API/tests/Chat_NEW/integration/test_history_selection_api.py
comm -13 <scratchpad>/sweep0_15615.txt <scratchpad>/t2_fail_15615.txt
rm -f <scratchpad>/t2_fail_15615.txt
uvx bandit -q tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected: 0 failed, netprobe lines 0, and `comm` prints nothing. `BOUNDARY` had 11 test cases at the base and gains 2 rows and 6 helper rows (19). Bandit prints no finding on a line this task added (`uvx` may download bandit; if no download is allowed, record "Bandit not run").

- [ ] **Step 7: Close TASK-15615 and commit.**

```bash
BACKLOG task edit TASK-15615 --check-ac 1 --check-ac 2 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "<measured: t2red '7 failed, 3 passed' and the server-key row's diff; t2green '10 passed'; create_chat_completion 2904 lines; tests/Docs; t2files counts and the empty comm; Bandit. Known limits: a key the lookup counts as own that the runtime resolves off the vendor's host is still refused after moderation; a failed lookup counts as not own (fails closed); an off-list request makes one own-key lookup before moderation.>" \
  --final-summary "Chat's first admin-override check (/chat/completions, before input moderation) refuses a model off allowed_models with model_not_allowed unless the user's own key serves the provider. It asks users_own_key_serves (the pre-resolution own-key test of ADR-083's quota exemption) for the user the credential runtime resolves for, and only when the allowlist alone would refuse, so other requests make no extra read; a chat macro, which runs later on the server's key, is always held to the list. A request the server's key would refuse is again refused before input moderation, topic monitoring, supervised policy and their audit records run; the user's own key still runs the model; a disabled provider or denied model is refused before moderation on every key; the credential runtime still checks the list on the key it resolves. ADR-086 gains an amendment." \
  -s Done
git add -A backlog/tasks
git add tldw_Server_API/app/api/v1/endpoints/chat.py \
  tldw_Server_API/tests/Chat/integration/test_chat_provider_override_store_boundary.py \
  Docs/ADR/086-authoritative-provider-credential-scope.md Docs/Published/ADR/086-authoritative-provider-credential-scope.md
git commit -m "fix(chat): refuse an off-list model on the server's key before input moderation (TASK-15615)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: `TASK-15615 - Chat refuses an off-list model on the server's key before input moderation runs (Done)`; the commit lists 5 files.

---

### Task 3: TASK-15616, the three test gaps

**Files:**
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py` (`_fake_validate`, `:1003-1006`)
- Test: `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py` (append after `:558`)
- Test: `tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py` (append after `:486`)
- Modify: `backlog/tasks/task-15616 - Close-the-test-gaps-PR-3544-s-final-review-found.md` (backlog-py)

**Interfaces:**
- Consumes: Task 2's `_override_error_before_the_key` (mutation M-a reverts its call).
- Consumes: `admin_llm_providers_service.upsert_override(provider, payload) -> LLMProviderOverrideResponse` and `get_override(provider) -> LLMProviderOverrideResponse`; `_open_sqlite_override_repo(tmp_path, monkeypatch)`; the `own_key_failover` fixture (`install("health_switch")`, `.dispatched`) and `_failover_request(stream)`; `users_own_key(monkeypatch, provider, api_key=...)`.

The three mutations, each applied with Edit, run, then undone with `git restore <file>`:
- **M-a** (`CHAT`, the first check): replace Task 2's three-line call with the line it replaced, `override_error = _llm_provider_overrides.validate_provider_override(provider, model, allowlist=chat_macro_intent is not None)` (16-space indent).
- **M-b** (`tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py:760`, in `_parse_override_row`): `denied_models=denied_models,` becomes `denied_models=None,`.
- **M-c** (`CHAT`, the failover check, `:5303` at the base): `allowlist=call_credential_source(refreshed_args) != USER_KEY_SOURCE` becomes `allowlist=True`.

- [ ] **Step 1: Item (1), the honest double.** In `test_chat_completion_revalidates_default_model`, replace:

```python
    def _fake_validate(provider: str, model: Optional[str], **_kwargs):
        if model == "bad-model":
            return {"error_code": "model_not_allowed", "message": "Model not allowed"}
        return None
```

with:

```python
    def _fake_validate(provider: str, model: Optional[str], *, allowlist: bool = True):
        # As the real check: an off-list model is refused only where the allowlist applies (TASK-15616).
        if model == "bad-model" and allowlist:
            return {"error_code": "model_not_allowed", "message": "Model not allowed"}
        return None
```

- [ ] **Step 2: Item (2), the real round trip.** Append to the AuthNZ_SQLite file (two blank lines before):

```python
@pytest.mark.asyncio
async def test_an_admin_sets_and_clears_denied_models_through_the_real_repo_sqlite(tmp_path, monkeypatch) -> None:
    """TASK-15616 (#3544's final review, M3): the admin API's service writes denied_models to the real repo, and each
    response, and a later read, comes back through the store's refresh and _parse_override_row, with nothing stubbed.
    Padding and blanks are dropped; an empty list clears the column (Review Focus 5)."""
    from tldw_Server_API.app.api.v1.schemas.admin_schemas import LLMProviderOverrideRequest
    from tldw_Server_API.app.services import admin_llm_providers_service as service

    await _open_sqlite_override_repo(tmp_path, monkeypatch)

    written = await service.upsert_override("openai", LLMProviderOverrideRequest(denied_models=[" gpt-4o ", "", "o1"]))
    read_back = await service.get_override("openai")
    cleared = await service.upsert_override("openai", LLMProviderOverrideRequest(denied_models=[]))

    assert (written.denied_models, read_back.denied_models, cleared.denied_models) == (
        ["gpt-4o", "o1"],
        ["gpt-4o", "o1"],
        None,
    )
```

- [ ] **Step 3: Item (3), the failover check.** Append to `test_chat_llm_tokens_quota_402.py` (two blank lines before):

```python
@pytest.mark.parametrize(
    ("own_key", "expected"),
    [(True, (200, None, ["anthropic"])), (False, (403, "model_not_allowed", []))],
    ids=["own-key-fallback", "server-key-fallback"],
)
def test_a_failover_runs_a_model_off_the_allowlist_only_on_the_users_own_key(
    authenticated_client, own_key_failover, monkeypatch, own_key, expected
):
    """TASK-15616 (#3544's final review, M3): the health check fails openai over to anthropic, whose default model is
    off anthropic's allowed_models. On the user's own anthropic key the fallback dispatches; on the server's key it is
    refused model_not_allowed. chat.py's failover check takes the allowlist from the fallback key's source."""
    from tldw_Server_API.app.core.Provider_Credentials.llm_provider_overrides import (
        LLMProviderOverride,
        set_llm_provider_overrides_cache_for_tests,
    )
    from tldw_Server_API.tests.helpers.llm_quota import users_own_key

    own_key_failover.install("health_switch")
    if own_key:
        users_own_key(monkeypatch, "anthropic", api_key="sk-users-anthropic-key")
    set_llm_provider_overrides_cache_for_tests(
        {"anthropic": LLMProviderOverride(provider="anthropic", allowed_models=["claude-only-this"])}
    )

    response = authenticated_client.post("/api/v1/chat/completions", json=_failover_request(False).model_dump())

    assert (
        response.status_code,
        (response.json().get("detail") or {}).get("error_code"),
        own_key_failover.dispatched,
    ) == expected
```

- [ ] **Step 4: Run them to see them pass.** The four targets; each `<target N>` in a command below is the path written out:

1. `tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py::test_chat_completion_revalidates_default_model`
2. `tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py::test_an_admin_sets_and_clears_denied_models_through_the_real_repo_sqlite`
3. `tldw_Server_API/tests/Admin/test_admin_llm_provider_denied_models.py`
4. `tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py::test_a_failover_runs_a_model_off_the_allowlist_only_on_the_users_own_key`

```bash
RUN t3green -p no:xdist <target 1> <target 2> <target 3> <target 4>
```

Expected: `5 passed`, netprobe lines 0.

- [ ] **Step 5: RED for item (1), mutation M-a.** Apply M-a, then:

```bash
RUN t3red_a -p no:xdist <target 1>
git restore tldw_Server_API/app/api/v1/endpoints/chat.py
```

Expected: `1 failed` at `assert response.status_code == status.HTTP_403_FORBIDDEN` (predicted `assert 200 == 403`: the first check no longer refuses, and nothing later does, since the double is not the store). With the old double (`**_kwargs`, no `allowlist` test) the same mutation passes: that was the gap.

- [ ] **Step 6: RED for item (2), mutation M-b.** Apply M-b, then:

```bash
RUN t3red_b -p no:xdist <target 2> <target 3>
git restore tldw_Server_API/app/core/Provider_Credentials/llm_provider_overrides.py
```

Expected: `1 failed, 1 passed`. The new test fails with `(None, None, None) == (['gpt-4o', 'o1'], ['gpt-4o', 'o1'], None)`; the stubbed Admin test passes under the mutation (the gap).

- [ ] **Step 7: RED for item (3), mutation M-c.** Apply M-c, then:

```bash
RUN t3red_c -p no:xdist <target 4>
git restore tldw_Server_API/app/api/v1/endpoints/chat.py
git status --short tldw_Server_API/app
```

Expected: `1 failed, 1 passed`: `[own-key-fallback]` fails with `(403, 'model_not_allowed', []) == (200, None, ['anthropic'])`; `[server-key-fallback]` passes (the credential runtime refuses that fallback first; Known limits). `git status` prints nothing.

- [ ] **Step 8: The three files whole, against the baseline.**

```bash
FAILED_TO=<scratchpad>/t3_fail_15615.txt RUN t3files -n 2 \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py \
  tldw_Server_API/tests/Admin/test_admin_llm_provider_denied_models.py \
  tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py
comm -13 <scratchpad>/sweep0_15615.txt <scratchpad>/t3_fail_15615.txt
rm -f <scratchpad>/t3_fail_15615.txt
```

Expected: 0 failed, netprobe lines 0, `comm` prints nothing. (The simplified file runs whole in Task 2's file run and Task 5's sweep.)

- [ ] **Step 9: Close TASK-15616 and commit.**

```bash
BACKLOG task edit TASK-15616 --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "<measured: t3green '5 passed'; M-a (the TASK-15615 first-check call reverted) fails item 1; M-b (_parse_override_row denied_models=None) fails item 2 while the stubbed Admin test passes; M-c (failover allowlist=True) fails [own-key-fallback] only; t3files 0 failed, comm empty. Docs: none relevant. Bandit: skipped, test-only change. Known limit: the server-key fallback is refused by the credential runtime before the failover check, so only the own-key row pins that check.>" \
  --final-summary "Three test gaps from PR #3544's final review are closed. The simplified chat test's validate_provider_override double now refuses model_not_allowed only when called with the allowlist on, as the real check does, so the test pins chat's first-check refusal (it fails when the TASK-15615 call is reverted). A new SQLite test sets denied_models through the admin service (upsert_override, get_override) and reads it back through the real repo, the store refresh and _parse_override_row, padding and blanks dropped and an empty list clearing it. A new chat test pins the failover check: a fallback to a model off allowed_models dispatches on the user's own key and is refused model_not_allowed on the server's key; mutating the check to allowlist=True fails it." \
  -s Done
git add -A backlog/tasks
git add tldw_Server_API/tests/Chat/integration/test_chat_endpoint_simplified.py \
  tldw_Server_API/tests/AuthNZ_SQLite/test_authnz_llm_provider_overrides_repo_sqlite.py \
  tldw_Server_API/tests/Chat/integration/test_chat_llm_tokens_quota_402.py
git commit -m "test(llm): pin chat's first check, the denylist round trip and the failover check (TASK-15616)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: `TASK-15616 - Close the test gaps PR #3544's final review found (Done)`; the commit lists 4 files.

---

### Task 4: TASK-15618, the mixed provider listing

**Files:**
- Merge: `origin/dev` (it must hold #3554)
- Test: `tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py` (append after `:321` at #3554's head)
- Modify: `backlog/tasks/task-15618 - Pin-the-mixed-provider-listing-server-key-entries-next-to-an-own-key-entry.md` (backlog-py)

**Interfaces:**
- Consumes (from #3554, as at `6428b61fc3`): in `tldw_Server_API/app/api/v1/endpoints/llm_providers.py`, `_users_listing`'s loop ending
  ```text
          if isinstance(entry, dict) and entry.get("name") in own_keys:
              providers.append(_own_key_entry(entry))
          else:
              applied = _llm_provider_overrides.apply_llm_provider_overrides_to_listing({"providers": [entry]})
              providers += applied["providers"]
  ```
  and, in the test file, `_listing_app(monkeypatch, *, stored, ...) -> (TestClient, requester)`, `_vendor(monkeypatch, answers) -> list[str]`, `_openai_in(body)` and `_ROUTES`. If any differs after the merge, adapt the test to it and say so in the notes; if `_users_listing` no longer has that `else` branch, stop and report.

- [ ] **Step 1: Merge `origin/dev`, after checking it holds #3554.**

```bash
git fetch origin dev
git cat-file -e origin/dev:tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py && echo present || echo absent
```

Expected: `present`. If `absent`, #3554 has not merged: stop here and report to the controller (Tasks 0 to 3 stay committed; Task 5 waits).

```bash
git merge --no-edit origin/dev
git log --oneline -1
grep -n 'applied = _llm_provider_overrides.apply_llm_provider_overrides_to_listing({"providers": \[entry\]})' tldw_Server_API/app/api/v1/endpoints/llm_providers.py
```

Expected: `Merge made by the 'ort' strategy.` with no conflict (this branch and #3554 share no file); a merge commit at the head; one `grep` line. On a conflict: `git merge --abort` and report.

- [ ] **Step 2: The file before the change.**

```bash
RUN t4base -p no:xdist tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
```

Expected: `13 passed` (the count at `6428b61fc3`; if #3554 changed before merging, its file's count), netprobe lines 0 predicted (record the count).

- [ ] **Step 3: Write the test.** Append to `test_llm_providers_own_key.py` (two blank lines before):

```python
@pytest.mark.parametrize(
    ("groq_override", "expected"),
    [
        ({"denied_models": ["groq-denied"]}, (["groq-allowed"], True, True)),
        ({"allowed_models": ["groq-allowed"]}, (["groq-allowed"], True, True)),
        ({"is_enabled": False}, ([], False, False)),
    ],
    ids=["denied-model", "allowlist", "provider-disabled"],
)
def test_a_users_own_key_leaves_the_override_on_their_server_key_entries(
    monkeypatch: pytest.MonkeyPatch,
    groq_override: dict[str, Any],
    expected: tuple[list[str], bool, bool],
) -> None:
    """TASK-15618 (#3554's final review, M1): user 7 lists OpenAI on their own key, and Groq runs on the server's key.
    In that mixed listing the Groq entry is still held to the admin override, as for a user with no own key: a denied
    model and a model off allowed_models are not listed, and a disabled provider is listed unusable."""
    client, requester = _listing_app(monkeypatch, stored={(7, "openai"): "sk-user-7"})
    monkeypatch.setattr(
        "tldw_Server_API.app.core.config_llm_providers.get_api_keys", lambda: {"groq": "sk-server-groq"}
    )
    _vendor(monkeypatch, {"sk-user-7": ["gpt-7"], "sk-server-groq": ["groq-allowed", "groq-denied"]})
    set_llm_provider_overrides_cache_for_tests({"groq": LLMProviderOverride(provider="groq", **groq_override)})
    requester.user = 7

    body = client.get(_ROUTES[0]).json()
    groq = next(p for p in body["providers"] if p["name"] == "groq")

    assert (_openai_in(body)["models"], groq["models"], groq.get("enabled"), groq["provider_enabled"]) == (
        ["gpt-7"],
        *expected,
    )
```

- [ ] **Step 4: Run it.**

```bash
RUN t4green -p no:xdist tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
```

Expected: `16 passed` (Step 2's count + 3), the same netprobe count as Step 2.

- [ ] **Step 5: RED, the review's mutation.** In `llm_providers.py`, replace the two lines of `_users_listing`'s `else:` branch (`applied = ...` and `providers += applied["providers"]`) with one line, `providers.append(entry)` (12-space indent).

```bash
RUN t4red -p no:xdist tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
git restore tldw_Server_API/app/api/v1/endpoints/llm_providers.py
git status --short tldw_Server_API/app
```

Expected: `3 failed, 13 passed`: the three new rows, every existing test passing (the gap). `[denied-model]` and `[allowlist]` fail on Groq's models (`['groq-allowed', 'groq-denied']`); `[provider-disabled]` fails on `enabled` (`None`, not `False`; R12). `git status` prints nothing.

- [ ] **Step 6: Close TASK-15618 and commit.**

```bash
BACKLOG task edit TASK-15618 --check-ac 1 --check-ac 2 --check-ac 3 \
  --check-dod 1 --check-dod 2 --check-dod 3 --check-dod 4 --check-dod 5 --check-dod 6 \
  --append-notes "<measured: origin/dev held #3554 (merge commit <sha>); t4base <n> passed; t4green <n+3> passed; t4red (else branch replaced by providers.append(entry)) 3 failed, <n> passed. Docs: none relevant. Bandit: skipped, test-only change. Known limit: for a disabled server-key provider only 'enabled' differs under the mutation; the builder already lists it unusable with no listed models.>" \
  --final-summary "The mixed provider listing is pinned. A user with their own OpenAI key, next to a Groq entry on the server's key, still sees Groq held to the admin override: a model on Groq's denied_models and a model off its allowed_models are not listed, and a Groq the admin disabled is listed unusable (enabled and provider_enabled false, no models). Replacing _users_listing's per-entry override application with a plain append fails all three rows. The branch merges dev after #3554." \
  -s Done
git add -A backlog/tasks
git add tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
git commit -m "test(llm): pin the override on server-key entries in an own-key listing (TASK-15618)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"
```

Expected: `TASK-15618 - Pin the mixed provider listing: server-key entries next to an own-key entry (Done)`; the commit lists 2 files.

---

### Task 5: Verification, the PR body, hand-off

- [ ] **Step 1: The sweep.** The baseline's files plus the own-key listing file:

```bash
FAILED_TO=<scratchpad>/sweep1_15615.txt RUN sweep1 -n 2 $(cat <scratchpad>/sweep_files_15615.txt) \
  tldw_Server_API/tests/Chat_NEW/unit/test_llm_providers_own_key.py
comm -13 <scratchpad>/sweep0_15615.txt <scratchpad>/sweep1_15615.txt
comm -23 <scratchpad>/sweep0_15615.txt <scratchpad>/sweep1_15615.txt
```

Expected: both `comm` commands print nothing; 0 failed predicted; netprobe lines equal to the baseline's plus Task 4 Step 2's. Passed: the baseline's + 8 (`BOUNDARY`: 2 rows, 6 helper rows) + 2 (failover) + 1 (SQLite) + 16 (the listing file), adjusted for tests the `dev` merge added to the sweep files. If the first `comm` prints ids, classify each: one of this plan's tests is a bug to fix here; a test the merge brought that fails the same way on `origin/dev` is pre-existing (show it); anything else, stop and report. Then `rm -f <scratchpad>/sweep0_15615.txt <scratchpad>/sweep1_15615.txt <scratchpad>/sweep_files_15615.txt`.

- [ ] **Step 2: tests/lint, tests/Docs and the gates.**

```bash
RUN lint -n 2 tldw_Server_API/tests/lint --ignore=tldw_Server_API/tests/lint/test_runtime_package_discovery.py
RUN docs5 -n 2 tldw_Server_API/tests/Docs
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/test_quality_triage.py --enforce
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_shard_coverage.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/check_backlog_task_ids.py
PATH="$(dirname $PY):$PATH" $PY Helper_Scripts/ci/scope_predicate_ratchet.py
git diff --name-only --diff-filter=AM origin/dev...HEAD -- '*.py' | xargs "$(dirname $PY)/ruff" check
git diff --check origin/dev...HEAD
PATH="$(dirname $PY):$PATH" PYTHONPATH=. $PY Helper_Scripts/export_openapi_schema.py \
  --check apps/packages/api-types/openapi.fingerprint.json --check-paths apps/packages/api-types/paths.d.ts
```

Expected: `tests/lint` 0 failed, the function-size and endpoint-monkeypatch ratchets included (an intermittent native abort in `test_rg_route_map_lint.py` under `-n 2` also hits `origin/dev`: re-run that file with `-p no:xdist` and expect it to pass); `tests/Docs` 0 failed. The four scripts exit 0. `ruff` reports nothing that `origin/dev`'s copy of the same file does not (compare with `git show origin/dev:<file> | "$(dirname $PY)/ruff" check --stdin-filename <file> -`). `git diff --check` prints nothing. The OpenAPI check exits 0 (no route or schema changed).

- [ ] **Step 3: The PR body.** In the PR body draft, replace each `_(implementer: ...)_` placeholder under `## Measured` with the measured value. Change nothing else.

- [ ] **Step 4: Hand off.** Do not push. Report to the controller: the five commits (plan, TASK-15617, TASK-15615, TASK-15616, TASK-15618) plus the merge commit, every measured value, any `Expected:` that differed and why, and anything stopped or skipped. The controller pushes, runs the final whole-branch review (`/superpowers:requesting-code-review` in a clean session, given Review Focus and the Rulings), records it under `## Reviews`, and marks #3555 ready.

## Rulings (review round 1, 2026-11-22)

The independent dry run: Ready to execute; 0 Blocker, 0 Important, 7 Minor. Every `Expected:` it ran matched. These rulings override the task text above.

| # | Finding | Ruling |
|---|---|---|
| RR1 | M1: no test checks which user id the own-key lookup is asked about. Passing `None` at the call site passes every test, and in production it would refuse every own-key user's off-list model before moderation. | **Fix.** In Task 2, `test_chat_refuses_before_moderation_what_the_key_may_not_run` wraps the own-key stub so it records the user id it receives. For the `own-key` row, assert that the recorded id equals the authenticated test user's id (and is not `None`). Add a mutation step: passing `None` at the call site fails that row. |
| RR2 | M2: Review Focus 1 claims more than its test pins. The call site's `macro=chat_macro_intent is not None` has no test, on `dev` too. | **Reword, no new test.** Review Focus 1 and the Task 2 docstring say: "pinned at the helper; the call site's macro flag is unpinned, as on `dev`". Add a Known-limits line in the plan and the PR body: "No test combines a chat macro with a provider override, so the call site's macro flag is unpinned (as on `dev`)." A macro-plus-override chat test is out of scope for a minors PR. |
| RR3 | M3: the ADR-086 amendment says "This runtime", and Known limit 1 names the wrong checker. | **Fix the wording.** In the ADR paragraph (Task 2 Step 5), write "The credential runtime still checks `allowed_models` on the credential it resolves". In Known limit 1 (plan and PR body), write: "A key the own-key lookup counts as the user's own but that resolves off the vendor's host (a base URL stored on a trusted caller's credential, or a trusted caller's per-request base URL) passes chat's first check and is refused `model_not_allowed` after moderation, by the credential runtime or the adapter guard, as on `dev`." Refresh the Published mirror as Task 2 Step 5 already says. |
| RR4 | M4: a reverse mismatch is missing from Known limits. | **Add the line** to the plan's and the PR body's Known limits: "A trusted caller whose stored key carries the vendor's base URL, while the server config points that provider at a gateway, is counted as not own, so an off-list model is refused before moderation; the ADR-083 quota exemption already counts it the same way." |
| RR5 | M5: TASK-15618 AC #2 says "hides", but the test asserts the provider is listed as unusable. TASK-15616 AC #2 says "through the admin API", but R8 meets it through the service functions. | **Amend the ACs.** In Task 4's backlog step: `BACKLOG task edit TASK-15618 --remove-ac 2 --ac "A test asserts that user's listing shows a server-key provider the admin disabled as unusable (enabled and provider_enabled false, no models)"`. Then check the re-added AC by its new index (print the task to find it). In Task 3's backlog step, add to `--append-notes`: "AC #2: admin API = the admin service functions the routes call (R8); the HTTP route is not patched, because the endpoint-monkeypatch ratchet is at its ceiling." |
| RR6 | M6: Task 0's private-file grep misses the PR body draft's name, and Task 5 Step 3 names it in the committed copy. | **Fix.** Add `pr_body` to the grep's alternation: `(brief|plan|final_review|impl_report|design_review|pr_body)_`. In the committed copy, Task 5 Step 3 says "the PR body draft" instead of the private path. The `netprobe_15427.py` mention stays, as in earlier plans. |
| RR7 | M7: Bandit can't run offline here. | **No change.** Record "Bandit not run (not installed)" in Measured. Don't download it. |
