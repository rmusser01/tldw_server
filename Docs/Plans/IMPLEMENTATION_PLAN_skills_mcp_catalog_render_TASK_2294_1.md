# Skills MCP Catalog and Safe Render Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Expose user-scoped `skills.list`, `skills.get`, and policy-gated `skills.render` through MCP Unified without model calls, tool execution, supporting-file content exposure, or authored Skill mutation.

**Architecture:** Add two catalog-focused service operations to `SkillsService`, extend the standalone MCP profile runtime with a canonical `skill_name` permission subject, and add a thin host-side `SkillsModule`. Reuse existing integrity verification, approval leases, MCP execution/reporting, and `SkillExecutor` dry-run behavior. Keep all blocking registry and filesystem work off the event loop.

**Tech Stack:** Python 3.10+, asyncio, FastAPI MCP runtime, SQLite-backed `CharactersRAGDB`, Pydantic policy models, pytest, Bandit.

**Design:** `Docs/Design/2026-07-13-skills-mcp-catalog-render-design.md`

**Backlog:** `TASK-2294.1`

## Global Constraints

- Work only in `.worktrees/skills-mcp-catalog-render` on `codex/skills-mcp-catalog-render`.
- Use TDD: add a focused failing test, run it red, implement the minimum behavior, and run it green.
- Do not add model invocation, tool execution, workflow/job orchestration, frontend changes, new persistence, or a second policy evaluator.
- Never return raw `SKILL.md`, supporting-file names/content, filesystem paths, database paths, hashes, or raw exceptions from MCP tools.
- `skills.render` always forces `dry_run=True`, always passes `context=None` to `SkillExecutor`, and never accepts execution controls from callers.
- Catalog operations include only `user_invocable=true`, `disable_model_invocation=false`, non-deleted, integrity-approved rows for the authenticated user.
- Page items and total come from one matching row set and one integrity-filtering pass.
- `arguments` and `q` are preserved verbatim after exact type/length validation; do not call `BaseModule.sanitize_input()` for these non-executing text fields.
- Render input is limited to 10,000 characters; rendered output defaults to and cannot exceed 100,000 characters.
- Use the existing MCP gateway, `Skill(...)` permission rules, and approval leases. Do not evaluate profile documents inside `SkillsModule`.
- Run Bandit on every touched Python path before completion.

## File Map

- Modify `tldw_Server_API/app/core/Skills/skills_service.py`: one-pass model-visible page, metadata lookup, and offloaded verified load.
- Modify `tldw_Server_API/tests/Skills/unit/test_skills_service.py`: service visibility, pagination, integrity, and thread-offload coverage.
- Modify `apps/mcp-unified/src/mcp_unified/profiles/subjects.py`: extract bounded `skill_name` subjects.
- Modify `apps/mcp-unified/src/mcp_unified/profiles/permission_rules.py`: lowercase Skill patterns and subjects consistently.
- Modify `apps/mcp-unified/src/mcp_unified/policy_grants/models.py`: admit canonical Skill subjects to existing approval leases.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py`: canonical Skill matching.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_simulation.py`: extraction and simulated deny/ask/lease behavior.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py`: real profile-runtime deny/ask/lease behavior.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_grant_manager.py`: Skill approval-grant validation coverage.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_policy_grant_stores.py`: in-memory and SQLite Skill grant persistence coverage.
- Create `tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py`: tool schemas, context binding, service delegation, rendering, and safe errors.
- Create `tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py`: module contracts and user-isolation integration tests.
- Modify `tldw_Server_API/Config_Files/mcp_modules.yaml`: enable the read-only Skills module.
- Modify `tldw_Server_API/app/core/MCP_unified/module_surface.py`: classify Skills as read-only.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py`: configuration and dynamic registration tests.
- Modify `tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py`: read-only module-surface classification test.
- Modify `Docs/MCP/Unified/Modules.md`: operator-facing tool, bounds, and permission documentation.
- Update `backlog/tasks/task-2294.1 - Expose-Skills-catalog-and-safe-render-through-MCP.md` only through Backlog MCP.

---

## Stage 1: Model-Visible Skills Service Primitives

**Goal**: Provide one-pass catalog pagination, exact metadata lookup, and event-loop-safe verified loading.

**Success Criteria**: The service returns only model-visible, integrity-approved Skills; page and total use one matching row query; supporting content is never returned by metadata methods; blocking work runs in worker threads; existing `get_skill()` responses are unchanged.

**Tests**: `tldw_Server_API/tests/Skills/unit/test_skills_service.py`

**Status**: Complete

### Task 1.1: Write failing catalog service tests

- [x] Import `AsyncMock` from `unittest.mock`, then add tests using the existing `service` fixture for these behaviors:

```python
@pytest.mark.asyncio
async def test_list_model_visible_skills_page_filters_and_counts_once(service, monkeypatch):
    await service.create_skill("visible", "---\nuser-invocable: true\n---\nVisible")
    await service.create_skill("hidden", "---\nuser-invocable: false\n---\nHidden")
    await service.create_skill(
        "manual-only",
        "---\ndisable-model-invocation: true\n---\nManual",
    )
    await service._sync_registry_async(force=True)
    monkeypatch.setattr(service, "_sync_registry_async", AsyncMock())
    calls = 0
    original = service._get_db().list_skill_registry

    def counted_list(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service._get_db(), "list_skill_registry", counted_list)
    items, total = await service.list_model_visible_skills_page(limit=10, offset=0)

    assert [item.name for item in items] == ["visible"]
    assert total == 1
    assert calls == 1


@pytest.mark.asyncio
async def test_get_model_visible_skill_metadata_hides_non_model_skills(service):
    await service.create_skill(
        "manual-only",
        "---\ndisable-model-invocation: true\n---\nManual",
    )

    with pytest.raises(SkillNotFoundError):
        await service.get_model_visible_skill_metadata("manual-only")
```

- [x] Add an integrity-resolver fixture or monkeypatch `_is_skill_allowed` so an integrity-blocked row is omitted and the exact lookup raises `SkillNotFoundError`.
- [x] Add a supporting-file Skill and assert the metadata-only helpers do not parse or return `content`, `raw_content`, or `supporting_files`. Keep `directory_path` and `content_hash` available only as internal `SkillMetadata` fields; Stage 3 verifies that the MCP formatter never exposes them.
- [x] Add an offload test that records `threading.get_ident()` inside the new synchronous page helper and confirms it differs from the test event-loop thread.
- [x] Add a regression test that `get_skill()` still returns the same content/supporting-file payload after its filesystem work is moved to a worker thread.
- [x] Run the focused tests and confirm they fail because the new methods do not exist:

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Skills/unit/test_skills_service.py \
  -k 'model_visible or verified_load_offload'
```

Expected: failures naming `list_model_visible_skills_page` or `get_model_visible_skill_metadata`.

### Task 1.2: Implement one-pass service operations

- [x] Add these public signatures to `SkillsService`:

```python
async def list_model_visible_skills_page(
    self,
    *,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> tuple[list[SkillMetadata], int]:
    await self._sync_registry_async()
    return await asyncio.to_thread(
        self._list_model_visible_skills_page_sync,
        q,
        limit,
        offset,
    )


async def get_model_visible_skill_metadata(self, name: str) -> SkillMetadata:
    normalized = self._normalize_and_validate_skill_name(name)
    await self._sync_registry_async()
    return await asyncio.to_thread(
        self._get_model_visible_skill_metadata_sync,
        normalized,
    )
```

- [x] Implement private synchronous helpers that:
  - after registry synchronization, query non-deleted rows once with fixed `sort="name"`, `order="asc"`, and no DB pagination;
  - require `user_invocable` and reject `disable_model_invocation`;
  - call `_is_skill_allowed(name, purpose="skill_discovery")` once per otherwise-visible row;
  - calculate `total = len(visible_rows)` before slicing;
  - return `SkillMetadata` objects without full-directory parsing.
- [x] Extract the current post-sync body of `get_skill()` into a synchronous private helper and call it through `asyncio.to_thread`. Preserve its exception types, return keys, integrity purpose, registry tombstoning, and optimistic version behavior exactly.
- [x] Run the focused service tests:

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/tests/Skills/unit/test_skills_service.py
```

Expected: all tests pass.

- [x] Commit Stage 1:

```bash
git add tldw_Server_API/app/core/Skills/skills_service.py \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py
git commit -m "feat(skills): add model-visible catalog service"
```

---

## Stage 2: Canonical Skill Permission Subjects

**Goal**: Make `Skill(pattern)` rules enforceable for `skills.render` through the existing gateway and approval-lease system.

**Success Criteria**: `skill_name` emits one lowercase Skill subject; Skill rule patterns, runtime subjects, and approval grants share the same canonical form; deny, ask, valid lease, expired lease, and allow behavior remain gateway-owned.

**Tests**: Profile permission, policy simulation, and FastAPI gateway runtime tests.

**Status**: Complete

### Task 2.1: Write failing subject and policy tests

- [x] Extend `test_subjects_module_extracts_permission_rule_subjects` with:

```python
subjects = extract_permission_rule_subjects(
    "skills.render",
    {"skill_name": "Review-Paper", "arguments": "--formal /* example */"},
)
pairs = {(subject_type, value) for subject_type, value, _argv in subjects}
assert ("skill", "review-paper") in pairs
assert all(subject_type != "skill" for subject_type, _, _ in extract_permission_rule_subjects(
    "skills.get", {"name": "Review-Paper"}
))
```

- [x] Add profile rule tests proving `Skill(REVIEW-*)` matches `review-paper` and approval grants created with mixed case normalize to the same value.
- [x] Replace the existing unsupported-Skill grant expectations with tests proving approval grants accept `subject_type="skill"`, normalize mixed-case values to lowercase, and retain existing expiry semantics in both in-memory and SQLite stores.
- [x] Add policy simulation tests for:
  - explicit `Skill(secret-*)` deny;
  - `Skill(review-*)` ask without a lease;
  - the same ask with a valid Skill approval lease;
  - the same ask with an expired Skill approval lease;
  - explicit `Skill(review-*)` allow;
  - unrelated Skill rules not blocking an otherwise allowed tool call.
- [x] Assert blank, non-string, nested, and generic `name` values emit no Skill subject, while oversized `skill_name` values retain the existing `max_subject_value_length` failure.
- [x] Add FastAPI profile-runtime tests using a backend descriptor for `skills.render` and assert denied/approval-required calls never reach the backend, while a valid lease delegates once with a redacted grant marker.
- [x] Run the tests and confirm they fail because no Skill subject is extracted:

```bash
source ../../.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_simulation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_grant_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_policy_grant_stores.py \
  -k 'skill and (permission or subject or approval or lease)'
```

### Task 2.2: Implement shared Skill subject normalization

- [x] In `subjects.py`, add only the explicit convention:

```python
SKILL_ARGUMENT_KEYS = frozenset({"skill_name"})

# Inside extract_permission_rule_subjects(...)
elif key in SKILL_ARGUMENT_KEYS:
    for item in _string_values(value):
        _append_permission_subject(subjects, "skill", item.lower(), None)
```

- [x] Export `SKILL_ARGUMENT_KEYS` in `subjects.py.__all__` for test and policy tooling parity.
- [x] In `permission_rules.py`, lowercase both sides of Skill matching:

```python
if tool_name == "Skill":
    return PolicyDecisionRule(
        rule_type="skill",
        outcome=outcome,
        source=source,
        pattern=normalized_specifier.lower(),
        reason_code=reason_code,
    )

# In _normalize_subject_value(...)
if subject_type in {"mcp", "skill"}:
    return normalized.lower()
```

- [x] Add `"skill"` to `APPROVAL_SUBJECT_TYPES` in `policy_grants/models.py`. Do not add a new grant type, table, storage path, or Skill-specific lease branch; existing `normalize_permission_subject_value()` and stores remain authoritative.

- [x] Run all five focused permission/grant suites and confirm they pass.
- [x] Run the standalone package boundary smoke that covers the edited package source:

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py \
  -k 'source or import or package'
```

- [x] Commit Stage 2:

```bash
git add apps/mcp-unified/src/mcp_unified/profiles/subjects.py \
  apps/mcp-unified/src/mcp_unified/profiles/permission_rules.py \
  apps/mcp-unified/src/mcp_unified/policy_grants/models.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_simulation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_grant_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_policy_grant_stores.py
git commit -m "feat(mcp): enforce canonical Skill permission subjects"
```

---

## Stage 3: Read-Only Skills MCP Module

**Goal**: Implement bounded list/get/render tools using authenticated user context and the service primitives.

**Success Criteria**: Tool schemas and validation match the design; user isolation is fail-closed; punctuation is preserved; all database handles close; rendering is always dry; safe errors map through existing MCP exception classes; omitted supporting material is disclosed only as a boolean.

**Tests**: New `test_skills_module.py` unit and integration coverage.

**Status**: Complete

### Task 3.1: Write failing module contract tests

- [x] Create `test_skills_module.py` with fixtures that build separate user directories, ChaChaNotes databases, `SkillsService` instances, and `RequestContext` values whose numeric-string `user_id` values and `db_paths["chacha"]` point at each user database.
- [x] Assert numeric-string user IDs such as `"1"` are converted to positive integers, while empty, non-numeric, zero, negative, and boolean user IDs fail with `skills_user_context_required`.
- [x] Assert `get_tools()` exposes exactly `skills.list`, `skills.get`, and `skills.render`, all with `tool["metadata"]["readOnlyHint"] is True`.
- [x] Assert validator behavior for unknown keys, booleans passed as integers, negative offsets, limit 0/101, query length 201, missing/invalid names, and arguments length 10,001.
- [x] Assert `q="flags --all /* literal */"` and `arguments="--formal /* literal */\nnext"` reach service/executor unchanged.
- [x] Assert list search is fixed to name ascending and returns `skills`, `count`, integrity-filtered `total`, effective `limit`, `offset`, and `next_offset`; verify the final page returns `next_offset=None`.
- [x] Assert `skills.get` returns the same metadata formatter shape as one list item, without a wrapper or content fields.
- [x] Assert list/get never return IDs, timestamps, content, supporting-file details, paths, or hashes.
- [x] Assert hidden, model-disabled, deleted, integrity-blocked, and other-user Skills are absent or return `skill_not_found`.
- [x] Assert render output includes:

```python
assert result == {
    "skill_name": "review-paper",
    "rendered_prompt": "Review issue 42",
    "declared_tools": ["rag.search"],
    "model_override": None,
    "execution_mode": "inline",
    "supporting_files_omitted": False,
    "dry_run": True,
    "version": 1,
}
```

- [x] Add a Skill with one supporting file and assert only `supporting_files_omitted=True` is disclosed.
- [x] Monkeypatch `SkillExecutor._execute_forked` and `_execute_inline` to raise if called; render an inline Skill and a fork Skill and assert neither method runs.
- [x] Configure `max_rendered_skill_chars=10`, render 11 characters, and assert `ValueError("rendered_skill_too_large: limit=10")` without prompt text.
- [x] Parameterize missing, boolean, non-integer, below-minimum, valid, and above-maximum module settings; assert invalid types use defaults and integers clamp to the documented bounds.
- [x] Assert missing user context raises `PermissionError("skills_user_context_required")`, integrity races raise `PermissionError("context_integrity_blocked")`, and unexpected storage failures expose only `skills_unavailable` inside module-level tests.
- [x] Capture logs for a storage exception containing sentinel content and a sentinel path; assert the log includes only operation, numeric user ID, and exception type, with neither sentinel disclosed.
- [x] Assert request-scoped databases close after successful list/get/render, not-found, integrity rejection, oversized output, and unexpected storage failure.
- [x] Force `SkillsService` construction to fail after `CharactersRAGDB` opens and assert that database is closed before the bounded failure propagates.
- [x] Block a worker-side catalog operation, cancel the module call, release the operation, and assert cancellation propagates only after the operation finishes and the database closes.
- [x] Run the new test file and confirm import failure because `skills_module.py` does not exist.

### Task 3.2: Implement `SkillsModule`

- [x] Create constants and clamping helpers:

```python
DEFAULT_LIST_PAGE_SIZE = 50
MAX_LIST_PAGE_SIZE = 100
MAX_QUERY_CHARS = 200
MAX_ARGUMENT_CHARS = 10_000
HARD_MAX_RENDERED_SKILL_CHARS = 100_000
```

- [x] Implement `on_initialize()` so integer settings are clamped to their documented ranges; missing, boolean, and non-integer values use the defaults. The effective `list_page_size` is the `skills.list` schema and runtime default.
- [x] Instantiate one stateless `SkillExecutor()` in `on_initialize()` and retain it as `self._executor`; it receives no request context, model client, or tool registry.
- [x] Implement all three `create_tool_definition()` schemas with `category` set to `search` or `retrieval` and `readOnlyHint=True`.
- [x] Implement `validate_tool_arguments()` with per-tool exact key allowlists. Reject `bool` for integer fields. Validate but do not rewrite `q` or `arguments`; normalize whitespace-only `q` only when passing it to the service.
- [x] Do not call `sanitize_input()`. Start dispatch with a shallow dictionary copy after confirming `arguments` is a dictionary.
- [x] Implement a request-scoped context manager/helper that:
  - converts `context.user_id` with `int(str(value))`, rejects booleans, and requires the result to be greater than zero;
  - requires trusted `context.db_paths["chacha"]`;
  - constructs `CharactersRAGDB` from that path and `SkillsService(user_id, Path(chacha_path).parent, db)` together through `asyncio.to_thread`;
  - closes the database inside the worker if `SkillsService` construction fails after `CharactersRAGDB` opens;
  - retains each offloaded task and, on cancellation, awaits the in-flight task before propagating cancellation so cleanup cannot race active database work;
  - closes `db.close_all_connections()` through `asyncio.to_thread` in `finally` on every path after a database has opened, including lookup, render, and response-size failures.
- [x] Format catalog metadata with only:

```python
{
    "name": metadata.name,
    "description": metadata.description,
    "argument_hint": metadata.argument_hint,
    "user_invocable": metadata.user_invocable,
    "disable_model_invocation": metadata.disable_model_invocation,
    "declared_tools": list(metadata.allowed_tools),
    "model": metadata.model,
    "context": metadata.context,
    "runtime": build_skill_runtime_metadata(
        context=metadata.context,
        allowed_tools=metadata.allowed_tools,
        model=metadata.model,
        disable_model_invocation=metadata.disable_model_invocation,
    ),
    "version": metadata.version,
}
```

- [x] Implement `skills.list` by calling `list_model_visible_skills_page(q=effective_q, limit=effective_limit, offset=effective_offset)` exactly once and returning:

```python
count = len(items)
next_value = offset + count
return {
    "skills": [self._format_metadata(item) for item in items],
    "count": count,
    "total": total,
    "limit": limit,
    "offset": offset,
    "next_offset": next_value if next_value < total else None,
}
```

- [x] Implement `skills.get` with one `get_model_visible_skill_metadata(name)` call and return `self._format_metadata(metadata)` directly.

- [x] For render, call metadata lookup first, then verified `get_skill()`, recheck the parsed `user_invocable` and `disable_model_invocation` flags for races, and call:

```python
result = await self._executor.execute(
    skill_data,
    arguments,
    context=None,
    dry_run=True,
)
```

- [x] Check `len(result.rendered_prompt)` before constructing the response. Return `declared_tools=list(result.allowed_tools)` and `supporting_files_omitted=bool(skill_data.get("supporting_files"))`.
- [x] Translate exceptions narrowly:
  - validation and not-found to bounded `ValueError` messages handled as invalid params;
  - missing user context and render-time `ContextIntegrityBlocked` to bounded `PermissionError` messages;
  - storage/parser failures to `RuntimeError("skills_unavailable")`, which the MCP runtime sanitizes in production.
- [x] For module-owned storage/parser failures, log only operation, numeric user ID, and exception class name. Do not interpolate exception messages, arguments, rendered content, database paths, or filesystem paths.
- [x] Run `test_skills_module.py`, then rerun the complete Skills service suite.
- [x] Commit Stage 3:

```bash
git add tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py
git commit -m "feat(mcp): add Skills catalog and dry render tools"
```

---

## Stage 4: Registration, Surface, and Operator Documentation

**Goal**: Make the module available through production configuration and document its safe operating contract.

**Success Criteria**: Default loading registers all three tools; status classifies Skills as read-only; operator docs explain bounds, permission rules, dry-run guarantees, registry maintenance, and supporting-file omission.

**Tests**: Dynamic module catalog, module surface, and authenticated user-isolation tests.

**Status**: Complete

### Task 4.1: Write failing registration tests

- [x] Add a dynamic config test asserting the `skills` entry follows `prompts`, is enabled, points to `SkillsModule`, uses version `0.1.0`, has `department=knowledge`, `max_concurrent=10`, and has the two exact settings.
- [x] Add a registration test using a temporary MCP module config and assert `find_module_for_tool()` resolves all three tool names.
- [x] Extend the module-surface test to assert `skills` appears under `read_only` with no explicit opt-in requirement.
- [x] Run the tests and confirm they fail because configuration and surface classification are absent.

### Task 4.2: Register and document the module

- [x] Add the YAML entry from the approved design immediately after `prompts`.
- [x] Add this risk-tier entry:

```python
"skills": ("read_only", "Discover and safely render user-owned Skills without execution."),
```

- [x] Add a `Skills Module` section to `Docs/MCP/Unified/Modules.md` documenting:
  - `skills.list`, `skills.get`, and `skills.render`;
  - metadata-only discovery and model-visible filtering;
  - `Skill(name)` deny/ask/approval/allow evaluation for render;
  - 10,000-character arguments and 100,000-character hard output ceiling;
  - `declared_tools` is not effective authorization;
  - `supporting_files_omitted` means the rendered body may not be self-contained;
  - registry synchronization may update derived index rows;
  - the module intentionally bypasses generic SQL-token sanitization for bounded non-executing prompt text.
- [x] Update the read-only tier table to include `skills`.
- [x] Run dynamic registration, module surface, and Skills module tests.
- [x] Commit Stage 4:

```bash
git add tldw_Server_API/Config_Files/mcp_modules.yaml \
  tldw_Server_API/app/core/MCP_unified/module_surface.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  Docs/MCP/Unified/Modules.md
git commit -m "docs(mcp): register and document Skills tools"
```

---

## Stage 5: Verification and Closeout

**Goal**: Prove the scoped feature is stable, secure, documented, and ready for review against `dev`.

**Success Criteria**: Focused and regression suites pass, Bandit reports no new findings, diff checks pass, only task-related files changed, Backlog records evidence, and the branch is ready for code review.

**Tests**: Full focused matrix below.

**Status**: In Progress

### Task 5.1: Run focused verification

- [ ] Run service and module tests:

```bash
source ../../.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py \
  tldw_Server_API/tests/Skills/unit/test_skill_executor.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py
```

- [ ] Run permission and gateway tests:

```bash
source ../../.venv/bin/activate
python -m pytest -q \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_simulation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_grant_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_policy_grant_stores.py \
  -k 'skill or permission_rule or approval_lease'
```

- [ ] Run the standalone package boundary tests affected by `apps/mcp-unified` changes:

```bash
source ../../.venv/bin/activate
python -m pytest -q tldw_Server_API/app/core/MCP_unified/tests/test_runtime_package_boundary.py
```

- [ ] Run Bandit on every touched Python path:

```bash
source ../../.venv/bin/activate
python -m bandit -r \
  apps/mcp-unified/src/mcp_unified/profiles/subjects.py \
  apps/mcp-unified/src/mcp_unified/profiles/permission_rules.py \
  apps/mcp-unified/src/mcp_unified/policy_grants/models.py \
  tldw_Server_API/app/core/Skills/skills_service.py \
  tldw_Server_API/app/core/MCP_unified/modules/implementations/skills_module.py \
  tldw_Server_API/app/core/MCP_unified/module_surface.py \
  tldw_Server_API/tests/Skills/unit/test_skills_service.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_profile_permission_rules.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_simulation.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_fastapi_package.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_gateway_policy_grant_manager.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_policy_grant_stores.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_skills_module.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_dynamic_module_catalog.py \
  tldw_Server_API/app/core/MCP_unified/tests/test_basic_functionality.py \
  -s B101 \
  -f json -o /tmp/bandit_TASK_2294_1.json
```

- [ ] Run repository hygiene checks:

```bash
git diff --check
git status --short
git diff --stat origin/dev...HEAD
```

- [ ] Inspect `/tmp/bandit_TASK_2294_1.json`; fix any new finding in touched code before proceeding.

### Task 5.2: Review and finalize tracking

- [ ] Review the final diff against every design decision and Backlog acceptance criterion. Confirm no model/tool execution, frontend changes, workflow code, new persistence, or unrelated formatting entered the branch.
- [ ] Update every stage status in this plan from `Not Started` to `Complete` only after its commands pass.
- [ ] Use Backlog MCP to record modified files, implementation plan, verification commands/results, Bandit result, known skips, and final summary for `TASK-2294.1`.
- [ ] Check all acceptance criteria and Definition of Done items only when evidence exists.
- [ ] Commit tracking-only closeout changes:

```bash
git add Docs/Plans/IMPLEMENTATION_PLAN_skills_mcp_catalog_render_TASK_2294_1.md \
  'backlog/tasks/task-2294.1 - Expose-Skills-catalog-and-safe-render-through-MCP.md'
git commit -m "chore: close TASK-2294.1 verification"
```

- [ ] Use `superpowers:requesting-code-review` for a final correctness and scope review before pushing or opening a PR against `dev`.

## Plan Self-Review Checklist

- [x] Every design requirement maps to a stage and test.
- [x] Exact interfaces and field names are consistent across service, permission, module, config, docs, and tests.
- [x] The placeholder scan returns no plan-failure language.
- [x] All synchronous database/filesystem/integrity work added by this task is explicitly offloaded.
- [x] Discovery-time and render-race integrity errors have distinct, non-enumerating behavior.
- [x] Supporting files are never returned and omission is never silent.
- [x] Permission evaluation remains in the shared gateway and uses existing approval leases.
- [x] Verification includes focused tests, regression tests, package boundaries, Bandit, and diff hygiene.
