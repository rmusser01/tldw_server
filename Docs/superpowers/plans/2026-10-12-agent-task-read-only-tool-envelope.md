# agent_task Read-Only Tool Envelope Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute tool-requesting `agent_task` definitions inside a read-only envelope — eligible tools derived from the MCP catalogue, executed through the existing chat tool machinery, `approval_required` for anything outside.

**Architecture:** One derived filter (catalogue `canExecute` ∩ `MODULE_RISK_TIER == "read_only"`) feeds a bounded 5-round tool-calling loop inside `_execute_agent_task`; enforcement is the existing `execute_assistant_tool_calls` allow-catalog; out-of-envelope attempts classify by set membership. Spec: `Docs/superpowers/specs/2026-10-12-agent-task-read-only-tool-envelope-design.md` (PR #3228).

**Tech Stack:** Python 3.11, FastAPI/pydantic server, MCP Unified protocol, pytest.

**Spec:** Docs/superpowers/specs/2026-10-12-agent-task-read-only-tool-envelope-design.md

## Global Constraints

- Owner rulings: agent_task-only scope; bounds-exhausted = `succeeded` + truncation note.
- Eligibility signal: `MODULE_RISK_TIERS` declaration + catalogue `canExecute` — never name heuristics; catalogue failure → empty set (fail closed).
- `approval_required` never degrades health; Phase-4D gate untouched.
- Loop runs inside the existing consumer execution timeout; OpenAI-style tool messages (adapters translate).
- Run venv: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python`.

---

### Task 1: Extract `_run_generation` (shared chat-call base)

**Files:**
- Modify: `tldw_Server_API/app/core/Scheduled_Tasks/automation_executors.py`
- Test: existing `tldw_Server_API/tests/Notifications/test_automation_executors.py` (unchanged — refactor pinned by existing tests)

**Interfaces:**
- Produces: `async def _run_generation(definition: DefinitionRow, user_prompt: str) -> str` — resolves target, builds call_kwargs, calls `perform_chat_api_call_async`, returns stripped text; raises `RuntimeError("automation executor received an empty completion")` on empty.

- [ ] **Step 1: Run existing executor tests (baseline)**

Run: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.venv/bin/python -m pytest tldw_Server_API/tests/Notifications/test_automation_executors.py -q -p no:cacheprovider`
Expected: all PASS

- [ ] **Step 2: Extract the shared body**

```python
async def _run_generation(definition: DefinitionRow, user_prompt: str) -> str:
    """One generation-only completion (shared by both executor families)."""
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async

    target = resolve_execution_target(definition)
    call_kwargs: dict[str, Any] = {
        "messages": [{"role": "user", "content": user_prompt}],
        "system_message": _definition_system_prompt(definition),
        "max_tokens": target["max_tokens"],
    }
    if target["provider"]:
        call_kwargs["api_provider"] = target["provider"]
    if target["model"]:
        call_kwargs["model"] = target["model"]

    response = await perform_chat_api_call_async(**call_kwargs)
    text = (extract_openai_content(response) or "").strip()
    if not text:
        raise RuntimeError("automation executor received an empty completion")
    return text
```

Rewrite `_execute_generation_only` and `_execute_agent_task` to call it (agent_task: resolve message from store first, then `return await _run_generation(definition, raw_message)`).

- [ ] **Step 3: Re-run tests; commit**

Run: same command; Expected: PASS
```bash
git add tldw_Server_API/app/core/Scheduled_Tasks/automation_executors.py
git commit -m "refactor(scheduled-tasks): extract _run_generation shared chat-call base"
```

### Task 2: Eligible-set derivation

**Files:**
- Modify: `tldw_Server_API/app/core/Scheduled_Tasks/automation_executors.py`
- Test: `tldw_Server_API/tests/Notifications/test_automation_executors.py` (append)

**Interfaces:**
- Produces: `async def derive_read_only_eligible_tools(owner_id: int) -> list[dict[str, Any]]` — catalogue entries with `canExecute` truthy whose `MODULE_RISK_TIERS.get(entry["module"], ("",""))[0] == "read_only"`; ANY exception → `[]` (fail closed).

- [ ] **Step 1: Write failing tests**

```python
@pytest.mark.asyncio
async def test_eligible_set_filters_by_declared_tier_and_permission(monkeypatch) -> None:
    catalogue = {"tools": [
        {"name": "media_search", "module": "media", "canExecute": True},
        {"name": "notes_write", "module": "notes", "canExecute": True},
        {"name": "chats_get", "module": "chats", "canExecute": False},
        {"name": "mystery_tool", "module": "not_a_module", "canExecute": True},
    ]}

    class _FakeToolsExecutor:
        async def list_tools(self, *, user_id, client_id):
            return catalogue

    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em
    monkeypatch.setattr(em, "_automation_tools_executor", lambda: _FakeToolsExecutor())
    eligible = await em.derive_read_only_eligible_tools(7)
    assert [t["name"] for t in eligible] == ["media_search"]

@pytest.mark.asyncio
async def test_eligible_set_fails_closed_on_catalogue_error(monkeypatch) -> None:
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em

    class _Broken:
        async def list_tools(self, *, user_id, client_id):
            raise RuntimeError("mcp down")

    monkeypatch.setattr(em, "_automation_tools_executor", lambda: _Broken())
    assert await em.derive_read_only_eligible_tools(7) == []
```

- [ ] **Step 2: Run; verify FAIL** (`derive_read_only_eligible_tools` undefined)

- [ ] **Step 3: Implement**

```python
def _automation_tools_executor():
    from tldw_Server_API.app.core.Tools.tool_executor import ToolExecutor
    return ToolExecutor()


async def derive_read_only_eligible_tools(owner_id: int) -> list[dict[str, Any]]:
    """Catalogue entries the owner can execute from read_only-declared modules.

    Fail closed: any catalogue failure yields an empty set — nothing is
    presented, nothing can execute (spec §3).
    """
    try:
        from tldw_Server_API.app.core.MCP_Unified.module_surface import MODULE_RISK_TIERS
        listing = await _automation_tools_executor().list_tools(
            user_id=str(owner_id), client_id="scheduled_automation"
        )
    except Exception:  # noqa: BLE001 — fail closed by construction
        return []
    eligible = []
    for entry in listing.get("tools", []):
        if not isinstance(entry, dict) or not entry.get("canExecute"):
            continue
        tier = MODULE_RISK_TIERS.get(str(entry.get("module")), ("", ""))
        if tier[0] == "read_only":
            eligible.append(entry)
    return eligible
```

- [ ] **Step 4: Run; PASS. Commit** `feat(scheduled-tasks): derive read-only eligible set from the MCP catalogue`

### Task 3: The bounded tool loop

**Files:**
- Modify: `tldw_Server_API/app/core/Scheduled_Tasks/automation_executors.py`
- Test: same file (append)

**Interfaces:**
- Consumes: Task 1's `_run_generation` parts (target resolution), Task 2's eligible list.
- Produces: `async def run_read_only_envelope(definition, raw_message, eligible: list[dict], *, max_rounds: int = 5) -> EnvelopeResult` where `EnvelopeResult = dataclass(status: str, text: str, note: str | None)` — `status` ∈ `{"succeeded", "approval_required", "failed"}`; `note` carries truncation / not-presented text. Membership violation → `status="approval_required"`, note names the attempted tool.

- [ ] **Step 1: Write failing tests** (fake chat + fake executor; assert: tool round then final answer; bounds → succeeded+note; out-of-set call → approval_required; presented-empty → succeeded+note)

```python
@pytest.mark.asyncio
async def test_envelope_runs_tool_round_then_answers(monkeypatch) -> None:
    import tldw_Server_API.app.core.Scheduled_Tasks.automation_executors as em
    calls = []

    async def _fake_llm(**kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            return {"choices": [{"message": {"content": "", "tool_calls": [
                {"id": "c1", "type": "function",
                 "function": {"name": "media_search", "arguments": "{\"q\": \"x\"}"}}]}}]}
        return {"choices": [{"message": {"content": "final answer"}}]}

    monkeypatch.setattr(em, "_envelope_llm_call", _fake_llm)

    async def _fake_exec(**kwargs):
        from tldw_Server_API.app.core.Chat.tool_auto_exec import ToolExecutionBatchResult
        # use the real batch builder via a minimal stub: return one success record
        rec = em._stub_tool_success("c1", "media_search", "result text")
        return em._batch_from_records([rec])

    monkeypatch.setattr(em, "_envelope_execute_calls", _fake_exec)
    result = await em.run_read_only_envelope(_definition({}, family="agent_task"), "do it",
        [{"name": "media_search", "module": "media"}])
    assert result.status == "succeeded" and result.text == "final answer"
    assert calls[1]["messages"][-1]["role"] == "tool"
```

Plus three siblings: `test_envelope_bounds_exhausted_succeeds_with_note` (always tool_calls; assert note contains "truncat"), `test_envelope_out_of_set_call_is_approval_required` (tool name "notes_write"; assert status + note names it), `test_envelope_presented_empty_note` (presented [] with non-empty eligible → status succeeded, note "not presented").

- [ ] **Step 2: Run; FAIL**

- [ ] **Step 3: Implement** — loop skeleton (exact code in the executor file):

```python
@dataclass
class EnvelopeResult:
    status: str
    text: str
    note: str | None = None


async def _envelope_llm_call(**kwargs):  # seam for tests
    from tldw_Server_API.app.core.Chat.chat_service import perform_chat_api_call_async
    return await perform_chat_api_call_async(**kwargs)


async def _envelope_execute_calls(**kwargs):  # seam for tests
    from tldw_Server_API.app.core.Chat.tool_auto_exec import execute_assistant_tool_calls
    return await execute_assistant_tool_calls(**kwargs)


async def run_read_only_envelope(definition, raw_message, eligible, *, max_rounds=5):
    from tldw_Server_API.app.core.Chat.chat_service import present_chat_tools
    eligible_names = {t["name"] for t in eligible}
    presented = present_chat_tools(
        tools=[{"type": "function", "function": {"name": t["name"],
                "description": t.get("description", ""), "parameters": t.get("inputSchema", {})}}
               for t in eligible],
        allow_catalog=sorted(eligible_names), rollout_mode=None,
        provider_key="", provider_allowlist=None, streaming=False,
        presentation_variant=None,
    )
    if not presented.llm_tools:
        text = await _run_generation(definition, raw_message)
        return EnvelopeResult("succeeded", text, "tools eligible but not presented")
    messages = [{"role": "user", "content": raw_message}]
    for round_index in range(max_rounds):
        response = await _envelope_llm_call(
            messages=messages, tools=presented.llm_tools,
            system_message=_definition_system_prompt(definition),
            max_tokens=resolve_execution_target(definition)["max_tokens"])
        message = (response.get("choices") or [{}])[0].get("message", {})
        tool_calls = message.get("tool_calls") or []
        content = (message.get("content") or "").strip()
        if not tool_calls:
            return EnvelopeResult("succeeded", content or "(empty)", None)
        for call in tool_calls:
            name = ((call.get("function") or {}).get("name") or "")
            if name not in eligible_names:  # membership, never deny-string parsing
                return EnvelopeResult("approval_required", content or "",
                                      f"attempted non-envelope tool: {name}")
        batch = await _envelope_execute_calls(
            tool_calls=tool_calls, user_id=str(definition.owner_id),
            client_id="scheduled_automation", max_tool_calls=8, timeout_ms=30000,
            allow_catalog=sorted(eligible_names), attach_idempotency=True,
            idempotency_seed=f"automation:{definition.id}:{round_index}")
        messages.append({"role": "assistant", "content": content, "tool_calls": tool_calls})
        for record in batch.records:
            messages.append({"role": "tool", "tool_call_id": record.tool_call_id,
                             "content": record.output or ""})
    text = await _run_generation(definition, raw_message)
    return EnvelopeResult("succeeded", text,
                          f"tool rounds truncated after {max_rounds}")
```

(Plus `_stub_tool_success` / `_batch_from_records` test helpers on the module.)

- [ ] **Step 4: Run; PASS. Commit** `feat(scheduled-tasks): bounded read-only tool loop with membership-classified refusals`

### Task 4: Consumer gate routing + outcome wiring

**Files:**
- Modify: `tldw_Server_API/app/core/Scheduled_Tasks/agent_task_jobs.py` (tools-gate block)
- Test: `tldw_Server_API/tests/Notifications/test_agent_task_jobs_consumer.py` (extend)

**Interfaces:**
- Consumes: `derive_read_only_eligible_tools`, `run_read_only_envelope`, `EnvelopeResult`.
- Produces: routing — family `agent_task` + all requested names ⊆ eligible → executor receives `payload["_envelope_eligible"] = [names]`; otherwise unchanged `approval_required`.

- [ ] **Step 1: Failing tests** — extend the matrix: `test_gate_routes_agent_task_readonly_tools_to_envelope` (agent_task def, tools ["media_search"], fake catalogue all-read-only, executor spy sees `_envelope_eligible` in payload and returns "done" → run succeeded); existing `test_tool_requesting_config_terminates_approval_required` still passes for recurring_question.

- [ ] **Step 2: FAIL; Step 3: Implement** — in the tools gate, before the `approval_required` `_finish`:

```python
eligible = await derive_read_only_eligible_tools(user_id) if definition.family == "agent_task" else []
eligible_names = {entry["name"] for entry in eligible}
requested = _requested_tool_names(definition)
if definition.family == "agent_task" and requested and requested <= eligible_names:
    payload = {**payload, "_envelope_eligible": sorted(eligible_names)}
    # fall through to executor dispatch with the enriched payload
elif _phase1_tools_requested(definition):
    ...  # existing approval_required _finish (unchanged)
```

In the executor dispatch result handling: map `EnvelopeResult.status` — `"approval_required"` → `_finish(status="approval_required", error="tools_require_read_only_envelope", summary=result.note)`; note appended to summaries otherwise.

- [ ] **Step 4: Run full consumer suite; PASS. Commit** `feat(scheduled-tasks): route read-only tool requests through the envelope`

### Task 5: agent_ref resolution + task-file close-out

**Files:**
- Modify: `automation_executors.py` (`_definition_system_prompt` extension)
- Modify: `backlog/tasks/task-15158 - agent-task-phase2-message-store-and-read-only-envelope.md`
- Test: append registry tests (fake `AgentRegistry` entry → prompt override; unknown ref → `LookupError`)

- [ ] **Step 1: Failing test** — `test_agent_ref_resolves_system_prompt` (registry stub returns entry with prompt "You are triage."; assert the LLM call's system_message); `test_unknown_agent_ref_fails_honestly` (registry returns None → `LookupError("agent_ref ... unknown or blocked")`).
- [ ] **Step 2: FAIL; Step 3: Implement** — extend `_definition_system_prompt(definition)`: when `input.agent_ref` present, look up via a `_resolve_agent_registry()` seam; entry prompt wins, fallback to generation-only prompt; None/blocked → raise LookupError with the availability reason.
- [ ] **Step 4: PASS; tick task-13264 ACs; commit** `feat(scheduled-tasks): agent_ref persona resolution for the envelope`

### Task 6: Full verification + PR

- [ ] Run the four scheduling suites (`test_agent_task_jobs_consumer`, `test_automation_executors`, `test_automation_message_store`, `test_scheduled_task_execution_target_bound`) — all green; ruff clean on touched files.
- [ ] Push branch `feat/agent-task-tool-envelope`; PR vs dev referencing spec #3228; the standard review/Qodo/rebase/merge loop.
