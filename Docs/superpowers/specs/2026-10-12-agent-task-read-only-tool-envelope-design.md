# Design: agent_task read-only tool envelope — bounded tool-calling for scheduled automations

Status: **Approved design** (owner rulings 2026-10-12: scope = agent_task
only; bounds-exhausted outcome = succeeded with truncation note).
Implements ADR-184 decision 2A step 2 (tldw_chatbook
`backlog/decisions/184-agent-task-server-execution-message-refs-and-approval-escalation.md`);
carries TASK-13264 to its last open item.

## 1. Purpose

Let certified deployments execute `agent_task` definitions that request
tools — inside a read-only envelope. Step 1 (landed, tldw_server #3042)
made tool-requesting runs terminate `approval_required` instead of the
phase-1 skip. This step converts that absolute refusal into a bounded
capability: side-effect-free tools execute in-loop; anything else still
terminates `approval_required`, now including runtime attempts.

## 2. Locked decisions (owner-ruled)

1. **Scope: `agent_task` only.** Tool-requesting `recurring_question`
   definitions keep terminating `approval_required` — their primary path
   is generation-only; tools there are an authoring anomaly. The consumer
   gate routes to the envelope only for `agent_task`.
2. **Bounds-exhausted outcome: `succeeded` with an explicit truncation
   note.** The loop's iteration cap is a cost control, not a failure; the
   run produced output and the summary says it may be incomplete.

## 3. Tool eligibility — declared tier + verified permission, fail closed everywhere

**The classification signal is the MCP registry's explicit module risk tier,
not the tool's name.** `MCP_Unified/module_surface.py::MODULE_RISK_TIERS`
curates every module: `read_only` (media, knowledge, rag, chats, prompts,
prompts_catalog, cooking, skills, mcp_discovery), `write`, `local_files`,
`local_process`, `external_network`. Name-token heuristics
(`determine_permission_tier`) are NOT trusted for unattended execution — a
read-ish name proves nothing about a tool's effects, and treating it as
proof would be fail-open by construction.

**At the consumer gate:** a tool-requesting `agent_task` definition enters
the envelope only when every tool in `input.tools` satisfies the full
closed chain:

1. The MCP catalog resolves the name (via `Tools/ToolExecutor.list_tools`)
   to a registered tool with `canExecute` for the owner.
2. That tool's module maps to `MODULE_RISK_TIER == "read_only"`.

Any failure in the chain — non-`read_only` module, unresolvable name,
permission miss, **or the catalog listing itself failing (MCP unavailable
→ the intersection cannot be verified)** — terminates the run
`approval_required`. No path executes on an unverified tool set.

**At runtime (belt and braces):** the gate's conclusion is never assumed.
A model tool-call naming a tool outside the verified eligible set records
the attempted name in the run summary and terminates the run
`approval_required`. Unknown/unverifiable at runtime is
`approval_required`, never execute-anyway.

## 4. The loop

An extension of `_execute_agent_task` in `automation_executors.py`
(the ADR-named seam), after the store-resolved message (1A read path
unchanged):

1. Resolve `agent_ref` through `AgentRegistry` for the agent's system
   prompt / persona context. Unknown or blocked ref → honest failed run
   (the registry's availability classification already exists).
2. Bounded loop (default **5** iterations):
   - Present the eligible tool schemas via `present_chat_tools` — rollout
     mode, provider allowlists, and presentation variants stay consistent
     with interactive chat rather than a second policy path.
   - Call `perform_chat_api_call_async(messages, tools=…, tool_choice=…)`
     (the adapter request builder copies `tools`/`tool_choice` through;
     verified).
   - If the response carries `tool_calls`: execute via
     `execute_assistant_tool_calls` — its caps, timeouts, allow-catalog,
     and idempotency keys are reused as-is — append results as tool
     messages, continue the loop.
   - If the response carries a final text answer: return it.
3. Everything runs inside the existing consumer execution timeout — no
   new timer machinery.
4. Extract `_run_generation(definition, user_prompt)` from the shared
   chat-call boilerplate (code-review ask from #3042) so the
   generation-only and tool paths share one base.

## 5. Outcomes

| Event | Status | Summary |
|---|---|---|
| Final text answer | `succeeded` | the answer |
| Loop bounds exhausted | `succeeded` | answer-so-far + explicit truncation note |
| Model/provider error | `failed` | precise reason |
| Unauthorized / side-effecting tool attempt (runtime) | `approval_required` | attempted tool name (actionable pass-back) |
| Unknown/blocked `agent_ref`; unresolvable message ref | `failed` | precise reason (existing taxonomy) |

Health mapping unchanged (`approval_required` never degrades health).

## 6. Certification — untouched

The Phase-4D deployment-certification gate still front-runs all
`agent_task` dispatch. The envelope only matters for certified
deployments; nothing here weakens or bypasses the readiness check.

## 7. Files

- `core/Scheduled_Tasks/automation_executors.py` — the loop, agent-ref
  resolution, `_run_generation` extraction
- `core/Scheduled_Tasks/agent_task_jobs.py` — gate refinement
  (family + classify-then-route)
- Consumed as-is: `MCP_Unified/module_surface.py` (MODULE_RISK_TIERS),
  `Chat/tool_auto_exec.py`,
  `Tools/tool_executor.py` (MCP), `Agent_Client_Protocol/agent_registry.py`,
  `present_chat_tools`

## 8. Testing

- Loop tests against a tier-gated fake MCP executor: happy path
  (tool call → result → final answer), bounds exhaustion (succeeded +
  note), unauthorized tool name (approval_required with the name),
  agent_ref unknown/blocked (failed with reason)
- Gate routing (fail-closed matrix): agent_task + every tool
  read-only-declared + canExecute → envelope; any non-read-only module,
  unresolvable name, permission miss, or CATALOG UNAVAILABLE →
  approval_required; tool-requesting recurring_question
  → still approval_required; no-tools agent_task → generation-only
- Existing suites stay green; consumer replay semantics unchanged
