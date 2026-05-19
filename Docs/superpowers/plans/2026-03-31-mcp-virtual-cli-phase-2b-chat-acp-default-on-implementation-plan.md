# MCP Virtual CLI Phase 2b Chat + ACP Default-On Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Promote the existing `run(command)` run-first surface to default-on for the stable `chat + ACP` provider-model cohort while preserving typed-tool fallback, rollback controls, and the phase 2a execution/runtime architecture.

**Architecture:** Extend the existing phase 2a rollout contract rather than adding a new surface layer. The implementation should add a third rollout mode (`default_on`), ship explicit stable-cohort defaults in config, keep the existing `cohort` telemetry label name with refined values, and reuse the existing chat presenter, ACP presenter, and metrics seams. Chat and ACP must stay behaviorally symmetric, but phase 2b should not widen the cohort selector beyond the current `provider:model` allowlist contract.

**Tech Stack:** Python, FastAPI chat stack, ACP adapters/runners, existing MetricsRegistry/OpenTelemetry helpers, pytest, Bandit, INI-style config, `.env` docs.

---

## Implementation Boundaries

- Do not add new command families or change the phase 1 `run(command)` runtime.
- Do not hide typed tools completely.
- Do not widen the cohort selector beyond the current `provider:model` allowlist contract.
- Do not reinterpret governance or approval semantics.
- Keep the existing `cohort` metric label name; refine values instead of adding a new `posture` label in phase 2b.
- Keep code-level safe fallback as `off`; move default-on behavior through shipped config/profile defaults.

## File Map

### Rollout/config contract

- Modify: `tldw_Server_API/app/core/config.py`
  - Expand run-first rollout modes to `off | gated | default_on`.
  - Add one shared helper for deriving the run-first `cohort` label from rollout mode, eligibility, and ineligible reason.
- Modify: `tldw_Server_API/Config_Files/config.txt`
  - Ship the stable-cohort phase 2b defaults in `[Chat-Module]` and `[ACP]`.
- Modify: `tldw_Server_API/Config_Files/.env.example`
  - Document the new mode vocabulary and optional override env vars without enabling insecure or implicit defaults.

### Chat surface

- Modify: `tldw_Server_API/app/core/Chat/run_first_presentation.py`
  - Treat `default_on` as an enabled run-first mode and preserve out-of-cohort fallback behavior.
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
  - Map phase 2b rollout state into the existing `_chat_run_first_*` metadata and telemetry context.
  - Replace silent chat run-first metric suppression with logged, non-fatal handling.
- Modify: `tldw_Server_API/app/core/Chat/chat_metrics.py`
  - Preserve existing counters while accepting the refined `cohort` values.
- Modify: `tldw_Server_API/app/core/Chat/README.md`
  - Update the phase 2a documentation block to reflect phase 2b default-on semantics and rollback posture.

### ACP surface

- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_tool_presentation.py`
  - Treat `default_on` as an enabled run-first mode.
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_adapter.py`
  - Derive the refined `cohort` label from rollout mode plus presentation eligibility.
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/metrics.py`
  - Preserve metric family names while accepting the refined `cohort` values.

### Test coverage

- Modify: `tldw_Server_API/tests/Chat/unit/test_run_first_rollout_config.py`
- Modify: `tldw_Server_API/tests/Governance/test_governance_metrics_and_rollout.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_run_first_presentation.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_metrics_integration.py`
- Modify: `tldw_Server_API/tests/Chat/integration/test_chat_run_first_rollout.py`
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_tool_presentation.py`
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_adapter.py`
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_metrics.py`

## Task 1: Expand The Rollout Contract And Ship Stable-Cohort Defaults

**Files:**
- Modify: `tldw_Server_API/app/core/config.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Modify: `tldw_Server_API/Config_Files/.env.example`
- Modify: `tldw_Server_API/tests/Chat/unit/test_run_first_rollout_config.py`
- Modify: `tldw_Server_API/tests/Governance/test_governance_metrics_and_rollout.py`

- [ ] **Step 1: Write the failing config tests for `default_on` and cohort mapping**

Add coverage for:

- `resolve_chat_run_first_rollout_mode()` accepting `default_on`
- `resolve_acp_run_first_rollout_mode()` accepting `default_on`
- one shared helper that maps rollout state to phase 2b `cohort` values

```python
def test_resolve_chat_run_first_rollout_mode_accepts_default_on(monkeypatch):
    from tldw_Server_API.app.core import config

    monkeypatch.setenv("CHAT_RUN_FIRST_ROLLOUT_MODE", "default_on")
    assert config.resolve_chat_run_first_rollout_mode() == "default_on"


def test_resolve_run_first_cohort_label_maps_default_on_out_of_cohort():
    from tldw_Server_API.app.core import config

    assert config.resolve_run_first_cohort_label(
        "default_on",
        eligible=False,
        ineligible_reason="provider_not_in_rollout_allowlist",
    ) == "out_of_cohort"
```

- [ ] **Step 2: Run the config tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_run_first_rollout_config.py \
  tldw_Server_API/tests/Governance/test_governance_metrics_and_rollout.py -v
```

Expected: FAIL because `default_on` is not accepted yet and the shared cohort-label helper does not exist.

- [ ] **Step 3: Implement the minimal config contract**

In `tldw_Server_API/app/core/config.py`:

- expand `_RUN_FIRST_ROLLOUT_MODES`
- add a shared helper used by both chat and ACP

```python
_RUN_FIRST_ROLLOUT_MODES = {"off", "gated", "default_on"}


def resolve_run_first_cohort_label(
    rollout_mode: str | None,
    *,
    eligible: bool,
    ineligible_reason: str | None = None,
) -> str:
    token = str(rollout_mode or "").strip().lower()
    if token == "gated":
        return "gated"
    if token == "default_on":
        if eligible:
            return "default_on"
        if str(ineligible_reason or "").strip() == "provider_not_in_rollout_allowlist":
            return "out_of_cohort"
        return "default_on"
    return "override_off"
```

In `tldw_Server_API/Config_Files/config.txt`, add explicit phase 2b shipped defaults in:

- `[Chat-Module]`
- `[ACP]`

Use the stable cohort already encoded in phase 2a tests/examples:

```ini
run_first_rollout_mode = default_on
run_first_provider_allowlist = openai:gpt-4o-mini,anthropic:claude-3-7-sonnet
run_first_presentation_variant = chat_phase2b_v1
```

and the ACP equivalent with `acp_phase2b_v1`.

In `tldw_Server_API/Config_Files/.env.example`, document the same env knobs as optional overrides, but leave them commented so copying the template does not silently widen rollout behavior.

- [ ] **Step 4: Run the config tests again to verify they pass**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/Config_Files/.env.example \
  tldw_Server_API/tests/Chat/unit/test_run_first_rollout_config.py \
  tldw_Server_API/tests/Governance/test_governance_metrics_and_rollout.py
git commit -m "feat: add phase 2b run-first rollout defaults"
```

## Task 2: Promote Chat Run-First Presentation To `default_on`

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/run_first_presentation.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_run_first_presentation.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py`

- [ ] **Step 1: Write the failing chat presentation tests**

Add coverage for:

- `rollout_mode="default_on"` plus in-cohort provider keeps `run` first and injects prompt guidance
- `rollout_mode="default_on"` plus out-of-cohort provider keeps tools visible but makes the session ineligible
- chat request metadata uses the refined `cohort` values

```python
def test_present_chat_tools_orders_run_first_when_default_on_and_in_cohort():
    presented = present_chat_tools(
        tools=[RUN_TOOL, NOTES_TOOL],
        allow_catalog=["run", "notes.*"],
        rollout_mode="default_on",
        provider_key="openai:gpt-4o-mini",
        provider_allowlist=["openai:gpt-4o-mini"],
        streaming=False,
    )
    assert presented.eligible is True
    assert [tool["function"]["name"] for tool in presented.llm_tools] == ["run", "notes.search"]


async def test_build_call_params_marks_out_of_cohort_for_default_on_provider_mismatch(monkeypatch):
    ...
    assert cleaned_args["_chat_run_first_cohort"] == "out_of_cohort"
```

- [ ] **Step 2: Run the chat presentation tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_run_first_presentation.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py -v
```

Expected: FAIL because `default_on` is not treated as enabled and the chat metadata still only emits `gated` or `control`.

- [ ] **Step 3: Implement the minimal chat behavior changes**

In `tldw_Server_API/app/core/Chat/run_first_presentation.py`:

- treat `default_on` as an enabled mode alongside `gated`
- keep `provider_not_in_rollout_allowlist` as the explicit ineligible reason for out-of-cohort default-on sessions

```python
rollout_enabled = rollout_token in {"gated", "default_on"}
eligible = rollout_enabled and run_present and provider_allowed
```

In `tldw_Server_API/app/core/Chat/chat_service.py`:

- use `resolve_run_first_cohort_label(...)` when setting `_chat_run_first_cohort`
- keep `tool_choice` unset or `auto`
- preserve the existing effective-tool-set invariant

```python
call_params["_chat_run_first_cohort"] = resolve_run_first_cohort_label(
    rollout_mode,
    eligible=run_first_presentation.eligible,
    ineligible_reason=run_first_presentation.ineligible_reason,
)
```

- [ ] **Step 4: Run the chat presentation tests again**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/Chat/run_first_presentation.py \
  tldw_Server_API/app/core/Chat/chat_service.py \
  tldw_Server_API/tests/Chat/unit/test_run_first_presentation.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py
git commit -m "feat: promote chat run-first default posture"
```

## Task 3: Promote ACP Run-First Presentation To `default_on`

**Files:**
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_tool_presentation.py`
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_adapter.py`
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_tool_presentation.py`
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_adapter.py`

- [ ] **Step 1: Write the failing ACP presentation tests**

Add coverage for:

- `default_on` enabling run-first ordering for in-cohort ACP sessions
- `default_on` preserving visible fallback tools but no prompt fragment for out-of-cohort sessions
- ACP metrics context using the refined `cohort` values

```python
def test_present_acp_tools_orders_run_first_for_default_on_session():
    presented = present_acp_tools(
        session_id="s1",
        tools=[NOTES_TOOL, RUN_TOOL],
        rollout_mode="default_on",
        provider_key="openai:gpt-4o-mini",
        provider_allowlist=["openai:gpt-4o-mini"],
    )
    assert presented.eligible is True
    assert presented.effective_tool_names == ["run", "notes.search"]


async def test_mcp_adapter_send_prompt_llm_driven_uses_default_on_cohort(...):
    ...
    assert runner_kwargs["run_first_metrics_context"]["cohort"] == "default_on"
```

- [ ] **Step 2: Run the ACP presentation tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_tool_presentation.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_adapter.py -v
```

Expected: FAIL because ACP still treats only `gated` as enabled and still maps cohort to `gated` or `control`.

- [ ] **Step 3: Implement the minimal ACP behavior changes**

In `tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_tool_presentation.py`:

```python
rollout_enabled = rollout_token in {"gated", "default_on"}
```

In `tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_adapter.py`:

- use `resolve_run_first_cohort_label(...)`
- keep the ACP prompt fragment and presented tool ordering unchanged apart from the new mode semantics

```python
"cohort": resolve_run_first_cohort_label(
    rollout_mode,
    eligible=presented_tools.eligible,
    ineligible_reason=presented_tools.ineligible_reason,
),
```

- [ ] **Step 4: Run the ACP presentation tests again**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_tool_presentation.py \
  tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_adapter.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_tool_presentation.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_adapter.py
git commit -m "feat: promote ACP run-first default posture"
```

## Task 4: Refine Telemetry Labels And Make Chat Metric Failures Observable

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/chat_service.py`
- Modify: `tldw_Server_API/app/core/Chat/chat_metrics.py`
- Modify: `tldw_Server_API/app/core/Agent_Client_Protocol/metrics.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_metrics_integration.py`
- Modify: `tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py`
- Modify: `tldw_Server_API/tests/Chat/integration/test_chat_run_first_rollout.py`
- Modify: `tldw_Server_API/tests/Agent_Client_Protocol/test_acp_metrics.py`

- [ ] **Step 1: Write the failing telemetry tests**

Add coverage for:

- `cohort="default_on"` and `cohort="out_of_cohort"` on the existing chat and ACP metrics
- chat run-first metric emission logging a warning or debug message instead of silently suppressing failures
- end-to-end chat rollout integration using `default_on` rather than `gated`

```python
def test_chat_metrics_records_phase2b_cohort_labels():
    collector = ChatMetricsCollector()
    collector.metrics.run_first_rollout = MagicMock()

    collector.track_run_first_rollout(
        presentation_variant="chat_phase2b_v1",
        cohort="default_on",
        provider="openai",
        model="gpt-4o-mini",
        streaming=False,
        eligible=True,
        ineligible_reason=None,
    )

    _, labels = collector.metrics.run_first_rollout.add.call_args.args
    assert labels["cohort"] == "default_on"


async def test_emit_chat_run_first_rollout_metrics_logs_failures(monkeypatch):
    metrics = SimpleNamespace(track_run_first_rollout=Mock(side_effect=RuntimeError("metrics down")))
    warning = Mock()
    monkeypatch.setattr(chat_service.logger, "warning", warning)
    ...
    assert warning.called
```

- [ ] **Step 2: Run the telemetry tests to verify they fail**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_chat_metrics_integration.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py \
  tldw_Server_API/tests/Chat/integration/test_chat_run_first_rollout.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_metrics.py -v
```

Expected: FAIL because the current labels still use `gated/control` semantics and chat still suppresses run-first metric failures silently.

- [ ] **Step 3: Implement the telemetry refinements**

In `tldw_Server_API/app/core/Chat/chat_service.py`:

- keep the same metric families
- replace `contextlib.suppress(Exception)` in the run-first emission helpers with logged, non-fatal handling

```python
try:
    metrics.track_run_first_rollout(**context)
except Exception as exc:
    logger.warning("Chat run-first rollout metric emission failed: {}", exc)
```

In `tldw_Server_API/app/core/Chat/chat_metrics.py` and `tldw_Server_API/app/core/Agent_Client_Protocol/metrics.py`:

- do not rename the `cohort` label
- accept the new phase 2b value set without changing counter names

- [ ] **Step 4: Run the telemetry tests again**

Run the same command from Step 2.

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/Chat/chat_service.py \
  tldw_Server_API/app/core/Chat/chat_metrics.py \
  tldw_Server_API/app/core/Agent_Client_Protocol/metrics.py \
  tldw_Server_API/tests/Chat/unit/test_chat_metrics_integration.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py \
  tldw_Server_API/tests/Chat/integration/test_chat_run_first_rollout.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_metrics.py
git commit -m "feat: refine phase 2b run-first telemetry"
```

## Task 5: Update Surface Documentation And Run Final Verification

**Files:**
- Modify: `tldw_Server_API/app/core/Chat/README.md`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Modify: `tldw_Server_API/Config_Files/.env.example`
- Verify the files changed in Tasks 1-4

- [ ] **Step 1: Write the failing docs-oriented checks**

Use existing docs/config assertions instead of inventing a new checker. Add or extend lightweight tests only if current suites do not cover:

- `default_on` appearing in run-first config docs/examples
- the stable provider-model cohort being explicitly documented in shipped config
- chat README describing phase 2b default-on posture and rollback semantics

If a test is needed, prefer extending:

- `tldw_Server_API/tests/Chat/unit/test_run_first_rollout_config.py`
- `tldw_Server_API/tests/Governance/test_governance_metrics_and_rollout.py`

- [ ] **Step 2: Update the docs**

In `tldw_Server_API/app/core/Chat/README.md`, replace the phase 2a-specific wording with phase 2b wording:

- `default_on` is now the normal posture for the stable cohort
- `gated` remains for controlled experiments
- `off` is the rollback posture
- `cohort` remains the metric label name with phase 2b values

Keep the existing phase 1 and phase 2a architecture explanation intact; only update the rollout posture and telemetry wording.

- [ ] **Step 3: Run the targeted phase 2b verification suite**

Run:

```bash
source .venv/bin/activate && python -m pytest \
  tldw_Server_API/tests/Chat/unit/test_run_first_rollout_config.py \
  tldw_Server_API/tests/Governance/test_governance_metrics_and_rollout.py \
  tldw_Server_API/tests/Chat/unit/test_run_first_presentation.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_tool_autoexec.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_streaming_tool_autoexec.py \
  tldw_Server_API/tests/Chat/unit/test_chat_metrics_integration.py \
  tldw_Server_API/tests/Chat/integration/test_chat_run_first_rollout.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_tool_presentation.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_mcp_adapter.py \
  tldw_Server_API/tests/Agent_Client_Protocol/test_acp_metrics.py -v
```

Expected: PASS.

- [ ] **Step 4: Run Bandit and diff hygiene**

Run:

```bash
source .venv/bin/activate && python -m bandit -r \
  tldw_Server_API/app/core/config.py \
  tldw_Server_API/app/core/Chat/run_first_presentation.py \
  tldw_Server_API/app/core/Chat/chat_service.py \
  tldw_Server_API/app/core/Chat/chat_metrics.py \
  tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_tool_presentation.py \
  tldw_Server_API/app/core/Agent_Client_Protocol/adapters/mcp_adapter.py \
  tldw_Server_API/app/core/Agent_Client_Protocol/metrics.py

git diff --check
```

Expected:

- Bandit reports no new issues in touched code
- `git diff --check` is clean

- [ ] **Step 5: Commit**

```bash
git add \
  tldw_Server_API/app/core/Chat/README.md \
  tldw_Server_API/Config_Files/config.txt \
  tldw_Server_API/Config_Files/.env.example
git commit -m "docs: finalize phase 2b run-first default-on rollout"
```
