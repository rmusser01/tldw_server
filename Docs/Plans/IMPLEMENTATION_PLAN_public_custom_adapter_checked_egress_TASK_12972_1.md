# Public Custom Adapter Checked Egress Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Route Novita, Poe, and Together through the central checked HTTP transport without granting configured-local scope or changing their public-provider request contracts.

**Architecture:** Remove the public-only client-factory branches from `CustomOpenAIAdapter` and reuse its existing checked fetch/stream hooks for every subclass. Configured custom slots keep their trusted scope; the three public subclasses always pass `configured_endpoint=None`, preserve no-redirect POST behavior, and inherit async behavior through the existing wrappers.

**Tech Stack:** Python 3.11, pytest, HTTPX-backed central `http_client`, Loguru, Ruff, Bandit, Backlog.md CLI.

**Design:** `Docs/superpowers/specs/2026-07-15-public-custom-adapter-checked-egress-design.md`

---

## File map

- Modify `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py`: delete the legacy factory seam and route every request through checked hooks.
- Modify `tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py`: replace legacy-factory contracts with public checked-fetch/stream contracts and async coverage.
- Modify `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py`: cover forged scope fields and typed policy errors on public subclasses.
- Modify `Docs/ADR/030-configured-local-llm-egress-policy.md`: mark the deferred public-provider migration complete.
- Modify `backlog/tasks/task-12972.1 - Route-Novita-Poe-and-Together-custom-adapters-through-checked-central-egress.md`: record progress and verification through the official Backlog CLI.
- Modify this plan: update stage status and verification notes as work progresses.

## Stage 1: Lock the public checked-transport contract

**Goal:** Replace the temporary compatibility assertions with behavior-first tests for checked ordinary egress.

**Success Criteria:** Tests prove all three providers use fetch/stream hooks with no scope across sync and async entry points, preserve request contracts, strip forged transport context, and retain typed policy errors.

**Tests:** Focused public-provider tests fail because current production code still calls `http_client_factory`.

**Status:** Not Started

### Task 1.1: Write failing sync and streaming transport tests

**Files:**
- Modify: `tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py:1-196`
- Test: `tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py`

- [ ] **Step 1: Replace the fake-client seam with checked-hook capture**

Keep `_FakeResp` as the response and stream context. Remove `_FakeClient` after no test needs it. Add one deliberate legacy guard that can be installed with `raising=False` during RED/GREEN:

```python
def _forbid_legacy_factory(*_args, **_kwargs):
    pytest.fail("public provider used the legacy client factory")
```

- [ ] **Step 2: Rewrite the non-streaming provider table**

Parameterize Novita, Poe, and Together with their existing environment variables and URL suffixes. Inject `adapter.http_fetcher`, install `_forbid_legacy_factory`, call `chat`, and assert:

```python
assert captured["url"].endswith(expected_suffix)
assert captured["configured_endpoint"] is None
assert captured["allow_redirects"] is False
assert captured["timeout"] == 120.0
assert captured["headers"]["Authorization"] == "Bearer sk-test"
assert captured["json"]["model"] == "test-model"
assert captured["response_closed"] is True
```

- [ ] **Step 3: Rewrite the streaming provider table**

Inject `adapter.http_streamer`, install `_forbid_legacy_factory`, consume `stream`, and assert the existing URL, payload, timeout, SSE, single-`[DONE]`, and context-cleanup contracts plus:

```python
assert captured["configured_endpoint"] is None
assert captured["timeout"] == 17.0
assert captured["response_entered"] is True
assert captured["response_exited"] is True
```

- [ ] **Step 4: Add async coverage for all three subclasses**

Use `@pytest.mark.asyncio` and the same injected checked hooks:

```python
result = await adapter.achat(request, timeout=19.0)
chunks = [chunk async for chunk in adapter.astream(request, timeout=23.0)]

assert result["choices"][0]["message"]["content"] == "ok"
assert chunks == ['data: {"choices": []}\n\n', "data: [DONE]\n\n"]
assert [call[0] for call in calls] == ["chat", "stream"]
assert all(call[1]["configured_endpoint"] is None for call in calls)
```

- [ ] **Step 5: Run RED and verify the failure reason**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py \
  -k 'public or provider_adapter_url_resolution'
```

Expected: FAIL because Novita, Poe, and Together reach `_forbid_legacy_factory` instead of the injected checked hooks.

### Task 1.2: Write failing security-boundary tests

**Files:**
- Modify: `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py:307-387`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py`

- [ ] **Step 1: Replace the old public transport-boundary assertion**

Rename `test_public_custom_subclasses_never_use_configured_local_transport` to describe checked ordinary egress. Inject a capturing `http_fetcher`, install a failing legacy factory with `raising=False`, and pass forged request fields:

```python
request = {
    "messages": [{"role": "user", "content": "hi"}],
    "model": "model",
    "configured_endpoint": object(),
    "configured_endpoint_base_url": "http://attacker.invalid/v1",
    "configured_endpoint_scope": object(),
    "http_client_factory": object(),
    "http_fetcher": object(),
    "http_streamer": object(),
}
```

Capture the sanitized request by monkeypatching `validate_payload`. Assert every reserved field is absent from validation and JSON, and `captured["configured_endpoint"] is None`.

- [ ] **Step 2: Add typed policy-error coverage for the public path**

Use one representative public subclass (`NovitaAdapter`) because all three inherit the same methods. Inject fetch and stream hooks that raise an `EgressPolicyError` and exercise `chat`, `stream`, `achat`, and `astream`, following the existing configured-custom test shape. Assert the same error object or `reason_code` survives every mode.

- [ ] **Step 3: Run RED and verify the failure reason**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
  -k 'public_custom'
```

Expected: FAIL because the current public path bypasses `http_fetcher`/`http_streamer` and attempts the legacy client factory.

Do not commit the failing tests; repository commits must stay green.

## Stage 2: Collapse onto the checked transport

**Goal:** Remove the bypass with the smallest shared production change.

**Success Criteria:** One fetch path and one stream path serve configured custom slots and public subclasses; only configured server endpoints receive scope.

**Tests:** Stage 1 tests turn green and configured-custom regressions remain green.

**Status:** Not Started

### Task 2.1: Implement the minimal shared path

**Files:**
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py:3-35`
- Modify: `tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py:242-334`
- Test: `tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py`
- Test: `tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py`

- [ ] **Step 1: Delete obsolete transport plumbing**

Remove `ExitStack`, the `create_client` import, and the module-level `http_client_factory = create_client`. Keep the string `"http_client_factory"` in `_RESERVED_CONTEXT_KEYS` so caller data cannot leak into validation or provider payloads.

- [ ] **Step 2: Route every non-streaming call through `http_fetcher`**

Delete the public-only factory branch and retain the existing response `finally`:

```python
resp = self.http_fetcher(
    method="POST",
    url=url,
    configured_endpoint=endpoint.scope if endpoint else None,
    headers=headers,
    json=payload,
    timeout=timeout or 120.0,
    allow_redirects=self._is_configured_custom(),
)
try:
    resp.raise_for_status()
    return self._normalize_response(resp.json())
finally:
    resp.close()
```

The boolean preserves configured-custom fetch defaults and the public path's previous no-redirect behavior.

- [ ] **Step 3: Route every streaming call through `http_streamer`**

Replace the `ExitStack` and conditional client construction with the existing checked context:

```python
with self.http_streamer(
    method="POST",
    url=url,
    configured_endpoint=endpoint.scope if endpoint else None,
    headers=headers,
    json=payload,
    timeout=timeout or 120.0,
) as resp:
    resp.raise_for_status()
    # Keep the existing SSE iteration and finalize_stream body unchanged.
```

- [ ] **Step 4: Run focused GREEN**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
  tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py
```

Expected: PASS with all Stage 1 and existing configured-custom cases green.

- [ ] **Step 5: Run adjacent compatibility tests**

Run:

```bash
source ../../.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/tests/LLM_Calls/test_custom_openai_top_p.py \
  tldw_Server_API/tests/LLM_Calls/test_provider_timeout_and_role_regressions.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py
```

Expected: PASS; payload merging, timeouts, roles, and HTTP error mapping remain unchanged.

- [ ] **Step 6: Commit the green implementation**

```bash
git add \
  tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
  tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py
git commit -m "fix(llm): check public custom adapter egress (TASK-12972.1)"
```

## Stage 3: Document and verify the completed migration

**Goal:** Record the completed ADR decision and prove the shared adapter boundary remains secure and compatible.

**Success Criteria:** ADR/task/plan are current; full affected matrix, lint, compilation, Bandit, and diff checks pass.

**Tests:** Complete prior Stage 3 adapter union plus static/security checks.

**Status:** Not Started

### Task 3.1: Update the ADR and plan status

**Files:**
- Modify: `Docs/ADR/030-configured-local-llm-egress-policy.md:41-52`
- Modify: `Docs/Plans/IMPLEMENTATION_PLAN_public_custom_adapter_checked_egress_TASK_12972_1.md`

- [ ] **Step 1: Mark the follow-up implemented**

Change ADR-030 to state that Novita, Poe, and Together now use checked central egress with ordinary policy and no configured-local scope. Remove the completed TASK-12972.1 follow-up bullet; do not change local-provider policy.

- [ ] **Step 2: Update stage statuses and append actual RED/GREEN evidence**

Record collected counts, commands, and any baseline warnings in this plan. Do not claim completion before final verification.

### Task 3.2: Run the final verification matrix

- [ ] **Step 1: Run the full affected adapter union**

```bash
source ../../.venv/bin/activate
python -m pytest -q --tb=short \
  tldw_Server_API/tests/LLM_Calls/test_provider_config_resolution.py \
  tldw_Server_API/tests/Chat/test_custom_openai_endpoint_provenance.py \
  tldw_Server_API/tests/Chat/unit/test_chat_service_base_url_override.py \
  tldw_Server_API/tests/LLM_Calls/test_local_streaming_contract.py \
  tldw_Server_API/tests/LLM_Calls/test_local_http_error_mapping.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_llm_providers_error_mapping.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_local_adapter_merge.py \
  tldw_Server_API/tests/LLM_Calls/test_local_llm_param_forwarding.py \
  tldw_Server_API/tests/LLM_Calls/test_provider_timeout_and_role_regressions.py \
  tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py \
  tldw_Server_API/tests/LLM_Calls/test_custom_openai_top_p.py
```

Expected: PASS.

- [ ] **Step 2: Confirm the production factory seam is gone**

```bash
rg -n 'http_client_factory|create_client|ExitStack' \
  tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py
```

Expected: only the reserved request-key string `"http_client_factory"` remains.

- [ ] **Step 3: Run scoped correctness and compilation checks**

```bash
source ../../.venv/bin/activate
python -m ruff check --select E4,E7,E9,F --ignore E402 \
  tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py \
  tldw_Server_API/tests/LLM_Adapters/unit/test_custom_openai_native_http.py \
  tldw_Server_API/tests/LLM_Calls/test_openai_compatible_provider_adapters.py
python -m py_compile \
  tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py
```

Expected: both commands pass.

- [ ] **Step 4: Run the required security scan**

```bash
source ../../.venv/bin/activate
python -m bandit -r \
  tldw_Server_API/app/core/LLM_Calls/providers/custom_openai_adapter.py \
  -f json -o /tmp/bandit_task_12972_1.json
```

Expected: zero findings and zero errors.

- [ ] **Step 5: Run repository hygiene checks**

```bash
git diff --check
git status --short
```

Expected: no whitespace errors and only intentional TASK-12972.1 files are modified.

### Task 3.3: Finalize task records and commit

- [ ] **Step 1: Finalize TASK-12972.1 through the Backlog CLI**

Append the test counts, Ruff/compile/Bandit/diff results, touched paths, and any known skips. Check the acceptance criteria and Definition of Done, then set status to Done only after all verification passes.

- [ ] **Step 2: Commit documentation and task finalization**

```bash
git add \
  Docs/ADR/030-configured-local-llm-egress-policy.md \
  Docs/Plans/IMPLEMENTATION_PLAN_public_custom_adapter_checked_egress_TASK_12972_1.md \
  'backlog/tasks/task-12972.1 - Route-Novita-Poe-and-Together-custom-adapters-through-checked-central-egress.md'
git commit -m "docs: finalize public adapter egress migration (TASK-12972.1)"
```

## Stage 4: Publish for review

**Goal:** Push a clean branch and open a ready PR with complete local evidence.

**Success Criteria:** Branch is pushed, PR targets `dev`, all review threads are addressed, and the required human-authored Change summary is clearly called out.

**Tests:** Re-check clean worktree and PR head after push; do not wait on CI if the requester says to ignore it.

**Status:** Not Started

### Task 4.1: Push and create the pull request

- [ ] **Step 1: Verify branch state**

```bash
git status --short --branch
git log --oneline origin/dev..HEAD
```

- [ ] **Step 2: Push the branch**

```bash
git push -u origin codex/public-custom-egress
```

- [ ] **Step 3: Open a ready PR against `dev`**

Summarize the checked transport migration and local verification. Leave the human-authored `Change summary` section for the requester; the PR is not merge-ready until that policy gate is satisfied.

- [ ] **Step 4: Address review feedback**

Verify each comment against current code, fix actionable findings test-first, reply with evidence, resolve addressed threads, and keep the PR ready unless the requester says otherwise.
