# Service Prompt Registry and Resolver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Provide one typed allowlist, strict template contract, and deterministic per-part resolution path for all migrated service prompts.

**Architecture:** A code-native immutable registry is populated only from approved inventory rows. Packaged content stays in existing prompt assets. The resolver accepts explicit request parts, an optional verified pin bundle, an active-user-revision provider whose exact asset is approved by the latest signed Context Integrity manifest, and deployment overrides, then returns an immutable full bundle with per-part provenance. It has no database or FastAPI imports.

**Tech Stack:** Python dataclasses/enums, existing prompt loader/YAML assets, Hypothesis, pytest.

---

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate.

## Task 1: Define registry contracts and the first read-only entries

**Files:**

- Create: `tldw_Server_API/app/core/Service_Prompts/__init__.py`
- Create: `tldw_Server_API/app/core/Service_Prompts/models.py`
- Create: `tldw_Server_API/app/core/Service_Prompts/registry.py`
- Create: `tldw_Server_API/app/core/Service_Prompts/settings.py`
- Create: `tldw_Server_API/app/services/startup_service_prompts.py`
- Modify: `tldw_Server_API/app/main.py`
- Modify: `tldw_Server_API/Config_Files/config.txt`
- Test: `tldw_Server_API/tests/Service_Prompts/test_registry.py`
- Test: `tldw_Server_API/tests/Services/test_startup_service_prompts.py`
- Reference: `Docs/Design/service-prompt-inventory.md`

- [ ] Create the implementation Backlog task and link the approved inventory rows.
- [ ] Write failing tests for duplicate definition/part IDs, empty bundles, missing visible parts, hidden editable parts, invalid placeholder contracts, invalid budgets, and deterministic catalog ordering.
- [ ] Implement frozen `ServicePromptDefinition`, `ServicePromptPartDefinition`, `TemplateVariableContract`, `PromptExecutionContext`, and `ServicePromptRegistry` types. Definitions include category/tags/workflow IDs, localization label/description keys with English fallbacks, compatibility module/key/environment mappings, assembly order, contract version, safe sample values, output-contract eligibility evidence, sensitivity, deprecation/replacement metadata, and rollout availability.
- [ ] Enforce code-defined IDs, atomic multipart bundles, `editable` only when `visible`, explicit locked assembly order, `literal|template` mode, 64 KiB part and 256 KiB definition limits, declared sample coverage, required/optional variables, per-template/per-variable/repetition/assembly budgets, and explicit reject-or-deterministically-truncate policy for oversized runtime values.
- [ ] Register only the first approved read-only inventory slice. Do not add deferred or excluded rows and do not scan directories at runtime.
- [ ] Parse `TLDW_SERVICE_PROMPTS_MODE` in `settings.py` with exact values `enabled|read_only|bypass_stored_overrides`, default `read_only`, and startup failure on unknown values. `startup_service_prompts.py` validates mode, registry contracts/assets, sample rendering, compatibility mappings, and trusted default availability, then publishes capability state on `app.state`; wire it through the existing startup lifecycle in `main.py`.
- [ ] Run `python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_registry.py tldw_Server_API/tests/Services/test_startup_service_prompts.py` and commit: `feat: add typed service prompt registry (<task-id>)`.

## Task 2: Implement the strict template parser and renderer

**Files:**

- Create: `tldw_Server_API/app/core/Service_Prompts/templates.py`
- Test: `tldw_Server_API/tests/Service_Prompts/test_templates.py`

- [ ] Write failing example and property tests for literal text, declared `{name}` variables, escaped literal braces (`{{` and `}}`), missing/extra variables, repeated variables, malformed braces, dotted/index access, format specifiers/conversions, filters, calls, comments, loops, Unicode, reject/truncate policy, repetition overflow, per-variable overflow, and rendered-output overflow.
- [ ] Confirm the tests fail without a parser.
- [ ] Implement a linear parser for exact placeholders `{` + identifier `[A-Za-z_][A-Za-z0-9_]*` + `}`, with `{{`/ `}}` as escaped literal braces; do not add Jinja or another template dependency.
- [ ] Preserve literal bytes apart from documented brace unescaping and substitution. Return part/variable/line/column diagnostics without echoing surrounding prompt text, and validate UTF-8 byte counts plus repetition/assembled budgets.
- [ ] Rerun `python -m pytest -q tldw_Server_API/tests/Service_Prompts/test_templates.py` and commit: `feat: validate service prompt templates (<task-id>)`.

## Task 3: Add packaged and strict deployment sources

**Files:**

- Create: `tldw_Server_API/app/core/Service_Prompts/sources.py`
- Modify: `tldw_Server_API/app/core/Utils/prompt_loader.py`
- Test: `tldw_Server_API/tests/Service_Prompts/test_sources.py`
- Test: `tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py`

- [ ] Write failing tests for packaged defaults, configured file overrides, blank env values, missing/unreadable files, integrity-blocked files, invalid UTF-8, oversized content, and unchanged legacy fallback behavior.
- [ ] Add a registry-only strict loader that uses the existing env variable convention and Context Integrity resolver. If a nonblank override is configured, any read/integrity/contract failure raises `ServicePromptConfigurationError`; it never falls back.
- [ ] Keep `load_prompt()` behavior unchanged for non-migrated keys. Share only safe file-path/parsing helpers instead of adding a second YAML loader.
- [ ] Rerun both focused test files and commit: `feat: fail closed for managed prompt overrides (<task-id>)`.

## Task 4: Implement per-part resolution and provenance

**Files:**

- Create: `tldw_Server_API/app/core/Service_Prompts/resolver.py`
- Test: `tldw_Server_API/tests/Service_Prompts/test_resolver.py`
- Test: `tldw_Server_API/tests/Service_Prompts/test_resolver_properties.py`

- [ ] Write failing table and property tests covering every precedence combination: explicit request → verified pin → approved user → deployment → packaged.
- [ ] Test mixed-source multipart bundles, locked/hidden parts, unknown parts, rejection of partial stored revisions, allowed partial explicit overrides, bypass mode, read-only mode, userless maintenance, store errors, active rows missing signed-manifest trust, and deterministic canonical bundle digests.
- [ ] Define injected protocols for `ApprovedRevisionSource` and `VerifiedPinSource`; keep imports one-way so persistence and Jobs depend on the core resolver.
- [ ] Implement `resolve(context: PromptExecutionContext, service_prompt_id, variables)` returning immutable `ResolvedServicePrompt` with ordered render-ready/rendered parts, source kind, revision/snapshot digest, trusted server-default bundle digest, canonical content digest, contract/schema version, and locked-section markers.
- [ ] Select an active stored override atomically as the complete editable-part bundle; only explicit request overrides may replace a declared subset part by part. Userless activity deliberately uses trusted server defaults, and store/ownership errors fail rather than crossing users or silently falling back.
- [ ] Require the active pointer/revision digest to match the stable signed state asset and the exact owner/definition/revision bytes to match its immutable signed revision asset before accepting stored user parts; require verified pin evidence before accepting pinned parts. Pending saves and acknowledgement-only generation changes must not invalidate the prior active override. An explicit request override is authenticated by the containing request/job and does not require operator approval. Resolve once per request/batch/job and pass the immutable result into lower-level loops.
- [ ] Implement `enabled`, `read_only`, and `bypass_stored_overrides` from one parsed settings object. `bypass_stored_overrides` skips user rows but still honors explicit, pin, deployment, and packaged sources.
- [ ] Rerun all Service_Prompts unit tests and commit: `feat: resolve service prompts with provenance (<task-id>)`.

## Task 5: Add preview-safe bundle serialization

**Files:**

- Create: `tldw_Server_API/app/core/Service_Prompts/preview.py`
- Test: `tldw_Server_API/tests/Service_Prompts/test_preview.py`

- [ ] Write failing tests proving preview uses the same parser/resolver, never executes an LLM, caps returned content at the definition budget, marks locked parts, and redacts hidden parts.
- [ ] Implement a thin serializer over `ResolvedServicePrompt`; do not create a second resolution path.
- [ ] Rerun `python -m pytest -q tldw_Server_API/tests/Service_Prompts`.
- [ ] Commit: `feat: preview resolved service prompts safely (<task-id>)`.

## Task 6: Verify and secure

- [ ] Run `python -m pytest -q tldw_Server_API/tests/Service_Prompts tldw_Server_API/tests/Utils/test_prompt_loader_env_overrides.py tldw_Server_API/tests/Utils/test_prompt_loader_paths.py`.
- [ ] Run `python -m bandit -r tldw_Server_API/app/core/Service_Prompts tldw_Server_API/app/core/Utils/prompt_loader.py tldw_Server_API/app/services/startup_service_prompts.py tldw_Server_API/app/main.py -f json -o /tmp/bandit_service_prompt_registry.json` and review the JSON.
- [ ] Run `git diff --check` and confirm no dependency was added.
- [ ] Update the Backlog task and commit: `test: verify service prompt resolver contracts (<task-id>)`.
