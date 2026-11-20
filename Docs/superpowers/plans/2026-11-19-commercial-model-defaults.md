# Commercial model defaults and selectable catalogs implementation plan

> **For agentic workers:** Use superpowers:executing-plans or superpowers:subagent-driven-development. Steps use checkboxes to track verification.

**Goal** Ship current commercial text-generation defaults and model-appropriate output budgets, then backport the verified behavior to hosted.

**Architecture** Keep provider discovery authoritative. Store bundled model and budget choices at the configuration and adapter boundary. Keep published capabilities and historical pricing in their existing catalogs.

**Tech stack** Python 3.12, ConfigParser, pytest, hosted backend patches.

**Spec** [Commercial model defaults design](../specs/2026-11-19-commercial-model-defaults-design.md)

## Review focus

- An explicitly configured output limit, including OpenAI's `max_completion_tokens`, wins over a bundled default.
- Changing the model or using a custom endpoint cannot inherit the larger bundled limit for a different model or endpoint.
- A retired item can still have a historical price while the picker excludes it.
- A provider-specific request field or non-streaming restriction can make a valid model ceiling unsuitable for that transport.
- A hosted patch has to replay against the frozen upstream pin and the existing patch sequence.

## 1. Audit defaults and selectable data

- [x] Record one row per commercial adapter in `Docs/Development/Commercial_Model_Defaults.md`, with supported endpoints, old and current IDs, dated official evidence, advertised output where known, and runtime budget. Cover providers without a shipped default and region or account limits.
- [x] Add a test in `tldw_Server_API/tests/LLM_Calls/test_commercial_model_defaults.py` that reads the bundled configuration and checks the verified defaults, including `deepseek-flash`. Add assertions that retired static entries remain priced while absent from selection.
- [x] Run the new test before updating the data. The old DeepSeek ID and retired static entries must fail the new assertions.
- [x] Refresh `tldw_Server_API/Config_Files/config.txt`, matching fallbacks in `app/core/config.py`, and relevant adapter-level fallback IDs. Hide confirmed retirements and enrich current entries in `model_pricing.json` and `app/core/LLM_Calls/model_metadata.py`. Keep authenticated discovery and explicit configured selections unchanged.

## 2. Apply and verify output defaults

- [x] In the same test module, exercise the resolved budget with official endpoints and mocked adapter payloads. Assert that DeepSeek Flash sends 200,000, a 4,096 request or operator override still wins, alternate models and custom endpoints retain the previous fallback, and Anthropic's streaming and non-streaming budgets respect the transport.
- [x] Run those assertions against the old provider payloads. Confirm the new defaults fail and the existing override cases pass.
- [x] Add `apply_commercial_output_default(provider: str, request: dict[str, Any]) -> None` at the common bound-request boundary. Resolve the model and official endpoint before applying a verified bundled budget. Keep configured limits first and allow the adapter to produce the native field. Handle model-specific sampling or streaming requirements in the relevant adapter.
- [x] Run the focused defaults, adapter, summary, inventory, and pricing tests. Run Ruff, Bandit, and git diff checks on the affected scope. Review the entire public diff, commit the verified work, and open the `dev` PR.

## 3. Backport to hosted

- [x] Translate the verified public diff into a numbered backend patch for the hosted pin `b2c791d060645da16c93471d65c9395551e611b2`. Adapt catalog metadata to the pinned module layout. Preserve existing hosted overlays and DeepSeek-specific explicit limits in other features.
- [x] Stage the patch so the materializer includes it. Run the complete patch replay and repeat equivalent offline config, inventory, and payload assertions against the generated runtime.
- [x] Review the hosted diff and patch replay. Update the hosted Backlog record, commit, and open the hosted `main` PR with the public PR link and distinct verification results.
