# Moderation

The Moderation module provides configurable content policy checks, review
workflows, supervised policy evaluation, governance helpers, category taxonomy,
and family-wizard policy materialization. It is safety-sensitive and should stay
explicit about when text is blocked, warned, redacted, queued for review, or
allowed.

## Start Here

- Runtime checks: `moderation_service.py`.
- Supervised and governance policy helpers: `supervised_policy.py`,
  `governance_utils.py`, `governance_io.py`, and `conflict_resolution.py`.
- Reviews: `review_service.py` and `review_store.py`.
- Taxonomy and matching: `category_taxonomy.py` and `semantic_matcher.py`.
- API endpoint: `app/api/v1/endpoints/moderation.py`; family setup endpoint:
  `app/api/v1/endpoints/family_wizard.py`.
- Tests: moderation tests under `tests/unit/`, chat integration moderation tests,
  Guardian family-wizard tests, and
  `tests/AuthNZ_Unit/test_moderation_permissions_claims.py`.

## Responsibilities

- Load and apply moderation policy from config and runtime overrides.
- Evaluate literal, regex, category, and optional PII-style rules.
- Support supervised policies and governance schedules/chat-type scoping.
- Persist and process moderation review items.
- Materialize family-wizard settings into moderation/governance policy records.

## Module Map

- `moderation_service.py` evaluates text and applies block/warn/redact actions.
- `supervised_policy.py` composes policy decisions and review requirements.
- `review_service.py` and `review_store.py` own review lifecycle data.
- `category_taxonomy.py` defines category naming and metadata.
- `semantic_matcher.py` supports similarity-style policy matching.
- `family_wizard_materializer.py` converts family wizard input into policy data.

## How It Connects

- Chat and other text-generation endpoints call moderation before or after model
  output depending on route policy.
- Governance and Monitoring share policy schedule/chat-type utility functions.
- AuthNZ permission tests protect moderation endpoint claims.

## Extension Points

- Add rule sources in `moderation_service.py` only after defining conflict and
  precedence behavior.
- Add review fields through `review_store.py`, schemas/endpoints, and review
  tests together.
- Keep governance helper functions pure so they can be reused by Monitoring and
  supervised policy code.

## Testing

- Runtime text checks: `tests/unit/test_moderation_check_text_snippet.py`,
  `tests/unit/test_moderation_redact_categories.py`, and
  `tests/Chat_NEW/integration/test_moderation.py`.
- Endpoint and permission coverage: `tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
  and `tests/unit/test_moderation_test_endpoint_sample.py`.
- Family/governance behavior: `tests/Guardian/test_family_wizard_endpoints.py`,
  `tests/Guardian/test_family_wizard_materialization.py`, and related Guardian
  family-wizard tests. There is no direct
  `tests/Moderation/` directory in this tree; use the adjacent unit and chat
  integration moderation tests listed above.

## Gotchas

- Regex rules can create performance and false-positive risks. Keep scan limits,
  replacement limits, and tests for edge cases.
- Do not log raw moderated content when the failure path may include sensitive
  user input.
