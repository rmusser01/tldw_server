# Moderation Backend Review Design

Date: 2026-04-07
Topic: Moderation backend module review
Status: Approved design

## Objective

Review the backend Moderation module in `tldw_server` for concrete issues, bugs, risks, and improvement opportunities with a findings-first, risk-weighted audit.

The review should prioritize actionable defects and behaviorally significant risks, while also calling out lower-severity maintainability, testability, and design improvements that would reduce future regression risk.

## Scope

This review is centered on the backend moderation surface:

- `tldw_Server_API/app/core/Moderation/moderation_service.py`
- `tldw_Server_API/app/api/v1/endpoints/moderation.py`
- `tldw_Server_API/app/api/v1/schemas/moderation_schemas.py`
- `tldw_Server_API/app/api/v1/endpoints/chat.py` only where needed to confirm the real moderation enforcement contract
- moderation-focused backend tests under `tldw_Server_API/tests/`

This includes:

- moderation policy construction and merging
- blocklist parsing and validation behavior
- per-user overrides and runtime overrides
- effective-policy inspection and tester behavior
- real caller behavior for input and output moderation in the chat backend
- moderation admin authorization and permission-gating behavior
- blocklist persistence, optimistic concurrency, and reload semantics
- test coverage that defines or implies moderation backend contracts

This review excludes:

- moderation playground UI and frontend service code
- unrelated chat or workflow behavior except where needed to confirm moderation contracts
- implementation or remediation work during the review phase unless explicitly requested later

## Approaches Considered

### Recommended: Risk-based backend review

Trace the moderation flow end to end across service, endpoints, schemas, and tests, focusing on high-risk seams instead of treating every file with equal depth.

Why this is preferred:

- moderation is configuration-heavy, so many real problems live at merge and boundary points rather than isolated functions
- it balances speed and confidence better than a pure static skim
- it yields findings that are easier to triage than a flat commentary pass

### Alternative: Fast static review

Inspect the code and tests without targeted behavioral cross-checking.

Trade-offs:

- fastest option
- useful for a quick health check
- lower confidence on semantic mismatches, persistence behavior, and admin-surface edge cases

### Alternative: Deep assurance review

Perform the risk-based review and then add targeted runtime verification or adversarial probing for parsing, persistence, and regex-safety claims.

Trade-offs:

- highest confidence
- slower and more likely to expand beyond review into validation work
- not necessary unless the static evidence leaves important ambiguity

## Chosen Method

Use the recommended risk-based backend review with five explicit review axes:

1. `Policy construction and merge behavior`
   Inspect global config loading, runtime overrides, per-user overrides, and reload behavior to see whether effective policy state can drift, downgrade, or become internally inconsistent.
2. `Rule parsing and safety`
   Inspect blocklist grammar parsing, `#categories`, `-> action` handling, regex parsing, dangerous-pattern heuristics, and failure handling for malformed rules.
3. `Enforcement semantics`
   Compare `check_text`, redaction, action evaluation, sanitized snippet generation, category gating, phase gating, and tester behavior for semantic mismatches or surprising contracts.
4. `Persistence and concurrency`
   Review blocklist writes, override writes, file persistence, atomicity assumptions, ETag handling, and optimistic concurrency behavior.
5. `Test adequacy`
   Use unit and integration tests to determine what behavior is actually pinned down, what is only implied, and which high-risk branches appear under-tested.

The review should use this initial audit seed set before expanding further:

- `tldw_Server_API/tests/unit/test_moderation_blocklist_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_check_text_snippet.py`
- `tldw_Server_API/tests/unit/test_moderation_effective_settings.py`
- `tldw_Server_API/tests/unit/test_moderation_env_parse.py`
- `tldw_Server_API/tests/unit/test_moderation_etag_handling.py`
- `tldw_Server_API/tests/unit/test_moderation_redact_categories.py`
- `tldw_Server_API/tests/unit/test_moderation_runtime_overrides_bool.py`
- `tldw_Server_API/tests/unit/test_moderation_test_endpoint_sample.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_contract.py`
- `tldw_Server_API/tests/unit/test_moderation_user_override_validation.py`
- `tldw_Server_API/tests/AuthNZ_Unit/test_moderation_permissions_claims.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation.py`
- `tldw_Server_API/tests/Chat_NEW/integration/test_moderation_categories.py`

## Evidence Model

The review will rely on:

- direct source inspection of the moderation service, endpoint layer, and schemas
- direct inspection of moderation-focused backend tests
- targeted inspection of closely related callers only when needed to confirm moderation behavior
- recent git history when it helps explain churn or regression-prone areas

The review is primarily static and read-first. Runtime verification may be added selectively if a specific suspected issue cannot be classified confidently from source and tests alone.

When certainty is limited, observations should be labeled as probable risks or open questions rather than overstated as confirmed defects.

For persistence and concurrency claims, the bar should be stricter. If the review raises an issue involving atomicity, reload safety, `If-Match` or ETag conflict behavior, or file-write durability, it should perform targeted verification whenever feasible instead of relying on source inspection alone.

## Findings Model

The final output should be findings-first and split into clear evidence bands.

### Confirmed findings

These are the primary deliverable. They include:

- correctness bugs
- security or privacy issues
- unsafe parsing or persistence behavior
- API contract mismatches
- missing tests that leave a high-risk moderation path effectively unguarded

Each finding should include:

- severity
- confidence
- concise issue statement
- why it matters in runtime, safety, or operational terms
- concrete file and line references

### Probable risks

These are issues that appear material but are not fully proven from the available evidence. They should still include impact and file references, but they must be labeled clearly so they are not mistaken for confirmed defects.

### Improvements

These are lower-severity but still worthwhile observations, such as:

- maintainability problems that increase regression risk
- surprising semantics that may confuse future maintainers or operators
- validation or structure gaps that are not current defects but are likely future bug sources
- test additions that materially improve confidence

Minor style commentary and low-signal cleanup should be omitted.

### Open questions

If behavior remains ambiguous after source and test inspection, list the ambiguity explicitly instead of folding it into either confirmed findings or improvements.

## Severity and Prioritization

Severity should be balanced across:

- correctness and functional behavior
- security and privacy exposure
- persistence, admin, and operational safety
- maintainability when it materially affects future reliability

When findings are comparable, rank higher the issue with the larger blast radius, higher probability of silent policy failure, greater chance of unsafe moderation gaps, or greater likelihood of repeated regressions.

## Review Focus Areas

The review should bias toward these concrete questions:

- Can config, runtime overrides, and per-user overrides combine into unexpected effective policies?
- Can malformed blocklist entries or override rules be silently accepted, misparsed, or applied differently than documented?
- Do input/output phase handling, category handling, and action handling behave consistently across service methods, admin tester surfaces, and the real `chat.py` caller path?
- Are blocklist and override persistence paths safe under reloads, partial failures, and concurrent admin edits?
- Do schemas and endpoint behavior align, or are there request/response edge cases where the API contract is looser or stricter than the service expects?
- Do moderation admin endpoints enforce the intended role and permission boundaries, and do their tests actually pin that down?
- Do the tests materially protect the highest-risk moderation semantics, or do they mostly cover happy-path behavior?

## Execution Boundaries

- The review remains backend-only.
- Cross-module behavior may be noted only where it directly defines moderation backend semantics.
- The review remains non-invasive and should not silently turn into remediation work.
- The final product is a code-review style findings report, not a fix implementation plan.

## Constraints

- Do not broaden this run into a full chat or frontend moderation review.
- Do not present speculation as a confirmed defect.
- Do not bury concrete defects under general cleanup commentary.
- Do not recommend broad refactors unless they clearly reduce a real moderation risk.

## Final Deliverable

The final response to the user will:

- list findings first, ordered by severity
- separate confirmed findings, probable risks, improvements, and open questions
- include file references and concrete behavioral impact
- mention important open questions or assumptions only where needed

The goal is a backend moderation review artifact that can be used directly to prioritize follow-up fixes.

## Success Criteria

The design is successful when:

- the review stays within the backend moderation surface
- findings are evidence-backed and actionable
- real defects and behaviorally significant risks are prioritized above cleanup
- lower-severity design, validation, and testability improvements are included only when they materially matter
- the final report is easy to use as a triage document for follow-up remediation
