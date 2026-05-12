# Moderation Review And Rules Remediation Design

Date: 2026-05-12
Backlog: TASK-303
Status: Draft approved for planning

## Purpose

The moderation area needs two distinct jobs:

1. Review moderation events and make decisions.
2. Configure the rules, policies, blocklists, overrides, and tester that produce those events.

The current WebUI/extension implementation only serves the second job. It is mounted as
`/moderation-playground`, while `/moderation` is not a reachable product route. The approved
direction is:

- `/moderation` becomes the moderation review queue.
- The existing rule configuration surface moves to `/moderation/rules`.
- `/moderation-playground` remains as a compatibility redirect to `/moderation/rules`.

This design turns the UX audit findings into a staged implementation plan that can be split into
reviewable tasks.

## Current Evidence

The audit found these current implementation anchors:

- Next route wrapper: `apps/tldw-frontend/pages/moderation-playground.tsx`.
- Shared route: `apps/packages/ui/src/routes/option-moderation-playground.tsx`.
- Shared route registry maps `/moderation-playground` in `apps/packages/ui/src/routes/route-registry.tsx`.
- Extension route registry maps `/moderation-playground` in `apps/tldw-frontend/extension/routes/route-registry.tsx`.
- Main shell: `apps/packages/ui/src/components/Option/ModerationPlayground/ModerationPlaygroundShell.tsx`.
- Rule panels: `PolicySettingsPanel`, `BlocklistStudioPanel`, `UserOverridesPanel`, `TestSandboxPanel`, and `AdvancedPanel`.
- Docs currently describe an admin moderation tab in `Docs/Code_Documentation/Moderation-Guardrails.md`.

Observed UX issues to address:

- `/moderation` does not route to a moderation experience.
- The current page is rules configuration, not review.
- Blocklist comments and blank lines can appear as active `literal block` rows.
- Raw replace, upload replace, and deletes lack preview, rollback, or audit context.
- The tester can show "Content Allowed" for risky-looking samples without explaining the active policy reason.
- Context-bar icon buttons lack accessible names.
- Several labels are visual only and not programmatically associated with controls.
- A 390px mobile viewport showed horizontal overflow.
- Review queue concepts are missing: needs-review status, item context, approve/block/redact/escalate/dismiss, undo, bulk review, completion, and audit trail.

## Product Model

### `/moderation` - Moderation Review

Primary user goal: decide what to do with moderation events.

This route should answer:

- What needs review?
- Why was this item flagged or allowed?
- What context is safe and necessary to inspect?
- What action should I take?
- Can I undo or audit the decision?
- How much work remains?

### `/moderation/rules` - Content Rules

Primary user goal: configure the system that produces moderation outcomes.

This route should answer:

- Is moderation enabled?
- What policy applies globally and per user?
- Which rules are active?
- Are rule edits valid before saving?
- What changed and can it be recovered?
- Why would a sample pass, block, redact, or warn?

## Target Route And Navigation Design

Route behavior:

- Add `/moderation` as the canonical review route.
- Add `/moderation/rules` as the canonical configuration route.
- Redirect `/moderation-playground` to `/moderation/rules`.
- Preserve WebUI and extension parity for route registry, sidebar shortcuts, settings nav, tutorials, and page inventory.

Navigation labels:

- `/moderation`: "Moderation Review".
- `/moderation/rules`: "Content Rules".
- Existing "Content Controls" language can remain as a broader grouping label only if the group contains both routes.
- Avoid "Playground" except for sandbox/test subfeatures.

Entry and exit flows:

- Review route should link to Content Rules when the user needs to adjust policy or blocklists.
- Rules route should link back to Review when test results or configuration status imply review work exists.
- Empty review state should offer "Open Content Rules" only as a secondary action, not as the primary page purpose.

## Review Queue MVP

The first review queue should be narrow and operational rather than a general analytics dashboard.

### Queue List

Show a dense list/table of moderation review items with:

- Status: needs review, approved, blocked, redacted, dismissed, escalated.
- Severity or priority when available.
- Category and phase: user message or AI response.
- Source type and source id where available.
- User id, session id, or anonymous local identifier as applicable.
- Created time and last decision time.
- Short excerpt with sensitive content safely redacted according to the backend contract.
- Recommended action if the backend provides one.

Required controls:

- Status filter.
- Category/severity filter.
- Source/user filter.
- Date sort.
- Search by safe excerpt or item id.
- Refresh.

### Item Detail

Selecting an item opens a detail pane or detail route with:

- Safe content excerpt and surrounding context.
- Source/provenance: source type, source id, route/workflow that generated it, and timestamp.
- Effective policy snapshot used at flag time.
- Matched rules with sanitized sample, category, action, rule id, and confidence if present.
- Prior decisions and comments.
- Available actions with clear consequences.

### Decision Actions

MVP actions:

- Approve.
- Block.
- Redact.
- Dismiss as not actionable.
- Escalate.

Every action should require:

- Decision reason, either optional for low-risk actions or required for block/escalate.
- Confirmation for destructive or bulk decisions.
- Immediate undo for recent single-item decisions.
- Audit event persisted by the backend.

### Empty, Loading, Error, And Permission States

Required states:

- Empty: no items need review, with counts for other statuses if available.
- Loading: skeleton list and disabled actions.
- Backend unreachable: actionable diagnostics and retry.
- Auth/permission denied: state the required permission and link to settings or docs.
- Partial data: show queue with an inline warning when details or audit trail cannot load.

## Content Rules Remediation

The current configuration page should be retained but renamed and hardened.

### Blocklist Studio

Fix display semantics:

- Use backend parsed metadata or lint results for `pattern_type`.
- Render comments as comments, blank lines as spacers or hidden rows, and active rules as active rules.
- Count only active rules in status badges.

Add power controls:

- Search pattern text.
- Filter by type, action, category, active/comment/empty.
- Sort by line number, action, category, or pattern.
- Active-only toggle by default.
- Optional bulk delete for selected active rules after preview.

Add safety:

- Raw editor save must show a diff preview before replacing the file.
- Upload replace must show file summary, diff, lint result, and a backup/download option.
- Delete rule should support at least one Stage 2 recovery path: undo in the current session, server-backed restore from previous version, or mandatory backup export before destructive save.
- Show version, ETag or equivalent revision, last loaded time, and conflict/reload guidance.

### Policy Settings

Clarify read-only versus editable settings:

- Keep server-controlled settings visibly read-only.
- Explain which editable runtime overrides affect testing immediately.
- Show unsaved changes by section.
- Persist-to-disk should keep its confirmation, but also state the target file/config scope.

### User Overrides

Improve recovery and clarity:

- Require stronger confirmation for deleting many overrides.
- Show a per-user effective policy summary before save.
- Show last saved time if backend data supports it.
- Keep table filter and selection, but add clear empty states for no override, no match, and disabled per-user overrides.

### Test Sandbox

Make tester outcomes explainable:

- Add a "Why this result?" panel for every run.
- Explain engine disabled, phase disabled, PII disabled, category not enabled, no matching rule, matched rule, user override, and global fallback.
- Group quick samples by expected behavior under the current policy when feasible.
- Surface the effective policy in human-readable rows before raw JSON.
- Keep raw JSON for debugging behind a details element.

### Advanced

Reduce destructive ambiguity:

- Split "Download" and "Upload and replace" into separate visual groups.
- Require preview/lint before uploads can replace active files.
- Add a clear warning that import affects server-wide or per-user configuration.
- Keep server config JSON read-only and label it as diagnostic.

## Accessibility And Responsive Requirements

Accessibility fixes:

- Add `aria-label` to icon-only context-bar actions such as Quick Test and Reload.
- Add accessible names to close/clear controls.
- Use `htmlFor`/`id` or `aria-labelledby` for input, textarea, and select labels.
- Use `aria-pressed` or radio semantics for segmented controls.
- Preserve visible focus and logical tab order through tab bars, tables, modals, and decision actions.
- Ensure error messages are announced or associated with their inputs.

Responsive fixes:

- No horizontal overflow at 390px, tablet, and desktop widths.
- Header, PageShell, status badges, tab bars, and category chips must not widen the viewport.
- Wide tables should scroll internally without causing page-level horizontal scroll.
- Decision/detail layouts should collapse into list-first then detail below on mobile.

## Data And API Contract Direction

Review data should be backend-owned. The frontend should not infer moderation events solely from local test history or blocklist rows.

Review content must also be backend-sanitized. The frontend should render only fields the backend explicitly marks safe for the moderator's permission level. Raw unsafe content, private identifiers, full surrounding context, and unredacted matched samples should not be exposed or exported unless a backend capability explicitly authorizes that access for the current user.

Proposed review item shape:

```ts
type ModerationReviewStatus =
  | "needs_review"
  | "approved"
  | "blocked"
  | "redacted"
  | "dismissed"
  | "escalated"

type ModerationDecisionAction =
  | "approve"
  | "block"
  | "redact"
  | "dismiss"
  | "escalate"

type ModerationReviewItem = {
  id: string
  status: ModerationReviewStatus
  phase: "input" | "output"
  source_type?: string
  source_id?: string
  user_id?: string
  session_id?: string
  created_at: string
  updated_at?: string
  severity?: "low" | "medium" | "high" | "critical"
  category?: string
  safe_fields: {
    excerpt: true
    context?: boolean
    user_id?: boolean
    session_id?: boolean
    source_id?: boolean
  }
  excerpt: string
  context?: {
    before?: string
    after?: string
  }
  effective_policy: Record<string, unknown>
  matches: Array<{
    rule_id?: string
    pattern_type?: "literal" | "regex" | "pii" | "category"
    category?: string
    action: "pass" | "block" | "redact" | "warn"
    sample?: string
    confidence?: number
  }>
  recommended_action?: ModerationDecisionAction | null
  decision?: ModerationDecision
}

type ModerationDecision = {
  id: string
  action: ModerationDecisionAction
  resulting_status: Exclude<ModerationReviewStatus, "needs_review">
  reason?: string
  decided_at: string
  decided_by?: string
  can_undo?: boolean
  undo_until?: string
}
```

Candidate endpoints:

- `GET /api/v1/moderation/review/items`
  - Query params: `status`, `category`, `severity`, `source_type`, `source_id`, `user_id`, `q`, `date_from`, `date_to`, `sort`, `cursor`, `limit`.
  - Response: `{items, next_cursor?, counts_by_status?, partial?: boolean, warnings?: string[]}`.
- `GET /api/v1/moderation/review/items/{item_id}`
  - Response: item detail with backend-sanitized context, policy snapshot, matches, and decision history allowed for the current permission level.
- `POST /api/v1/moderation/review/items/{item_id}/decision`
  - Request: `{action: ModerationDecisionAction, reason?: string, expected_status?: ModerationReviewStatus}`.
  - Response: `{item, decision, undo_token?}`.
- `POST /api/v1/moderation/review/items/{item_id}/undo`
  - Request: `{decision_id: string, undo_token?: string}`.
  - Response: `{item, undone_decision_id}`.
- `POST /api/v1/moderation/review/bulk-decision`
  - Request: `{item_ids: string[], action: ModerationDecisionAction, reason?: string, expected_status?: ModerationReviewStatus}`.
  - Response: `{updated: string[], failed: Array<{item_id: string, error: string}>}`.
- `GET /api/v1/moderation/review/audit`
  - Query params: `item_id`, `decision_id`, `actor`, `action`, `date_from`, `date_to`, `cursor`, `limit`.
  - Response: `{events, next_cursor?}` where event content is sanitized for export/display.

Minimum permissions:

- `MODERATION_REVIEW_READ`: list and inspect sanitized review items.
- `MODERATION_REVIEW_DECIDE`: make single-item decisions.
- `MODERATION_REVIEW_BULK_DECIDE`: make bulk decisions.
- `MODERATION_AUDIT_READ`: view/export audit records.
- `SYSTEM_CONFIGURE`: retain existing rule/policy configuration access.

If backend support is not ready, the first frontend implementation slice should use a typed mock fixture behind the existing E2E/smoke setup only. Production UI should make unsupported review state explicit rather than pretending data exists.

## Staged Implementation Plan

### Stage 1: Route And Naming Foundation

Goal: make routes match user intent.

Deliverables:

- `/moderation` route exists as review route.
- `/moderation/rules` hosts the existing configuration UI.
- `/moderation-playground` redirects to `/moderation/rules`.
- WebUI and extension route registries, page inventory, smoke mappings, settings nav, header shortcuts, docs, and tutorial route patterns are updated.
- First review route can show an honest empty/unsupported state if backend review data is not ready.

Success criteria:

- Direct navigation to `/moderation` no longer 404s.
- Direct navigation to `/moderation/rules` loads current rule configuration.
- Existing `/moderation-playground` links do not break.
- Visible naming distinguishes review from rules.

### Stage 2: Content Rules Hardening

Goal: make the existing configuration page safe and trustworthy.

Deliverables:

- Blocklist comments and empty lines no longer appear as active block rules.
- Blocklist search/filter/sort and active-only view exist.
- Raw replace and upload replace require preview and lint.
- Tester includes human-readable result explanation.
- Destructive actions have undo, version restore, or explicit backup path.

Success criteria:

- Comments are rendered as comments or hidden from active-rule counts.
- A sample that passes because moderation or PII is disabled says so clearly.
- Replacing a rule file cannot happen without previewing the resulting change.

### Stage 3: Accessibility And Responsive Pass

Goal: clear the audit blockers before adding the review workflow.

Deliverables:

- Accessible names and programmatic labels are fixed.
- Keyboard access works for tab bars, context actions, tables, modals, and tester controls.
- 390px mobile viewport has no page-level horizontal overflow.
- Wide tables scroll inside their containers.

Success criteria:

- CDP/Playwright checks can find named Quick Test and Reload buttons.
- Inputs and selects expose usable labels.
- Mobile `document.documentElement.scrollWidth <= window.innerWidth` for target routes.

### Stage 4: Review Backend Contract And Seed Fixtures

Goal: create the minimum backend-owned review substrate before production review UI.

Deliverables:

- Review item list/detail/decision/undo/audit endpoints or equivalent route contract.
- Pagination, filters, search, sort, and counts contract for the queue list.
- Sanitized review item fields and permission-gated context exposure.
- Decision action vocabulary mapped to resulting queue statuses.
- Minimal audit event persistence for review decisions.
- Seed or fixture data for local smoke tests and empty/error states.

Success criteria:

- Frontend can request review items without inventing production data from local tester history.
- A decision action returns the updated item, decision record, and undo affordance when available.
- The contract prevents raw unsafe/private content exposure unless a backend permission explicitly allows it.

### Stage 5: Review Queue MVP

Goal: implement the first usable `/moderation` review workflow.

Deliverables:

- Review list with status/category/user/date filters.
- Detail pane or detail route with context, policy snapshot, matched rules, and provenance.
- Single-item decisions: approve, block, redact, dismiss, escalate.
- Undo for recent single-item decisions.
- Loading, empty, error, partial data, and permission states.

Success criteria:

- A user can understand what needs review, inspect why it was flagged, decide an action, and recover from an immediate mistake.
- The UI does not require reading raw JSON to understand the moderation reason.

### Stage 6: Audit Trail And Recovery

Goal: make review and configuration accountable.

Deliverables:

- Item-level decision history.
- Rule/config change log.
- Last changed by/when where backend data supports it.
- Restore path for rule file versions or explicit backup restore workflow.
- Export for review/audit records.

Stage 2 must ship a minimum recovery path before destructive rules edits are allowed. This stage expands that minimum into durable history, cross-session restore, and exportable audit records.

Success criteria:

- A moderator can explain who changed what, when, and why.
- A bad rules edit or wrong decision has a documented recovery path.

### Stage 7: Power-User Efficiency

Goal: support repeated moderation work without slowing occasional admins.

Deliverables:

- Bulk review actions with preview and confirmation.
- Keyboard shortcuts for next item, approve, block, dismiss, and escalate.
- Persisted filters, sort, and selected view.
- Saved views such as high severity unreviewed and PII matches.
- Queue completion and health summary.

Success criteria:

- Returning users can process a queue without repeatedly reconfiguring filters.
- Bulk actions cannot silently affect hidden or filtered-out items.

### Stage 8: Fixtures And Regression Coverage

Goal: make the end state testable without depending on live unsafe content.

Deliverables:

- Fixtures for review items across statuses, categories, severities, phases, users, and sources.
- Fixtures for enabled/disabled engine, PII on/off, invalid regex, many rules, many overrides, permissions, backend 500, and import conflicts.
- E2E coverage for `/moderation`, `/moderation/rules`, redirect behavior, decisions, undo, mobile layout, and accessibility labels.

Success criteria:

- The review queue and rules pages can be validated in CI/smoke without production data.
- Regression tests cover the issues identified by the audit.

## Error Handling Principles

- Prefer inline, actionable error states over generic toast-only failures.
- Preserve any partial data that is safe to show.
- Error text should name the failed action and next available action.
- Permission errors should identify required capability, not just say access denied.
- Conflict errors on rule versions should offer reload, compare, or retry.

## Out Of Scope

- General backend moderation theory or policy taxonomy redesign.
- Rewriting unrelated WebUI layout systems except where needed to fix `/moderation` and `/moderation/rules`.
- Adding live automated enforcement changes beyond the contracts needed to review and configure moderation.
- Broad repo-wide TypeScript cleanup unrelated to these routes.

## Risks And Mitigations

- Risk: Review queue needs backend storage that may not exist yet.
  - Mitigation: make Stage 1 route honest with empty/unsupported state, then create backend-contract implementation tasks before full review UI.

- Risk: Existing `/moderation-playground` links are already in docs/tests/bookmarks.
  - Mitigation: keep redirect and update docs over time.

- Risk: Configuration and review concerns get mixed again.
  - Mitigation: keep `/moderation` for decisions and `/moderation/rules` for policy/rules; cross-link only for task handoff.

- Risk: Destructive rule editing remains unsafe if only frontend confirmations are added.
  - Mitigation: pair frontend preview with backend version/ETag conflict handling and backup/restore.

## Recommended Task Breakdown After Spec Approval

1. Route and naming split for `/moderation` and `/moderation/rules`.
2. Content Rules blocklist display/search/filter and tester explainability.
3. Content Rules destructive-action preview and recovery.
4. Accessibility and responsive hardening for both moderation routes.
5. Backend review item contract, permissions, audit persistence, and seed fixtures.
6. Review Queue MVP frontend.
7. Audit trail, durable recovery, and export.
8. Power-user bulk review and saved views.
9. End-to-end moderation regression suite.

Each implementation task should include WebUI and extension parity checks when the affected route is shared.
