# Research Studio UX Remediation Design

Date: 2026-05-12
Owner: Codex collaboration session
Status: Draft for user review
Backlog: TASK-304

## Summary

Research Studio is the intended user-facing name for the existing
`WorkspacePlayground` research workspace surface. The current implementation has
strong foundations: a three-pane desktop layout, mobile Sources / Chat / Studio
tabs, source-gated generation, generated artifact workflows, onboarding
prompts, and skip links. The audit identified several user-facing gaps around
route naming, mobile deep links, empty states, planned work products, degraded
health handling, progressive disclosure, and returning-user efficiency.

This design turns those audit findings into a staged remediation series. The
primary product decision is:

- `/research-studio` becomes the canonical user-facing route.
- `/workspace-playground` remains a backward-compatible alias.
- `/workspace-studio` becomes a compatibility alias instead of a 404.
- User-facing labels standardize on "Research Studio."
- Stable internal identifiers that are already persisted or exported keep their
  current `workspace-playground` names unless a separate migration is approved.

The remediation is intentionally staged so each slice can ship and be reviewed
independently. It is not a general frontend cleanup and should not drift into
unrelated chat, media, backend architecture, or workspace persistence work.

## Goals

1. Make Research Studio reachable and name-consistent across WebUI, extension,
   docs, navigation, search, tutorials, and helper copy.
2. Preserve backward compatibility for existing links, saved routes, exported
   bundles, telemetry state, and persisted workspace records.
3. Support canonical mobile deep links through `?tab=sources|chat|studio`,
   with `?tab=studio` as the documented way to open Studio on mobile.
4. Make Studio work-product-first: actionable work products lead the experience,
   and planned work products are hidden until usable.
5. Replace disabled-control-first empty states with clear source-selection and
   readiness guidance.
6. Let degraded-but-reachable health states render the app with visible warning
   affordances, while preserving blocking behavior for unreachable or explicitly
   unusable states.
7. Improve returning-user speed through persistence, compact mobile behavior,
   and focused accessibility checks.

## Non-Goals

- Do not rename persisted storage keys, export bundle formats, telemetry storage
  keys, event names, or server-side data fields in this remediation series.
- Do not redesign the entire app shell, global sidebar, chat model selector, or
  media ingestion flow.
- Do not invent new work-product systems beside the existing Research Studio
  template and artifact machinery.
- Do not implement non-actionable planned work products as visible end-user
  affordances.
- Do not solve backend health semantics in one pass if the health payload does
  not yet expose capability-specific details.

## Current Repo Anchors

The current WebUI page is implemented as a Next page at
`apps/tldw-frontend/pages/workspace-playground.tsx`, which dynamically imports
the shared route component. The extension route registry also exposes
`/workspace-playground` with a nav label token that renders as "Research
Studio."

Important implementation anchors:

- `apps/tldw-frontend/pages/workspace-playground.tsx`
- `apps/tldw-frontend/extension/routes/route-registry.tsx`
- `apps/packages/ui/src/routes/route-paths.ts`
- `apps/packages/ui/src/routes/route-registry.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/index.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/index.tsx`
- `apps/packages/ui/src/components/Option/WorkspacePlayground/StudioPane/WorkProductTemplateChooser.tsx`
- `apps/packages/ui/src/tutorials/definitions/workspace-playground.ts`
- `apps/tldw-frontend/components/navigation/RouteRedirect.tsx`
- `apps/tldw-frontend/components/networking/ServerReadinessGate.tsx`
- `Docs/Operations/Research_Studio_Trust_Status_Telemetry_Runbook.md`

The existing `RouteRedirect` helper already supports query/hash preservation and
route-alias telemetry. That should be reused for web aliases instead of
creating a new redirect primitive.

## Evidence From Audit

Browser/CDP observations:

- `http://127.0.0.1:3000/workspace-studio` returned a 404 with "We could not
  find that route."
- `http://127.0.0.1:3000/workspace-playground` loaded the implemented Research
  Studio surface.
- Desktop showed Sources, Chat, and Studio panes together.
- Mobile showed Sources / Chat / Studio tabs; direct CDP activation of Studio
  worked.
- Studio showed many disabled generation controls before the no-source hint.
- "More outputs (6)" expanded even when all revealed outputs were disabled.
- Slides and Audio settings appeared before source selection or output intent.
- Repeated `/api/v1/health` responses with HTTP 206 caused the readiness gate
  to retry before the app became usable.

Captured audit artifacts:

- `/private/tmp/workspace-studio-cdp-audit/desktop-requested-route.png`
- `/private/tmp/workspace-studio-cdp-audit/desktop-implemented-studio-anchor.png`
- `/private/tmp/workspace-studio-cdp-audit/desktop-desktop-more-outputs.png`
- `/private/tmp/workspace-studio-cdp-audit/mobile-direct-cdp-studio-tab.png`
- `/private/tmp/workspace-studio-cdp-audit/audit-data.json`

User decisions after the audit:

- "Research Studio" is the proper product name.
- `/research-studio` should be canonical.
- `/workspace-playground` and `/workspace-studio` should redirect or alias to
  the canonical route.
- Mobile deep links should use `?tab=studio`.
- Health capability semantics are not yet clear.
- Planned work products should be hidden until actionable.
- Studio should default to a work-product-first model.

## Design Principles

### Compatibility Before Renaming

Route and label cleanup must not become a data migration. User-facing names can
change safely; persisted identifiers need a separate migration plan. Internal
constants and filenames may keep `workspace-playground` names until there is a
low-risk reason to rename them.

### Work Products Before Output Types

Research Studio should ask users what work product they want, not force them to
choose a low-level generation mode first. Output types remain useful but should
be secondary to the work-product path.

### Explain Preconditions Before Disabling Actions

Disabled controls prevent accidental or expensive work, but they should not be
the first thing a user has to interpret. Source selection, selected source
count, and generation readiness should appear before generation actions.

### Degraded Is Not Always Blocked

A degraded backend state can still allow browsing, source review, or even
chatting. The app should render when the API is reachable and give scoped
warnings. Blocking should be reserved for unreachable or explicitly unusable
states.

### Mobile Deep Links Are Route State

Mobile tab state should be represented in the URL when it matters. `?tab=studio`
is the canonical contract; hash anchors remain legacy compatibility.

## Staged Remediation Series

### Stage 0: Tracking And Baseline

Goal: preserve the audit evidence and create clear work orders before behavior
changes begin.

Scope:

- Keep TASK-304 as the parent design record.
- Later implementation planning should create child tasks for each stage.
- Attach this spec and the audit screenshot locations to those tasks.
- Capture current route behavior, Studio no-source behavior, and mobile tab
  behavior as baseline evidence.

Success criteria:

- Parent task references this design spec.
- Each future implementation stage has a distinct task with acceptance
  criteria, test expectations, and dependencies.
- No code behavior changes are made in this stage.

Verification:

- Backlog task records are present and linked.
- Spec exists under `Docs/superpowers/specs/`.

### Stage 1A: Canonical Route Compatibility

Goal: make Research Studio reachable at the correct canonical route while
preserving existing links.

Scope:

- Add `/research-studio` as the canonical WebUI page.
- Add `/workspace-playground` redirect or alias to `/research-studio`.
- Add `/workspace-studio` redirect or alias to `/research-studio`.
- Preserve query params and hashes during redirects where practical.
- Update shared and extension route registries so the extension can reach the
  same canonical surface.
- Keep internal component names and persisted identifiers stable.

Route behavior:

- `/research-studio` renders the Research Studio surface.
- `/research-studio?tab=studio` renders the same surface and opens Studio on
  mobile after Stage 2.
- `/workspace-playground?shared=abc` redirects or aliases to
  `/research-studio?shared=abc`.
- `/workspace-studio?tab=studio` redirects or aliases to
  `/research-studio?tab=studio`.

Implementation notes:

- Reuse `RouteRedirect` for Next route aliases where a redirect page is enough.
- For the extension, prefer registry-level aliases or `Navigate` redirects that
  preserve path state according to existing extension routing conventions.
- Do not use static extension redirects that drop `search` or `hash` state.
  The Research Studio aliases must preserve at least `?tab=...`, `?shared=...`,
  and any existing prefill or source-transfer query params.
- If a Next redirect page is used for a legacy route, account for the fact that
  non-settings routes are currently wrapped by `ServerReadinessGate`. Either
  land degraded-health pass-through before browser-visible alias verification,
  or provide a narrow redirect-only bypass so legacy route correction is not
  blocked by API health.
- Consider adding `RESEARCH_STUDIO_PATH = "/research-studio"` and keeping
  `WORKSPACE_PLAYGROUND_PATH = "/workspace-playground"` as a legacy constant
  during migration.
- `VIEWPORT_CONSTRAINED_PATHS` should include the canonical route and maintain
  compatibility for the legacy route until callers are updated.

Success criteria:

- The canonical route does not 404.
- Both legacy paths reach the same user-visible surface.
- Existing shared links keep their query params.
- Extension routing has parity with the WebUI route behavior.

Verification:

- Focused route tests for WebUI redirect/alias behavior.
- Extension route-registry tests for `/research-studio` and legacy aliases.
- CDP smoke checks for desktop `/research-studio` and
  `/workspace-studio?tab=studio`.

### Stage 1B: User-Facing Naming Sweep

Goal: align user-visible information scent around "Research Studio" without
breaking internal persistence.

Scope:

- Update nav labels, command palette entries, header shortcuts, docs references,
  tutorial labels, Quick Chat helper route descriptions, and visible copy from
  "Workspace Playground" or "Workspace Studio" to "Research Studio."
- Inventory known handoff and entry-point callers before editing paths. This
  includes Knowledge QA handoff links, Shared With Me links, header shortcuts,
  Quick Chat helper docs, route-path builders, tutorial route normalization,
  extension options routes, extension sidepanel/popout route state, and any
  source-transfer or prefill utilities that currently target
  `/workspace-playground`.
- Keep local storage keys, export bundle format names, telemetry storage keys,
  telemetry event types, and generated artifact field names stable.
- If internal source names remain `WorkspacePlayground`, document that as a
  compatibility boundary rather than a product name.

Explicitly preserve:

- `tldw.workspace-playground.bundle`
- `tldw:workspace:playground:telemetry`
- `tldw:workspace-playground:*` local storage keys
- Existing tutorial IDs such as `workspace-playground-basics` unless a separate
  tutorial migration is approved
- Existing event names such as `workspace-playground:discuss-artifact`

Success criteria:

- Users see "Research Studio" in navigation and help surfaces.
- Existing persisted workspaces, exports, telemetry, and local preferences keep
  working.
- No user-facing documentation points users to `/workspace-studio`.

Verification:

- Text search confirms intended visible strings and documents changed.
- Tests covering route labels, header shortcuts, and helper docs remain green.
- Manual audit of docs that intentionally retain legacy names for compatibility
  notes.

### Stage 2: Mobile Route-State Contract

Goal: make `?tab=studio` the canonical mobile deep-link mechanism.

Scope:

- Parse `tab` query param with allowed values `sources`, `chat`, and `studio`.
- Default to Chat when `tab` is absent or invalid.
- On mobile, initialize and update `activeTab` from the parsed route state.
- On desktop, keep panes visible/collapsible but use `tab=studio` as a focus
  hint when useful.
- Preserve `?tab=studio` through route aliases.
- Treat `#workspace-studio-panel` as legacy compatibility: it may focus the
  Studio panel on desktop, but docs should use `?tab=studio`.

Implementation notes:

- Avoid scattering route parsing across effects. Create one helper for Research
  Studio route-state normalization.
- Reuse existing `focusWorkspacePane("studio")` behavior where possible.
- Tab normalization must preserve unrelated query params. For example,
  `?shared=abc&tab=studio` remains shared workspace state plus Studio focus,
  not just Studio focus.
- Avoid creating browser history churn when syncing URL state from tab clicks.
  The first implementation may read the URL without writing every tab change
  back, unless product explicitly wants persistent tab URLs after manual clicks.

Success criteria:

- `/research-studio?tab=studio` opens Studio on mobile.
- `/research-studio?tab=sources` opens Sources on mobile.
- Invalid tabs fall back to Chat without an error state.
- Legacy aliases preserve the query param.

Verification:

- Component tests for route-state parsing.
- WorkspacePlayground tests for initial mobile tab selection.
- CDP mobile screenshot for `/research-studio?tab=studio`.

### Stage 3A: Degraded Health Pass-Through

Goal: avoid false blocking when the API is reachable but reports degraded
health.

Scope:

- Treat reachable degraded health as app-enterable.
- Render Research Studio with a visible warning or status affordance naming the
  degraded state when available.
- Keep unreachable, malformed, or explicitly unhealthy responses in the existing
  retry/blocking path.
- Avoid claiming generation is safe unless the health payload confirms the
  relevant dependencies.

Implementation notes:

- If `ServerReadinessGate` has already been fixed on another branch or task,
  Research Studio implementation should reuse that shared gate behavior instead
  of duplicating a local workaround.
- The warning should link to health diagnostics when available.
- The app should not silently wait for 15 seconds when the API is degraded but
  reachable.
- This stage may need to land before route-alias browser verification if the
  common local development state is degraded. Otherwise `/workspace-studio`
  could stop being a 404 but still appear blocked behind readiness retries.

Success criteria:

- HTTP 200/206 degraded health can enter the app.
- Unreachable API still blocks/retries.
- Users can see that the server is degraded and where to inspect details.

Verification:

- Focused readiness-gate tests for degraded enterable and unhealthy blocked
  states.
- CDP smoke with mocked or observed degraded health.

### Stage 3B: Capability-Aware Health Follow-Up

Goal: define which capabilities remain safe under degraded backend states once
the backend payload semantics are clear.

Scope:

- Define a Research Studio capability matrix:
  - browse existing workspace
  - manage local workspace sources
  - use chat
  - generate artifacts
  - export or download artifacts
  - sync/share workspace
- Map each capability to backend health checks when the payload exposes enough
  detail.
- Disable or warn only at the capability boundary, not at the entire app
  boundary.

Open decision:

- The exact health checks that gate chat and generation are not yet known. This
  stage should start with an investigation task or backend contract review.

Success criteria:

- Health degradation affects the smallest relevant Research Studio action.
- Users understand whether they can browse, chat, generate, or export.
- Generation and expensive actions avoid unsafe attempts when dependencies are
  known unavailable.

Verification:

- Unit tests for capability derivation.
- UI tests for warning/disabled generation states.
- Backend contract notes or docs update.

### Stage 4: Work-Product-First Studio IA

Goal: make actionable work products the primary Studio decision.

Scope:

- Hide planned work products until they are actionable.
- Lead Studio with available work products, starting with Executive Brief and
  any other implemented templates.
- Move raw output types such as Summary, Report, Flashcards, Quiz, Slides, Data
  Table, Timeline, Compare Sources, Mind Map, and Audio Summary into a secondary
  "Other outputs" or "More outputs" area that appears after readiness context.
- Keep output-type generation available where it already works; do not remove
  implemented output functionality.
- Ensure the selected work product drives any relevant default output type,
  prompt, template metadata, and artifact metadata.

Implementation notes:

- `WorkProductTemplateChooser` currently marks non-executive templates as
  planned. It should filter unavailable templates out of the visible list rather
  than showing disabled "Planned" cards.
- If only one work product is actionable, the UI should still read as a
  work-product-first flow, with secondary output types clearly separated.
- Recent output types may remain useful for power users but should not override
  the work-product-first hierarchy.

Success criteria:

- Planned templates are not visible to end users.
- The primary Studio path is work-product selection.
- Existing output generation remains discoverable as secondary actions.

Verification:

- WorkProductTemplateChooser tests for hidden planned templates.
- StudioPane tests for visible actionable work product and secondary outputs.
- CDP screenshot verifies no "Planned" badges in the end-user default state.

### Stage 5: No-Source And Progressive Disclosure States

Goal: show the source requirement before unavailable generation actions.

Scope:

- Move "Select sources to generate" guidance to the top of Studio.
- Show selected source count and a source-selection CTA near the top.
- When zero sources are selected, show a source-empty state before work products
  and output types.
- Hide or visually subordinate generation controls until the source requirement
  is met.
- Hide Slides settings until Slides is selected or relevant.
- Hide Audio Settings until Audio Summary is selected or relevant.
- Change "More outputs" so it does not expand into a wall of disabled actions
  when no sources are selected.

Implementation notes:

- Preserve source-gating behavior that prevents accidental generation.
- Tooltips are not enough for no-source explanation because mobile users and
  keyboard users may not discover them.
- The CTA should route users to the Sources tab on mobile and Sources pane on
  desktop, reusing existing pane focus helpers.

Success criteria:

- A first-time user sees what Studio needs before seeing disabled generation.
- No-source Studio provides a direct Add/select sources action.
- Advanced output settings are progressively disclosed.

Verification:

- StudioPane no-source tests.
- Keyboard/focus test for Add/select sources CTA.
- CDP desktop and mobile screenshots for no-source state.

### Stage 6: Returning-User Efficiency

Goal: reduce repeated setup friction for users who already understand Research
Studio.

Scope:

- Persist the last active mobile tab where appropriate.
- Ensure onboarding dismissal remains persistent.
- Persist advanced settings collapsed/expanded state only where it improves
  repeat use without hiding critical warnings.
- Preserve or improve recent work-product/output shortcuts.
- Compact the mobile header/status stack after first-use or after onboarding is
  dismissed.

Implementation notes:

- Persistence should use safe storage helpers and tolerate storage failures.
- Do not persist states that could hide urgent warnings.
- If both URL `?tab=...` and persisted last tab exist, URL takes priority.

Success criteria:

- Returning mobile users can land directly in Studio when intended.
- Dismissed onboarding does not keep consuming vertical space.
- Frequent output/work-product choices are easy to repeat.

Verification:

- Storage resilience tests.
- Mobile responsive tests for compacted chrome.
- CDP mobile screenshots before and after onboarding dismissal.

### Stage 7: Accessibility, Documentation, And Release Verification

Goal: close the remediation with focused verification instead of broad frontend
cleanup.

Scope:

- Increase or preserve effective hit targets for key Research Studio controls.
- Verify keyboard focus order across Sources, Chat, and Studio.
- Confirm skip links still work after route and tab changes.
- Update user-facing docs and developer docs for `/research-studio`.
- Record legacy alias behavior in docs where appropriate.
- Run focused tests and CDP screenshots across desktop and mobile.

Success criteria:

- Main Research Studio controls meet practical desktop and mobile target sizes.
- Keyboard users can reach the source CTA, work products, outputs, generated
  artifacts, and quick notes in a coherent order.
- Docs point to `/research-studio` and mention legacy redirects only where
  needed.
- WebUI and extension route parity is verified.

Verification:

- Focused unit/component tests.
- Route-registry tests.
- Tutorial route tests.
- CDP screenshots for:
  - `/research-studio`
  - `/workspace-playground`
  - `/workspace-studio`
  - `/research-studio?tab=studio` on mobile
- Manual scan for visible "Workspace Playground" references that should now be
  "Research Studio."

## Dependency Order

The recommended implementation order is:

1. Stage 0.
2. Stage 3A.
3. Stage 1A.
4. Stage 2.
5. Stage 1B.
6. Stage 4.
7. Stage 5.
8. Stage 6.
9. Stage 3B when backend health semantics are ready.
10. Stage 7 as the release-closeout pass.

Reasoning:

- Degraded-health pass-through should land before browser-visible route alias
  rollout when local API health is commonly degraded. That keeps route
  verification from being hidden behind the readiness screen.
- Route compatibility should land before copy/docs references are changed.
- Mobile `?tab=studio` should land before docs promote it.
- Health pass-through can be shared with other WebUI surfaces if already
  underway elsewhere.
- Work-product-first IA and no-source empty states touch the same Studio area
  and should be coordinated, but they remain separable for review.
- Capability-aware health should not block the rest of the Research Studio UX
  remediation while its backend contract remains uncertain.

## Risk Register

### Risk: Route rename breaks existing links or tests

Mitigation:

- Keep `/workspace-playground` as an alias.
- Preserve query params.
- Add route tests before broad naming changes.

### Risk: Internal rename causes persisted data churn

Mitigation:

- Do not rename storage keys, export formats, telemetry keys, or event names in
  this series.
- Document compatibility names as internal implementation details.

### Risk: Planned work products disappear and users lose roadmap visibility

Mitigation:

- Hide planned products in the end-user Studio panel, but keep roadmap/status
  visibility in docs, changelog, or admin/dev surfaces if needed.

### Risk: Health pass-through lets users try broken generation

Mitigation:

- Stage 3A only changes app entry behavior.
- Stage 3B adds capability-aware gating once backend health semantics are known.
- Generation actions should still surface request-level failures clearly.

### Risk: Work-product-first hides familiar output types

Mitigation:

- Keep implemented output types as secondary actions.
- Track recent outputs for power users.
- Use tests to ensure existing output actions remain reachable after source
  readiness.

### Risk: Mobile URL and persisted tab state conflict

Mitigation:

- URL state wins over persisted last tab.
- Invalid URL tab values fall back to Chat.
- Keep write-back behavior conservative to avoid history churn.

## Testing Strategy

The implementation plans derived from this design should prefer focused,
stage-specific verification:

- Route and alias tests for Stage 1A.
- Header/nav/helper copy tests for Stage 1B.
- Route-state parser and mobile tab tests for Stage 2.
- Readiness-gate tests for Stage 3A.
- Capability derivation tests for Stage 3B.
- WorkProductTemplateChooser and StudioPane tests for Stages 4 and 5.
- Storage resilience and responsive tests for Stage 6.
- CDP/browser screenshots for Stage 7.

Full repo-wide frontend typecheck or smoke suites can be reserved for the final
release pass unless a stage touches shared route or readiness infrastructure
with broader risk.

## Documentation Updates

Documentation should converge on:

- Route: `/research-studio`
- Product name: Research Studio
- Mobile deep link: `/research-studio?tab=studio`
- Legacy compatibility: `/workspace-playground` and `/workspace-studio`
  redirect or alias to `/research-studio`

Docs that mention internal implementation paths may keep `WorkspacePlayground`
when referring to code, but should clarify that the user-facing product is
Research Studio.

Update source documentation first. Generated documentation outputs such as
`Docs/site` should be rebuilt through the normal docs pipeline only if that is
part of the stage's accepted verification scope.

## Acceptance Checklist For The Series

- `/research-studio` is the canonical user-facing WebUI and extension route.
- `/workspace-playground` and `/workspace-studio` do not 404.
- Query params are preserved across aliases.
- `?tab=studio` opens Studio on mobile.
- Visible labels and docs use "Research Studio."
- Planned work products are hidden until actionable.
- Studio defaults to work-product-first selection.
- No-source Studio explains source requirements before showing generation
  actions.
- Advanced output settings are progressively disclosed.
- Degraded-but-reachable health can render the app with warning visibility.
- Capability-aware health follow-up is scoped separately if backend semantics
  are not yet sufficient.
- Returning-user state reduces repeated mobile and onboarding friction.
- Accessibility and CDP verification cover desktop and mobile.
