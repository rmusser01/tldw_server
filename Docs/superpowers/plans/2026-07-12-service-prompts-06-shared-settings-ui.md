# Shared Service Prompt Settings UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let users browse, preview, edit, review, reset, and restore eligible service prompts from one responsive settings experience shared by the Next.js WebUI and browser extension.

**Architecture:** Add a small typed API service and one shared settings component using existing request/auth and Ant Design patterns. Register the same route in shared options, extension options, and Next.js. OpenAPI controls generic route visibility; the page fetches authenticated capability and lifecycle data. The sidepanel remains compact and links into the full editor.

**Tech Stack:** React 19/TypeScript, Ant Design, existing `tldwRequest`, settings route registry/navigation, Vitest/Testing Library, Playwright.

---

Before every commit below, satisfy the umbrella plan's mandatory per-commit gate.

## Product UI constraints

- This is a calm expert/power-user workbench used during long research or operator sessions, sometimes in the compact extension. It inherits the user's existing light/dark theme rather than forcing one.
- Use the existing restrained neutral palette and semantic state tokens. Blue identifies primary action/current selection; warnings/errors use established state components, not decorative color.
- Prefer an aligned list/detail work surface with dividers over nested cards. Use skeletons that preserve layout, concise recovery panels, and standard controls.
- Keep explanatory prose to 65–75 characters per line, prompt content in the existing mono/editor treatment, and ordinary labels in the product UI font.
- Use 150–250 ms state transitions only where existing components already do; respect reduced motion and add no decorative choreography.
- Routine approval/reset confirmation is inline progressive disclosure, not a modal-first flow.

## Task 1: Add typed client and capability detection

**Files:**

- Create: `apps/packages/ui/src/services/service-prompts.ts`
- Create: `apps/packages/ui/src/services/__tests__/service-prompts.test.ts`
- Modify: `apps/packages/ui/src/services/tldw/server-capabilities.ts`
- Modify: `apps/packages/ui/src/services/__tests__/server-capabilities.test.ts`

- [ ] Create the implementation Backlog task and link this plan.
- [ ] Write failing tests for `/capability`, `/catalog`, detail, preview, override, reset, acknowledge-default, history, revision, restore, and Context Integrity review/approve/reject; cover auth-aware requests, URL encoding, abort signals, typed error mapping, cursor pagination, ETags, mutation IDs, and no local persistence of prompt content.
- [ ] Implement a focused client over `tldwRequest`; do not add methods to the already-broad `TldwApiClient` and do not add a state library.
- [ ] Add `hasServicePrompts` derived from `/api/v1/service-prompts/capability` presence in OpenAPI, including fallback spec/cache version updates.
- [ ] Keep `mode`, `availability`, and `can_approve_pending` out of the generic capability cache; fetch them through the authenticated service.
- [ ] Run `bunx vitest run ../packages/ui/src/services/__tests__/service-prompts.test.ts ../packages/ui/src/services/__tests__/server-capabilities.test.ts` from `apps/tldw-frontend`.
- [ ] Commit: `feat: add service prompt settings client (<task-id>)`.

## Task 2: Build the catalog/detail editor

**Files:**

- Create: `apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx`
- Modify: `apps/packages/ui/src/assets/locale/en/settings.json`
- Modify: `apps/tldw-frontend/public/_locales/en/settings.json`

- [ ] Write failing component tests for disconnected, unsupported, unavailable, quarantined, incompatible, read-only, and bypass banners; empty/error/retry states; search across labels/descriptions/tags/workflows; category and Customized/Server default/Upstream changed/Needs attention filters; keyboard selection; multipart editing; locked/hidden parts; dirty navigation; and responsive single-column behavior.
- [ ] Build a bordered/divided two-pane desktop work surface and explicit list→detail mobile flow using existing settings tokens/components; avoid nested cards. Show title, description, category, experimental label, active source, pending status, and last update without exposing implementation module names.
- [ ] Render one labeled lightweight textarea per visible editable part, required/optional variable chips with cursor insertion, read-only presentation for locked visible parts, and only a presence/change/digest marker for hidden parts.
- [ ] Keep unsaved content in component memory only. Preserve it through connection failures, provide Retry and Copy Draft, use native close protection plus an in-app dirty warning before route/definition changes, and restore focus to the originating list item on mobile back navigation.
- [ ] Use layout-preserving skeletons, existing recovery/state panels, an accessible error summary that focuses the first invalid part, explicit text/status icons in addition to color, accessible labels/descriptions, visible focus, near-44px touch targets, and live regions for save/preview results.
- [ ] Use existing semantic tokens for both themes and existing form/button vocabulary for default, hover, focus, active, disabled, loading, and error states; add no raw colors or one-off alert styles.
- [ ] Rerun the component test and commit: `feat: build service prompt settings editor (<task-id>)`.

## Task 3: Add preview, save, approval, reset, and history flows

**Files:**

- Modify: `apps/packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx`
- Modify: `apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/ServicePromptDiff.tsx`
- Create: `apps/packages/ui/src/components/Option/Settings/__tests__/ServicePromptDiff.test.tsx`

- [ ] Write failing tests for preview/assembly order, validation errors, stale-generation conflicts, pending-save/supersession messaging, mutation retry, approve/reject review links, escaped canonical diff with unchanged visible regions collapsed, explicit single-user approval, reset confirmation, upstream Compare/Keep override/Use server default actions, default-digest acknowledgement, cursor history pagination, non-restorable state events, revision inspection, and restore-as-pending.
- [ ] Preview through the API only; label it “rendered prompt preview” and never imply model execution.
- [ ] Save to `PUT /{definition_id}/override` with the current `expected_generation` and a stable client mutation ID retained only for the in-flight/retry attempt; on 409 preserve the draft and offer reload/diff rather than overwriting.
- [ ] Show approve/reject controls only when `can_approve_pending`; load the Context Integrity review endpoint, render escaped plain-text diffs only, require inline confirmation, and clearly state that approval changes future runs while already queued jobs keep their authenticated pin.
- [ ] Reset through inline destructive confirmation naming the definition; keep cancel/reset actions adjacent and return focus after either choice. History remains available after reset.
- [ ] Restore creates a new pending revision and then refreshes detail/history; never label an imported/old revision active until approval succeeds.
- [ ] Reuse `DiffLine`, `computeDiffSync`, `computeDiffWithWorker`, worker/sampling guards, and sampling helpers from `apps/packages/ui/src/components/Media/diff-worker-client.ts`; add a small settings-specific inline renderer rather than reusing modal-first `DiffViewModal` or watchlist-specific `FlowCheckDiffPanel`. Show unified escaped plain text with unchanged regions collapsed, part-level markers, and an optional side-by-side desktop view. Hidden parts show safe changed/unchanged markers and digests only. Do not add a diff dependency.
- [ ] Rerun the component tests and commit: `feat: manage service prompt revision lifecycle (<task-id>)`.

## Task 4: Register shared WebUI and extension routes

**Files:**

- Modify: `apps/packages/ui/src/routes/option-settings-route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-registry.tsx`
- Modify: `apps/tldw-frontend/extension/routes/route-registry.tsx`
- Modify: `apps/packages/ui/src/routes/route-capabilities.ts`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav-config.ts`
- Modify: `apps/packages/ui/src/components/Layouts/settings-nav.ts`
- Create: `apps/tldw-frontend/pages/settings/service-prompts.tsx`
- Create: `apps/packages/ui/src/routes/__tests__/service-prompts-route.test.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/route-capabilities.test.ts`

- [ ] Write failing route tests for `/settings/service-prompts` parity across shared options, extension options, and Next.js, plus capability-hidden behavior when OpenAPI lacks the route.
- [ ] Register one lazy shared component in all three registries using existing `createSettingsRoute`/dynamic import patterns.
- [ ] Add “Service prompts” under `aiModels` at order 2.5. Keep `/settings/prompt` unchanged under the knowledge workspace and explain the distinction in page copy.
- [ ] Mark the route experimental on the page. Add a time-bounded navigation beta badge following `SETTINGS_BETA_BADGE_WINDOWS`; do not make authorization depend on the badge.
- [ ] Rerun route/navigation tests and commit: `feat: register service prompt settings routes (<task-id>)`.

## Task 5: Add the sidepanel deep link

**Files:**

- Modify: `apps/packages/ui/src/components/Sidepanel/Settings/body.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Settings/__tests__/service-prompts-link.test.tsx`

- [ ] Write failing tests that the link appears only with `hasServicePrompts`, opens `options.html#/settings/service-prompts` in extension context, and uses `/settings/service-prompts` in WebUI-compatible context.
- [ ] Add one compact settings row/link using the sidepanel's existing section vocabulary; do not add a standalone nested card, embed the editor, or duplicate lifecycle state in the sidepanel.
- [ ] Reuse the existing browser runtime/tabs fallback pattern and include `noopener,noreferrer` for window fallback.
- [ ] Rerun the sidepanel test and commit: `feat: link sidepanel to service prompt settings (<task-id>)`.

## Task 6: Add end-to-end coverage

**Files:**

- Create: `apps/tldw-frontend/e2e/workflows/service-prompts-settings.spec.ts`
- Modify: `apps/tldw-frontend/e2e/smoke/all-pages.spec.ts` only if routes are enumerated manually

- [ ] Add API fixtures for enabled, read-only, unavailable, pending, and stale-conflict states without calling an external LLM.
- [ ] Cover catalog search/filter, variable insertion, editing two parts, preview, save-to-pending, explicit approval, active provenance, history restore, upstream compare/keep/reset, Copy Draft after connection failure, mobile layout, keyboard navigation, and extension-style hash routing.
- [ ] Assert hidden content never appears in DOM, screenshots, or request fixtures. Verify both themes, reduced motion, 200% text zoom, no page-level horizontal overflow at 390px/extension width, and keyboard-only completion of save/review/reset.
- [ ] After the inventory-generated canary domain plan is implemented, add a real-server workflow using the existing mock LLM/provider harness: edit and approve that exact domain in WebUI, verify the affected workflow receives it, edit the same account through the extension-options hash route, and verify WebUI observes the new pending revision. Keep state-only UI fixtures for unavailable/error permutations; do not migrate domain code in this UI plan.
- [ ] Run `bunx playwright test e2e/workflows/service-prompts-settings.spec.ts --reporter=line`.
- [ ] Commit: `test: cover service prompt settings workflow (<task-id>)`.

## Task 7: Frontend verification

- [ ] From `apps/tldw-frontend`, run `bunx vitest run ../packages/ui/src/services/__tests__/service-prompts.test.ts ../packages/ui/src/services/__tests__/server-capabilities.test.ts ../packages/ui/src/components/Option/Settings/__tests__/ServicePromptsSettings.test.tsx ../packages/ui/src/components/Option/Settings/__tests__/ServicePromptDiff.test.tsx ../packages/ui/src/routes/__tests__/service-prompts-route.test.tsx ../packages/ui/src/routes/__tests__/route-capabilities.test.ts ../packages/ui/src/components/Sidepanel/Settings/__tests__/service-prompts-link.test.tsx`.
- [ ] Run `bunx eslint ../packages/ui/src/services/service-prompts.ts ../packages/ui/src/components/Option/Settings/ServicePromptsSettings.tsx ../packages/ui/src/components/Option/Settings/ServicePromptDiff.tsx ../packages/ui/src/routes/option-settings-route-registry.tsx ../packages/ui/src/routes/route-capabilities.ts ../packages/ui/src/components/Layouts/settings-nav-config.ts ../packages/ui/src/components/Layouts/settings-nav.ts extension/routes/route-registry.tsx pages/settings/service-prompts.tsx`.
- [ ] Run the Playwright command from task 6 and `git diff --check`.
- [ ] Update the Backlog task with screenshots or accessibility notes, focused test output, touched files, and final summary.
- [ ] Commit: `chore: verify shared service prompt settings UI (<task-id>)`.

Bandit is not required for this TypeScript-only plan. Backend API verification remains mandatory before E2E sign-off.
