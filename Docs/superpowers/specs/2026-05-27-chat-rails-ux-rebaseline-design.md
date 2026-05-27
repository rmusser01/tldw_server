# Chat Rails UX Rebaseline And Remediation Design

**Date:** 2026-05-27
**Surface:** WebUI `/chat` and directly connected browser-extension chat handoff
**Status:** Draft for review
**Backlog:** TASK-516

## Goal

Create a clean `origin/dev`-based branch for `/chat` remediation, verify that the main chat cockpit side rails are present and functional there, then redo the UX evaluation against that correct baseline before implementing fixes.

The first objective is not to redesign `/chat` from the stale audit branch. The first objective is to restore confidence in the main `dev` chat surface and make the remediation list reflect the rail-enabled product.

## Context

The previous `/chat` audit was performed from `codex/chat-sidebar-tools-first`, a branch that is far diverged from `origin/dev`. In that branch, the chat cockpit rail files were absent from the tracked checkout.

After fetching `origin/dev`, the clean baseline contains these rail components and tests:

- `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
- `apps/packages/ui/src/components/Option/Playground/PlaygroundRailSection.tsx`
- `apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx`
- `apps/packages/ui/src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx`

Therefore, "there are no siderails" was a branch-baseline problem, not confirmed product behavior on current `dev`.

## Scope

In scope:

- Create and work from a clean branch/worktree based on `origin/dev`.
- Verify the rail-enabled `/chat` page on desktop and narrow/mobile widths.
- Re-run the `/chat` UX evaluation with the rails present.
- Re-check directly connected extension chat entry and handoff paths only where they launch, capture context for, or hand off into chat.
- Convert the refreshed findings into a prioritized remediation plan.

Out of scope:

- Broad app redesign.
- Rewriting the browser extension outside chat capture/handoff.
- Replacing the existing cockpit architecture if it is functional.
- Implementing fixes before the refreshed rail-enabled audit is complete.

## Branch And Worktree Strategy

Use a clean worktree rooted at:

`/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`

Branch:

`codex/chat-rails-ux-rebaseline`

Base:

`origin/dev`

The current dirty checkout remains only as historical context. It must not be used as the implementation baseline for chat rail remediation.

## Rebaseline Workflow

1. Confirm branch provenance.
   - `git status --short --branch` should show the clean remediation branch tracking `origin/dev`.
   - Record `pwd`, `git branch --show-current`, `git rev-parse --short HEAD`, `git rev-parse --short origin/dev`, and `git merge-base --is-ancestor origin/dev HEAD`.
   - The expected first audit baseline is `HEAD == origin/dev`; if the branch has local spec-only commits, record the exact merge base and explain which commits are local planning changes.
   - `git ls-files` should show the cockpit rail files listed above.

2. Verify the rail-enabled `/chat` UI.
   - Start the WebUI from the clean worktree.
   - Open `/chat`.
   - Confirm the desktop layout exposes the cockpit shell, context rail, runtime inspector, status strip, focus toggle, and character rail where expected.
   - Confirm focus mode hides rails and returns them without losing chat state.

3. Verify responsive behavior.
   - At approximately `390px` width, `/chat` must not require horizontal page scrolling.
   - Rail content should collapse into drawers, collapsibles, or a deliberate cockpit reveal pattern.
   - Composer, transcript, and header controls must remain reachable.

4. Redo the UX evaluation.
   - Evaluate first-time and power-user journeys separately.
   - Record observed UI behavior first; use code evidence only where live observation is unavailable.
   - Reclassify each previous finding as fixed, still reproduces, changed by rail baseline, or not applicable.
   - Write the refreshed audit to `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`.
   - The audit must include, for each observation: route, viewport, user journey, observed behavior, evidence source, prior-finding classification, severity, confidence, and whether it is eligible for the first implementation plan.

5. Re-check extension handoff.
   - Verify sidepanel chat starters, capture handoff, media ingest handoff, and "open full-screen" behavior.
   - Confirm whether full-screen opens the rail-enabled chat route with state/context preserved.

6. Produce a remediation plan.
   - Split fixes into small, reviewable slices.
   - Each implementation slice must have its own Backlog task before file edits.
   - Do not merge stale audit assumptions into implementation without current rail-enabled evidence.

## Prior Findings Reconciliation

The refreshed audit should explicitly revisit the earlier findings:

| Previous Finding | Rebaseline Decision Needed |
| --- | --- |
| Mobile `/chat` horizontal overflow | Verify on rail-enabled `origin/dev`; this may be solved or altered by cockpit/focus mode. |
| Initial connection/setup modal | Re-test against current readiness behavior and degraded-health pass-through. |
| Too many first-run controls | Re-evaluate with cockpit/focus rails present; determine whether progressive disclosure is already improved. |
| Dense settings modal | Re-test; likely still relevant but may be less central if rails expose common controls. |
| Prompt picker empty state | Re-test. |
| Compare disabled without reason | Re-test. |
| Character/persona timeline ambiguity | Re-test on current tracked identity and character rail implementation. |
| Search & Context preview opacity | Re-test with context rail. |
| Extension full-screen/dashboard handoff | Re-test; likely still requires targeted verification. |
| Duplicate accessible sidebar labels | Re-test with current app shell and chat shell. |

## Success Criteria

- The clean branch is created from `origin/dev`.
- The rail components are verified as present in the clean checkout.
- A live `/chat` audit is completed against the rail-enabled page.
- Findings are separated into fixed-by-baseline, still-reproducing, and new rail-specific issues.
- The first implementation plan after this spec targets only verified current issues.
- No code implementation starts from the stale `codex/chat-sidebar-tools-first` branch.

## Verification Plan

Planning/spec verification:

- `git diff --check`
- Backlog task references this spec.
- Bandit skipped for this design-only Markdown task.

Rail-enabled audit verification:

- Focused rail component tests where available.
- Browser verification on desktop and narrow/mobile viewports.
- Screenshot artifacts are required for desktop cockpit, desktop focus, mobile focus, mobile cockpit or rail drawer, and extension sidepanel/debug sidepanel.
- Store screenshot artifacts under `Docs/Reviews/artifacts/chat-rails-ux-rebaseline-2026-05-27/` unless the implementation plan chooses a more appropriate existing artifact directory.
- If backend is reachable, use real server data instead of mocked route payloads.
- If backend is unavailable, record that limitation and cite source files for fallback evidence.
- If screenshots are impossible because live browser access is unavailable, the audit must include DOM/accessibility snapshots plus the explicit reason screenshots could not be captured.

## Risks

- The rail-enabled `origin/dev` page may still have regressions hidden by previous branch confusion.
- Local dirty worktrees can make provenance unclear; commands must report working directory and branch when evidence is collected.
- Some previous UX issues may disappear on the correct baseline; implementation must not preserve outdated findings for momentum.
- Extension sidepanel and main WebUI chat may share code but differ in shell constraints, so parity must be verified rather than assumed.

## Open Questions

- Should implementation prioritize rail restoration/regression tests first, or user-facing UX fixes first if the rails already pass?
- Should full-screen extension handoff target `/chat`, an options hash route, or a dedicated chat-resume URL contract?
