# Chat Rails UX Rebaseline Audit - 2026-05-27

## Baseline

- Worktree: `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`
- Branch: `codex/chat-rails-ux-rebaseline`
- Pre-artifact baseline/provenance capture commit: `69a80b4b5` (`git rev-parse --short HEAD` output captured before the audit artifact commit).
- origin/dev: `efe42fe0c`
- Merge-base expectation: `git merge-base --is-ancestor origin/dev HEAD` produced no stdout and exited `0` during the pre-artifact baseline capture.
- Backend: Not captured for Task 1.
- WebUI URL: Not captured for Task 1.
- Rail source files:

```text
apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundRailSection.tsx
apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx
apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx
```

## Required Evidence

- Desktop cockpit screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-cockpit.png`
- Desktop focus screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-focus.png`
- Mobile focus screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png`
- Mobile cockpit screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png`
- Extension sidepanel screenshot: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png`
- Evidence JSON: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json`

## Prior Finding Reclassification

| ID | Prior finding | Current route/viewport | Classification | Evidence | Severity | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- |
| C1 | Mobile `/chat` horizontal overflow | | | | | |
| C2 | First-run connection/setup feedback | | | | | |
| C3 | First-run control overload | | | | | |
| C4 | Dense settings modal | | | | | |
| C5 | Prompt picker empty state | | | | | |
| C6 | Compare disabled without reason | | | | | |
| C7 | Character/persona timeline ambiguity | | | | | |
| C8 | Search & Context preview opacity | | | | | |
| C9 | Extension full-screen/dashboard handoff | | | | | |
| C10 | Duplicate accessible sidebar labels | | | | | |

## Refreshed Findings

| ID | Severity | Journey | Route | Viewport | Evidence | UX issue | User impact | Recommended solution | Effort | Confidence | First-plan eligible |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |

## Notes

- Observed behavior:
- Limitations:
- Non-goals:
