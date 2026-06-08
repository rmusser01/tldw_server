# Mermaid Chat Card Browser QA Harness Design

## Problem

The Mermaid chat/card feature is implemented and merged, but browser-level QA is not stable enough. The previous closeout verified focused unit tests and frontend compile, but full Browser verification was blocked by readiness gates, first-run setup, and Browser policy rejecting a temporary `file://` harness.

This leaves a gap: future changes to chat Markdown, Mermaid rendering, artifact cards, or readiness gates can regress real browser rendering without a deterministic route that exercises the relevant UI states.

## Goal

Add a stable browser QA harness for assistant-facing Mermaid chat/card rendering. The harness should run inside the Next WebUI, avoid backend readiness and first-run dependencies, and expose fixed test cases for Playwright and manual Browser inspection.

## Non-Goals

- Do not add a user-visible product page.
- Do not change Mermaid rendering behavior.
- Do not render Mermaid in user messages.
- Do not add Graphviz/DOT rendering.
- Do not add PNG export, server-side rendering, or Mermaid prompt templates.
- Do not depend on a running backend server.

## Route

Add `apps/tldw-frontend/pages/__debug__/mermaid-chat-cards.tsx`.

The route should:

- be client-rendered with `ssr: false`, matching existing debug route style;
- use real shared UI components rather than a parallel renderer;
- render static fixtures only;
- expose stable `data-testid` markers for Playwright;
- be visually simple and utilitarian.

The page is debug-only by path. It should not be added to normal navigation.

## Route Governance

Add the route to the existing route metadata registry:

- `apps/packages/ui/src/routes/route-metadata.ts`

The route metadata entry should use `surface: "internal_qa_debug"` and `smoke: "exclude"` so broad route governance understands that this is an intentional debug page, not a product route.

Do not add the new route to `apps/tldw-frontend/e2e/smoke/page-inventory.ts`. Route-contract coverage expects smoke-excluded metadata routes to stay out of the broad page inventory. This route has its own focused Playwright smoke test instead.

## Gate Bypass

`pages/_app.tsx` should treat `routePath` values under `/__debug__` as debug routes that bypass `ServerReadinessGate` and `FirstRunGate`.

Rationale:

- existing smoke tests already rely on `/__debug__/sidepanel-chat` as a reduced-noise surface;
- Mermaid browser QA should not require a backend or first-run local storage;
- bypassing only `/__debug__` paths keeps production routes unchanged.

The bypass should be explicit, for example:

```ts
const isDebugRoute = routePath.startsWith("/__debug__")
const shouldBypassGates =
  isPublicAuthRoute || isSettingsRoute || isSetupRoute || isDebugRoute
```

## Fixtures

The route should include these sections:

1. Assistant Mermaid render
   - Uses `Markdown` with `enableMermaidDiagrams`.
   - Contains a closed `mermaid` fence.
   - Expected result: inline Mermaid diagram block renders.

2. User-message unchanged
   - Renders the same kind of Mermaid fence as plain text in a user-message styled block.
   - Expected result: source text remains visible and no Mermaid diagram block is created inside this section.

3. Setting-off fallback
   - Uses `Markdown` without `enableMermaidDiagrams`.
   - Expected result: Mermaid source renders as a code block.

4. Invalid Mermaid fallback
   - Uses `Markdown` with `enableMermaidDiagrams`.
   - Contains invalid Mermaid source.
   - Expected result: the existing local fallback text `Unable to render Mermaid diagram.` is visible, the raw source remains visible, and the rest of the page remains usable.

5. Graphviz/DOT fallback
   - Uses a `dot` or `graphviz` fenced code block.
   - Expected result: source remains a code block and is not treated as Mermaid.

6. Artifact-style Mermaid card
   - Uses `MermaidDiagramBlock` directly, matching artifact-card usage without depending on chat state.
   - Expected result: diagram block renders and exposes preview/copy controls.

## Playwright Coverage

Add `apps/tldw-frontend/e2e/smoke/mermaid-chat-cards.spec.ts`.

The smoke test should:

- navigate to `/__debug__/mermaid-chat-cards`;
- assert the harness root is visible;
- assert `server-readiness-recovery` and `first-run-gate-overlay` are absent, proving the route is not blocked by the same gates that blocked the previous browser closeout;
- assert assistant Mermaid produces a Mermaid diagram block;
- assert user-message fixture retains raw Mermaid source and has no nested Mermaid diagram block;
- assert setting-off fixture renders a code block/source fallback;
- assert invalid Mermaid fixture shows `Unable to render Mermaid diagram.` plus source text without breaking the page;
- assert Graphviz/DOT fixture remains code;
- assert artifact-style fixture renders a Mermaid diagram block and action controls.

The route should wrap each fixture in a stable section test id, such as `mermaid-harness-assistant`, `mermaid-harness-user`, `mermaid-harness-disabled`, `mermaid-harness-invalid`, `mermaid-harness-graphviz`, and `mermaid-harness-artifact`. Tests should query inside those sections and use roles/button labels for real `MermaidDiagramBlock` controls instead of depending on generated SVG internals. Do not add component-level test ids to `MermaidDiagramBlock` unless the harness cannot otherwise assert the real behavior.

## Error Handling

The route should be resilient if Mermaid fails to load or render:

- the page still loads;
- invalid Mermaid failure remains local to that fixture;
- raw source remains inspectable;
- Playwright assertions should allow the renderer's existing async/fallback behavior instead of depending on exact generated SVG internals.

## Accessibility

The harness is not a product surface, but the rendered components should still preserve the accessibility behavior of the real components:

- Mermaid diagram blocks should retain their existing accessible diagram labeling;
- buttons should keep labels or tooltips already provided by the shared components;
- the route should use headings to separate fixtures.

## Acceptance Criteria

- `/__debug__/mermaid-chat-cards` loads without backend readiness or first-run setup.
- The route does not show `server-readiness-recovery` or `first-run-gate-overlay`.
- The route uses real `Markdown` and `MermaidDiagramBlock` components.
- Assistant Mermaid, user-message unchanged, setting-off fallback, invalid fallback, Graphviz/DOT fallback, and artifact-style block are all represented.
- Playwright smoke coverage verifies the fixture states with stable selectors.
- Route metadata classifies the page as internal QA/debug with `smoke: "exclude"`, and the page is not added to the broad smoke page inventory.
- Existing focused Mermaid unit tests still pass.
- Frontend compile passes.
- No app source behavior changes outside debug-route gating and QA harness code.

## Verification

Expected verification commands:

```bash
cd apps
bun install --frozen-lockfile

cd packages/ui
bunx vitest run \
  src/components/Common/__tests__/Mermaid.test.tsx \
  src/components/Common/__tests__/MermaidDiagramBlock.test.tsx \
  src/components/Common/__tests__/MermaidPreviewDialog.test.tsx \
  src/components/Common/__tests__/Markdown.mermaid.test.tsx \
  src/components/Common/__tests__/Markdown.github-code-blocks.test.tsx \
  src/components/Common/__tests__/CodeBlock.artifacts.test.tsx \
  src/components/Common/Playground/__tests__/Message.mermaid-rendering.test.tsx \
  src/components/Common/QuickChatHelper/__tests__/QuickChatMessage.mermaid.test.tsx \
  src/components/Sidepanel/Chat/__tests__/ArtifactsPanel.mermaid.test.tsx \
  src/components/Option/Settings/__tests__/ChatSettings.test.tsx

cd ../../tldw-frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run compile
npx playwright test e2e/smoke/mermaid-chat-cards.spec.ts
```

Bandit is not applicable to this frontend-only change, but the final task record should state that explicitly.
