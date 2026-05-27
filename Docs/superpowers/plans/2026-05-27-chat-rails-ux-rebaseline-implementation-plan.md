# Chat Rails UX Rebaseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Verify the `/chat` cockpit rails from the clean `origin/dev` baseline, lock in rail regression coverage first, route extension full-screen handoff to `/chat`, and produce a refreshed rail-enabled UX audit before user-facing UX fixes.

**Architecture:** Keep the existing `Playground` cockpit architecture and sidepanel chat shell. Add only targeted regression guards and documentation artifacts unless current verification proves a rail or `/chat` handoff regression. Use the existing real-server Playwright cockpit spec as the browser verification anchor and the existing rail component tests as the unit/integration anchor.

**Tech Stack:** React, Next.js, WXT extension shell, Vitest, Testing Library, Playwright, Backlog.md.

---

## File Map

- Modify: `backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md`
  - Tracks planning, verification, and final handoff notes for the design/planning slice.
- Create: `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`
  - Refreshed audit report for the rail-enabled `/chat` baseline.
- Create: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/`
  - Screenshot and JSON evidence directory. Use this existing `Docs/Reviews/assets` convention instead of the spec's fallback `Docs/Reviews/artifacts` path.
- Modify or create: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts`
  - Source-level guard that fails if `Playground.tsx` stops rendering/importing the cockpit shell, rails, runtime inspector, and character rail.
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
  - Add explicit provenance/no-horizontal-overflow assertions and ensure screenshots cover required audit states.
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/SidepanelHeaderSimple.tsx`
  - Change full-screen handoff from `/options.html#/` to `/options.html#/chat`.
- Create or modify: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx`
  - Verifies full-screen button opens `/chat` through the extension options URL.

## Backlog Ownership

- `TASK-516` covers the design, implementation plan, refreshed audit document, evidence artifacts, and final handoff notes.
- `TASK-517` covers code-changing rail regression work in Tasks 2 and 3.
- `TASK-518` covers code-changing sidepanel full-screen handoff work in Task 4.
- Before editing files in Tasks 2, 3, or 4, set the relevant task to `In Progress`, record touched files, and keep verification results in that task. `TASK-516` remains the umbrella for the rebaseline/audit artifacts.

## Task 1: Record Baseline Provenance And Audit Artifact Skeleton

**Files:**
- Create: `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`
- Create: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/README.md`
- Modify: `backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md`

- [ ] **Step 1: Capture exact baseline commands**

Run from the clean worktree:

```bash
pwd
git branch --show-current
git rev-parse --short HEAD
git rev-parse --short origin/dev
git merge-base --is-ancestor origin/dev HEAD
git ls-files apps/packages/ui/src/components/Option/Playground | rg 'Cockpit|Rail|Inspector|CharacterControl'
```

Expected:

- `pwd` is `/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline`.
- Branch is `codex/chat-rails-ux-rebaseline`.
- `HEAD` is either equal to `origin/dev` or ahead only by planning commits.
- `git merge-base --is-ancestor origin/dev HEAD` exits `0`.
- Rail files are listed.

- [ ] **Step 2: Write the audit skeleton**

Create `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`:

```markdown
# Chat Rails UX Rebaseline Audit - 2026-05-27

## Baseline

- Worktree:
- Branch:
- HEAD:
- origin/dev:
- Merge-base expectation:
- Backend:
- WebUI URL:

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
```

- [ ] **Step 3: Add artifact directory README**

Create `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/README.md`:

```markdown
# Chat Rails UX Rebaseline Evidence

This directory contains screenshots and browser evidence captured from the clean `origin/dev`-based `/chat` rail rebaseline.

Required files:

- `desktop-cockpit.png`
- `desktop-focus.png`
- `mobile-focus.png`
- `mobile-cockpit.png`
- `extension-sidepanel.png`
- `evidence.json`
```

- [ ] **Step 4: Update Backlog notes**

Add an implementation note to `TASK-516` with the baseline command outputs and the audit artifact path.

- [ ] **Step 5: Commit**

```bash
git add Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/README.md \
  "backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md"
git commit -m "docs(chat): start rails UX rebaseline audit"
```

## Task 2: Add Rail Restoration Regression Guards First

**Files:**
- Create or modify: `apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts`
- Modify task: `backlog/tasks/task-517 - Add-chat-cockpit-rail-regression-guards.md`
- Existing references:
  - `apps/packages/ui/src/components/Option/Playground/Playground.tsx`
  - `apps/packages/ui/src/components/Option/Playground/PlaygroundCockpitShell.tsx`
  - `apps/packages/ui/src/components/Option/Playground/PlaygroundContextRail.tsx`
  - `apps/packages/ui/src/components/Option/Playground/PlaygroundRuntimeInspector.tsx`
  - `apps/packages/ui/src/components/Option/Playground/CharacterControlRail.tsx`

- [ ] **Step 1: Write the failing source guard**

Create `Playground.cockpit-regression.guard.test.ts`:

```ts
import { readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { describe, expect, it } from "vitest"

const testDir = path.dirname(fileURLToPath(import.meta.url))
const playgroundPath = path.resolve(testDir, "../Playground.tsx")
const cockpitShellPath = path.resolve(testDir, "../PlaygroundCockpitShell.tsx")

describe("Playground cockpit regression guard", () => {
  it("keeps the main /chat cockpit shell and rails wired into Playground", () => {
    const source = readFileSync(playgroundPath, "utf8")

    expect(source).toContain("PlaygroundCockpitShell")
    expect(source).toContain("PlaygroundContextRail")
    expect(source).toContain("PlaygroundRuntimeInspector")
    expect(source).toContain("CharacterControlRail")
    expect(source).toContain("<PlaygroundCockpitShell")
    expect(source).toContain("<PlaygroundContextRail")
    expect(source).toContain("<PlaygroundRuntimeInspector")
    expect(source).toContain("<CharacterControlRail")
  })

  it("keeps cockpit shell test ids and mobile rail state available", () => {
    const source = readFileSync(cockpitShellPath, "utf8")

    expect(source).toContain("playground-cockpit-shell")
    expect(source).toContain("playground-cockpit-left-rail")
    expect(source).toContain("playground-cockpit-right-rail")
    expect(source).toContain("playground-cockpit-mobile-rails")
    expect(source).toContain("Enter focus chat")
    expect(source).toContain("Show cockpit panels")
  })

  it("keeps focus mode as a reversible state rather than a separate route", () => {
    const source = readFileSync(playgroundPath, "utf8")

    expect(source).toContain("playgroundChatLayoutMode")
    expect(source).toContain("\"cockpit\"")
    expect(source).toContain("\"focus\"")
  })
})
```

Expected initially: if the current code already has these strings, the test may pass immediately. That is acceptable because this is a restoration guard against future regressions.

- [ ] **Step 2: Run the new guard**

```bash
cd apps/packages/ui
bunx vitest run src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts
```

Expected: pass on current `origin/dev`; fail if the rails are removed from `Playground.tsx`.

- [ ] **Step 3: Run the existing rail unit/integration set**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx \
  src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx \
  src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx
```

Expected: all focused rail tests pass. Document unrelated baseline failures separately if they occur.

- [ ] **Step 4: Commit**

```bash
git add apps/packages/ui/src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts \
  "backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md" \
  "backlog/tasks/task-517 - Add-chat-cockpit-rail-regression-guards.md"
git commit -m "test(chat): guard cockpit rail wiring"
```

## Task 3: Harden Real-Server Rail Verification And Evidence

**Files:**
- Modify: `apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts`
- Modify: `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`
- Modify task: `backlog/tasks/task-517 - Add-chat-cockpit-rail-regression-guards.md`
- Add generated evidence files under: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/`

- [ ] **Step 1: Add provenance and overflow helpers to the Playwright spec**

Add helpers near the other helpers in `chat-cockpit.real-server.spec.ts`:

```ts
const assertNoHorizontalOverflow = async (page: Page) => {
  const metrics = await page.evaluate(() => ({
    innerWidth: window.innerWidth,
    docScrollWidth: document.documentElement.scrollWidth,
    bodyScrollWidth: document.body.scrollWidth,
  }));

  expect(metrics.docScrollWidth).toBeLessThanOrEqual(metrics.innerWidth + 1);
  expect(metrics.bodyScrollWidth).toBeLessThanOrEqual(metrics.innerWidth + 1);
};
```

- [ ] **Step 2: Use the helper in desktop and mobile rail tests**

In the existing desktop cockpit test after initial render:

```ts
await assertNoHorizontalOverflow(page);
```

In the existing mobile cockpit test after `/chat` loads, after cockpit panels open, and after returning to focus:

```ts
await assertNoHorizontalOverflow(page);
```

- [ ] **Step 3: Ensure required screenshot states exist**

Map the existing screenshots to the audit:

- `chat-cockpit-desktop-initial.png` -> `desktop-cockpit.png`
- `chat-cockpit-desktop-focus.png` -> `desktop-focus.png`
- `chat-cockpit-mobile-focus.png` -> `mobile-focus.png`
- `chat-cockpit-mobile-context.png` or `chat-cockpit-mobile-runtime.png` -> `mobile-cockpit.png`

After the Playwright run, copy or save these into `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/`.

- [ ] **Step 4: Check backend health before browser verification**

```bash
curl -sf http://127.0.0.1:8000/api/v1/health
```

Expected: JSON health response. Record the exact `status`, `auth_mode`, and degraded checks in the audit. If this command fails, do not invent browser results; record backend unavailable and use source/file evidence as fallback.

- [ ] **Step 5: Run the real-server cockpit spec**

Use the live backend and no route stubbing:

```bash
cd apps/tldw-frontend
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
TLDW_SERVER_URL=http://127.0.0.1:8000 \
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
TLDW_WEB_URL=http://localhost:18014 \
TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18014' \
bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line
```

Expected: existing rail/cockpit tests pass. If the live backend is unavailable or degraded, record the limitation and the exact health result in the audit.

- [ ] **Step 6: Copy required screenshots into review artifacts**

From the worktree root after the Playwright run:

```bash
mkdir -p Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline
cp "$(find apps/tldw-frontend/test-results -name 'chat-cockpit-desktop-initial.png' -print -quit)" \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-cockpit.png
cp "$(find apps/tldw-frontend/test-results -name 'chat-cockpit-desktop-focus.png' -print -quit)" \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-focus.png
cp "$(find apps/tldw-frontend/test-results -name 'chat-cockpit-mobile-focus.png' -print -quit)" \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png
cp "$(find apps/tldw-frontend/test-results -name 'chat-cockpit-mobile-context.png' -print -quit)" \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png
```

Expected: all four files exist in `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/`. If Playwright stores a different mobile cockpit state, copy the runtime screenshot instead and note the source filename in the audit.

- [ ] **Step 7: Write `evidence.json`**

Create `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/evidence.json`:

```json
{
  "worktree": "/Users/macbook-dev/Documents/GitHub/tldw_server2/.worktrees/chat-rails-ux-rebaseline",
  "branch": "codex/chat-rails-ux-rebaseline",
  "head": "",
  "originDev": "",
  "backend": "http://127.0.0.1:8000",
  "webui": "http://localhost:18014",
  "viewportChecks": [
    {
      "route": "/chat",
      "viewport": "1440x960",
      "state": "desktop-cockpit",
      "horizontalOverflow": false
    },
    {
      "route": "/chat",
      "viewport": "390x844",
      "state": "mobile-focus",
      "horizontalOverflow": false
    },
    {
      "route": "/chat",
      "viewport": "390x844",
      "state": "mobile-cockpit",
      "horizontalOverflow": false
    }
  ]
}
```

Fill `head` and `originDev` with actual command output.

- [ ] **Step 8: Update the audit report**

Fill `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md` with:

- baseline command output;
- screenshot paths;
- Playwright command/result;
- first pass classification for C1 mobile overflow and cockpit rail presence.

- [ ] **Step 9: Commit**

```bash
git add apps/tldw-frontend/e2e/workflows/chat-cockpit.real-server.spec.ts \
  Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline \
  "backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md" \
  "backlog/tasks/task-517 - Add-chat-cockpit-rail-regression-guards.md"
git commit -m "test(chat): verify rails against real server"
```

## Task 4: Route Extension Full-Screen Handoff To `/chat`

**Files:**
- Modify: `apps/packages/ui/src/components/Sidepanel/Chat/SidepanelHeaderSimple.tsx`
- Create: `apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx`
- Modify task: `backlog/tasks/task-518 - Route-sidepanel-full-screen-chat-handoff-to-chat.md`
- Optional reference: `apps/packages/ui/src/routes/route-paths.ts`

- [ ] **Step 1: Write the failing test**

Create `SidepanelHeaderSimple.fullscreen-route.test.tsx`:

```tsx
// @vitest-environment jsdom
import React from "react"
import { render, screen } from "@testing-library/react"
import userEvent from "@testing-library/user-event"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"

const browserMocks = vi.hoisted(() => ({
  createTab: vi.fn(() => Promise.resolve({ id: 1 })),
  getURL: vi.fn((path: string) => `chrome-extension://tldw${path}`)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: { getURL: browserMocks.getURL },
    tabs: { create: browserMocks.createTab }
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("antd", () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/hooks/useMessage", () => ({
  useMessage: () => ({ temporaryChat: false })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({ error: vi.fn() })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: { hasPersona: false } })
}))

vi.mock("../TtsClipsDrawer", () => ({
  TtsClipsDrawer: () => <div data-testid="tts-clips-drawer" />
}))

import { SidepanelHeaderSimple } from "../SidepanelHeaderSimple"

describe("SidepanelHeaderSimple full-screen route", () => {
  it("opens the rail-enabled full app chat route", async () => {
    const user = userEvent.setup()

    render(
      <MemoryRouter>
        <SidepanelHeaderSimple activeTitle="Sidepanel chat" />
      </MemoryRouter>
    )

    await user.click(screen.getByTestId("chat-open-full-screen"))

    expect(browserMocks.getURL).toHaveBeenCalledWith("/options.html#/chat")
    expect(browserMocks.createTab).toHaveBeenCalledWith({
      url: "chrome-extension://tldw/options.html#/chat"
    })
  })
})
```

Expected before implementation: fails because the component currently uses `/options.html#/`.

- [ ] **Step 2: Implement the minimal route change**

In `SidepanelHeaderSimple.tsx`, change:

```ts
const url = browser.runtime.getURL("/options.html#/")
```

to:

```ts
const url = browser.runtime.getURL("/options.html#/chat")
```

- [ ] **Step 3: Run the sidepanel header tests**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx \
  src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.tts-clips-lazy-mount.test.ts
```

Expected: both pass.

- [ ] **Step 4: Update the refreshed audit**

Classify the previous extension full-screen handoff finding as:

- `still reproduces before Task 4`;
- `fixed by Task 4` after the test and code change pass.

- [ ] **Step 5: Commit**

```bash
git add apps/packages/ui/src/components/Sidepanel/Chat/SidepanelHeaderSimple.tsx \
  apps/packages/ui/src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx \
  Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md \
  "backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md" \
  "backlog/tasks/task-518 - Route-sidepanel-full-screen-chat-handoff-to-chat.md"
git commit -m "fix(chat): open sidepanel full screen to chat"
```

## Task 5: Complete The Refreshed Rail-Enabled UX Evaluation

**Files:**
- Modify: `Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md`
- Add/update artifacts under: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/`

Completion note: this task records each requested subflow as either directly observed or explicitly not revalidated with the blocking evidence. The full real-server cockpit suite still has non-rail baseline failures, so the refreshed audit must not claim green-path coverage for first-send/streaming, retry, prompt picker, long sessions, or compare/export/share unless those flows are separately rerun and pass.

- [x] **Step 1: Re-run desktop first-time journey**

Use the browser on `/chat` at desktop width. Record:

- whether cockpit rails are visible by default;
- whether focus mode is reachable;
- whether setup/readiness blocks or degrades gracefully;
- first-send behavior, streaming, stop, queue, retry, and save/title behavior;
- prompt/model/persona/RAG discoverability with rails present.

Required viewport and artifact:

- `1440x960`
- `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-cockpit.png`

Status: completed as an evidence-limited rebaseline. Rails, focus reachability, readiness, composer reachability, and discoverability were recorded. First-send, streaming, stop, queue, retry, and save/title were recorded as not newly revalidated because the live cockpit suite is blocked by captured non-rail baseline failures.

- [x] **Step 2: Re-run desktop power-user journey**

Record:

- model/provider settings workflow;
- prompt picker;
- Search & Context/context rail flow;
- MCP/tools runtime inspector flow;
- character rail overlay/tracked chat flow;
- history/sidebar flow;
- long session controls.

Required artifact:

- `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/desktop-focus.png`

Status: completed as an evidence-limited rebaseline. Model/provider, context/runtime, character, and focus/cockpit observations were recorded. Prompt picker, history/sidebar, long-session controls, and compare/export/share were recorded as not newly revalidated in this slice.

- [x] **Step 3: Re-run mobile journey**

At `390x844`, record:

- focus default;
- cockpit panel reveal;
- context/runtime tabs;
- composer reachability;
- horizontal overflow metrics;
- screenshots.

Required viewport and artifacts:

- `390x844`
- `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-focus.png`
- `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/mobile-cockpit.png`

- [x] **Step 4: Re-run extension handoff journey**

Use `http://localhost:<port>/__debug__/sidepanel-chat` when the packaged extension is not being driven directly. Record:

- sidepanel starter state;
- capture/handoff affordances that are visible;
- full-screen button target;
- sidepanel-to-`/chat` state preservation limitations.

Exact screenshot procedure if using the Next debug route:

1. Start WebUI in one terminal:

```bash
cd apps/tldw-frontend
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 bun run dev -- -H 127.0.0.1 -p 18014
```

2. Capture the sidepanel debug screenshot from another terminal:

```bash
cd apps/tldw-frontend
bunx playwright screenshot \
  --browser=chromium \
  --viewport-size=390,844 \
  http://localhost:18014/__debug__/sidepanel-chat \
  ../../Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png
```

Expected: `Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline/extension-sidepanel.png` exists. If the packaged extension is used instead, record the browser/profile and save the screenshot to the same path.

- [x] **Step 5: Fill all audit tables**

Every finding must include:

- route;
- viewport;
- observed behavior;
- evidence path or source file;
- prior-finding classification;
- severity;
- confidence;
- whether it is eligible for the first implementation plan.

- [ ] **Step 6: Commit**

```bash
git add Docs/Reviews/CHAT_RAILS_UX_REBASELINE_2026_05_27.md \
  Docs/Reviews/assets/2026-05-27-chat-rails-ux-rebaseline \
  "backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md"
git commit -m "docs(chat): rebaseline rail-enabled UX audit"
```

## Task 6: Final Verification And Handoff

**Files:**
- Modify: `backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md`

- [x] **Step 1: Run focused verification**

```bash
cd apps/packages/ui
bunx vitest run \
  src/components/Option/Playground/__tests__/Playground.cockpit-regression.guard.test.ts \
  src/components/Option/Playground/__tests__/Playground.cockpit-shell.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-controls.test.tsx \
  src/components/Option/Playground/__tests__/Playground.cockpit-a11y.test.tsx \
  src/components/Option/Playground/__tests__/PlaygroundContextRail.first-slice.test.tsx \
  src/components/Option/Playground/__tests__/PlaygroundRuntimeInspector.first-slice.test.tsx \
  src/components/Option/Playground/__tests__/CharacterControlRail.test.tsx \
  src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.fullscreen-route.test.tsx \
  src/components/Sidepanel/Chat/__tests__/SidepanelHeaderSimple.tts-clips-lazy-mount.test.ts
```

Expected: all focused tests pass.

- [x] **Step 2: Run real-server Playwright if backend is available**

```bash
cd apps/tldw-frontend
TLDW_E2E_SERVER_URL=http://127.0.0.1:8000 \
TLDW_E2E_API_KEY=THIS-IS-A-SECURE-KEY-123-FAKE-KEY \
TLDW_SERVER_URL=http://127.0.0.1:8000 \
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000 \
TLDW_WEB_URL=http://localhost:18014 \
TLDW_WEB_CMD='bun run dev -- -H 127.0.0.1 -p 18014' \
bunx playwright test e2e/workflows/chat-cockpit.real-server.spec.ts --project=chromium --reporter=line
```

Expected: passes, or audit records backend unavailability/degradation as a limitation.

- [x] **Step 3: Run Markdown/static checks**

```bash
git diff --check
git diff --cached --check
```

Expected: no output.

- [x] **Step 4: Bandit decision**

Skip Bandit for this slice if only frontend TypeScript, Markdown, and screenshots are touched. Record the skip in `TASK-516`.

- [x] **Step 5: Update Backlog final summary**

Record:

- test commands and results;
- screenshots/audit path;
- whether rails passed;
- first eligible UX remediation slice after rebaseline.

- [ ] **Step 6: Commit final task update**

```bash
git add "backlog/tasks/task-516 - Design-chat-rails-UX-rebaseline-and-remediation-from-origin-dev.md"
git commit -m "docs(chat): close rails UX rebaseline planning"
```

## Expected Next Work After This Plan

Do not implement broad UX fixes in this plan. After the refreshed audit is complete, create separate Backlog tasks for the first verified user-facing remediation slices. The likely candidates are:

- first-run readiness/setup feedback if still reproducing;
- prompt picker empty-state improvements if still reproducing;
- compare-mode disabled-state reason if still reproducing;
- context preview/source transparency if still weak with the rails present;
- extension dashboard route cleanup if still confirmed separate from full-screen `/chat`.
