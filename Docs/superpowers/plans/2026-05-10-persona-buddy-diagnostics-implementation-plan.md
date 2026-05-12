# Persona/Buddy Diagnostics Surface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add the first Stage 1 Persona/Buddy reliability slice: a compact, read-only diagnostics surface for Persona Garden / Persona Live / Buddy shell state.

**Architecture:** Keep V1 frontend-only and reference-backed by existing runtime state. A pure diagnostics projector converts current persona, Buddy, Live, wake, MCP capability, and visual diagnostic inputs into a small state model; a presentational panel renders it with the existing design-system state components; the Persona sidepanel route wires current state into the projector without changing backend contracts or healthy control flows.

**Tech Stack:** Next.js/React, TypeScript, Vitest, Testing Library, existing `StatePanel` / `RecoveryCallout` design-system components, existing Persona visual diagnostics helpers.

---

## Context

This plan follows the merged Stage 0 audit in `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md` and GitHub issue `#1511`.

The Stage 1 scope is intentionally narrow:

- Add read-only diagnostics for the existing Persona/Buddy runtime.
- Use existing client/runtime state only.
- Do not add new MCP tools, backend endpoints, renderer behavior, native/background wake support, Persona Chat quality/eval work, or VN/CYOA changes.
- Keep visual-pack failures fail-open: the assistant should continue to render or fall back while diagnostics explain what is degraded.

## File Structure

- Create: `apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts`
  - Pure TypeScript diagnostics model and projector.
  - Converts route/runtime inputs into `healthy`, `unavailable`, `degraded`, or `recovering` plus diagnostic rows.
  - No React imports.

- Create: `apps/packages/ui/src/components/PersonaGarden/PersonaBuddyDiagnosticsPanel.tsx`
  - Small presentational component around the diagnostics projection.
  - Reuses `StatePanel` or `RecoveryCallout` from `apps/packages/ui/src/components/ui/state/`.

- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts`
  - Unit coverage for the projector.

- Create: `apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx`
  - Rendering coverage for the compact panel.

- Modify: `apps/packages/ui/src/components/PersonaGarden/LiveSessionPanel.tsx`
  - Add an optional `diagnostics?: React.ReactNode` slot.
  - Render the slot without changing existing controls, assistant voice, error, transcript, or composer ordering semantics.

- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Build diagnostics inputs from existing route state.
  - Pass the diagnostics panel into `LiveSessionPanel`.
  - Use `capabilities.hasMcp` as the current MCP transport readiness signal; mark persona-visuals tool readiness as unknown when the route does not have a more specific client-side signal.

- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`
  - Add one route-level degraded-state assertion proving the diagnostics surface appears while existing Persona Live controls remain available.

- Modify: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`
  - Add a brief implementation note linking the Stage 1 diagnostics slice to issue `#1511` and the implementation plan.

- Modify: `backlog/tasks/task-228 - Add-Persona-Buddy-diagnostics-surface.md`
  - Keep Backlog status, plan, verification, and final summary current.

## State Model

Use a small, explicit projection type so behavior is testable without rendering the full route.

```ts
export type PersonaBuddyDiagnosticState =
  | "healthy"
  | "unavailable"
  | "degraded"
  | "recovering";

export type PersonaBuddyDiagnosticRow = {
  label: string;
  value: string;
  state?: PersonaBuddyDiagnosticState;
  detail?: string;
};

export type PersonaBuddyDiagnostics = {
  state: PersonaBuddyDiagnosticState;
  title: string;
  message: string;
  rows: PersonaBuddyDiagnosticRow[];
};
```

Severity rules, highest priority first:

1. `unavailable`: server capabilities are still unavailable, persona support is missing, or no selected persona can be resolved.
2. `recovering`: Live session is reconnecting/connecting, or live voice recovery mode is active.
3. `degraded`: existing feature is available but impaired, such as Live websocket error, wake warning, text-only TTS fallback, visual pack load/render diagnostic with warning/error severity, or MCP unavailable.
4. `healthy`: core Persona Live / Buddy state is usable, with informational rows allowed.

Important visual-pack nuance:

- `no_active_pack` is informational by default because a persona can validly use the default buddy avatar.
- broken, unsupported, missing, or failed visual packs are degraded but fail open.

## Task 1: Pure Diagnostics Projector

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts`

- [ ] **Step 1: Write failing unit tests**

Cover these cases:

```ts
it("returns healthy diagnostics for connected persona live state", () => {
  const diagnostics = buildPersonaBuddyDiagnostics({
    selectedPersona: { id: "persona-1", name: "Ada" },
    profileState: "loaded",
    buddySummary: "Uses the active persona profile.",
    capabilities: { hasPersona: true, hasMcp: true },
    liveSession: { connected: true, connecting: false, sessionId: "session-1" },
    liveVoice: { state: "idle", recoveryMode: "none" },
    wake: { armed: true, detectorState: "ready" },
    visual: { packLoadStatus: "loaded", diagnostic: null },
  });

  expect(diagnostics.state).toBe("healthy");
  expect(diagnostics.rows).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ label: "Persona", value: "Ada" }),
      expect.objectContaining({ label: "Live session", value: "Connected" }),
    ]),
  );
});

it("marks missing persona support unavailable", () => {
  const diagnostics = buildPersonaBuddyDiagnostics({
    selectedPersona: null,
    profileState: "idle",
    buddySummary: null,
    capabilities: { hasPersona: false, hasMcp: false },
    liveSession: { connected: false, connecting: false, sessionId: null },
    liveVoice: { state: "idle", recoveryMode: "none" },
    wake: { armed: false, detectorState: "idle" },
    visual: { packLoadStatus: "idle", diagnostic: null },
  });

  expect(diagnostics.state).toBe("unavailable");
  expect(diagnostics.message).toMatch(/persona/i);
});

it("marks reconnecting live voice state recovering", () => {
  const diagnostics = buildPersonaBuddyDiagnostics({
    selectedPersona: { id: "persona-1", name: "Ada" },
    profileState: "loaded",
    buddySummary: "Ready",
    capabilities: { hasPersona: true, hasMcp: true },
    liveSession: { connected: false, connecting: true, sessionId: "session-1" },
    liveVoice: { state: "listening", recoveryMode: "reconnect" },
    wake: { armed: true, detectorState: "ready" },
    visual: { packLoadStatus: "loaded", diagnostic: null },
  });

  expect(diagnostics.state).toBe("recovering");
  expect(diagnostics.rows).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ label: "Live session", state: "recovering" }),
    ]),
  );
});

it("marks broken visual packs degraded without treating no active pack as broken", () => {
  const diagnostics = buildPersonaBuddyDiagnostics({
    selectedPersona: { id: "persona-1", name: "Ada" },
    profileState: "loaded",
    buddySummary: "Ready",
    capabilities: { hasPersona: true, hasMcp: true },
    liveSession: { connected: true, connecting: false, sessionId: "session-1" },
    liveVoice: { state: "idle", recoveryMode: "none" },
    wake: { armed: true, detectorState: "ready" },
    visual: {
      packLoadStatus: "error",
      diagnostic: {
        code: "missing_manifest",
        severity: "warning",
        message: "Visual pack manifest is missing.",
      },
    },
  });

  expect(diagnostics.state).toBe("degraded");
  expect(diagnostics.rows).toEqual(
    expect.arrayContaining([
      expect.objectContaining({ label: "Visual pack", state: "degraded" }),
    ]),
  );
});
```

Run:

```bash
bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts
```

Expected: FAIL because `personaBuddyDiagnostics.ts` does not exist.

- [ ] **Step 2: Implement the projector**

Implementation requirements:

- Export stable input/output types.
- Keep copy concise and actionable.
- Do not import React.
- Accept nullable/unknown values from the route without throwing.
- Convert missing optional state to `"Unavailable"` or `"Unknown"` rows instead of hiding the row.

Suggested helper shape:

```ts
const stateRank: Record<PersonaBuddyDiagnosticState, number> = {
  healthy: 0,
  degraded: 1,
  recovering: 2,
  unavailable: 3,
};

function worseState(
  current: PersonaBuddyDiagnosticState,
  candidate: PersonaBuddyDiagnosticState,
): PersonaBuddyDiagnosticState {
  return stateRank[candidate] > stateRank[current] ? candidate : current;
}
```

- [ ] **Step 3: Run projector tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts
```

Expected: PASS.

- [ ] **Step 4: Commit Task 1**

```bash
git add apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts \
  apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts \
  "backlog/tasks/task-228 - Add-Persona-Buddy-diagnostics-surface.md"
git commit -m "feat(ui): add persona buddy diagnostics projector"
```

## Task 2: Diagnostics Panel Component

**Files:**
- Create: `apps/packages/ui/src/components/PersonaGarden/PersonaBuddyDiagnosticsPanel.tsx`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx`

- [ ] **Step 1: Write failing render tests**

```tsx
it("renders compact diagnostics rows", () => {
  render(
    <PersonaBuddyDiagnosticsPanel
      diagnostics={{
        state: "degraded",
        title: "Persona Buddy degraded",
        message: "Visual pack needs attention.",
        rows: [
          { label: "Persona", value: "Ada", state: "healthy" },
          { label: "Visual pack", value: "Missing manifest", state: "degraded" },
        ],
      }}
    />,
  );

  expect(screen.getByText("Persona Buddy degraded")).toBeInTheDocument();
  expect(screen.getByText("Visual pack")).toBeInTheDocument();
  expect(screen.getByText("Missing manifest")).toBeInTheDocument();
});
```

Run:

```bash
bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx
```

Expected: FAIL because the component does not exist.

- [ ] **Step 2: Implement the panel**

Implementation requirements:

- Use the existing `StatePanel` component from `apps/packages/ui/src/components/ui/state/StatePanel.tsx` unless its layout conflicts with Persona Live.
- Keep it compact and unblocked: do not add modals, route changes, or hidden required interactions.
- Add `data-testid="persona-buddy-diagnostics"` to the root.
- Map diagnostic rows to `StatePanelDiagnostic[]`.
- Prefer existing tones/copy patterns; do not create a new design-system variant.

- [ ] **Step 3: Run panel tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx
```

Expected: PASS.

- [ ] **Step 4: Commit Task 2**

```bash
git add apps/packages/ui/src/components/PersonaGarden/PersonaBuddyDiagnosticsPanel.tsx \
  apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx \
  "backlog/tasks/task-228 - Add-Persona-Buddy-diagnostics-surface.md"
git commit -m "feat(ui): add persona buddy diagnostics panel"
```

## Task 3: Persona Live Route Integration

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/LiveSessionPanel.tsx`
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Modify: `apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx`

- [ ] **Step 1: Write a failing route-level degraded-state test**

Add a test that mounts the Persona sidepanel with:

- persona capability enabled,
- Live tab active,
- selected persona resolved,
- a visual-pack degraded signal or wake warning,
- normal Live controls still visible.

Expected assertions:

```ts
expect(await screen.findByTestId("persona-buddy-diagnostics")).toBeInTheDocument();
expect(screen.getByText(/degraded/i)).toBeInTheDocument();
expect(screen.getByRole("button", { name: /connect|disconnect|start/i })).toBeInTheDocument();
```

Use existing mocks in `sidepanel-persona.test.tsx`; do not create a separate route harness unless the current test file cannot express the state.

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx -t "diagnostics"
```

Expected: FAIL because the route does not render the diagnostics panel.

- [ ] **Step 2: Add a diagnostics slot to `LiveSessionPanel`**

Change props:

```tsx
export interface LiveSessionPanelProps {
  controls: React.ReactNode;
  assistantVoice: React.ReactNode;
  diagnostics?: React.ReactNode;
  error?: React.ReactNode;
  pendingPlan?: React.ReactNode;
  transcript: React.ReactNode;
  composer: React.ReactNode;
}
```

Render `diagnostics` near the voice/status controls, before transcript content, so it is visible but does not interrupt the transcript/composer workflow.

- [ ] **Step 3: Build diagnostics inputs in `sidepanel-persona.tsx`**

Use existing route state only:

- selected persona id/name from current persona selection/profile state,
- profile loading/error/loaded state,
- Buddy summary/dormant state from current Buddy shell context,
- Live websocket/session state from `usePersonaLiveSession`,
- live voice state/recovery/warnings from `useLiveVoiceController`,
- wake armed/detector/warning fields already passed to `AssistantVoiceCard`,
- visual feedback and visual diagnostics already available from existing helpers or from a narrow route-local check,
- `capabilities.hasMcp` for MCP transport readiness.

Do not add a backend request for diagnostics in this task.

- [ ] **Step 4: Render `PersonaBuddyDiagnosticsPanel` in the Live tab**

Pass the rendered diagnostics node into:

```tsx
<LiveSessionPanel
  controls={liveControls}
  assistantVoice={assistantVoiceCard}
  diagnostics={<PersonaBuddyDiagnosticsPanel diagnostics={personaBuddyDiagnostics} />}
  error={liveSessionStatusPanels}
  pendingPlan={pendingPlanCard}
  transcript={transcriptPanel}
  composer={composerPanel}
/>
```

If route variables differ, keep the same intent: diagnostics is a sibling of existing Live status controls, not a replacement.

- [ ] **Step 5: Run focused route test**

Run:

```bash
bunx vitest run apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx -t "diagnostics"
```

Expected: PASS.

- [ ] **Step 6: Commit Task 3**

```bash
git add apps/packages/ui/src/components/PersonaGarden/LiveSessionPanel.tsx \
  apps/packages/ui/src/routes/sidepanel-persona.tsx \
  apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx \
  "backlog/tasks/task-228 - Add-Persona-Buddy-diagnostics-surface.md"
git commit -m "feat(ui): surface persona buddy diagnostics in live view"
```

## Task 4: Documentation, Verification, And Backlog Closeout

**Files:**
- Modify: `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`
- Modify: `backlog/tasks/task-228 - Add-Persona-Buddy-diagnostics-surface.md`

- [ ] **Step 1: Add a Stage 1 implementation note to the audit**

Add a short note under the Stage 1 diagnostics recommendation:

```md
Implementation tracking: Stage 1 Persona/Buddy diagnostics is tracked by
GitHub issue #1511 and the implementation plan in
Docs/superpowers/plans/2026-05-10-persona-buddy-diagnostics-implementation-plan.md.
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts \
  apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx \
  apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx -t "diagnostics"
```

Expected: PASS.

- [ ] **Step 3: Run related visual diagnostics regression test**

Run:

```bash
bunx vitest run apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts
```

Expected: PASS.

- [ ] **Step 4: Run diff and security checks**

Run:

```bash
git diff --check
```

Expected: no whitespace errors.

Bandit: skipped for this implementation if only TypeScript/Markdown/Backlog files changed. If Python files are touched, run Bandit on those touched Python paths using the repository virtual environment.

- [ ] **Step 5: Update Backlog task**

Mark acceptance criteria complete only after verification passes. Record:

- files changed,
- test commands and results,
- Bandit skip or output,
- PR URL when available.

- [ ] **Step 6: Commit Task 4**

```bash
git add Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md \
  "backlog/tasks/task-228 - Add-Persona-Buddy-diagnostics-surface.md"
git commit -m "docs: track persona buddy diagnostics stage one"
```

## Final Verification Before PR

Run:

```bash
bunx vitest run apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts \
  apps/packages/ui/src/components/PersonaGarden/__tests__/PersonaBuddyDiagnosticsPanel.test.tsx \
  apps/packages/ui/src/routes/__tests__/sidepanel-persona.test.tsx -t "diagnostics"
bunx vitest run apps/packages/ui/src/components/Common/PersonaBuddy/__tests__/personaVisualDiagnostics.test.ts
git diff --check
```

Expected:

- All Vitest commands pass.
- `git diff --check` reports no errors.
- No Bandit run is required unless Python files are touched.

## PR Notes

The PR description should include a human-editable `Change summary` placeholder and mention:

- This is Stage 1 of issue `#1511` under epic `#1510`.
- The diagnostics surface is read-only and frontend-only.
- The implementation uses existing Persona/Buddy runtime state and visual diagnostic helpers.
- No new MCP tools, backend endpoints, renderer behavior, native wake support, Persona Chat quality changes, or VN/CYOA changes are included.
