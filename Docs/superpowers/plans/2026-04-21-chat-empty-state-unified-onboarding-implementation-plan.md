# Chat Empty State Unified Onboarding Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the stacked chat empty-state blocks with one cohesive onboarding card while keeping the current actions and routes unchanged.

**Architecture:** Rewrite `PlaygroundEmpty` so it renders one outer onboarding shell with internal sections for hero content, actions, starter modes, and footer help. Keep the change local to the chat empty-state component and update the focused Vitest coverage to guard the new layout contract.

**Tech Stack:** React, Tailwind utility classes, Vitest, Testing Library

---

### Task 1: Guard The Unified Empty-State Contract

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

- [ ] **Step 1: Write the failing tests**

Add assertions that expect:
- one unified onboarding shell test id
- the primary actions and guided-mode buttons to exist inside that shell
- the old standalone `Start with a guided mode:` heading to be absent
- the disconnected `Open Settings` button to still exist inside the unified shell

- [ ] **Step 2: Run the focused tests to verify failure**

Run:
`bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

Expected: FAIL because `PlaygroundEmpty` does not yet render the new unified shell contract.

### Task 2: Implement The Single Onboarding Card

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`
- Test: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

- [ ] **Step 1: Write the minimal implementation**

Replace the stacked `FeatureEmptyState` + guided-mode section + footer section structure with one outer onboarding card that contains:
- hero content
- action row
- guided mode deck
- footer tips and tour link

- [ ] **Step 2: Run the focused tests to verify they pass**

Run:
`bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

Expected: PASS

### Task 3: Verify Formatting And Touched Scope

**Files:**
- Modify: `apps/packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx`
- Modify: `apps/packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

- [ ] **Step 1: Normalize formatting**

Run:
`bunx prettier --write ../packages/ui/src/components/Option/Playground/PlaygroundEmpty.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

- [ ] **Step 2: Re-run focused verification**

Run:
`bunx vitest run ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.test.tsx ../packages/ui/src/components/Option/Playground/__tests__/PlaygroundEmpty.disconnected.test.tsx`

Expected: PASS

- [ ] **Step 3: Run repo-required security validation**

Run:
`source .venv/bin/activate && python -m bandit -r apps/packages/ui/src/components/Option/Playground -f json -o /tmp/bandit_chat_empty_state_unified_onboarding.json`

Expected: no actionable findings for the touched scope.
