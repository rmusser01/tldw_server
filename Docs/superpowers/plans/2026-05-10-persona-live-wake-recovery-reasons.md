# Persona Live Wake Recovery Reasons Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Map existing Persona Live voice and wake degraded states to stable reason codes and concise recovery copy surfaced through the current hook and Buddy diagnostics UI.

**Architecture:** Keep runtime behavior unchanged and add reason metadata beside the existing warning strings. The hook remains the source of truth for live voice and wake codes; Buddy diagnostics consumes those codes to render specific recovery copy instead of inferring from ad hoc warning text.

**Tech Stack:** React hooks, TypeScript, Vitest, Testing Library, existing Persona Garden diagnostics components.

**Tracking:** Backlog `TASK-233`; GitHub issue `#1519`; Stage 0 audit `Docs/Reviews/PERSONA_BUDDY_CURRENT_STATE_AUDIT_2026_05_10.md`.

---

## File Structure

- Modify `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`
  - Add exported reason-code types.
  - Store `warningReasonCode` and `wakeWarningReasonCode` alongside existing `warning` and `wakeWarning`.
  - Set/clear codes at the same points existing warning copy is set/cleared.
- Modify `apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts`
  - Accept the hook reason codes in `liveVoice` and `wake` inputs.
  - Map known reason codes to concise recovery copy and diagnostic states.
- Modify `apps/packages/ui/src/routes/sidepanel-persona.tsx`
  - Pass the new hook codes into `buildPersonaBuddyDiagnostics`.
- Modify `apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx`
  - Cover controller reason codes for no wake phrases, detector unavailable/error, wake rejection, manual mode, TTS fallback, disconnected/reconnect, and teardown clearing.
- Modify `apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts`
  - Cover diagnostics output for reason-coded live voice and wake states.
- Update `backlog/tasks/task-233 - Map-Persona-Live-voice-and-wake-recovery-reasons.md`
  - Record plan path, touched files, verification, and final summary.

## Task 1: Hook Reason Codes

**Files:**
- Modify: `apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx`
- Test: `apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx`

- [ ] **Step 1: Write failing hook tests**

Add expectations such as:

```ts
expect(result.current.wakeWarningReasonCode).toBe("wake_not_configured")
expect(result.current.wakeWarningReasonCode).toBe("wake_detector_unavailable")
expect(result.current.wakeWarningReasonCode).toBe("wake_activation_rejected_not_saved_in_profile")
expect(result.current.warningReasonCode).toBe("voice_manual_mode_required")
expect(result.current.warningReasonCode).toBe("voice_tts_unavailable_text_only")
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx --maxWorkers=1
```

Expected: FAIL because the controller does not expose reason-code fields yet.

- [ ] **Step 3: Implement minimal hook metadata**

Add exported string-literal union types:

```ts
export type PersonaLiveVoiceWarningReasonCode =
  | "live_voice_disconnected"
  | "server_stt_unavailable"
  | "live_voice_source_pending"
  | "barge_in_disabled"
  | "voice_capture_error"
  | "voice_no_transcript"
  | "voice_manual_mode_required"
  | "voice_tts_unavailable_text_only"
  | "voice_commit_ignored_already_committed"
  | "voice_trigger_not_heard"
  | "voice_empty_command_after_trigger"

export type PersonaWakeWarningReasonCode =
  | "wake_not_configured"
  | "wake_detector_unavailable"
  | "wake_detector_permission_denied"
  | "wake_detector_error"
  | "wake_activation_disconnected"
  | "wake_activation_send_failed"
  | "wake_activation_rejected_not_saved_in_profile"
  | "wake_activation_rejected_missing_from_runtime_config"
  | "wake_activation_rejected_phrase_not_configured"
  | "wake_activation_rejected"
```

Set these codes wherever the existing hook sets `warning` or `wakeWarning`; clear them whenever the corresponding warning is cleared.

- [ ] **Step 4: Run tests to verify GREEN**

Run the same hook test command. Expected: PASS.

## Task 2: Diagnostics Mapping

**Files:**
- Modify: `apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts`
- Test: `apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts`

- [ ] **Step 1: Write failing diagnostics tests**

Add tests that pass reason codes into `buildPersonaBuddyDiagnostics` and assert rows use specific recovery copy:

```ts
expect(diagnostics.rows).toEqual(
  expect.arrayContaining([
    expect.objectContaining({
      label: "Wake",
      value: "Permission needed",
      detail: expect.stringMatching(/manual controls remain available/i)
    })
  ])
)
```

- [ ] **Step 2: Run tests to verify RED**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts --maxWorkers=1
```

Expected: FAIL because diagnostics does not accept or map reason codes.

- [ ] **Step 3: Implement minimal mapper**

Add small records for known live voice and wake reason codes. Keep copy short and action-oriented, and avoid broken-state language when fallback controls remain available.

- [ ] **Step 4: Run diagnostics tests to verify GREEN**

Run the same diagnostics test command. Expected: PASS.

## Task 3: Route Wiring

**Files:**
- Modify: `apps/packages/ui/src/routes/sidepanel-persona.tsx`
- Test: existing hook and diagnostics tests

- [ ] **Step 1: Wire reason codes into diagnostics input**

Pass:

```ts
warningReasonCode: liveVoiceController.warningReasonCode
wakeWarningReasonCode: liveVoiceController.wakeWarningReasonCode
```

- [ ] **Step 2: Run focused tests**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts --maxWorkers=1
```

Expected: PASS.

## Task 4: Verification and Closeout

**Files:**
- Modify: `backlog/tasks/task-233 - Map-Persona-Live-voice-and-wake-recovery-reasons.md`

- [ ] **Step 1: Run final focused verification**

Run:

```bash
cd apps/packages/ui
bunx vitest run src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts src/components/PersonaGarden/__tests__/LiveSessionPanel.test.tsx --maxWorkers=1
git diff --check
```

Expected: PASS and no whitespace errors.

- [ ] **Step 2: Bandit scope decision**

This slice touches frontend TypeScript and Backlog metadata only. Record Bandit as not applicable unless backend Python changes are added.

- [ ] **Step 3: Update Backlog task**

Mark acceptance criteria complete, record verification commands/results, link this plan, and add final summary.

- [ ] **Step 4: Commit**

```bash
git add Docs/superpowers/plans/2026-05-10-persona-live-wake-recovery-reasons.md \
  "backlog/tasks/task-233 - Map-Persona-Live-voice-and-wake-recovery-reasons.md" \
  apps/packages/ui/src/hooks/usePersonaLiveVoiceController.tsx \
  apps/packages/ui/src/hooks/__tests__/usePersonaLiveVoiceController.test.tsx \
  apps/packages/ui/src/components/PersonaGarden/personaBuddyDiagnostics.ts \
  apps/packages/ui/src/components/PersonaGarden/__tests__/personaBuddyDiagnostics.test.ts \
  apps/packages/ui/src/routes/sidepanel-persona.tsx
git commit -m "feat(ui): map persona live recovery reasons"
```

## Self-Review

- The plan keeps the slice in Persona/Buddy live voice and wake paths, not VN code.
- It does not add new runtime wake capabilities; it only names and surfaces existing states.
- It avoids snapshot data, backend schema work, and MCP surface changes outside the current Stage 1 issue.
- It preserves existing warning copy for the visible Assistant Voice card while improving diagnostics and testability.
