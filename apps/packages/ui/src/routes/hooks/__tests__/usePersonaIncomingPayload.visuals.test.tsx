import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import { usePersonaIncomingPayload } from "../usePersonaIncomingPayload"

const noop = () => undefined

describe("usePersonaIncomingPayload visual state overrides", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(1_000)
    usePersonaVisualRuntimeStore.setState({ override: null })
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("stores bounded visual_state_override payloads for the active persona session", () => {
    const liveVoiceController = { handlePayload: vi.fn() }
    const { result } = renderHook(() => {
      const [approvedStepMap, setApprovedStepMap] = React.useState<Record<number, boolean>>({})
      const [pendingApprovals, setPendingApprovals] = React.useState<any[]>([])
      const [pendingPlan, setPendingPlan] = React.useState<any>(null)
      const [resolvedApprovalSnapshot, setResolvedApprovalSnapshot] =
        React.useState<{ key: string; toolName: string } | null>(null)
      const [setupTestOutcome, setSetupTestOutcome] = React.useState<any>(null)
      const [setupLiveDetour, setSetupLiveDetour] = React.useState<any>(null)
      const [setupTestResumeNote, setSetupTestResumeNote] = React.useState<string | null>(null)

      void approvedStepMap
      void pendingApprovals
      void pendingPlan
      void resolvedApprovalSnapshot
      void setupTestOutcome
      void setupLiveDetour
      void setupTestResumeNote

      return usePersonaIncomingPayload({
        appendLog: vi.fn(),
        clearResolvedApprovalFadeTimer: vi.fn(),
        consumeSetupHandoffAction: vi.fn(),
        emitSetupAnalyticsEvent: vi.fn(),
        liveVoiceController,
        personaId: "persona-1",
        personaSetupWizardCurrentStep: "test",
        personaSetupWizardIsSetupRequired: false,
        resolvedApprovalSnapshot,
        sessionId: "session-1",
        setApprovedStepMap,
        setPendingApprovals,
        setPendingPlan,
        setResolvedApprovalSnapshot,
        setSetupTestOutcome,
        setSetupLiveDetour,
        setSetupTestResumeNote,
        setupLiveDetourRef: React.useRef(null),
        setupHandoffRef: React.useRef(null),
        activeTabRef: React.useRef("live"),
        setupWizardAwaitingLiveResponseRef: React.useRef(false),
        setupWizardLastLiveTextRef: React.useRef("")
      })
    })

    act(() => {
      result.current({
        type: "visual_state_override",
        state: "speaking",
        duration_ms: 750,
        reason: "persona_visuals.trigger_state"
      })
    })

    expect(liveVoiceController.handlePayload).toHaveBeenCalled()
    expect(usePersonaVisualRuntimeStore.getState().override).toEqual({
      personaId: "persona-1",
      sessionId: "session-1",
      state: "speaking",
      reason: "persona_visuals.trigger_state",
      expiresAt: 1_750
    })
  })
})
