import React from "react"
import { act, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

import { usePersonaVisualRuntimeStore } from "@/store/persona-visual-runtime"
import { usePersonaIncomingPayload } from "../usePersonaIncomingPayload"

const noop = () => undefined

it("does not publish rejected late voice replies but retains ordinary text replies", () => {
  const appendLog = vi.fn()
  const handlePayload = vi.fn((): boolean | void => false)
  const { result } = renderHook(() => usePersonaIncomingPayload({
    appendLog, liveVoiceController: { handlePayload }, personaId: "persona-1", sessionId: "session-1",
    clearResolvedApprovalFadeTimer: noop, consumeSetupHandoffAction: noop,
    emitSetupAnalyticsEvent: noop, personaSetupWizardCurrentStep: "test",
    personaSetupWizardIsSetupRequired: false, resolvedApprovalSnapshot: null,
    setApprovedStepMap: noop, setPendingApprovals: noop, setPendingPlan: noop,
    setResolvedApprovalSnapshot: noop, setSetupTestOutcome: noop, setSetupLiveDetour: noop,
    setSetupTestResumeNote: noop, setupLiveDetourRef: React.useRef(null),
    setupHandoffRef: React.useRef(null), activeTabRef: React.useRef("live"),
    setupWizardAwaitingLiveResponseRef: React.useRef(false), setupWizardLastLiveTextRef: React.useRef("")
  }))
  act(() => result.current({ event: "assistant_delta", text_delta: "late voice" }))
  expect(appendLog).not.toHaveBeenCalled()
  handlePayload.mockReturnValue(undefined)
  act(() => result.current({ event: "assistant_delta", text_delta: "ordinary text reply" }))
  expect(appendLog).toHaveBeenCalledWith("assistant", "ordinary text reply")
  appendLog.mockClear()
  act(() => result.current({ event: "partial_transcript", transcript: "blue note", text_delta: "blue note" }))
  act(() => result.current({ event: "partial_transcript", transcript: "blue notebook", text_delta: "book" }))
  expect(appendLog).not.toHaveBeenCalled()
  act(() => result.current({ event: "notice", reason_code: "VOICE_TURN_COMMITTED", transcript: "blue notebook", message: "Voice turn committed." }))
  expect(appendLog.mock.calls.filter(([kind]) => kind === "user")).toEqual([["user", "blue notebook"]])

})

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

  it("stores safe custom visual_state_override payloads for active-pack resolution", () => {
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
        state: "tool.notes_search",
        duration_ms: 1250,
        reason: "mcp_runtime.notes.search"
      })
    })

    expect(usePersonaVisualRuntimeStore.getState().override).toEqual({
      personaId: "persona-1",
      sessionId: "session-1",
      state: "tool.notes_search",
      reason: "mcp_runtime.notes.search",
      expiresAt: 2_250
    })
  })
})
