import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import {
  usePersonaSetupOrchestrator,
  type UsePersonaSetupOrchestratorDeps
} from "../usePersonaSetupOrchestrator"

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { fetchWithAuth: vi.fn() }
}))

const buildDeps = (): UsePersonaSetupOrchestratorDeps => ({
  selectedPersonaId: "persona-1",
  setSelectedPersonaId: vi.fn(),
  isCompanionMode: false,
  activeTab: "profiles",
  setActiveTab: vi.fn(),
  connected: false,
  connecting: false,
  catalog: [{ id: "persona-1", name: "Helper" }],
  setCatalog: vi.fn(),
  personaProfileLoading: false,
  savedPersonaSetup: {
    status: "in_progress",
    current_step: "voice",
    run_id: "setup-1"
  },
  setSavedPersonaSetup: vi.fn(),
  savedPersonaVoiceDefaults: null,
  setSavedPersonaVoiceDefaults: vi.fn(),
  savedPersonaProfileVersion: 1,
  setSavedPersonaProfileVersion: vi.fn(),
  emitSetupAnalyticsEventRef: { current: vi.fn() },
  confirmDiscardUnsavedStateDraftsRef: { current: () => true },
  triggerRecoveryReconnectRef: { current: vi.fn() },
  setupLiveDetourRef: { current: null },
  setupHandoffRef: { current: null },
  setupHandoffFocusRequestRef: { current: null },
  activeTabRef: { current: "profiles" },
  setupWizardAwaitingLiveResponseRef: { current: false },
  setupWizardLastLiveTextRef: { current: "" }
})

describe("setup voice-default checkpoint handoff", () => {
  it.each([
    { oldSuccess: true, roundTrip: false },
    { oldSuccess: false, roundTrip: false },
    { oldSuccess: true, roundTrip: true },
    { oldSuccess: false, roundTrip: true }
  ])(
    "isolates a pending checkpoint after a persona switch (old success: $oldSuccess, round trip: $roundTrip)",
    async ({ oldSuccess, roundTrip }) => {
      type ProfileResponse = Awaited<
        ReturnType<typeof tldwClient.fetchWithAuth>
      >
      const responses: Array<(value: ProfileResponse) => void> = []
      vi.mocked(tldwClient.fetchWithAuth).mockImplementation(
        () =>
          new Promise((resolve) => {
            responses.push(resolve)
          })
      )
      const deps = buildDeps()
      const { result, rerender } = renderHook(
        (props) => usePersonaSetupOrchestrator(props),
        { initialProps: deps }
      )
      let firstSave!: Promise<void>
      act(() => {
        firstSave = result.current.handleSetupVoiceDefaultsSaved({
          id: "persona-1",
          version: 2
        })
      })
      expect(result.current.setupWizardSaving).toBe(true)

      rerender({
        ...deps,
        selectedPersonaId: "persona-2",
        savedPersonaProfileVersion: 7
      })
      const currentPersonaId = roundTrip ? "persona-1" : "persona-2"
      if (roundTrip) {
        rerender({
          ...deps,
          selectedPersonaId: currentPersonaId,
          savedPersonaProfileVersion: 7
        })
      }
      let secondSave!: Promise<void>
      act(() => {
        secondSave = result.current.handleSetupVoiceDefaultsSaved({
          id: currentPersonaId,
          version: 8
        })
      })
      vi.mocked(deps.setSavedPersonaProfileVersion).mockClear()
      await act(async () => {
        responses[0]({
          ok: oldSuccess,
          error: "Old persona conflict",
          text: async () => "",
          json: async () => ({ id: "persona-1", version: 3 })
        })
        await firstSave
      })
      expect(deps.setSavedPersonaProfileVersion).not.toHaveBeenCalled()
      expect(result.current.currentSetupWizardError).toBeNull()
      expect(result.current.setupWizardSaving).toBe(true)

      await act(async () => {
        responses[1]({
          ok: true,
          text: async () => "",
          json: async () => ({ id: currentPersonaId, version: 9 })
        })
        await secondSave
      })
      expect(deps.setSavedPersonaProfileVersion).toHaveBeenCalledWith(9)
      expect(result.current.setupWizardSaving).toBe(false)
    }
  )
})
