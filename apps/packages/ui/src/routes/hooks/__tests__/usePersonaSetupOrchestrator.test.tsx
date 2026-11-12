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

type ProfileResponse = Awaited<ReturnType<typeof tldwClient.fetchWithAuth>>
const responseBody = (body: unknown): ProfileResponse => ({
  ok: true,
  json: async () => body,
  text: async () => ""
})

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

describe("owned Finish: current setup attempt", () => {
  const response = (ok = true): ProfileResponse => ({
    ok,
    error: ok ? undefined : "Profile changed",
    status: ok ? 200 : 409,
    text: async () => "",
    json: async () => ({
      id: "persona-1", version: 2,
      voice_defaults: { confirmation_mode: "always" },
      setup: { status: "completed", current_step: "test", run_id: "setup-1" }
    })
  })
  const patches = () => vi.mocked(tldwClient.fetchWithAuth).mock.calls.filter(
    ([, init]) => init?.method === "PATCH"
  )
  const ready = () => {
    vi.mocked(tldwClient.fetchWithAuth).mockReset()
    const deps = buildDeps()
    deps.savedPersonaSetup = {
      ...deps.savedPersonaSetup!, current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"]
    }
    deps.savedPersonaVoiceDefaults = { confirmation_mode: "always" }
    return deps
  }

  it.each([
    [null, "dry_run"],
    [{ kind: "dry_run_failure", message: "Failed" }, "dry_run"],
    [{ kind: "dry_run_no_match", heardText: "unknown" }, "dry_run"],
    [{ kind: "live_sent", text: "pending" }, "live_session"],
    [{ kind: "live_success", text: "question", responseText: "answer" }, "dry_run"],
    [{ kind: "dry_run_match", heardText: "search" }, "live_session"],
    [{ kind: "dry_run_match", heardText: "search" }, "invalid"]
  ] as const)("owned Finish: refuses a missing, failed, pending or mismatched outcome (%s, %s)", async (outcome, completion) => {
    const deps = ready()
    vi.mocked(tldwClient.fetchWithAuth).mockResolvedValue(response())
    const { result } = renderHook(() => usePersonaSetupOrchestrator(deps))
    if (outcome) act(() => result.current.setSetupTestOutcome(outcome))
    await act(async () => {
      await result.current.completePersonaSetup(completion as "dry_run" | "live_session")
    })
    expect(patches()).toHaveLength(0)
    expect(result.current.setupHandoff).toBeNull()
    expect(deps.setSavedPersonaSetup).not.toHaveBeenCalled()
    expect(vi.mocked(deps.emitSetupAnalyticsEventRef.current).mock.calls.some(
      ([event]) => event.eventType === "setup_completed"
    )).toBe(false)
  })

  it.each([null, 0])("owned Finish: refuses a missing or zero profile version (%s)", async (version) => {
    const deps = ready()
    deps.savedPersonaProfileVersion = version
    vi.mocked(tldwClient.fetchWithAuth).mockResolvedValue(response())
    const { result } = renderHook(() => usePersonaSetupOrchestrator(deps))
    act(() => result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" }))
    await act(async () => { await result.current.completePersonaSetup("dry_run") })
    expect(patches()).toHaveLength(0)
    expect(result.current.setupHandoff).toBeNull()
  })

  it.each(["version", "run"] as const)(
    "owned Finish: a real version-2 dry-run match cannot finish after a new %s on the test step",
    async (change) => {
      const deps = ready()
      deps.savedPersonaProfileVersion = 2
      vi.mocked(tldwClient.fetchWithAuth).mockImplementation((path, init) => {
        if (String(path).includes("/voice-commands/test") && init?.method === "POST") {
          return Promise.resolve(responseBody({
            heard_text: "search notes", matched: true, command_name: "Search Notes"
          }))
        }
        return Promise.resolve(response())
      })
      const { result, rerender } = renderHook(
        (props) => usePersonaSetupOrchestrator(props), { initialProps: deps }
      )
      await act(async () => { await result.current.handleRunSetupDryRun("search notes") })
      expect(result.current.setupTestOutcome).toMatchObject({
        kind: "dry_run_match", heardText: "search notes", commandName: "Search Notes"
      })
      if (change === "version") rerender({ ...deps, savedPersonaProfileVersion: 3 })
      else rerender({ ...deps, savedPersonaSetup: { ...deps.savedPersonaSetup!, run_id: "setup-2" } })
      act(() => result.current.setSetupReviewSummaryDraft({
        starterCommands: { mode: "skipped" }, confirmationMode: "always", connection: { mode: "skipped" }
      }))
      await act(async () => { await result.current.completePersonaSetup("dry_run") })
      expect(patches()).toHaveLength(0)
      expect(result.current.setupHandoff).toBeNull()
    }
  )

  it("owned Finish: carries the successful profile version into one PATCH", async () => {
    const deps = ready()
    vi.mocked(tldwClient.fetchWithAuth).mockResolvedValue(response())
    const { result } = renderHook(() => usePersonaSetupOrchestrator(deps))
    act(() => result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" }))
    await act(async () => { await result.current.completePersonaSetup("dry_run") })
    expect(patches()).toHaveLength(1)
    expect(String(patches()[0][0])).toContain("expected_version=1")
    expect(result.current.setupHandoff?.completionType).toBe("dry_run")
  })

  it("owned Finish: submits duplicate completion calls once", async () => {
    const deps = ready()
    vi.mocked(tldwClient.fetchWithAuth).mockResolvedValue(response())
    const { result } = renderHook(() => usePersonaSetupOrchestrator(deps))
    act(() => result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" }))
    await act(async () => {
      await Promise.all([
        result.current.completePersonaSetup("dry_run"),
        result.current.completePersonaSetup("dry_run")
      ])
    })
    expect(patches()).toHaveLength(1)
  })

  it.each(["persona", "run", "version", "auth", "authority"] as const)(
    "owned Finish: a pending summary read cannot PATCH after %s changes",
    async (change) => {
      const deps = buildDeps()
      deps.savedPersonaSetup = {
        ...deps.savedPersonaSetup!, current_step: "test",
        completed_steps: ["persona", "voice", "commands", "safety"]
      }
      vi.mocked(tldwClient.fetchWithAuth).mockReset()
      const reads: Array<{ path: string; resolve: (value: ProfileResponse) => void }> = []
      vi.mocked(tldwClient.fetchWithAuth).mockImplementation(
        (path, init) => init?.method === "PATCH"
          ? Promise.resolve(response())
          : new Promise<ProfileResponse>(resolve => reads.push({ path: String(path), resolve }))
      )
      const { result, rerender } = renderHook(
        (props) => usePersonaSetupOrchestrator(props), { initialProps: deps }
      )
      act(() => result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" }))
      let finish!: Promise<void>
      act(() => { finish = result.current.completePersonaSetup("dry_run") })
      expect(reads).toHaveLength(2)
      if (change === "persona") rerender({ ...deps, selectedPersonaId: "persona-2" })
      if (change === "run") rerender({ ...deps, savedPersonaSetup: { ...deps.savedPersonaSetup!, run_id: "setup-2" } })
      if (change === "version") rerender({ ...deps, savedPersonaProfileVersion: 2 })
      if (change === "auth" || change === "authority") {
        act(() => window.dispatchEvent(new CustomEvent(
          change === "auth" ? "tldw:auth-principal-changed" : "tldw:config-updated",
          { detail: change === "authority" ? { authorityChanged: true } : { kind: "logout" } }
        )))
      }
      await act(async () => {
        reads.forEach(({ path, resolve }) => resolve(responseBody(
          path.includes("/voice-commands") ? { commands: [], total: 0 } : []
        )))
        await finish
      })
      expect(patches()).toHaveLength(0)
      expect(result.current.setupHandoff).toBeNull()
    }
  )

  it.each([true, false])("owned Finish: ignores a late PATCH result after persona changes (ok: %s)", async (ok) => {
    const deps = ready()
    let resolvePatch!: (value: ProfileResponse) => void
    vi.mocked(tldwClient.fetchWithAuth).mockImplementation(
      () => new Promise<ProfileResponse>(resolve => { resolvePatch = resolve })
    )
    const { result, rerender } = renderHook(
      (props) => usePersonaSetupOrchestrator(props), { initialProps: deps }
    )
    act(() => {
      result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" })
      result.current.setSetupReviewSummaryDraft({
        starterCommands: { mode: "skipped" }, confirmationMode: "always", connection: { mode: "skipped" }
      })
    })
    let finish!: Promise<void>
    act(() => { finish = result.current.completePersonaSetup("dry_run") })
    await act(async () => { await Promise.resolve() })
    expect(patches()).toHaveLength(1)
    rerender({ ...deps, selectedPersonaId: "persona-2", savedPersonaProfileVersion: 7 })
    vi.mocked(deps.setSavedPersonaSetup).mockClear()
    await act(async () => {
      resolvePatch(response(ok))
      await finish
    })
    expect(deps.setSavedPersonaSetup).not.toHaveBeenCalled()
    expect(result.current.setupHandoff).toBeNull()
    expect(result.current.setupStepErrors.test).toBeFalsy()
  })

  it.each([true, false])("owned Finish: late PATCH %s cannot clear a newer persona save", async (ok) => {
    const deps = ready()
    const pending: Array<(value: ProfileResponse) => void> = []
    vi.mocked(tldwClient.fetchWithAuth).mockImplementation(
      () => new Promise<ProfileResponse>(resolve => pending.push(resolve))
    )
    const { result, rerender } = renderHook(
      (props) => usePersonaSetupOrchestrator(props), { initialProps: deps }
    )
    act(() => {
      result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" })
      result.current.setSetupReviewSummaryDraft({
        starterCommands: { mode: "skipped" }, confirmationMode: "always", connection: { mode: "skipped" }
      })
    })
    let oldFinish!: Promise<void>
    act(() => { oldFinish = result.current.completePersonaSetup("dry_run") })
    await act(async () => { await Promise.resolve() })
    expect(pending).toHaveLength(1)
    rerender({ ...deps, selectedPersonaId: "persona-2", savedPersonaProfileVersion: 7,
      savedPersonaSetup: { ...deps.savedPersonaSetup!, current_step: "voice", run_id: "setup-2" }
    })
    let newSave!: Promise<void>
    act(() => { newSave = result.current.handleSetupVoiceDefaultsSaved({ id: "persona-2", version: 8 }) })
    expect(pending).toHaveLength(2)
    vi.mocked(deps.setSavedPersonaSetup).mockClear()
    await act(async () => {
      pending[0](response(ok))
      await oldFinish
    })
    expect(result.current.setupWizardSaving).toBe(true)
    expect(result.current.setupHandoff).toBeNull()
    expect(deps.setSavedPersonaSetup).not.toHaveBeenCalled()
    await act(async () => {
      pending[1](responseBody({ id: "persona-2", version: 9 }))
      await newSave
    })
    expect(result.current.setupWizardSaving).toBe(false)
  })

  it("owned Finish: unmount retires a pending completion PATCH", async () => {
    const deps = ready()
    let resolvePatch!: (value: ProfileResponse) => void
    vi.mocked(tldwClient.fetchWithAuth).mockImplementation(
      () => new Promise<ProfileResponse>(resolve => { resolvePatch = resolve })
    )
    const { result, unmount } = renderHook(() => usePersonaSetupOrchestrator(deps))
    act(() => {
      result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" })
      result.current.setSetupReviewSummaryDraft({
        starterCommands: { mode: "skipped" }, confirmationMode: "always", connection: { mode: "skipped" }
      })
    })
    let finish!: Promise<void>
    act(() => { finish = result.current.completePersonaSetup("dry_run") })
    await act(async () => { await Promise.resolve() })
    expect(patches()).toHaveLength(1)
    unmount()
    vi.mocked(deps.setSavedPersonaSetup).mockClear()
    vi.mocked(deps.emitSetupAnalyticsEventRef.current).mockClear()
    await act(async () => {
      resolvePatch(response())
      await finish
    })
    expect(deps.setSavedPersonaSetup).not.toHaveBeenCalled()
    expect(vi.mocked(deps.emitSetupAnalyticsEventRef.current).mock.calls.some(
      ([event]) => event.eventType === "setup_completed"
    )).toBe(false)
  })

  it("owned Finish: controlled CAS conflict leaves setup incomplete", async () => {
    const deps = ready()
    vi.mocked(tldwClient.fetchWithAuth).mockResolvedValue(response(false))
    const { result } = renderHook(() => usePersonaSetupOrchestrator(deps))
    act(() => result.current.setSetupTestOutcome({ kind: "dry_run_match", heardText: "search" }))
    await act(async () => { await result.current.completePersonaSetup("dry_run") })
    expect(patches()).toHaveLength(1)
    expect(result.current.setupHandoff).toBeNull()
    expect(result.current.setupStepErrors.test).toContain("Profile changed")
    expect(deps.setSavedPersonaSetup).not.toHaveBeenCalled()
  })
})

describe("owned reviewed setup: asynchronous scope ownership", () => {
  const readyDeps = () => {
    const deps = buildDeps()
    deps.savedPersonaSetup = {
      ...deps.savedPersonaSetup!, current_step: "test",
      completed_steps: ["persona", "voice", "commands", "safety"]
    }
    deps.savedPersonaProfileVersion = 2
    vi.mocked(tldwClient.fetchWithAuth).mockReset()
    return deps
  }
  const matchedDryRun = { heard_text: "search notes", matched: true, command_name: "Search Notes" }
  const completionPatches = () => vi.mocked(tldwClient.fetchWithAuth).mock.calls.filter(
    ([, init]) => init?.method === "PATCH"
  )

  it.each(["run", "version", "authority"] as const)(
    "owned reviewed setup: held Finish releases saving immediately when %s changes",
    async (change) => {
      const deps = readyDeps()
      const reads: Array<{ path: string; resolve: (value: ProfileResponse) => void }> = []
      vi.mocked(tldwClient.fetchWithAuth).mockImplementation((path, init) => {
        if (String(path).endsWith("/voice-commands/test") && init?.method === "POST") {
          return Promise.resolve(responseBody(matchedDryRun))
        }
        if (init?.method === "GET" && ["/voice-commands", "/connections"].some(suffix => String(path).endsWith(suffix))) {
          return new Promise<ProfileResponse>(resolve => reads.push({ path: String(path), resolve }))
        }
        return Promise.resolve(responseBody({ id: "persona-1", version: 3,
          setup: { status: "completed", current_step: "test", run_id: "setup-1" }
        }))
      })
      const { result, rerender } = renderHook(
        (props) => usePersonaSetupOrchestrator(props), { initialProps: deps }
      )
      await act(async () => { await result.current.handleRunSetupDryRun("search notes") })
      expect(result.current.setupTestOutcome).toMatchObject({ kind: "dry_run_match", heardText: "search notes" })
      let finish!: Promise<void>
      act(() => { finish = result.current.completePersonaSetup("dry_run") })
      await act(async () => { await Promise.resolve() })
      expect(reads).toHaveLength(2)
      expect(result.current.setupWizardSaving).toBe(true)
      if (change === "run") rerender({ ...deps, savedPersonaSetup: { ...deps.savedPersonaSetup!, run_id: "setup-2" } })
      if (change === "version") rerender({ ...deps, savedPersonaProfileVersion: 3 })
      if (change === "authority") act(() => window.dispatchEvent(
        new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } })
      ))
      const savingAfterChange = result.current.setupWizardSaving
      await act(async () => {
        reads.forEach(({ path, resolve }) => resolve(responseBody(
          path.endsWith("/voice-commands") ? { commands: [], total: 0 } : []
        )))
        await finish
      })
      expect(savingAfterChange).toBe(false)
      expect(completionPatches()).toHaveLength(0)
      expect(result.current.setupHandoff).toBeNull()
      expect(deps.setSavedPersonaSetup).not.toHaveBeenCalled()
      expect(vi.mocked(deps.emitSetupAnalyticsEventRef.current).mock.calls.some(
        ([event]) => event.eventType === "setup_completed"
      )).toBe(false)
    }
  )

  it.each(["persona", "run", "version", "authority"] as const)(
    "owned reviewed setup: held dry run releases loading immediately when %s changes",
    async (change) => {
      const deps = readyDeps()
      let resolveOld!: (value: ProfileResponse) => void
      vi.mocked(tldwClient.fetchWithAuth).mockImplementation(() =>
        new Promise<ProfileResponse>(resolve => { resolveOld = resolve })
      )
      const { result, rerender } = renderHook(
        (props) => usePersonaSetupOrchestrator(props), { initialProps: deps }
      )
      let oldRun!: Promise<void>
      act(() => { oldRun = result.current.handleRunSetupDryRun("old phrase") })
      expect(result.current.setupWizardDryRunLoading).toBe(true)
      if (change === "persona") rerender({ ...deps, selectedPersonaId: "persona-2" })
      if (change === "run") rerender({ ...deps, savedPersonaSetup: { ...deps.savedPersonaSetup!, run_id: "setup-2" } })
      if (change === "version") rerender({ ...deps, savedPersonaProfileVersion: 3 })
      if (change === "authority") act(() => window.dispatchEvent(
        new CustomEvent("tldw:config-updated", { detail: { authorityChanged: true } })
      ))
      const loadingAfterChange = result.current.setupWizardDryRunLoading
      await act(async () => {
        resolveOld(responseBody(matchedDryRun))
        await oldRun
      })
      expect(loadingAfterChange).toBe(false)
      expect(result.current.setupTestOutcome).toBeNull()
      expect(result.current.setupWizardDryRunLoading).toBe(false)
    }
  )

  it("owned reviewed setup: old dry-run completion cannot clear a newer run's loading", async () => {
    const deps = readyDeps()
    const pending: Array<(value: ProfileResponse) => void> = []
    vi.mocked(tldwClient.fetchWithAuth).mockImplementation(() =>
      new Promise<ProfileResponse>(resolve => pending.push(resolve))
    )
    const { result, rerender } = renderHook(
      (props) => usePersonaSetupOrchestrator(props), { initialProps: deps }
    )
    let oldRun!: Promise<void>
    act(() => { oldRun = result.current.handleRunSetupDryRun("old phrase") })
    rerender({ ...deps, savedPersonaSetup: { ...deps.savedPersonaSetup!, run_id: "setup-2" } })
    let newRun!: Promise<void>
    act(() => { newRun = result.current.handleRunSetupDryRun("new phrase") })
    expect(pending).toHaveLength(2)
    await act(async () => {
      pending[0](responseBody({ ...matchedDryRun, heard_text: "old phrase" }))
      await oldRun
    })
    expect(result.current.setupWizardDryRunLoading).toBe(true)
    expect(result.current.setupTestOutcome).toBeNull()
    await act(async () => {
      pending[1](responseBody({ ...matchedDryRun, heard_text: "new phrase" }))
      await newRun
    })
    expect(result.current.setupWizardDryRunLoading).toBe(false)
    expect(result.current.setupTestOutcome).toMatchObject({ kind: "dry_run_match", heardText: "new phrase" })
  })

  it("owned reviewed setup: invalid completion type cannot use an otherwise valid live success", async () => {
    const deps = readyDeps()
    vi.mocked(tldwClient.fetchWithAuth).mockResolvedValue(responseBody({
      id: "persona-1", version: 3, setup: { status: "completed", current_step: "test", run_id: "setup-1" }
    }))
    const { result } = renderHook(() => usePersonaSetupOrchestrator(deps))
    act(() => result.current.setSetupTestOutcome({ kind: "live_success", text: "question", responseText: "answer" }))
    await act(async () => {
      await result.current.completePersonaSetup("invalid" as "dry_run" | "live_session")
    })
    expect(completionPatches()).toHaveLength(0)
    expect(result.current.setupHandoff).toBeNull()
    expect(deps.setSavedPersonaSetup).not.toHaveBeenCalled()
    expect(vi.mocked(deps.emitSetupAnalyticsEventRef.current).mock.calls.some(
      ([event]) => event.eventType === "setup_completed"
    )).toBe(false)
  })
})
