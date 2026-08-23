import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getConfig: vi.fn(),
  getCurrentUser: vi.fn(),
  submit: vi.fn(),
  status: vi.fn(),
  onCompleted: vi.fn(),
  onStopWaiting: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: (...args: unknown[]) => mocks.getConfig(...args),
    submitPresentationGeneration: (...args: unknown[]) => mocks.submit(...args),
    getPresentationGenerationStatus: (...args: unknown[]) => mocks.status(...args)
  }
}))

vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: (...args: unknown[]) => mocks.getCurrentUser(...args) }
}))

const capability = {
  enabled: true,
  reason: null,
  transport: "slides_generation_job",
  source_kinds: ["prompt", "chat", "media", "notes", "rag"],
  provider: "openai",
  model: "gpt-5-mini",
  adapter_id: "openai-responses",
  endpoint_identity: "https://api.openai.com/v1",
  generation_config_revision: `sha256:${"c".repeat(64)}`,
  input_limits: { max_request_bytes: 4_194_304, max_source_chars: 40, max_source_tokens: 50_000, max_audience_chars: 20, max_source_identifier_bytes: 256, max_note_ids: 100, max_rag_query_chars: 20_000, max_rag_top_k: 100 },
  output_limits: { max_provider_response_bytes: 8_388_608, max_document_bytes: 1_048_576 }
} as const

const withLimits = (limits: Partial<Record<keyof typeof capability.input_limits, number>>) => ({
  ...capability,
  input_limits: { ...capability.input_limits, ...limits }
})

const validDraft = {
  source: "Bounded source",
  presentationType: "tech-sharing",
  audience: "Engineers",
  slideCount: 8,
  visualDirection: "dark-technical",
  deliveryStyle: "speaker-led"
}

const pendingReceipt = {
  generation_id: "018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
  status: "queued",
  status_url: "/api/v1/slides/generations/018f2f4a-6f79-7a27-a1aa-7bb60777d9f1",
  presentation_id: null
}

const loadSubject = () =>
  vi.importActual<typeof import("../useStandaloneHtmlGeneration")>(
    ["..", "useStandaloneHtmlGeneration"].join("/")
  )

const setup = async () => {
  const module = await loadSubject()
  const hook = renderHook(() => module.useStandaloneHtmlGeneration({
    capability: capability as any,
    onCompleted: mocks.onCompleted,
    onStopWaiting: mocks.onStopWaiting
  }))
  await waitFor(() => expect(hook.result.current.scopeReady).toBe(true))
  act(() => hook.result.current.replaceDraft(validDraft as any))
  return { ...hook, module }
}

const storedValues = (): string =>
  Array.from({ length: sessionStorage.length }, (_, index) => sessionStorage.getItem(sessionStorage.key(index)!) ?? "").join("\n")

describe("useStandaloneHtmlGeneration", () => {
  beforeEach(() => {
    vi.useRealTimers()
    mocks.getConfig.mockReset()
    mocks.getCurrentUser.mockReset()
    mocks.submit.mockReset()
    mocks.status.mockReset()
    mocks.onCompleted.mockReset()
    mocks.onStopWaiting.mockReset()
    sessionStorage.clear()
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://tldw.example/base", authMode: "multi-user", accessToken: "secret-token" })
    mocks.getCurrentUser.mockResolvedValue({ id: 42, username: "researcher", is_active: true })
  })

  afterEach(() => {
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it("rejects NUL, unpaired surrogates, effective limits, and invalid slide counts before state or persistence", async () => {
    const { result } = await setup()
    const before = result.current.draft

    act(() => expect(result.current.updateField("source", "bad\u0000source")).toBe(false))
    expect(result.current.draft).toEqual(before)
    act(() => expect(result.current.updateField("source", "bad\ud800source")).toBe(false))
    act(() => expect(result.current.updateField("source", "x".repeat(41))).toBe(false))
    act(() => expect(result.current.updateField("audience", "x".repeat(21))).toBe(false))
    act(() => expect(result.current.updateField("slideCount", 31)).toBe(false))
    expect(Array.from({ length: sessionStorage.length }, (_, index) => sessionStorage.key(index))).not.toContain("bad\u0000source")
  })

  it("fails closed when the current trusted principal cannot be confirmed", async () => {
    mocks.getCurrentUser.mockRejectedValue(new Error("authentication unavailable"))
    const module = await loadSubject()
    const { result } = renderHook(() => module.useStandaloneHtmlGeneration({
      capability: capability as any,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))

    await waitFor(() => expect(result.current.scopeError).toBe("Current server and account could not be confirmed."))
    expect(result.current.scopeReady).toBe(false)
    await act(async () => result.current.submit())
    expect(mocks.submit).not.toHaveBeenCalled()
  })

  it("ignores a late first scope Retry after a second Retry confirms current authority", async () => {
    mocks.getConfig.mockRejectedValueOnce(new Error("configuration unavailable"))
    const module = await loadSubject()
    const hook = renderHook(() => module.useStandaloneHtmlGeneration({
      capability: capability as any,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))

    await waitFor(() => expect(hook.result.current.scopeError).toBe(
      "Current server and account could not be confirmed."
    ))

    const retiredDraft = {
      schemaVersion: 1,
      timestamp: Date.now(),
      values: { ...validDraft, source: "Retired authority source" },
      generationConfigRevision: capability.generation_config_revision
    }
    sessionStorage.setItem(
      module.buildStandaloneHtmlStorageKeys({
        serverOrigin: "https://retired.example",
        principalId: "84"
      }).draft,
      JSON.stringify(retiredDraft)
    )

    let resolveRetiredConfig: ((value: unknown) => void) | undefined
    let resolveRetiredUser: ((value: unknown) => void) | undefined
    mocks.getConfig
      .mockReturnValueOnce(new Promise((resolve) => { resolveRetiredConfig = resolve }))
      .mockResolvedValueOnce({ serverUrl: "https://tldw.example/base" })
    mocks.getCurrentUser
      .mockReturnValueOnce(new Promise((resolve) => { resolveRetiredUser = resolve }))
      .mockResolvedValueOnce({ id: 42 })

    let firstRetry!: Promise<void>
    act(() => { firstRetry = hook.result.current.retryScope() })
    expect(hook.result.current.scopeReady).toBe(false)
    expect(hook.result.current.draft.source).toBe("")

    await act(async () => hook.result.current.retryScope())
    expect(hook.result.current.scopeReady).toBe(true)
    expect(hook.result.current.draft.source).toBe("")

    await act(async () => {
      resolveRetiredConfig?.({ serverUrl: "https://retired.example/base" })
      resolveRetiredUser?.({ id: 84 })
      await firstRetry
    })
    expect(hook.result.current.scopeReady).toBe(true)
    expect(hook.result.current.draft.source).toBe("")
    expect(mocks.getConfig).toHaveBeenCalledTimes(3)
    expect(mocks.getCurrentUser).toHaveBeenCalledTimes(3)
    hook.unmount()
  })

  it("keeps the exact in-memory draft across repeated same-scope retries when storage is unavailable", async () => {
    const setItem = vi.spyOn(
      Object.getPrototypeOf(window.sessionStorage) as Storage,
      "setItem"
    ).mockImplementation(() => {
      throw new DOMException("quota unavailable", "QuotaExceededError")
    })
    const hook = await setup()
    const latestSource = "Latest source after quota failure"
    act(() => expect(hook.result.current.updateField(
      "source",
      latestSource
    )).toBe(true))
    await waitFor(() => expect(hook.result.current.storageWarning).toBe(
      "Reload recovery is unavailable."
    ))

    let resolveFirstConfig: ((value: unknown) => void) | undefined
    let resolveFirstUser: ((value: unknown) => void) | undefined
    mocks.getConfig
      .mockReturnValueOnce(new Promise((resolve) => { resolveFirstConfig = resolve }))
      .mockResolvedValueOnce({ serverUrl: "https://tldw.example/base" })
    mocks.getCurrentUser
      .mockReturnValueOnce(new Promise((resolve) => { resolveFirstUser = resolve }))
      .mockResolvedValueOnce({ id: 42 })

    let firstRetry!: Promise<void>
    act(() => { firstRetry = hook.result.current.retryScope() })
    expect(hook.result.current.scopeReady).toBe(false)
    expect(hook.result.current.draft.source).toBe("")

    await act(async () => hook.result.current.retryScope())
    expect(hook.result.current.scopeReady).toBe(true)
    expect(hook.result.current.draft.source).toBe(latestSource)

    await act(async () => {
      resolveFirstConfig?.({ serverUrl: "https://tldw.example/base" })
      resolveFirstUser?.({ id: 42 })
      await firstRetry
    })
    expect(hook.result.current.draft.source).toBe(latestSource)
    expect(hook.result.current.storageWarning).toBe("Reload recovery is unavailable.")
    setItem.mockRestore()
    hook.unmount()
  })

  it.each([
    {
      name: "failed",
      submitResult: {
        ...pendingReceipt,
        status: "failed",
        progress_text: "Provider reached terminal state",
        error_code: "provider_failed",
        error_message: "Provider failed"
      },
      phase: "failed",
      backendStatus: "failed",
      progressText: "Provider reached terminal state",
      safeError: "provider_failed"
    },
    {
      name: "ambiguous",
      submitResult: Object.assign(new Error("Network error"), { status: 0 }),
      phase: "ambiguous",
      backendStatus: null,
      progressText: null,
      safeError: "generation_submission_unknown"
    }
  ])("keeps $name authority across an immediate second same-scope boundary", async ({
    name,
    submitResult,
    phase,
    backendStatus,
    progressText,
    safeError
  }) => {
    if (name === "ambiguous") mocks.submit.mockRejectedValue(submitResult)
    else mocks.submit.mockResolvedValue(submitResult)
    const hook = await setup()
    await act(async () => hook.result.current.submit())
    await waitFor(() => expect(hook.result.current.phase).toBe(phase))
    expect(hook.result.current.draftRecoveryAvailable).toBe(true)

    mocks.getConfig.mockRejectedValueOnce(new Error("scope temporarily unavailable"))
    await act(async () => hook.result.current.retryScope())
    expect(hook.result.current.scopeReady).toBe(false)

    await act(async () => {
      await hook.result.current.retryScope()
      await hook.result.current.retryScope()
    })

    expect(hook.result.current.scopeReady).toBe(true)
    expect(hook.result.current.phase).toBe(phase)
    expect(hook.result.current.backendStatus).toBe(backendStatus)
    expect(hook.result.current.progressText).toBe(progressText)
    expect(hook.result.current.safeError).toBe(safeError)
    expect(hook.result.current.snapshot).not.toBeNull()
    expect(hook.result.current.recoveryAvailable).toBe(true)
    expect(hook.result.current.draftRecoveryAvailable).toBe(true)
    expect(mocks.submit).toHaveBeenCalledTimes(1)
    hook.unmount()
  })

  it("keeps same-scope in-memory source when the sessionStorage getter throws and scrubs it on mismatch", async () => {
    const descriptor = Object.getOwnPropertyDescriptor(window, "sessionStorage")
    Object.defineProperty(window, "sessionStorage", {
      configurable: true,
      get: () => {
        throw new DOMException("storage unavailable", "SecurityError")
      }
    })
    try {
      const module = await loadSubject()
      const hook = renderHook(() => module.useStandaloneHtmlGeneration({
        capability: capability as any,
        onCompleted: mocks.onCompleted,
        onStopWaiting: mocks.onStopWaiting
      }))
      await waitFor(() => expect(hook.result.current.scopeReady).toBe(true))
      act(() => hook.result.current.updateField("source", "Getter-failed current source"))
      expect(hook.result.current.storageWarning).toBe("Reload recovery is unavailable.")

      await act(async () => hook.result.current.retryScope())
      expect(hook.result.current.scopeReady).toBe(true)
      expect(hook.result.current.draft.source).toBe("Getter-failed current source")

      act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))
      expect(hook.result.current.scopeReady).toBe(false)
      expect(hook.result.current.draft.source).toBe("")
      hook.unmount()
    } finally {
      if (descriptor) Object.defineProperty(window, "sessionStorage", descriptor)
      else Reflect.deleteProperty(window, "sessionStorage")
    }
  })

  it("pauses quarantined polling and Resume continues status without another provider submission", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockResolvedValue({
      receipt: { ...pendingReceipt, status: "running" },
      retryAfterMs: 10_000
    })
    const hook = await setup()
    const setItem = vi.spyOn(
      Object.getPrototypeOf(window.sessionStorage) as Storage,
      "setItem"
    ).mockImplementation(() => {
      throw new DOMException("quota unavailable", "QuotaExceededError")
    })
    await act(async () => hook.result.current.submit())
    await waitFor(() => expect(hook.result.current.phase).toBe("polling"))
    await waitFor(() => expect(mocks.status).toHaveBeenCalledTimes(1))

    await act(async () => hook.result.current.retryScope())
    expect(hook.result.current.phase).toBe("stopped")
    expect(hook.result.current.draft.source).toBe("Bounded source")
    expect(hook.result.current.snapshot).not.toBeNull()
    expect(hook.result.current.recoveryAvailable).toBe(true)

    await act(async () => hook.result.current.resume())
    expect(mocks.submit).toHaveBeenCalledTimes(1)
    expect(mocks.status).toHaveBeenCalledTimes(2)
    setItem.mockRestore()
    hook.unmount()
  })

  it("replays a quarantined pre-receipt attempt with the exact idempotency key and request", async () => {
    mocks.submit
      .mockRejectedValueOnce(Object.assign(new Error("Network error"), { status: 0 }))
      .mockResolvedValueOnce({
        ...pendingReceipt,
        status: "failed",
        error_code: "provider_failed",
        error_message: "Provider failed"
      })
    const hook = await setup()
    vi.spyOn(
      Object.getPrototypeOf(window.sessionStorage) as Storage,
      "setItem"
    ).mockImplementation(() => {
      throw new DOMException("quota unavailable", "QuotaExceededError")
    })

    await act(async () => hook.result.current.submit())
    await waitFor(() => expect(hook.result.current.phase).toBe("ambiguous"))
    const [firstRequest, firstOptions] = mocks.submit.mock.calls[0]

    await act(async () => hook.result.current.retryScope())
    expect(hook.result.current.phase).toBe("ambiguous")
    expect(hook.result.current.recoveryAvailable).toBe(true)

    await act(async () => hook.result.current.resume())
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(2))
    const [secondRequest, secondOptions] = mocks.submit.mock.calls[1]
    expect(secondRequest).toEqual(firstRequest)
    expect(secondOptions.idempotencyKey).toBe(firstOptions.idempotencyKey)
    hook.unmount()
  })

  it("captures and persists an immutable canonical snapshot before POST with a random URL-safe key", async () => {
    let resolveSubmit: ((value: unknown) => void) | undefined
    mocks.submit.mockReturnValue(new Promise((resolve) => { resolveSubmit = resolve }))
    const randomSpy = vi.spyOn(globalThis.crypto, "getRandomValues").mockImplementation(((array: Uint8Array) => {
      array.forEach((_value, index) => { array[index] = index })
      return array
    }) as any)
    const { result } = await setup()

    let submitting!: Promise<void>
    act(() => { submitting = result.current.submit() })
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    act(() => { void result.current.submit() })
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    expect(result.current.locked).toBe(true)
    expect(result.current.snapshot).toEqual({
      generation_mode: "standalone_html",
      generation_config_revision: capability.generation_config_revision,
      source: { kind: "prompt", prompt: "Bounded source" },
      html_options: { presentation_type: "tech-sharing", audience: "Engineers", slide_count: 8, visual_direction: "dark-technical", delivery_style: "speaker-led" }
    })
    const [sentRequest, options] = mocks.submit.mock.calls[0]
    expect(sentRequest).toEqual(result.current.snapshot)
    expect(options.idempotencyKey).toMatch(/^[A-Za-z0-9._~-]{16,200}$/)
    expect(Array.from({ length: sessionStorage.length }, (_, index) => sessionStorage.key(index))).not.toContain(options.idempotencyKey)

    resolveSubmit?.({ ...pendingReceipt, status: "failed", error_code: "provider_failed", error_message: "Provider failed" })
    await act(async () => submitting)
    randomSpy.mockRestore()
  })

  it("unlocks a definitive pre-admission rejection and gives the corrected request a new key", async () => {
    mocks.submit
      .mockRejectedValueOnce(Object.assign(new Error("rejected"), {
        status: 422,
        details: { error_code: "generation_request_invalid" }
      }))
      .mockResolvedValueOnce({ ...pendingReceipt, status: "failed", error_code: "provider_failed", error_message: "Provider failed" })
    const { result } = await setup()

    await act(async () => result.current.submit())
    const rejectedKey = mocks.submit.mock.calls[0][1].idempotencyKey
    expect(result.current.phase).toBe("rejected")
    expect(result.current.safeError).toBe("generation_request_invalid")
    expect(result.current.snapshot).toBeNull()
    expect(result.current.locked).toBe(false)

    act(() => result.current.updateField("source", "Corrected source"))
    await act(async () => result.current.submit())
    expect(mocks.submit.mock.calls[1][1].idempotencyKey).not.toBe(rejectedKey)
    expect(mocks.submit.mock.calls[1][0].source.prompt).toBe("Corrected source")
  })

  it("replays an ambiguous pre-202 outcome with the exact request and key", async () => {
    const ambiguous = Object.assign(new Error("Network error"), { status: 0 })
    mocks.submit.mockRejectedValueOnce(ambiguous).mockResolvedValueOnce(pendingReceipt)
    const first = await setup()

    await act(async () => first.result.current.submit())
    expect(first.result.current.phase).toBe("ambiguous")
    const [firstRequest, firstOptions] = mocks.submit.mock.calls[0]
    first.unmount()

    const replay = renderHook(() => first.module.useStandaloneHtmlGeneration({
      capability: capability as any,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(replay.result.current.phase).toBe("stopped"))
    await act(async () => replay.result.current.resume())
    const [secondRequest, secondOptions] = mocks.submit.mock.calls[1]
    expect(secondRequest).toEqual(firstRequest)
    expect(secondOptions.idempotencyKey).toBe(firstOptions.idempotencyKey)
    expect(replay.result.current.locked).toBe(true)
    replay.unmount()
  })

  it("uses new keys only for confirmed different requests and terminal Try again", async () => {
    mocks.submit
      .mockRejectedValueOnce(Object.assign(new Error("Network error"), { status: 0 }))
      .mockResolvedValueOnce({ ...pendingReceipt, status: "failed", error_code: "generation_quarantined", error_message: "Quarantined" })
      .mockResolvedValueOnce(pendingReceipt)
    const { result } = await setup()

    await act(async () => result.current.submit())
    const firstKey = mocks.submit.mock.calls[0][1].idempotencyKey
    act(() => result.current.startDifferent())
    expect(result.current.locked).toBe(false)
    await act(async () => result.current.submit())
    const secondKey = mocks.submit.mock.calls[1][1].idempotencyKey
    expect(secondKey).not.toBe(firstKey)
    expect(result.current.phase).toBe("failed")

    await act(async () => result.current.tryAgain())
    const thirdKey = mocks.submit.mock.calls[2][1].idempotencyKey
    expect(thirdKey).not.toBe(secondKey)
  })

  it("polls only real states, honors bounded Retry-After, and hands off a completed presentation", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status
      .mockResolvedValueOnce({ receipt: { ...pendingReceipt, status: "running", progress_text: "Checking document" }, retryAfterMs: 50_000 })
      .mockResolvedValueOnce({ receipt: { generation_id: pendingReceipt.generation_id, status: "completed", status_url: pendingReceipt.status_url, presentation_id: "presentation-9", content_kind: "standalone_html" }, retryAfterMs: null })
    const { result } = await setup()
    vi.useFakeTimers()

    await act(async () => result.current.submit())
    expect(result.current.backendStatus).toBe("running")
    expect(result.current.progressText).toBe("Checking document")
    expect(mocks.onCompleted).not.toHaveBeenCalled()

    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(mocks.onCompleted).toHaveBeenCalledWith("presentation-9")
    expect(result.current.phase).toBe("completed")
    expect(sessionStorage.length).toBe(0)
  })

  it("Stop waiting and Forget remain local while Resume continues polling", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockReturnValue(new Promise(() => undefined))
    const { result } = await setup()
    await act(async () => result.current.submit())

    act(() => result.current.stopWaiting())
    expect(result.current.phase).toBe("stopped")
    expect(mocks.onStopWaiting).toHaveBeenCalledTimes(1)
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    act(() => result.current.forget())
    expect(result.current.recoveryAvailable).toBe(false)
    expect(sessionStorage.length).toBe(0)
    expect(mocks.status).toHaveBeenCalledTimes(1)
  })

  it("rehydrates an admitted job after reload and resumes status polling without another POST", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockReturnValue(new Promise(() => undefined))
    const first = await setup()
    await act(async () => first.result.current.submit())
    expect(first.result.current.recoveryAvailable).toBe(true)
    first.unmount()

    const second = renderHook(() => first.module.useStandaloneHtmlGeneration({
      capability: capability as any,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(second.result.current.phase).toBe("stopped"))
    await act(async () => { void second.result.current.resume() })
    expect(mocks.submit).toHaveBeenCalledTimes(1)
    expect(mocks.status).toHaveBeenCalledTimes(2)
    second.unmount()
  })

  it("keeps an existing receipt recoverable when current generation capability is unavailable", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockReturnValue(new Promise(() => undefined))
    const first = await setup()
    await act(async () => first.result.current.submit())
    first.result.current.stopWaiting()
    first.unmount()

    const second = renderHook(() => (first.module.useStandaloneHtmlGeneration as any)({
      capability: null,
      contentMaxSlides: 30,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(second.result.current.phase).toBe("stopped"))
    expect(second.result.current.recoveryAvailable).toBe(true)
    await act(async () => { void second.result.current.resume() })
    expect(mocks.status).toHaveBeenCalledTimes(2)
    second.unmount()
  })

  it("probes source-free recovery after scope confirmation and leaves records unread on outage", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockReturnValue(new Promise(() => undefined))
    const first = await setup()
    await act(async () => first.result.current.submit())
    first.result.current.stopWaiting()
    first.unmount()
    const keys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    const getSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "getItem")

    await expect((first.module as any).probeStandaloneHtmlRecovery()).resolves.toEqual({ kind: "resume" })
    expect(getSpy).toHaveBeenCalledWith(keys.resume)
    expect(getSpy).not.toHaveBeenCalledWith(keys.draft)

    sessionStorage.removeItem(keys.resume)
    getSpy.mockClear()
    await expect((first.module as any).probeStandaloneHtmlRecovery()).resolves.toEqual({ kind: "draft" })
    expect(getSpy).not.toHaveBeenCalledWith(keys.draft)

    getSpy.mockClear()
    mocks.getConfig.mockRejectedValue(new Error("temporary outage"))
    await expect((first.module as any).probeStandaloneHtmlRecovery()).resolves.toBeNull()
    expect(getSpy).not.toHaveBeenCalled()
    expect(sessionStorage.getItem(keys.draft)).not.toBeNull()
    getSpy.mockRestore()
  })

  it("retains a trusted recovery kind through storage outage and clears it on definitive none", async () => {
    const first = await setup()
    first.unmount()
    const keys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    const probe = renderHook(() => first.module.useStandaloneHtmlRecoveryProbe())
    await waitFor(() => expect(probe.result.current.status).toBe("available"))
    expect(probe.result.current.kind).toBe("draft")

    const keySpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "key")
      .mockImplementation(() => { throw new DOMException("storage unavailable", "SecurityError") })
    await act(async () => probe.result.current.retry())
    keySpy.mockRestore()

    expect(probe.result.current.status).toBe("unavailable")
    expect(probe.result.current.kind).toBe("draft")

    sessionStorage.removeItem(keys.draft)
    await act(async () => probe.result.current.retry())
    expect(probe.result.current.status).toBe("none")
    expect(probe.result.current.kind).toBeNull()
    probe.unmount()
  })

  it("clears a trusted recovery kind when a confirmed scope switch cannot inspect storage", async () => {
    const first = await setup()
    first.unmount()
    const oldKeys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    const probe = renderHook(() => first.module.useStandaloneHtmlRecoveryProbe())
    await waitFor(() => expect(probe.result.current.kind).toBe("draft"))

    mocks.getConfig.mockResolvedValue({ serverUrl: "https://other.example/base" })
    mocks.getCurrentUser.mockResolvedValue({ id: 77 })
    const keySpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "key")
      .mockImplementation(() => { throw new DOMException("storage unavailable", "SecurityError") })
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
    await waitFor(() => expect(probe.result.current.status).toBe("unavailable"))
    keySpy.mockRestore()

    expect(probe.result.current.kind).toBeNull()
    expect(sessionStorage.getItem(oldKeys.draft)).toBeNull()
    probe.unmount()
  })

  it("invalidates deferred recovery probing on logout and deletes its last trusted namespace", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockReturnValue(new Promise(() => undefined))
    const first = await setup()
    await act(async () => first.result.current.submit())
    first.result.current.stopWaiting()
    first.unmount()
    const keys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    const probe = renderHook(() => first.module.useStandaloneHtmlRecoveryProbe())
    await waitFor(() => expect(probe.result.current.status).toBe("available"))

    let resolveOldScope: ((value: unknown) => void) | undefined
    mocks.getConfig.mockReturnValue(new Promise((resolve) => { resolveOldScope = resolve }))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
    await waitFor(() => expect(probe.result.current.status).toBe("checking"))
    const callsBeforeLogout = mocks.getConfig.mock.calls.length
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "logout" }
    })))

    expect(probe.result.current.status).toBe("unavailable")
    expect(probe.result.current.kind).toBeNull()
    expect(mocks.getConfig).toHaveBeenCalledTimes(callsBeforeLogout)
    expect(sessionStorage.getItem(keys.draft)).toBeNull()
    expect(sessionStorage.getItem(keys.resume)).toBeNull()

    resolveOldScope?.({ serverUrl: "https://tldw.example/base" })
    await act(async () => Promise.resolve())
    expect(probe.result.current.status).toBe("unavailable")
    probe.unmount()
  })

  it.each([
    [401, "auth_lost", "stopped", null],
    [404, "missing", "missing", "generation_not_found"],
    [429, "throttled", "stopped", "generation_status_throttled"],
    [503, "outage", "stopped", "generation_status_unavailable"]
  ])("retains recoverable form state for polling HTTP %s", async (
    status,
    phase,
    restoredPhase,
    restoredError
  ) => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockRejectedValue(Object.assign(new Error("safe failure"), { status }))
    const { result } = await setup()
    await act(async () => result.current.submit())

    await waitFor(() => expect(result.current.phase).toBe(phase))
    expect(result.current.snapshot?.source.kind === "prompt" ? result.current.snapshot.source.prompt : null).toBe("Bounded source")
    expect(result.current.recoveryAvailable).toBe(true)
    const submitCalls = mocks.submit.mock.calls.length
    const statusCalls = mocks.status.mock.calls.length

    await act(async () => result.current.retryScope())

    expect(result.current.phase).toBe(restoredPhase)
    expect(result.current.safeError).toBe(restoredError)
    expect(result.current.snapshot?.source.kind === "prompt" ? result.current.snapshot.source.prompt : null).toBe("Bounded source")
    expect(result.current.recoveryAvailable).toBe(true)
    expect(result.current.draftRecoveryAvailable).toBe(true)
    expect(mocks.submit).toHaveBeenCalledTimes(submitCalls)
    expect(mocks.status).toHaveBeenCalledTimes(statusCalls)
  })

  it.each([429, 503])("retries transient polling HTTP %s with bounded backoff", async (status) => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status
      .mockRejectedValueOnce(Object.assign(new Error("temporary"), { status, retryAfterMs: 60_000 }))
      .mockResolvedValueOnce({
        receipt: {
          ...pendingReceipt,
          status: "completed",
          presentation_id: "presentation-after-retry",
          content_kind: "standalone_html"
        },
        retryAfterMs: null
      })
    const { result } = await setup()
    vi.useFakeTimers()
    await act(async () => result.current.submit())
    expect(result.current.phase).toBe(status === 429 ? "throttled" : "outage")

    await act(async () => { await vi.advanceTimersByTimeAsync(10_000) })
    expect(mocks.status).toHaveBeenCalledTimes(2)
    expect(mocks.onCompleted).toHaveBeenCalledWith("presentation-after-retry")
  })

  it.each([
    [{ ...pendingReceipt, status: "cancelled", error_code: "generation_cancelled" }, "cancelled"],
    [{ ...pendingReceipt, status: "failed", error_code: "generation_quarantined", error_message: "Quarantined" }, "failed"],
    [{ ...pendingReceipt, status: "completed", presentation_id: null, content_kind: "standalone_html" }, "completed_missing_binding"]
  ])("preserves the draft for terminal receipt state %#", async (receipt, phase) => {
    mocks.submit.mockResolvedValue(receipt)
    const { result } = await setup()
    await act(async () => result.current.submit())
    expect(result.current.phase).toBe(phase)
    expect(result.current.draft.source).toBe("Bounded source")
    const submitCalls = mocks.submit.mock.calls.length
    const statusCalls = mocks.status.mock.calls.length

    await act(async () => result.current.retryScope())

    expect(result.current.phase).toBe(phase)
    expect(result.current.draft.source).toBe("Bounded source")
    expect(result.current.snapshot).not.toBeNull()
    expect(result.current.recoveryAvailable).toBe(true)
    expect(result.current.draftRecoveryAvailable).toBe(true)
    expect(mocks.submit).toHaveBeenCalledTimes(submitCalls)
    expect(mocks.status).toHaveBeenCalledTimes(statusCalls)
  })

  it("normalizes a quarantined in-flight submission to ambiguous without automatic replay", async () => {
    let resolveSubmit: ((value: unknown) => void) | undefined
    mocks.submit.mockReturnValue(new Promise((resolve) => { resolveSubmit = resolve }))
    const hook = await setup()

    await act(async () => hook.result.current.submit())
    await waitFor(() => expect(hook.result.current.phase).toBe("submitting"))
    expect(mocks.submit).toHaveBeenCalledTimes(1)

    await act(async () => hook.result.current.retryScope())

    expect(hook.result.current.phase).toBe("ambiguous")
    expect(hook.result.current.snapshot).not.toBeNull()
    expect(hook.result.current.recoveryAvailable).toBe(true)
    expect(mocks.submit).toHaveBeenCalledTimes(1)
    resolveSubmit?.(pendingReceipt)
    await act(async () => Promise.resolve())
    expect(hook.result.current.phase).toBe("ambiguous")
    expect(mocks.submit).toHaveBeenCalledTimes(1)
    hook.unmount()
  })

  it("uses principal and canonical origin scoped 24-hour records and expires invalid recovery before reading values", async () => {
    const module = await loadSubject()
    const keys = module.buildStandaloneHtmlStorageKeys({ serverOrigin: "https://tldw.example", principalId: "42" })
    sessionStorage.setItem(keys.draft, JSON.stringify({ schemaVersion: 1, timestamp: Date.now() - 86_400_001, values: { ...validDraft, source: "EXPIRED SECRET" }, generationConfigRevision: capability.generation_config_revision }))
    sessionStorage.setItem(keys.resume, JSON.stringify({ generationId: null, idempotencyKey: "A".repeat(24), requestDigest: "bad", timestamp: Date.now() }))

    const { result } = renderHook(() => module.useStandaloneHtmlGeneration({ capability: capability as any, onCompleted: mocks.onCompleted, onStopWaiting: mocks.onStopWaiting }))
    await waitFor(() => expect(result.current.scopeReady).toBe(true))
    expect(result.current.draft.source).toBe("")
    expect(result.current.recoveryAvailable).toBe(false)
    expect(sessionStorage.getItem(keys.draft)).toBeNull()
    expect(sessionStorage.getItem(keys.resume)).toBeNull()
  })

  it("keeps in-memory work usable and warns when sessionStorage fails", async () => {
    const setSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "setItem").mockImplementation(() => { throw new DOMException("quota", "QuotaExceededError") })
    const { result } = await setup()
    act(() => result.current.updateField("source", "still in memory"))
    expect(result.current.draft.source).toBe("still in memory")
    expect(setSpy).toHaveBeenCalled()
    await waitFor(() => expect(result.current.storageWarning).toBe("Reload recovery is unavailable."))
    setSpy.mockRestore()
  })

  it("keeps ambiguous replay resumable in memory when sessionStorage quota fails", async () => {
    const setSpy = vi.spyOn(Object.getPrototypeOf(window.sessionStorage) as Storage, "setItem")
      .mockImplementation(() => { throw new DOMException("quota", "QuotaExceededError") })
    mocks.submit
      .mockRejectedValueOnce(Object.assign(new Error("network"), { status: 0 }))
      .mockResolvedValueOnce(pendingReceipt)
    mocks.status.mockReturnValue(new Promise(() => undefined))
    const { result } = await setup()

    await act(async () => result.current.submit())
    const [request, options] = mocks.submit.mock.calls[0]
    expect(result.current.phase).toBe("ambiguous")
    expect(result.current.recoveryAvailable).toBe(true)

    await act(async () => result.current.resume())
    expect(mocks.submit.mock.calls[1][0]).toEqual(request)
    expect(mocks.submit.mock.calls[1][1].idempotencyKey).toBe(options.idempotencyKey)
    setSpy.mockRestore()
  })

  it("preserves the draft but removes replay metadata after a definitive 422", async () => {
    mocks.submit.mockRejectedValue(Object.assign(new Error("invalid"), {
      status: 422,
      details: { error_code: "generation_request_invalid" }
    }))
    const first = await setup()
    await act(async () => first.result.current.submit())
    expect(first.result.current.phase).toBe("rejected")
    expect(storedValues()).toContain("Bounded source")
    expect(storedValues()).not.toContain(pendingReceipt.generation_id)

    await act(async () => first.result.current.retryScope())
    expect(first.result.current.phase).toBe("rejected")
    expect(first.result.current.safeError).toBe("generation_request_invalid")
    expect(first.result.current.draft.source).toBe("Bounded source")
    expect(first.result.current.snapshot).toBeNull()
    expect(first.result.current.recoveryAvailable).toBe(false)
    expect(first.result.current.draftRecoveryAvailable).toBe(true)
    expect(mocks.submit).toHaveBeenCalledTimes(1)
    first.unmount()

    const second = renderHook(() => first.module.useStandaloneHtmlGeneration({
      capability: null,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(second.result.current.scopeReady).toBe(true))
    expect(second.result.current.draft.source).toBe("Bounded source")
    expect(second.result.current.recoveryAvailable).toBe(false)
    expect(second.result.current.draftRecoveryAvailable).toBe(true)
    second.unmount()
  })

  it("keeps a terminally edited draft across reload without digest-deleting it", async () => {
    mocks.submit.mockResolvedValue({
      ...pendingReceipt,
      status: "failed",
      error_code: "provider_failed",
      error_message: "Provider failed"
    })
    const first = await setup()
    await act(async () => first.result.current.submit())
    act(() => first.result.current.updateField("source", "Edited after failure"))
    first.unmount()

    const second = renderHook(() => first.module.useStandaloneHtmlGeneration({
      capability: null,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(second.result.current.scopeReady).toBe(true))
    expect(second.result.current.draft.source).toBe("Edited after failure")
    expect(second.result.current.recoveryAvailable).toBe(false)
    expect(second.result.current.draftRecoveryAvailable).toBe(true)
    second.unmount()
  })

  it("rejects stale POST completion after principal and origin replacement", async () => {
    let resolveSubmit: ((value: unknown) => void) | undefined
    mocks.submit.mockReturnValue(new Promise((resolve) => { resolveSubmit = resolve }))
    const { result } = await setup()
    act(() => { void result.current.submit() })
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))

    mocks.getConfig.mockResolvedValue({ serverUrl: "https://other.example" })
    mocks.getCurrentUser.mockResolvedValue({ id: 77 })
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")))
    await waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(result.current.scopeReady).toBe(true))
    resolveSubmit?.({
      ...pendingReceipt,
      status: "completed",
      presentation_id: "old-principal-presentation",
      content_kind: "standalone_html"
    })
    await act(async () => Promise.resolve())

    expect(mocks.onCompleted).not.toHaveBeenCalled()
    expect(result.current.snapshot).toBeNull()
    expect(storedValues()).not.toContain("old-principal-presentation")
    expect(sessionStorage.length).toBe(0)
  })

  it("rejects stale poll completion after an auth boundary", async () => {
    let resolveStatus: ((value: unknown) => void) | undefined
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockReturnValue(new Promise((resolve) => { resolveStatus = resolve }))
    const { result } = await setup()
    await act(async () => result.current.submit())

    mocks.getCurrentUser.mockResolvedValue({ id: 77 })
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed")))
    await waitFor(() => expect(mocks.getConfig).toHaveBeenCalledTimes(2))
    await waitFor(() => expect(result.current.scopeReady).toBe(true))
    resolveStatus?.({
      receipt: {
        ...pendingReceipt,
        status: "completed",
        presentation_id: "old-poll-presentation",
        content_kind: "standalone_html"
      },
      retryAfterMs: null
    })
    await act(async () => Promise.resolve())

    expect(mocks.onCompleted).not.toHaveBeenCalled()
    expect(storedValues()).not.toContain("old-poll-presentation")
    expect(result.current.snapshot).toBeNull()
  })

  it("logout invalidates deferred old-session scope work and deletes the last trusted namespace", async () => {
    const first = await setup()
    const oldKeys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    expect(sessionStorage.getItem(oldKeys.draft)).toContain("Bounded source")

    let resolveOldScope: ((value: unknown) => void) | undefined
    mocks.getConfig.mockReturnValue(new Promise((resolve) => { resolveOldScope = resolve }))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
    const callsBeforeLogout = mocks.getConfig.mock.calls.length
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "logout" }
    })))

    expect(first.result.current.draft.source).toBe("")
    expect(sessionStorage.getItem(oldKeys.draft)).toBeNull()
    expect(sessionStorage.getItem(oldKeys.resume)).toBeNull()
    expect(mocks.getConfig).toHaveBeenCalledTimes(callsBeforeLogout)

    resolveOldScope?.({ serverUrl: "https://tldw.example/base" })
    await act(async () => Promise.resolve())
    expect(first.result.current.draft.source).toBe("")
    expect(sessionStorage.getItem(oldKeys.draft)).toBeNull()
  })

  it("deletes the last trusted namespace when a switch follows a verification outage", async () => {
    const first = await setup()
    const oldKeys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    mocks.getConfig.mockRejectedValueOnce(new Error("temporary verification outage"))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
    await waitFor(() => expect(first.result.current.scopeError).toBe("Current server and account could not be confirmed."))
    expect(sessionStorage.getItem(oldKeys.draft)).not.toBeNull()

    mocks.getConfig.mockResolvedValue({ serverUrl: "https://other.example/base" })
    mocks.getCurrentUser.mockResolvedValue({ id: 77, username: "other", is_active: true })
    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "login" }
    })))
    await waitFor(() => expect(first.result.current.scopeReady).toBe(true))
    expect(sessionStorage.getItem(oldKeys.draft)).toBeNull()
    expect(sessionStorage.getItem(oldKeys.resume)).toBeNull()
  })

  it("deletes the last trusted namespace when logout follows a verification outage", async () => {
    const first = await setup()
    const oldKeys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    mocks.getConfig.mockRejectedValue(new Error("temporary verification outage"))
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
    await waitFor(() => expect(first.result.current.scopeError).toBe("Current server and account could not be confirmed."))
    expect(sessionStorage.getItem(oldKeys.draft)).not.toBeNull()
    const callsBeforeLogout = mocks.getConfig.mock.calls.length

    act(() => window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "logout" }
    })))
    expect(sessionStorage.getItem(oldKeys.draft)).toBeNull()
    expect(sessionStorage.getItem(oldKeys.resume)).toBeNull()
    expect(mocks.getConfig).toHaveBeenCalledTimes(callsBeforeLogout)
    expect(first.result.current.draft.source).toBe("")
  })

  it("scrubs hydrated source and removes the trusted namespace on a confirmed Slides scope mismatch", async () => {
    const first = await setup()
    const oldKeys = first.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })
    expect(sessionStorage.getItem(oldKeys.draft)).toContain("Bounded source")
    const callsBeforeMismatch = mocks.getConfig.mock.calls.length

    act(() => window.dispatchEvent(new CustomEvent("tldw:slides-scope-mismatch")))

    expect(first.result.current.draft.source).toBe("")
    expect(first.result.current.scopeReady).toBe(false)
    expect(sessionStorage.getItem(oldKeys.draft)).toBeNull()
    expect(sessionStorage.getItem(oldKeys.resume)).toBeNull()
    expect(mocks.getConfig).toHaveBeenCalledTimes(callsBeforeMismatch)
  })

  it("invalidates deferred POST work on pagehide while preserving pre-admission recovery", async () => {
    let resolveSubmit: ((value: unknown) => void) | undefined
    mocks.submit.mockReturnValue(new Promise((resolve) => { resolveSubmit = resolve }))
    const { result } = await setup()
    act(() => { void result.current.submit() })
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))

    act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
    expect(result.current.snapshot).toBeNull()
    expect((mocks.submit.mock.calls[0][1] as { abortSignal?: AbortSignal }).abortSignal?.aborted).toBe(true)
    resolveSubmit?.({
      ...pendingReceipt,
      status: "completed",
      presentation_id: "late-bfcache-presentation",
      content_kind: "standalone_html"
    })
    await act(async () => Promise.resolve())
    expect(mocks.onCompleted).not.toHaveBeenCalled()
    expect(storedValues()).not.toContain("late-bfcache-presentation")
  })

  it("commits a source-free submitted state during pagehide before a bfcache snapshot can retain it", async () => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockReturnValue(new Promise(() => undefined))
    const hook = await setup()
    await act(async () => hook.result.current.submit())
    await waitFor(() => expect(hook.result.current.snapshot).not.toBeNull())
    const keys = hook.module.buildStandaloneHtmlStorageKeys({
      serverOrigin: "https://tldw.example",
      principalId: "42"
    })

    let observedSnapshot: unknown = "not observed"
    let observedSource = "not observed"
    let observedPhase = "not observed"
    const observePagehide = () => {
      observedSnapshot = hook.result.current.snapshot
      observedSource = hook.result.current.draft.source
      observedPhase = hook.result.current.phase
    }
    window.addEventListener("pagehide", observePagehide)
    act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
    window.removeEventListener("pagehide", observePagehide)

    expect(observedSnapshot).toBeNull()
    expect(observedSource).toBe("")
    expect(observedPhase).toBe("idle")
    expect(sessionStorage.getItem(keys.draft)).toContain("Bounded source")
    expect(sessionStorage.getItem(keys.resume)).not.toBeNull()

    mocks.getCurrentUser.mockResolvedValue({ id: 84, username: "other", is_active: true })
    act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))
    await waitFor(() => expect(hook.result.current.scopeReady).toBe(true))
    expect(hook.result.current.snapshot).toBeNull()
    expect(hook.result.current.draft.source).toBe("")
    expect(sessionStorage.getItem(keys.draft)).toBeNull()
    expect(sessionStorage.getItem(keys.resume)).toBeNull()
  })

  it("scrubs refs and ignores late POST completion after unmount", async () => {
    let resolveSubmit: ((value: unknown) => void) | undefined
    mocks.submit.mockReturnValue(new Promise((resolve) => { resolveSubmit = resolve }))
    const hook = await setup()
    act(() => { void hook.result.current.submit() })
    await waitFor(() => expect(mocks.submit).toHaveBeenCalledTimes(1))
    const before = storedValues()
    hook.unmount()
    expect((mocks.submit.mock.calls[0][1] as { abortSignal?: AbortSignal }).abortSignal?.aborted).toBe(true)

    resolveSubmit?.({
      ...pendingReceipt,
      status: "completed",
      presentation_id: "late-unmounted-presentation",
      content_kind: "standalone_html"
    })
    await act(async () => Promise.resolve())
    expect(mocks.onCompleted).not.toHaveBeenCalled()
    expect(storedValues()).toBe(before)
  })

  it("counts Unicode scalar values and clamps slides to the content capability", async () => {
    const module = await loadSubject()
    const effectiveCapability = withLimits({ max_source_chars: 2 })
    const { result } = renderHook(() => (module.useStandaloneHtmlGeneration as any)({
      capability: effectiveCapability,
      contentMaxSlides: 5,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(result.current.scopeReady).toBe(true))

    act(() => expect(result.current.updateField("source", "😀😀")).toBe(true))
    act(() => expect(result.current.updateField("source", "😀😀😀")).toBe(false))
    act(() => expect(result.current.updateField("slideCount", 6)).toBe(false))
    expect(result.current.fieldErrors.slideCount).toContain("1 to 5")
  })

  it("enforces canonical request UTF-8 bytes before POST", async () => {
    const module = await loadSubject()
    const { result } = renderHook(() => module.useStandaloneHtmlGeneration({
      capability: withLimits({ max_request_bytes: 150 }) as any,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(result.current.scopeReady).toBe(true))
    act(() => result.current.replaceDraft(validDraft as any))

    await act(async () => result.current.submit())
    expect(mocks.submit).not.toHaveBeenCalled()
    expect(result.current.editError).toBe("Request exceeds the 150 byte limit.")
  })

  it("routes terminal Try again through nonblank validation", async () => {
    mocks.submit.mockResolvedValueOnce({
      ...pendingReceipt,
      status: "failed",
      error_code: "provider_failed",
      error_message: "Provider failed"
    })
    const { result } = await setup()
    await act(async () => result.current.submit())
    act(() => result.current.updateField("source", ""))

    await act(async () => result.current.tryAgain())
    expect(mocks.submit).toHaveBeenCalledTimes(1)
  })

  it("revalidates a terminal retry after the effective slide cap decreases", async () => {
    mocks.submit.mockResolvedValueOnce({
      ...pendingReceipt,
      status: "failed",
      error_code: "provider_failed",
      error_message: "Provider failed"
    })
    const module = await loadSubject()
    const { result, rerender } = renderHook(
      ({ contentMaxSlides }) => (module.useStandaloneHtmlGeneration as any)({
        capability,
        contentMaxSlides,
        onCompleted: mocks.onCompleted,
        onStopWaiting: mocks.onStopWaiting
      }),
      { initialProps: { contentMaxSlides: 30 } }
    )
    await waitFor(() => expect(result.current.scopeReady).toBe(true))
    act(() => result.current.replaceDraft(validDraft as any))
    await act(async () => result.current.submit())
    rerender({ contentMaxSlides: 5 })

    await act(async () => result.current.tryAgain())
    expect(mocks.submit).toHaveBeenCalledTimes(1)
    expect(result.current.fieldErrors.slideCount).toContain("1 to 5")
  })

  it("handles generation_configuration_changed as a non-replayable draft-preserving correction", async () => {
    const refreshed = vi.fn().mockResolvedValue({
      ...capability,
      generation_config_revision: `sha256:${"d".repeat(64)}`
    })
    mocks.submit.mockRejectedValue(Object.assign(new Error("changed"), {
      status: 409,
      details: { error_code: "generation_configuration_changed" }
    }))
    const module = await loadSubject()
    const { result } = renderHook(() => (module.useStandaloneHtmlGeneration as any)({
      capability,
      contentMaxSlides: 30,
      onCapabilitiesChanged: refreshed,
      onCompleted: mocks.onCompleted,
      onStopWaiting: mocks.onStopWaiting
    }))
    await waitFor(() => expect(result.current.scopeReady).toBe(true))
    act(() => result.current.replaceDraft(validDraft as any))

    await act(async () => result.current.submit())
    expect(result.current.phase).toBe("configuration_changed")
    expect(result.current.draft.source).toBe("Bounded source")
    expect(result.current.snapshot).toBeNull()
    expect(result.current.recoveryAvailable).toBe(false)
    expect(refreshed).toHaveBeenCalledTimes(1)

    await act(async () => result.current.retryScope())
    expect(result.current.phase).toBe("configuration_changed")
    expect(result.current.safeError).toBe("generation_configuration_changed")
    expect(result.current.draft.source).toBe("Bounded source")
    expect(result.current.snapshot).toBeNull()
    expect(result.current.recoveryAvailable).toBe(false)
    expect(result.current.draftRecoveryAvailable).toBe(true)
    expect(mocks.submit).toHaveBeenCalledTimes(1)
  })

  it("bounds receipt progress and never renders an arbitrary error payload", async () => {
    mocks.submit.mockResolvedValueOnce({
      ...pendingReceipt,
      status: "running",
      progress_text: "p".repeat(2_000)
    })
    mocks.status.mockResolvedValueOnce({
      receipt: {
        ...pendingReceipt,
        status: "failed",
        error_code: "PRIVATE SOURCE: do not render",
        error_message: "PRIVATE SOURCE: do not render"
      },
      retryAfterMs: null
    })
    const { result } = await setup()
    await act(async () => result.current.submit())

    expect(result.current.progressText).toBeNull()
    expect(result.current.safeError).toBe("generation_failed")
    expect(JSON.stringify(result.current)).not.toContain("PRIVATE SOURCE")
  })

  it("flushes then clears on pagehide, guardedly rehydrates on pageshow, and clears on principal/config change", async () => {
    const { result } = await setup()
    expect(result.current.draft.source).toBe("Bounded source")

    act(() => window.dispatchEvent(new PageTransitionEvent("pagehide", { persisted: true })))
    expect(result.current.draft.source).toBe("")

    act(() => window.dispatchEvent(new PageTransitionEvent("pageshow", { persisted: true })))
    await waitFor(() => expect(result.current.draft.source).toBe("Bounded source"))

    mocks.getCurrentUser.mockResolvedValue({ id: 77, username: "other", is_active: true })
    act(() => window.dispatchEvent(new CustomEvent("tldw:config-updated")))
    await waitFor(() => expect(result.current.draft.source).toBe(""))
    expect(result.current.snapshot).toBeNull()
  })
})
