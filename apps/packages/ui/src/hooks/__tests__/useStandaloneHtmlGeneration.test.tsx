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

describe("useStandaloneHtmlGeneration", () => {
  beforeEach(() => {
    vi.useRealTimers()
    vi.clearAllMocks()
    sessionStorage.clear()
    mocks.getConfig.mockResolvedValue({ serverUrl: "https://tldw.example/base", authMode: "multi-user", accessToken: "secret-token" })
    mocks.getCurrentUser.mockResolvedValue({ id: 42, username: "researcher", is_active: true })
  })

  afterEach(() => vi.useRealTimers())

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

  it.each([
    [401, "auth_lost"],
    [404, "missing"],
    [429, "throttled"],
    [503, "outage"]
  ])("retains recoverable form state for polling HTTP %s", async (status, phase) => {
    mocks.submit.mockResolvedValue(pendingReceipt)
    mocks.status.mockRejectedValue(Object.assign(new Error("safe failure"), { status }))
    const { result } = await setup()
    await act(async () => result.current.submit())

    await waitFor(() => expect(result.current.phase).toBe(phase))
    expect(result.current.snapshot?.source.prompt).toBe("Bounded source")
    expect(result.current.recoveryAvailable).toBe(true)
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
