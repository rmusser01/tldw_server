import React from "react"
import { act, cleanup, renderHook } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import fixture from "@/utils/__fixtures__/service-prompt-rendering.json"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

const mocks = vi.hoisted(() => ({
  load: vi.fn(), request: vi.fn(), configListeners: new Set<() => void>(),
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: (...args: unknown[]) => mocks.request(...args) }))
vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) => mocks.load(...args),
  subscribeToServicePromptConfigChanges: (listener: () => void) => {
    mocks.configListeners.add(listener)
    return () => mocks.configListeners.delete(listener)
  },
}))
vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: <T,>(_key: string, initial: T) => React.useState(initial),
}))
import { useWritingFeedback } from "../useWritingFeedback"

const requestScope = { config: { serverUrl: "https://server-a.test", authMode: "multi-user" as const }, userId: "owner-a" }
type FeedbackId = "writing.feedback.mood" | "writing.feedback.echo"
let scope: AbortController
let snapshots: ServicePromptSnapshot[]
let custom: boolean

function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => { resolve = yes; reject = no })
  return { promise, resolve, reject }
}
function setup() {
  return renderHook((editorText: string) => useWritingFeedback({
    editorText, isOnline: true, isGenerating: false, selectedModel: "test-model",
  }), { initialProps: "" })
}
async function tick(ms = 0) {
  await act(async () => { await vi.advanceTimersByTimeAsync(ms) })
}

beforeEach(() => {
  vi.useFakeTimers()
  vi.setSystemTime(60_000)
  vi.clearAllMocks()
  vi.spyOn(console, "warn").mockImplementation(() => {})
  scope = new AbortController()
  snapshots = []
  custom = true
  mocks.request.mockResolvedValue({ choices: [{ message: { content: "calm" } }] })
  mocks.load.mockImplementation(async ([id]: FeedbackId[], { signal }: { signal: AbortSignal }) => {
    const parts = custom
      ? Object.fromEntries(Object.keys(fixture.defaults[id]).map((key) => [key, `Custom ${key} {literal}`]))
      : fixture.defaults[id]
    const snapshot: ServicePromptSnapshot = {
      scopeKey: "scope-a", requestScope, capability: "supported",
      definitions: { [id]: { definition: { id, parts: [] }, parts, source: custom ? "user" : "packaged", revision: null } },
      scopeSignal: signal, scopeInvalidatedSignal: scope.signal, release: vi.fn(),
    }
    snapshots.push(snapshot)
    return snapshot
  })
})
afterEach(() => {
  cleanup()
  vi.useRealTimers()
})

describe("Writing feedback Service Prompts", () => {
  it.each(["prompt_lookup", "chat_generation"])("reports safe diagnostics for %s failures without logging private payloads", async (phase) => {
    const error = Object.assign(new Error("private prompt and credential"), { status: 403, body: "private response" })
    if (phase === "prompt_lookup") mocks.load.mockRejectedValueOnce(error)
    else mocks.request.mockRejectedValueOnce(error)
    const hook = setup()
    act(() => hook.result.current.setEchoEnabled(true))
    hook.rerender("secret".repeat(100))
    await tick()
    expect(console.warn).toHaveBeenCalledExactlyOnceWith("Writing feedback request failed", { kind: "echo", phase })
    expect(hook.result.current.echoReactions).toEqual([])
  })

  it.each(["cancelled", "scope_changed"])("does not warn for %s request control flow", async (reason) => {
    const response = deferred<unknown>()
    mocks.request.mockReturnValueOnce(response.promise)
    const hook = setup()
    act(() => hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick()
    if (reason === "cancelled") act(() => hook.result.current.setEchoEnabled(false))
    await act(async () => response.reject(reason === "cancelled"
      ? new DOMException("Aborted", "AbortError")
      : Object.assign(new Error("Scope changed"), { status: 412, details: { detail: { code: "request_config_scope_changed" } } })))
    expect(console.warn).not.toHaveBeenCalled()
  })

  it.each(["request_error", "empty", "invalid_mood"])("releases the new lease after %s without retaining invisible feedback", async (outcome) => {
    if (outcome === "request_error") mocks.request.mockRejectedValueOnce(new Error("Provider failed"))
    else mocks.request.mockResolvedValueOnce({ choices: [{ message: { content: outcome === "empty" ? "" : "angry" } }] })
    const hook = setup()
    act(() => outcome === "invalid_mood" ? hook.result.current.setMoodEnabled(true) : hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick(10_000)
    expect(snapshots[0].release).toHaveBeenCalledOnce()
    expect(hook.result.current.currentMood).toBeNull()
    expect(hook.result.current.echoReactions).toEqual([])
  })

  it.each(["invalid", "failed", "empty"])("keeps the existing visible-feedback lease when a replacement is %s", async (outcome) => {
    const hook = setup()
    act(() => hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick()
    if (outcome === "invalid") {
      const loader = mocks.load.getMockImplementation()!
      mocks.load.mockImplementationOnce(async (...args) => {
        const snapshot = await loader(...args)
        snapshot.definitions["writing.feedback.echo"].parts = {}
        return snapshot
      })
    } else if (outcome === "failed") mocks.request.mockRejectedValueOnce(new Error("Provider failed"))
    else mocks.request.mockResolvedValueOnce({ choices: [{ message: { content: "" } }] })
    await tick(30_000)
    hook.rerender("x".repeat(1000))
    await tick()
    expect(hook.result.current.echoReactions).toHaveLength(1)
    expect(snapshots[0].release).not.toHaveBeenCalled()
    expect(snapshots[1].release).toHaveBeenCalledOnce()
    act(() => scope.abort())
    expect(hook.result.current.echoReactions).toEqual([])
    expect(snapshots[0].release).toHaveBeenCalledOnce()
  })
  it("does not let a stale scope error clear feedback from a newer request", async () => {
    const response = deferred<unknown>()
    mocks.request.mockReturnValueOnce(response.promise)
    const hook = setup()
    act(() => hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick()
    act(() => scope.abort())
    scope = new AbortController()
    hook.rerender("x".repeat(1000))
    await tick()
    expect(hook.result.current.echoReactions).toHaveLength(1)
    await act(async () => response.reject(Object.assign(new Error("Scope changed"), {
      status: 412, details: { detail: { code: "request_config_scope_changed" } },
    })))
    expect(hook.result.current.echoReactions).toHaveLength(1)
    expect(snapshots.at(-1)!.release).not.toHaveBeenCalled()
  })

  it("releases completed feedback leases on unmount", async () => {
    const hook = setup()
    act(() => hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick()
    hook.unmount()
    expect(snapshots[0].release).toHaveBeenCalledOnce()
    expect(mocks.configListeners.size).toBe(0)
  })

  it.each(["writing.feedback.mood", "writing.feedback.echo"] as const)("does not dispatch an incomplete %s snapshot", async (id) => {
    const loader = mocks.load.getMockImplementation()!
    mocks.load.mockImplementationOnce(async (...args) => {
      const snapshot = await loader(...args)
      snapshot.definitions[id].parts = {}
      return snapshot
    })
    const hook = setup()
    act(() => id.endsWith("mood") ? hook.result.current.setMoodEnabled(true) : hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick(10_000)
    expect(mocks.request).not.toHaveBeenCalled()
    expect(snapshots[0].release).toHaveBeenCalledOnce()
  })
  it("does not look up prompts or generate while feedback is disabled", async () => {
    const hook = setup()
    hook.rerender("a".repeat(1000))
    await tick(60_000)
    expect(mocks.load).not.toHaveBeenCalled()
    expect(mocks.request).not.toHaveBeenCalled()
  })

  it.each([true, false])("sends mood guidance with locked output and capped passage (custom=%s)", async (useCustom) => {
    custom = useCustom
    const hook = setup()
    act(() => hook.result.current.setMoodEnabled(true))
    hook.rerender("prefix" + "m".repeat(500))
    await tick(10_000)
    expect(mocks.request).toHaveBeenCalledWith(expect.objectContaining({
      servicePromptConfig: { ...requestScope.config, expectedUserId: "owner-a" },
      headers: { "Content-Type": "application/json", "X-TLDW-Expected-User-ID": "owner-a" },
      abortSignal: snapshots[0].scopeSignal,
      body: {
        model: "test-model", temperature: 0.7, max_tokens: 100,
        messages: [
          { role: "system", content: (useCustom ? "Custom system_semantics {literal}" : "You are a mood classifier.") + " Respond with exactly one word." },
          { role: "user", content: (useCustom ? "Custom classification_semantics {literal}" : "Classify the emotional mood of this text.") + " Respond with ONLY one word from: tense, romantic, melancholic, action, calm, mysterious, humorous\n\nText: " + "m".repeat(500) },
        ],
      },
    }))
    expect(hook.result.current.currentMood).toBe("calm")
  })

  it("rejects values outside the fixed mood enum", async () => {
    mocks.request.mockResolvedValue({ choices: [{ message: { content: "angry" } }] })
    const hook = setup()
    act(() => hook.result.current.setMoodEnabled(true))
    hook.rerender("Some passage")
    await tick(10_000)
    expect(hook.result.current.currentMood).toBeNull()
    expect(hook.result.current.moodAnalyzing).toBe(false)
  })

  it.each([true, false])("rotates all five Echo instructions with unchanged identities and context (custom=%s)", async (useCustom) => {
    custom = useCustom
    const hook = setup()
    act(() => hook.result.current.setEchoEnabled(true))
    for (const [index, persona] of ["Alex", "Sam", "Max", "Riley", "Jordan", "Alex"].entries()) {
      if (index) await tick(30_000)
      hook.rerender("prefix" + "e".repeat(1000 + index * 500))
      await tick()
      const body = mocks.request.mock.calls.at(-1)![0].body
      expect(body).toEqual({
        model: "test-model", temperature: 0.7, max_tokens: 100,
        messages: [
          { role: "system", content: useCustom ? `Custom ${persona.toLowerCase()}_system {literal}` : fixture.defaults["writing.feedback.echo"][`${persona.toLowerCase()}_system` as keyof typeof fixture.defaults["writing.feedback.echo"]] },
          { role: "user", content: "React to this passage:\n\n" + "e".repeat(1000) },
        ],
      })
      expect(hook.result.current.echoReactions[0].persona).toBe(persona)
      expect(hook.result.current.charsSinceLastEcho).toBe(0)
    }
    expect(mocks.load).toHaveBeenCalledTimes(6)
    expect(snapshots.slice(0, -1).every((snapshot) => vi.mocked(snapshot.release).mock.calls.length === 1)).toBe(true)
  })

  it("does not send a default request after a failed prompt lookup", async () => {
    mocks.load.mockRejectedValue(new Error("Forbidden"))
    const hook = setup()
    act(() => { hook.result.current.setEchoEnabled(true); hook.result.current.setMoodEnabled(true) })
    hook.rerender("x".repeat(500))
    await tick(10_000)
    expect(mocks.request).not.toHaveBeenCalled()
    expect(hook.result.current.echoAnalyzing).toBe(false)
    expect(hook.result.current.moodAnalyzing).toBe(false)
  })

  it("clears both completed feedback channels on scope invalidation and releases leases", async () => {
    const hook = setup()
    act(() => { hook.result.current.setEchoEnabled(true); hook.result.current.setMoodEnabled(true) })
    hook.rerender("x".repeat(500))
    await tick(10_000)
    expect(hook.result.current.currentMood).toBe("calm")
    expect(hook.result.current.echoReactions).toHaveLength(1)
    act(() => scope.abort())
    expect(hook.result.current.currentMood).toBeNull()
    expect(hook.result.current.echoReactions).toEqual([])
    expect(hook.result.current.charsSinceLastEcho).toBe(0)
    expect(snapshots.every((snapshot) => vi.mocked(snapshot.release).mock.calls.length === 1)).toBe(true)
  })

  it.each(["mood", "echo"] as const)("discards a late %s response after account/server invalidation", async (kind) => {
    const response = deferred<unknown>()
    mocks.request.mockReturnValue(response.promise)
    const hook = setup()
    act(() => kind === "mood" ? hook.result.current.setMoodEnabled(true) : hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick(10_000)
    expect(mocks.request).toHaveBeenCalledTimes(1)
    act(() => scope.abort())
    expect(mocks.request.mock.calls[0][0].abortSignal.aborted).toBe(true)
    await act(async () => response.resolve({ choices: [{ message: { content: "calm" } }] }))
    expect(hook.result.current.currentMood).toBeNull()
    expect(hook.result.current.echoReactions).toEqual([])
    expect(hook.result.current.moodAnalyzing).toBe(false)
    expect(hook.result.current.echoAnalyzing).toBe(false)
  })

  it.each(["auth", "config", "disable", "unmount"])("does not dispatch after delayed lookup and %s boundary", async (boundary) => {
    const pending = deferred<ServicePromptSnapshot>()
    const loader = mocks.load.getMockImplementation()!
    mocks.load.mockImplementationOnce((...args) => { void loader(...args); return pending.promise })
    const hook = setup()
    act(() => hook.result.current.setEchoEnabled(true))
    hook.rerender("x".repeat(500))
    await tick()
    act(() => {
      if (boundary === "auth") window.dispatchEvent(new Event("tldw:auth-credentials-changed"))
      if (boundary === "config") mocks.configListeners.forEach((listener) => listener())
      if (boundary === "disable") hook.result.current.setEchoEnabled(false)
      if (boundary === "unmount") hook.unmount()
    })
    await act(async () => pending.resolve(snapshots[0]))
    expect(mocks.request).not.toHaveBeenCalled()
    expect(snapshots[0].release).toHaveBeenCalledOnce()
  })
})
