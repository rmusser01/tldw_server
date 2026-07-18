import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getSetting: vi.fn(async () => true),
  invoke: vi.fn(async () => ({ content: "Scoped title" })),
  pageAssistModel: vi.fn<(options?: unknown) => Promise<any>>()
}))

vi.mock("@/models", () => ({
  pageAssistModel: mocks.pageAssistModel
}))

vi.mock("@/services/settings/registry", () => ({
  coerceBoolean: (value: unknown) => Boolean(value),
  defineSetting: (key: string) => key,
  getSetting: (...args: unknown[]) => mocks.getSetting(...args),
  setSetting: vi.fn()
}))

vi.mock("@/libs/reasoning", () => ({
  removeReasoning: (value: string) => value
}))

import { generateTitle } from "../title"

const requestScope = Object.freeze({
  config: Object.freeze({
    serverUrl: "https://scope.example",
    authMode: "multi-user" as const
  }),
  userId: 7
})

describe("generateTitle request scope", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.pageAssistModel.mockResolvedValue({ invoke: mocks.invoke })
  })

  it("uses the captured request scope and signal for title generation", async () => {
    const controller = new AbortController()

    await expect(generateTitle(
      "model-1",
      "question",
      "fallback",
      { signal: controller.signal, requestScope }
    )).resolves.toBe("Scoped title")

    expect(mocks.pageAssistModel).toHaveBeenCalledWith({
      model: "model-1",
      toolChoice: "none",
      saveToDb: false,
      requestScope
    })
    expect(mocks.invoke).toHaveBeenCalledWith(
      expect.any(Array),
      { signal: controller.signal }
    )
  })

  it("does not turn a scope abort into a fallback title", async () => {
    const controller = new AbortController()
    const abortError = new Error("scope changed")
    abortError.name = "AbortError"
    mocks.invoke.mockImplementationOnce(async () => {
      controller.abort()
      throw abortError
    })

    await expect(generateTitle(
      "model-1",
      "question",
      "fallback",
      { signal: controller.signal, requestScope }
    )).rejects.toBe(abortError)
  })

  it("does not turn a structured request-scope 412 into a fallback title", async () => {
    const scopeChangedError = Object.assign(new Error("scope changed"), {
      status: 412,
      details: { code: "request_config_scope_changed" }
    })
    mocks.invoke.mockRejectedValueOnce(scopeChangedError)

    await expect(generateTitle(
      "model-1",
      "question",
      "fallback",
      { requestScope }
    )).rejects.toBe(scopeChangedError)
  })
})
