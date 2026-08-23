import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  getSetting: vi.fn(async () => true),
  invoke: vi.fn(async () => ({ content: "Scoped title" })),
  loadServicePromptSnapshot: vi.fn(),
  pageAssistModel: vi.fn<(options?: unknown) => Promise<any>>()
}))

vi.mock("@/models", () => ({
  pageAssistModel: mocks.pageAssistModel
}))

vi.mock("@/services/settings/registry", async () => {
  const actual = await vi.importActual<typeof import("@/services/settings/registry")>(
    "@/services/settings/registry"
  )
  return {
    ...actual,
    coerceBoolean: (value: unknown) => Boolean(value),
    defineSetting: (key: string) => key,
    getSetting: (...args: unknown[]) => mocks.getSetting(...args),
    setSetting: vi.fn()
  }
})

vi.mock("@/libs/reasoning", () => ({
  removeReasoning: (value: string) => value
}))

vi.mock("@/services/service-prompts", async () => {
  const actual = await vi.importActual<typeof import("@/services/service-prompts")>(
    "@/services/service-prompts"
  )
  return {
    ...actual,
    loadServicePromptSnapshot: (...args: unknown[]) =>
      mocks.loadServicePromptSnapshot(...args)
  }
})

import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { generateTitle } from "../title"

const requestScope = Object.freeze({
  config: Object.freeze({
    serverUrl: "https://scope.example",
    authMode: "multi-user" as const
  }),
  userId: 7
})

const snapshotFor = (template = "Title for {query}") => {
  const scopeController = new AbortController()
  const scopeInvalidatedController = new AbortController()
  const snapshot = Object.freeze({
    scopeKey: "scope-key",
    requestScope,
    capability: "supported" as const,
    definitions: Object.freeze({
      "chat.title.generation": Object.freeze({
        definition: Object.freeze({
          id: "chat.title.generation",
          parts: Object.freeze([Object.freeze({
            key: "user_template",
            mode: "template" as const,
            required_variables: Object.freeze(["query"])
          })])
        }),
        parts: Object.freeze({ user_template: template }),
        source: "user" as const,
        revision: "123e4567-e89b-42d3-a456-426614174000"
      })
    }),
    scopeSignal: scopeController.signal,
    scopeInvalidatedSignal: scopeInvalidatedController.signal,
    release: vi.fn()
  }) as ServicePromptSnapshot
  return { snapshot, scopeInvalidatedController }
}

describe("generateTitle service-prompt scope", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.getSetting.mockResolvedValue(true)
    mocks.invoke.mockResolvedValue({ content: "Scoped title" })
    mocks.pageAssistModel.mockResolvedValue({ invoke: mocks.invoke })
    mocks.loadServicePromptSnapshot.mockResolvedValue(snapshotFor().snapshot)
  })

  it("renders the custom title template once and sends its bytes to the provider", async () => {
    const { snapshot } = snapshotFor("Custom title for literal {{query}}: {query}")
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshot)

    const result = await generateTitle(
      "model-1",
      "What changed?",
      "fallback",
      { requestScope }
    )

    const invokedMessages = mocks.invoke.mock.calls[0]?.[0]
    expect(invokedMessages[0].content).toBe(
      "Custom title for literal {query}: What changed?"
    )
    expect(result).toBe("Scoped title")
    expect(snapshot.release).toHaveBeenCalledOnce()
  })

  it.each([
    ["custom", "Custom title: What changed?"],
    ["packaged", "Packaged title: What changed?"]
  ])("renders the $name template into hand-derived provider bytes", async (_name, expected) => {
    const template = expected.replace("What changed?", "{query}")
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshotFor(template).snapshot)

    await generateTitle("model-1", "What changed?", "fallback", { requestScope })

    expect(mocks.invoke.mock.calls[0]?.[0][0].content).toBe(expected)
  })

  it("returns the fallback without a snapshot or model request when disabled", async () => {
    mocks.getSetting.mockResolvedValueOnce(false)

    await expect(generateTitle("model-1", "question", "fallback", { requestScope }))
      .resolves.toBe("fallback")

    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
  })

  it("returns the fallback when reading the setting fails", async () => {
    mocks.getSetting.mockRejectedValueOnce(new Error("secret setting error"))

    await expect(generateTitle("model-1", "question", "fallback", { requestScope }))
      .resolves.toBe("fallback")

    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
    expect(mocks.pageAssistModel).not.toHaveBeenCalled()
  })

  it.each([
    ["snapshot", () => mocks.loadServicePromptSnapshot.mockRejectedValueOnce(new Error("secret snapshot error"))],
    ["render", () => mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshotFor("{not_query}").snapshot)],
    ["model", () => mocks.pageAssistModel.mockRejectedValueOnce(new Error("secret model error"))]
  ])("returns the fallback on an ordinary $name failure", async (_name, arrange) => {
    arrange()

    await expect(generateTitle("model-1", "question", "fallback", { requestScope }))
      .resolves.toBe("fallback")
  })

  it("logs a generic fallback error without authored prompt or error text", async () => {
    const secretPrompt = "secret authored title prompt {query}"
    const secretError = "secret provider error"
    const log = vi.spyOn(console, "error").mockImplementation(() => undefined)
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshotFor(secretPrompt).snapshot)
    mocks.pageAssistModel.mockRejectedValueOnce(new Error(secretError))

    await expect(generateTitle("model-1", "question", "fallback", { requestScope }))
      .resolves.toBe("fallback")

    expect(log).toHaveBeenCalledWith("Error generating title")
    expect(log.mock.calls.flat().join(" ")).not.toContain(secretPrompt)
    expect(log.mock.calls.flat().join(" ")).not.toContain(secretError)
    log.mockRestore()
  })

  it("releases the snapshot after an ordinary fallback", async () => {
    const { snapshot } = snapshotFor()
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshot)
    mocks.pageAssistModel.mockRejectedValueOnce(new Error("model unavailable"))

    await expect(generateTitle("model-1", "question", "fallback", { requestScope }))
      .resolves.toBe("fallback")

    expect(snapshot.release).toHaveBeenCalledOnce()
  })

  it("rethrows an expected request-scope mismatch", async () => {
    const scopeChanged = Object.assign(new Error("scope changed"), {
      status: 412,
      details: { detail: { code: "request_config_scope_changed" } }
    })
    mocks.loadServicePromptSnapshot.mockRejectedValueOnce(scopeChanged)

    await expect(generateTitle("model-1", "question", "fallback", { requestScope }))
      .rejects.toBe(scopeChanged)
  })

  it("rethrows canonical scope invalidation and releases the snapshot", async () => {
    const { snapshot, scopeInvalidatedController } = snapshotFor()
    const abortError = new Error("scope invalidated")
    abortError.name = "AbortError"
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshot)
    mocks.invoke.mockImplementationOnce(async () => {
      scopeInvalidatedController.abort()
      throw abortError
    })

    await expect(generateTitle("model-1", "question", "fallback", { requestScope }))
      .rejects.toMatchObject({
        status: 412,
        details: { detail: { code: "request_config_scope_changed" } }
      })

    expect(snapshot.release).toHaveBeenCalledOnce()
  })

  it("rethrows a caller abort and releases the snapshot", async () => {
    const { snapshot } = snapshotFor()
    const controller = new AbortController()
    const abortError = new Error("caller aborted")
    abortError.name = "AbortError"
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshot)
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

    expect(snapshot.release).toHaveBeenCalledOnce()
  })

  it.each([
    ["model creation", "caller"],
    ["model creation", "scope"],
    ["invocation", "caller"],
    ["invocation", "scope"]
  ])("does not return a signal-ignoring provider result after $phase $cancellation cancellation", async (phase, cancellation) => {
    const { snapshot, scopeInvalidatedController } = snapshotFor()
    const caller = new AbortController()
    const cancel = () => {
      if (cancellation === "scope") {
        scopeInvalidatedController.abort()
      } else {
        caller.abort()
      }
    }
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(snapshot)
    if (phase === "model creation") {
      mocks.pageAssistModel.mockImplementationOnce(async () => {
        cancel()
        return { invoke: mocks.invoke }
      })
    } else {
      mocks.invoke.mockImplementationOnce(async () => {
        cancel()
        return { content: "late provider title" }
      })
    }

    const title = generateTitle(
      "model-1",
      "question",
      "fallback",
      { signal: caller.signal, requestScope }
    )

    if (cancellation === "scope") {
      await expect(title).rejects.toMatchObject({
        status: 412,
        details: { detail: { code: "request_config_scope_changed" } }
      })
    } else {
      await expect(title).rejects.toMatchObject({ name: "AbortError" })
    }
    if (phase === "model creation") {
      expect(mocks.invoke).not.toHaveBeenCalled()
    } else {
      expect(mocks.invoke).toHaveBeenCalledOnce()
    }
    expect(snapshot.release).toHaveBeenCalledOnce()
  })
})
