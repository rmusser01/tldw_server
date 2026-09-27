// @vitest-environment jsdom
import { act, renderHook } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { usePlaygroundImageGen } from "../usePlaygroundImageGen"

const mocks = vi.hoisted(() => ({
  createChatCompletion: vi.fn(),
  initialize: vi.fn(),
  loadServicePromptSnapshot: vi.fn(),
  releaseSnapshot: vi.fn(),
  notificationError: vi.fn(),
  scopeController: new AbortController(),
  scopeInvalidatedController: new AbortController(),
  requestScope: {
    config: { serverUrl: "https://captured.example" },
    userId: null
  }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: () => ({ data: [], isLoading: false })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: (...args: unknown[]) => mocks.initialize(...args),
    createChatCompletion: (...args: unknown[]) =>
      mocks.createChatCompletion(...args)
  }
}))

vi.mock("@/services/service-prompts", () => ({
  loadServicePromptSnapshot: (...args: unknown[]) =>
    mocks.loadServicePromptSnapshot(...args)
}))

vi.mock("@/utils/resolve-api-provider", () => ({
  resolveApiProviderForModel: vi.fn(async () => "custom")
}))

const completionResponse = (content: string) => ({
  json: async () => ({ choices: [{ message: { content } }] })
})

const servicePromptSnapshot = (includeDefinition = true) => ({
  scopeKey: "captured-scope",
  requestScope: mocks.requestScope,
  capability: "supported",
  definitions: includeDefinition
    ? {
        "image.prompt.refinement": {
          definition: {
            id: "image.prompt.refinement",
            parts: [
              {
                key: "system_semantics",
                mode: "literal",
                required_variables: []
              },
              {
                key: "rewrite_semantics",
                mode: "literal",
                required_variables: []
              }
            ]
          },
          parts: {
            system_semantics:
              "Preserve the subject and favor painterly detail.",
            rewrite_semantics: "Return a vivid, compact production prompt."
          },
          source: "user",
          revision: "11111111-1111-4111-8111-111111111111"
        }
      }
    : {},
  scopeSignal: mocks.scopeController.signal,
  scopeInvalidatedSignal: mocks.scopeInvalidatedController.signal,
  release: mocks.releaseSnapshot
})

const baseDeps = () => ({
  imageBackendDefaultTrimmed: "local-sd",
  imageBackendOptions: [{ value: "local-sd", label: "Local SD" }],
  imageEventSyncChatMode: "off" as const,
  imageEventSyncGlobalDefault: "off" as const,
  updateChatSettings: vi.fn(),
  setImageEventSyncGlobalDefault: vi.fn(),
  messages: [
    {
      isBot: true,
      message: "Lana stands by the rainy neon alley.",
      moodLabel: "focused"
    }
  ],
  selectedCharacterName: "Lana",
  selectedModel: "deepseek-chat",
  currentApiProvider: "custom",
  formMessage: "",
  sendMessage: vi.fn(async () => undefined),
  textAreaFocus: vi.fn(),
  notificationApi: { error: mocks.notificationError },
  t: (_key: string, fallback?: string | { defaultValue?: string }) =>
    typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key,
  setToolsPopoverOpen: vi.fn()
})

const prepareHook = () => {
  const rendered = renderHook(() => usePlaygroundImageGen(baseDeps()))
  act(() => {
    rendered.result.current.setImageGenerateBackend("local-sd")
    rendered.result.current.setImageGeneratePrompt(
      "Portrait of Lana in neon rain."
    )
  })
  return rendered
}

describe("usePlaygroundImageGen Service Prompts", () => {
  beforeEach(() => {
    mocks.scopeController = new AbortController()
    mocks.scopeInvalidatedController = new AbortController()
    mocks.initialize.mockReset()
    mocks.initialize.mockResolvedValue(undefined)
    mocks.createChatCompletion.mockReset()
    mocks.createChatCompletion.mockResolvedValue(
      completionResponse("Prompt: cinematic painterly portrait of Lana")
    )
    mocks.releaseSnapshot.mockReset()
    mocks.notificationError.mockReset()
    mocks.loadServicePromptSnapshot.mockReset()
    mocks.loadServicePromptSnapshot.mockResolvedValue(servicePromptSnapshot())
  })

  it("binds one scoped snapshot and its custom semantics to each refinement", async () => {
    const { result } = prepareHook()

    await act(async () => {
      await result.current.handleRefineImagePromptDraft()
    })

    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledOnce()
    expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledWith([
      "image.prompt.refinement"
    ])
    const [requestBody, requestOptions] =
      mocks.createChatCompletion.mock.calls[0] ?? []
    expect(requestBody.messages[0].content).toContain(
      "Preserve the subject and favor painterly detail."
    )
    expect(requestBody.messages[1].content).toContain(
      "Return a vivid, compact production prompt."
    )
    expect(requestOptions).toEqual({
      signal: mocks.scopeController.signal,
      requestScope: mocks.requestScope
    })
    expect(result.current.imagePromptRefineCandidate).toBe(
      "cinematic painterly portrait of Lana"
    )
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
  })

  it("does not commit a candidate after the captured scope is invalidated", async () => {
    mocks.createChatCompletion.mockImplementationOnce(async () => {
      mocks.scopeInvalidatedController.abort()
      mocks.scopeController.abort()
      return completionResponse("Prompt: stale account result")
    })
    const { result } = prepareHook()

    await act(async () => {
      await result.current.handleRefineImagePromptDraft()
    })

    expect(result.current.imagePromptRefineCandidate).toBe("")
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
  })

  it("does not dispatch after the captured scope is already invalidated", async () => {
    mocks.scopeInvalidatedController.abort()
    const { result } = prepareHook()

    await act(async () => {
      await result.current.handleRefineImagePromptDraft()
    })

    expect(mocks.createChatCompletion).not.toHaveBeenCalled()
    expect(result.current.imagePromptRefineCandidate).toBe("")
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
  })

  it("fails closed when a supported snapshot omits the requested definition", async () => {
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce(
      servicePromptSnapshot(false)
    )
    const { result } = prepareHook()

    await act(async () => {
      await result.current.handleRefineImagePromptDraft()
    })

    expect(mocks.createChatCompletion).not.toHaveBeenCalled()
    expect(result.current.imagePromptRefineCandidate).toBe("")
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
  })

  it("preserves empty-result reporting without committing a candidate", async () => {
    mocks.createChatCompletion.mockResolvedValueOnce(completionResponse(""))
    const { result } = prepareHook()

    await act(async () => {
      await result.current.handleRefineImagePromptDraft()
    })

    expect(result.current.imagePromptRefineCandidate).toBe("")
    expect(mocks.notificationError).toHaveBeenCalledWith({
      message: "Prompt refinement failed",
      description: "Refiner returned an empty prompt. Try again."
    })
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
  })

  it("preserves ordinary provider error reporting without committing a candidate", async () => {
    mocks.createChatCompletion.mockRejectedValueOnce(
      new Error("Provider unavailable")
    )
    const { result } = prepareHook()

    await act(async () => {
      await result.current.handleRefineImagePromptDraft()
    })

    expect(result.current.imagePromptRefineCandidate).toBe("")
    expect(mocks.notificationError).toHaveBeenCalledWith({
      message: "Prompt refinement failed",
      description: "Provider unavailable"
    })
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
  })
})
