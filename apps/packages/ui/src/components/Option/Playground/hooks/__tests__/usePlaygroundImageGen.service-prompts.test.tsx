// @vitest-environment jsdom
import { act, renderHook, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"

import type { ImageBackendConfigMap } from "@/services/image-generation"
import type { ChatSubmitResult } from "@/hooks/chat/chat-action-utils"
import { usePlaygroundImageGen } from "../usePlaygroundImageGen"

const mocks = vi.hoisted(() => ({
  createChatCompletion: vi.fn(),
  getImageBackendConfigs: vi.fn(async (_options?: { throwOnError?: boolean }): Promise<ImageBackendConfigMap> => ({})),
  resolveApiProviderForModel: vi.fn(async (_args: unknown) => "custom"),
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
  resolveApiProviderForModel: (args: unknown) => mocks.resolveApiProviderForModel(args)
}))

vi.mock("@/services/image-generation", async original => ({
  ...(await original<typeof import("@/services/image-generation")>()),
  getImageBackendConfigs: (options?: { throwOnError?: boolean }) => mocks.getImageBackendConfigs(options)
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
  sendMessage: vi.fn(async (): Promise<ChatSubmitResult | void> => undefined),
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

const heldSend = () => {
  let resolve!: (value: ChatSubmitResult | void) => void,
    reject!: (reason: Error) => void;
  const promise = new Promise<ChatSubmitResult | void>((done, fail) => {
    resolve = done;
    reject = fail;
  });
  return { promise, resolve, reject };
};
describe("image send completion ownership", () => {
  beforeEach(() => mocks.notificationError.mockReset());
  const setup = (
    sendMessage: (
      payload: Record<string, unknown>,
    ) => Promise<ChatSubmitResult | void>,
    captureCompletionGuard = (_focus?: boolean) => () => true,
  ) => {
    const deps = { ...baseDeps(), sendMessage, captureCompletionGuard };
    const view = renderHook(() => usePlaygroundImageGen(deps));
    act(() => {
      view.result.current.openImageGenerateModal();
      view.result.current.setImageGeneratePrompt("Original image");
    });
    return { ...view, deps };
  };
  it.each(["submitted", "failed", "skipped"] as const)(
    "preserves a reopened modal after an old %s result",
    async (status) => {
      const old = heldSend(),
        next = heldSend(),
        send = vi
          .fn()
          .mockReturnValueOnce(old.promise)
          .mockReturnValueOnce(next.promise);
      const view = setup(send);
      let first!: Promise<void>, second!: Promise<void>;
      act(() => {
        first = view.result.current.submitImageGenerateModal();
      });
      act(() => {
        view.result.current.closeImageGenerateModal();
        view.result.current.openImageGenerateModal();
        view.result.current.setImageGeneratePrompt("New image");
      });
      act(() => {
        second = view.result.current.submitImageGenerateModal();
      });
      await act(async () => {
        old.resolve(
          status === "submitted"
            ? { status }
            : status === "failed"
              ? { status, errorMessage: "Failed" }
              : { status, reason: "Skipped" },
        );
        await first;
      });
      expect(view.result.current.imageGenerateModalOpen).toBe(true);
      expect(view.result.current.imageGenerateSubmitting).toBe(true);
      expect(view.deps.textAreaFocus).not.toHaveBeenCalled();
      await act(async () => {
        next.resolve({ status: "submitted" });
        await second;
      });
      expect(view.result.current.imageGenerateModalOpen).toBe(false);
      expect(view.result.current.imageGenerateSubmitting).toBe(false);
    },
  );
  it("does not notify a new source after a held rejection and releases its own pending state", async () => {
    const send = heldSend();
    let current = true;
    const guard = () => {
      const began = current;
      return () => began && current;
    };
    const view = setup(() => send.promise, guard);
    let pending!: Promise<void>;
    act(() => {
      pending = view.result.current.submitImageGenerateModal();
    });
    current = false;
    view.deps.captureCompletionGuard = () => () => false;
    view.rerender();
    expect(view.result.current.imageGenerateSubmitting).toBe(false);
    await act(async () => {
      send.reject(new Error("Old request failed"));
      await pending;
    });
    expect(mocks.notificationError).not.toHaveBeenCalled();
  });
  it.each(["failed", "skipped"] as const)(
    "keeps the current modal available after %s",
    async (status) => {
      const view = setup(async () =>
        status === "failed"
          ? { status, errorMessage: "Failed" }
          : { status, reason: "Skipped" },
      );
      await act(async () => {
        await view.result.current.submitImageGenerateModal();
      });
      expect(view.result.current.imageGenerateModalOpen).toBe(true);
      expect(view.result.current.imageGenerateSubmitting).toBe(false);
      expect(view.deps.textAreaFocus).not.toHaveBeenCalled();
    },
  );
});


// Exercise the production hook; configuration, transport, and source authority are I/O doubles.
const held = <T,>() => {
  let resolve!: (value: T) => void, reject!: (error: Error) => void
  const promise = new Promise<T>((done, fail) => { resolve = done; reject = fail })
  return { promise, resolve, reject }
}
const imagePayload = (prompt = "Prompt: current refined image") => ({ choices: [{ message: { content: prompt } }] })
const setupLifetime = async () => {
  let owner = 0
  const captureForOwner = (renderedOwner: number) => () => () => owner === renderedOwner
  const deps = { ...baseDeps(), captureCompletionGuard: captureForOwner(owner) }
  const view = renderHook(() => usePlaygroundImageGen(deps))
  await act(async () => { view.result.current.openImageGenerateModal() })
  act(() => { view.result.current.setImageGeneratePrompt("Original image") })
  const retireSource = () => {
    owner++
    deps.captureCompletionGuard = captureForOwner(owner)
    view.rerender()
  }
  return { ...view, deps, retireSource }
}
type ImageHook = ReturnType<typeof usePlaygroundImageGen>
const settingsFields: Array<{ name: string; read: (hook: ImageHook) => unknown; edit: (hook: ImageHook, other: boolean) => void }> = [
  { name: "backend", read: h => h.imageGenerateBackend, edit: (h, other) => h.setImageGenerateBackend(other ? "other" : "local-sd") },
  { name: "format", read: h => h.imageGenerateFormat, edit: (h, other) => h.setImageGenerateFormat(other ? "jpg" : "png") },
  { name: "negative prompt", read: h => h.imageGenerateNegativePrompt, edit: (h, other) => h.setImageGenerateNegativePrompt(other ? "Other" : "Keep") },
  { name: "width", read: h => h.imageGenerateWidth, edit: (h, other) => h.setImageGenerateWidth(other ? 640 : 384) },
  { name: "height", read: h => h.imageGenerateHeight, edit: (h, other) => h.setImageGenerateHeight(other ? 640 : 384) },
  { name: "steps", read: h => h.imageGenerateSteps, edit: (h, other) => h.setImageGenerateSteps(other ? 40 : 20) },
  { name: "CFG scale", read: h => h.imageGenerateCfgScale, edit: (h, other) => h.setImageGenerateCfgScale(other ? 8 : 4) },
  { name: "seed", read: h => h.imageGenerateSeed, edit: (h, other) => h.setImageGenerateSeed(other ? 42 : 12) },
  { name: "sampler", read: h => h.imageGenerateSampler, edit: (h, other) => h.setImageGenerateSampler(other ? "Other" : "Keep") },
  { name: "model", read: h => h.imageGenerateModel, edit: (h, other) => h.setImageGenerateModel(other ? "Other" : "Keep") },
  { name: "extra params", read: h => h.imageGenerateExtraParams, edit: (h, other) => h.setImageGenerateExtraParams(other ? '{"other":true}' : '{"keep":true}') }
]
const savedConfigs = { "local-sd": { format: "webp" as const, negativePrompt: "Saved", width: 999, height: 999, steps: 99, cfgScale: 9, seed: 99, sampler: "Saved", model: "Saved", extraParams: { saved: true } } }

describe("image modal asynchronous lifetime", () => {
  beforeEach(() => {
    mocks.scopeController = new AbortController()
    mocks.scopeInvalidatedController = new AbortController()
    mocks.initialize.mockReset().mockResolvedValue(undefined)
    mocks.getImageBackendConfigs.mockReset().mockResolvedValue({})
    mocks.resolveApiProviderForModel.mockReset().mockResolvedValue("custom")
    mocks.loadServicePromptSnapshot.mockReset().mockResolvedValue(servicePromptSnapshot())
    mocks.createChatCompletion.mockReset().mockResolvedValue(completionResponse("Prompt: current refined image"))
    mocks.releaseSnapshot.mockReset()
    mocks.notificationError.mockReset()
  })

  it("hydrates a current modal with all acknowledged saved settings", async () => {
    const view = await setupLifetime()
    mocks.getImageBackendConfigs.mockResolvedValueOnce(savedConfigs)
    await act(async () => { await view.result.current.hydrateImageGenerateSettings("local-sd") })
    expect(view.result.current).toMatchObject({ imageGenerateFormat: "webp", imageGenerateNegativePrompt: "Saved", imageGenerateWidth: 999, imageGenerateHeight: 999, imageGenerateSteps: 99, imageGenerateCfgScale: 9, imageGenerateSeed: 99, imageGenerateSampler: "Saved", imageGenerateModel: "Saved", imageGenerateExtraParams: JSON.stringify({ saved: true }, null, 2) })
    expect(mocks.notificationError).not.toHaveBeenCalled()
  })

  it.each(["source", "close", "reopen"] as const)("keeps a replacement's settings after a held read retires by %s", async retirement => {
    const view = await setupLifetime(), read = held<typeof savedConfigs>()
    act(() => { view.result.current.setImageGenerateWidth(384); view.result.current.setImageGenerateNegativePrompt("Current") })
    mocks.getImageBackendConfigs.mockResolvedValue({ "local-sd": { width: 384, negativePrompt: "Current" } }).mockReturnValueOnce(read.promise)
    let pending!: Promise<void>
    act(() => { pending = view.result.current.hydrateImageGenerateSettings("local-sd") })
    if (retirement === "source") view.retireSource()
    else await act(async () => { view.result.current.closeImageGenerateModal(); if (retirement === "reopen") view.result.current.openImageGenerateModal() })
    await act(async () => { read.resolve(savedConfigs); await pending })
    expect(view.result.current.imageGenerateWidth).toBe(384)
    expect(view.result.current.imageGenerateNegativePrompt).toBe("Current")
    expect(mocks.notificationError).not.toHaveBeenCalled()
  })

  it.each(settingsFields)("preserves edit and Undo of $name during a held settings read", async field => {
    const view = await setupLifetime(), read = held<typeof savedConfigs>()
    act(() => { field.edit(view.result.current, false); view.result.current.setImageGenerateWidth(384) })
    const original = field.read(view.result.current)
    mocks.getImageBackendConfigs.mockReturnValueOnce(read.promise)
    let pending!: Promise<void>
    act(() => { pending = view.result.current.hydrateImageGenerateSettings("local-sd") })
    expect(view.result.current.imageGenerateBusy).toBe(false)
    act(() => { field.edit(view.result.current, true); field.edit(view.result.current, false) })
    await act(async () => { read.resolve(savedConfigs); await pending })
    expect(field.read(view.result.current)).toEqual(original)
    expect(view.result.current.imageGenerateWidth).toBe(384)
  })

  it("keeps the latest backend A/B/A settings request when old reads resolve last", async () => {
    const view = await setupLifetime()
    const reads = [held<typeof savedConfigs>(), held<typeof savedConfigs>(), held<typeof savedConfigs>()]
    for (const read of reads) mocks.getImageBackendConfigs.mockReturnValueOnce(read.promise)
    const pending: Promise<void>[] = []
    for (const backend of ["local-sd", "other", "local-sd"]) {
      act(() => {
        view.result.current.setImageGenerateBackend(backend)
        pending.push(view.result.current.hydrateImageGenerateSettings(backend))
      })
    }
    await act(async () => { reads[2].resolve({ "local-sd": { ...savedConfigs["local-sd"], width: 303 } }); await pending[2] })
    await act(async () => { reads[1].resolve({ "local-sd": { ...savedConfigs["local-sd"], width: 202 } }); await pending[1] })
    await act(async () => { reads[0].resolve({ "local-sd": { ...savedConfigs["local-sd"], width: 101 } }); await pending[0] })
    expect(view.result.current.imageGenerateBackend).toBe("local-sd")
    expect(view.result.current.imageGenerateWidth).toBe(303)
  })

  it("keeps current values and reports a failed settings read", async () => {
    const view = await setupLifetime()
    act(() => { view.result.current.setImageGenerateWidth(384); view.result.current.setImageGenerateNegativePrompt("Keep") })
    mocks.getImageBackendConfigs.mockRejectedValueOnce(new Error("Saved settings unavailable"))
    await act(async () => { await view.result.current.hydrateImageGenerateSettings("local-sd") })
    expect(view.result.current.imageGenerateWidth).toBe(384)
    expect(view.result.current.imageGenerateNegativePrompt).toBe("Keep")
    expect(mocks.notificationError).toHaveBeenCalledOnce()
  })

  it("retains settings when the actual registry storage read fails", async () => {
    const actual = await vi.importActual<typeof import("@/services/image-generation")>("@/services/image-generation")
    const registry = await vi.importActual<typeof import("@/services/settings/registry")>("@/services/settings/registry")
    const storage = registry.getStorageForSetting({ key: "imageBackendConfigs", defaultValue: {} })
    const read = vi.spyOn(storage, "get").mockRejectedValue(new Error("Saved settings unavailable"))
    try {
      const view = await setupLifetime()
      act(() => { view.result.current.setImageGenerateWidth(384); view.result.current.setImageGenerateNegativePrompt("Keep") })
      mocks.getImageBackendConfigs.mockImplementationOnce(actual.getImageBackendConfigs)
      await act(async () => { await view.result.current.hydrateImageGenerateSettings("local-sd") })
      expect(read).toHaveBeenCalledWith("imageBackendConfigs")
      expect(view.result.current.imageGenerateWidth).toBe(384)
      expect(view.result.current.imageGenerateNegativePrompt).toBe("Keep")
      expect(mocks.notificationError).toHaveBeenCalledWith({ message: "Image settings unavailable", description: "Saved settings unavailable" })
    } finally { read.mockRestore() }
  })

  it("retains the advisory default contract for other backend settings readers", async () => {
    const actual = await vi.importActual<typeof import("@/services/image-generation")>("@/services/image-generation")
    const registry = await vi.importActual<typeof import("@/services/settings/registry")>("@/services/settings/registry")
    const storage = registry.getStorageForSetting({ key: "imageBackendConfigs", defaultValue: {} })
    const read = vi.spyOn(storage, "get").mockRejectedValue(new Error("Saved settings unavailable"))
    try { expect(await actual.getImageBackendConfigs()).toEqual({}) }
    finally { read.mockRestore() }
  })

  it("stays quiet after a retired settings read rejects", async () => {
    const view = await setupLifetime(), read = held<typeof savedConfigs>()
    mocks.getImageBackendConfigs.mockReturnValueOnce(read.promise)
    let pending!: Promise<void>
    act(() => { pending = view.result.current.hydrateImageGenerateSettings("local-sd") })
    view.retireSource()
    act(() => { view.result.current.setImageGenerateWidth(384) })
    await act(async () => { read.reject(new Error("Retired settings unavailable")); await pending })
    expect(view.result.current.imageGenerateWidth).toBe(384)
    expect(mocks.notificationError).not.toHaveBeenCalled()
  })

  it("rejects a stale rendered refinement before initialization or snapshot loading", async () => {
    const view = await setupLifetime(), old = view.result.current.handleRefineImagePromptDraft
    view.retireSource()
    await act(async () => { await old() })
    expect(mocks.initialize).not.toHaveBeenCalled()
    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
    expect(mocks.createChatCompletion).not.toHaveBeenCalled()
  })

  it("rejects a stale settings callback before reading configuration", async () => {
    const view = await setupLifetime(), old = view.result.current.hydrateImageGenerateSettings
    mocks.getImageBackendConfigs.mockClear()
    view.retireSource()
    await act(async () => { await old("local-sd") })
    expect(mocks.getImageBackendConfigs).not.toHaveBeenCalled()
    expect(mocks.notificationError).not.toHaveBeenCalled()
  })

  it("releases retired refinement busy state before its held initialization completes", async () => {
    const view = await setupLifetime(), initialize = held<void>()
    mocks.initialize.mockReturnValueOnce(initialize.promise)
    let pending!: Promise<void>
    act(() => { pending = view.result.current.handleRefineImagePromptDraft() })
    expect(view.result.current.imagePromptRefineSubmitting).toBe(true)
    view.retireSource()
    expect(view.result.current.imageGenerateBusy).toBe(false)
    await act(async () => { initialize.resolve(); await pending })
    expect(mocks.loadServicePromptSnapshot).not.toHaveBeenCalled()
  })

  const boundaries = ["initialize", "snapshot", "provider", "response", "json"] as const
  it.each(boundaries.flatMap(boundary => ["source", "reopen"].flatMap(retirement => ["success", "failure"].map(outcome => ({ boundary, retirement, outcome })))))(
    "retires a held $boundary $outcome after $retirement without adopting the next source",
    async ({ boundary, retirement, outcome }) => {
      const view = await setupLifetime(), pause = held<unknown>()
      const json = vi.fn(async () => imagePayload())
      mocks.createChatCompletion.mockResolvedValue({ json })
      const entries = { initialize: mocks.initialize, snapshot: mocks.loadServicePromptSnapshot, provider: mocks.resolveApiProviderForModel, response: mocks.createChatCompletion, json }
      const current = entries[boundary] as ReturnType<typeof vi.fn>
      current.mockReturnValueOnce(pause.promise)
      let pending!: Promise<void>
      act(() => { pending = view.result.current.handleRefineImagePromptDraft() })
      await waitFor(() => expect(current).toHaveBeenCalledOnce())
      if (retirement === "source") view.retireSource()
      else await act(async () => { view.result.current.closeImageGenerateModal(); view.result.current.openImageGenerateModal() })
      await act(async () => {
        if (outcome === "failure") pause.reject(new Error("Retired boundary failed"))
        else pause.resolve(boundary === "snapshot" ? servicePromptSnapshot() : boundary === "provider" ? "custom" : boundary === "response" ? { json } : boundary === "json" ? imagePayload() : undefined)
        await pending
      })
      const order = boundaries.indexOf(boundary)
      for (const next of boundaries.slice(order + 1)) expect(entries[next]).not.toHaveBeenCalled()
      expect(view.result.current.imagePromptRefineCandidate).toBe("")
      expect(mocks.notificationError).not.toHaveBeenCalled()
      expect(mocks.releaseSnapshot).toHaveBeenCalledTimes(boundary === "initialize" || (boundary === "snapshot" && outcome === "failure") ? 0 : 1)
    }
  )

  it.each(["refine", "generate"] as const)("refuses another current operation while $0 is pending", async operation => {
    const view = await setupLifetime(), wire = held<ReturnType<typeof completionResponse>>(), send = heldSend()
    mocks.createChatCompletion.mockReturnValueOnce(wire.promise)
    view.deps.sendMessage.mockReturnValueOnce(send.promise)
    let pending!: Promise<void>
    act(() => { pending = operation === "refine" ? view.result.current.handleRefineImagePromptDraft() : view.result.current.submitImageGenerateModal() })
    if (operation === "refine") await waitFor(() => expect(mocks.createChatCompletion).toHaveBeenCalledOnce())
    let duplicateRefine!: Promise<void>, duplicateGenerate!: Promise<void>
    act(() => { duplicateRefine = view.result.current.handleRefineImagePromptDraft(); duplicateGenerate = view.result.current.submitImageGenerateModal() })
    try {
      await act(async () => {})
      expect(mocks.createChatCompletion).toHaveBeenCalledTimes(operation === "refine" ? 1 : 0)
      expect(view.deps.sendMessage).toHaveBeenCalledTimes(operation === "generate" ? 1 : 0)
    } finally {
      await act(async () => { wire.resolve(completionResponse("Prompt: current image")); send.resolve({ status: "submitted" }); await Promise.allSettled([pending, duplicateRefine, duplicateGenerate]) })
    }
  })

  it.each(["success", "failure"] as const)("does not release a newer generation after old refinement %s", async outcome => {
    const view = await setupLifetime(), wire = held<ReturnType<typeof completionResponse>>(), send = heldSend()
    mocks.createChatCompletion.mockReturnValueOnce(wire.promise)
    let original!: Promise<void>, current!: Promise<void>
    act(() => { original = view.result.current.handleRefineImagePromptDraft() })
    await waitFor(() => expect(mocks.createChatCompletion).toHaveBeenCalledOnce())
    await act(async () => { view.result.current.closeImageGenerateModal(); view.result.current.openImageGenerateModal() })
    view.deps.sendMessage.mockReturnValueOnce(send.promise)
    act(() => { current = view.result.current.submitImageGenerateModal() })
    expect(view.result.current.imageGenerateSubmitting).toBe(true)
    await act(async () => { if (outcome === "success") wire.resolve(completionResponse("Prompt: retired image")); else wire.reject(new Error("Retired failure")); await original })
    expect(view.result.current.imageGenerateSubmitting).toBe(true)
    expect(view.result.current.imagePromptRefineSubmitting).toBe(false)
    expect(view.result.current.imagePromptRefineCandidate).toBe("")
    expect(mocks.notificationError).not.toHaveBeenCalled()
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
    await act(async () => { send.resolve({ status: "submitted" }); await current })
    expect(view.result.current.imageGenerateBusy).toBe(false)
  })

  it.each(["success", "failure"] as const)("keeps a newer refinement pending after the old refinement %s", async outcome => {
    const view = await setupLifetime(), oldWire = held<ReturnType<typeof completionResponse>>(), newWire = held<ReturnType<typeof completionResponse>>()
    const newRelease = vi.fn()
    mocks.createChatCompletion.mockReturnValueOnce(oldWire.promise).mockReturnValueOnce(newWire.promise)
    let original!: Promise<void>, current!: Promise<void>
    act(() => { original = view.result.current.handleRefineImagePromptDraft() })
    await waitFor(() => expect(mocks.createChatCompletion).toHaveBeenCalledOnce())
    await act(async () => { view.result.current.closeImageGenerateModal(); view.result.current.openImageGenerateModal() })
    mocks.loadServicePromptSnapshot.mockResolvedValueOnce({ ...servicePromptSnapshot(), release: newRelease })
    act(() => { current = view.result.current.handleRefineImagePromptDraft() })
    await waitFor(() => expect(mocks.createChatCompletion).toHaveBeenCalledTimes(2))
    await act(async () => { if (outcome === "success") oldWire.resolve(completionResponse("Prompt: retired image")); else oldWire.reject(new Error("Retired failure")); await original })
    expect(view.result.current.imagePromptRefineSubmitting).toBe(true)
    expect(view.result.current.imagePromptRefineCandidate).toBe("")
    expect(mocks.notificationError).not.toHaveBeenCalled()
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
    expect(newRelease).not.toHaveBeenCalled()
    await act(async () => { newWire.resolve(completionResponse("Prompt: current new image")); await current })
    expect(view.result.current.imagePromptRefineSubmitting).toBe(false)
    expect(view.result.current.imagePromptRefineCandidate).toBe("current new image")
    expect(newRelease).toHaveBeenCalledOnce()
  })

  it.each(["apply", "reject"].flatMap(action => ["refine", "generate"].map(operation => ({ action, operation }))))(
    "refuses $action of a retained candidate during current $operation",
    async ({ action, operation }) => {
      const view = await setupLifetime()
      await act(async () => { await view.result.current.handleRefineImagePromptDraft() })
      const originalCandidate = view.result.current.imagePromptRefineCandidate
      expect(originalCandidate).toBe("current refined image")
      const wire = held<ReturnType<typeof completionResponse>>(), send = heldSend()
      mocks.createChatCompletion.mockReturnValueOnce(wire.promise)
      view.deps.sendMessage.mockReturnValueOnce(send.promise)
      let pending!: Promise<void>
      act(() => { pending = operation === "refine" ? view.result.current.handleRefineImagePromptDraft() : view.result.current.submitImageGenerateModal() })
      if (operation === "refine") await waitFor(() => expect(mocks.createChatCompletion).toHaveBeenCalledTimes(2))
      try {
        act(() => { if (action === "apply") view.result.current.applyRefinedImagePromptCandidate(); else view.result.current.rejectRefinedImagePromptCandidate() })
        expect(view.result.current.imageGeneratePrompt).toBe("Original image")
        expect(view.result.current.imageGenerateRefineMetadata).toBeUndefined()
        expect(view.result.current.imagePromptRefineCandidate).toBe(originalCandidate)
        expect(view.result.current.imageGenerateBusy).toBe(true)
      } finally {
        await act(async () => { wire.resolve(completionResponse("Prompt: newer image")); send.resolve({ status: "submitted" }); await pending })
      }
    }
  )

  it.each(["source", "close", "reopen"] as const)("retires a completed candidate by %s before old or current callbacks can adopt it", async retirement => {
    const view = await setupLifetime()
    await act(async () => { await view.result.current.handleRefineImagePromptDraft() })
    const oldApply = view.result.current.applyRefinedImagePromptCandidate, oldReject = view.result.current.rejectRefinedImagePromptCandidate
    expect(view.result.current.imagePromptRefineCandidate).toBe("current refined image")
    if (retirement === "source") view.retireSource()
    else await act(async () => { view.result.current.closeImageGenerateModal(); if (retirement === "reopen") view.result.current.openImageGenerateModal() })
    expect(view.result.current.imagePromptRefineCandidate).toBe("")
    act(() => { oldApply(); oldReject(); view.result.current.applyRefinedImagePromptCandidate(); view.result.current.rejectRefinedImagePromptCandidate() })
    expect(view.result.current.imageGeneratePrompt).toBe("Original image")
    expect(view.result.current.imageGenerateRefineMetadata).toBeUndefined()
  })

  it.each(["apply", "reject"] as const)("rejects an old $0 callback after a newer candidate publishes in the same modal", async action => {
    const view = await setupLifetime()
    await act(async () => { await view.result.current.handleRefineImagePromptDraft() })
    const old = action === "apply" ? view.result.current.applyRefinedImagePromptCandidate : view.result.current.rejectRefinedImagePromptCandidate
    mocks.createChatCompletion.mockResolvedValueOnce(completionResponse("Prompt: newer candidate"))
    await act(async () => { await view.result.current.handleRefineImagePromptDraft() })
    expect(view.result.current.imagePromptRefineCandidate).toBe("newer candidate")
    act(() => { old() })
    expect(view.result.current.imageGeneratePrompt).toBe("Original image")
    expect(view.result.current.imagePromptRefineCandidate).toBe("newer candidate")
    act(() => { view.result.current.applyRefinedImagePromptCandidate() })
    expect(view.result.current.imageGeneratePrompt).toBe("newer candidate")
    expect(view.result.current.imageGenerateRefineMetadata).toMatchObject({ model: "deepseek-chat" })
    expect(view.result.current.imagePromptRefineCandidate).toBe("")
  })

  it("retires applied refinement metadata with its source while keeping the editable prompt", async () => {
    const view = await setupLifetime()
    await act(async () => { await view.result.current.handleRefineImagePromptDraft() })
    const oldApply = view.result.current.applyRefinedImagePromptCandidate
    act(() => { view.result.current.applyRefinedImagePromptCandidate() })
    expect(view.result.current.imageGeneratePrompt).toBe("current refined image")
    expect(view.result.current.imageGenerateRefineMetadata).toMatchObject({ model: "deepseek-chat" })
    act(() => { view.result.current.setImageGeneratePrompt("Edited after applying") })
    act(() => { oldApply() })
    expect(view.result.current.imageGeneratePrompt).toBe("Edited after applying")
    view.retireSource()
    expect(view.result.current.imageGenerateRefineMetadata).toBeUndefined()
    expect(view.result.current.imageGeneratePrompt).toBe("Edited after applying")
  })

  it("keeps current Reject available after an ordinary refinement failure", async () => {
    const view = await setupLifetime()
    await act(async () => { await view.result.current.handleRefineImagePromptDraft() })
    mocks.createChatCompletion.mockRejectedValueOnce(new Error("Provider unavailable"))
    await act(async () => { await view.result.current.handleRefineImagePromptDraft() })
    expect(view.result.current.imagePromptRefineCandidate).toBe("current refined image")
    expect(view.result.current.imageGenerateBusy).toBe(false)
    act(() => { view.result.current.rejectRefinedImagePromptCandidate() })
    expect(view.result.current.imagePromptRefineCandidate).toBe("")
    expect(view.result.current.imageGeneratePrompt).toBe("Original image")
  })

  it("releases a late snapshot after unmount without continuing the request", async () => {
    const view = await setupLifetime(), snapshot = held<ReturnType<typeof servicePromptSnapshot>>()
    mocks.loadServicePromptSnapshot.mockReturnValueOnce(snapshot.promise)
    let pending!: Promise<void>
    act(() => { pending = view.result.current.handleRefineImagePromptDraft() })
    await waitFor(() => expect(mocks.loadServicePromptSnapshot).toHaveBeenCalledOnce())
    view.unmount()
    await act(async () => { snapshot.resolve(servicePromptSnapshot()); await pending })
    expect(mocks.createChatCompletion).not.toHaveBeenCalled()
    expect(mocks.releaseSnapshot).toHaveBeenCalledOnce()
    expect(mocks.notificationError).not.toHaveBeenCalled()
  })
})
