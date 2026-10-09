import { act, cleanup, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import {
  usePromptTemplates,
  type UsePromptTemplatesDeps
} from "../hooks/usePromptTemplates"
import { createStartupTemplateBundle } from "../startup-template-bundles"
import { useSelectedModel } from "@/hooks/chat/useSelectedModel"
import { useStoreMessageOption } from "@/store/option"
import type { Prompt } from "@/db/dexie/types"

// Actual application hook; these tests declare the persistence/identity boundary.
// Actual model store and device I/O acknowledgement controls live in
// useSelectedModel.persistence.test.tsx. No native storage is claimed here.
const createDeps = (
  overrides: Partial<UsePromptTemplatesDeps> = {}
): UsePromptTemplatesDeps => ({
  startupTemplatesRaw: "[]",
  setStartupTemplatesRaw: vi.fn(),
  promptLibrary: [],
  selectedModel: null,
  systemPrompt: "",
  selectedSystemPrompt: null,
  selectedQuickPrompt: null,
  selectedCharacter: null,
  ragPinnedResults: [],
  currentChatModelSettings: {},
  setSelectedModel: vi.fn(),
  setSelectedSystemPrompt: vi.fn(),
  setSelectedQuickPrompt: vi.fn(),
  setSystemPrompt: vi.fn(),
  applyAssistantSelection: vi.fn(async () => true),
  setRagPinnedResults: vi.fn(),
  updateChatModelSettings: vi.fn(),
  compareModeActive: true,
  setCompareSelectedModels: vi.fn(),
  setModeAnnouncement: vi.fn(),
  t: (key, fallback) => fallback || key,
  ...overrides
})
const bundle = createStartupTemplateBundle({
  name: "Setup",
  selectedModel: "openai:next",
  systemPrompt: "Exact proposed behavior.",
  presetKey: "balanced",
  ragPinnedResults: [{ id: "source-1", snippet: "Evidence" }]
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}

describe("startup preview incarnation", () => {
  it("retires a cancelled preview even when reopening the same bundle in one render batch", () => {
    const deps = createDeps()
    const { result } = renderHook(() => usePromptTemplates(deps))
    act(() => result.current.setStartupTemplatePreview(bundle))
    const older = result.current.startupTemplatePreviewLease!
    act(() => {
      result.current.setStartupTemplatePreview(null)
      result.current.setStartupTemplatePreview(bundle)
    })
    const newer = result.current.startupTemplatePreviewLease!
    expect({
      old: older.isCurrent(),
      new: newer.isCurrent(),
      same: older.id === newer.id
    }).toEqual({ old: false, new: true, same: false })
  })
  it("captures the reviewed source at Open and keeps it distinct from the preview lifetime", () => {
    let sourceRevision = 0
    const captureStartupTemplateSource = vi.fn(() => {
      const original = sourceRevision
      return {
        isCurrent: () => original === sourceRevision,
        continuationIsCurrent: () => true
      }
    })
    const deps = createDeps({ captureStartupTemplateSource })
    const { result } = renderHook(() => usePromptTemplates(deps))
    act(() => result.current.setStartupTemplatePreview(bundle))
    const lease = result.current.startupTemplatePreviewLease!
    sourceRevision++
    expect({
      source: lease.sourceIsCurrent(),
      preview: lease.isCurrent(),
      captures: captureStartupTemplateSource.mock.calls.length
    }).toEqual({ source: false, preview: true, captures: 1 })
  })
  it("retires the preview lease on owner unmount", () => {
    const { result, unmount } = renderHook(() =>
      usePromptTemplates(createDeps())
    )
    act(() => result.current.setStartupTemplatePreview(bundle))
    const lease = result.current.startupTemplatePreviewLease!
    unmount()
    expect(lease.isCurrent()).toBe(false)
  })
})

describe("startup template model and publication acknowledgements", () => {
  it("awaits model persistence before reporting success or publishing later settings", async () => {
    const held = deferred<boolean>()
    const deps = createDeps({ setSelectedModel: vi.fn(() => held.promise) })
    const { result } = renderHook(() => usePromptTemplates(deps))
    let finished = false
    let pending!: Promise<boolean>
    act(() => {
      pending = result.current
        .applyStartupTemplateBundle(bundle)
        .then((value) => {
          finished = true
          return value
        })
    })
    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect({
      finished,
      prompt: vi.mocked(deps.setSystemPrompt).mock.calls.length,
      settings: vi.mocked(deps.updateChatModelSettings).mock.calls.length
    }).toEqual({ finished: false, prompt: 0, settings: 0 })
    await act(async () => {
      held.resolve(true)
      expect(await pending).toBe(true)
    })
  })
  it("does not turn a superseded model write into a successful setup", async () => {
    const deps = createDeps({ setSelectedModel: vi.fn(async () => false) })
    const { result } = renderHook(() => usePromptTemplates(deps))
    let accepted: unknown
    await act(async () => {
      accepted = await result.current.applyStartupTemplateBundle(bundle)
    })
    expect({
      accepted,
      compare: vi.mocked(deps.setCompareSelectedModels).mock.calls.length,
      prompt: vi.mocked(deps.setSystemPrompt).mock.calls.length,
      pins: vi.mocked(deps.setRagPinnedResults).mock.calls.length
    }).toEqual({ accepted: false, compare: 0, prompt: 0, pins: 0 })
  })
  it("keeps a rejected model write observable and stops the remaining publications", async () => {
    const failure = new Error("Device model write rejected")
    const pending = Promise.reject(failure)
    void pending.catch(() => undefined)
    const deps = createDeps({ setSelectedModel: vi.fn(() => pending) })
    const { result } = renderHook(() => usePromptTemplates(deps))
    let rejected: unknown
    await act(async () => {
      await result.current.applyStartupTemplateBundle(bundle).catch((error) => {
        rejected = error
      })
    })
    expect({
      rejected,
      prompt: vi.mocked(deps.setSystemPrompt).mock.calls.length,
      settings: vi.mocked(deps.updateChatModelSettings).mock.calls.length
    }).toEqual({ rejected: failure, prompt: 0, settings: 0 })
  })
  it("refuses a retained Apply whose original source has already retired", async () => {
    const deps = createDeps()
    const { result } = renderHook(() => usePromptTemplates(deps))
    await act(async () => {
      expect(
        await result.current.applyStartupTemplateBundle(bundle, {
          isCurrent: () => false
        })
      ).toBe(false)
    })
    expect(deps.applyAssistantSelection).not.toHaveBeenCalled()
  })
  it.each([
    "setSelectedModel",
    "setCompareSelectedModels",
    "setSelectedSystemPrompt",
    "setSystemPrompt",
    "updateChatModelSettings"
  ] as const)(
    "stops its publication tail when %s synchronously retires the source",
    async (retiringSetter) => {
      let current = true
      const deps = createDeps()
      vi.mocked(deps[retiringSetter]).mockImplementation(() => {
        current = false
      })
      const { result } = renderHook(() => usePromptTemplates(deps))
      let accepted: unknown
      await act(async () => {
        accepted = await result.current.applyStartupTemplateBundle(bundle, {
          isCurrent: () => current
        })
      })
      expect({
        accepted,
        pinnedCalls: vi.mocked(deps.setRagPinnedResults).mock.calls.length
      }).toEqual({ accepted: false, pinnedCalls: 0 })
    }
  )
})


describe("exact startup application Retry with the actual model device boundary", () => {
  beforeEach(() => {
    window.localStorage.setItem(
      "selectedModel",
      JSON.stringify("openai:original")
    )
    useStoreMessageOption.getState().setSelectedModel("openai:original")
  })
  afterEach(() => {
    cleanup()
    vi.restoreAllMocks()
    window.localStorage.removeItem("selectedModel")
  })
  const mountApplication = () => {
    const deps = createDeps()
    const view = renderHook(() => {
      const first = useSelectedModel()
      const second = useSelectedModel()
      return {
        first,
        second,
        templates: usePromptTemplates({
          ...deps,
          selectedModel: first.selectedModel,
          setSelectedModel: first.setSelectedModel,
          beginSelectedModelSave: first.beginSelectedModelSave
        })
      }
    })
    return { ...view, deps }
  }
  const failDeviceWrite = () => {
    let fail = true
    const prototype = Object.getPrototypeOf(
      window.localStorage
    ) as globalThis.Storage
    const original = prototype.setItem
    const writes: string[] = []
    const failure = new Error("Model device write unavailable")
    vi.spyOn(prototype, "setItem").mockImplementation(function (
      this: globalThis.Storage,
      key: string,
      value: string
    ) {
      if (key === "selectedModel") {
        writes.push(value)
        if (fail) throw failure
      }
      return original.call(this, key, value)
    })
    return {
      writes,
      failure,
      recover: () => {
        fail = false
      }
    }
  }
  const ready = async (view: ReturnType<typeof mountApplication>) => {
    await waitFor(() =>
      expect([
        view.result.current.first.selectedModelIsLoading,
        view.result.current.second.selectedModelIsLoading
      ]).toEqual([false, false])
    )
  }
  it("retries the captured model operation after identity ACK without selecting again or resolving a changed library", async () => {
    const view = mountApplication()
    await ready(view)
    const record = {
      id: "library-original",
      title: "Original",
      content: "Frozen reviewed behavior"
    } as Prompt
    view.deps.promptLibrary = [record]
    view.rerender()
    const proposal = createStartupTemplateBundle({
      name: "Exact Retry",
      selectedModel: "openai:next",
      selectedSystemPromptId: record.id,
      systemPrompt: "Saved fallback"
    })
    const context = { applicationId: {}, isCurrent: () => true }
    const device = failDeviceWrite()
    await act(async () => {
      await expect(
        view.result.current.templates.applyStartupTemplateBundle(
          proposal,
          context
        )
      ).rejects.toBe(device.failure)
    })
    const revision = useStoreMessageOption.getState().selectedModelRevision
    record.content = "Library changed after failure"
    device.recover()
    await act(async () => {
      expect(
        await view.result.current.templates.applyStartupTemplateBundle(
          proposal,
          context
        )
      ).toBe(true)
    })
    expect({
      identities: vi.mocked(view.deps.applyAssistantSelection).mock.calls
        .length,
      revision: useStoreMessageOption.getState().selectedModelRevision,
      writes: device.writes,
      prompt: vi
        .mocked(view.deps.setSystemPrompt)
        .mock.calls.map((call) => call[0])
    }).toEqual({
      identities: 1,
      revision,
      writes: [JSON.stringify("openai:next"), JSON.stringify("openai:next")],
      prompt: ["Frozen reviewed behavior"]
    })
  })
  it.each(["other model", "same model newer revision"])(
    "refuses cached model Retry after another hook selects %s",
    async (change) => {
      const view = mountApplication()
      await ready(view)
      const context = { applicationId: {}, isCurrent: () => true }
      const device = failDeviceWrite()
      await act(async () => {
        await view.result.current.templates
          .applyStartupTemplateBundle(bundle, context)
          .catch(() => undefined)
      })
      device.recover()
      await act(async () => {
        expect(
          await view.result.current.second.setSelectedModel(
            change === "other model" ? "openai:newer" : bundle.selectedModel!
          )
        ).toBe(true)
      })
      const revision = useStoreMessageOption.getState().selectedModelRevision
      const writes = device.writes.slice()
      await act(async () => {
        expect(
          await view.result.current.templates.applyStartupTemplateBundle(
            bundle,
            context
          )
        ).toBe(false)
      })
      expect({
        identities: vi.mocked(view.deps.applyAssistantSelection).mock.calls
          .length,
        revision: useStoreMessageOption.getState().selectedModelRevision,
        writes: device.writes
      }).toEqual({ identities: 1, revision, writes })
    }
  )
  it("refuses a changed proposal that incorrectly reuses the failed application ID", async () => {
    const view = mountApplication()
    await ready(view)
    const context = { applicationId: {}, isCurrent: () => true }
    const device = failDeviceWrite()
    await act(async () => {
      await view.result.current.templates
        .applyStartupTemplateBundle(bundle, context)
        .catch(() => undefined)
    })
    device.recover()
    await act(async () => {
      expect(
        await view.result.current.templates.applyStartupTemplateBundle(
          { ...bundle, systemPrompt: "Different proposal" },
          context
        )
      ).toBe(false)
    })
    expect({
      identities: vi.mocked(view.deps.applyAssistantSelection).mock.calls
        .length,
      writes: device.writes
    }).toEqual({
      identities: 1,
      writes: [JSON.stringify(bundle.selectedModel)]
    })
  })
  it("does not treat a destination notification as identity acknowledgement", async () => {
    const view = mountApplication()
    await ready(view)
    let accept = false
    vi.mocked(view.deps.applyAssistantSelection).mockImplementation(
      async (_selection, context) => {
        context?.onDestination?.({
          historyId: null,
          serverChatId: null,
          isCurrent: () => true
        })
        return accept
      }
    )
    const context = { applicationId: {}, isCurrent: () => true }
    await act(async () => {
      expect(
        await view.result.current.templates.applyStartupTemplateBundle(
          bundle,
          context
        )
      ).toBe(false)
    })
    accept = true
    await act(async () => {
      expect(
        await view.result.current.templates.applyStartupTemplateBundle(
          bundle,
          context
        )
      ).toBe(true)
    })
    expect(view.deps.applyAssistantSelection).toHaveBeenCalledTimes(2)
  })
})

describe("terminal cached model receipts", () => {
  it("keeps a false model receipt terminal even if its declared guard remains current", async () => {
    const retry = vi.fn(async () => true)
    const deps = createDeps({
      beginSelectedModelSave: vi.fn(() => ({
        completion: Promise.resolve(false),
        isCurrent: () => true,
        retry
      }))
    })
    const { result } = renderHook(() => usePromptTemplates(deps))
    const context = { applicationId: {}, isCurrent: () => true }
    await act(async () => {
      expect(
        await result.current.applyStartupTemplateBundle(bundle, context)
      ).toBe(false)
    })
    await act(async () => {
      expect(
        await result.current.applyStartupTemplateBundle(bundle, context)
      ).toBe(false)
    })
    expect({
      identity: vi.mocked(deps.applyAssistantSelection).mock.calls.length,
      model: vi.mocked(deps.beginSelectedModelSave!).mock.calls.length,
      retry: retry.mock.calls.length
    }).toEqual({ identity: 1, model: 1, retry: 0 })
  })
})

describe("saved setup completion timing", () => {
  it("does not announce Applied or close an independent preview before the scene caller completes", async () => {
    const deps = createDeps()
    const { result } = renderHook(() => usePromptTemplates(deps))
    act(() => result.current.setStartupTemplatePreview(bundle))
    const lease = result.current.startupTemplatePreviewLease!
    await act(async () => {
      expect(await result.current.handleApplySavedRolePlaySetup(bundle)).toBe(
        true
      )
    })
    expect({
      messages: vi.mocked(deps.setModeAnnouncement).mock.calls.length,
      previewCurrent: lease.isCurrent()
    }).toEqual({ messages: 0, previewCurrent: true })
  })
})
