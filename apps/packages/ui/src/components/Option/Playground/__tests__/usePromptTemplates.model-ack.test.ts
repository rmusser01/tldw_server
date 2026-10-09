import { act, renderHook } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import {
  usePromptTemplates,
  type UsePromptTemplatesDeps
} from "../hooks/usePromptTemplates"
import { createStartupTemplateBundle } from "../startup-template-bundles"

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
