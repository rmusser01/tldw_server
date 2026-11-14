import { act, renderHook } from "@testing-library/react"
import { beforeEach, expect, it, vi } from "vitest"
import { webcrypto } from "node:crypto"
import { useResearchWebCapture } from "../use-research-web-capture"
import {
  getResearchWorkspaceOwner,
  readResearchWebCaptures
} from "../research-workspace-prefill"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { useWorkspaceStore } from "@/store/workspace"

const mocks = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  mode: "single-user" as "single-user" | "multi-user",
  origin: "https://capture.example",
  key: "synthetic-test-key",
  principal: "alice",
  verifiedUser: "alice",
  extract: vi.fn(),
  save: vi.fn(),
  getUser: vi.fn(),
  resolveConfig: vi.fn()
}))
const config = () => ({
  serverUrl: mocks.origin,
  authMode: mocks.mode,
  apiKey: mocks.key,
  accessToken:
    mocks.mode === "multi-user"
      ? `fixture.${btoa(JSON.stringify({ sub: mocks.principal }))}.signature`
      : undefined,
  orgId: "org-test"
})
vi.mock("@/utils/safe-storage", async (original) => ({
  ...(await original<typeof import("@/utils/safe-storage")>()),
  createSafeStorage: () => ({
    hasPersistentBackend: true,
    getAll: async () => Object.fromEntries(mocks.values),
    get: async (key: string) => structuredClone(mocks.values.get(key)),
    set: async (key: string, value: unknown) => {
      mocks.values.set(key, structuredClone(value));
    },
    watch: vi.fn(),
    unwatch: vi.fn(),
  }),
}));
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getConfig: async () => config(),
    initialize: async () => {},
    ensureConfigForRequest: () => mocks.resolveConfig(),
    saveWebClip: (...args: unknown[]) => mocks.save(...args)
  }
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: () => mocks.getUser() }
}))
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))
vi.mock("@/services/tldw/TldwMedia", () => ({
  tldwMedia: {
    extractPublicArticle: (...args: unknown[]) => mocks.extract(...args)
  }
}))
vi.mock("@/store/workspace", async () => {
  const { create } = await import("zustand")
  return {
    useWorkspaceStore: create(() => ({
      workspaceId: "workspace",
      sources: [],
      selectedSourceIds: [],
      selectedSourceFolderIds: []
    }))
  }
})
const original = {
  id: "original",
  mediaId: 1,
  title: "Result",
  type: "website" as const,
  url: "https://article.example/story",
  addedAt: new Date()
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
beforeEach(() => {
  vi.clearAllMocks()
  Object.defineProperty(globalThis, "crypto", {
    value: webcrypto,
    configurable: true
  })
  mocks.values.clear()
  mocks.mode = "single-user"
  mocks.origin = "https://capture.example"
  mocks.key = "synthetic-test-key"
  mocks.principal = mocks.verifiedUser = "alice"
  mocks.resolveConfig.mockImplementation(async () => config())
  mocks.getUser.mockImplementation(async () => ({ id: mocks.verifiedUser }))
  mocks.extract.mockResolvedValue({
    results: [
      {
        title: "Extracted",
        content: " complete text ",
        extraction_successful: true
      }
    ]
  })
  mocks.save.mockRejectedValue(new Error("Synthetic lost acknowledgement"))
  useWorkspaceStore.setState({
    workspaceId: "workspace",
    sources: [original],
    selectedSourceIds: [original.id],
    selectedSourceFolderIds: []
  })
})
it.each(["single-user", "multi-user"] as const)(
  "real %s snapshot is compatible with the unchanged public owner and explicit extraction",
  async (mode) => {
    mocks.mode = mode
    const publicOwner = await getResearchWorkspaceOwner()
    const snapshot = await loadServicePromptSnapshot([])
    expect(publicOwner).toBe(
      buildChatSurfaceScopeKeyFromConfig({ ...config(), apiKey: undefined })
    )
    if (mode === "single-user") expect(snapshot.scopeKey).not.toBe(publicOwner)
    else expect(snapshot.requestScope.userId).toBe("alice")
    snapshot.release()
    const view = renderHook(() => useResearchWebCapture("workspace"))
    await act(async () => {})
    expect(mocks.extract).not.toHaveBeenCalled()
    await act(async () => view.result.current.open(original))
    expect(view.result.current.error).toBeNull()
    expect(mocks.extract).not.toHaveBeenCalled()
    await act(async () => view.result.current.extract())
    expect(mocks.extract).toHaveBeenCalledOnce()
    const options = mocks.extract.mock.calls[0][1]
    expect(options.requestScope.config).not.toHaveProperty("apiKey")
    if (mode === "single-user")
      expect(options.requestScope.config.expectedSingleUserApiKeyScope).toMatch(
        /^key:sha256:/
      )
    else expect(options.requestScope.userId).toBe("alice")
    expect(view.result.current.preview?.text).toBe(" complete text ")
  }
)
it.each(["single-user", "multi-user"] as const)(
  "%s retries the original public-owner acceptance without creating a credential-key namespace",
  async (mode) => {
    mocks.mode = mode
    const owner = await getResearchWorkspaceOwner()
    const view = renderHook(() => useResearchWebCapture("workspace"))
    await act(async () => view.result.current.open(original))
    await act(async () => view.result.current.extract())
    await act(async () => view.result.current.save())
    expect(mocks.save).toHaveBeenCalledOnce()
    const accepted = mocks.save.mock.calls[0][0]
    expect([...mocks.values.keys()]).toEqual([
      `__tldw_research_workspace_prefill:${owner}:web-captures:${accepted.clip_id}`,
    ]);
    expect((await readResearchWebCaptures(owner, "workspace"))[0].body).toEqual(
      accepted
    )
    view.unmount()
    const reopened = renderHook(() => useResearchWebCapture("workspace"))
    await act(async () => reopened.result.current.open(original))
    await act(async () => reopened.result.current.save())
    expect(mocks.save.mock.calls[1][0]).toEqual(accepted)
    expect(mocks.extract).toHaveBeenCalledOnce()
  }
)
it.each(["origin", "principal"])(
  "rejects a different verified %s between public owner capture and snapshot resolution",
  async (change) => {
    mocks.mode = "multi-user"
    mocks.resolveConfig.mockImplementation(async () => {
      if (change === "origin") mocks.origin = "https://other.example"
      else mocks.verifiedUser = "bob"
      return config()
    })
    const view = renderHook(() => useResearchWebCapture("workspace"))
    await act(async () => view.result.current.open(original))
    await act(async () => view.result.current.extract())
    expect(mocks.extract).not.toHaveBeenCalled()
    expect(view.result.current.error).toBe("Capture account changed")
  }
)
it.each(["origin", "credential", "principal"])(
  "%s retirement aborts the real snapshot lease and discards a late extraction",
  async (change) => {
    if (change === "principal") mocks.mode = "multi-user"
    const gate = deferred<unknown>()
    mocks.extract.mockReturnValue(gate.promise)
    const view = renderHook(() => useResearchWebCapture("workspace"))
    await act(async () => view.result.current.open(original))
    let extracting!: Promise<void>
    act(() => {
      extracting = view.result.current.extract()
    })
    await act(async () => {})
    expect(mocks.extract).toHaveBeenCalledOnce()
    const signal = mocks.extract.mock.calls[0][1].signal as AbortSignal
    const previous = config()
    act(() => {
      if (change === "origin") mocks.origin = "https://other.example"
      if (change === "credential") mocks.key = "synthetic-replacement-key"
      if (change === "principal") mocks.principal = mocks.verifiedUser = "bob"
      window.dispatchEvent(
        new StorageEvent("storage", {
          key: "tldwConfig",
          oldValue: JSON.stringify(previous),
          newValue: JSON.stringify(config())
        })
      )
    })
    expect(signal.aborted).toBe(true)
    await act(async () => {
      gate.resolve({
        results: [{ content: "late text", extraction_successful: true }]
      })
      await extracting
    })
    expect(view.result.current.preview).toBeNull()
    expect(mocks.save).not.toHaveBeenCalled()
  }
)
it("late accepted save remains under its original public owner after principal retirement", async () => {
  mocks.mode = "multi-user"
  const owner = await getResearchWorkspaceOwner()
  const gate = deferred<unknown>()
  mocks.save.mockReturnValue(gate.promise)
  const view = renderHook(() => useResearchWebCapture("workspace"))
  await act(async () => view.result.current.open(original))
  await act(async () => view.result.current.extract())
  let saving!: Promise<void>
  act(() => {
    saving = view.result.current.save()
  })
  await vi.waitFor(() => expect(mocks.save).toHaveBeenCalledOnce())
  const accepted = mocks.save.mock.calls[0][0]
  act(() => {
    mocks.principal = mocks.verifiedUser = "bob"
    window.dispatchEvent(new Event("tldw:auth-principal-changed"))
  })
  await act(async () => {
    gate.resolve({ status: "saved" })
    await saving
  })
  expect((await readResearchWebCaptures(owner, "workspace"))[0].body).toEqual(
    accepted
  )
  expect(
    await readResearchWebCaptures(
      await getResearchWorkspaceOwner(),
      "workspace"
    )
  ).toEqual([])
  expect(useWorkspaceStore.getState().sources).toEqual([original])
})
