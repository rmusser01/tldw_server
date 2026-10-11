import { afterEach, beforeEach, expect, it, vi } from "vitest"
import type { ResearchWebCapture } from "../research-workspace-prefill"

const mocks = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  get: vi.fn(),
  set: vi.fn(),
  lock: vi.fn()
}))
vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    hasPersistentBackend: true,
    get: mocks.get,
    set: mocks.set
  })
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: () => () => {}
}))
const deferred = () => {
  let resolve!: () => void
  const promise = new Promise<void>((done) => { resolve = done })
  return { promise, resolve }
}
const capture = (clipId: string, workspaceId = "workspace-a"): ResearchWebCapture => ({
  ownerScope: "public-owner-a",
  workspaceId,
  sourceId: "original-source",
  body: {
    clip_id: clipId,
    source_url: "https://article.example/story",
    source_title: "Article",
    content: { html: "", text: "Accepted article" },
    workspace: { workspace_id: workspaceId },
    enhancements: {}
  } as ResearchWebCapture["body"]
})
const pin = {
  clipId: "clip-a",
  requestedUrl: "https://article.example/story",
  capturedAt: "2026-10-08T00:00:00Z",
  contentSha256: "a".repeat(64),
  mediaId: 71,
  versionNumber: 9,
  versionUuid: "version-nine"
}
beforeEach(() => {
  vi.resetModules()
  vi.clearAllMocks()
  mocks.values.clear()
  mocks.get.mockImplementation(async (key: string) => structuredClone(mocks.values.get(key)))
  mocks.set.mockImplementation(async (key: string, value: unknown) => {
    mocks.values.set(key, structuredClone(value))
  })
  const queues = new Map<string, Promise<unknown>>()
  mocks.lock.mockImplementation((key: string, work: () => Promise<unknown>) => {
    const next = (queues.get(key) || Promise.resolve()).catch(() => {}).then(work)
    queues.set(key, next)
    return next
  })
  vi.stubGlobal("navigator", Object.create(window.navigator, {
    locks: { value: { request: mocks.lock } }
  }))
})
afterEach(() => vi.unstubAllGlobals())

it.each(["different-clips", "same-clip-progress"])(
  "serializes independent JS contexts for %s without losing durable recovery",
  async (scenario) => {
    const firstContext = await import("../research-workspace-prefill")
    vi.resetModules()
    const secondContext = await import("../research-workspace-prefill")
    expect(firstContext.saveResearchWebCapture).not.toBe(secondContext.saveResearchWebCapture)
    const entered = deferred()
    const release = deferred()
    mocks.get.mockImplementationOnce(async (key: string) => {
      const previous = structuredClone(mocks.values.get(key))
      entered.resolve()
      await release.promise
      return previous
    })
    const first = firstContext.saveResearchWebCapture(capture("clip-a"))
    await entered.promise
    const callsBeforeSecond = mocks.get.mock.calls.length + mocks.lock.mock.calls.length
    const secondRecord = scenario === "different-clips"
      ? capture("clip-b", "workspace-b")
      : { ...capture("clip-a"), pin, attached: true }
    const second = secondContext.saveResearchWebCapture(secondRecord)
    await vi.waitFor(() => expect(
      mocks.get.mock.calls.length + mocks.lock.mock.calls.length
    ).toBeGreaterThan(callsBeforeSecond))
    release.resolve()
    await Promise.all([first, second])
    expect(await firstContext.readResearchWebCaptures("public-owner-a", secondRecord.workspaceId))
      .toContainEqual(secondRecord)
    if (scenario === "different-clips")
      expect(await secondContext.readResearchWebCaptures("public-owner-a", "workspace-a"))
        .toEqual([capture("clip-a")])
    expect(await firstContext.readResearchWebCaptures("other-owner", "workspace-a")).toEqual([])
  }
)
it("stale body-only and pin-only retries cannot erase recorded pin or attachment progress", async () => {
  const context = await import("../research-workspace-prefill")
  const accepted = { ...capture("clip-a"), pin, attached: true }
  await context.saveResearchWebCapture(accepted)
  await context.saveResearchWebCapture(capture("clip-a"))
  await context.saveResearchWebCapture({ ...capture("clip-a"), pin, attached: false })
  expect(await context.readResearchWebCaptures("public-owner-a", "workspace-a")).toEqual([accepted])
})
it("fails closed before writing when cross-context locking is unavailable", async () => {
  vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: undefined } }))
  const context = await import("../research-workspace-prefill")
  await expect(context.saveResearchWebCapture(capture("clip-a"))).rejects.toThrow(/Web Locks/)
  expect(mocks.set).not.toHaveBeenCalled()
})
it("keeps accepted body, destination and exact pin immutable", async () => {
  const context = await import("../research-workspace-prefill")
  const accepted = { ...capture("clip-a"), pin, attached: true }
  await context.saveResearchWebCapture(accepted)
  await expect(context.saveResearchWebCapture({
    ...capture("clip-a"), body: { ...capture("clip-a").body, content: { html: "", text: "Changed" } }
  })).rejects.toThrow(/cannot change/)
  await expect(context.saveResearchWebCapture(capture("clip-a", "workspace-b")))
    .rejects.toThrow(/cannot change/)
  await expect(context.saveResearchWebCapture({ ...capture("clip-a"), pin: { ...pin, versionNumber: 10 } }))
    .rejects.toThrow(/cannot change/)
  expect(await context.readResearchWebCaptures("public-owner-a", "workspace-a")).toEqual([accepted])
})
