import { beforeEach, describe, expect, it, vi } from "vitest"

import {
  extractSourceContentText,
  resolveSourceContentText
} from "../source-content-text"
import { SOURCE_CONTENT_CACHE_MAX_ENTRIES } from "@/store/workspace-slices/sources-slice"
import { useWorkspaceStore } from "@/store/workspace"
import { servicePromptAuthorityKey, type ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"

const getMediaDetailsMock = vi.hoisted(() => vi.fn())

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    getMediaDetails: (...args: unknown[]) => getMediaDetailsMock(...args)
  }
}))

const detailWithText = (text: string) => ({
  content: { text },
  source: { title: "Fetched title" }
})

const resetCache = () => {
  useWorkspaceStore.setState({ sourceContentCache: new Map() })
}

const requestScope: ServicePromptRequestScope = {
  config: { serverUrl: "https://source.test", authMode: "multi-user", orgId: 4 },
  userId: "alice"
}
const scoped = { requestScope }
const authorityKey = servicePromptAuthorityKey(requestScope)

describe("resolveSourceContentText (WP-45 store reuse)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    getMediaDetailsMock.mockReset()
    resetCache()
  })

  it("pins no-network on the second ask for the same (mediaId, version)", async () => {
    getMediaDetailsMock.mockResolvedValue(detailWithText("first fetch text"))

    const first = await resolveSourceContentText(7, scoped)
    expect(first).toEqual({
      text: "first fetch text",
      title: "Fetched title",
      fromCache: false
    })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(1)
    expect(getMediaDetailsMock).toHaveBeenCalledWith(
      7,
      expect.objectContaining({
        include_content: true,
        include_versions: false,
        include_version_content: false
      })
    )

    const second = await resolveSourceContentText(7, scoped)
    expect(second).toEqual({
      text: "first fetch text",
      title: "Fetched title",
      fromCache: true
    })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(1)
  })

  it("fetches only misses", async () => {
    getMediaDetailsMock.mockImplementation(async (mediaId: number) =>
      detailWithText(`text for ${mediaId}`)
    )

    await resolveSourceContentText(7, scoped)
    await resolveSourceContentText(8, scoped)
    await resolveSourceContentText(7, scoped)

    expect(getMediaDetailsMock).toHaveBeenCalledTimes(2)
    expect(getMediaDetailsMock).toHaveBeenCalledWith(8, expect.anything())
  })

  it("does not serve a versioned cache entry to an unversioned ask", async () => {
    useWorkspaceStore.getState().setSourceContentText(7, 3, "versioned text", null, authorityKey)
    getMediaDetailsMock.mockResolvedValue(detailWithText("unversioned text"))

    const unversioned = await resolveSourceContentText(7, scoped)
    expect(unversioned.text).toBe("unversioned text")
    expect(unversioned.fromCache).toBe(false)

    // The unversioned ask misses (version mismatch) and then overwrites the
    // entry under its own identity.
    expect(useWorkspaceStore.getState().getSourceContentText(7, null, authorityKey)).toBe(
      "unversioned text"
    )
    expect(useWorkspaceStore.getState().getSourceContentText(7, 3, authorityKey)).toBeNull()
  })

  it.each([
    ["server", { ...requestScope, config: { ...requestScope.config, serverUrl: "https://other.test" } }],
    ["user", { ...requestScope, userId: "bob" }],
    ["organization", { ...requestScope, config: { ...requestScope.config, orgId: 5 } }],
    ["auth source", { ...requestScope, config: { ...requestScope.config, authSource: "cookie-session" as const } }],
    ["key scope", { ...requestScope, config: { ...requestScope.config, expectedSingleUserApiKeyScope: "other-key-fingerprint" } }]
  ])("does not return another %s authority's text or title for the same media id", async (_label, otherScope) => {
    getMediaDetailsMock.mockResolvedValueOnce(detailWithText("Alice's source"))
      .mockResolvedValueOnce({ content: { text: "Other authority's source" }, source: { title: "Other title" } })
    await resolveSourceContentText(7, scoped)

    expect(await resolveSourceContentText(7, { requestScope: otherScope })).toEqual({
      text: "Other authority's source", title: "Other title", fromCache: false
    })
    expect(await resolveSourceContentText(7, { requestScope: otherScope })).toEqual({
      text: "Other authority's source", title: "Other title", fromCache: true
    })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(2)
  })

  it("reuses an explicit single-user key scope and isolates a rotated key", async () => {
    const singleUser: ServicePromptRequestScope = {
      config: { serverUrl: "https://source.test", authMode: "single-user", expectedSingleUserApiKeyScope: "key-fingerprint" },
      userId: null
    }
    getMediaDetailsMock.mockResolvedValue(detailWithText("Owned single-user source"))
    await resolveSourceContentText(7, { requestScope: singleUser })
    expect(await resolveSourceContentText(7, { requestScope: singleUser })).toMatchObject({
      text: "Owned single-user source", fromCache: true
    })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(1)
    getMediaDetailsMock.mockResolvedValueOnce(detailWithText("Rotated key's source"))
    expect(await resolveSourceContentText(7, {
      requestScope: { ...singleUser, config: { ...singleUser.config, expectedSingleUserApiKeyScope: "rotated-key-fingerprint" } }
    })).toMatchObject({ text: "Rotated key's source", fromCache: false })
  })

  it("retains only the credential-free authority alongside cached source data", async () => {
    getMediaDetailsMock.mockResolvedValue(detailWithText("Owned source"))
    await resolveSourceContentText(7, {
      requestScope: { ...requestScope, config: { ...requestScope.config, expectedRefreshToken: "fixture-refresh-value" } }
    })
    expect(useWorkspaceStore.getState().getSourceContentText(7, null, authorityKey)).toBe("Owned source")
    expect(JSON.stringify([...useWorkspaceStore.getState().sourceContentCache])).not.toContain("fixture-refresh-value")
    expect(useWorkspaceStore.getState().getSourceContentText(7, null)).toBeNull()
  })

  it("falls back to the owned network read when a partial store has no cache actions", async () => {
    getMediaDetailsMock.mockResolvedValue(detailWithText("Uncached partial-store source"))
    const getState = vi.spyOn(useWorkspaceStore, "getState").mockReturnValue({} as ReturnType<typeof useWorkspaceStore.getState>)
    try {
      expect(await resolveSourceContentText(7, scoped)).toEqual({
        text: "Uncached partial-store source", title: "Fetched title", fromCache: false
      })
    } finally {
      getState.mockRestore()
    }
  })

  it.each([
    ["unscoped", undefined],
    ["unresolved user", { ...requestScope, userId: null }],
    ["unresolved key", { config: { serverUrl: "https://source.test", authMode: "single-user" as const }, userId: null }],
    ["unresolved key with user", { config: { serverUrl: "https://source.test", authMode: "single-user" as const }, userId: "alice" }],
    ["blank key scope", { config: { serverUrl: "https://source.test", authMode: "single-user" as const, expectedSingleUserApiKeyScope: " " }, userId: null }],
    ["empty server", { ...requestScope, config: { ...requestScope.config, serverUrl: " " } }]
  ])("keeps %s reads out of the shared cache", async (_label, unknownScope) => {
    getMediaDetailsMock.mockResolvedValueOnce(detailWithText("Known source"))
      .mockResolvedValueOnce(detailWithText("First uncached read"))
      .mockResolvedValueOnce(detailWithText("Second uncached read"))
    await resolveSourceContentText(7, scoped)
    expect(await resolveSourceContentText(7, { requestScope: unknownScope })).toMatchObject({
      text: "First uncached read", fromCache: false
    })
    expect(await resolveSourceContentText(7, { requestScope: unknownScope })).toMatchObject({
      text: "Second uncached read", fromCache: false
    })
    expect(await resolveSourceContentText(7, scoped)).toMatchObject({ text: "Known source", fromCache: true })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(3)
  })

  it("refuses an aborted operation even when its authority has cached text", async () => {
    getMediaDetailsMock.mockResolvedValue(detailWithText("Owned source"))
    await resolveSourceContentText(7, scoped)
    const controller = new AbortController()
    controller.abort()
    await expect(resolveSourceContentText(7, { ...scoped, signal: controller.signal })).rejects.toMatchObject({ name: "AbortError" })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(1)
  })

  it("rejects a retired late read without replacing the new authority's cache", async () => {
    let finishOld!: (value: unknown) => void
    getMediaDetailsMock.mockImplementationOnce(() => new Promise(resolve => { finishOld = resolve }))
      .mockResolvedValueOnce(detailWithText("Bob's current source"))
    const controller = new AbortController()
    const oldRead = resolveSourceContentText(7, { ...scoped, signal: controller.signal })
    controller.abort()
    const bobScope = { requestScope: { ...requestScope, userId: "bob" } }
    expect(await resolveSourceContentText(7, bobScope)).toMatchObject({ text: "Bob's current source", fromCache: false })
    finishOld(detailWithText("Alice's retired source"))
    await expect(oldRead).rejects.toMatchObject({ name: "AbortError" })
    expect(await resolveSourceContentText(7, bobScope)).toMatchObject({ text: "Bob's current source", fromCache: true })
    expect(getMediaDetailsMock).toHaveBeenCalledTimes(2)
  })

  it("propagates fetch failures without caching", async () => {
    getMediaDetailsMock.mockRejectedValue(new Error("detail fetch failed"))

    await expect(resolveSourceContentText(42)).rejects.toThrow(
      "detail fetch failed"
    )
    expect(useWorkspaceStore.getState().getSourceContentText(42, null)).toBeNull()
  })

  it("does not cache empty extractions", async () => {
    getMediaDetailsMock.mockResolvedValue({ content: {} })

    const resolved = await resolveSourceContentText(9)
    expect(resolved.text).toBe("")
    expect(resolved.fromCache).toBe(false)
    expect(useWorkspaceStore.getState().getSourceContentText(9, null)).toBeNull()
  })

  it("bounds the in-memory cache to the entry cap", () => {
    const store = useWorkspaceStore.getState()
    for (let mediaId = 1; mediaId <= SOURCE_CONTENT_CACHE_MAX_ENTRIES + 1; mediaId += 1) {
      store.setSourceContentText(mediaId, null, `text ${mediaId}`)
    }

    const cache = useWorkspaceStore.getState().sourceContentCache
    expect(cache.size).toBe(SOURCE_CONTENT_CACHE_MAX_ENTRIES)
    expect(useWorkspaceStore.getState().getSourceContentText(1, null)).toBeNull()
    expect(useWorkspaceStore.getState().getSourceContentText(
      SOURCE_CONTENT_CACHE_MAX_ENTRIES + 1,
      null
    )).toBe(`text ${SOURCE_CONTENT_CACHE_MAX_ENTRIES + 1}`)
  })

  it("evicts the least-recently-used entry, not the smallest mediaId", () => {
    // A plain object iterates integer-like keys in ascending numeric order,
    // which used to evict the smallest id regardless of insertion recency.
    // The Map-based cache must evict by least-recent use instead: insert
    // mediaId 40 first, then 2..33 — at 33 entries the oldest ask (40) is
    // evicted, NOT the smallest id (2).
    const store = useWorkspaceStore.getState()
    store.setSourceContentText(40, null, "text 40")
    for (let mediaId = 2; mediaId <= SOURCE_CONTENT_CACHE_MAX_ENTRIES + 1; mediaId += 1) {
      store.setSourceContentText(mediaId, null, `text ${mediaId}`)
    }

    expect(useWorkspaceStore.getState().sourceContentCache.size).toBe(
      SOURCE_CONTENT_CACHE_MAX_ENTRIES
    )
    expect(useWorkspaceStore.getState().getSourceContentText(40, null)).toBeNull()
    expect(useWorkspaceStore.getState().getSourceContentText(2, null)).toBe("text 2")
    expect(useWorkspaceStore.getState().getSourceContentText(
      SOURCE_CONTENT_CACHE_MAX_ENTRIES + 1,
      null
    )).toBe(`text ${SOURCE_CONTENT_CACHE_MAX_ENTRIES + 1}`)
  })
})

describe("extractSourceContentText", () => {
  it("reads the first supported text candidate", () => {
    expect(extractSourceContentText({ content: { text: "a" } })).toBe("a")
    expect(extractSourceContentText({ content: "direct string" })).toBe(
      "direct string"
    )
    expect(extractSourceContentText({ content: { transcript: "t" } })).toBe("t")
    expect(extractSourceContentText({ processing: { analysis: "p" } })).toBe("p")
    expect(extractSourceContentText({ content: { text: "   " } })).toBe("")
    expect(extractSourceContentText(null)).toBe("")
  })
})
