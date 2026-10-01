import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import * as mediaHandoff from "@/services/tldw/media-chat-handoff"

const runtime = vi.hoisted(() => ({ extension: false }))
vi.mock("@/utils/browser-runtime", () => ({
  isExtensionRuntime: () => runtime.extension,
}))
vi.mock(
  "@plasmohq/storage",
  () => import("../../../../../tldw-frontend/extension/shims/plasmo-storage"),
)

import {
  buildDiscussMediaHint,
  getMediaChatHandoffMode,
  normalizeMediaChatHandoffPayload,
  parseMediaIdAsNumber,
} from "@/services/tldw/media-chat-handoff"

describe("media chat handoff helpers", () => {
  it("normalizes payload and keeps supported mode", () => {
    const payload = normalizeMediaChatHandoffPayload({
      mediaId: "42",
      url: "https://example.com/video",
      title: "Demo",
      content: "Summary text",
      mode: "rag_media",
      ownerScope: "server:alice",
    })

    expect(payload).toEqual({
      mediaId: "42",
      url: "https://example.com/video",
      title: "Demo",
      content: "Summary text",
      mode: "rag_media",
      ownerScope: "server:alice",
    })
  })

  it("defaults mode to normal when mode is not provided", () => {
    const payload = normalizeMediaChatHandoffPayload({
      mediaId: "9",
    })
    expect(payload).toEqual({ mediaId: "9" })
    expect(getMediaChatHandoffMode(payload || {})).toBe("normal")
  })

  it("parses numeric media id and rejects invalid values", () => {
    expect(parseMediaIdAsNumber({ mediaId: "123" })).toBe(123)
    expect(parseMediaIdAsNumber({ mediaId: "abc" })).toBeNull()
    expect(parseMediaIdAsNumber({ mediaId: "-5" })).toBeNull()
  })

  it("builds hint text from structured payload content", () => {
    expect(
      buildDiscussMediaHint({
        mediaId: "7",
        title: "Weekly Meeting",
        content: "Transcript excerpt",
      }),
    ).toContain("Chat with this media: Weekly Meeting")

    expect(
      buildDiscussMediaHint({
        mediaId: "7",
      }),
    ).toBe("Let's talk about media 7.")
  })
})

describe("tab-owned media handoff delivery", () => {
  const source = {
    ownerScope: "server:alice",
    mediaId: "1",
    title: "Source",
    content: " \nFull source\n\t ",
    mode: "normal" as const,
  }
  const makeTab = () => {
    const values = new Map<string, string>()
    return {
      getItem: (key: string) => values.get(key) ?? null,
      setItem: (key: string, value: string) => {
        values.set(key, value)
      },
      removeItem: (key: string) => {
        values.delete(key)
      },
    }
  }
  beforeEach(() => {
    runtime.extension = false
    localStorage.clear()
    sessionStorage.clear()
  })
  afterEach(() => {
    vi.restoreAllMocks()
    vi.unstubAllGlobals()
    vi.useRealTimers()
  })

  it("keeps a source in its initiating tab and delivers every character once", async () => {
    const initiating = makeTab(),
      unrelated = makeTab()
    const tab = vi
      .spyOn(window, "sessionStorage", "get")
      .mockReturnValue(initiating as Storage)
    const token = await mediaHandoff.createMediaChatHandoff(source)
    expect(mediaHandoff.buildMediaChatHandoffRoute(token)).toMatch(
      /^\/chat\?media_handoff=[a-z0-9-]+$/i,
    )
    expect(localStorage.length).toBe(0)
    tab.mockReturnValue(unrelated as Storage)
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ).toBeNull()
    tab.mockReturnValue(initiating as Storage)
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ).toEqual(source)
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ).toBeNull()
  })

  it("does not consume an unresolved or different owner's source", async () => {
    const token = await mediaHandoff.createMediaChatHandoff(source)
    expect(await mediaHandoff.readMediaChatHandoff(token, null)).toBeNull()
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:bob"),
    ).toBeNull()
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ).toEqual(source)
  })

  it("requires an owned source and never falls back to global storage", async () => {
    await expect(
      mediaHandoff.createMediaChatHandoff({ content: "Private" }),
    ).rejects.toThrow(/account/i)
    vi.spyOn(window, "sessionStorage", "get").mockImplementation(() => {
      throw new Error("Blocked")
    })
    await expect(mediaHandoff.createMediaChatHandoff(source)).rejects.toThrow(
      /storage/i,
    )
    expect(localStorage.length).toBe(0)
  })

  it("expires abandoned transfers and retains independent tokens", async () => {
    vi.useFakeTimers()
    const first = await mediaHandoff.createMediaChatHandoff(source)
    vi.advanceTimersByTime(mediaHandoff.MEDIA_CHAT_HANDOFF_TTL_MS + 1)
    const next = await mediaHandoff.createMediaChatHandoff({
      ...source,
      content: "Newer source",
    })
    expect(
      await mediaHandoff.consumeMediaChatHandoff(first, "server:alice"),
    ).toBeNull()
    expect(
      (await mediaHandoff.consumeMediaChatHandoff(next, "server:alice"))
        ?.content,
    ).toBe("Newer source")
  })

  it("preserves all selected media IDs and unbounded source content", async () => {
    const payload = {
      ...source,
      mediaIds: [1, 2],
      mode: "rag_media" as const,
      content: "a".repeat(32_001),
    }
    const token = await mediaHandoff.createMediaChatHandoff(payload)
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ).toEqual(payload)
  })

  it("leaves a source available when the destination rejects it after storage resolves", async () => {
    const token = await mediaHandoff.createMediaChatHandoff(source)
    expect(
      await mediaHandoff.consumeMediaChatHandoff(
        token,
        "server:alice",
        () => false,
      ),
    ).toBeNull()
    const claims = await Promise.all([
      mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
      mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ])
    expect(claims.filter(Boolean)).toEqual([source])
  })

  it("requires extension locking before creating a cross-tab transfer", async () => {
    runtime.extension = true
    vi.stubGlobal("navigator", { locks: undefined })
    await expect(
      mediaHandoff.createMediaChatHandoff(source, { newTab: true }),
    ).rejects.toThrow(/storage/i)
    expect(localStorage.length).toBe(0)
  })

  it("claims extension new-tab tokens once under the same owner without affecting other tokens", async () => {
    runtime.extension = true
    const tails = new Map<string, Promise<unknown>>()
    vi.stubGlobal(
      "navigator",
      Object.create(window.navigator, {
        locks: {
          value: {
            request: (key: string, work: () => unknown) => {
              const next = (tails.get(key) ?? Promise.resolve()).then(work)
              tails.set(
                key,
                next.catch(() => undefined),
              )
              return next
            },
          },
        },
      }),
    )
    const token = await mediaHandoff.createMediaChatHandoff(source, {
      newTab: true,
    })
    const other = await mediaHandoff.createMediaChatHandoff(
      { ...source, content: "Another source" },
      { newTab: true },
    )
    expect(
      await mediaHandoff.consumeMediaChatHandoff(token, "server:bob"),
    ).toBeNull()
    const claims = await Promise.all([
      mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
      mediaHandoff.consumeMediaChatHandoff(token, "server:alice"),
    ])
    expect(claims.filter(Boolean)).toEqual([source])
    expect(
      (await mediaHandoff.consumeMediaChatHandoff(other, "server:alice"))
        ?.content,
    ).toBe("Another source")
    vi.useFakeTimers()
    const expired = await mediaHandoff.createMediaChatHandoff(source, {
      newTab: true,
    })
    vi.advanceTimersByTime(mediaHandoff.MEDIA_CHAT_HANDOFF_TTL_MS + 1)
    expect(
      await mediaHandoff.consumeMediaChatHandoff(expired, "server:alice"),
    ).toBeNull()
  })
})
