import { beforeEach, describe, expect, it, vi } from "vitest"

const storageState = vi.hoisted(() => {
  const store = new Map<string, unknown>()

  return {
    store,
    get: vi.fn(async (key: string) => store.get(key)),
    set: vi.fn(async (key: string, value: unknown) => {
      store.set(key, value)
    }),
    remove: vi.fn(async (key: string) => {
      store.delete(key)
    }),
    removeMany: vi.fn(async (keys: string[]) => {
      keys.forEach((key) => store.delete(key))
    }),
    getAll: vi.fn(async () => Object.fromEntries(store.entries()))
  }
})

vi.mock("@/utils/safe-storage", () => ({
  createSafeStorage: () => ({
    get: storageState.get,
    set: storageState.set,
    remove: storageState.remove,
    removeMany: storageState.removeMany,
    getAll: storageState.getAll
  })
}))

import {
  buildSidepanelChatHandoffRoute,
  buildSidepanelHandoffMessageForModel,
  cleanupExpiredSidepanelChatHandoffs,
  consumeSidepanelChatHandoff,
  createSidepanelChatHandoff,
  readSidepanelChatHandoff,
  SIDEPANEL_CHAT_HANDOFF_MAX_DRAFT_CHARS,
  SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPET_CHARS,
  SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPETS,
  SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX,
  SIDEPANEL_CHAT_HANDOFF_TTL_MS,
  type SidepanelChatHandoffPackage
} from "@/services/sidepanel-chat-handoff"

const NOW = new Date("2026-05-28T18:30:00.000Z")

const storageKey = (id: string) => `${SIDEPANEL_CHAT_HANDOFF_STORAGE_PREFIX}${id}`

const buildStoredPackage = (
  overrides: Partial<SidepanelChatHandoffPackage> = {}
): SidepanelChatHandoffPackage => ({
  id: "handoff-existing",
  source: "sidepanel-chat",
  createdAt: NOW.toISOString(),
  expiresAt: new Date(NOW.getTime() + SIDEPANEL_CHAT_HANDOFF_TTL_MS).toISOString(),
  draft: { text: "saved draft" },
  ...overrides
})

const getQueryParams = (route: string) => {
  const queryIndex = route.indexOf("?")
  return new URLSearchParams(queryIndex >= 0 ? route.slice(queryIndex + 1) : "")
}

describe("sidepanel chat handoff storage service", () => {
  beforeEach(() => {
    vi.useFakeTimers()
    vi.setSystemTime(NOW)
    vi.clearAllMocks()
    storageState.store.clear()
  })

  it("creates bounded packages and verifies read-back before returning the id", async () => {
    const pkg = await createSidepanelChatHandoff({
      draftText: "D".repeat(SIDEPANEL_CHAT_HANDOFF_MAX_DRAFT_CHARS + 12),
      pageContext: {
        title: "Reference Page",
        url: "https://example.test/article",
        snippets: [
          {
            kind: "selection",
            label: "selection",
            text: "S".repeat(SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPET_CHARS + 20)
          },
          { kind: "visible-context", text: "visible context" },
          { kind: "captured-snippet", text: "captured context" },
          { kind: "selection", text: "second selection" },
          { kind: "captured-snippet", text: "dropped extra snippet" }
        ]
      },
      routeIntent: {
        path: "/chat",
        mode: "character",
        characterId: "char-1"
      }
    })

    expect(pkg.id).toBeTruthy()
    expect(pkg.source).toBe("sidepanel-chat")
    expect(pkg.createdAt).toBe(NOW.toISOString())
    expect(pkg.expiresAt).toBe(
      new Date(NOW.getTime() + SIDEPANEL_CHAT_HANDOFF_TTL_MS).toISOString()
    )
    expect(pkg.draft.text).toHaveLength(SIDEPANEL_CHAT_HANDOFF_MAX_DRAFT_CHARS)
    expect(pkg.draft.truncated).toBe(true)
    expect(pkg.pageContext?.snippets).toHaveLength(SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPETS)
    expect(pkg.pageContext?.snippets[0].text).toHaveLength(
      SIDEPANEL_CHAT_HANDOFF_MAX_SNIPPET_CHARS
    )
    expect(pkg.pageContext?.snippets[0].truncated).toBe(true)
    expect(pkg.pageContext?.truncated).toBe(true)
    expect(storageState.set).toHaveBeenCalledWith(storageKey(pkg.id), pkg)
    expect(storageState.get).toHaveBeenCalledWith(storageKey(pkg.id))

    await expect(readSidepanelChatHandoff(pkg.id)).resolves.toEqual(pkg)
  })

  it("throws and does not return a handoff id when storage set fails", async () => {
    storageState.set.mockRejectedValueOnce(new Error("storage quota exceeded"))

    await expect(
      createSidepanelChatHandoff({ draftText: "unsaved draft" })
    ).rejects.toThrow("storage quota exceeded")

    expect(storageState.store.size).toBe(0)
    expect(storageState.remove).toHaveBeenCalledTimes(1)
  })

  it("throws when read-back verification cannot read the saved package", async () => {
    storageState.get.mockResolvedValueOnce(null)

    await expect(
      createSidepanelChatHandoff({ draftText: "write without read-back" })
    ).rejects.toThrow("Sidepanel chat handoff could not be saved.")

    expect(storageState.store.size).toBe(0)
    expect(storageState.remove).toHaveBeenCalledTimes(1)
  })

  it("returns null and removes expired or malformed packages", async () => {
    const expired = buildStoredPackage({
      id: "expired",
      expiresAt: new Date(NOW.getTime() - 1).toISOString()
    })
    storageState.store.set(storageKey("expired"), expired)
    storageState.store.set(storageKey("malformed"), {
      id: "malformed",
      source: "sidepanel-chat"
    })

    await expect(readSidepanelChatHandoff("expired")).resolves.toBeNull()
    await expect(readSidepanelChatHandoff("malformed")).resolves.toBeNull()

    expect(storageState.store.has(storageKey("expired"))).toBe(false)
    expect(storageState.store.has(storageKey("malformed"))).toBe(false)

    storageState.store.set(storageKey("cleanup-expired"), expired)
    storageState.store.set(storageKey("cleanup-malformed"), { id: "bad" })

    await expect(cleanupExpiredSidepanelChatHandoffs()).resolves.toBe(2)
    expect(storageState.store.has(storageKey("cleanup-expired"))).toBe(false)
    expect(storageState.store.has(storageKey("cleanup-malformed"))).toBe(false)
  })

  it("keeps valid serialized packages during cleanup and removes expired or malformed serialized packages", async () => {
    const validSerialized = buildStoredPackage({ id: "serialized-valid" })
    const expiredSerialized = buildStoredPackage({
      id: "serialized-expired",
      expiresAt: new Date(NOW.getTime() - 1).toISOString()
    })

    storageState.store.set(storageKey("serialized-valid"), JSON.stringify(validSerialized))
    storageState.store.set(
      storageKey("serialized-expired"),
      JSON.stringify(expiredSerialized)
    )
    storageState.store.set(
      storageKey("serialized-malformed"),
      JSON.stringify({ id: "serialized-malformed" })
    )

    await expect(cleanupExpiredSidepanelChatHandoffs()).resolves.toBe(2)

    expect(storageState.store.has(storageKey("serialized-valid"))).toBe(true)
    expect(storageState.store.has(storageKey("serialized-expired"))).toBe(false)
    expect(storageState.store.has(storageKey("serialized-malformed"))).toBe(false)
    await expect(readSidepanelChatHandoff("serialized-valid")).resolves.toEqual(
      validSerialized
    )
  })

  it("consumes a package exactly once", async () => {
    const pkg = await createSidepanelChatHandoff({
      draftText: "consume this once"
    })

    await expect(consumeSidepanelChatHandoff(pkg.id)).resolves.toEqual(pkg)
    await expect(readSidepanelChatHandoff(pkg.id)).resolves.toBeNull()
    await expect(consumeSidepanelChatHandoff(pkg.id)).resolves.toBeNull()
  })

  it("merges handoff into normal and character /chat hash routes", () => {
    const normalRoute = buildSidepanelChatHandoffRoute("#/chat", "handoff-normal")
    const normalParams = getQueryParams(normalRoute)
    expect(normalRoute).toBe("#/chat?handoff=handoff-normal")
    expect(normalParams.get("handoff")).toBe("handoff-normal")

    const characterRoute = buildSidepanelChatHandoffRoute(
      "#/chat?mode=character&characterId=char-1",
      "handoff-character"
    )
    const characterParams = getQueryParams(characterRoute)
    expect(characterRoute.startsWith("#/chat?")).toBe(true)
    expect(characterParams.get("mode")).toBe("character")
    expect(characterParams.get("characterId")).toBe("char-1")
    expect(characterParams.get("handoff")).toBe("handoff-character")
  })

  it("never puts draft text or snippet text into the route", async () => {
    const pkg = await createSidepanelChatHandoff({
      draftText: "selected-secret-draft",
      pageContext: {
        snippets: [{ kind: "selection", text: "snippet-secret-text" }]
      }
    })

    const route = buildSidepanelChatHandoffRoute(
      "#/chat?mode=character&characterId=char-1",
      pkg.id
    )

    expect(route).toContain(`handoff=${encodeURIComponent(pkg.id)}`)
    expect(route).not.toContain("selected-secret-draft")
    expect(route).not.toContain("snippet-secret-text")
    expect(route).not.toContain("draft")
    expect(route).not.toContain("snippet")
  })

  it("builds messageForModel with visible sidepanel context", () => {
    const message = buildSidepanelHandoffMessageForModel("What should I inspect?", {
      title: "Debugging Guide",
      url: "https://example.test/debugging",
      snippets: [
        {
          kind: "selection",
          label: "selected text",
          text: "The failing assertion mentions read-back verification."
        },
        {
          kind: "visible-context",
          text: "The visible paragraph describes storage cleanup."
        }
      ]
    })

    expect(message).toBe(
      [
        "Sidepanel page context:",
        "Title: Debugging Guide",
        "URL: https://example.test/debugging",
        "Snippet 1 (selected text): The failing assertion mentions read-back verification.",
        "Snippet 2: The visible paragraph describes storage cleanup.",
        "User draft:",
        "What should I inspect?"
      ].join("\n")
    )
  })
})
