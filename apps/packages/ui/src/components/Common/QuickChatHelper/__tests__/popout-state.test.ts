import { beforeEach, describe, expect, it, vi } from "vitest"
import { act, renderHook, waitFor } from "@testing-library/react"
import {
  buildQuickChatPopoutState,
  parseQuickChatPopoutState,
  restoreQuickChatPopoutState,
  useQuickChatPopoutRestore
} from "../popout-state"

const authority = vi.hoisted(() => ({ read: vi.fn() }))
vi.mock("@/services/recipe-persistence-uncertainty", () => ({
  resolveRecipePersistenceOwnerView: authority.read
}))
beforeEach(() => {
  authority.read
    .mockReset()
    .mockResolvedValue({ ownerId: "Alice", authorizationRevision: "resolved" })
})

describe("quick chat popout state", () => {
  it("stores normalized source route context", () => {
    const state = buildQuickChatPopoutState(
      {
        messages: [],
        modelOverride: null,
        assistantMode: "docs_rag"
      },
      "chrome-extension://abc/options.html#/research-workspace?tab=chat",
      "Alice"
    )

    expect(state.sourceRoute).toBe("/research-workspace")
  })

  it("parses valid popout state with route context", () => {
    const parsed = parseQuickChatPopoutState(
      {
        messages: [
          {
            id: "m1",
            role: "user",
            content: "How do I do this on this page?",
            timestamp: 123
          }
        ],
        modelOverride: "gpt-4o-mini",
        assistantMode: "docs_rag",
        sourceRoute: "/knowledge",
        ownerId: "Alice"
      },
      "Alice"
    )

    expect(parsed).not.toBeNull()
    expect(parsed?.assistantMode).toBe("docs_rag")
    expect(parsed?.sourceRoute).toBe("/knowledge")
    expect(parsed?.messages).toHaveLength(1)
  })

  it("rejects invalid popout state payloads", () => {
    const parsed = parseQuickChatPopoutState(
      {
        messages: [{ id: 123, role: "user", content: "x", timestamp: 1 }],
        assistantMode: "docs_rag",
        sourceRoute: "/chat",
        ownerId: "Alice"
      },
      "Alice"
    )
    expect(parsed).toBeNull()
  })
  it.each([
    [undefined, "Alice"],
    ["Alice", "Bob"],
    ["Alice", null]
  ])(
    "refuses an unowned or crossed popout snapshot (%s -> %s)",
    (ownerId, currentOwner) => {
      const parsed = parseQuickChatPopoutState(
        {
          messages: [
            {
              id: "private",
              role: "user",
              content: "Alice private question",
              timestamp: 1
            }
          ],
          ownerId,
          modelOverride: "Alice private model",
          assistantMode: "chat"
        },
        currentOwner
      )
      expect(parsed).toBeNull()
    }
  )

  it("refuses a missing creation owner", () => {
    expect(() =>
      buildQuickChatPopoutState(
        { messages: [], modelOverride: null, assistantMode: "chat" },
        "/notes",
        ""
      )
    ).toThrow("account is unavailable")
  })

  it("restores only an admitted current owner", async () => {
    const restore = vi.fn()
    const value = buildQuickChatPopoutState(
      {
        messages: [
          { id: "m", role: "user", content: "Alice private", timestamp: 1 }
        ],
        modelOverride: null,
        assistantMode: "chat"
      },
      "/notes",
      "Alice"
    )
    await restoreQuickChatPopoutState(value, restore, () => true)
    expect(restore).toHaveBeenCalledWith(value)
    authority.read.mockResolvedValueOnce({
      ownerId: "Bob",
      authorizationRevision: "resolved"
    })
    restore.mockClear()
    await restoreQuickChatPopoutState(value, restore, () => true)
    expect(restore).not.toHaveBeenCalled()
  })

  it.each(["principal", "storage", "unmount"])(
    "fences a delayed restore across %s",
    async (change) => {
      let resolve!: (owner: {
        ownerId: string
        authorizationRevision: string
      }) => void
      const held = new Promise<{
        ownerId: string
        authorizationRevision: string
      }>((yes) => {
        resolve = yes
      })
      authority.read.mockReturnValueOnce(held)
      let mounted = true
      const restore = vi.fn()
      const value = buildQuickChatPopoutState(
        {
          messages: [
            { id: "m", role: "user", content: "Alice private", timestamp: 1 }
          ],
          modelOverride: null,
          assistantMode: "chat"
        },
        "/notes",
        "Alice"
      )
      const pending = restoreQuickChatPopoutState(value, restore, () => mounted)
      const settled = pending.catch(() => undefined)
      try {
        if (change === "principal")
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        if (change === "storage")
          window.dispatchEvent(
            new StorageEvent("storage", {
              key: "tldwConfig",
              oldValue: JSON.stringify({
                serverUrl: "http://old",
                apiKey: "old"
              }),
              newValue: JSON.stringify({
                serverUrl: "http://new",
                apiKey: "new"
              })
            })
          )
        if (change === "unmount") mounted = false
        resolve({ ownerId: "Alice", authorizationRevision: "resolved" })
        await settled
        expect(restore).not.toHaveBeenCalled()
      } finally {
        resolve({ ownerId: "Alice", authorizationRevision: "resolved" })
        await settled
      }
    }
  )

  it("retains an admitted transfer during an authority-preserving refresh", async () => {
    const restore = vi.fn()
    const value = buildQuickChatPopoutState(
      { messages: [], modelOverride: null, assistantMode: "chat" },
      "/notes",
      "Alice"
    )
    const pending = restoreQuickChatPopoutState(value, restore, () => true)
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: false }
      })
    )
    await pending
    expect(restore).toHaveBeenCalledWith(value)
  })
})

it("retires a held first key when a different transfer becomes current", async () => {
  sessionStorage.clear()
  let resolve!: (owner: {
    ownerId: string
    authorizationRevision: string
  }) => void
  const held = new Promise<{ ownerId: string; authorizationRevision: string }>(
    (yes) => {
      resolve = yes
    }
  )
  authority.read.mockReturnValueOnce(held)
  const first = buildQuickChatPopoutState(
    {
      messages: [{ id: "old", role: "user", content: "First", timestamp: 1 }],
      modelOverride: null,
      assistantMode: "chat"
    },
    "/notes",
    "Alice"
  )
  const second = { ...first, sourceRoute: "/media" }
  const firstKey = "quickchat_1760000000001",
    secondKey = "quickchat_1760000000002"
  sessionStorage.setItem(firstKey, JSON.stringify(first))
  sessionStorage.setItem(secondKey, JSON.stringify(second))
  const restore = vi.fn()
  const view = renderHook(
    ({ stateKey }) => useQuickChatPopoutRestore(stateKey, restore),
    { initialProps: { stateKey: firstKey } }
  )
  try {
    await waitFor(() => expect(authority.read).toHaveBeenCalledOnce())
    view.rerender({ stateKey: secondKey })
    await waitFor(() => expect(restore).toHaveBeenCalledExactlyOnceWith(second))
    await act(async () => {
      resolve({ ownerId: "Alice", authorizationRevision: "resolved" })
      await held
    })
    expect(restore).toHaveBeenCalledExactlyOnceWith(second)
    expect(sessionStorage.getItem(firstKey)).toBeNull()
    expect(sessionStorage.getItem(secondKey)).toBeNull()
  } finally {
    view.unmount()
    await act(async () => {
      resolve({ ownerId: "Alice", authorizationRevision: "resolved" })
      await held
    })
    sessionStorage.clear()
  }
})
