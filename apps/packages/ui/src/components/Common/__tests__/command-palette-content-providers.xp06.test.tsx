/**
 * XP-06 slice 1: the web palette's Chats and Notes providers search the
 * server and open results at their addresses (/chat?chat=<id>,
 * /notes?note=<id>).
 */
import React from "react"
import { act, renderHook } from "@testing-library/react"
import { MemoryRouter, useLocation } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { useCommandPaletteContentProviders } from "../command-palette-content-providers"

const mocks = vi.hoisted(() => ({
  initialize: vi.fn(async () => undefined),
  searchConversationsWithMeta: vi.fn(),
  searchNotes: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: mocks.initialize,
    searchConversationsWithMeta: mocks.searchConversationsWithMeta,
    searchNotes: mocks.searchNotes
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string | { defaultValue?: string }) =>
      typeof defaultValue === "string" ? defaultValue : defaultValue?.defaultValue ?? key
  })
}))

const renderProviders = () => {
  const location = { current: "" }
  const wrapper = ({ children }: { children: React.ReactNode }) => (
    <MemoryRouter initialEntries={["/media"]}>{children}</MemoryRouter>
  )
  const hook = renderHook(
    () => {
      const current = useLocation()
      location.current = `${current.pathname}${current.search}`
      return useCommandPaletteContentProviders()
    },
    { wrapper }
  )
  return { hook, location }
}

describe("useCommandPaletteContentProviders (XP-06)", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    mocks.searchConversationsWithMeta.mockResolvedValue({
      chats: [
        { id: "c1", title: "Explain LoRA", created_at: "2026-01-01T00:00:00Z" },
        { id: "c2", title: "", created_at: "2026-01-02T00:00:00Z" }
      ],
      total: 2
    })
    mocks.searchNotes.mockResolvedValue({
      items: [
        {
          id: "n1",
          title: "LoRA notes",
          content: "Low-rank adapters",
          snippet: [
            { text: "Fine-tune with ", match: false },
            { text: "LoRA", match: true },
            { text: " adapters", match: false }
          ]
        }
      ]
    })
  })

  it("keeps the same providers across renders", () => {
    const { hook } = renderProviders()
    const first = hook.result.current
    hook.rerender()
    expect(hook.result.current).toBe(first)
  })

  it("searches chats by title and content, five at most, and opens one at its address", async () => {
    const { hook, location } = renderProviders()
    const chats = hook.result.current.find((provider) => provider.id === "chats")!
    const signal = new AbortController().signal

    const items = await chats.search("lora", signal)

    expect(mocks.searchConversationsWithMeta).toHaveBeenCalledWith(
      expect.objectContaining({ query: "lora", search_in: "title,content", limit: 5 }),
      expect.objectContaining({ signal })
    )
    expect(items.map((entry) => [entry.category, entry.label, entry.targetPath])).toEqual([
      ["chat", "Explain LoRA", "/chat?chat=c1"],
      ["chat", "Untitled chat", "/chat?chat=c2"]
    ])

    act(() => items[0].action())
    expect(location.current).toBe("/chat?chat=c1")
  })

  it("searches notes, five at most, and opens one at its address", async () => {
    const { hook, location } = renderProviders()
    const notes = hook.result.current.find((provider) => provider.id === "notes")!
    const signal = new AbortController().signal

    const items = await notes.search("lora", signal)

    // Only `items`: the deprecated `notes`/`results` copies are not needed.
    expect(mocks.searchNotes).toHaveBeenCalledWith(
      "lora",
      { limit: 5, legacy_aliases: false },
      { signal }
    )
    expect(items.map((entry) => [entry.category, entry.label, entry.targetPath])).toEqual([
      ["note", "LoRA notes", "/notes?note=n1"]
    ])
    // The match in context, as plain text.
    expect(items[0].description).toBe("Fine-tune with LoRA adapters")

    act(() => items[0].action())
    expect(location.current).toBe("/notes?note=n1")
  })
})
