// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen, within } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ConversationContextPopover } from "../ConversationContextPopover"
import type { ConversationContextComposition } from "@/types/conversation-context"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback || _key
  })
}))

vi.mock("antd", async () => {
  const ReactModule =
    await vi.importActual<typeof import("react")>("react")

  return {
    Popover: ({
      children,
      content
    }: {
      children: React.ReactNode
      content: React.ReactNode
    }) => (
      <div>
        {children}
        <div data-testid="conversation-context-popover-content">
          {content}
        </div>
      </div>
    ),
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>
  }
})

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(async () => undefined),
    listWorldBooks: vi.fn(async () => ({ world_books: [] })),
    listDictionaries: vi.fn(async () => ({ dictionaries: [] }))
  }
}))

vi.mock("../CharacterSelect", () => ({
  CharacterSelect: ({
    selectedCharacterId
  }: {
    selectedCharacterId: string | null
  }) => (
    <button type="button" data-testid="mock-character-select">
      Character slot {selectedCharacterId ?? "none"}
    </button>
  )
}))

const buildComposition = (): ConversationContextComposition => ({
  selection: {
    chatId: "chat-1",
    characterId: "42",
    worldBookIds: [3],
    dictionaryIds: [7]
  },
  inputText: "EV",
  transformedInputText: "Echo Vault",
  pieces: [
    {
      kind: "character",
      id: "42",
      name: "Archivist",
      source: "explicit_chat",
      status: "configured"
    },
    {
      kind: "worldbook",
      id: 3,
      source: "explicit_chat",
      status: "matched",
      diagnostics: [{ entry_id: 11, world_book_id: 3 }]
    },
    {
      kind: "worldbook",
      id: 4,
      source: "character_inherited",
      status: "configured"
    },
    {
      kind: "dictionary",
      id: 7,
      source: "explicit_chat",
      status: "active",
      diagnostics: {
        replacements: 1,
        entries_used: [7]
      }
    }
  ],
  previewSections: [
    {
      name: "Dictionaries",
      content: "Echo Vault",
      source: "explicit_chat"
    },
    {
      name: "Worldbooks",
      content: "Echo Vault lore.",
      source: "explicit_chat"
    }
  ],
  providerMessages: [
    {
      role: "system",
      content: "Worldbooks:\nEcho Vault lore."
    },
    {
      role: "user",
      content: "Echo Vault"
    }
  ],
  readiness: "ready",
  warnings: []
})

const buildBlankComposition = (): ConversationContextComposition => ({
  selection: {
    chatId: "chat-1",
    characterId: null,
    worldBookIds: [3],
    dictionaryIds: [7]
  },
  inputText: "EV",
  transformedInputText: "Echo Vault",
  pieces: [
    {
      kind: "worldbook",
      id: 3,
      source: "explicit_chat",
      status: "configured"
    },
    {
      kind: "dictionary",
      id: 7,
      source: "explicit_chat",
      status: "active",
      diagnostics: {
        replacements: 1,
        entries_used: [7]
      }
    }
  ],
  previewSections: [
    {
      name: "Dictionaries",
      content: "Echo Vault",
      source: "explicit_chat"
    }
  ],
  providerMessages: [
    {
      role: "user",
      content: "Echo Vault"
    }
  ],
  readiness: "ready",
  warnings: []
})

describe("ConversationContextPopover", () => {
  it("renders character selection as one slot inside conversation context", () => {
    render(
      <ConversationContextPopover
        chatId="chat-1"
        selectedCharacterId="42"
        setSelectedCharacterId={vi.fn()}
        composition={buildComposition()}
        compositionStatus="ready"
      />
    )

    expect(screen.getByTestId("conversation-context-trigger")).toBeInTheDocument()
    expect(screen.getByTestId("mock-character-select")).toHaveTextContent(
      "Character slot 42"
    )
  })

  it("shows worldbook and dictionary diagnostics from the client composition", () => {
    render(
      <ConversationContextPopover
        chatId="chat-1"
        selectedCharacterId="42"
        setSelectedCharacterId={vi.fn()}
        composition={buildComposition()}
        compositionStatus="ready"
      />
    )

    const content = screen.getByTestId("conversation-context-popover-content")
    expect(within(content).getAllByText("Worldbooks").length).toBeGreaterThan(0)
    expect(within(content).getByText("1 matched / 2 configured")).toBeInTheDocument()
    expect(within(content).getAllByText("Dictionaries").length).toBeGreaterThan(0)
    expect(within(content).getByText("1 active / 1 configured")).toBeInTheDocument()
    expect(within(content).getByText("Echo Vault lore.")).toBeInTheDocument()
  })

  it("does not block blank chat when no context assets are selected", () => {
    render(
      <ConversationContextPopover
        chatId={null}
        selectedCharacterId={null}
        setSelectedCharacterId={vi.fn()}
        composition={null}
        compositionStatus="idle"
      />
    )

    const trigger = screen.getByTestId("conversation-context-trigger")
    expect(trigger).toHaveAttribute("aria-disabled", "false")
    expect(screen.getByText("No optional context")).toBeInTheDocument()
  })

  it("keeps the trigger size stable between loading and ready states", () => {
    const { rerender } = render(
      <ConversationContextPopover
        chatId="chat-1"
        selectedCharacterId={null}
        setSelectedCharacterId={vi.fn()}
        composition={null}
        compositionStatus="loading"
      />
    )

    const loadingClassName =
      screen.getByTestId("conversation-context-trigger").className

    rerender(
      <ConversationContextPopover
        chatId="chat-1"
        selectedCharacterId="42"
        setSelectedCharacterId={vi.fn()}
        composition={buildComposition()}
        compositionStatus="ready"
      />
    )

    const readyClassName =
      screen.getByTestId("conversation-context-trigger").className
    expect(loadingClassName).toContain("min-w-[44px]")
    expect(readyClassName).toContain("min-w-[44px]")
    expect(loadingClassName).toContain("sm:min-w-[104px]")
    expect(readyClassName).toContain("sm:min-w-[104px]")
  })

  it("persists selected worldbooks through conversation-scoped settings", () => {
    const saveSelection = vi.fn(async () => undefined)
    render(
      <ConversationContextPopover
        chatId="chat-1"
        selectedCharacterId={null}
        setSelectedCharacterId={vi.fn()}
        composition={buildBlankComposition()}
        compositionStatus="ready"
        saveSelection={saveSelection}
        worldBookOptions={[
          { id: 3, name: "Echo Vault" },
          { id: 5, name: "Lore Atlas" }
        ]}
        dictionaryOptions={[{ id: 7, name: "Glossary" }]}
      />
    )

    fireEvent.click(screen.getByLabelText("Lore Atlas"))

    expect(saveSelection).toHaveBeenCalledWith({
      worldBookIds: [3, 5],
      dictionaryIds: [7]
    })
  })

  it("persists selected dictionaries through nested and compatibility settings", () => {
    const saveSelection = vi.fn(async () => undefined)
    render(
      <ConversationContextPopover
        chatId="chat-1"
        selectedCharacterId={null}
        setSelectedCharacterId={vi.fn()}
        composition={buildBlankComposition()}
        compositionStatus="ready"
        saveSelection={saveSelection}
        worldBookOptions={[{ id: 3, name: "Echo Vault" }]}
        dictionaryOptions={[
          { id: 7, name: "Glossary" },
          { id: 8, name: "Aliases" }
        ]}
      />
    )

    fireEvent.click(screen.getByLabelText("Aliases"))

    expect(saveSelection).toHaveBeenCalledWith({
      worldBookIds: [3],
      dictionaryIds: [7, 8]
    })
  })

  it("does not offer destructive asset edits before a chat is persisted", () => {
    const saveSelection = vi.fn(async () => undefined)
    render(
      <ConversationContextPopover
        chatId={null}
        selectedCharacterId={null}
        setSelectedCharacterId={vi.fn()}
        composition={buildComposition()}
        compositionStatus="ready"
        saveSelection={saveSelection}
        worldBookOptions={[{ id: 5, name: "Lore Atlas" }]}
        dictionaryOptions={[{ id: 8, name: "Aliases" }]}
      />
    )

    expect(screen.getByLabelText("Lore Atlas")).toBeDisabled()
    expect(screen.getByLabelText("Aliases")).toBeDisabled()
    fireEvent.click(screen.getByLabelText("Lore Atlas"))
    expect(saveSelection).not.toHaveBeenCalled()
  })
})
