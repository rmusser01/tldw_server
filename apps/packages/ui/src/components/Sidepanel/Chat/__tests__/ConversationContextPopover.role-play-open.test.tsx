import React from "react"
import { render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ConversationContextPopover } from "../ConversationContextPopover"

type MockPopoverProps = {
  children?: React.ReactNode
  content?: React.ReactNode
  open?: boolean
}

vi.mock("antd", () => ({
  Popover: ({ children, content, open }: MockPopoverProps) => (
    <>
      {children}
      {open ? <div data-testid="mock-popover-content">{content}</div> : null}
    </>
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

vi.mock("../CharacterSelect", () => ({
  CharacterSelect: ({
    openRequest
  }: {
    openRequest?: { id: number; tab?: string }
  }) => (
    <div data-testid="character-select-open-request">
      {openRequest ? `${openRequest.id}:${openRequest.tab ?? ""}` : "none"}
    </div>
  )
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: vi.fn(),
    listWorldBooks: vi.fn(),
    listDictionaries: vi.fn()
  }
}))

const renderPopover = () =>
  render(
    <ConversationContextPopover
      chatId="chat-1"
      selectedCharacterId={null}
      setSelectedCharacterId={vi.fn()}
      composition={null}
      compositionStatus="idle"
      worldBookOptions={[]}
      dictionaryOptions={[]}
    />
  )

describe("ConversationContextPopover role-play picker behavior", () => {
  it("opens the mounted assistant picker when the sidepanel event is dispatched", async () => {
    renderPopover()

    window.dispatchEvent(
      new CustomEvent("tldw:open-sidepanel-assistant-select", {
        detail: { tab: "persona" }
      })
    )

    await waitFor(() => {
      expect(screen.getByTestId("character-select-open-request")).toHaveTextContent(
        "1:persona"
      )
    })

    expect(screen.getByRole("button", { name: "Conversation context" }))
      .toHaveAttribute("aria-expanded", "true")
  })
})
