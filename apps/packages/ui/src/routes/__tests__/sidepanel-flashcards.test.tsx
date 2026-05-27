import React from "react"
import userEvent from "@testing-library/user-event"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import SidepanelFlashcards, {
  readSelectedTextFromPage
} from "../sidepanel-flashcards"

const flashcardMocks = vi.hoisted(() => ({
  createFlashcardMutateAsync: vi.fn(),
  useFlashcardsEnabled: vi.fn(),
  useDecksQuery: vi.fn()
}))

const browserMocks = vi.hoisted(() => ({
  runtimeGetURL: vi.fn((path: string) => `chrome-extension://flashcards${path}`),
  tabsCreate: vi.fn(async () => undefined),
  tabsQuery: vi.fn(async () => [
    {
      id: 42,
      title: "Selection Source",
      url: "https://example.test/source"
    }
  ]),
  executeScript: vi.fn(async () => [
    {
      result: "Key concept from the active page"
    }
  ])
}))

vi.mock("@/components/Flashcards/hooks", () => ({
  useCreateFlashcardMutation: () => ({
    mutateAsync: flashcardMocks.createFlashcardMutateAsync,
    isPending: false
  }),
  useFlashcardsEnabled: () => flashcardMocks.useFlashcardsEnabled(),
  useDecksQuery: (...args: unknown[]) => flashcardMocks.useDecksQuery(...args)
}))

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: browserMocks.runtimeGetURL
    },
    tabs: {
      create: browserMocks.tabsCreate,
      query: browserMocks.tabsQuery
    },
    scripting: {
      executeScript: browserMocks.executeScript
    }
  }
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string; deckName?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return (fallbackOrOptions.defaultValue || key).replace(
          "{{deckName}}",
          fallbackOrOptions.deckName || ""
        )
      }
      return key
    }
  })
}))

describe("sidepanel flashcards route", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    flashcardMocks.createFlashcardMutateAsync.mockResolvedValue({
      uuid: "card-1",
      deck_id: 7
    })
    flashcardMocks.useFlashcardsEnabled.mockReturnValue({
      isOnline: true,
      capsLoading: false,
      flashcardsUnsupported: false,
      flashcardsEnabled: true
    })
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: [
        {
          id: 7,
          name: "Biology",
          description: null,
          deleted: false,
          client_id: "test",
          version: 1
        }
      ],
      isLoading: false,
      isError: false
    })
  })

  afterEach(() => {
    vi.restoreAllMocks()
  })

  it("shows explicit full-workspace and selected-text capture actions without auto-opening a tab", () => {
    render(<SidepanelFlashcards />)

    expect(
      screen.getByRole("heading", { name: "Flashcards" })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Capture selected page text into a deck, or open the full Flashcards workspace."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open full Flashcards" })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Create one editable card in the sidepanel. Use full Flashcards for generation, imports, and review."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Capture page selection" })
    ).toBeInTheDocument()
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
  })

  it("opens the full Flashcards workspace when the user chooses that action", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Open full Flashcards" })
    )

    expect(browserMocks.runtimeGetURL).toHaveBeenCalledWith(
      "/options.html#/flashcards"
    )
    expect(browserMocks.tabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://flashcards/options.html#/flashcards"
    })
  })

  it("shows an inline error when tab and fallback window opening fail", async () => {
    browserMocks.tabsCreate.mockRejectedValueOnce(new Error("tab open failed"))
    vi.spyOn(window, "open").mockReturnValue(null)
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Open full Flashcards" })
    )

    expect(window.open).toHaveBeenCalledWith(
      "chrome-extension://flashcards/options.html#/flashcards",
      "_blank"
    )
    expect(
      await screen.findByText(
        "Could not open Flashcards. Check popup permissions and try again."
      )
    ).toBeInTheDocument()
  })

  it("reads selected text from focused form controls in the injected page helper", () => {
    const input = document.createElement("input")
    input.value = "Before selected phrase after"
    document.body.append(input)

    try {
      input.focus()
      input.setSelectionRange(7, 22)

      expect(readSelectedTextFromPage()).toBe("selected phrase")
    } finally {
      input.remove()
    }
  })

  it("captures the active page selection into an editable sidepanel draft", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(browserMocks.tabsQuery).toHaveBeenCalledWith({
      active: true,
      currentWindow: true
    })
    expect(browserMocks.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      func: expect.any(Function)
    })
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()
    expect(screen.getByTestId("sidepanel-flashcards-deck-select")).toHaveTextContent(
      "Biology"
    )
    expect(screen.getByLabelText("Front")).toHaveValue("Selection Source")
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  it("saves the edited sidepanel draft with deck and page provenance", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    fireEvent.change(screen.getByLabelText("Front"), {
      target: { value: "Edited question" }
    })
    fireEvent.change(screen.getByLabelText("Back"), {
      target: { value: "Edited answer" }
    })
    await user.click(screen.getByRole("button", { name: "Save card" }))

    await waitFor(() => {
      expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledTimes(1)
    })
    expect(flashcardMocks.createFlashcardMutateAsync).toHaveBeenCalledWith({
      deck_id: 7,
      front: "Edited question",
      back: "Edited answer",
      model_type: "basic",
      is_cloze: false,
      reverse: false,
      source_ref_type: "manual",
      source_ref_id: "https://example.test/source"
    })
    expect(screen.getByText("Saved to Biology.")).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Draft flashcard" })
    ).not.toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open full Flashcards" })
    ).toBeInTheDocument()
  })

  it("keeps the draft in place when sidepanel save fails", async () => {
    flashcardMocks.createFlashcardMutateAsync.mockRejectedValueOnce(
      new Error("save failed")
    )
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    await user.click(screen.getByRole("button", { name: "Save card" }))

    expect(
      await screen.findByText("Could not save flashcard. Check your connection and try again.")
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Back")).toHaveValue(
      "Key concept from the active page"
    )
  })

  it("requires a deck before saving a sidepanel draft", async () => {
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: [],
      isLoading: false,
      isError: false
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Create a deck in full Flashcards before saving here.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("does not show true-empty deck guidance while decks are still loading", async () => {
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: undefined,
      isLoading: true,
      isError: false
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.queryByText("Create a deck in full Flashcards before saving here.")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("differentiates deck load errors from true empty deck state", async () => {
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: true
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Could not load decks. Check your connection and try again.")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Create a deck in full Flashcards before saving here.")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("explains when flashcards are unavailable instead of saying there are no decks", async () => {
    flashcardMocks.useFlashcardsEnabled.mockReturnValue({
      isOnline: false,
      capsLoading: false,
      flashcardsUnsupported: false,
      flashcardsEnabled: false
    })
    flashcardMocks.useDecksQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      isError: false
    })
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Flashcards are unavailable. Check the server connection and try again.")
    ).toBeInTheDocument()
    expect(
      screen.queryByText("Create a deck in full Flashcards before saving here.")
    ).not.toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Save card" })).toBeDisabled()
  })

  it("clears a previous draft when a later capture attempt fails", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )
    expect(
      screen.getByRole("heading", { name: "Draft flashcard" })
    ).toBeInTheDocument()

    browserMocks.executeScript.mockResolvedValueOnce([{ result: "" }])
    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Select text on the page first.")
    ).toBeInTheDocument()
    expect(
      screen.queryByRole("heading", { name: "Draft flashcard" })
    ).not.toBeInTheDocument()
  })

  it("uses capture wording when no active page tab is available", async () => {
    browserMocks.tabsQuery.mockResolvedValueOnce([])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Open a page tab before capturing page selection.")
    ).toBeInTheDocument()
  })

  it("keeps the user in place when no page text is selected", async () => {
    browserMocks.executeScript.mockResolvedValueOnce([{ result: "" }])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Capture page selection" })
    )

    expect(
      screen.getByText("Select text on the page first.")
    ).toBeInTheDocument()
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
  })
})
