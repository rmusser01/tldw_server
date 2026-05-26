import React from "react"
import userEvent from "@testing-library/user-event"
import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import SidepanelFlashcards from "../sidepanel-flashcards"

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
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

describe("sidepanel flashcards route", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("shows explicit full-workspace and selected-text capture actions without auto-opening a tab", () => {
    render(<SidepanelFlashcards />)

    expect(
      screen.getByRole("heading", { name: "Flashcards" })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Study, manage, and create cards in the full Flashcards workspace."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Open full Flashcards" })
    ).toBeInTheDocument()
    expect(
      screen.getByText(
        "Turn the current page selection into editable flashcard drafts."
      )
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Generate from page selection" })
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

  it("captures the active page selection into the Flashcards generate workflow", async () => {
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate from page selection" })
    )

    expect(browserMocks.tabsQuery).toHaveBeenCalledWith({
      active: true,
      currentWindow: true
    })
    expect(browserMocks.executeScript).toHaveBeenCalledWith({
      target: { tabId: 42 },
      func: expect.any(Function)
    })
    const openedUrl = browserMocks.tabsCreate.mock.calls.at(-1)?.[0]?.url
    expect(openedUrl).toContain("/options.html#/flashcards?")
    const search = openedUrl?.split("?")[1] || ""
    const params = new URLSearchParams(search)
    expect(params.get("generate")).toBe("1")
    expect(params.get("generate_text")).toBe("Key concept from the active page")
    expect(params.get("generate_source_title")).toBe("Selection Source")
  })

  it("keeps the user in place when no page text is selected", async () => {
    browserMocks.executeScript.mockResolvedValueOnce([{ result: "" }])
    const user = userEvent.setup()
    render(<SidepanelFlashcards />)

    await user.click(
      screen.getByRole("button", { name: "Generate from page selection" })
    )

    expect(
      screen.getByText("Select text on the page first.")
    ).toBeInTheDocument()
    expect(browserMocks.tabsCreate).not.toHaveBeenCalled()
  })
})
