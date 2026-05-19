// @vitest-environment jsdom

import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter } from "react-router-dom"
import { DemoModeProvider } from "@/context/demo-mode"

vi.mock("../CompanionHomePage", () => ({
  CompanionHomePage: ({ surface }: { surface: "options" | "sidepanel" }) => (
    <div data-testid="companion-home-page">{surface}</div>
  )
}))

const mockGetURL = vi.fn((path: string) => `chrome-extension://test${path}`)
const mockTabsCreate = vi.fn().mockResolvedValue({})

vi.mock("wxt/browser", () => ({
  browser: {
    runtime: {
      getURL: (...args: unknown[]) => mockGetURL(...(args as [string]))
    },
    tabs: {
      create: (...args: unknown[]) => mockTabsCreate(...(args as [object]))
    }
  }
}))

import { CompanionHomeShell } from "../CompanionHomeShell"

function Wrapper({ children }: { children: React.ReactNode }) {
  const client = new QueryClient({
    defaultOptions: { queries: { retry: false } }
  })
  return (
    <QueryClientProvider client={client}>
      <MemoryRouter>
        <DemoModeProvider>{children}</DemoModeProvider>
      </MemoryRouter>
    </QueryClientProvider>
  )
}

describe("CompanionHomeShell", () => {
  it("renders functional quick actions for the options surface", () => {
    render(<CompanionHomeShell surface="options" />, { wrapper: Wrapper })

    expect(screen.getByTestId("companion-home-shell")).toBeInTheDocument()
    expect(screen.getByTestId("companion-home-page")).toHaveTextContent("options")
    expect(screen.getByRole("link", { name: /Open Chat/i })).toHaveAttribute(
      "href",
      "/chat"
    )
    expect(screen.getByRole("link", { name: /Open Knowledge/i })).toHaveAttribute(
      "href",
      "/knowledge"
    )
  })

  it("offers a forced-chat escape hatch on the sidepanel surface", () => {
    render(<CompanionHomeShell surface="sidepanel" />, { wrapper: Wrapper })

    expect(screen.getByTestId("companion-home-page")).toHaveTextContent("sidepanel")
    expect(screen.getByRole("link", { name: /Open Chat/i })).toHaveAttribute(
      "href",
      "/?view=chat"
    )
    expect(screen.getByRole("link", { name: /Open Settings/i })).toHaveAttribute(
      "href",
      "/settings"
    )
  })

  it("renders Media Library and Quizzes deep-link buttons on sidepanel", () => {
    render(<CompanionHomeShell surface="sidepanel" />, { wrapper: Wrapper })

    const mediaButton = screen.getByRole("button", { name: /Open Media Library/i })
    const quizButton = screen.getByRole("button", { name: /Open Quizzes/i })

    expect(mediaButton).toBeInTheDocument()
    expect(quizButton).toBeInTheDocument()

    // Buttons should have correct test IDs
    expect(screen.getByTestId("companion-home-action-media")).toBeInTheDocument()
    expect(screen.getByTestId("companion-home-action-quiz")).toBeInTheDocument()
  })

  it("opens options page route when Media Library button is clicked", () => {
    mockGetURL.mockClear()
    mockTabsCreate.mockClear()

    render(<CompanionHomeShell surface="sidepanel" />, { wrapper: Wrapper })

    fireEvent.click(screen.getByRole("button", { name: /Open Media Library/i }))

    expect(mockGetURL).toHaveBeenCalledWith("/options.html#/media")
    expect(mockTabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://test/options.html#/media"
    })
  })

  it("opens options page route when Quizzes button is clicked", () => {
    mockGetURL.mockClear()
    mockTabsCreate.mockClear()

    render(<CompanionHomeShell surface="sidepanel" />, { wrapper: Wrapper })

    fireEvent.click(screen.getByRole("button", { name: /Open Quizzes/i }))

    expect(mockGetURL).toHaveBeenCalledWith("/options.html#/quiz")
    expect(mockTabsCreate).toHaveBeenCalledWith({
      url: "chrome-extension://test/options.html#/quiz"
    })
  })

  it("does not render deep-link buttons on the options surface", () => {
    render(<CompanionHomeShell surface="options" />, { wrapper: Wrapper })

    expect(screen.queryByRole("button", { name: /Open Media Library/i })).not.toBeInTheDocument()
    expect(screen.queryByRole("button", { name: /Open Quizzes/i })).not.toBeInTheDocument()
  })
})
