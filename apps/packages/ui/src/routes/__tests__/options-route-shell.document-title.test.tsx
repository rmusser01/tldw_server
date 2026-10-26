// @vitest-environment jsdom
/**
 * XP-19 (#3101, G14): the extension options page's tab title follows the
 * route ("Notes | tldw", "Chat | tldw", as on the WebUI) and a chat's title
 * does not stick after leaving the chat.
 */
import React from "react"
import { act, fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes, useNavigate } from "react-router-dom"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, options?: { defaultValue?: string }) => options?.defaultValue ?? key
  })
}))

// The route table itself is not under test: each route renders a stand-in
// that, like the real Chat page, may set its own title once it has loaded.
vi.mock("../app-route", () => ({
  RouteShell: () => {
    const navigate = useNavigate()
    return (
      <>
        <button type="button" onClick={() => navigate("/notes")}>Go to Notes</button>
        <button type="button" onClick={() => navigate("/prompts")}>Go to Prompts</button>
        <button type="button" onClick={() => navigate("/media/123/details")}>Go to an unknown page</button>
        <Routes>
          <Route path="/chat" element={<ChatStandIn />} />
          <Route path="*" element={null} />
        </Routes>
      </>
    )
  }
}))

const ChatStandIn = () => {
  React.useEffect(() => {
    document.title = "Git rebase gone wrong - recover commits"
  }, [])
  return <p>Chat</p>
}

import { OptionsRouteShell } from "../options-route-shell"

const renderShell = (path: string) =>
  render(
    <MemoryRouter initialEntries={[path]}>
      <OptionsRouteShell />
    </MemoryRouter>
  )

describe("options page titles (XP-19)", () => {
  beforeEach(() => {
    document.title = "tldw Assistant — Options"
  })

  it("names the page the route shows", () => {
    renderShell("/notes")

    expect(document.title).toBe("Notes | tldw")
  })

  it("does not keep a chat's title after leaving the chat", () => {
    renderShell("/chat")
    expect(document.title).toBe("Git rebase gone wrong - recover commits")

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Go to Prompts" }))
    })
    expect(document.title).toBe("Prompts | tldw")

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Go to Notes" }))
    })
    expect(document.title).toBe("Notes | tldw")
  })

  it("falls back to the options page name for pages without a title", () => {
    renderShell("/notes")

    act(() => {
      fireEvent.click(screen.getByRole("button", { name: "Go to an unknown page" }))
    })

    expect(document.title).toBe("tldw Assistant — Options")
  })
})
