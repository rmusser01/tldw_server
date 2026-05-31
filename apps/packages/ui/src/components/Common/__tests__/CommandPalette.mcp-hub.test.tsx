import React from "react"
import { describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter, useLocation } from "react-router-dom"

import { CommandPalette } from "../CommandPalette"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, defaultValue?: string) => defaultValue ?? key,
  }),
}))

describe("CommandPalette MCP Hub discoverability", () => {
  const RouteProbe = () => {
    const location = useLocation()
    return <div data-testid="route-probe">{location.pathname}</div>
  }

  it("shows Go to MCP Hub when the palette opens with an empty query", async () => {
    render(
      <MemoryRouter>
        <CommandPalette />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    expect(
      screen.getByRole("option", { name: /Go to MCP Hub/i })
    ).toBeInTheDocument()
  })

  it("shows only one MCP Hub route result when searching for mcp", async () => {
    render(
      <MemoryRouter>
        <CommandPalette />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()

    const searchInput = screen.getByPlaceholderText(/Type a command or search/i)
    fireEvent.change(searchInput, { target: { value: "mcp" } })

    expect(screen.getAllByRole("option", { name: /MCP Hub/i })).toHaveLength(1)
  })

  it("opens MCP Hub on its canonical top-level route", async () => {
    render(
      <MemoryRouter initialEntries={["/settings"]}>
        <CommandPalette />
        <RouteProbe />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()
    const goToMcpHub = screen.getByRole("option", { name: /Go to MCP Hub/i })

    expect(goToMcpHub).toHaveAttribute("data-target-path", "/mcp-hub")
    fireEvent.click(goToMcpHub)

    expect(screen.getByTestId("route-probe")).toHaveTextContent("/mcp-hub")
  })

  it("keeps the specific Theme setting when searching a shared settings route", async () => {
    render(
      <MemoryRouter>
        <CommandPalette />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()

    const searchInput = screen.getByPlaceholderText(/Type a command or search/i)
    fireEvent.change(searchInput, { target: { value: "theme" } })

    expect(screen.getByRole("option", { name: /Theme/i })).toBeInTheDocument()
    expect(
      screen.queryByRole("option", { name: /Go to Settings/i })
    ).not.toBeInTheDocument()
  })

  it("keeps multiple settings that intentionally share the same route", async () => {
    render(
      <MemoryRouter>
        <CommandPalette />
      </MemoryRouter>
    )

    window.dispatchEvent(new CustomEvent("tldw:open-command-palette"))

    expect(await screen.findByRole("dialog")).toBeInTheDocument()

    const searchInput = screen.getByPlaceholderText(/Type a command or search/i)
    fireEvent.change(searchInput, { target: { value: "language" } })

    expect(screen.getAllByRole("option")).toHaveLength(3)
    expect(screen.getByText("Language")).toBeInTheDocument()
    expect(screen.getByText("OCR language")).toBeInTheDocument()
    expect(screen.getByText("Speech recognition language")).toBeInTheDocument()
  })
})
