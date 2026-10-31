import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ConfigProvider } from "antd"
import userEvent from "@testing-library/user-event"

import { ChatModelSelectorDropdown } from "../ChatModelSelectorDropdown"
import { PlaygroundModelCatalogControls } from "../PlaygroundModelCatalogControls"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({ t: (key: string, fallback?: string) => fallback ?? key })
}))
vi.mock("react-router-dom", () => ({
  Link: ({ children, to, ...rest }: React.AnchorHTMLAttributes<HTMLAnchorElement> & { to: string }) => (
    <a href={to} {...rest}>{children}</a>
  )
}))
vi.mock("@/components/Common/ProviderIcon", () => ({
  ProviderIcons: () => null
}))

describe("Chat model tooltip and dropdown interaction", () => {
  it("hides hover help while selecting and restores it after closing", async () => {
    const select = vi.fn()
    function Selector() {
      const [open, setOpen] = React.useState(false)
      return <ChatModelSelectorDropdown
        apiModelLabel="Llama.cpp / Gemma"
        connectionStatusLabel="Healthy"
        modelDropdownOpen={open}
        modelDropdownMenuItems={[{ key: "next", label: "Next model", onClick: select }]}
        resolvedProviderKey="llamacpp"
        selectedModel="llamacpp:Gemma"
        setModelDropdownOpen={setOpen}
        setModelSearchQuery={() => undefined}
      />
    }
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector /></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    fireEvent.mouseEnter(trigger)
    await waitFor(() => expect(screen.getByRole("tooltip")).toBeVisible())
    fireEvent.click(trigger)
    const option = await screen.findByRole("menuitem", { name: "Next model" })
    expect(trigger).not.toHaveAttribute("title")
    await waitFor(() => expect(screen.queryByRole("tooltip")).not.toBeInTheDocument())
    fireEvent.click(option)
    expect(select).toHaveBeenCalledTimes(1)
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(trigger).toHaveAttribute("title", "Llama.cpp / Gemma")
    fireEvent.mouseLeave(trigger)
    fireEvent.mouseEnter(trigger)
    await waitFor(() => expect(screen.getByRole("tooltip")).toBeVisible())
  })
})


describe("Chat model picker accessibility", () => {
  function Selector({ catalog = false }: { catalog?: boolean }) {
    const [open, setOpen] = React.useState(false)
    const [search, setSearch] = React.useState("")
    const [sort, setSort] = React.useState<"favorites" | "az" | "provider" | "localFirst">("favorites")
    return <ChatModelSelectorDropdown
      apiModelLabel="Llama.cpp / Gemma"
      connectionStatusLabel="Healthy"
      selectedModel="llamacpp:Gemma"
      modelDropdownOpen={open}
      modelDropdownMenuItems={[{ key: "gemma", label: "A model with a very long full name" }]}
      resolvedProviderKey="llamacpp"
      setModelDropdownOpen={setOpen}
      modelSearchQuery={search}
      setModelSearchQuery={setSearch}
      modelSortMode={sort}
      setModelSortMode={setSort}
      catalogControls={catalog ? <PlaygroundModelCatalogControls
        t={(_key, fallback) => String(fallback)}
        modelListScope="configured"
        setModelListScope={() => undefined}
        modelSearchQuery={search}
        setModelSearchQuery={setSearch}
        modelSortMode={sort}
        setModelSortMode={setSort}
      /> : undefined}
    />
  }

  it("advertises the menu it opens, preserving full model names", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector /></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    expect(trigger).toHaveAttribute("aria-haspopup", "menu")
    fireEvent.click(trigger)
    const option = await screen.findByRole("menuitem", { name: "A model with a very long full name" })
    await waitFor(() => expect(option).toBeVisible())
  })

  it.each([false, true])("lets Escape dismiss from the search control in catalog=%s", async catalog => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector catalog={catalog} /></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    fireEvent.click(trigger)
    const search = await screen.findByPlaceholderText("Search models")
    search.focus()
    fireEvent.keyDown(search, { key: "Escape", keyCode: 27 })
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(trigger).toHaveFocus()
  })

  it("keeps catalog, search, sort, menu and help reachable with Tab", async () => {
    const user = userEvent.setup()
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector catalog /></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    fireEvent.click(trigger)
    const search = await screen.findByRole("textbox", { name: "Search models" })
    await waitFor(() => expect(search).toHaveFocus())
    await user.tab({ shift: true })
    expect(screen.getByRole("radio", { name: "Ready to use" })).toHaveFocus()
    await user.tab()
    expect(search).toHaveFocus()
    await user.tab()
    expect(screen.getByRole("combobox", { name: "Sort models" })).toHaveFocus()
    await user.tab()
    expect(screen.getByRole("menu")).toHaveFocus()
    await user.tab()
    const help = screen.getByRole("link", { name: "Help me choose a model" })
    expect(help).toHaveFocus()
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    fireEvent.keyDown(help, { key: "Escape", keyCode: 27 })
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(trigger).toHaveFocus()
  })

  it("continues Tab outside the picker after leaving Help", async () => {
    const user = userEvent.setup()
    render(<ConfigProvider theme={{ token: { motion: false } }}><button>Before picker</button><Selector /><button>After picker</button></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    fireEvent.click(trigger)
    const search = await screen.findByRole("textbox", { name: "Search models" })
    await waitFor(() => expect(search).toHaveFocus())
    await user.tab()
    await user.tab()
    await user.tab()
    expect(screen.getByRole("link", { name: "Help me choose a model" })).toHaveFocus()
    await user.tab()
    expect(document.body).toHaveFocus()
    await user.tab()
    expect(screen.getByRole("button", { name: "Before picker" })).toHaveFocus()
    expect(trigger).toHaveAttribute("aria-expanded", "false")
  })

  it("closes an open picker once when its trigger is clicked from Search", async () => {
    const user = userEvent.setup()
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector /></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    await user.click(trigger)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Search models" })).toHaveFocus())
    await user.click(trigger)
    expect(trigger).toHaveAttribute("aria-expanded", "false")
    expect(trigger).toHaveFocus()
  })

  it("closes nested Sort before its parent picker with Escape", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector /></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    fireEvent.click(trigger)
    const sort = await screen.findByRole("combobox", { name: "Sort models" })
    await waitFor(() => expect(sort).toBeVisible())
    sort.focus()
    fireEvent.mouseDown(sort)
    await waitFor(() => expect(sort).toHaveAttribute("aria-expanded", "true"))
    fireEvent.keyDown(sort, { key: "Escape", keyCode: 27 })
    await waitFor(() => expect(sort).toHaveAttribute("aria-expanded", "false"))
    expect(trigger).toHaveAttribute("aria-expanded", "true")
    expect(sort).toHaveFocus()
    fireEvent.keyDown(sort, { key: "Escape", keyCode: 27 })
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(trigger).toHaveFocus()
  })

  it.each(["close", "unmount"])("retires opening focus on %s before the delayed frames finish", retirement => {
    const frames = new Map<number, FrameRequestCallback>()
    let nextFrame = 0
    const request = vi.spyOn(window, "requestAnimationFrame").mockImplementation(callback => {
      frames.set(++nextFrame, callback)
      return nextFrame
    })
    const cancel = vi.spyOn(window, "cancelAnimationFrame").mockImplementation(id => { frames.delete(id) })
    const flushFrame = () => {
      const pending = [...frames.values()]
      frames.clear()
      act(() => pending.forEach(callback => callback(0)))
    }
    const view = (show = true) => <ConfigProvider theme={{ token: { motion: true } }}>{show && <Selector />}<button>Other action</button></ConfigProvider>
    const { unmount, rerender } = render(view())
    try {
      const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
      trigger.focus()
      fireEvent.click(trigger)
      const search = screen.getByRole("textbox", { name: "Search models" })
      const outside = screen.getByRole("button", { name: "Other action" })
      outside.focus()
      if (retirement === "close") {
        fireEvent.mouseDown(outside)
        fireEvent.click(outside)
        expect(trigger).toHaveAttribute("aria-expanded", "false")
        // Default leave motion retains the popup while its earlier focus frame is held.
        expect(search.isConnected).toBe(true)
      } else {
        rerender(view(false))
        expect(search.isConnected).toBe(false)
      }
      for (let frame = 0; frame < 4; frame++) flushFrame()
      expect(outside).toHaveFocus()
    } finally {
      unmount()
      request.mockRestore()
      cancel.mockRestore()
    }
  })

  it("preserves focus on an outside action when dismissing the picker", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector /><button>Other action</button></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    fireEvent.click(trigger)
    await screen.findByRole("textbox", { name: "Search models" })
    const outside = screen.getByRole("button", { name: "Other action" })
    outside.focus()
    fireEvent.mouseDown(outside)
    fireEvent.click(outside)
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(outside).toHaveFocus()
  })

  it.each([false, true])("names search and sort in catalog=%s, retains filtering and closes with Escape", async catalog => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><Selector catalog={catalog} /></ConfigProvider>)
    const trigger = screen.getByRole("button", { name: "Llama.cpp / Gemma" })
    fireEvent.click(trigger)
    const search = await screen.findByRole("textbox", { name: "Search models" })
    await waitFor(() => expect(screen.getByRole("combobox", { name: "Sort models" })).toBeVisible())
    await waitFor(() => expect(search).toHaveFocus())
    fireEvent.change(search, { target: { value: "Gemma" } })
    expect(search).toHaveValue("Gemma")
    fireEvent.keyDown(search, { key: "Escape", keyCode: 27 })
    await waitFor(() => expect(trigger).toHaveAttribute("aria-expanded", "false"))
    expect(trigger).toHaveFocus()
    fireEvent.click(trigger)
    await waitFor(() => expect(screen.getByRole("textbox", { name: "Search models" })).toHaveValue(""))
  })
})
