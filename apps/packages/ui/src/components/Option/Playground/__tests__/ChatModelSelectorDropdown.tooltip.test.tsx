import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ConfigProvider } from "antd"

import { ChatModelSelectorDropdown } from "../ChatModelSelectorDropdown"

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
