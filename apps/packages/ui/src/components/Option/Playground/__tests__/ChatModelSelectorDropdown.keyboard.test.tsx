// @vitest-environment jsdom
/**
 * AX-01 (#3122): the composer's model picker works from the keyboard.
 * ArrowDown moves from the search box, or from a closed Sort control, into
 * the model list; ArrowUp on the first model returns to the search; Enter
 * picks the focused model.
 */
import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { ConfigProvider } from "antd"

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

function Picker({ catalog, onPick }: { catalog: boolean; onPick: (key: string) => void }) {
  const [open, setOpen] = React.useState(false)
  const [search, setSearch] = React.useState("")
  const [sort, setSort] = React.useState<"favorites" | "az" | "provider" | "localFirst">("favorites")
  return (
    <ChatModelSelectorDropdown
      apiModelLabel="Llama.cpp / Gemma"
      connectionStatusLabel="Healthy"
      selectedModel="llamacpp:Gemma"
      modelDropdownOpen={open}
      modelDropdownMenuItems={[
        { key: "first", label: "First model", onClick: () => { onPick("first"); setOpen(false) } },
        { key: "second", label: "Second model", onClick: () => { onPick("second"); setOpen(false) } }
      ]}
      resolvedProviderKey="llamacpp"
      setModelDropdownOpen={setOpen}
      modelSearchQuery={search}
      setModelSearchQuery={setSearch}
      modelSortMode={sort}
      setModelSortMode={setSort}
      catalogControls={catalog ? (
        <PlaygroundModelCatalogControls
          t={(_key, fallback) => String(fallback)}
          modelListScope="configured"
          setModelListScope={() => undefined}
          modelSearchQuery={search}
          setModelSearchQuery={setSearch}
          modelSortMode={sort}
          setModelSortMode={setSort}
        />
      ) : undefined}
    />
  )
}

const openPicker = async (catalog: boolean, onPick = vi.fn()) => {
  render(
    <ConfigProvider theme={{ token: { motion: false } }}>
      <Picker catalog={catalog} onPick={onPick} />
    </ConfigProvider>
  )
  fireEvent.click(screen.getByRole("button", { name: "Llama.cpp / Gemma" }))
  const search = await screen.findByRole("textbox", { name: "Search models" })
  await waitFor(() => expect(search).toHaveFocus())
  const first = screen.getByRole("menuitem", { name: "First model" })
  return { search, first, onPick }
}

describe.each([false, true])("model picker keyboard (catalog controls=%s)", (catalog) => {
  it("moves from the search into the list with ArrowDown and back with ArrowUp", async () => {
    const { search, first } = await openPicker(catalog)

    fireEvent.keyDown(search, { key: "ArrowDown", keyCode: 40 })
    expect(first).toHaveFocus()
    fireEvent.keyDown(first, { key: "ArrowUp", keyCode: 38 })
    expect(search).toHaveFocus()
  })

  it("moves from a closed Sort control into the list with ArrowDown", async () => {
    await openPicker(catalog)
    const sort = screen.getByRole("combobox", { name: "Sort models" })
    sort.focus()

    fireEvent.keyDown(sort, { key: "ArrowDown", keyCode: 40 })
    expect(screen.getByRole("menuitem", { name: "First model" })).toHaveFocus()
    expect(sort).toHaveAttribute("aria-expanded", "false")
  })

  it("picks the focused model with Enter", async () => {
    const { search, first, onPick } = await openPicker(catalog)

    fireEvent.keyDown(search, { key: "ArrowDown", keyCode: 40 })
    expect(first).toHaveFocus()
    fireEvent.keyDown(first, { key: "Enter", keyCode: 13 })
    expect(onPick).toHaveBeenCalledWith("first")
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Llama.cpp / Gemma" })).toHaveAttribute("aria-expanded", "false")
    )
  })
})
