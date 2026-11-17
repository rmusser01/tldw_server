// @vitest-environment jsdom
import React from "react"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { SlashCommandMenu } from "../SlashCommandMenu"
import { MentionsMenu } from "../MentionsMenu"
import { MentionsDropdown } from "@/components/Option/Playground/MentionsDropdown"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

// Native geometry is checked separately. This double checks active-row scrolling.
const originalScroll = Object.getOwnPropertyDescriptor(
  HTMLElement.prototype,
  "scrollIntoView"
)
const visible = vi.fn(function (this: HTMLElement) {
  return this.textContent
})
const items = Array.from({ length: 12 }, (_, i) => ({
  id: String(i),
  command: `command-${i}`,
  label: `Item ${i}`
}))
const actions = { onActiveIndexChange: vi.fn(), onSelect: vi.fn() }

describe("composer menu active-option visibility", () => {
  beforeEach(() => {
    visible.mockClear()
    Object.defineProperty(HTMLElement.prototype, "scrollIntoView", {
      configurable: true,
      value: visible
    })
  })
  afterEach(() => {
    cleanup()
    if (originalScroll)
      Object.defineProperty(
        HTMLElement.prototype,
        "scrollIntoView",
        originalScroll
      )
    else Reflect.deleteProperty(HTMLElement.prototype, "scrollIntoView")
  })

  it.each(["slash", "mentions"])(
    "reveals a lower active option in the %s menu",
    (kind) => {
      const menu = (index: number) =>
        kind === "slash" ? (
          <SlashCommandMenu
            open
            commands={items}
            activeIndex={index}
            {...actions}
          />
        ) : (
          <MentionsMenu open items={items} activeIndex={index} {...actions} />
        )
      const view = render(menu(0))
      visible.mockClear()
      view.rerender(menu(11))
      expect(visible.mock.contexts).toContain(
        screen.getByRole("option", { name: /Item 11/ })
      )
      expect(visible).toHaveBeenCalledWith({
        block: "nearest",
        inline: "nearest"
      })
    }
  )

  it("reveals the selected tab after keyboard navigation wraps to the last tab", () => {
    render(
      <MentionsDropdown
        show
        tabs={items.map((item, index) => ({
          id: index,
          title: item.label,
          url: "https://example.com/" + index
        }))}
        mentionPosition={{ start: 0, end: 1, query: "" }}
        textareaRef={{ current: null }}
        onSelectTab={vi.fn()}
        onClose={vi.fn()}
        refetchTabs={async () => {}}
        onMentionsOpen={async () => {}}
      />
    )
    visible.mockClear()
    fireEvent.keyDown(document, { key: "ArrowUp" })
    expect(visible.mock.contexts).toContain(
      screen.getByText("Item 11").closest("button")
    )
  })
})
