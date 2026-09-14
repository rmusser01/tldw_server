import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { PersonaGardenTabs } from "../PersonaGardenTabs"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, options: any) => options?.defaultValue
  })
}))

it("moves section focus and selection with arrows, Home and End", () => {
  const Harness = () => {
    const [activeKey, onChange] = React.useState("live")
    return (
      <PersonaGardenTabs
        activeKey={activeKey}
        onChange={onChange}
        items={[
          { key: "live", label: "Live", content: "Conversation" },
          { key: "profile", label: "Profile", content: "Edit profile" },
          { key: "visuals", label: "Buddies", content: "Buddy gallery" }
        ]}
      />
    )
  }
  render(<Harness />)
  const tabs = screen.getAllByRole("tab")
  tabs[0].focus()
  fireEvent.keyDown(tabs[0], { key: "ArrowRight" })
  expect(tabs[1]).toHaveFocus()
  expect(tabs[1]).toHaveAttribute("aria-selected", "true")
  fireEvent.keyDown(tabs[1], { key: "End" })
  expect(tabs[2]).toHaveFocus()
  fireEvent.keyDown(tabs[2], { key: "ArrowRight" })
  expect(tabs[0]).toHaveFocus()
  fireEvent.keyDown(tabs[0], { key: "ArrowLeft" })
  expect(tabs[2]).toHaveFocus()
  fireEvent.keyDown(tabs[2], { key: "Home" })
  expect(tabs[0]).toHaveFocus()
  expect(tabs.filter((tab) => tab.tabIndex === 0)).toEqual([tabs[0]])
})
