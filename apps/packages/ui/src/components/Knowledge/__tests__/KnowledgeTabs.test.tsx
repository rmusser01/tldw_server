import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { KnowledgeTabs } from "../KnowledgeTabs"
import type { KnowledgeTab } from "../KnowledgePanel"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string) => fallback ?? _key
  })
}))

const TabsHarness: React.FC = () => {
  const [activeTab, setActiveTab] = React.useState<KnowledgeTab>("qa-search")
  return (
    <KnowledgeTabs
      activeTab={activeTab}
      onTabChange={setActiveTab}
      contextCount={3}
    />
  )
}

describe("KnowledgeTabs keyboard navigation", () => {
  it("switches tabs with numeric keys 1/2/3/4", () => {
    render(<TabsHarness />)

    const tablist = screen.getByRole("tablist", {
      name: "Search & Context sections"
    })

    fireEvent.keyDown(tablist, { key: "2" })
    expect(screen.getByRole("tab", { name: "File Search" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    fireEvent.keyDown(tablist, { key: "3" })
    expect(screen.getByRole("tab", { name: "Settings" })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    fireEvent.keyDown(tablist, { key: "4" })
    expect(screen.getByRole("tab", { name: /Context/ })).toHaveAttribute(
      "aria-selected",
      "true"
    )

    fireEvent.keyDown(tablist, { key: "1" })
    expect(screen.getByRole("tab", { name: "QA Search" })).toHaveAttribute(
      "aria-selected",
      "true"
    )
  })
})
