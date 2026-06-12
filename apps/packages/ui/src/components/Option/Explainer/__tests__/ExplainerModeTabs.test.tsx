import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { ExplainerModeTabs } from "../ExplainerModeTabs"

describe("ExplainerModeTabs", () => {
  it("uses roving tabindex so only the active tab is in the tab order", () => {
    render(<ExplainerModeTabs activeMode="goal" onModeChange={vi.fn()} />)

    expect(screen.getByRole("tab", { name: "Goal" })).toHaveAttribute("tabindex", "0")
    expect(screen.getByRole("tab", { name: "Sources" })).toHaveAttribute("tabindex", "-1")
  })

  it("moves selection with arrow keys", () => {
    const onModeChange = vi.fn()
    render(<ExplainerModeTabs activeMode="goal" onModeChange={onModeChange} />)

    fireEvent.keyDown(screen.getByRole("tab", { name: "Goal" }), { key: "ArrowRight" })
    expect(onModeChange).toHaveBeenCalledWith("sources")

    onModeChange.mockClear()
    fireEvent.keyDown(screen.getByRole("tab", { name: "Goal" }), { key: "ArrowLeft" })
    expect(onModeChange).toHaveBeenCalledWith("sources")
  })

  it("supports Home and End keys", () => {
    const onModeChange = vi.fn()
    render(<ExplainerModeTabs activeMode="sources" onModeChange={onModeChange} />)

    fireEvent.keyDown(screen.getByRole("tab", { name: "Sources" }), { key: "Home" })
    expect(onModeChange).toHaveBeenCalledWith("goal")

    onModeChange.mockClear()
    fireEvent.keyDown(screen.getByRole("tab", { name: "Goal" }), { key: "End" })
    expect(onModeChange).toHaveBeenCalledWith("sources")
  })
})
