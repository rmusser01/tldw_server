// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, afterEach, describe, expect, it, vi } from "vitest"

import { ThemeAdvancedEditor } from "../ThemeAdvancedEditor"
import { ThemeQuickEditor } from "../ThemeQuickEditor"
import { getBuiltinPresets } from "@/themes/presets"
import { generateThemeId } from "@/themes/validation"

vi.mock("@/themes/apply-theme", () => ({
  applyThemeTokens: vi.fn(),
  clearThemeTokens: vi.fn()
}))

const baseTheme = getBuiltinPresets()[0]

describe("Theme editor regressions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  afterEach(() => {
    vi.useRealTimers()
  })

  it("treats transient themes without ids as create mode in the advanced editor", () => {
    const onSave = vi.fn()
    const onClose = vi.fn()
    const draftTheme = {
      ...baseTheme,
      id: "",
      name: "Draft Quick Theme",
      builtin: false,
    }

    render(
      <ThemeAdvancedEditor
        open
        onClose={onClose}
        onSave={onSave}
        onDelete={vi.fn()}
        isDark={false}
        editingTheme={draftTheme}
        activeTheme={baseTheme}
      />
    )

    expect(screen.getByText("Create Theme (Advanced)")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Delete" })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Save" }))

    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        id: generateThemeId("Draft Quick Theme"),
        name: "Draft Quick Theme"
      })
    )
  })

  it("generates a unique default name and id for new quick themes", () => {
    vi.useFakeTimers()
    vi.setSystemTime(new Date("2026-04-18T22:58:00.123Z"))

    const onSave = vi.fn()
    const onClose = vi.fn()
    const expectedName = "Quick Theme 2026-04-18 22:58:00.123"

    render(
      <ThemeQuickEditor
        open
        onClose={onClose}
        onSave={onSave}
        isDark={false}
        activeTheme={baseTheme}
      />
    )

    fireEvent.click(screen.getByRole("button", { name: "Apply" }))

    expect(onSave).toHaveBeenCalledWith(
      expect.objectContaining({
        name: expectedName,
        id: generateThemeId(expectedName)
      })
    )
    expect(onClose).toHaveBeenCalled()
  })
})
