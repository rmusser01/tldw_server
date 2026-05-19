// @vitest-environment jsdom
import React from "react"
import { fireEvent, render, screen } from "@testing-library/react"
import { Modal } from "antd"
import { beforeEach, describe, expect, it, vi } from "vitest"

import { ThemePicker } from "../ThemePicker"
import { getBuiltinPresets } from "@/themes/presets"
import { useTheme } from "@/hooks/useTheme"

vi.mock("@/hooks/useTheme", () => ({
  useTheme: vi.fn()
}))

vi.mock("../ThemeQuickEditor", () => ({
  ThemeQuickEditor: () => null
}))

vi.mock("../ThemeAdvancedEditor", () => ({
  ThemeAdvancedEditor: () => null
}))

vi.mock("@/themes/import-export", () => ({
  downloadThemeJson: vi.fn(),
  parseImportedTheme: vi.fn()
}))

vi.mock("@/themes/custom-themes", () => ({
  duplicateTheme: vi.fn((theme, name) => ({
    ...theme,
    id: `${theme.id}-copy`,
    name
  }))
}))

const baseTheme = getBuiltinPresets()[0]

describe("ThemePicker regressions", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    vi.mocked(useTheme).mockReturnValue({
      mode: "light",
      modePreference: "light",
      setModePreference: vi.fn(),
      themeId: "custom-1",
      setThemeId: vi.fn(),
      themeDefinition: baseTheme,
      presets: [
        baseTheme,
        {
          ...baseTheme,
          id: "custom-1",
          name: "Custom Theme",
          builtin: false
        }
      ],
      customThemes: [
        {
          ...baseTheme,
          id: "custom-1",
          name: "Custom Theme",
          builtin: false
        }
      ],
      saveCustomTheme: vi.fn(),
      deleteCustomTheme: vi.fn()
    } as any)
    vi.spyOn(Modal, "confirm").mockImplementation(() => ({
      destroy: vi.fn(),
      update: vi.fn()
    }) as any)
  })

  it("confirms before deleting a custom theme", async () => {
    render(<ThemePicker />)

    fireEvent.click(screen.getByTitle("Delete theme"))

    const themeApi = vi.mocked(useTheme).mock.results[0]?.value

    expect(themeApi?.deleteCustomTheme).not.toHaveBeenCalled()
    expect(Modal.confirm).toHaveBeenCalledTimes(1)

    const confirmConfig = vi.mocked(Modal.confirm).mock.calls[0][0]
    await confirmConfig.onOk?.()

    expect(themeApi?.deleteCustomTheme).toHaveBeenCalledWith("custom-1")
  })
})
