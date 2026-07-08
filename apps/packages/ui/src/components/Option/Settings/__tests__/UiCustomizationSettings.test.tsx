import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"

import { UiCustomizationSettings } from "../ui-customization"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@/hooks/useSetting", () => ({
  useSetting: () => [[], vi.fn()]
}))

vi.mock("@/components/Common/Settings/ThemePicker", () => ({
  ThemePicker: () => <div>Theme picker</div>
}))

vi.mock("../system-settings", () => ({
  SystemSettings: () => <div>System basics</div>
}))

describe("UiCustomizationSettings", () => {
  it("owns theme and system display basics", () => {
    render(<UiCustomizationSettings />)

    expect(screen.getByText("UI customization")).toBeInTheDocument()
    expect(screen.getByText("Theme picker")).toBeInTheDocument()
    expect(screen.getByText("System basics")).toBeInTheDocument()
  })

  it("keeps dense shortcut and display controls collapsed by default", () => {
    render(<UiCustomizationSettings />)

    expect(screen.getByTestId("sidebar-shortcuts-disclosure")).not.toHaveAttribute("open")
    expect(screen.getByTestId("playground-shortcuts-disclosure")).not.toHaveAttribute("open")
    expect(screen.getByTestId("theme-display-disclosure")).not.toHaveAttribute("open")
    expect(screen.getByTestId("system-display-disclosure")).not.toHaveAttribute("open")
  })
})
