import { render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { SettingsBody } from "../body"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (_key: string, fallback: unknown) => [fallback, vi.fn()]
}))

vi.mock("@/components/Common/Settings/ThemePicker", () => ({
  ThemePicker: () => <div>Theme picker</div>
}))

const runtimeGlobals = globalThis as typeof globalThis & {
  browser?: unknown
  chrome?: unknown
}
const originalBrowser = runtimeGlobals.browser
const originalChrome = runtimeGlobals.chrome

describe("SettingsBody", () => {
  afterEach(() => {
    runtimeGlobals.browser = originalBrowser
    runtimeGlobals.chrome = originalChrome
  })

  it("renders compact operational shortcuts", () => {
    runtimeGlobals.browser = undefined
    runtimeGlobals.chrome = undefined

    render(<SettingsBody />)

    expect(screen.getByRole("heading", { name: "Settings shortcuts" })).toBeInTheDocument()
    expect(screen.getByRole("link", { name: "Setup & Recovery" })).toHaveAttribute(
      "href",
      "/settings"
    )
    expect(screen.getByRole("link", { name: "Setup & Recovery" })).toHaveAttribute(
      "target",
      "_blank"
    )
    expect(screen.getByRole("link", { name: "Preferences" })).toHaveAttribute(
      "href",
      "/settings/preferences"
    )
    expect(screen.getByRole("link", { name: "UI customization" })).toHaveAttribute(
      "href",
      "/settings/ui"
    )
    expect(screen.getByRole("link", { name: "Data & Administration" })).toHaveAttribute(
      "href",
      "/settings/data"
    )
    expect(screen.getByText("Theme picker")).toBeInTheDocument()
    expect(screen.queryByText(/embedding defaults/i)).not.toBeInTheDocument()
  })

  it("opens full options routes when rendered in an extension runtime", () => {
    runtimeGlobals.browser = {
      runtime: {
        getURL: vi.fn((path: string) => `chrome-extension://test${path}`)
      }
    }

    render(<SettingsBody />)

    expect(screen.getByRole("link", { name: "Setup & Recovery" })).toHaveAttribute(
      "href",
      "chrome-extension://test/options.html#/settings"
    )
    expect(screen.getByRole("link", { name: "Preferences" })).toHaveAttribute(
      "href",
      "chrome-extension://test/options.html#/settings/preferences"
    )
  })

  it("opens full options routes when only chrome.runtime is available", () => {
    runtimeGlobals.browser = undefined
    runtimeGlobals.chrome = {
      runtime: {
        getURL: vi.fn((path: string) => `chrome-extension://chrome-only${path}`)
      }
    }

    render(<SettingsBody />)

    expect(screen.getByRole("link", { name: "Setup & Recovery" })).toHaveAttribute(
      "href",
      "chrome-extension://chrome-only/options.html#/settings"
    )
  })
})
