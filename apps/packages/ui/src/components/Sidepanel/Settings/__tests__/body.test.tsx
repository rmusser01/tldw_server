import { fireEvent, render, screen } from "@testing-library/react"
import { afterEach, describe, expect, it, vi } from "vitest"

import { SettingsBody } from "../body"

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback?: string | { defaultValue?: string }) =>
      typeof fallback === "string" ? fallback : fallback?.defaultValue ?? _key
  })
}))

const storage = vi.hoisted(() => ({
  values: new Map<string, unknown>(),
  write: vi.fn()
}))

vi.mock("@plasmohq/storage/hook", () => ({
  useStorage: (key: string, fallback: unknown) => [
    storage.values.has(key) ? storage.values.get(key) : fallback,
    (next: unknown) => storage.write(key, next)
  ]
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
    storage.values.clear()
    storage.write.mockReset()
    vi.unstubAllGlobals()
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

  it("uses the actual WebUI route when the browser shim exposes getURL without an extension id", () => {
    runtimeGlobals.browser = { runtime: { id: undefined, getURL: (path: string) => `https://webui.test${path}` } }
    runtimeGlobals.chrome = undefined
    render(<SettingsBody />)
    expect(screen.getByRole("link", { name: "Server & auth" })).toHaveAttribute("href", "/settings/tldw")
  })

  it("opens full options routes when rendered in an extension runtime", () => {
    runtimeGlobals.browser = {
      runtime: {
        id: "test-extension",
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
        id: "chrome-extension",
        getURL: vi.fn((path: string) => `chrome-extension://chrome-only${path}`)
      }
    }

    render(<SettingsBody />)

    expect(screen.getByRole("link", { name: "Setup & Recovery" })).toHaveAttribute(
      "href",
      "chrome-extension://chrome-only/options.html#/settings"
    )
  })

  it("keeps the side panel chat input pinned by default and lets the user turn it off (XS-09)", () => {
    render(<SettingsBody />)
    const toggle = screen.getByRole("switch", {
      name: "Keep the side panel chat input visible while scrolling"
    })
    expect(toggle).toBeChecked()
    fireEvent.click(toggle)
    expect(storage.write).toHaveBeenCalledWith("sidepanelStickyChatInput", false)
  })

  it("ignores the full-page sticky input setting (XS-09)", () => {
    storage.values.set("stickyChatInput", false)
    render(<SettingsBody />)
    expect(
      screen.getByRole("switch", {
        name: "Keep the side panel chat input visible while scrolling"
      })
    ).toBeChecked()
  })
})
