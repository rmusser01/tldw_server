import { readFileSync } from "node:fs"
import { resolve } from "node:path"

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

const originalBrowser = (globalThis as typeof globalThis & { browser?: unknown })
  .browser

describe("SettingsBody", () => {
  afterEach(() => {
    ;(globalThis as typeof globalThis & { browser?: unknown }).browser =
      originalBrowser
  })

  it("renders compact operational shortcuts without embedding provider probing", () => {
    ;(globalThis as typeof globalThis & { browser?: unknown }).browser =
      undefined

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
    expect(screen.getByText("Theme picker")).toBeInTheDocument()

    const source = readFileSync(
      resolve(process.cwd(), "../packages/ui/src/components/Sidepanel/Settings/body.tsx"),
      "utf8"
    )
    expect(source).not.toContain("defaultEmbeddingModelForRag")
  })

  it("opens full options routes when rendered in an extension runtime", () => {
    ;(globalThis as typeof globalThis & { browser?: unknown }).browser = {
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
})
