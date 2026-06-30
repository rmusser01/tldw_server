import React from "react"
import { render, screen } from "@testing-library/react"
import { describe, expect, it, vi } from "vitest"
import { MemoryRouter, Route, Routes } from "react-router-dom"

import enOption from "@/assets/locale/en/option.json"
import enSettings from "@/assets/locale/en/settings.json"
import { SettingsLayout } from "../SettingsOptionLayout"

const localeNamespaces: Record<string, unknown> = {
  common: { close: "Close" },
  option: enOption,
  settings: enSettings
}

const resolveLocaleToken = (token: string): string | undefined => {
  const [namespace, keyPath] = token.split(":")
  if (!namespace || !keyPath) return undefined

  const value = keyPath.split(".").reduce<unknown>((current, segment) => {
    if (!current || typeof current !== "object") return undefined
    return (current as Record<string, unknown>)[segment]
  }, localeNamespaces[namespace])

  return typeof value === "string" ? value : undefined
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (token: string, fallback?: string) =>
      resolveLocaleToken(token) ?? fallback ?? token
  })
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: null,
    loading: false
  })
}))

vi.mock("@/utils/sidepanel", () => ({
  isSidepanelSupported: () => false,
  openSidepanel: vi.fn()
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>()
  return {
    ...actual,
    setSetting: vi.fn()
  }
})

vi.mock("@/utils/settings-return", () => ({
  getSettingsReturnTo: () => null
}))

const renderSettingsLayout = () =>
  render(
    <MemoryRouter initialEntries={["/settings/provider-keys"]}>
      <Routes>
        <Route
          path="*"
          element={
            <SettingsLayout>
              <div>content</div>
            </SettingsLayout>
          }
        />
      </Routes>
    </MemoryRouter>
  )

describe("settings navigation labels", () => {
  it("renders provider keys as user-facing navigation text", () => {
    renderSettingsLayout()

    expect(screen.queryAllByText("settings:providerKeys.navTitle")).toHaveLength(0)
    expect(screen.getByRole("link", { name: /provider keys/i })).toBeVisible()
  })
})
