import React from "react"
import { act, render, screen } from "@testing-library/react"
import { useTranslation } from "react-i18next"
import { expect, it, vi } from "vitest"
import i18n, { ensureLanguageLoaded } from "../lib/i18n-web"
import settingsEs from "@tldw/ui/assets/locale/es/settings.json"
import optionEs from "@tldw/ui/assets/locale/es/option.json"

vi.mock("wxt/browser", () => ({
  browser: { storage: { local: { get: async () => ({}) } } }
}))

function PreferencesHeading() {
  const { t } = useTranslation("settings")
  return <h2>{t("preferencesSettings.title")}</h2>
}

it("updates mounted translations after the selected language loads without a reload", async () => {
  await i18n.changeLanguage("en")
  render(<PreferencesHeading />)
  expect(
    screen.getByRole("heading", { name: "General preferences" })
  ).toBeVisible()

  act(() => {
    void i18n.changeLanguage("es")
  })
  await act(async () => {
    await ensureLanguageLoaded("es")
  })

  expect(
    screen.getByRole("heading", {
      name: settingsEs.preferencesSettings.title
    })
  ).toBeVisible()
})

it("loads the existing Notes route translations for the selected language", async () => {
  await ensureLanguageLoaded("es")
  expect(i18n.hasResourceBundle("es", "option")).toBe(true)
  expect(i18n.getFixedT("es", "option")("notesSearch.newTooltip")).toBe(
    optionEs.notesSearch.newTooltip
  )
})

it("keeps the latest selected language when another language finishes loading", async () => {
  const pending = ensureLanguageLoaded("fr")
  await i18n.changeLanguage("en")
  await pending
  expect(i18n.language).toBe("en")
  expect(i18n.t("settings:preferencesSettings.title")).toBe(
    "General preferences"
  )
})
