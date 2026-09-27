import { fireEvent, render, screen } from "@testing-library/react"
import { createInstance } from "i18next"
import { I18nextProvider, initReactI18next } from "react-i18next"
import { afterEach, describe, expect, it, vi } from "vitest"
import { ServerOverviewHint } from "../ServerOverviewHint"

const serverGuide = "https://github.com/rmusser01/tldw_server/blob/main/Docs/Getting_Started/README.md"
const locales = import.meta.glob<{ default: Record<string, unknown> }>(
  "../../../assets/locale/*/settings.json", { eager: true }
)

afterEach(() => vi.restoreAllMocks())

async function openGuide(settings: Record<string, unknown>) {
  const i18n = createInstance()
  await i18n.use(initReactI18next).init({
    lng: "en", fallbackLng: false,
    resources: { en: { settings } },
    interpolation: { escapeValue: false }
  })
  const open = vi.spyOn(window, "open").mockReturnValue(null)
  render(<I18nextProvider i18n={i18n}><ServerOverviewHint /></I18nextProvider>)
  fireEvent.click(screen.getByRole("button"))
  return open
}

describe("server setup guide destination", () => {
  it.each(Object.entries(locales))("opens server documentation for %s", async (_path, locale) => {
    const open = await openGuide(locale.default)
    expect(open).toHaveBeenCalledWith(serverGuide, "_blank", "noopener,noreferrer")
  })

  it("uses the maintained server fallback when no translations are present", async () => {
    const open = await openGuide({})
    expect(open).toHaveBeenCalledWith(serverGuide, "_blank", "noopener,noreferrer")
  })

  it("retains a supplied server documentation override", async () => {
    const override = "https://docs.example.test/server-setup"
    const open = await openGuide({ serverOverview: { docsUrl: override } })
    expect(open).toHaveBeenCalledWith(override, "_blank", "noopener,noreferrer")
  })
})
