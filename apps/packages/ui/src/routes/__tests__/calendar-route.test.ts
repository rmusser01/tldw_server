import { readFileSync } from "node:fs"
import { dirname, resolve } from "node:path"
import { fileURLToPath } from "node:url"
import { createInstance } from "i18next"
import { describe, expect, it } from "vitest"
import optionEnglish from "../../assets/locale/en/option.json"

const testDirectory = dirname(fileURLToPath(import.meta.url))
const frontendRoot = resolve(testDirectory, "../../../../../tldw-frontend")
const routeSource = () =>
  readFileSync(resolve(testDirectory, "../option-calendar.tsx"), "utf8")

describe("calendar route wiring", () => {
  it("provides the shared route imported by the Next page", () => {
    const page = readFileSync(
      resolve(frontendRoot, "pages/calendar.tsx"),
      "utf8"
    )

    expect(page).toContain('import("@/routes/option-calendar")')
    expect(routeSource()).toContain("CalendarPage")
    expect(routeSource()).toContain("RouteErrorBoundary")
  })

  it("registers the extension route with the Calendar navigation label", () => {
    const registry = readFileSync(
      resolve(frontendRoot, "extension/routes/route-registry.tsx"),
      "utf8"
    )
    const extensionRoute = readFileSync(
      resolve(frontendRoot, "extension/routes/option-calendar.tsx"),
      "utf8"
    )

    expect(registry).toContain('path: "/calendar"')
    expect(registry).toContain('labelToken: "option:calendar.nav"')
    expect(extensionRoute).toContain("CalendarPage")
  })

  it("resolves the Calendar navigation label from the web locale", async () => {
    const i18n = createInstance()
    await i18n.init({
      lng: "en",
      resources: { en: { option: optionEnglish } }
    })

    expect(i18n.t("option:calendar.nav")).toBe("Calendar")
  })
})
