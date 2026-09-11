import { afterEach, describe, expect, it, vi } from "vitest"
import { seedManualUatBrowser } from "../browser-uat-seed.mjs"

const settings = {
  webUrl: "https://webui.example.test/app",
  serverUrl: "https://api.example.test",
  apiKey: "uat-regression-credential",
  legacyBootstrap: true
}

afterEach(() => {
  localStorage.clear()
  vi.unstubAllGlobals()
})

describe("manual UAT browser seed", () => {
  it.each(["file:///trusted/index.html", "data:text/html,example", "ftp://webui.example.test"])(
    "does not install credentials for an unsupported WebUI URL: %s", (webUrl) => {
      vi.stubGlobal("location", { origin: new URL(webUrl).origin })
      seedManualUatBrowser({ ...settings, webUrl })
      expect(localStorage.length).toBe(0)
    }
  )

  it.each([
    "https://attacker.example.test",
    "https://webui.example.test.attacker.test",
    "https://webui.example.test:8443",
    "http://webui.example.test",
    "null"
  ])("does not install credentials into %s", (origin) => {
    vi.stubGlobal("location", { origin })
    seedManualUatBrowser(settings)
    expect(localStorage.length).toBe(0)
  })

  it("keeps configured-server credentials available on the selected WebUI", () => {
    vi.stubGlobal("location", { origin: "https://webui.example.test" })
    seedManualUatBrowser(settings)
    expect(JSON.parse(localStorage.getItem("tldwConfig")!)).toEqual({
      serverUrl: settings.serverUrl,
      authMode: "single-user",
      apiKey: settings.apiKey
    })
    expect(localStorage.getItem("apiKey")).toBe(settings.apiKey)
    expect(localStorage.getItem("tldw-api-host")).toBe(settings.serverUrl)
  })

  it("preserves the minimal seed used by probes that exercise setup", () => {
    vi.stubGlobal("location", { origin: "https://webui.example.test" })
    seedManualUatBrowser({ ...settings, legacyBootstrap: false })
    expect(Array.from({ length: localStorage.length }, (_, index) => localStorage.key(index)).sort())
      .toEqual(["isMigrated", "tldwConfig"])
  })
})
