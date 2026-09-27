import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { runInNewContext } from "node:vm"
import { seedManualUatBrowser } from "../browser-uat-seed.mjs"

const settings = {
  webUrl: "https://webui.example.test/app",
  serverUrl: "https://api.example.test",
  apiKey: "uat-regression-credential",
  legacyBootstrap: true
}

beforeEach(() => {
  // Track the descriptors so the document-memory facade is isolated per test.
  vi.stubGlobal("localStorage", localStorage)
  vi.stubGlobal("sessionStorage", sessionStorage)
})

afterEach(() => {
  localStorage.clear()
  vi.unstubAllGlobals()
})

describe("manual UAT browser seed", () => {
  it("keeps seed and subsequent application writes out of both native stores", () => {
    const nativeLocal = localStorage
    const nativeSession = sessionStorage
    vi.stubGlobal("location", { origin: "https://webui.example.test" })
    seedManualUatBrowser(settings)
    localStorage.setItem("appConfig", JSON.stringify({ apiKey: settings.apiKey }))
    sessionStorage.setItem("manual-session-key", settings.apiKey)
    expect(JSON.parse(localStorage.getItem("appConfig")!)).toEqual({ apiKey: settings.apiKey })
    expect(sessionStorage.getItem("manual-session-key")).toBe(settings.apiKey)
    expect(nativeLocal.length).toBe(0)
    expect(nativeSession.length).toBe(0)
  })

  it("does not seed or partially replace storage when a native descriptor is locked", () => {
    const nativeLocal = { setItem: vi.fn() }
    const nativeSession = { setItem: vi.fn() }
    const context = { location: { origin: "https://webui.example.test" }, URL }
    Object.defineProperties(context, {
      localStorage: { value: nativeLocal, configurable: true },
      sessionStorage: { value: nativeSession, configurable: false }
    })
    expect(() => runInNewContext(
      `(${seedManualUatBrowser.toString()})(${JSON.stringify(settings)})`, context
    )).toThrow("Cannot install document-memory UAT sessionStorage")
    expect(Reflect.get(context, "localStorage")).toBe(nativeLocal)
    expect(Reflect.get(context, "sessionStorage")).toBe(nativeSession)
    expect(nativeLocal.setItem).not.toHaveBeenCalled()
    expect(nativeSession.setItem).not.toHaveBeenCalled()
  })

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
    const nativeLocal = localStorage
    const nativeSession = sessionStorage
    vi.stubGlobal("location", { origin })
    seedManualUatBrowser(settings)
    expect(localStorage.length).toBe(0)
    expect(localStorage).toBe(nativeLocal)
    expect(sessionStorage).toBe(nativeSession)
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
