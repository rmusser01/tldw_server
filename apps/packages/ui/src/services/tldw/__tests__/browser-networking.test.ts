import { describe, expect, it } from "vitest"

import {
  buildBrowserHttpBase,
  buildBrowserWebSocketBase,
  detectBrowserNetworkingIssue,
  isCookieSessionBrowserTransport,
  isLoopbackHost,
  resolveBrowserTransport,
  resolveWebUiQuickstartServerUrl
} from "@/services/tldw/browser-networking"

describe("browser-networking", () => {
  it("resolves same-origin http and ws bases for quickstart webui pages", () => {
    const resolved = resolveBrowserTransport({
      surface: "webui-page",
      deploymentMode: "quickstart",
      pageOrigin: "https://webui.example.test",
      apiOrigin: ""
    })

    expect(buildBrowserHttpBase(resolved)).toBe("")
    expect(buildBrowserWebSocketBase(resolved)).toBe(
      "wss://webui.example.test"
    )
  })

  it("requires an explicit absolute api origin in advanced mode", () => {
    expect(() =>
      resolveBrowserTransport({
        surface: "webui-page",
        deploymentMode: "advanced",
        pageOrigin: "http://192.168.5.184:8080",
        apiOrigin: ""
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)

    expect(() =>
      resolveBrowserTransport({
        surface: "webui-page",
        deploymentMode: "advanced",
        pageOrigin: "http://192.168.5.184:8080",
        apiOrigin: "/api"
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)
  })

  it("treats bracketed ipv6 loopback as loopback", () => {
    expect(isLoopbackHost("[::1]")).toBe(true)
  })

  it("flags advanced loopback mismatch for webui pages", () => {
    expect(
      detectBrowserNetworkingIssue({
        surface: "webui-page",
        deploymentMode: "advanced",
        pageOrigin: "http://192.168.5.184:8080",
        apiOrigin: "http://127.0.0.1:8000"
      })
    ).toEqual({
      kind: "loopback_api_not_browser_reachable",
      apiOrigin: "http://127.0.0.1:8000",
      pageOrigin: "http://192.168.5.184:8080"
    })
  })

  it("canonicalizes loopback quickstart hosts back to the current webui origin", () => {
    expect(
      resolveWebUiQuickstartServerUrl({
        surface: "webui-page",
        deploymentMode: "quickstart",
        pageOrigin: "http://192.168.5.184:3000",
        apiOrigin: ""
      })
    ).toBe("http://192.168.5.184:3000")
  })

  it("does not block loopback mismatch for extension or browser-app surfaces", () => {
    expect(
      detectBrowserNetworkingIssue({
        surface: "extension",
        deploymentMode: "advanced",
        pageOrigin: "chrome-extension://abcd",
        apiOrigin: "http://127.0.0.1:8000"
      })
    ).toBeUndefined()

    expect(
      detectBrowserNetworkingIssue({
        surface: "browser-app",
        deploymentMode: "advanced",
        pageOrigin: "app://tldw",
        apiOrigin: "http://127.0.0.1:8000"
      })
    ).toBeUndefined()
  })

  it("activates cookie sessions only for quickstart same-origin http pages", () => {
    const base = {
      authMode: "single-user",
      authSource: "cookie-session",
      transportKind: "same-origin",
      transportMode: "quickstart",
      pageOrigin: "https://webui.example.test"
    } as const

    expect(isCookieSessionBrowserTransport(base)).toBe(true)
    expect(
      isCookieSessionBrowserTransport({ ...base, transportMode: "advanced" })
    ).toBe(false)
    expect(
      isCookieSessionBrowserTransport({ ...base, transportMode: "hosted" })
    ).toBe(false)
    expect(
      isCookieSessionBrowserTransport({
        ...base,
        pageOrigin: "chrome-extension://extension-id"
      })
    ).toBe(false)
  })
})
