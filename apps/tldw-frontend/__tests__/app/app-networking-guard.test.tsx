import path from "node:path"
import { pathToFileURL } from "node:url"
import React from "react"
import { render, screen } from "@testing-library/react"
import { renderToString } from "react-dom/server"
import { afterAll, beforeEach, describe, expect, it, vi } from "vitest"

import { ConfigurationGuard } from "@web/components/networking/ConfigurationGuard"
import App from "@web/pages/_app"

const mockRouter = {
  pathname: "/media",
  asPath: "/media",
  push: vi.fn(),
  replace: vi.fn(),
  prefetch: vi.fn(() => Promise.resolve(true))
}

const DummyPage = () => <div data-testid="page-content">Page</div>

vi.mock("next/router", () => ({
  useRouter: () => mockRouter
}))

vi.mock("next/dynamic", () => ({
  default: () =>
    ({
      children
    }: {
      children: React.ReactNode
    }) => <div data-testid="option-layout">{children}</div>
}))

vi.mock("@web/components/AppProviders", () => ({
  AppProviders: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@web/lib/configured-auth-state", () => ({
  loadConfiguredAuthConfig: async () => null,
  loadTldwAuth: async () => ({
    getCurrentUser: async () => ({ username: "test-user" })
  })
}))

const originalApiUrl = process.env.NEXT_PUBLIC_API_URL
const originalDeploymentMode = process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
const originalWindowLocation = window.location

const setWindowLocation = (href: string) => {
  Object.defineProperty(window, "location", {
    configurable: true,
    value: new URL(href)
  })
}

const renderApp = () => render(<App Component={DummyPage} pageProps={{}} />)

describe("app networking guard", () => {
  beforeEach(() => {
    mockRouter.pathname = "/media"
    mockRouter.asPath = "/media"
    mockRouter.push.mockClear()
    mockRouter.replace.mockClear()
    mockRouter.prefetch.mockClear()
    delete process.env.NEXT_PUBLIC_API_URL
    delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    setWindowLocation("http://localhost:3000/media")
  })

  afterAll(() => {
    if (originalApiUrl === undefined) {
      delete process.env.NEXT_PUBLIC_API_URL
    } else {
      process.env.NEXT_PUBLIC_API_URL = originalApiUrl
    }

    if (originalDeploymentMode === undefined) {
      delete process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE
    } else {
      process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = originalDeploymentMode
    }

    Object.defineProperty(window, "location", {
      configurable: true,
      value: originalWindowLocation
    })
  })

  it("blocks app rendering when advanced mode targets a loopback api from a lan page origin", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
    setWindowLocation("http://192.168.1.50:3000/media")

    renderApp()

    expect(
      await screen.findByTestId("networking-config-error")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("page-content")).toBeNull()
  })

  it("shows remediation text explaining the api is only reachable from the host machine", async () => {
    process.env.NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE = "advanced"
    process.env.NEXT_PUBLIC_API_URL = "http://127.0.0.1:8000"
    setWindowLocation("http://192.168.1.50:3000/media")

    renderApp()

    expect(
      await screen.findByText(/only reachable from the host machine/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/set the webui api url to a lan-reachable address/i)
    ).toBeInTheDocument()
  })

  it("rejects quickstart mode when TLDW_INTERNAL_API_ORIGIN is missing", async () => {
    const validatorPath = path.resolve(
      __dirname,
      "..",
      "..",
      "scripts",
      "validate-networking-config.mjs"
    )
    const moduleUrl = pathToFileURL(validatorPath)
    moduleUrl.searchParams.set("t", `${Date.now()}-${Math.random()}`)
    const { validateNetworkingConfig } = await import(moduleUrl.href)

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart"
      })
    ).toThrow(/TLDW_INTERNAL_API_ORIGIN/i)
  })

  it("rejects quickstart mode when NEXT_PUBLIC_API_URL is set", async () => {
    const validatorPath = path.resolve(
      __dirname,
      "..",
      "..",
      "scripts",
      "validate-networking-config.mjs"
    )
    const moduleUrl = pathToFileURL(validatorPath)
    moduleUrl.searchParams.set("t", `${Date.now()}-${Math.random()}`)
    const { validateNetworkingConfig } = await import(moduleUrl.href)

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "quickstart",
        TLDW_INTERNAL_API_ORIGIN: "http://app:8000",
        NEXT_PUBLIC_API_URL: "http://192.168.1.20:8000"
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)
  })

  it("requires an absolute NEXT_PUBLIC_API_URL in advanced mode", async () => {
    const validatorPath = path.resolve(
      __dirname,
      "..",
      "..",
      "scripts",
      "validate-networking-config.mjs"
    )
    const moduleUrl = pathToFileURL(validatorPath)
    moduleUrl.searchParams.set("t", `${Date.now()}-${Math.random()}`)
    const { validateNetworkingConfig } = await import(moduleUrl.href)

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced"
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)

    expect(() =>
      validateNetworkingConfig({
        NEXT_PUBLIC_TLDW_DEPLOYMENT_MODE: "advanced",
        NEXT_PUBLIC_API_URL: "/api"
      })
    ).toThrow(/NEXT_PUBLIC_API_URL/i)
  })

  it("does not render app content during server rendering before the guard resolves", () => {
    const originalWindow = globalThis.window
    vi.stubGlobal("window", undefined)

    try {
      const html = renderToString(
        <ConfigurationGuard>
          <div data-testid="server-page-content">Page</div>
        </ConfigurationGuard>
      )

      expect(html).not.toContain("server-page-content")
      expect(html).not.toContain("Page")
    } finally {
      vi.stubGlobal("window", originalWindow)
    }
  })
})
