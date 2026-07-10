import { describe, expect, it, vi } from "vitest"

vi.mock("~/components/Layouts/Layout", () => ({
  default: () => null
}))

vi.mock("~/hooks/useDarkmode", () => ({
  useDarkMode: () => ({ mode: "light" })
}))

vi.mock("@/components/Common/PageAssistLoader", () => ({
  PageAssistLoader: () => null
}))

vi.mock("@/hooks/useAutoButtonTitles", () => ({
  useAutoButtonTitles: () => undefined
}))

vi.mock("@/store/layout-ui", () => ({
  useLayoutUiStore: () => undefined
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({ capabilities: null, loading: false })
}))

vi.mock("@/services/settings/registry", async (importOriginal) => {
  const actual =
    await importOriginal<typeof import("@/services/settings/registry")>()
  return { ...actual, setSetting: vi.fn() }
})

import { getRouteBootstrapNamespaces } from "@/routes/app-route"

describe("route namespace bootstrap", () => {
  it("loads Watchlists copy before rendering the Watchlists route", () => {
    expect(getRouteBootstrapNamespaces("options", "/watchlists")).toContain(
      "watchlists"
    )
  })
})
