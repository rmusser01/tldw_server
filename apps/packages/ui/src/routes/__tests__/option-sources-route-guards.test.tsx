import React from "react"
import { render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

import OptionSourcesNew from "../option-sources-new"
import OptionSourcesDetail from "../option-sources-detail"

const renderRoute = (ui: React.ReactElement) =>
  render(<MemoryRouter initialEntries={["/sources"]}>{ui}</MemoryRouter>)

const onlineMocks = vi.hoisted(() => ({
  useServerOnline: vi.fn()
}))

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn()
}))

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
}))

vi.mock("@/components/Layouts/Layout", () => ({
  __esModule: true,
  default: ({ children }: { children: React.ReactNode }) => (
    <div data-testid="option-layout">{children}</div>
  )
}))

vi.mock("@/components/Common/RouteErrorBoundary", () => ({
  RouteErrorBoundary: ({
    routeId,
    children
  }: {
    routeId: string
    children: React.ReactNode
  }) => <div data-testid={`route-boundary-${routeId}`}>{children}</div>
}))

vi.mock("@/components/Option/Sources/SourceForm", () => ({
  SourceForm: ({ mode }: { mode: string }) => <div data-testid={`source-form-${mode}`} />
}))

vi.mock("@/components/Option/Sources/SourceDetailPage", () => ({
  SourceDetailPage: ({ sourceId }: { sourceId: string }) => (
    <div data-testid="source-detail-page">{sourceId}</div>
  )
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useParams: () => ({ sourceId: "42" })
  }
})

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => onlineMocks.useServerOnline()
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.useServerCapabilities()
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionMocks.useConnectionUxState()
}))

describe("sources option route guards", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    onlineMocks.useServerOnline.mockReturnValue(true)
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: { hasIngestionSources: true }
    })
  })

  it("blocks the new route with an offline state when the server is unavailable", () => {
    onlineMocks.useServerOnline.mockReturnValue(false)
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "error_unreachable",
      hasCompletedFirstRun: true
    })

    renderRoute(<OptionSourcesNew />)

    expect(
      screen.getByText("Can't reach your tldw server right now.")
    ).toBeInTheDocument()
    expect(screen.queryByTestId("source-form-create")).not.toBeInTheDocument()
  })

  it("blocks the detail route when ingestion sources are unsupported", () => {
    capabilityMocks.useServerCapabilities.mockReturnValue({
      loading: false,
      capabilities: { hasIngestionSources: false }
    })

    renderRoute(<OptionSourcesDetail />)

    expect(
      screen.getByText("The connected server does not advertise ingestion source management.")
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      "/api/v1/ingestion-sources"
    )
    expect(screen.queryByTestId("source-detail-page")).not.toBeInTheDocument()
  })

  it("renders the new and detail routes when ingestion sources are supported", () => {
    const { rerender } = renderRoute(<OptionSourcesNew />)

    expect(screen.getByTestId("route-boundary-sources-new")).toBeVisible()
    expect(screen.getByTestId("source-form-create")).toBeVisible()

    rerender(
      <MemoryRouter initialEntries={["/sources/42"]}>
        <OptionSourcesDetail />
      </MemoryRouter>
    )

    expect(screen.getByTestId("route-boundary-sources-detail")).toBeVisible()
    expect(screen.getByTestId("source-detail-page")).toHaveTextContent("42")
  })
})
