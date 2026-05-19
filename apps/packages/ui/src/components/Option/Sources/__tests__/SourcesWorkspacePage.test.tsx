import React from "react"
import { fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const hookMocks = vi.hoisted(() => ({
  useIngestionSourcesQuery: vi.fn(),
  useSyncIngestionSourceMutation: vi.fn(),
  useUpdateIngestionSourceMutation: vi.fn()
}))

const capabilityMocks = vi.hoisted(() => ({
  useServerCapabilities: vi.fn()
}))

const onlineMocks = vi.hoisted(() => ({
  useServerOnline: vi.fn()
}))

const connectionMocks = vi.hoisted(() => ({
  useConnectionUxState: vi.fn()
}))

const routerMocks = vi.hoisted(() => ({
  navigate: vi.fn()
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

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => onlineMocks.useServerOnline()
}))

vi.mock("@/hooks/useConnectionState", () => ({
  useConnectionUxState: () => connectionMocks.useConnectionUxState()
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => capabilityMocks.useServerCapabilities()
}))

vi.mock("@/hooks/use-ingestion-sources", () => ({
  useIngestionSourcesQuery: (...args: unknown[]) => hookMocks.useIngestionSourcesQuery(...args),
  useSyncIngestionSourceMutation: (...args: unknown[]) =>
    hookMocks.useSyncIngestionSourceMutation(...args),
  useUpdateIngestionSourceMutation: (...args: unknown[]) =>
    hookMocks.useUpdateIngestionSourceMutation(...args)
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>("react-router-dom")
  return {
    ...actual,
    useNavigate: () => routerMocks.navigate
  }
})

import { SourcesWorkspacePage } from "@/components/Option/Sources/SourcesWorkspacePage"

const renderWorkspace = (ui: React.ReactElement) =>
  render(<MemoryRouter initialEntries={["/sources"]}>{ui}</MemoryRouter>)

describe("SourcesWorkspacePage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    onlineMocks.useServerOnline.mockReturnValue(true)
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "connected_ok",
      hasCompletedFirstRun: true
    })
    capabilityMocks.useServerCapabilities.mockReturnValue({
      capabilities: { hasIngestionSources: true },
      loading: false
    })
    hookMocks.useIngestionSourcesQuery.mockReturnValue({
      data: {
        sources: [
          {
            id: "12",
            source_type: "archive_snapshot",
            sink_type: "notes",
            policy: "canonical",
            enabled: true,
            config: { label: "Archive Notes" },
            last_sync_status: "completed",
            last_successful_sync_summary: {
              changed_count: 3,
              degraded_count: 2,
              conflict_count: 1
            }
          }
        ],
        total: 1
      },
      isLoading: false
    })
    hookMocks.useSyncIngestionSourceMutation.mockReturnValue({
      mutateAsync: vi.fn(async () => ({ status: "queued" })),
      isPending: false
    })
    hookMocks.useUpdateIngestionSourceMutation.mockImplementation(() => ({
      mutateAsync: vi.fn(async () => ({ id: "12", enabled: false })),
      isPending: false
    }))
  })

  it("renders sources with quick actions and degraded/conflict counts", async () => {
    const syncMutate = vi.fn(async () => ({ status: "queued" }))
    hookMocks.useSyncIngestionSourceMutation.mockReturnValue({
      mutateAsync: syncMutate,
      isPending: false
    })

    renderWorkspace(<SourcesWorkspacePage />)

    expect(await screen.findByText("Archive Notes")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Sync now" })).toBeEnabled()
    expect(screen.getByText(/Degraded 2/i)).toBeInTheDocument()
    expect(screen.getByText(/Detached 1/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Sync now" }))
    fireEvent.click(screen.getByRole("button", { name: "New source" }))
    fireEvent.click(screen.getByRole("button", { name: "Open detail" }))

    await waitFor(() => {
      expect(syncMutate).toHaveBeenCalledWith("12")
    })
    expect(routerMocks.navigate).toHaveBeenCalledWith("/sources/new")
    expect(routerMocks.navigate).toHaveBeenCalledWith("/sources/12")
  })

  it("shows credential repair guidance when the server is reachable but auth is missing", () => {
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "error_auth",
      hasCompletedFirstRun: true
    })

    renderWorkspace(<SourcesWorkspacePage />)

    expect(
      screen.getByText("Add your credentials before Sources can load data.")
    ).toBeInTheDocument()
  })

  it("keeps the unreachable message for genuine connectivity failures", () => {
    connectionMocks.useConnectionUxState.mockReturnValue({
      uxState: "error_unreachable",
      hasCompletedFirstRun: true
    })

    renderWorkspace(<SourcesWorkspacePage />)

    expect(
      screen.getByText("Can't reach your tldw server right now.")
    ).toBeInTheDocument()
  })

  it("renders a feature unavailable state when ingestion sources are unsupported", () => {
    capabilityMocks.useServerCapabilities.mockReturnValue({
      capabilities: { hasIngestionSources: false },
      loading: false
    })

    renderWorkspace(<SourcesWorkspacePage />)

    expect(
      screen.getByRole("heading", { name: "Sources are unavailable on this server" })
    ).toBeInTheDocument()
    expect(
      screen.getByText("The connected server does not advertise ingestion source management.")
    ).toBeInTheDocument()
    expect(screen.getByLabelText("Diagnostics")).toHaveTextContent(
      "/api/v1/ingestion-sources"
    )
    expect(hookMocks.useIngestionSourcesQuery).toHaveBeenCalledWith(undefined, {
      enabled: false
    })
  })

  it("moves raw sources load failures into diagnostics", () => {
    const rawError = Object.assign(
      new Error("Request failed: 404 (GET /api/v1/ingestion-sources)"),
      { status: 404 }
    )
    hookMocks.useIngestionSourcesQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: rawError,
      refetch: vi.fn()
    })

    renderWorkspace(<SourcesWorkspacePage />)

    expect(
      screen.getByRole("heading", { name: "Sources are unavailable on this server" })
    ).toBeInTheDocument()
    expect(
      screen.getByText("The connected server does not advertise ingestion source management.")
    ).toBeInTheDocument()

    const diagnostics = screen.getByLabelText("Diagnostics")
    expect(within(diagnostics).getByText("/api/v1/ingestion-sources")).toBeInTheDocument()
    expect(within(diagnostics).getByText("404")).toBeInTheDocument()
    expect(
      within(diagnostics).getByText("Request failed: 404 (GET /api/v1/ingestion-sources)")
    ).toBeInTheDocument()
  })

  it("wires enable disable actions through the update mutation", async () => {
    const updateMutate = vi.fn(async () => ({ id: "12", enabled: false }))
    hookMocks.useUpdateIngestionSourceMutation.mockReturnValue({
      mutateAsync: updateMutate,
      isPending: false
    })

    renderWorkspace(<SourcesWorkspacePage />)

    fireEvent.click(await screen.findByRole("button", { name: "Disable" }))

    await waitFor(() => {
      expect(updateMutate).toHaveBeenCalledWith({
        enabled: false
      })
    })
  })

  it("shows an admin badge in admin mode", async () => {
    renderWorkspace(<SourcesWorkspacePage mode="admin" />)

    expect(await screen.findByText("Admin view")).toBeInTheDocument()
  })
})
