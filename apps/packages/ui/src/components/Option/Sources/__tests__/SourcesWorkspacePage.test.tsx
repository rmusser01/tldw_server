import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
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

    expect(screen.getByText("Unavailable")).toBeInTheDocument()
    expect(screen.getByText("Sources are unavailable")).toBeInTheDocument()
    expect(
      screen.getByText("This server does not expose the ingestion sources capability.")
    ).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Check server setup" })).toBeInTheDocument()
    expect(hookMocks.useIngestionSourcesQuery).toHaveBeenCalledWith(undefined, {
      enabled: false
    })
  })

  it("keeps raw source endpoint failures in diagnostics", () => {
    hookMocks.useIngestionSourcesQuery.mockReturnValue({
      data: undefined,
      isLoading: false,
      error: {
        status: 404,
        message: "Not Found (GET /api/v1/ingestion-sources)"
      }
    })

    renderWorkspace(<SourcesWorkspacePage />)

    const heading = screen.getByRole("heading", {
      name: "Sources are unavailable"
    })
    const primaryState = heading.closest("div")
    const diagnostics = screen.getByLabelText("Diagnostics")

    expect(primaryState).not.toHaveTextContent("/api/v1/ingestion-sources")
    expect(diagnostics).toHaveTextContent("/api/v1/ingestion-sources")
    expect(diagnostics).toHaveTextContent("404")
    expect(diagnostics).toHaveTextContent("Not Found (GET /api/v1/ingestion-sources)")
    expect(screen.getByRole("button", { name: "Try again" })).toBeInTheDocument()
  })

  it("uses a shared empty state with a creation action when there are no sources", () => {
    hookMocks.useIngestionSourcesQuery.mockReturnValue({
      data: {
        sources: [],
        total: 0
      },
      isLoading: false
    })

    renderWorkspace(<SourcesWorkspacePage />)

    expect(screen.getByText("Empty")).toBeInTheDocument()
    expect(screen.getByText("No sources yet")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Create source" }))

    expect(routerMocks.navigate).toHaveBeenCalledWith("/sources/new")
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
