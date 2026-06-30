import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { NoResultsRecovery } from "../panels/NoResultsRecovery"
import type { KnowledgeSourceHealthState } from "../types"

vi.mock("@/store/quick-ingest", () => ({
  useQuickIngestStore: (selector: (state: { recentlyIngestedDocs: unknown[] }) => unknown) =>
    selector({ recentlyIngestedDocs: [] }),
}))

vi.mock("react-router-dom", async () => {
  const actual = await vi.importActual<typeof import("react-router-dom")>(
    "react-router-dom"
  )
  return {
    ...actual,
    Link: ({ to, ...props }: { to: string } & React.AnchorHTMLAttributes<HTMLAnchorElement>) => (
      <a data-router-link="true" href={to} {...props} />
    ),
  }
})

const defaultProps = {
  onOpenQuickIngest: vi.fn(),
  onEnableWeb: vi.fn(),
  onShowNearestMatches: vi.fn(),
  webEnabled: false,
}

const unavailableHealthState: KnowledgeSourceHealthState = {
  loading: false,
  error: null,
  loadedAt: "2026-05-16T00:00:00Z",
  sources: [
    {
      sourceId: "media_db",
      label: "Documents & Media",
      available: false,
      searchable: false,
      itemCount: null,
      indexedCount: null,
      lastUpdated: null,
      lastIndexed: null,
      indexStatus: "unavailable",
      embeddingStatus: "unavailable",
      disabledReason: "no_retriever_configured",
      workspaceScoped: false,
      hiddenByDefault: false,
      privacyNote: null,
    },
  ],
  bySource: {
    media_db: {
      sourceId: "media_db",
      label: "Documents & Media",
      available: false,
      searchable: false,
      itemCount: null,
      indexedCount: null,
      lastUpdated: null,
      lastIndexed: null,
      disabledReason: "no_retriever_configured",
      indexStatus: "unavailable",
      embeddingStatus: "unavailable",
      workspaceScoped: false,
      hiddenByDefault: false,
      privacyNote: null,
    },
  },
}

describe("NoResultsRecovery source diagnostics", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("distinguishes unavailable and empty selected sources", () => {
    render(
      <NoResultsRecovery
        {...defaultProps}
        sourceStatus={{
          media_db: { status: "empty", count: 0, reason: "no_matching_entries" },
          world_books: {
            status: "unavailable",
            count: 0,
            reason: "no_retriever_configured",
          },
        }}
      />
    )

    expect(screen.getByText("Search diagnostics")).toBeInTheDocument()
    expect(screen.getByText(/Documents & Media: empty/i)).toBeInTheDocument()
    expect(screen.getByText(/World Books: unavailable/i)).toBeInTheDocument()
  })

  it("distinguishes pre-query health from post-query source diagnostics", () => {
    render(
      <NoResultsRecovery
        {...defaultProps}
        selectedSources={["media_db"]}
        sourceHealth={unavailableHealthState}
        sourceStatus={{
          media_db: { status: "empty", count: 0, reason: "no_matching_entries" },
        }}
      />
    )

    expect(screen.getByText("Source readiness")).toBeInTheDocument()
    expect(screen.getByText(/Documents & Media: unavailable/i)).toBeInTheDocument()
    expect(screen.getByText("Search diagnostics")).toBeInTheDocument()
    expect(screen.getByText(/Documents & Media: empty/i)).toBeInTheDocument()
  })

  it("uses copy that preserves the user's provider/privacy decision for web recovery", () => {
    render(<NoResultsRecovery {...defaultProps} />)

    expect(
      screen.getByText(/Web fallback uses your configured server default provider/i)
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Queries stay on your tldw server unless you enable web fallback/i)
    ).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Enable web fallback" }))
    expect(defaultProps.onEnableWeb).toHaveBeenCalledOnce()
  })

  it("uses concrete recovery handoff labels instead of ambiguous add-source copy", () => {
    render(<NoResultsRecovery {...defaultProps} />)

    fireEvent.click(screen.getByRole("button", { name: "Open Quick Ingest" }))
    expect(defaultProps.onOpenQuickIngest).toHaveBeenCalledOnce()

    expect(screen.getByRole("link", { name: "Open source page" })).toHaveAttribute(
      "href",
      "/sources"
    )
    expect(screen.getByRole("link", { name: "Open source page" })).toHaveAttribute(
      "data-router-link",
      "true"
    )
    expect(screen.queryByRole("button", { name: "Add sources" })).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Broaden source scope" })
    ).not.toBeInTheDocument()
    expect(
      screen.queryByRole("button", { name: "Show nearest matches" })
    ).not.toBeInTheDocument()
  })

  it("renders trust reason diagnostics for insufficient evidence recovery", () => {
    render(
      <NoResultsRecovery
        {...defaultProps}
        trustReasonCodes={["no_evidence", "missing_inspectable_evidence"]}
      />
    )

    expect(screen.getByText("Recovery reasons")).toBeInTheDocument()
    expect(screen.getByText(/No searchable evidence was returned/i)).toBeInTheDocument()
    expect(
      screen.getByText(/Cited sources do not include inspectable excerpts/i)
    ).toBeInTheDocument()
  })
})
