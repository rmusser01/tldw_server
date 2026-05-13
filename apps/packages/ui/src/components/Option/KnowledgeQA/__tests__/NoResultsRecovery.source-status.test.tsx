import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { NoResultsRecovery } from "../panels/NoResultsRecovery"

vi.mock("@/store/quick-ingest", () => ({
  useQuickIngestStore: (selector: (state: { recentlyIngestedDocs: unknown[] }) => unknown) =>
    selector({ recentlyIngestedDocs: [] }),
}))

const defaultProps = {
  onBroadenScope: vi.fn(),
  onEnableWeb: vi.fn(),
  onShowNearestMatches: vi.fn(),
  webEnabled: false,
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

    expect(screen.getByText("Source diagnostics")).toBeInTheDocument()
    expect(screen.getByText(/Documents & Media: empty/i)).toBeInTheDocument()
    expect(screen.getByText(/World Books: unavailable/i)).toBeInTheDocument()
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
})
