import { fireEvent, render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { NoResultsRecovery } from "../panels/NoResultsRecovery"

let recentlyIngestedDocs: unknown[] = []

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, fallback: string) => fallback,
  }),
}))

vi.mock("@/store/quick-ingest", () => ({
  useQuickIngestStore: (selector: (state: { recentlyIngestedDocs: unknown[] }) => unknown) =>
    selector({ recentlyIngestedDocs }),
}))

describe("NoResultsRecovery", () => {
  beforeEach(() => {
    recentlyIngestedDocs = []
  })

  it("offers source and web recovery while hiding nearest matches without candidates", () => {
    const onBroadenScope = vi.fn()
    const onEnableWeb = vi.fn()
    const onShowNearestMatches = vi.fn()

    render(
      <NoResultsRecovery
        onBroadenScope={onBroadenScope}
        onEnableWeb={onEnableWeb}
        onShowNearestMatches={onShowNearestMatches}
        webEnabled={false}
        webAvailable
        hasNearestMatches={false}
      />
    )

    expect(screen.getByText("No results found")).toBeInTheDocument()
    expect(screen.getByText("Try different keywords or fewer constraints.")).toBeInTheDocument()
    expect(screen.getByText("Broaden the question before adding details.")).toBeInTheDocument()
    expect(screen.getByText("Confirm your sources were ingested and indexed.")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Show nearest matches" })).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Broaden source scope" }))
    fireEvent.click(screen.getByRole("button", { name: "Enable web search" }))

    expect(onBroadenScope).toHaveBeenCalledTimes(1)
    expect(onEnableWeb).toHaveBeenCalledTimes(1)
    expect(onShowNearestMatches).not.toHaveBeenCalled()
  })

  it("surfaces nearest matches, web availability, and recent indexing guidance only when relevant", () => {
    recentlyIngestedDocs = [{ id: "doc-1" }]
    const onShowNearestMatches = vi.fn()
    const { rerender } = render(
      <NoResultsRecovery
        onBroadenScope={vi.fn()}
        onEnableWeb={vi.fn()}
        onShowNearestMatches={onShowNearestMatches}
        webEnabled
        webAvailable
        hasNearestMatches
      />
    )

    expect(screen.getByText("Web search enabled")).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Show nearest matches" })).toBeEnabled()
    expect(screen.getByText(/may still be indexing/i)).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Show nearest matches" }))
    expect(onShowNearestMatches).toHaveBeenCalledTimes(1)

    rerender(
      <NoResultsRecovery
        onBroadenScope={vi.fn()}
        onEnableWeb={vi.fn()}
        onShowNearestMatches={vi.fn()}
        webEnabled={false}
        webAvailable={false}
        hasNearestMatches={false}
      />
    )

    expect(screen.getByText("Web search unavailable on this server")).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Enable web search" })).not.toBeInTheDocument()
  })
})
