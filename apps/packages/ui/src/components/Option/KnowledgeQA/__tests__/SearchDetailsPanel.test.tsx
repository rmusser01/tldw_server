import { render, screen } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { SearchDetailsPanel } from "../SearchDetailsPanel"

const state = {
  searchDetails: null as any,
  isSearching: false,
  results: [] as Array<Record<string, any>>,
}

vi.mock("../KnowledgeQAProvider", () => ({
  useKnowledgeQA: () => ({
    searchDetails: state.searchDetails,
    isSearching: state.isSearching,
    results: state.results,
  }),
}))

describe("SearchDetailsPanel", () => {
  beforeEach(() => {
    state.searchDetails = null
    state.isSearching = false
    state.results = []
  })

  it("does not render without search detail data", () => {
    const { container } = render(<SearchDetailsPanel />)
    expect(container).toBeEmptyDOMElement()
  })

  it("renders collapsible runtime search details", () => {
    state.results = [
      { score: 0.84 },
      { score: 0.62 },
    ]
    state.searchDetails = {
      expandedQueries: ["q1 variation", "q1 synonym"],
      rerankingEnabled: true,
      rerankingStrategy: "hybrid",
      averageRelevance: 0.84,
      webFallbackEnabled: true,
      webFallbackTriggered: true,
      webFallbackEngine: "duckduckgo",
      faithfulnessScore: 0.91,
      faithfulnessTotalClaims: 4,
      faithfulnessSupportedClaims: 3,
      faithfulnessUnsupportedClaims: 1,
      verificationRate: 0.88,
      verificationCoverage: 0.75,
      verificationReportAvailable: true,
      retrievalLatencyMs: 182,
      documentsConsidered: 42,
      chunksConsidered: 318,
      documentsReturned: 10,
      candidatesConsidered: 42,
      candidatesReturned: 10,
      candidatesRejected: 32,
      alsoConsidered: [
        {
          id: "cand-1",
          title: "Candidate source 1",
          score: 0.41,
          reason: "Below threshold",
        },
      ],
      whyTheseSources: {
        topicality: 0.91,
        diversity: 0.45,
        freshness: null,
      },
    }

    render(<SearchDetailsPanel />)

    expect(screen.getByText("Search details")).toBeInTheDocument()
    expect(screen.getByText(/q1 variation, q1 synonym/)).toBeInTheDocument()
    expect(screen.getByText(/Enabled \(hybrid\)/)).toBeInTheDocument()
    expect(screen.getByText("84%")).toBeInTheDocument()
    expect(screen.getByText(/Triggered \(duckduckgo\)/)).toBeInTheDocument()
    expect(screen.getByText(/faithfulness 91%/i)).toBeInTheDocument()
    expect(screen.getByText(/supported 3/i)).toBeInTheDocument()
    expect(screen.getByText(/unsupported 1/i)).toBeInTheDocument()
    const candidatesLine = screen.getByText(/Candidates considered:/i).closest("div")
    expect(candidatesLine).not.toBeNull()
    expect(candidatesLine?.textContent).toContain("returned 10")
    expect(candidatesLine?.textContent).toContain("rejected 32")
    expect(candidatesLine?.textContent).toContain("retained 24%")
    expect(
      screen.getByText(/considered 42 documents • 318 chunks scanned • returned 10 sources/i)
    ).toBeInTheDocument()
    expect(screen.getByText(/182 ms/i)).toBeInTheDocument()
    expect(screen.getByText("Candidate source 1")).toBeInTheDocument()
    expect(screen.getByText("Threshold")).toBeInTheDocument()
    expect(
      screen.getByText(/\(41% • -21 pts vs weakest included\) — Below threshold/i)
    ).toBeInTheDocument()
  })

  it("explains missing retrieval telemetry without rendering N/A rows", () => {
    state.searchDetails = {
      expandedQueries: [],
      rerankingEnabled: false,
      rerankingStrategy: null,
      averageRelevance: null,
      webFallbackEnabled: false,
      webFallbackTriggered: false,
      webFallbackEngine: null,
      documentsConsidered: null,
      chunksConsidered: null,
      documentsReturned: null,
      candidatesConsidered: null,
      candidatesReturned: null,
      candidatesRejected: null,
      retrievalLatencyMs: null,
      alsoConsidered: [],
      whyTheseSources: null,
    }

    render(<SearchDetailsPanel />)

    expect(screen.getByText("No query expansion terms were reported.")).toBeInTheDocument()
    expect(screen.getByText("No retrieval counts were reported for this search.")).toBeInTheDocument()
    expect(screen.getByText("No closest-miss candidates were reported.")).toBeInTheDocument()
    expect(screen.queryByText("N/A")).not.toBeInTheDocument()
  })

  it("summarizes materialized evidence origins and unavailable results", () => {
    state.results = [
      {
        score: 0.84,
        evidenceOrigin: "local_library",
        sourceStatus: "searched",
      },
      {
        score: 0.67,
        evidenceOrigin: "web_fallback",
        sourceStatus: "searched",
      },
      {
        score: 0,
        evidenceOrigin: "local_library",
        sourceStatus: "unavailable",
        unavailableReason: "deleted_or_unavailable",
      },
    ]
    state.searchDetails = {
      expandedQueries: [],
      rerankingEnabled: false,
      rerankingStrategy: null,
      averageRelevance: null,
      webFallbackEnabled: true,
      webFallbackTriggered: true,
      webFallbackEngine: "duckduckgo",
      documentsConsidered: null,
      chunksConsidered: null,
      documentsReturned: 3,
      candidatesConsidered: null,
      candidatesReturned: 3,
      candidatesRejected: null,
      retrievalLatencyMs: null,
      alsoConsidered: [],
      whyTheseSources: null,
    }

    render(<SearchDetailsPanel />)

    const evidenceLine = screen.getByText(/Evidence origins:/i).closest("div")
    expect(evidenceLine).not.toBeNull()
    expect(evidenceLine?.textContent).toContain("local library 2")
    expect(evidenceLine?.textContent).toContain("web fallback 1")
    expect(evidenceLine?.textContent).toContain("unavailable 1")
  })
})
