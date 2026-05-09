import { describe, it, expect, vi, afterEach, beforeEach } from "vitest"
import { render, cleanup } from "@testing-library/react"
import { QuickInsightsTab } from "../QuickInsightsTab"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { useConnectionStore } from "@/store/connection"

const mockUseDocumentInsights = vi.fn()
const mockUseGenerateInsightsMutation = vi.fn()

vi.mock("@/hooks/document-workspace", async () => {
  const actual = await vi.importActual<any>("@/hooks/document-workspace")
  return {
    ...actual,
    useDocumentInsights: (...args: any[]) => mockUseDocumentInsights(...args),
    useGenerateInsightsMutation: () => mockUseGenerateInsightsMutation(),
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue || _key,
  }),
}))

describe("QuickInsightsTab loading state", () => {
  beforeEach(() => {
    useDocumentWorkspaceStore.setState({ activeDocumentId: 1 })
    const prev = useConnectionStore.getState().state
    useConnectionStore.setState({
      state: { ...prev, isConnected: true, mode: "normal" },
    })
    mockUseDocumentInsights.mockReturnValue({
      data: null,
      isLoading: true,
      error: null,
    })
    mockUseGenerateInsightsMutation.mockReturnValue({
      mutate: vi.fn(),
      isPending: false,
      error: null,
    })
  })

  afterEach(() => {
    cleanup()
    useDocumentWorkspaceStore.setState({ activeDocumentId: null })
    const prev = useConnectionStore.getState().state
    useConnectionStore.setState({
      state: { ...prev, isConnected: false, mode: "normal" },
    })
    vi.clearAllMocks()
  })

  it("renders the loading branch through the canonical LoadingState primitive", () => {
    const { container } = render(<QuickInsightsTab />)

    expect(
      container.querySelector('[data-ds-component="LoadingState"]')
    ).toBeInTheDocument()
  })
})
