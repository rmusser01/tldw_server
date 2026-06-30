import { cleanup, render, screen } from "@testing-library/react"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { ReferencesTab } from "../ReferencesTab"
import { useDocumentWorkspaceStore } from "@/store/document-workspace"
import { useConnectionStore } from "@/store/connection"

const mockUseDocumentReferences = vi.fn()

vi.mock("@/hooks/document-workspace", async () => {
  const actual = await vi.importActual<any>("@/hooks/document-workspace")
  return {
    ...actual,
    useDocumentReferences: (...args: any[]) => mockUseDocumentReferences(...args)
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (_key: string, defaultValue?: string) => defaultValue || _key
  })
}))

const renderReferencesTab = () => {
  const queryClient = new QueryClient()

  return render(
    <QueryClientProvider client={queryClient}>
      <ReferencesTab />
    </QueryClientProvider>
  )
}

describe("ReferencesTab design-system empty states", () => {
  let previousWorkspaceState = useDocumentWorkspaceStore.getState()
  let previousConnectionStoreState = useConnectionStore.getState()

  beforeEach(() => {
    previousWorkspaceState = useDocumentWorkspaceStore.getState()
    previousConnectionStoreState = useConnectionStore.getState()

    useDocumentWorkspaceStore.setState({ activeDocumentId: 1 })
    useConnectionStore.setState({
      state: {
        ...previousConnectionStoreState.state,
        isConnected: true,
        mode: "normal"
      }
    })
    mockUseDocumentReferences.mockReturnValue({
      data: null,
      isLoading: false,
      error: null,
      isFetching: false
    })
  })

  afterEach(() => {
    cleanup()
    useDocumentWorkspaceStore.setState(previousWorkspaceState)
    useConnectionStore.setState(previousConnectionStoreState)
    vi.clearAllMocks()
  })

  it("renders the server-unavailable state through the design-system EmptyState", () => {
    const previousConnectionState = useConnectionStore.getState().state
    useConnectionStore.setState({
      state: {
        ...previousConnectionState,
        isConnected: false,
        mode: "normal"
      }
    })

    const { container } = renderReferencesTab()

    const message = screen.getByText("Connect to your server in Settings to use this feature")
    expect(message.closest('[data-ds-component="EmptyState"]')).not.toBeNull()
    expect(container.querySelectorAll('[data-ds-component="EmptyState"]')).toHaveLength(1)
  })

  it("renders reference load errors through the design-system EmptyState", () => {
    mockUseDocumentReferences.mockReturnValue({
      data: null,
      isLoading: false,
      error: new Error("Reference endpoint unavailable"),
      isFetching: false
    })

    const { container } = renderReferencesTab()

    const title = screen.getByText("Failed to load references")
    const emptyState = title.closest('[data-ds-component="EmptyState"]')
    expect(emptyState).not.toBeNull()
    expect(emptyState?.querySelector("svg")).toHaveClass("text-error")
    expect(screen.getByText("Reference endpoint unavailable")).toBeInTheDocument()
    expect(container.querySelectorAll('[data-ds-component="EmptyState"]')).toHaveLength(1)
  })
})
