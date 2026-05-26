import { render, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { AddSourceModal } from "../SourcesPane/AddSourceModal"

const { mockListMedia } = vi.hoisted(() => ({
  mockListMedia: vi.fn()
}))

const workspaceStoreState = {
  addSourceModalOpen: true,
  addSourceModalTab: "existing" as const,
  addSourceProcessing: false,
  addSourceError: null as string | null,
  sources: [] as Array<{ mediaId: number }>,
  closeAddSourceModal: vi.fn(),
  setAddSourceModalTab: vi.fn(),
  setAddSourceProcessing: vi.fn(),
  setAddSourceError: vi.fn(),
  addSource: vi.fn(),
  workspaceTag: "workspace:test"
}

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      _key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return _key
    }
  })
}))

vi.mock("@/hooks/useMediaQuery", () => ({
  useMobile: () => false
}))

vi.mock("@/store/workspace", () => ({
  useWorkspaceStore: (
    selector: (state: typeof workspaceStoreState) => unknown
  ) => selector(workspaceStoreState)
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    uploadMedia: vi.fn(),
    addMedia: vi.fn(),
    webSearch: vi.fn().mockResolvedValue({ results: [] }),
    searchMedia: vi.fn().mockResolvedValue({ results: [] }),
    listMedia: mockListMedia,
    updateMediaKeywords: vi.fn()
  }
}))

describe("AddSourceModal Stage 3 performance", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    workspaceStoreState.addSourceModalOpen = true
    workspaceStoreState.addSourceModalTab = "existing"
    mockListMedia.mockResolvedValue({
      media: [
        {
          id: 42,
          title: "Cached Library Item",
          type: "pdf"
        }
      ]
    })
  })

  it("reuses cached library media on reopen within cache TTL", async () => {
    const firstRender = render(<AddSourceModal />)

    await waitFor(() => {
      expect(mockListMedia).toHaveBeenCalledTimes(1)
    })

    firstRender.unmount()

    render(<AddSourceModal />)

    await waitFor(() => {
      expect(mockListMedia).toHaveBeenCalledTimes(1)
    })
  })
})
