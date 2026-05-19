// @vitest-environment jsdom

import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { DatasetsTab } from "../DatasetsTab"
import { useEvaluationsStore } from "@/store/evaluations"

const loadSpy = vi.fn()
const closeViewerSpy = vi.fn()

vi.mock("antd", async () => {
  const actual = await vi.importActual<any>("antd")

  return {
    ...actual,
    Pagination: ({ current = 1, pageSize = 10, total = 0, onChange }: any) => {
      const totalPages = Math.ceil(total / pageSize)
      return (
        <div>
          {Array.from({ length: totalPages }, (_, index) => {
            const page = index + 1
            return (
              <button
                key={page}
                type="button"
                aria-current={current === page ? "page" : undefined}
                onClick={() => onChange?.(page, pageSize)}
              >
                Page {page}
              </button>
            )
          })}
        </div>
      )
    }
  }
})

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
            count?: number
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue.replace(
          "{{count}}",
          String(defaultValueOrOptions.count ?? "")
        )
      }
      return key
    }
  })
}))

vi.mock("../../hooks/useDatasets", () => ({
  useDatasetsList: () => ({
    data: {
      ok: true,
      data: {
        data: [
          {
            id: "dataset-1",
            name: "Dataset One",
            sample_count: 6,
            created: 0,
            created_by: "user_123"
          }
        ]
      }
    },
    isLoading: false,
    isError: false
  }),
  useCreateDataset: () => ({
    mutateAsync: vi.fn(),
    isPending: false
  }),
  useDeleteDataset: () => ({
    mutateAsync: vi.fn(),
    isPending: false
  }),
  useLoadDatasetSamples: () => ({
    mutate: loadSpy,
    mutateAsync: loadSpy,
    isPending: false
  }),
  useCloseDatasetViewer: () => closeViewerSpy,
  parseSamplesJson: vi.fn(() => ({ samples: null, error: null }))
}))

vi.mock("../../components", () => ({
  CopyButton: () => null,
  DatasetUpload: () => null,
  JsonEditor: () => null
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("DatasetsTab sample pagination", () => {
  const originalMatchMedia = window.matchMedia

  beforeAll(() => {
    if (typeof window.matchMedia !== "function") {
      Object.defineProperty(window, "matchMedia", {
        writable: true,
        value: vi.fn().mockImplementation((query: string) => ({
          matches: false,
          media: query,
          onchange: null,
          addListener: vi.fn(),
          removeListener: vi.fn(),
          addEventListener: vi.fn(),
          removeEventListener: vi.fn(),
          dispatchEvent: vi.fn()
        }))
      })
    }
  })

  afterAll(() => {
    Object.defineProperty(window, "matchMedia", {
      writable: true,
      value: originalMatchMedia
    })
  })

  beforeEach(() => {
    vi.clearAllMocks()
    useEvaluationsStore.getState().resetStore()
    useEvaluationsStore.setState({
      viewingDataset: {
        id: "dataset-1",
        name: "Dataset One",
        sample_count: 6,
        created: 0,
        created_by: "user_123"
      },
      datasetSamples: Array.from({ length: 5 }, (_, index) => ({
        input: `sample-${index + 1}`
      })),
      datasetSamplesPage: 1,
      datasetSamplesPageSize: 5,
      datasetSamplesTotal: 6
    })
  })

  it("requests the next dataset page from the API instead of slicing the cached samples client-side", async () => {
    render(<DatasetsTab />)

    expect(screen.getByText(/sample-1/i)).toBeInTheDocument()
    expect(screen.queryByText(/sample-6/i)).not.toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Page 2" }))

    await waitFor(() => {
      expect(loadSpy).toHaveBeenCalledWith({
        datasetId: "dataset-1",
        page: 2,
        pageSize: 5
      })
    })

    expect(screen.getByText(/sample-1/i)).toBeInTheDocument()
    expect(screen.queryByText(/sample-6/i)).not.toBeInTheDocument()
  })
})
