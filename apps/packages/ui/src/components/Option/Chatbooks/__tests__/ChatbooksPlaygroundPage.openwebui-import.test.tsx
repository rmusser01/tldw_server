import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { useQuery } from "@tanstack/react-query"
import { ChatbooksPlaygroundPage } from "../ChatbooksPlaygroundPage"

const { useQueryMock, tldwClientMock } = vi.hoisted(() => ({
  useQueryMock: vi.fn(),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    listChatbookExportJobs: vi.fn(async () => ({ jobs: [] })),
    listChatbookImportJobs: vi.fn(async () => ({ jobs: [] })),
    getChatbookExportJob: vi.fn(),
    getChatbookImportJob: vi.fn(),
    downloadChatbookExport: vi.fn(),
    cancelChatbookExportJob: vi.fn(),
    cancelChatbookImportJob: vi.fn(),
    cleanupChatbooks: vi.fn(),
    removeChatbookExportJob: vi.fn(),
    removeChatbookImportJob: vi.fn(),
    exportChatbook: vi.fn(),
    previewChatbook: vi.fn(),
    importChatbook: vi.fn()
  }
}))

vi.mock("@tanstack/react-query", () => ({
  useQuery: useQueryMock
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      defaultValueOrOptions?:
        | string
        | {
            defaultValue?: string
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) return defaultValueOrOptions.defaultValue
      return key
    }
  })
}))

vi.mock("antd", async (importOriginal) => {
  const actual = await importOriginal<typeof import("antd")>()
  const React = await import("react")
  const Select = ({
    value,
    onChange,
    options = [],
    disabled,
    className,
    mode,
    placeholder
  }: any) => (
    <select
      aria-label={placeholder || "select"}
      className={className}
      disabled={disabled}
      value={Array.isArray(value) ? value[0] || "" : value || ""}
      onChange={(event) => {
        const nextValue = event.target.value
        onChange?.(mode === "tags" ? (nextValue ? [nextValue] : []) : nextValue)
      }}
    >
      {(options as Array<{ value: string; label: React.ReactNode }>).map((option) => (
        <option key={option.value} value={option.value}>
          {option.label}
        </option>
      ))}
    </select>
  )
  return {
    ...actual,
    Select
  }
})

vi.mock("@/hooks/useServerOnline", () => ({
  useServerOnline: () => true
}))

vi.mock("@/hooks/useServerCapabilities", () => ({
  useServerCapabilities: () => ({
    capabilities: {
      hasChatbooks: true
    }
  })
}))

vi.mock("@/hooks/useAntdNotification", () => ({
  useAntdNotification: () => ({
    success: vi.fn(),
    error: vi.fn(),
    info: vi.fn(),
    warning: vi.fn()
  })
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: tldwClientMock
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequest: vi.fn()
}))

vi.mock("@/components/Common/PageShell", () => ({
  PageShell: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

vi.mock("@/components/Common/WorkspaceConnectionGate", () => ({
  default: ({ children }: { children: React.ReactNode }) => <>{children}</>
}))

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

describe("ChatbooksPlaygroundPage OpenWebUI import mode", () => {
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
    vi.mocked(useQuery).mockReturnValue({
      data: { items: [], total: 0 },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn()
    } as any)
  })

  it("switches the import tab to OpenWebUI JSON mode and hides archive-only controls", async () => {
    const { container } = render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))

    expect(screen.getByText("Drop a .zip or .chatbook archive or click to browse")).toBeInTheDocument()
    expect(screen.getByText("Import media")).toBeInTheDocument()
    expect(screen.getByText("Import embeddings")).toBeInTheDocument()

    const sourceSelect = screen
      .getAllByRole("combobox")
      .find((element) => (element as HTMLSelectElement).value === "chatbook")
    expect(sourceSelect).not.toBeNull()
    fireEvent.change(sourceSelect!, { target: { value: "openwebui_json" } })

    await waitFor(() => {
      expect(screen.getByText("Drop an OpenWebUI .json export or click to browse")).toBeInTheDocument()
    })
    expect(screen.queryByText("Import media")).not.toBeInTheDocument()
    expect(screen.queryByText("Import embeddings")).not.toBeInTheDocument()
    expect(container.querySelector(".ant-upload-drag input[type=\"file\"]")).toHaveAttribute("accept", ".json")
  })
})
