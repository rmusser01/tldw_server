import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { afterAll, beforeAll, beforeEach, describe, expect, it, vi } from "vitest"
import { useQuery } from "@tanstack/react-query"
import { ChatbooksPlaygroundPage } from "../ChatbooksPlaygroundPage"

const { capabilitiesMock, useQueryMock, tldwClientMock } = vi.hoisted(() => ({
  capabilitiesMock: {
    hasChatbooks: true
  },
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
    importChatbook: vi.fn(),
    previewOpenWebUIHydration: vi.fn(),
    createOpenWebUIHydrationJob: vi.fn(),
    getOpenWebUIHydrationJob: vi.fn()
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
    capabilities: capabilitiesMock
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

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((resolvePromise, rejectPromise) => {
    resolve = resolvePromise
    reject = rejectPromise
  })
  return { promise, resolve, reject }
}

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const expectInsideDesignSystemAlert = (text: string | RegExp) => {
  const node = screen.getByText(text)
  expect(node.closest('[data-ds-component="Alert"]')).toHaveAttribute(
    "data-ds-component",
    "Alert"
  )
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
    capabilitiesMock.hasChatbooks = true
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

  it("previews OpenWebUI database users and submits the selected source user", async () => {
    tldwClientMock.previewChatbook.mockResolvedValueOnce({
      openwebui_db_preview: {
        user_count: 2,
        users: [
          {
            source_user_id: "user-a",
            display_label: "Alice",
            email: "alice@example.test",
            chat_count: 2,
            folder_count: 1,
            message_count: 12,
            branched_chat_count: 1,
            duplicate_chat_count: 0,
            archived_chat_count: 0,
            pinned_chat_count: 1,
            attachment_reference_count: 3,
            warning_count: 0,
            warnings: []
          },
          {
            source_user_id: "user-b",
            display_label: "Bob",
            email: "bob@example.test",
            chat_count: 1,
            folder_count: 0,
            message_count: 4,
            branched_chat_count: 0,
            duplicate_chat_count: 0,
            archived_chat_count: 0,
            pinned_chat_count: 0,
            attachment_reference_count: 0,
            warning_count: 0,
            warnings: []
          }
        ],
        warnings: []
      }
    })
    tldwClientMock.importChatbook.mockResolvedValueOnce({ success: true })

    const { container } = render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))
    expect(screen.getByRole("option", { name: "OpenWebUI database" })).toBeInTheDocument()
    const sourceSelect = screen
      .getAllByRole("combobox")
      .find((element) => (element as HTMLSelectElement).value === "chatbook")
    fireEvent.change(sourceSelect!, { target: { value: "openwebui_db" } })

    await waitFor(() => {
      expect(screen.getByText("Drop an OpenWebUI webui.db or .sqlite database or click to browse")).toBeInTheDocument()
    })
    expect(container.querySelector(".ant-upload-drag input[type=\"file\"]")).toHaveAttribute("accept", ".db,.sqlite")
    expect(screen.queryByText("Import media")).not.toBeInTheDocument()
    expect(screen.queryByText("Import embeddings")).not.toBeInTheDocument()

    const uploadInput = container.querySelector(".ant-upload-drag input[type=\"file\"]") as HTMLInputElement
    fireEvent.change(uploadInput, {
      target: {
        files: [new File(["SQLite format 3"], "webui.db", { type: "application/vnd.sqlite3" })]
      }
    })

    await waitFor(() => {
      expect(tldwClientMock.previewChatbook).toHaveBeenCalledWith(
        expect.objectContaining({ name: "webui.db" }),
        { source_format: "openwebui_db" }
      )
    })
    await waitFor(() => {
      expect(screen.getByText("Alice")).toBeInTheDocument()
      expect(screen.getByText("alice@example.test")).toBeInTheDocument()
    })

    const importButton = screen.getByRole("button", { name: "Import chatbook" })
    expect(importButton).toBeDisabled()

    fireEvent.change(screen.getByLabelText("Select source user"), {
      target: { value: "user-a" }
    })

    await waitFor(() => {
      expect(screen.getByText("Destination: OpenWebUI / Alice (user-a) / source folders")).toBeInTheDocument()
    })
    fireEvent.click(importButton)

    await waitFor(() => {
      expect(tldwClientMock.importChatbook).toHaveBeenCalledWith(
        expect.objectContaining({ name: "webui.db" }),
        expect.objectContaining({
          source_format: "openwebui_db",
          selected_openwebui_user_id: "user-a",
          import_media: false,
          import_embeddings: false
        })
      )
    })
  })

  it("ignores stale OpenWebUI preview responses when a newer file is selected", async () => {
    const firstPreview = deferred<any>()
    const secondPreview = deferred<any>()
    tldwClientMock.previewChatbook
      .mockImplementationOnce(() => firstPreview.promise)
      .mockImplementationOnce(() => secondPreview.promise)

    const { container } = render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))
    const sourceSelect = screen
      .getAllByRole("combobox")
      .find((element) => (element as HTMLSelectElement).value === "chatbook")
    fireEvent.change(sourceSelect!, { target: { value: "openwebui_json" } })

    const firstUploadInput = container.querySelector(".ant-upload-drag input[type=\"file\"]") as HTMLInputElement
    fireEvent.change(firstUploadInput, {
      target: {
        files: [new File(["[]"], "first.json", { type: "application/json" })]
      }
    })

    const secondUploadInput = container.querySelector(".ant-upload-drag input[type=\"file\"]") as HTMLInputElement
    fireEvent.change(secondUploadInput, {
      target: {
        files: [new File(["[]"], "second.json", { type: "application/json" })]
      }
    })

    await waitFor(() => {
      expect(tldwClientMock.previewChatbook).toHaveBeenCalledTimes(2)
    })

    await act(async () => {
      secondPreview.resolve({
        openwebui_preview: {
          chat_count: 22,
          message_count: 44,
          branched_chat_count: 0,
          duplicate_chat_count: 0,
          attachment_reference_count: 0,
          malformed_chat_count: 0,
          warnings: []
        }
      })
      await secondPreview.promise
    })

    expect(screen.getByText("22")).toBeInTheDocument()

    await act(async () => {
      firstPreview.resolve({
        openwebui_preview: {
          chat_count: 11,
          message_count: 22,
          branched_chat_count: 0,
          duplicate_chat_count: 0,
          attachment_reference_count: 0,
          malformed_chat_count: 0,
          warnings: []
        }
      })
      await firstPreview.promise
    })

    expect(screen.getByText("22")).toBeInTheDocument()
    expect(screen.queryByText("11")).not.toBeInTheDocument()
  })

  it("previews OpenWebUI hydration before enabling job creation", async () => {
    tldwClientMock.previewOpenWebUIHydration.mockResolvedValueOnce({
      summary: {
        referenced_files: 3,
        resolved_files: 2,
        image_files: 1,
        media_files: 1,
        missing_files: 1,
        unsupported_files: 0,
        failed_files: 0,
        hydrated_images: 0,
        registered_media_files: 0,
        already_hydrated: 0,
        processed_files: 0,
        warning_count: 1
      },
      warnings: ["Missing 1 source file"]
    })
    tldwClientMock.createOpenWebUIHydrationJob.mockResolvedValueOnce({
      job_id: "hydration-job-1",
      status: "queued"
    })

    render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))
    const sourceSelect = screen
      .getAllByRole("combobox")
      .find((element) => (element as HTMLSelectElement).value === "chatbook")
    fireEvent.change(sourceSelect!, { target: { value: "openwebui_json" } })

    fireEvent.change(screen.getByLabelText("OpenWebUI data root"), {
      target: { value: "/srv/openwebui" }
    })
    fireEvent.change(screen.getByLabelText("Imported conversation IDs"), {
      target: { value: "conv-a\nconv-b" }
    })
    fireEvent.change(screen.getByLabelText("OpenWebUI source user id"), {
      target: { value: "ow-user" }
    })

    const runButton = screen.getByRole("button", { name: "Run hydration job" })
    expect(runButton).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Preview attachments" }))

    await waitFor(() => {
      expect(tldwClientMock.previewOpenWebUIHydration).toHaveBeenCalledWith({
        openwebui_data_root: "/srv/openwebui",
        scope: {
          conversation_ids: ["conv-a", "conv-b"],
          source_user_id: "ow-user"
        },
        process_supported_files: false
      })
    })
    await waitFor(() => {
      expect(screen.getByText("Referenced files")).toBeInTheDocument()
      expect(screen.getByText("Missing 1 source file")).toBeInTheDocument()
    })
    expectInsideDesignSystemAlert("Hydration warnings")

    fireEvent.click(runButton)

    await waitFor(() => {
      expect(tldwClientMock.createOpenWebUIHydrationJob).toHaveBeenCalledWith({
        openwebui_data_root: "/srv/openwebui",
        scope: {
          conversation_ids: ["conv-a", "conv-b"],
          source_user_id: "ow-user"
        },
        process_supported_files: false
      })
    })
    expect(screen.getByText("queued")).toBeInTheDocument()
  })

  it("requires a fresh hydration preview after selecting a new OpenWebUI preview file", async () => {
    tldwClientMock.previewChatbook.mockResolvedValue({
      openwebui_preview: {
        chat_count: 1,
        message_count: 2,
        branched_chat_count: 0,
        duplicate_chat_count: 0,
        attachment_reference_count: 1,
        malformed_chat_count: 0,
        warnings: []
      }
    })
    tldwClientMock.previewOpenWebUIHydration.mockResolvedValue({
      summary: {
        referenced_files: 1,
        resolved_files: 1,
        image_files: 1,
        media_files: 0,
        missing_files: 0,
        unsupported_files: 0,
        failed_files: 0,
        hydrated_images: 0,
        registered_media_files: 0,
        already_hydrated: 0,
        processed_files: 0,
        warning_count: 0
      },
      warnings: []
    })

    const { container } = render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))
    const sourceSelect = screen
      .getAllByRole("combobox")
      .find((element) => (element as HTMLSelectElement).value === "chatbook")
    fireEvent.change(sourceSelect!, { target: { value: "openwebui_json" } })

    const uploadInput = container.querySelector(".ant-upload-drag input[type=\"file\"]") as HTMLInputElement
    fireEvent.change(uploadInput, {
      target: {
        files: [new File(["[]"], "first.json", { type: "application/json" })]
      }
    })
    await waitFor(() => {
      expect(tldwClientMock.previewChatbook).toHaveBeenCalledTimes(1)
    })
    await waitFor(() => {
      expect(screen.getByText("OpenWebUI preview")).toBeInTheDocument()
    })

    fireEvent.change(screen.getByLabelText("OpenWebUI data root"), {
      target: { value: "/srv/openwebui" }
    })
    fireEvent.change(screen.getByLabelText("Imported conversation IDs"), {
      target: { value: "conv-a" }
    })

    const runButton = screen.getByRole("button", { name: "Run hydration job" })
    fireEvent.click(screen.getByRole("button", { name: "Preview attachments" }))
    await waitFor(() => {
      expect(runButton).toBeEnabled()
    })

    const secondUploadInput = container.querySelector(".ant-upload-drag input[type=\"file\"]") as HTMLInputElement
    fireEvent.change(secondUploadInput, {
      target: {
        files: [new File(["[]"], "second.json", { type: "application/json" })]
      }
    })
    await waitFor(() => {
      expect(tldwClientMock.previewChatbook).toHaveBeenCalledTimes(2)
    })

    expect(runButton).toBeDisabled()
    fireEvent.click(runButton)
    expect(tldwClientMock.createOpenWebUIHydrationJob).not.toHaveBeenCalled()
  }, 10000)

  it("requires a fresh hydration preview after opting into supported-file processing", async () => {
    tldwClientMock.previewOpenWebUIHydration.mockResolvedValue({
      summary: {
        referenced_files: 1,
        resolved_files: 1,
        image_files: 0,
        media_files: 1,
        missing_files: 0,
        unsupported_files: 0,
        failed_files: 0,
        hydrated_images: 0,
        registered_media_files: 0,
        already_hydrated: 0,
        processed_files: 1,
        warning_count: 0
      },
      warnings: []
    })

    render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))
    const sourceSelect = screen
      .getAllByRole("combobox")
      .find((element) => (element as HTMLSelectElement).value === "chatbook")
    fireEvent.change(sourceSelect!, { target: { value: "openwebui_json" } })

    fireEvent.change(screen.getByLabelText("OpenWebUI data root"), {
      target: { value: "/srv/openwebui" }
    })
    fireEvent.change(screen.getByLabelText("Imported conversation IDs"), {
      target: { value: "conv-a" }
    })

    const runButton = screen.getByRole("button", { name: "Run hydration job" })
    fireEvent.click(screen.getByRole("button", { name: "Preview attachments" }))
    await waitFor(() => {
      expect(runButton).toBeEnabled()
    })

    fireEvent.click(screen.getByRole("switch", { name: "Process supported files" }))
    expect(runButton).toBeDisabled()

    fireEvent.click(screen.getByRole("button", { name: "Preview attachments" }))
    await waitFor(() => {
      expect(tldwClientMock.previewOpenWebUIHydration).toHaveBeenLastCalledWith(
        expect.objectContaining({
          process_supported_files: true
        })
      )
    })
    await waitFor(() => {
      expect(runButton).toBeEnabled()
    })
  }, 10000)

  it("disables OpenWebUI hydration controls when Chatbooks capability is unavailable", async () => {
    const { rerender } = render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))
    const sourceSelect = screen
      .getAllByRole("combobox")
      .find((element) => (element as HTMLSelectElement).value === "chatbook")
    fireEvent.change(sourceSelect!, { target: { value: "openwebui_json" } })

    capabilitiesMock.hasChatbooks = false
    rerender(<ChatbooksPlaygroundPage />)

    expect(screen.getByLabelText("OpenWebUI data root")).toBeDisabled()
    expect(screen.getByLabelText("Imported conversation IDs")).toBeDisabled()
    expect(screen.getByRole("switch", { name: "Process supported files" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Preview attachments" })).toBeDisabled()
    expect(screen.getByRole("button", { name: "Run hydration job" })).toBeDisabled()
    expectInsideDesignSystemAlert("Chatbooks is not available on this server.")
  }, 10000)
})
