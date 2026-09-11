import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useQuery } from "@tanstack/react-query"
import { ChatbooksPlaygroundPage } from "../ChatbooksPlaygroundPage"

const fullAccountScope = {
  mode: "full_account" as const,
  total_items: 12,
  pointer_only_count: 2,
  sensitive_category_count: 3,
  warning_count: 1,
  estimated_size_bytes: 2 * 1024 * 1024,
  categories: [
    {
      category: "notes",
      label: "Notes",
      count: 3,
      restore_status: "restorable" as const,
      sensitivity: "personal" as const,
      warning: null
    },
    {
      category: "media_pointers",
      label: "Media pointers",
      count: 2,
      restore_status: "pointer_only" as const,
      sensitivity: "personal" as const,
      warning: "External source bytes are not stored by tldw."
    },
    {
      category: "sensitive_user_values",
      label: "Sensitive user values",
      count: 1,
      restore_status: "restorable" as const,
      sensitivity: "secret" as const,
      warning: null
    }
  ]
}

const { capabilitiesMock, useQueryMock, tldwClientMock } = vi.hoisted(() => ({
  capabilitiesMock: {
    hasChatbooks: true
  },
  useQueryMock: vi.fn(),
  tldwClientMock: {
    initialize: vi.fn(async () => undefined),
    getChatbookExportScope: vi.fn(),
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
            count?: number
            name?: string
            seconds?: number
          }
    ) => {
      if (typeof defaultValueOrOptions === "string") return defaultValueOrOptions
      if (defaultValueOrOptions?.defaultValue) {
        return defaultValueOrOptions.defaultValue
          .replace("{{count}}", String(defaultValueOrOptions.count ?? ""))
          .replace("{{name}}", String(defaultValueOrOptions.name ?? ""))
          .replace("{{seconds}}", String(defaultValueOrOptions.seconds ?? ""))
      }
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

if (!(globalThis as any).ResizeObserver) {
  ;(globalThis as any).ResizeObserver = class ResizeObserver {
    observe() {}
    unobserve() {}
    disconnect() {}
  }
}

const selectExportMode = (value: "full_account" | "selective") => {
  const modeSelect = screen
    .getAllByRole("combobox")
    .find((element) => (element as HTMLSelectElement).value === "full_account")
  expect(modeSelect).toBeDefined()
  fireEvent.change(modeSelect!, { target: { value } })
}

describe("ChatbooksPlaygroundPage backup-all flow", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    capabilitiesMock.hasChatbooks = true
    tldwClientMock.getChatbookExportScope.mockResolvedValue(fullAccountScope)
    tldwClientMock.listChatbookExportJobs.mockResolvedValue({ jobs: [] })
    tldwClientMock.listChatbookImportJobs.mockResolvedValue({ jobs: [] })
    vi.mocked(useQuery).mockReturnValue({
      data: { items: [], total: 0 },
      isLoading: false,
      isError: false,
      error: null,
      refetch: vi.fn()
    } as any)
  })

  it("shows the Backup & Import heading and full-account scope summary", async () => {
    render(<ChatbooksPlaygroundPage />)

    expect(
      screen.getByRole("heading", { name: "Chatbooks Backup & Import" })
    ).toBeInTheDocument()
    expect(screen.getAllByText("Backup supported account data").length)
      .toBeGreaterThan(0)
    expect(screen.getByText(
      "This portable backup does not include Service Prompt overrides."
    )).toBeInTheDocument()

    await waitFor(() => {
      expect(screen.getByText("Backup all scope")).toBeInTheDocument()
    })
    expect(screen.getByText("Notes · 3")).toBeInTheDocument()
    expect(screen.getByText("Media pointers · 2")).toBeInTheDocument()
    expect(screen.getByText("Total items")).toBeInTheDocument()
    expect(screen.getByText("Pointer-only items")).toBeInTheDocument()
    expect(screen.getAllByText("Sensitive categories").length).toBeGreaterThan(0)
    expect(screen.getByText("Warnings")).toBeInTheDocument()
    expect(screen.getByText("Estimated size")).toBeInTheDocument()
    expect(screen.getByText("2.00 MB")).toBeInTheDocument()
    expect(screen.getByText("External source bytes are not stored by tldw.")).toBeInTheDocument()
  })

  it("starts Backup all without content selections", async () => {
    tldwClientMock.exportChatbook.mockResolvedValueOnce({ job_id: "export-job-1" })

    render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("button", { name: "Backup all" }))

    await waitFor(() => {
      expect(tldwClientMock.exportChatbook).toHaveBeenCalledTimes(1)
    })
    expect(tldwClientMock.exportChatbook).toHaveBeenCalledWith(
      expect.not.objectContaining({ content_selections: expect.anything() })
    )
    expect(tldwClientMock.exportChatbook).toHaveBeenCalledWith(
      expect.objectContaining({
        name: expect.stringMatching(/backup/i),
        description: expect.stringMatching(/account/i),
        include_media: true,
        include_embeddings: true,
        include_generated_content: true,
        media_quality: "original",
        format_version: "1.1.0"
      })
    )
  })

  it("keeps selective export blocking zero-item allowlists", async () => {
    render(<ChatbooksPlaygroundPage />)

    selectExportMode("selective")
    fireEvent.change(screen.getByPlaceholderText("Name"), {
      target: { value: "Selective export" }
    })
    fireEvent.change(screen.getByPlaceholderText("Description"), {
      target: { value: "No items selected" }
    })
    fireEvent.click(screen.getByRole("button", { name: "Export selected" }))

    await waitFor(() => {
      expect(tldwClientMock.exportChatbook).not.toHaveBeenCalled()
    })
  }, 10_000)

  it("omits archive import media and embedding flags so default restore handles all archive data", async () => {
    tldwClientMock.previewChatbook.mockResolvedValueOnce({
      manifest: {
        name: "Full archive",
        author: "Tester",
        description: "Archive restore",
        total_size_bytes: 2048,
        total_notes: 1,
        total_characters: 2,
        content_items: [
          { id: "character-1", type: "character", title: "Character one" },
          { id: "character-2", type: "character", title: "Character two" }
        ],
        account_inventory: [
          { category: "account_profiles", label: "Account profile" },
          { category: "account_settings", label: "Account settings" },
          { category: "characters", label: "Characters" }
        ],
        account_inventory_summary: {
          counts: {
            account_profiles: 1,
            account_settings: 1,
            characters: 2
          },
          sensitive_category_count: 1,
          warning_count: 1,
          warnings: ["Review imported provider settings."],
          post_write_verification: true
        }
      }
    })
    tldwClientMock.importChatbook.mockResolvedValueOnce({ success: true })

    const { container } = render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Import" }))
    const uploadInput = container.querySelector(
      ".ant-upload-drag input[type=\"file\"]"
    ) as HTMLInputElement
    fireEvent.change(uploadInput, {
      target: {
        files: [new File(["archive"], "backup.chatbook", { type: "application/zip" })]
      }
    })

    await waitFor(() => {
      expect(tldwClientMock.previewChatbook).toHaveBeenCalledWith(
        expect.objectContaining({ name: "backup.chatbook" }),
        { source_format: "chatbook" }
      )
    })
    expect(screen.getByText("What will be restored")).toBeInTheDocument()
    expect(screen.getByText("Account profile · 1")).toBeInTheDocument()
    expect(screen.getByText("Account settings · 1")).toBeInTheDocument()
    expect(screen.getByText("Verified")).toBeInTheDocument()
    expect(screen.getAllByText("Sensitive categories").length).toBeGreaterThan(0)
    expect(screen.getAllByText("All in archive").length).toBeGreaterThan(0)
    expect(screen.queryByText("Selected: 0")).not.toBeInTheDocument()
    fireEvent.click(screen.getByText("Review 1 warning"))
    expect(
      screen.getByText("Review imported provider settings.")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Import chatbook" }))

    await waitFor(() => {
      expect(tldwClientMock.importChatbook).toHaveBeenCalledTimes(1)
    })
    expect(tldwClientMock.importChatbook).toHaveBeenCalledWith(
      expect.objectContaining({ name: "backup.chatbook" }),
      expect.not.objectContaining({
        import_media: expect.anything(),
        import_embeddings: expect.anything()
      })
    )
  })

  it("shows completed job archive size, warning count, and verification status", async () => {
    tldwClientMock.listChatbookExportJobs.mockResolvedValueOnce({
      jobs: [
        {
          job_id: "job-1",
          status: "completed",
          chatbook_name: "Full account backup",
          created_at: "2026-07-09T12:00:00Z",
          file_size_bytes: 12 * 1024,
          metadata: {
            warning_count: 4,
            post_write_verification: true,
            archive_size_bytes: 12 * 1024
          }
        }
      ]
    })

    render(<ChatbooksPlaygroundPage />)

    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }))

    await waitFor(() => {
      expect(screen.getByText("Full account backup")).toBeInTheDocument()
    })
    expect(screen.getByText("12.0 KB")).toBeInTheDocument()
    expect(screen.getByText("4")).toBeInTheDocument()
    expect(screen.getByText("Verified")).toBeInTheDocument()
  })

  it("renders a completed historical job with stale zero progress as complete", async () => {
    tldwClientMock.listChatbookExportJobs.mockResolvedValueOnce({
      jobs: [
        {
          job_id: "completed-stale-progress",
          status: "completed",
          chatbook_name: "Historical backup",
          created_at: "2026-07-09T12:00:00Z",
          progress_percentage: 0
        }
      ]
    })

    render(<ChatbooksPlaygroundPage />)
    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }))

    await waitFor(() => {
      expect(screen.getByText("Historical backup")).toBeInTheDocument()
    })
    const progressBars = screen.getAllByRole("progressbar")
    expect(progressBars.length).toBeGreaterThan(0)
    progressBars.forEach((bar) => expect(bar).toHaveAttribute("aria-valuenow", "100"))
    expect(screen.queryByText("0%")).not.toBeInTheDocument()
  })

  it("does not render failed terminal jobs at zero or complete progress", async () => {
    tldwClientMock.listChatbookImportJobs.mockResolvedValueOnce({
      jobs: [
        {
          job_id: "failed-stale-progress",
          status: "failed",
          created_at: "2026-07-09T12:00:00Z",
          progress_percentage: 0
        },
        {
          job_id: "failed-complete-progress",
          status: "failed",
          created_at: "2026-07-09T12:01:00Z",
          progress_percentage: 100
        }
      ]
    })

    render(<ChatbooksPlaygroundPage />)
    fireEvent.click(screen.getByRole("tab", { name: "Jobs" }))

    await waitFor(() => {
      expect(screen.getByText("Job ID: failed-stale-progress")).toBeInTheDocument()
    })
    expect(screen.queryByRole("progressbar")).not.toBeInTheDocument()
    expect(screen.queryByText("100%")).not.toBeInTheDocument()
  })

  it("treats a legacy timezone-naive API timestamp as UTC before local formatting", async () => {
    const originalTimezone = process.env.TZ
    process.env.TZ = "America/Los_Angeles"
    const localeSpy = vi
      .spyOn(Date.prototype, "toLocaleString")
      .mockImplementation(function () {
        return this.toISOString()
      })
    tldwClientMock.listChatbookExportJobs.mockResolvedValueOnce({
      jobs: [
        {
          job_id: "legacy-naive-time",
          status: "completed",
          chatbook_name: "UTC backup",
          created_at: "2026-07-09T12:00:00"
        }
      ]
    })

    try {
      render(<ChatbooksPlaygroundPage />)
      fireEvent.click(screen.getByRole("tab", { name: "Jobs" }))

      await waitFor(() => {
        expect(screen.getByText("UTC backup")).toBeInTheDocument()
      })
      expect(screen.getAllByText("2026-07-09T12:00:00.000Z").length).toBeGreaterThan(0)
      expect(screen.queryByText("2026-07-09T19:00:00.000Z")).not.toBeInTheDocument()
    } finally {
      localeSpy.mockRestore()
      if (originalTimezone === undefined) delete process.env.TZ
      else process.env.TZ = originalTimezone
    }
  })
})
