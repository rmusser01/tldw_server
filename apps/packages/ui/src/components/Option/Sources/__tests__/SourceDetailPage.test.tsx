import React from "react"
import { fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { beforeEach, describe, expect, it, vi } from "vitest"

const hookMocks = vi.hoisted(() => ({
  useIngestionSourceDetailQuery: vi.fn(),
  useIngestionSourceItemsQuery: vi.fn(),
  useSyncIngestionSourceMutation: vi.fn(),
  useUploadIngestionSourceArchiveMutation: vi.fn(),
  useReattachIngestionSourceItemMutation: vi.fn()
}))

vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallbackOrOptions?: string | { defaultValue?: string }
    ) => {
      if (typeof fallbackOrOptions === "string") return fallbackOrOptions
      if (fallbackOrOptions && typeof fallbackOrOptions === "object") {
        return fallbackOrOptions.defaultValue || key
      }
      return key
    }
  })
}))

vi.mock("@/hooks/use-ingestion-sources", () => ({
  useIngestionSourceDetailQuery: (...args: unknown[]) =>
    hookMocks.useIngestionSourceDetailQuery(...args),
  useIngestionSourceItemsQuery: (...args: unknown[]) =>
    hookMocks.useIngestionSourceItemsQuery(...args),
  useSyncIngestionSourceMutation: (...args: unknown[]) =>
    hookMocks.useSyncIngestionSourceMutation(...args),
  useUploadIngestionSourceArchiveMutation: (...args: unknown[]) =>
    hookMocks.useUploadIngestionSourceArchiveMutation(...args),
  useReattachIngestionSourceItemMutation: (...args: unknown[]) =>
    hookMocks.useReattachIngestionSourceItemMutation(...args)
}))

vi.mock("@/components/Option/Sources/SourceForm", () => ({
  SourceForm: ({ mode }: { mode: string }) => <div>{`source-form-${mode}`}</div>
}))

import { SourceDetailPage } from "@/components/Option/Sources/SourceDetailPage"

const renderDetail = (ui: React.ReactElement) =>
  render(<MemoryRouter initialEntries={["/sources/42"]}>{ui}</MemoryRouter>)

const expectInsideDesignSystemAlert = (text: string | RegExp) => {
  const node = screen.getByText(text)
  const alert = node.closest('[data-ds-component="Alert"]')
  expect(alert).toHaveAttribute("data-ds-component", "Alert")
  return alert
}

describe("SourceDetailPage", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    hookMocks.useIngestionSourceDetailQuery.mockReturnValue({
      data: {
        id: "42",
        source_type: "archive_snapshot",
        sink_type: "notes",
        policy: "canonical",
        enabled: true,
        config: { label: "Archive Notes" },
        last_successful_snapshot_id: "7",
        last_error: "Archive member could not be parsed",
        last_successful_sync_summary: {
          changed_count: 2,
          degraded_count: 1,
          conflict_count: 1
        }
      },
      isLoading: false
    })
    hookMocks.useIngestionSourceItemsQuery.mockReturnValue({
      data: {
        items: [
          {
            id: "501",
            source_id: "42",
            normalized_relative_path: "notes/a.md",
            sync_status: "conflict_detached"
          },
          {
            id: "502",
            source_id: "42",
            normalized_relative_path: "docs/report.pdf",
            sync_status: "degraded_ingestion_error"
          }
        ],
        total: 2
      },
      isLoading: false
    })
    hookMocks.useSyncIngestionSourceMutation.mockReturnValue({
      mutateAsync: vi.fn(async () => ({ status: "queued" })),
      isPending: false
    })
    hookMocks.useUploadIngestionSourceArchiveMutation.mockReturnValue({
      mutateAsync: vi.fn(async () => ({ status: "queued" })),
      isPending: false
    })
    hookMocks.useReattachIngestionSourceItemMutation.mockReturnValue({
      mutateAsync: vi.fn(async () => ({ id: "501" })),
      isPending: false
    })
  })

  it("shows detached items and allows reattach without hiding degraded state", async () => {
    const reattachMutate = vi.fn(async () => ({ id: "501" }))
    hookMocks.useReattachIngestionSourceItemMutation.mockReturnValue({
      mutateAsync: reattachMutate,
      isPending: false
    })

    renderDetail(<SourceDetailPage sourceId="42" />)

    expect(await screen.findByText("conflict_detached")).toBeInTheDocument()
    expect(screen.getByText("degraded_ingestion_error")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Reattach" }))

    await waitFor(() => {
      expect(reattachMutate).toHaveBeenCalledWith("501")
    })
  })

  it("shows sync and archive controls plus immutable source hints", async () => {
    const syncMutate = vi.fn(async () => ({ status: "queued" }))
    hookMocks.useSyncIngestionSourceMutation.mockReturnValue({
      mutateAsync: syncMutate,
      isPending: false
    })

    renderDetail(<SourceDetailPage sourceId="42" />)

    expect(await screen.findByRole("button", { name: "Sync now" })).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Upload archive" })).toBeInTheDocument()
    const errorAlert = expectInsideDesignSystemAlert(
      "Archive member could not be parsed"
    )
    expect(errorAlert).toHaveAttribute("role", "alert")
    const identityAlert = expectInsideDesignSystemAlert(
      "Source identity is locked after the first successful sync."
    )
    expect(identityAlert).toHaveAttribute("role", "status")
    expect(identityAlert).toHaveAttribute("aria-live", "polite")

    fireEvent.click(screen.getByRole("button", { name: "Sync now" }))

    await waitFor(() => {
      expect(syncMutate).toHaveBeenCalledWith("42")
    })
  })

  it("filters tracked items between detached and degraded states", async () => {
    renderDetail(<SourceDetailPage sourceId="42" />)

    expect(await screen.findByText("notes/a.md")).toBeInTheDocument()
    expect(screen.getByText("docs/report.pdf")).toBeInTheDocument()

    fireEvent.click(screen.getByRole("button", { name: "Detached" }))

    await waitFor(() => {
      expect(screen.getByText("notes/a.md")).toBeInTheDocument()
      expect(screen.queryByText("docs/report.pdf")).not.toBeInTheDocument()
    })

    fireEvent.click(screen.getByRole("button", { name: "Degraded" }))

    await waitFor(() => {
      expect(screen.queryByText("notes/a.md")).not.toBeInTheDocument()
      expect(screen.getByText("docs/report.pdf")).toBeInTheDocument()
    })
  })
})
