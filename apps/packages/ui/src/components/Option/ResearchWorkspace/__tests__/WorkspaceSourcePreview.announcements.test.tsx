import React from "react"
import { ConfigProvider } from "antd"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceSource } from "@/types/workspace"
import type { WorkspaceSourcePreviewResponse } from "@/services/tldw/domains/workspace-api"
import { useWorkspaceStore } from "@/store/workspace"
import { WorkspaceSourcePreview } from "../SourcesPane/WorkspaceSourcePreview"

const request = vi.hoisted(() => vi.fn())
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: { getWorkspaceSourcePreview: request } }))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: vi.fn(async () => ({ scopeKey: "owner-7", userId: 7, clientPrincipalVerified: true }))
}))
vi.mock("wxt/browser", () => ({ browser: {} }))
vi.mock("@/store/workspace", async () => {
  const { create } = await import("zustand")
  return { useWorkspaceStore: create<{ workspaceId: string; sources: WorkspaceSource[] }>(() => ({ workspaceId: "", sources: [] })) }
})
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (key: string, fallback?: string) => fallback || key }) }))

const source: WorkspaceSource = { id: "s1", mediaId: 1, title: "Report", type: "pdf", status: "ready", addedAt: new Date("2026-09-29") }
const preview: WorkspaceSourcePreviewResponse = {
  workspace_id: "ws-a", source_id: "s1", media_id: 1, title: "Report", source_type: "pdf", url: null,
  state: "queryable", status_reason: "source_queryable",
  readiness: { metadata_ready: true, text_extracted: true, fts_ready: true, vector_ready: true, citation_ready: true, summary_ready: false, tool_accessible: true },
  content_available: true, preview_mode: "available", unavailable_reason: null,
  text_preview: "Captured report text", text_total_chars: 20, text_truncated: false, snippets: [], generated_at: "2026-09-29T00:00:00Z"
}
const renderPreview = (currentSource: WorkspaceSource = source) => render(
  <ConfigProvider theme={{ token: { motion: false } }}>
    <WorkspaceSourcePreview workspaceId="ws-a" source={currentSource} onClose={vi.fn()} />
  </ConfigProvider>
)

describe("Canonical source preview announcements", () => {
  beforeEach(() => {
    request.mockReset()
    useWorkspaceStore.setState({ workspaceId: "ws-a", sources: [source] })
  })

  it.each([
    ["title", "Canonical report", "Report"],
    ["source type", "html / Ready", "pdf / Ready"]
  ])("renders the canonical %s instead of staged metadata", async (_, canonical, staged) => {
    request.mockResolvedValue({ ...preview, title: "Canonical report", source_type: "html" })
    renderPreview()
    await screen.findByText("Captured report text")
    expect(screen.queryByText(canonical)).toBeInTheDocument()
    expect(screen.queryByText(staged)).not.toBeInTheDocument()
  })

  it("links the canonical URL instead of the staged URL", async () => {
    request.mockResolvedValue({ ...preview, url: "https://canonical.test/report" })
    renderPreview({ ...source, url: "https://staged.test/report" })
    await screen.findByText("Captured report text")
    expect(screen.queryByRole("link", { name: "https://canonical.test/report" })).toHaveAttribute(
      "href", "https://canonical.test/report"
    )
    expect(screen.queryByRole("link", { name: "https://staged.test/report" })).not.toBeInTheDocument()
  })

  it.each([
    null,
    "javascript:alert(1)",
    "java\tscript:alert(1)",
    "data:text/html,<script>alert(1)</script>",
    "/tmp/report.pdf",
    "report.pdf",
    "//source.test/report",
    "mailto:source@test.example",
    "https://[invalid"
  ])("suppresses the staged link when the canonical URL is %s", async (url) => {
    request.mockResolvedValue({ ...preview, url })
    renderPreview({ ...source, url: "https://staged.test/report" })
    expect(screen.getByRole("link")).toHaveAttribute("href", "https://staged.test/report")
    await screen.findByText("Captured report text")
    expect(screen.queryByRole("link")).not.toBeInTheDocument()
  })

  it("announces pending content without making captured text a live region", async () => {
    let resolve!: (value: WorkspaceSourcePreviewResponse) => void
    request.mockReturnValue(new Promise<WorkspaceSourcePreviewResponse>(yes => { resolve = yes }))
    renderPreview()
    expect(screen.getByRole("status")).toHaveTextContent("Loading captured content...")
    expect(screen.getByRole("status")).toHaveAttribute("aria-live", "polite")
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    await act(async () => resolve(preview))
    expect(screen.getByText("Captured report text").closest("[aria-live]")).toBeNull()
    expect(screen.queryByText("Loading captured content...")).not.toBeInTheDocument()
  })

  it("announces failure and then a pending retry without announcing its button", async () => {
    request.mockRejectedValueOnce(new Error("503 unavailable"))
    renderPreview()
    const alert = await screen.findByRole("alert")
    expect(alert).toHaveTextContent("Source preview could not load.")
    expect(alert).toHaveTextContent("503 unavailable")
    expect(alert).toHaveAttribute("aria-atomic", "true")
    const retry = screen.getByRole("button", { name: "Retry preview" })
    expect(alert).not.toContainElement(retry)
    request.mockReturnValue(new Promise(() => {}))
    fireEvent.click(retry)
    expect(screen.getByRole("status")).toHaveTextContent("Loading captured content...")
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
  })

  it.each([
    ["pending", "extraction_pending", "Text extraction has not completed yet."],
    ["missing_media", "media_not_found", "Media item is missing or unavailable."],
    ["failed", "extraction_failed", "Source extraction or indexing failed. Preview content is unavailable."],
    ["empty", "no_text", "No captured text is available for this source."]
  ] as const)("announces unavailable %s content", async (mode, reason, message) => {
    request.mockResolvedValue({ ...preview, content_available: false, text_preview: null, preview_mode: mode, unavailable_reason: reason })
    renderPreview()
    const loadingStatus = screen.getByRole("status")
    const status = await screen.findByText(message)
    expect(status).toBe(loadingStatus)
    expect(status).toHaveAttribute("role", "status")
    expect(status).toHaveAttribute("aria-live", "polite")
    expect(status).toHaveAttribute("aria-atomic", "true")
  })

  it("does not announce a late failure after the account boundary closes the preview", async () => {
    let reject!: (error: Error) => void
    request.mockReturnValue(new Promise((_, no) => { reject = no }))
    renderPreview()
    await waitFor(() => expect(request).toHaveBeenCalledTimes(1))
    act(() => window.dispatchEvent(new Event("tldw:auth-principal-changed")))
    await act(async () => reject(new Error("Private stale error")))
    expect(screen.queryByRole("status")).not.toBeInTheDocument()
    expect(screen.queryByRole("alert")).not.toBeInTheDocument()
    expect(screen.queryByText("Private stale error")).not.toBeInTheDocument()
  })
})
