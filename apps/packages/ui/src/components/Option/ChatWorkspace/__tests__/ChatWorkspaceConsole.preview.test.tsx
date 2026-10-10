import React from "react"
import { ConfigProvider } from "antd"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type { WorkspaceSource } from "@/types/workspace"
import type { ServicePromptScope } from "@/services/service-prompts"
import type { WorkspaceSourcePreviewResponse } from "@/services/tldw/domains/workspace-api"
import { useWorkspaceStore } from "@/store/workspace"
import { ChatWorkspaceConsole } from "../ChatWorkspaceConsole"

const { previewRequest, resolveScope } = vi.hoisted(() => ({
  previewRequest: vi.fn(),
  resolveScope: vi.fn()
}))

vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: { getWorkspaceSourcePreview: previewRequest }
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: resolveScope
}))
vi.mock("wxt/browser", () => ({ browser: {} }))
vi.mock("@/store/workspace", async () => {
  const { create } = await import("zustand")
  return {
    useWorkspaceStore: create<{
      workspaceId: string
      sources: WorkspaceSource[]
    }>(() => ({
      workspaceId: "",
      sources: []
    }))
  }
})
vi.mock("../WorkspaceChatPanel", () => ({ WorkspaceChatPanel: () => null }))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string, values?: Record<string, unknown>) =>
      (fallback || key).replace(/\{\{(\w+)\}\}/g, (_, name) =>
        String(values?.[name] ?? name)
      )
  })
}))

const owner: ServicePromptScope = {
  config: {
    serverUrl: "https://server.test",
    authMode: "multi-user",
    authSource: "manual"
  },
  scopeKey: "server-owner-7",
  userId: 7,
  clientPrincipalVerified: true
}
const source: WorkspaceSource = {
  id: "s1",
  mediaId: 1,
  title: "Captured report",
  type: "pdf",
  status: "ready",
  url: "https://source.test/report",
  addedAt: new Date("2026-09-29T00:00:00Z")
}
const preview = (
  changes: Partial<WorkspaceSourcePreviewResponse> = {}
): WorkspaceSourcePreviewResponse => ({
  workspace_id: "ws-a",
  source_id: "s1",
  media_id: 1,
  title: "Captured report",
  source_type: "pdf",
  url: source.url,
  state: "queryable",
  status_reason: "source_queryable",
  readiness: {
    metadata_ready: true,
    text_extracted: true,
    fts_ready: true,
    vector_ready: true,
    citation_ready: true,
    summary_ready: false,
    tool_accessible: true
  },
  content_available: true,
  preview_mode: "available",
  unavailable_reason: null,
  text_preview: "Captured <script>literal</script> report",
  text_total_chars: 2400,
  text_truncated: true,
  snippets: [
    {
      id: "chunk-1",
      source_id: "s1",
      media_id: 1,
      kind: "chunk",
      text: "Evidence from the captured report",
      start_char: 0,
      end_char: 33,
      chunk_index: 0,
      chunk_uuid: null,
      chunk_type: null
    }
  ],
  generated_at: "2026-09-29T00:00:00Z",
  ...changes
})
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (error: Error) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}

function Harness({
  workspaceId = "ws-a",
  currentSource = source
}: {
  workspaceId?: string
  currentSource?: WorkspaceSource | null
}) {
  const [browsedSourceId, setBrowsedSourceId] = React.useState<string | null>(
    null
  )
  const [staged, setStaged] = React.useState(false)
  return (
    <ConfigProvider theme={{ token: { motion: false } }}>
      <ChatWorkspaceConsole
        workspaceId={workspaceId}
        workspaceReady={true}
        workspaceName="Research"
        sources={currentSource ? [currentSource] : []}
        browsedSourceId={browsedSourceId}
        stagedSources={
          staged
            ? [
                {
                  sourceId: "s1",
                  mediaId: 1,
                  title: source.title,
                  type: "pdf",
                  scopeLabel: "Research",
                  availability: "ready"
                }
              ]
            : []
        }
        selectedModelLabel="Model"
        hasModelSelected={true}
        selectedPersonaLabel={null}
        assistantSource="none"
        backendAvailable={true}
        chatBackendAvailable={true}
        streaming={false}
        onBrowseSource={setBrowsedSourceId}
        onCloseBrowseSource={() => setBrowsedSourceId(null)}
        onStageSources={() => setStaged(true)}
        onUnstageSource={() => setStaged(false)}
        onClearStagedSources={() => setStaged(false)}
        onRuntimeStateChange={() => undefined}
      />
    </ConfigProvider>
  )
}
const browse = () =>
  fireEvent.click(
    screen.getByRole("button", { name: "Browse Captured report" })
  )

function StoreHarness() {
  const workspaceId = useWorkspaceStore((state) => state.workspaceId)
  const sources = useWorkspaceStore((state) => state.sources)
  return (
    <Harness workspaceId={workspaceId} currentSource={sources[0] ?? null} />
  )
}

describe("Chat Workspace canonical Browse preview", () => {
  beforeEach(() => {
    useWorkspaceStore.setState({ workspaceId: "ws-a", sources: [source] })
    previewRequest.mockReset().mockResolvedValue(preview())
    resolveScope.mockReset().mockResolvedValue(owner)
  })

  it.each([
    ["workspace", "scope"],
    ["workspace", "transport"],
    ["source", "scope"],
    ["source", "transport"],
    ["media-ID", "scope"],
    ["media-ID", "transport"]
  ] as const)(
    "fences batched Zustand %s ABA during %s without an intervening React commit",
    async (change, phase) => {
      const pendingScope = deferred<ServicePromptScope>()
      const pendingPreview = deferred<WorkspaceSourcePreviewResponse>()
      if (phase === "scope")
        resolveScope.mockReturnValueOnce(pendingScope.promise)
      else previewRequest.mockReturnValueOnce(pendingPreview.promise)
      render(<StoreHarness />)
      browse()
      await waitFor(() => expect(resolveScope).toHaveBeenCalledTimes(1))
      if (phase === "transport")
        await waitFor(() => expect(previewRequest).toHaveBeenCalledTimes(1))
      const signal = resolveScope.mock.calls[0][0].signal as AbortSignal
      const initial = useWorkspaceStore.getState()
      act(() => {
        if (change === "workspace") {
          useWorkspaceStore.setState({ workspaceId: "ws-b" })
          useWorkspaceStore.setState({ workspaceId: "ws-a" })
        } else if (change === "source") {
          useWorkspaceStore.setState({ sources: [] })
          useWorkspaceStore.setState({ sources: initial.sources })
        } else {
          useWorkspaceStore.setState({ sources: [{ ...source, mediaId: 2 }] })
          useWorkspaceStore.setState({ sources: initial.sources })
        }
        expect(signal.aborted).toBe(true)
      })
      await act(async () => {
        pendingScope.resolve(owner)
        pendingPreview.resolve(
          preview({ text_preview: "Batched stale content" })
        )
      })
      expect(previewRequest).toHaveBeenCalledTimes(phase === "scope" ? 0 : 1)
      expect(
        screen.queryByText("Batched stale content")
      ).not.toBeInTheDocument()
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      browse()
      expect(
        await screen.findByText("Captured <script>literal</script> report")
      ).toBeInTheDocument()
    }
  )

  it("opens captured content, chunks and truncation without staging, then closes and reopens", async () => {
    render(<Harness />)
    browse()
    const content = await screen.findByText(
      "Captured <script>literal</script> report"
    )
    expect(content.querySelector("script")).toBeNull()
    expect(
      screen.getByText("Evidence from the captured report")
    ).toBeInTheDocument()
    expect(
      screen.getByText("Showing first 40 of 2,400 characters.")
    ).toBeInTheDocument()
    expect(screen.getByRole("link", { name: source.url })).toHaveAttribute(
      "rel",
      "noreferrer"
    )
    expect(
      screen.getByRole("button", { name: "Stage Captured report for chat" })
    ).toBeEnabled()
    expect(previewRequest).toHaveBeenCalledWith(
      "ws-a",
      "s1",
      { max_chars: 3000, chunk_limit: 3 },
      { requestScope: owner, signal: expect.any(AbortSignal) }
    )
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await waitFor(() =>
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
    )
    browse()
    expect(
      await screen.findByText("Captured <script>literal</script> report")
    ).toBeInTheDocument()
  })

  it.each(["/tmp/captured-report.pdf", "captured-report.pdf", "//source.test/report", "mailto:source@test.example"])(
    "keeps non-HTTP provenance %s inert while retaining captured content",
    async (url) => {
      render(<Harness currentSource={{ ...source, url }} />)
      browse()
      expect(
        await screen.findByText("Captured <script>literal</script> report")
      ).toBeInTheDocument()
      expect(screen.queryByRole("link", { name: url })).not.toBeInTheDocument()
    }
  )

  it("shows failure details and retries only the preview", async () => {
    previewRequest.mockRejectedValueOnce(new Error("503 preview unavailable"))
    render(<Harness />)
    browse()
    expect(
      await screen.findByText("503 preview unavailable")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Retry preview" }))
    expect(
      await screen.findByText("Captured <script>literal</script> report")
    ).toBeInTheDocument()
    expect(
      screen.getByRole("button", { name: "Stage Captured report for chat" })
    ).toBeEnabled()
  })

  it.each([
    ["pending", "Text extraction has not completed yet."],
    ["missing_media", "Media item is missing or unavailable."],
    [
      "failed",
      "Source extraction or indexing failed. Preview content is unavailable."
    ],
    ["empty", "No captured text is available for this source."]
  ] as const)(
    "shows %s content state without disabling Browse",
    async (mode, text) => {
      previewRequest.mockResolvedValue(
        preview({
          preview_mode: mode,
          content_available: false,
          text_preview: null,
          snippets: []
        })
      )
      render(<Harness currentSource={{ ...source, status: "processing" }} />)
      browse()
      expect(await screen.findByText(text)).toBeInTheDocument()
      expect(
        screen.getByRole("button", { name: "Stage Captured report for chat" })
      ).toBeDisabled()
    }
  )

  it("never dispatches if closed while the captured owner is resolving", async () => {
    const scope = deferred<ServicePromptScope>()
    resolveScope.mockReturnValue(scope.promise)
    render(<Harness />)
    browse()
    expect(
      await screen.findByText("Loading captured content...")
    ).toBeInTheDocument()
    fireEvent.click(screen.getByRole("button", { name: "Close" }))
    await act(async () => {
      scope.resolve(owner)
    })
    expect(previewRequest).not.toHaveBeenCalled()
  })

  it.each(["workspace", "source"])(
    "discards late content after %s ABA and aborts old transport",
    async (change) => {
      const old = deferred<WorkspaceSourcePreviewResponse>()
      previewRequest.mockReturnValueOnce(old.promise)
      const view = render(<Harness />)
      browse()
      await waitFor(() => expect(previewRequest).toHaveBeenCalledTimes(1))
      const signal = previewRequest.mock.calls[0][3]?.signal as
        | AbortSignal
        | undefined
      if (change === "workspace") view.rerender(<Harness workspaceId="ws-b" />)
      else view.rerender(<Harness currentSource={{ ...source, id: "s2" }} />)
      view.rerender(<Harness />)
      browse()
      expect(
        await screen.findByText("Captured <script>literal</script> report")
      ).toBeInTheDocument()
      await act(async () => {
        old.resolve(preview({ text_preview: "Stale private content" }))
      })
      expect(
        screen.queryByText("Stale private content")
      ).not.toBeInTheDocument()
      expect(signal?.aborted).toBe(true)
    }
  )

  it.each(["tldw:auth-principal-changed", "tldw:config-updated"])(
    "closes and aborts on %s",
    async (eventName) => {
      const old = deferred<WorkspaceSourcePreviewResponse>()
      previewRequest.mockReturnValueOnce(old.promise)
      render(<Harness />)
      browse()
      await waitFor(() => expect(previewRequest).toHaveBeenCalledTimes(1))
      const signal = previewRequest.mock.calls[0][3]?.signal as
        | AbortSignal
        | undefined
      act(() =>
        window.dispatchEvent(
          new CustomEvent(eventName, { detail: { authorityChanged: true } })
        )
      )
      await act(async () => {
        old.resolve(preview({ text_preview: "Other owner content" }))
      })
      expect(signal?.aborted).toBe(true)
      expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
      expect(screen.queryByText("Other owner content")).not.toBeInTheDocument()
    }
  )

  it("rejects an owner change before scope resolution can dispatch", async () => {
    const scope = deferred<ServicePromptScope>()
    resolveScope.mockReturnValue(scope.promise)
    render(<Harness />)
    browse()
    await screen.findByText("Loading captured content...")
    act(() => window.dispatchEvent(new Event("tldw:auth-credentials-changed")))
    await act(async () => {
      scope.resolve(owner)
    })
    expect(previewRequest).not.toHaveBeenCalled()
    expect(screen.queryByRole("dialog")).not.toBeInTheDocument()
  })

  it.each([
    { workspace_id: "other-workspace" },
    { source_id: "other-source" },
    { snippets: [{ ...preview().snippets[0], source_id: "other-source" }] },
    {
      media_id: 2,
      snippets: [{ ...preview().snippets[0], media_id: 2 }]
    },
    { snippets: [{ ...preview().snippets[0], media_id: 2 }] }
  ])(
    "does not render content with mismatched response IDs %j",
    async (changes) => {
      previewRequest.mockResolvedValue(preview(changes))
      render(<Harness />)
      browse()
      expect(
        await screen.findByText("Source preview could not load.")
      ).toBeInTheDocument()
      expect(
        screen.getByText(
          "Source preview did not match the captured workspace source."
        )
      ).toBeInTheDocument()
      expect(
        screen.queryByText("Captured <script>literal</script> report")
      ).not.toBeInTheDocument()
      expect(
        screen.queryByText("Evidence from the captured report")
      ).not.toBeInTheDocument()
    }
  )

  it("does not expose unsafe links from source metadata", async () => {
    render(
      <Harness currentSource={{ ...source, url: "java\tscript:alert(1)" }} />
    )
    browse()
    await screen.findByText("Captured <script>literal</script> report")
    expect(
      screen.queryByRole("link", { name: /script:alert/ })
    ).not.toBeInTheDocument()
  })
})
