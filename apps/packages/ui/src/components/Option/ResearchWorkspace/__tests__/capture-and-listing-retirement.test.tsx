import { act, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { ConfigProvider, message } from "antd"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { useTranslation } from "react-i18next"
import { useWorkspaceStore } from "@/store/workspace"
import { serverWorkspaceMetadata } from "@/store/__tests__/workspace-activation.fixtures"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"
import { useArtifactExport } from "../StudioPane/hooks/useArtifactExport"
import type { GeneratedArtifact } from "@/types/workspace"

const boundary = vi.hoisted(() => ({
  request: vi.fn(), keywords: vi.fn(),
  changed: new Set<(invalidated: boolean) => void>()
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: boundary.request }))
vi.mock("@/services/note-keywords", () => ({ getNoteKeywords: boundary.keywords }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => ({ scopeKey: "owner-a" }),
  loadServicePromptSnapshot: () => { throw new Error("Unexpected save") }
}))
vi.mock("@/services/chat-account-boundary", () => ({
  watchChatAccountChanges: (callback: (invalidated: boolean) => void) => {
    boundary.changed.add(callback)
    return () => { boundary.changed.delete(callback) }
  }
}))
vi.mock("@/components/Common/MarkdownPreview", () => ({ MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div> }))
vi.mock("react-i18next", () => ({ useTranslation: () => ({ t: (_key: string, fallback: string) => fallback }) }))

const privateRows = [{ id: "alice-note", title: "Alice private title", content: "Alice private excerpt", keywords: ["private-tag"] }]
const artifact: GeneratedArtifact = {
  id: "artifact", type: "report", title: "Captured report", content: "Captured report body",
  status: "completed", createdAt: new Date("2026-10-08T00:00:00Z")
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(yes => { resolve = yes })
  return { promise, resolve }
}
const retire = () => {
  for (const changed of boundary.changed) changed(true)
}
const ArtifactCapture = () => {
  const [messageApi, context] = message.useMessage()
  const { t } = useTranslation()
  const actions = useArtifactExport({ messageApi, t, isMobile: false, generatedArtifacts: [artifact],
    removeArtifact: useWorkspaceStore.getState().removeArtifact,
    restoreArtifact: useWorkspaceStore.getState().restoreArtifact,
    captureToCurrentNote: useWorkspaceStore.getState().captureToCurrentNote })
  return <>{context}<button onClick={() => actions.handleSaveArtifactToNotes(artifact, "append")}>Append</button>
    <button onClick={() => actions.handleSaveArtifactToNotes(artifact, "replace")}>Replace</button></>
}

describe("mounted canonical capture and fetched-listing retirement", () => {
  beforeEach(() => {
    vi.clearAllMocks()
    boundary.changed.clear()
    localStorage.clear()
    useWorkspaceStore.getState().reset()
    useWorkspaceStore.getState().initializeWorkspace("Local draft")
    useWorkspaceStore.getState().setCurrentNote({ title: "Local title", content: "Unsent local body", keywords: ["local-tag"], isDirty: true })
    boundary.request.mockResolvedValue(privateRows)
    boundary.keywords.mockResolvedValue(["private-keyword"])
  })
  afterEach(() => { vi.useRealTimers() })

  it.each(["Append", "Replace"])("visibly refuses canonical artifact %s without changing the draft", async action => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><ArtifactCapture /></ConfigProvider>)
    // The event handler was created while local; it must inspect the destination at click time.
    act(() => { useWorkspaceStore.setState({ serverWorkspace: {
      scopeKey: "owner-a", sourceSignature: "", selectedSourceSignature: "", metadata: serverWorkspaceMetadata, notes: []
    } }) })
    const before = useWorkspaceStore.getState().currentNote
    fireEvent.click(screen.getByRole("button", { name: action }))
    expect(await screen.findByText("Server notes are view-only")).toBeVisible()
    expect(useWorkspaceStore.getState().currentNote).toBe(before)
    expect(screen.queryByText(/Output (added|replaced)/)).toBeNull()
    expect(boundary.request).not.toHaveBeenCalled()
  })

  it("keeps local artifact capture available", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><ArtifactCapture /></ConfigProvider>)
    fireEvent.click(screen.getByRole("button", { name: "Append" }))
    expect(await screen.findByText("Output added to your current note draft.")).toBeVisible()
    expect(useWorkspaceStore.getState().currentNote.content).toContain("Unsent local body")
    expect(useWorkspaceStore.getState().currentNote.content).toContain("Captured report body")
  })

  it.each(["account", "server"])("retires fetched rows and modal on %s retirement but retains the local draft", async boundaryKind => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><QuickNotesSection /></ConfigProvider>)
    await screen.findByRole("button", { name: "Alice private title" })
    fireEvent.click(screen.getByRole("button", { name: "Load note" }))
    const dialog = await screen.findByRole("dialog")
    await waitFor(() => expect(within(dialog).getByText("Alice private excerpt")).toBeVisible())
    expect(within(dialog).getByText("private-tag")).toBeVisible()
    const before = useWorkspaceStore.getState().currentNote
    act(() => retire())
    expect(screen.queryByRole("dialog")).toBeNull()
    expect(screen.queryByText("Alice private title")).toBeNull()
    expect(screen.queryByText("Alice private excerpt")).toBeNull()
    expect(screen.queryByText("private-tag")).toBeNull()
    expect(useWorkspaceStore.getState().currentNote).toBe(before)
    expect(screen.getByRole("textbox", { name: "Note content" })).toHaveValue("Unsent local body")
    boundary.request.mockResolvedValue([{ id: `${boundaryKind}-bob`, title: "Bob title", content: "Bob excerpt" }])
    fireEvent.click(screen.getByRole("button", { name: "Load note" }))
    await waitFor(() => expect(screen.getByText("Bob excerpt")).toBeVisible())
    expect(screen.queryByText("Alice private excerpt")).toBeNull()
  })

  it("retires fetched keyword suggestions without erasing the typed local keywords", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><QuickNotesSection /></ConfigProvider>)
    await waitFor(() => expect(boundary.keywords).toHaveBeenCalledOnce())
    fireEvent.change(screen.getByRole("combobox", { name: "Note keywords" }), { target: { value: "private" } })
    expect(await screen.findByRole("option", { name: "private-keyword" })).toBeInTheDocument()
    await waitFor(() => expect(screen.getByText("private-keyword", { selector: ".ant-select-item-option-content" })).toBeVisible())
    const before = useWorkspaceStore.getState().currentNote
    act(() => retire())
    expect(screen.queryByRole("option", { name: "private-keyword" })).toBeNull()
    expect(useWorkspaceStore.getState().currentNote).toBe(before)
    expect(screen.getByRole("combobox", { name: "Note keywords" })).toHaveValue("private")
  })

  it.each(["search", "pinned", "keywords"])("rejects the late %s reply after retirement", async operation => {
    const gate = deferred<unknown>()
    boundary.request.mockImplementation(({ path }: { path: string }) => {
      if (operation === "pinned" || (operation === "search" && path.startsWith("/api/v1/notes/?"))) return gate.promise
      return []
    })
    boundary.keywords.mockImplementation(() => operation === "keywords" ? gate.promise : [])
    render(<ConfigProvider theme={{ token: { motion: false } }}><QuickNotesSection /></ConfigProvider>)
    if (operation === "search") fireEvent.click(screen.getByRole("button", { name: "Load note" }))
    await waitFor(() => expect(operation === "keywords" ? boundary.keywords : boundary.request).toHaveBeenCalled())
    const before = useWorkspaceStore.getState().currentNote
    act(() => retire())
    await act(async () => { gate.resolve(operation === "keywords" ? ["private-keyword"] : privateRows); await gate.promise })
    expect(screen.queryByText("Alice private title")).toBeNull()
    expect(screen.queryByRole("dialog")).toBeNull()
    fireEvent.change(screen.getByRole("combobox", { name: "Note keywords" }), { target: { value: "private" } })
    expect(screen.queryByRole("option", { name: "private-keyword" })).toBeNull()
    expect(useWorkspaceStore.getState().currentNote.content).toBe(before.content)
  })

  it("cancels a queued old-owner search on retirement", async () => {
    vi.useFakeTimers()
    render(<ConfigProvider theme={{ token: { motion: false } }}><QuickNotesSection /></ConfigProvider>)
    await act(async () => { await Promise.resolve() })
    fireEvent.click(screen.getByRole("button", { name: "Load note" }))
    await act(async () => { await Promise.resolve() })
    fireEvent.change(screen.getByRole("textbox", { name: "Search notes" }), { target: { value: "Old owner query" } })
    const calls = boundary.request.mock.calls.length
    act(() => retire())
    await act(async () => { vi.advanceTimersByTime(300); await Promise.resolve() })
    expect(boundary.request).toHaveBeenCalledTimes(calls)
    expect(screen.queryByRole("dialog")).toBeNull()
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Unsent local body")
  })

  it("retains fetched rows across a non-authority config update", async () => {
    render(<ConfigProvider theme={{ token: { motion: false } }}><QuickNotesSection /></ConfigProvider>)
    await screen.findByRole("button", { name: "Alice private title" })
    act(() => { for (const changed of boundary.changed) changed(false) })
    expect(screen.getByRole("button", { name: "Alice private title" })).toBeVisible()
  })
})
