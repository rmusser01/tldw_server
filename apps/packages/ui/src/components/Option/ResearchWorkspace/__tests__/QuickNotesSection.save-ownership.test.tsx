import type { BgRequestInit } from "@/services/background-proxy"
import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"
import { useWorkspaceStore } from "@/store/workspace"
import type { WorkspaceNote } from "@/types/workspace"
import { readKnowledgeNoteProvenance, type KnowledgeNoteSource } from "@/utils/knowledge-note-provenance"

const mocks = vi.hoisted(() => ({
  serverUrl: "https://research-a.example",
  userId: "alice",
  request: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
  open: vi.fn()
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: string) => fallback || key
  })
}))
vi.mock("@/services/background-proxy", () => ({
  bgRequest: (...args: unknown[]) => mocks.request(...args)
}))
vi.mock("@/services/note-keywords", () => ({ getNoteKeywords: async () => [] }))
vi.mock("@/services/tldw/TldwApiClient", () => ({
  tldwClient: {
    initialize: async () => {},
    ensureConfigForRequest: async () => ({
      serverUrl: mocks.serverUrl,
      authMode: "multi-user",
      authSource: "cookie-session"
    })
  }
}))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: async () => ({ id: mocks.userId }) }
}))
vi.mock("@/services/tldw/deployment-mode", () => ({
  isHostedTldwDeployment: () => false
}))
vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: () => null
}))
vi.mock("antd", async () => ({
  ...(await vi.importActual<typeof import("antd")>("antd")),
  message: {
    useMessage: () => [
      {
        success: mocks.success,
        error: mocks.error,
        open: mocks.open,
        destroy: vi.fn(),
        warning: vi.fn()
      },
      null
    ]
  }
}))

const noteId = "9f0a165c-6f62-4b76-8cfb-e5d1a719f13b"
const otherId = "cb438f03-d609-4b4b-9f64-32bc769a55cb"
const draft = (id: string | undefined = noteId): WorkspaceNote => ({
  id,
  title: "Alice draft",
  content: "Original draft body",
  keywords: ["evidence"],
  isDirty: true
})
const deferred = () => {
  let resolve!: () => void
  const promise = new Promise<void>((done) => {
    resolve = done
  })
  return { promise, resolve }
}
const writes = () =>
  mocks.request.mock.calls.filter(([request]) =>
    ["PUT", "POST"].includes(request.method)
  )
const beginSave = async (boundary: "GET" | "PUT" | "POST") => {
  const gate = deferred()
  const initial = draft()
  if (boundary === "POST") initial.id = undefined
  if (boundary === "PUT") initial.version = 3
  useWorkspaceStore.setState({ currentNote: initial })
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return []
    if (request.method === "GET") {
      if (boundary === "GET") await gate.promise
      return { ...initial, version: 3 }
    }
    if (request.method === boundary) await gate.promise
    return { ...request.body, id: noteId, version: 4 }
  })
  const view = render(<QuickNotesSection />)
  fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }))
  await waitFor(() =>
    expect(
      mocks.request.mock.calls.some(
        ([request]) =>
          request.method === boundary && !request.path.includes("/search/")
      )
    ).toBe(true)
  )
  return { gate, view }
}
const finishSave = async (gate: ReturnType<typeof deferred>) => {
  await act(async () => {
    gate.resolve()
    await gate.promise
  })
  await waitFor(() =>
    expect(document.querySelector("button.ant-btn-loading")).toBeNull()
  )
}
function changeContext(change: string) {
  const state = useWorkspaceStore.getState()
  if (change.startsWith("workspace")) {
    useWorkspaceStore.setState({
      workspaceId: "workspace-b",
      workspaceTag: "workspace:b"
    })
    if (change === "workspace-return")
      useWorkspaceStore.setState({
        workspaceId: "workspace-a",
        workspaceTag: "workspace:a"
      })
  } else if (change.startsWith("note")) {
    state.loadNote({ ...draft(otherId), id: otherId })
    if (change === "note-return") state.setCurrentNote(draft())
  } else if (change.startsWith("clear")) {
    state.clearCurrentNote()
    if (change === "clear-then-type")
      state.updateNoteContent("A different unsaved note")
  } else if (change === "account") {
    mocks.userId = "bob"
    window.dispatchEvent(new Event("tldw:auth-principal-changed"))
  } else {
    mocks.serverUrl = "https://research-b.example"
    window.dispatchEvent(
      new CustomEvent("tldw:config-updated", {
        detail: { authorityChanged: true }
      })
    )
  }
}
beforeEach(() => {
  vi.clearAllMocks()
  mocks.serverUrl = "https://research-a.example"
  mocks.userId = "alice"
  useWorkspaceStore.setState({
    workspaceId: "workspace-a",
    workspaceTag: "workspace:a",
    currentNote: draft()
  })
})
describe("Quick Notes save ownership across requests and acknowledgment", () => {
  it.each([
    "server",
    "account",
    "workspace",
    "note",
    "clear",
    "workspace-return",
    "note-return"
  ] as const)(
    "does not write after %s changes during version GET",
    async (change) => {
      const { gate } = await beginSave("GET")
      act(() => changeContext(change))
      const replacement = useWorkspaceStore.getState().currentNote
      await finishSave(gate)
      expect(writes()).toEqual([])
      expect(useWorkspaceStore.getState().currentNote).toBe(replacement)
      expect(mocks.success).not.toHaveBeenCalled()
    }
  )
  it.each(["GET", "PUT", "POST"] as const)(
    "pins the entire %s operation to the captured account and server",
    async (boundary) => {
      const { gate } = await beginSave(boundary)
      await finishSave(gate)
      const request = writes()[0][0]
      expect(request.servicePromptConfig).toMatchObject({
        serverUrl: "https://research-a.example",
        expectedUserId: "alice"
      })
      expect(request.headers).toMatchObject({
        "X-TLDW-Expected-User-ID": "alice",
        "Content-Type": "application/json"
      })
      if (boundary !== "POST")
        expect(request.headers["expected-version"]).toBe("3")
      expect(request.abortSignal).toBeInstanceOf(AbortSignal)
      if (boundary === "GET") {
        const read = mocks.request.mock.calls.find(
          ([request]) =>
            request.method === "GET" &&
            request.path === `/api/v1/notes/${noteId}`
        )![0]
        expect(read.servicePromptConfig).toEqual(request.servicePromptConfig)
        expect(read.abortSignal).toBe(request.abortSignal)
      }
      expect(useWorkspaceStore.getState().currentNote).toMatchObject({
        id: noteId,
        version: 4,
        isDirty: false
      })
    }
  )
  for (const boundary of ["PUT", "POST"] as const) {
    it.each([
      "workspace",
      "note",
      "clear",
      "account",
      "server",
      "clear-then-type"
    ] as const)(
      `retains %s instead of a late ${boundary} acknowledgment`,
      async (change) => {
        const { gate } = await beginSave(boundary)
        act(() => changeContext(change))
        const replacement = useWorkspaceStore.getState().currentNote
        await finishSave(gate)
        expect(useWorkspaceStore.getState().currentNote).toBe(replacement)
        expect(mocks.success).not.toHaveBeenCalled()
      }
    )
  }
  it.each(["GET", "PUT", "POST"] as const)(
    "retains later human edits during %s and uses acknowledged identity/version on the next save",
    async (boundary) => {
      const { gate } = await beginSave(boundary)
      act(() => {
        useWorkspaceStore.getState().updateNoteTitle("Newer title")
        useWorkspaceStore.getState().updateNoteContent("Newer body")
        useWorkspaceStore.getState().updateNoteKeywords(["newer"])
      })
      await finishSave(gate)
      expect(useWorkspaceStore.getState().currentNote).toEqual({
        id: noteId,
        version: 4,
        title: "Newer title",
        content: "Newer body",
        keywords: ["newer"],
        isDirty: true
      })
      expect(screen.queryByText(/^Saved$/)).not.toBeInTheDocument()
      fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }))
      await waitFor(() => expect(writes()).toHaveLength(2))
      expect(writes()[1][0]).toMatchObject({
        path: `/api/v1/notes/${noteId}`,
        method: "PUT",
        headers: { "expected-version": "4" },
        body: {
          title: "Newer title",
          content: "Newer body",
          keywords: ["newer", "workspace:a"]
        }
      })
      await waitFor(() =>
        expect(useWorkspaceStore.getState().currentNote.isDirty).toBe(false)
      )
    }
  )
  it("does not acknowledge after unmount", async () => {
    const { gate, view } = await beginSave("PUT")
    const original = useWorkspaceStore.getState().currentNote
    view.unmount()
    await act(async () => {
      gate.resolve()
      await gate.promise
    })
    expect(useWorkspaceStore.getState().currentNote).toBe(original)
    expect(mocks.success).not.toHaveBeenCalled()
  })
})

it("backfills only the exact absent child and retries a lost acknowledgment unchanged", async () => {
  const history = { origin: "knowledge_qa", question: "Original question" }
  const marker = `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`
  useWorkspaceStore.setState({
    currentNote: {
      ...draft(),
      content: `Body\n\n${marker}`,
      version: 3,
      knowledge_provenance_state: "absent",
      knowledge_provenance_version: 0
    } as WorkspaceNote
  })
  let first = true
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return []
    if (request.method === "GET") return { ...draft(), version: 4 }
    if (first) {
      first = false
      throw new Error("Lost acknowledgment")
    }
    return {
      ...request.body,
      id: noteId,
      version: 4,
      knowledge_provenance_state: "active",
      knowledge_provenance_version: 1,
      knowledge_provenance_hash: `sha256:${"a".repeat(64)}`,
      knowledge_provenance: history
    }
  })
  render(<QuickNotesSection />)
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.error).toHaveBeenCalled())
  expect(writes()[0][0].body).toMatchObject({
    knowledge_provenance: history,
    expected_provenance_version: 0
  })
  act(() => {
    useWorkspaceStore.getState().updateNoteContent("Newer unsaved draft")
  })
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.success).toHaveBeenCalled())
  expect(writes()[1][0].body).toEqual(writes()[0][0].body)
  expect(writes()[0][0].headers["Idempotency-Key"]).toBeTruthy()
  expect(writes()[1][0].headers).toEqual(writes()[0][0].headers)
  expect(useWorkspaceStore.getState().currentNote).toMatchObject({
    content: "Newer unsaved draft",
    version: 4,
    isDirty: true,
    knowledge_provenance_version: 1
  })
})

it.each(["active", "deleted"] as const)(
  "displays only the %s retained source history",
  async (state) => {
    const history = {
      origin: "knowledge_qa" as const,
      question: "Original evidence question",
      sources: [
        {
          originalId: "foreign-note",
          mediaId: null,
          title: "Original source",
          type: "text" as const,
          sourceType: "notes",
          excerpt: "Retained excerpt",
          url: "javascript:alert(1)"
        }
      ]
    }
    useWorkspaceStore.setState({
      currentNote: {
        ...draft(),
        content: "Edited answer",
        knowledge_provenance_state: state,
        knowledge_provenance: state === "active" ? history : null,
        knowledge_provenance_version: 3
      }
    })
    mocks.request.mockResolvedValue([])
    render(<QuickNotesSection />)
    if (state === "active") {
      expect(screen.getByText("Original evidence question")).toBeInTheDocument()
      expect(screen.getByText("Retained excerpt")).toBeInTheDocument()
      expect(
        screen.queryByRole("link", { name: "Original source" })
      ).not.toBeInTheDocument()
    } else
      expect(screen.queryByText("Retained excerpt")).not.toBeInTheDocument()
    expect(
      mocks.request.mock.calls.every(
        ([request]) => !request.path.includes("foreign-note")
      )
    ).toBe(true)
  }
)

it("releases a definitively rejected request so corrected Quick Notes input gets a new identity", async () => {
  useWorkspaceStore.setState({
    currentNote: { ...draft(), version: 3, content: "Invalid body" }
  })
  mocks.request.mockImplementation(async (request) => {
    if (request.path.includes("/search/")) return []
    if (request.body?.content === "Invalid body")
      throw Object.assign(new Error("validation rejected"), { status: 422 })
    return { ...request.body, id: noteId, version: 4 }
  })
  render(<QuickNotesSection />)
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.error).toHaveBeenCalled())
  act(() => {
    useWorkspaceStore.getState().updateNoteContent("Corrected body")
  })
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.success).toHaveBeenCalled())
  expect(writes()[1][0].body.content).toBe("Corrected body")
  expect(writes()[1][0].headers["Idempotency-Key"]).not.toBe(
    writes()[0][0].headers["Idempotency-Key"]
  )
})

it.each([
  {
    status: 409,
    detail: { error_code: "notes_provenance_encryption_unsupported" },
    message: "Policy unavailable"
  },
  {
    status: 429,
    detail: "Rate limit exceeded for notes.create",
    message: "Rate limit exceeded for notes.create"
  },
  {
    status: 409,
    detail: { error_code: "notes_organization_sync_not_ready" },
    message: "Notes organization Sync is not ready for writes."
  }
])(
  "preserves a lost-ack create through normalized $status $message",
  async ({ status, detail, message }) => {
    const history = { origin: "knowledge_qa", question: "Original question" }
    const marker = `<!-- tldw-knowledge:v1:${encodeURIComponent(JSON.stringify(history))} -->`
    useWorkspaceStore.setState({
      currentNote: {
        ...draft(),
        id: undefined,
        content: `Original answer\n\n${marker}`
      }
    })
    mocks.request.mockImplementation(async (request) => {
      if (request.path.includes("/search/")) return []
      const attempts = writes().length
      if (attempts === 1) throw new Error("Lost acknowledgment")
      if (attempts === 2)
        throw Object.assign(new Error(message), { status, details: { detail } })
      return { ...request.body, id: noteId, version: 1 }
    })
    render(<QuickNotesSection />)
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1))
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(writes()).toHaveLength(2))
    await waitFor(() =>
      expect(screen.getByRole("button", { name: "Save" })).not.toBeDisabled()
    )
    act(() => {
      useWorkspaceStore.getState().updateNoteContent("Later draft")
    })
    fireEvent.click(screen.getByRole("button", { name: "Save" }))
    await waitFor(() => expect(mocks.success).toHaveBeenCalled())
    expect(writes()[2][0].headers).toEqual(writes()[0][0].headers)
    expect(writes()[2][0].body).toEqual(writes()[0][0].body)
    expect(writes()[0][0].body.knowledge_provenance).toEqual(history)
    expect(useWorkspaceStore.getState().currentNote.content).toBe("Later draft")
  }
)
it.each(["unchanged", "body", "new-history", "failed", "stale"])(
  "explicit captured provenance replacement preserves independent versions and %s drafts",
  async (change) => {
    const old = {
      origin: "knowledge_qa" as const,
      question: "original",
      sources: []
    }
    const replacement = {
      ...old,
      sources: [
        {
          originalId: "clip",
          excerpt: "",
          mediaId: 71,
          title: "Article",
          type: "website" as const,
          sourceType: "server_article",
          url: "https://example.org",
          snapshotMediaId: 71,
          originalVersion: 9
        }
      ]
    }
    const gate = deferred()
    useWorkspaceStore.setState({
      currentNote: {
        ...draft(),
        version: 40,
        knowledge_provenance_state: "active",
        knowledge_provenance_version: 2,
        knowledge_provenance: old,
        pendingKnowledgeProvenance: replacement
      } as WorkspaceNote
    })
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return []
      if (request.method === "GET")
        return {
          ...draft(),
          version: 41,
          knowledge_provenance_state: "active",
          knowledge_provenance_version: 3,
          knowledge_provenance: old
        }
      await gate.promise
      if (change === "failed") throw Error("offline")
      if (change === "stale") throw { status: 409, message: "version conflict" }
      return {
        ...request.body,
        id: noteId,
        version: 41,
        knowledge_provenance_state: "active",
        knowledge_provenance_version: 3,
        knowledge_provenance: replacement
      }
    })
    render(<QuickNotesSection />)
    fireEvent.click(screen.getByRole("button", { name: "Update" }))
    await waitFor(() => expect(writes()).toHaveLength(1))
    expect(writes()[0][0]).toMatchObject({
      headers: { "expected-version": "40" },
      body: {
        expected_provenance_version: 2,
        knowledge_provenance: replacement
      }
    })
    if (change === "body")
      act(() => {
        useWorkspaceStore.getState().updateNoteContent("later human edit")
      })
    if (change === "new-history")
      act(() => {
        useWorkspaceStore.setState({
          currentNote: {
            ...useWorkspaceStore.getState().currentNote,
            pendingKnowledgeProvenance: {
              ...replacement,
              question: "later citation"
            }
          }
        })
      })
    await finishSave(gate)
    const current = useWorkspaceStore.getState().currentNote
    if (change === "failed" || change === "stale")
      expect(current.pendingKnowledgeProvenance).toEqual(replacement)
    else if (change === "new-history")
      expect(current.pendingKnowledgeProvenance?.question).toBe(
        "later citation"
      )
    else expect(current.pendingKnowledgeProvenance).toBeUndefined()
    if (change === "body") {
      expect(current.content).toBe("later human edit")
      expect(current.isDirty).toBe(true)
    }
  }
)
it("version read discovering removed provenance cannot resend a pending replacement", async () => {
  const history = { origin: "reviewed_sources" as const, sources: [] }
  useWorkspaceStore.setState({
    currentNote: {
      ...draft(),
      version: undefined,
      knowledge_provenance_state: "active",
      knowledge_provenance_version: 2,
      knowledge_provenance: history,
      pendingKnowledgeProvenance: { ...history, question: "capture" }
    }
  })
  mocks.request.mockImplementation(async (request: BgRequestInit) =>
    request.path.includes("/search/")
      ? []
      : request.method === "GET"
        ? {
            ...draft(),
            version: 41,
            knowledge_provenance_state: "deleted",
            knowledge_provenance_version: 3
          }
        : { ...request.body, id: noteId, version: 42 }
  )
  render(<QuickNotesSection />)
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(writes()).toHaveLength(1))
  expect(writes()[0][0].body.knowledge_provenance).toBeUndefined()
  expect(writes()[0][0].body.expected_provenance_version).toBeUndefined()
})
it("retains explicit capture history when the acknowledgment does not confirm it", async () => {
  const replacement = {
    origin: "reviewed_sources" as const,
    question: "capture"
  }
  useWorkspaceStore.setState({
    currentNote: {
      ...draft(),
      version: 4,
      knowledge_provenance_state: "active",
      knowledge_provenance_version: 2,
      knowledge_provenance: { origin: "reviewed_sources" },
      pendingKnowledgeProvenance: replacement
    }
  })
  mocks.request.mockImplementation(async (request: BgRequestInit) =>
    request.path.includes("/search/")
      ? []
      : {
          ...request.body,
          id: noteId,
          version: 5,
          knowledge_provenance_state: "active",
          knowledge_provenance_version: 2,
          knowledge_provenance: { origin: "reviewed_sources" }
        }
  )
  render(<QuickNotesSection />)
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(writes()).toHaveLength(1))
  await waitFor(() =>
    expect(document.querySelector("button.ant-btn-loading")).toBeNull()
  )
  expect(
    useWorkspaceStore.getState().currentNote.pendingKnowledgeProvenance
  ).toEqual(replacement)
  expect(mocks.success).not.toHaveBeenCalled()
})

const sourceHistory = (id: string): KnowledgeNoteSource => ({
  originalId: id, excerpt: `Excerpt ${id}`, mediaId: 71,
  title: id, type: "website", sourceType: "web_capture",
  url: "https://article.example/story", snapshotMediaId: 71, originalVersion: 9
})
const originalHistory = {
  origin: "knowledge_qa" as const, question: "Original question",
  trust_state: "uncited_degraded_answer", trust_reason_codes: ["missing_citations"],
  sources: [sourceHistory("A"), sourceHistory("removed")]
}
const pendingHistory = { ...originalHistory, sources: [...originalHistory.sources, sourceHistory("C")] }
const remoteHistory = {
  ...originalHistory, question: "Remote question", trust_reason_codes: ["remote_review"],
  sources: [sourceHistory("A"), sourceHistory("B")]
}
const reconciledHistory = { ...remoteHistory, sources: [...remoteHistory.sources, sourceHistory("C")] }
const prepareCaptureConflict = async () => {
  useWorkspaceStore.setState({ currentNote: {
    ...draft(), content: "Unsaved captured message", version: 40,
    knowledge_provenance_state: "active", knowledge_provenance_version: 2,
    knowledge_provenance: originalHistory, pendingKnowledgeProvenance: pendingHistory
  } })
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return []
    if (request.method === "GET") return {
      ...draft(), title: "Remote title", content: "Remote body", version: 41,
      knowledge_provenance_state: "active", knowledge_provenance_version: 3,
      knowledge_provenance: remoteHistory
    }
    if (writes().length === 1) throw { status: 409, message: "version conflict" }
    return { ...request.body, id: noteId, version: 42,
      knowledge_provenance_state: "active", knowledge_provenance_version: 4,
      knowledge_provenance: request.body.knowledge_provenance || remoteHistory }
  })
  render(<QuickNotesSection />)
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.open).toHaveBeenCalled())
  await waitFor(() => expect(screen.getByRole("button", { name: "Update" })).not.toBeDisabled())
}
const reloadCaptureConflict = () => {
  render(mocks.open.mock.calls.at(-1)![0].content)
  fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
}
it("capture conflict retry reconciles whole canonical history without resurrecting removed references", async () => {
  await prepareCaptureConflict()
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.success).toHaveBeenCalled())
  const retry = writes()[1][0]
  expect(retry.body.knowledge_provenance).toEqual(reconciledHistory)
  expect(retry.body.expected_provenance_version).toBe(3)
  expect(retry.headers["expected-version"]).toBe("41")
  expect(retry.headers["Idempotency-Key"]).not.toBe(writes()[0][0].headers["Idempotency-Key"])
  expect(readKnowledgeNoteProvenance(retry.body.content)).toEqual(reconciledHistory)
})
it("reload latest retains unsaved capture provenance through the next explicit save", async () => {
  await prepareCaptureConflict()
  reloadCaptureConflict()
  await waitFor(() => expect(mocks.success).toHaveBeenCalled())
  const current = useWorkspaceStore.getState().currentNote
  expect(current.content).toContain("Remote body")
  expect(current.content).toContain("Unsaved captured message")
  expect(current.pendingKnowledgeProvenance).toEqual(reconciledHistory)
  expect(current.isDirty).toBe(true)
  const reload = mocks.request.mock.calls.filter(([request]) => request.method === "GET" && request.path === `/api/v1/notes/${noteId}`).at(-1)![0]
  expect(reload.servicePromptConfig).toMatchObject({ serverUrl: "https://research-a.example", expectedUserId: "alice" })
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(writes()).toHaveLength(2))
  expect(writes()[1][0].body.knowledge_provenance).toEqual(reconciledHistory)
  await waitFor(() => expect(useWorkspaceStore.getState().currentNote.pendingKnowledgeProvenance).toBeUndefined())
})
it.each(["workspace", "workspace-return", "note", "note-return", "clear", "account", "server", "edit"])(
  "reload latest cannot replace a retired %s capture draft",
  async (change) => {
    await prepareCaptureConflict()
    const gate = deferred()
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return []
      await gate.promise
      return { ...draft(), content: "Late remote body", version: 42,
        knowledge_provenance_state: "active", knowledge_provenance_version: 3,
        knowledge_provenance: remoteHistory }
    })
    const callsBefore = mocks.request.mock.calls.length
    reloadCaptureConflict()
    await waitFor(() => expect(mocks.request.mock.calls.length).toBeGreaterThan(callsBefore))
    await act(async () => {
      if (change === "edit")
        useWorkspaceStore.getState().updateNoteContent("Newer local capture draft")
      else changeContext(change)
    })
    const replacement = useWorkspaceStore.getState().currentNote
    await act(async () => { gate.resolve(); await gate.promise })
    expect(useWorkspaceStore.getState().currentNote).toBe(replacement)
    expect(mocks.success).not.toHaveBeenCalled()
  }
)

it.each([
  ["GET", "body"], ["PUT", "body"],
  ["GET", "capture-D"], ["PUT", "capture-D"]
] as const)(
  "rebased save acknowledgment retains remote history through a %s %s edit and second save",
  async (boundary, change) => {
    const localHistory = { ...originalHistory, sources: [sourceHistory("A")] }
    const localPending = { ...localHistory, sources: [...localHistory.sources, sourceHistory("C")] }
    const freshHistory = { ...remoteHistory, sources: [sourceHistory("B")] }
    const acceptedHistory = { ...freshHistory, sources: [...freshHistory.sources, sourceHistory("C")] }
    const nextHistory = change === "body" ? acceptedHistory
      : { ...acceptedHistory, sources: [...acceptedHistory.sources, sourceHistory("D")] }
    const getGate = deferred()
    const putGate = deferred()
    if (boundary === "PUT") getGate.resolve()
    useWorkspaceStore.setState({ currentNote: {
      ...draft(), version: undefined,
      knowledge_provenance_state: "active", knowledge_provenance_version: 2,
      knowledge_provenance: localHistory, pendingKnowledgeProvenance: localPending
    } })
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return []
      if (request.method === "GET") {
        await getGate.promise
        return { ...draft(), version: 41, knowledge_provenance_state: "active",
          knowledge_provenance_version: 3, knowledge_provenance: freshHistory }
      }
      const first = writes().length === 1
      if (first) await putGate.promise
      return { ...request.body, id: noteId, version: first ? 42 : 43,
        knowledge_provenance_state: "active", knowledge_provenance_version: first ? 4 : 5,
        knowledge_provenance: request.body.knowledge_provenance || acceptedHistory }
    })
    render(<QuickNotesSection />)
    fireEvent.click(screen.getByRole("button", { name: "Update" }))
    await waitFor(() => expect(mocks.request.mock.calls.some(([request]) =>
      request.method === boundary && request.path === `/api/v1/notes/${noteId}`
    )).toBe(true))
    await act(async () => {
      const state = useWorkspaceStore.getState()
      if (change === "body") state.updateNoteContent("Later human body")
      else state.captureToCurrentNote({
        content: "New capture D", title: "New capture D",
        provenance: { ...localPending, sources: [...localPending.sources, sourceHistory("D")] }
      })
    })
    const newerBody = useWorkspaceStore.getState().currentNote.content
    await act(async () => { getGate.resolve(); await getGate.promise })
    await waitFor(() => expect(writes()).toHaveLength(1))
    const firstBody = structuredClone(writes()[0][0].body)
    expect(firstBody.knowledge_provenance).toEqual(acceptedHistory)
    expect(firstBody.expected_provenance_version).toBe(3)
    expect(writes()[0][0].headers["expected-version"]).toBe("41")
    await act(async () => { putGate.resolve(); await putGate.promise })
    await waitFor(() => expect(mocks.success).toHaveBeenCalledTimes(1))
    const afterAck = structuredClone(useWorkspaceStore.getState().currentNote)
    expect(afterAck.content).toBe(newerBody)
    expect(afterAck.knowledge_provenance).toEqual(acceptedHistory)
    expect(afterAck.knowledge_provenance_version).toBe(4)
    expect(afterAck.version).toBe(42)
    expect(afterAck.isDirty).toBe(true)
    expect(writes()[0][0].body).toEqual(firstBody)
    await waitFor(() => expect(screen.getByRole("button", { name: "Update" })).not.toBeDisabled())
    fireEvent.click(screen.getByRole("button", { name: "Update" }))
    await waitFor(() => expect(mocks.success).toHaveBeenCalledTimes(2))
    const second = writes()[1][0]
    if (change === "body") expect(second.body.knowledge_provenance).toBeUndefined()
    else expect(second.body.knowledge_provenance).toEqual(nextHistory)
    expect(readKnowledgeNoteProvenance(second.body.content)).toEqual(nextHistory)
    expect(second.headers["expected-version"]).toBe("42")
    expect(second.headers["Idempotency-Key"]).not.toBe(writes()[0][0].headers["Idempotency-Key"])
    expect(afterAck.pendingKnowledgeProvenance).toEqual(change === "body" ? undefined : nextHistory)
    expect(useWorkspaceStore.getState().currentNote.knowledge_provenance).toEqual(nextHistory)
    expect(useWorkspaceStore.getState().currentNote.pendingKnowledgeProvenance).toBeUndefined()
    expect(useWorkspaceStore.getState().currentNote.knowledge_provenance?.sources?.map(source => source.originalId))
      .toEqual(change === "body" ? ["B", "C"] : ["B", "C", "D"])
  }
)
