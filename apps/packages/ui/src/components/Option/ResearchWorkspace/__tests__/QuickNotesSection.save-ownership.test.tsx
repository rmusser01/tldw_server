import {
  readSurfaceOfflineDraftQueue,
  retainSurfaceOfflineDraft,
} from "@/components/Notes/notes-manager-utils";
import { createNotesGraphAuthorityScope } from "@/components/Notes/hooks/useNotesGraphAuthorityScope";
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events";
import { Storage as PlasmoStorage } from "@plasmohq/storage";
vi.mock(
  "@plasmohq/storage",
  () =>
    import("../../../../../../../tldw-frontend/extension/shims/plasmo-storage"),
);
import type { BgRequestInit } from "@/services/background-proxy";
import React from "react";
import { Modal } from "antd";
import {
  act,
  fireEvent,
  render,
  screen,
  waitFor,
} from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { QuickNotesSection } from "../StudioPane/QuickNotesSection";
import { useWorkspaceStore, buildWorkspaceSnapshot } from "@/store/workspace";
import type { WorkspaceNote } from "@/types/workspace"

const mocks = vi.hoisted(() => ({
  serverUrl: "https://research-a.example",
  userId: "alice",
  authMode: "multi-user" as "multi-user" | "single-user",
  apiKey: "private-key-a",
  orgId: null as number | null,
  authSource: "cookie-session" as "cookie-session" | "manual",
  request: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
  open: vi.fn(),
}));
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
      authMode: mocks.authMode,
      apiKey: mocks.apiKey,
      authSource: mocks.authSource,
      orgId: mocks.orgId,
    }),
  },
}));
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: {
    getCurrentUser: async () => ({ id: mocks.userId, is_active: true }),
  },
}));
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
        warning: vi.fn(),
      },
      null,
    ],
  },
}));

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
  if (boundary === "POST") {
    initial.id = undefined;
    const queueId = crypto.randomUUID();
    vi.spyOn(crypto, "randomUUID")
      .mockReturnValueOnce(queueId)
      .mockReturnValueOnce(noteId);
  }
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
  const tails = new Map<string, Promise<unknown>>();
  vi.stubGlobal(
    "navigator",
    Object.create(window.navigator, {
      locks: {
        value: {
          request: (key: string, operation: () => unknown) => {
            const next = (tails.get(key) ?? Promise.resolve()).then(operation);
            tails.set(
              key,
              next.catch(() => undefined),
            );
            return next;
          },
        },
      },
    }),
  );
  vi.clearAllMocks();
  window.localStorage.clear();
  mocks.serverUrl = "https://research-a.example"
  mocks.userId = "alice"
  mocks.authMode = "multi-user";
  mocks.apiKey = "private-key-a";
  mocks.orgId = null;
  mocks.authSource = "cookie-session";
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
        pendingNoteWriteKey: expect.stringContaining(
          'surface:quick-notes:["workspace-a","workspace:a"]:',
        ),
        isDirty: true,
      });
      expect(screen.queryByText(/^Saved$/)).not.toBeInTheDocument();
      fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }));
      await waitFor(() => expect(writes()).toHaveLength(2));
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
        throw Object.assign(new Error(message), {
          status,
          details: { detail },
        });
      return { ...request.body, id: request.body.id, version: 1 };
    });
    render(<QuickNotesSection />);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(writes()).toHaveLength(2));
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

function CollapsibleQuickNotes() {
  const [expanded, setExpanded] = React.useState(true)
  return expanded ? <QuickNotesSection onCollapse={() => setExpanded(false)} /> :
    <button onClick={() => setExpanded(true)}>Reopen Quick Notes</button>
}

it("keeps the content base after a conflict until explicit reload and merge", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), version: 40 } })
  let remoteBody = "Remote edited body"
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return []
    if (request.method === "GET") return { ...draft(), content: remoteBody, version: 41 }
    if (request.headers?.["expected-version"] !== "41")
      throw Object.assign(new Error("version conflict"), { status: 409 })
    remoteBody = String(request.body.content)
    return { ...request.body, id: noteId, version: 42 }
  })
  render(<QuickNotesSection />)
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.open).toHaveBeenCalledTimes(1))
  expect(useWorkspaceStore.getState().currentNote.version).toBe(40)
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(mocks.open).toHaveBeenCalledTimes(2))
  expect(remoteBody).toBe("Remote edited body")
  expect(writes()[1][0].headers["expected-version"]).toBe("40")
  render(mocks.open.mock.calls[1][0].content)
  fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
  await waitFor(() => expect(useWorkspaceStore.getState().currentNote.version).toBe(41))
  expect(useWorkspaceStore.getState().currentNote.content).toContain("Remote edited body")
  expect(useWorkspaceStore.getState().currentNote.content).toContain("Original draft body")
  fireEvent.click(screen.getByRole("button", { name: "Update" }))
  await waitFor(() => expect(useWorkspaceStore.getState().currentNote.version).toBe(42))
})

it.each(["unchanged", "newer", "versioned-new"])("reconciles a committed create after collapse with %s edits", async (change) => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined, version: change === "versioned-new" ? 1 : undefined } })
  const notes = new Map<string, Record<string, unknown>>()
  const receipts = new Map<string, Record<string, unknown>>()
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return []
    const key = String(request.headers?.["Idempotency-Key"])
    if (receipts.has(key)) return receipts.get(key)
    const id = String(request.body.id || crypto.randomUUID())
    const saved = { ...request.body, id, version: 1 }
    notes.set(id, saved)
    receipts.set(key, saved)
    throw new Error("Committed, response dropped")
  })
  render(<CollapsibleQuickNotes />)
  fireEvent.click(screen.getByRole("button", { name: "Save" }))
  await waitFor(() => expect(mocks.error).toHaveBeenCalled())
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }))
  if (change === "newer") act(() => { useWorkspaceStore.getState().updateNoteContent("Newer local body") })
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }))
  fireEvent.click(await screen.findByRole("button", { name: "Save" }))
  await waitFor(() => expect(writes()).toHaveLength(2))
  expect(notes.size).toBe(1)
  await waitFor(() => expect(mocks.success).toHaveBeenCalled())
  expect(writes()[1][0].body).toEqual(writes()[0][0].body)
  expect(writes()[1][0].headers).toEqual(writes()[0][0].headers)
  expect(writes()[0][0].body.id).toMatch(/^[a-f0-9-]{36}$/)
  expect(useWorkspaceStore.getState().currentNote).toMatchObject({
    id: writes()[0][0].body.id,
    content: change === "newer" ? "Newer local body" : "Original draft body",
    isDirty: change === "newer"
  })
})

it("does not dispatch when persistent operation retention fails", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } })
  mocks.request.mockResolvedValue([])
  render(<QuickNotesSection />)
  await screen.findByRole("button", { name: "Save" })
  const prototype = Object.getPrototypeOf(window.localStorage) as Storage
  const original = prototype.setItem
  const write = vi.spyOn(prototype, "setItem").mockImplementation(function (key, value) {
    if (key.startsWith("tldw:notesOfflineDraftQueue:v1")) throw new Error("Storage full")
    return original.call(this, key, value)
  })
  fireEvent.click(screen.getByRole("button", { name: "Save" }))
  await waitFor(() => expect(mocks.error).toHaveBeenCalled())
  expect(writes()).toHaveLength(0)
  write.mockRestore()
})

it.each(["workspace", "account", "clear-then-type"])("does not recover an uncertain operation into another %s after collapse", async (change) => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } })
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return []
    if (writes().length === 1) throw new Error("Response lost")
    return { ...request.body, id: request.body.id, version: 1 }
  })
  render(<CollapsibleQuickNotes />)
  fireEvent.click(screen.getByRole("button", { name: "Save" }))
  await waitFor(() => expect(mocks.error).toHaveBeenCalled())
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }))
  act(() => { changeContext(change) })
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }))
  fireEvent.click(await screen.findByRole("button", { name: "Save" }))
  await waitFor(() => expect(mocks.success).toHaveBeenCalled())
  expect(writes()[1][0].headers["Idempotency-Key"]).not.toBe(writes()[0][0].headers["Idempotency-Key"])
  expect(writes()[1][0].body.id).not.toBe(writes()[0][0].body.id)
})

it("publishes the canonical create identity before collapse interrupts asynchronous checkpoint retirement", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } })
  const notes = new Map<string, Record<string, unknown>>()
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    const id = String(request.body.id || crypto.randomUUID());
    const saved = { ...request.body, id, version: 1 };
    notes.set(id, saved);
    return saved;
  });
  const gate = deferred();
  const set = PlasmoStorage.prototype.set;
  const retiring = vi
    .spyOn(PlasmoStorage.prototype, "set")
    .mockImplementation(async function (key, value) {
      if (value?.value?.metadata?.quickNotesAcceptedKey) await gate.promise;
      return set.call(this, key, value);
    });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() =>
    expect(
      retiring.mock.calls.some(
        ([, value]) => value?.value?.metadata?.quickNotesAcceptedKey,
      ),
    ).toBe(true),
  );
  expect(useWorkspaceStore.getState().currentNote.id).toBe(
    writes()[0][0].body.id,
  );
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    gate.resolve();
    await gate.promise;
  });
  retiring.mockRestore();
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  expect(
    document.body.contains(
      await screen.findByRole("button", { name: "Update" }),
    ),
  ).toBe(true);
  expect(notes.size).toBe(1);
});

it.each(["base-path", "org", "auth-source"])(
  "refuses retained Quick Notes adoption after a same-owner %s service change",
  async (change) => {
    mocks.serverUrl = "https://shared.example/server-a";
    useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return [];
      throw new Error("Committed, response dropped");
    });
    render(<CollapsibleQuickNotes />);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
    const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
    const retained = await readSurfaceOfflineDraftQueue(owner);
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    if (change === "base-path")
      mocks.serverUrl = "https://shared.example/server-b";
    if (change === "org") mocks.orgId = 2;
    if (change === "auth-source") mocks.authSource = "manual";
    expect(createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId)).toBe(
      owner,
    );
    fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(2));
    expect(writes()).toHaveLength(1);
    expect(await readSurfaceOfflineDraftQueue(owner)).toEqual(retained);
  },
);

it("refuses dispatch if the operation record succeeds but the real Workspace pointer persistence fails", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const originalEnvelope = localStorage.getItem(WORKSPACE_STORAGE_KEY);
  const prototype = Object.getPrototypeOf(localStorage) as Storage;
  const write = prototype.setItem;
  const failed = vi
    .spyOn(prototype, "setItem")
    .mockImplementation(function (key, value) {
      if (key.startsWith(WORKSPACE_STORAGE_KEY))
        throw new DOMException("Quota full", "QuotaExceededError");
      return write.call(this, key, value);
    });
  const notes = new Map<string, Record<string, unknown>>();
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    notes.set(String(request.body.id), request.body);
    throw new Error("Committed, response dropped");
  });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
  expect(
    Object.keys(
      await readSurfaceOfflineDraftQueue(
        createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId),
      ),
    ),
  ).toHaveLength(1);
  expect(localStorage.getItem(WORKSPACE_STORAGE_KEY)).toBe(originalEnvelope);
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  expect(
    useWorkspaceStore.getState().currentNote.pendingNoteWriteKey,
  ).toBeUndefined();
  failed.mockRestore();
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(2));
  expect(notes.size).toBe(1);
  expect(writes()).toHaveLength(1);
});

it("refuses dispatch while the existing Workspace adapter delays pointer persistence without replacing newer edits", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const storage = useWorkspaceStore.persist.getOptions().storage!;
  const originalSet = storage.setItem.bind(storage);
  const gate = deferred();
  const delayed = vi
    .spyOn(storage, "setItem")
    .mockImplementation(async (name, value) => {
      await gate.promise;
      return originalSet(name, value);
    });
  mocks.request.mockImplementation(async (request: BgRequestInit) =>
    request.path.includes("/search/")
      ? []
      : { ...request.body, id: request.body.id, version: 1 },
  );
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(delayed).toHaveBeenCalled());
  act(() => {
    useWorkspaceStore
      .getState()
      .updateNoteContent("Newer edit during delayed persistence");
  });
  await waitFor(() => expect(mocks.error).toHaveBeenCalled());
  expect(writes()).toHaveLength(0);
  expect(useWorkspaceStore.getState().currentNote.content).toBe(
    "Newer edit during delayed persistence",
  );
  await act(async () => {
    gate.resolve();
    await gate.promise;
  });
  delayed.mockRestore();
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  expect(useWorkspaceStore.getState().currentNote.content).toBe(
    "Newer edit during delayed persistence",
  );
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.success).toHaveBeenCalled());
  expect(writes()).toHaveLength(1);
  expect(writes()[0][0].body.content).toBe("Original draft body");
  expect(useWorkspaceStore.getState().currentNote.content).toBe(
    "Newer edit during delayed persistence",
  );
});

it("refuses dispatch when the real persisted Workspace snapshot cannot be read", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const storage = useWorkspaceStore.persist.getOptions().storage!;
  const read = vi
    .spyOn(storage, "getItem")
    .mockRejectedValue(new Error("Workspace storage unavailable"));
  mocks.request.mockResolvedValue([]);
  render(<QuickNotesSection />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalled());
  expect(writes()).toHaveLength(0);
  expect(
    Object.keys(
      await readSurfaceOfflineDraftQueue(
        createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId),
      ),
    ),
  ).toHaveLength(1);
  read.mockRestore();
});

it.each(["quota", "migration"])(
  "recovers a pointerless canonical UUID update after %s without replacing newer edits",
  async (boundary) => {
    useWorkspaceStore.setState({ currentNote: { ...draft(), version: 3 } });
    const serverSnapshot = buildWorkspaceSnapshot(useWorkspaceStore.getState());
    if (boundary === "migration")
      localStorage.setItem(
        "tldw:research-workspace:migration:tombstone:workspace-a",
        JSON.stringify({
          legacyWorkspaceId: "workspace-a",
          serverWorkspaceId: "workspace-a",
          migrationId: "migration-a",
          contentRetained: false,
          deletedAt: "2026-10-08T00:00:00Z",
        }),
      );
    const prototype = Object.getPrototypeOf(localStorage) as Storage;
    const originalWrite = prototype.setItem;
    const failing = vi
      .spyOn(prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (boundary === "quota" && key.startsWith(WORKSPACE_STORAGE_KEY))
          throw new DOMException("Quota full", "QuotaExceededError");
        return originalWrite.call(this, key, value);
      });
    const receipts = new Map<string, Record<string, unknown>>();
    let version = 3;
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return [];
      const key = String(request.headers?.["Idempotency-Key"]);
      if (receipts.has(key)) return receipts.get(key);
      expect(request.path).toBe(`/api/v1/notes/${noteId}`);
      expect(request.method).toBe("PUT");
      expect(request.headers?.["expected-version"]).toBe("3");
      version += 1;
      const saved = { ...request.body, id: noteId, version };
      receipts.set(key, saved);
      throw new Error("Committed update, response dropped");
    });
    render(<CollapsibleQuickNotes />);
    fireEvent.click(screen.getByRole("button", { name: "Update" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
    expect(writes()).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    await act(async () => {
      await useWorkspaceStore.persist.rehydrate();
    });
    failing.mockRestore();
    if (boundary === "migration")
      act(() => {
        useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
      });
    expect(
      useWorkspaceStore.getState().currentNote.pendingNoteWriteKey,
    ).toBeUndefined();
    act(() => {
      useWorkspaceStore
        .getState()
        .updateNoteContent("Newer edit after restoring the canonical Note");
    });
    fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
    fireEvent.click(screen.getByRole("button", { name: "Update" }));
    await waitFor(() => expect(mocks.success).toHaveBeenCalled());
    expect(writes()[1][0]).toMatchObject({
      body: writes()[0][0].body,
      headers: writes()[0][0].headers,
      path: writes()[0][0].path,
      method: "PUT",
    });
    expect(version).toBe(4);
    expect(useWorkspaceStore.getState().currentNote.content).toBe(
      "Newer edit after restoring the canonical Note",
    );
    expect(useWorkspaceStore.getState().currentNote.isDirty).toBe(true);
  },
);

it("allows a genuinely unbound new Quick Note in a migrated server Workspace", async () => {
  localStorage.setItem(
    "tldw:research-workspace:migration:tombstone:workspace-a",
    JSON.stringify({
      legacyWorkspaceId: "workspace-a",
      serverWorkspaceId: "workspace-a",
      migrationId: "migration-a",
      contentRetained: false,
      deletedAt: "2026-10-08T00:00:00Z",
    }),
  );
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  mocks.request.mockImplementation(async (request: BgRequestInit) =>
    request.path.includes("/search/")
      ? []
      : { ...request.body, id: request.body.id, version: 1 },
  );
  render(<QuickNotesSection />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() =>
    expect(
      mocks.error.mock.calls.length + mocks.success.mock.calls.length,
    ).toBeGreaterThan(0),
  );
  expect(writes()).toHaveLength(1);
  expect(mocks.error).toHaveBeenCalled();
  expect(
    Object.values(
      await readSurfaceOfflineDraftQueue(
        createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId),
      ),
    )[0].pendingWrite?.body.id,
  ).toBe(writes()[0][0].body.id);
});

it.each(["newer", "cleared", "replaced", "authority", "workspace", "multiple"])(
  "explicitly retries a migrated create after remount without adopting it into a %s draft",
  async (boundary) => {
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        contentRetained: false,
        deletedAt: "2026-10-08T00:00:00Z",
      }),
    );
    useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
    const serverSnapshot = buildWorkspaceSnapshot(useWorkspaceStore.getState());
    const receipts = new Map<string, Record<string, unknown>>();
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return [];
      const key = String(request.headers?.["Idempotency-Key"]);
      if (receipts.has(key)) return receipts.get(key);
      receipts.set(key, { ...request.body, id: request.body.id, version: 1 });
      throw new Error("Committed create, response dropped");
    });
    render(<CollapsibleQuickNotes />);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
    expect(writes()).toHaveLength(1);
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    await act(async () => {
      await useWorkspaceStore.persist.rehydrate();
    });
    act(() => {
      useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
    });
    expect(
      useWorkspaceStore.getState().currentNote.pendingNoteWriteKey,
    ).toBeUndefined();
    act(() => {
      if (boundary === "cleared") {
        useWorkspaceStore.getState().clearCurrentNote();
        useWorkspaceStore
          .getState()
          .updateNoteContent("Different draft after Clear");
      } else if (boundary === "replaced")
        useWorkspaceStore
          .getState()
          .loadNote({ ...draft(otherId), id: otherId, version: 8 });
      else
        useWorkspaceStore
          .getState()
          .updateNoteContent("Newer live draft after remount");
    });
    fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
    fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }));
    await waitFor(() => expect(mocks.open).toHaveBeenCalled());
    expect(writes()).toHaveLength(1);
    const action = mocks.open.mock.calls.at(-1)![0];
    const recovery = render(action.content);
    expect(screen.getByText(/Alice draft/)).toBeTruthy();
    if (boundary === "authority") mocks.serverUrl += "/other-service";
    if (boundary === "workspace")
      act(() => {
        useWorkspaceStore.setState({
          workspaceId: "workspace-b",
          workspaceTag: "workspace:b",
        });
      });
    if (boundary === "multiple") {
      const owner = createNotesGraphAuthorityScope(
        mocks.serverUrl,
        mocks.userId,
      );
      const entry = Object.values(await readSurfaceOfflineDraftQueue(owner))[0];
      await retainSurfaceOfflineDraft(owner, {
        ...entry,
        key: entry.key + ":other",
        pendingWrite: {
          ...entry.pendingWrite!,
          key: "other-key",
          body: { ...entry.pendingWrite!.body, id: otherId },
        },
      });
    }
    const liveDraft = useWorkspaceStore.getState().currentNote;
    fireEvent.click(
      screen.getByRole("button", { name: "Retry previous save" }),
    );
    if (["authority", "workspace", "multiple"].includes(boundary)) {
      await waitFor(() =>
        expect(mocks.error.mock.calls.length).toBeGreaterThan(1),
      );
      expect(writes()).toHaveLength(1);
    } else {
      await waitFor(() => expect(writes()).toHaveLength(2));
      expect(writes()[1][0]).toMatchObject({
        body: writes()[0][0].body,
        headers: writes()[0][0].headers,
        method: "POST",
      });
      expect(writes()[1][0].abortSignal).not.toBe(writes()[0][0].abortSignal);
      await waitFor(async () => {
        const rows = Object.values(
          await readSurfaceOfflineDraftQueue(
            createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId),
          ),
        );
        expect(rows).toHaveLength(1);
        expect(rows[0]).toMatchObject({
          noteId: writes()[0][0].body.id,
          baseVersion: 1,
          metadata: {
            quickNotesAcceptedKey: writes()[0][0].headers["Idempotency-Key"],
            quickNotesDirty: false,
          },
        });
        expect(rows[0].pendingWrite).toBeUndefined();
        expect(rows[0].metadata?.quickNotesAuthorityId).toBeTruthy();
      });
      expect(receipts.size).toBe(1);
    }
    expect(useWorkspaceStore.getState().currentNote).toBe(liveDraft);
    recovery.unmount();
  },
);

it.each(["ambiguous", "different-body-id"])(
  "refuses a pointerless canonical update with %s recovery",
  async (boundary) => {
    useWorkspaceStore.setState({ currentNote: { ...draft(), version: 3 } });
    const serverSnapshot = buildWorkspaceSnapshot(useWorkspaceStore.getState());
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return [];
      throw new Error("Committed update, response dropped");
    });
    render(<CollapsibleQuickNotes />);
    fireEvent.click(screen.getByRole("button", { name: "Update" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    await act(async () => {
      await useWorkspaceStore.persist.rehydrate();
    });
    act(() => {
      useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
    });
    const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
    const entry = Object.values(await readSurfaceOfflineDraftQueue(owner))[0];
    await retainSurfaceOfflineDraft(owner, {
      ...entry,
      key: entry.key + ":other",
      pendingWrite: {
        ...entry.pendingWrite!,
        key: "competing-update",
        body: {
          ...entry.pendingWrite!.body,
          ...(boundary === "different-body-id" ? { id: otherId } : {}),
        },
      },
    });
    if (boundary === "different-body-id") {
      const { retireSurfaceOfflineDraft } =
        await import("@/components/Notes/notes-manager-utils");
      await retireSurfaceOfflineDraft(
        owner,
        entry.key,
        entry.pendingWrite!.key,
      );
    }
    const retained = await readSurfaceOfflineDraftQueue(owner);
    fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
    fireEvent.click(screen.getByRole("button", { name: "Update" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(2));
    expect(writes()).toHaveLength(1);
    expect(await readSurfaceOfflineDraftQueue(owner)).toEqual(retained);
    expect(useWorkspaceStore.getState().currentNote).toEqual(
      serverSnapshot.currentNote,
    );
  },
);


it("refuses a rotated single-user key through the real captured Quick Notes snapshot after remount", async () => {
  mocks.authMode = "single-user";
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    throw new Error("Committed create, response dropped");
  });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
  const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
  const retained = await readSurfaceOfflineDraftQueue(owner);
  expect(writes()).toHaveLength(1);
  expect(JSON.stringify(retained)).not.toContain(mocks.apiKey);
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  mocks.apiKey = "private-key-b";
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(2));
  expect(writes()).toHaveLength(1);
  expect(await readSurfaceOfflineDraftQueue(owner)).toEqual(retained);
});

it.each(["422", "409", "policy", "uncertain"])(
  "handles explicit migrated recovery %s using only its rejected operation",
  async (boundary) => {
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        contentRetained: false,
        deletedAt: "2026-10-08T00:00:00Z",
      }),
    );
    useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
    const serverSnapshot = buildWorkspaceSnapshot(useWorkspaceStore.getState());
    const readback = deferred();
    let definitive = false;
    let deliberate = false;
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return [];
      if (request.method === "GET") {
        await readback.promise;
        return {
          ...draft(otherId),
          version: 19,
          knowledge_provenance_state: "active",
          knowledge_provenance_version: 19,
        };
      }
      if (deliberate)
        return { ...request.body, id: request.body.id, version: 1 };
      if (!definitive)
        throw new Error("Response lost before execution can be confirmed");
      if (boundary === "uncertain") throw new Error("Still uncertain");
      if (boundary === "policy")
        throw Object.assign(
          new Error("notes_provenance_encryption_unsupported"),
          { status: 409, code: "notes_provenance_encryption_unsupported" },
        );
      throw Object.assign(new Error("Rejected previous operation"), {
        status: Number(boundary),
      });
    });
    render(<CollapsibleQuickNotes />);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
    const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
    const retained = await readSurfaceOfflineDraftQueue(owner);
    fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
    await act(async () => {
      await useWorkspaceStore.persist.rehydrate();
    });
    act(() => {
      useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
      useWorkspaceStore
        .getState()
        .setCurrentNote({
          ...draft(boundary === "409" ? otherId : undefined),
          id: boundary === "409" ? otherId : undefined,
          content: "An unrelated deliberate new draft",
          version: boundary === "409" ? 8 : undefined,
        });
    });
    fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
    fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }));
    await waitFor(() => expect(mocks.open).toHaveBeenCalled());
    const recovery = render(mocks.open.mock.calls.at(-1)![0].content);
    const live = useWorkspaceStore.getState().currentNote;
    definitive = true;
    fireEvent.click(
      screen.getByRole("button", { name: "Retry previous save" }),
    );
    await waitFor(() => expect(writes()).toHaveLength(2));
    if (boundary === "409") {
      // Replace while the old shared catch would have a canonical head GET pending.
      await act(async () => {
        await Promise.resolve();
        await Promise.resolve();
      });
      act(() => {
        useWorkspaceStore
          .getState()
          .setCurrentNote({
            ...draft(noteId),
            id: noteId,
            content: "Replacement while readback would be pending",
            version: 11,
          });
      });
      const replacement = useWorkspaceStore.getState().currentNote;
      await act(async () => {
        readback.resolve();
        await readback.promise;
      });
      await waitFor(() =>
        expect(document.querySelector("button.ant-btn-loading")).toBeNull(),
      );
      expect(
        mocks.request.mock.calls.filter(
          ([r]) => r.method === "GET" && !r.path.includes("/search/"),
        ),
      ).toHaveLength(0);
      expect(useWorkspaceStore.getState().currentNote).toBe(replacement);
      expect(
        mocks.open.mock.calls.some(
          ([m]) => m.key === "workspace-note-version-conflict",
        ),
      ).toBe(false);
    } else {
      await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(2));
      expect(useWorkspaceStore.getState().currentNote).toBe(live);
    }
    if (boundary === "422" || boundary === "409") {
      expect(await readSurfaceOfflineDraftQueue(owner)).toEqual({});
      if (boundary === "422") {
        deliberate = true;
        fireEvent.click(screen.getByRole("button", { name: "Save" }));
        await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(3));
        expect(
          Object.values(await readSurfaceOfflineDraftQueue(owner))[0].pendingWrite
            ?.body.id,
        ).toBe(writes()[2][0].body.id);
        expect(writes()).toHaveLength(3);
        expect(writes()[2][0].body.content).toBe(
          "An unrelated deliberate new draft",
        );
        expect(writes()[2][0].body.id).not.toBe(writes()[0][0].body.id);
        expect(writes()[2][0].headers["Idempotency-Key"]).not.toBe(
          writes()[0][0].headers["Idempotency-Key"],
        );
      }
    } else expect(await readSurfaceOfflineDraftQueue(owner)).toEqual(retained);
    recovery.unmount();
  },
);

it("retains accepted create identity when canonical acknowledgement cannot persist to the real Workspace adapter", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
  const prototype = Object.getPrototypeOf(localStorage) as Storage;
  const originalWrite = prototype.setItem;
  let acceptedId: string | undefined;
  const serverNotes = new Map<string, Record<string, unknown>>();
  const failed = vi
    .spyOn(prototype, "setItem")
    .mockImplementation(function (key, value) {
      if (acceptedId && key.startsWith(WORKSPACE_STORAGE_KEY))
        throw new DOMException(
          "Quota full during acknowledgement",
          "QuotaExceededError",
        );
      return originalWrite.call(this, key, value);
    });
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    acceptedId = String(request.body.id);
    serverNotes.set(acceptedId, request.body);
    return { ...request.body, id: request.body.id, version: 1 };
  });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() =>
    expect(
      mocks.success.mock.calls.length + mocks.error.mock.calls.length,
    ).toBeGreaterThan(0),
  );
  expect(serverNotes.size).toBe(1);
  const accepted = acceptedId;
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  failed.mockRestore();
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  const restored = useWorkspaceStore.getState().currentNote;
  const retained = await readSurfaceOfflineDraftQueue(owner);
  expect({
    canonical: restored.id === accepted,
    recoverable: Object.keys(retained).length > 0,
  }).not.toEqual({ canonical: false, recoverable: false });
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }));
  await waitFor(() => expect(writes()).toHaveLength(2));
  expect(serverNotes.size).toBe(1);
});

it("does not create a second generated Note after acknowledged Workspace persistence loss and rehydrate", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
  const prototype = Object.getPrototypeOf(localStorage) as Storage;
  const originalWrite = prototype.setItem;
  let acceptedId: string | undefined;
  const serverNotes = new Map<string, Record<string, unknown>>();
  const failed = vi
    .spyOn(prototype, "setItem")
    .mockImplementation(function (key, value) {
      if (acceptedId && key.startsWith(WORKSPACE_STORAGE_KEY))
        throw new DOMException(
          "Quota full during acknowledgement",
          "QuotaExceededError",
        );
      return originalWrite.call(this, key, value);
    });
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    acceptedId = String(request.body.id);
    serverNotes.set(acceptedId, request.body);
    return { ...request.body, id: request.body.id, version: 1 };
  });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() =>
    expect(
      mocks.success.mock.calls.length + mocks.error.mock.calls.length,
    ).toBeGreaterThan(0),
  );
  expect(serverNotes.size).toBe(1);
  const accepted = acceptedId;
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  failed.mockRestore();
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  const restored = useWorkspaceStore.getState().currentNote;
  const retained = await readSurfaceOfflineDraftQueue(owner);
  expect(
    restored.id === accepted ||
      Object.values(retained).some(
        (entry) => entry.pendingWrite?.body.id === accepted,
      ),
  ).toBe(true);
  act(() => {
    useWorkspaceStore
      .getState()
      .updateNoteContent("Newer edit after acknowledgement storage loss");
  });
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }));
  await waitFor(() => expect(writes()).toHaveLength(2));
  await waitFor(() => expect(mocks.success).toHaveBeenCalled());
  expect(writes()[1][0]).toMatchObject({
    body: writes()[0][0].body,
    headers: writes()[0][0].headers,
    method: "POST",
  });
  expect(serverNotes.size).toBe(1);
  expect(useWorkspaceStore.getState().currentNote).toMatchObject({
    id: accepted,
    content: "Newer edit after acknowledgement storage loss",
    isDirty: true,
  });
});

it.each(["delayed", "unreadable", "replacement"])(
  "preserves the accepted operation through %s canonical acknowledgement readback",
  async (boundary) => {
    useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
    const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
    const storage = useWorkspaceStore.persist.getOptions().storage!;
    const gate = deferred();
    const getItem = storage.getItem.bind(storage);
    const setItem = storage.setItem.bind(storage);
    let acceptedId: string | undefined;
    const serverNotes = new Map<string, Record<string, unknown>>();
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return [];
      acceptedId = String(request.body.id);
      serverNotes.set(acceptedId, request.body);
      return { ...request.body, id: request.body.id, version: 1 };
    });
    const delayed = vi
      .spyOn(storage, "setItem")
      .mockImplementation(async (key, value) => {
        if (
          boundary === "delayed" &&
          acceptedId &&
          value.state.workspaceSnapshots?.["workspace-a"]?.currentNote.id ===
            acceptedId
        )
          await gate.promise;
        return setItem(key, value);
      });
    const reading = vi
      .spyOn(storage, "getItem")
      .mockImplementation(async (key) => {
        if (acceptedId && boundary === "unreadable")
          throw new Error("Acknowledgement snapshot unreadable");
        if (acceptedId && boundary === "replacement") await gate.promise;
        return getItem(key);
      });
    render(<CollapsibleQuickNotes />);
    fireEvent.click(screen.getByRole("button", { name: "Save" }));
    await waitFor(() => expect(writes()).toHaveLength(1));
    if (boundary === "replacement") {
      await waitFor(() => expect(reading.mock.calls.length).toBeGreaterThan(1));
      act(() => {
        useWorkspaceStore
          .getState()
          .loadNote({ ...draft(otherId), id: otherId, version: 8 });
      });
      const replacement = useWorkspaceStore.getState().currentNote;
      await act(async () => {
        gate.resolve();
        await gate.promise;
      });
      await waitFor(() =>
        expect(document.querySelector("button.ant-btn-loading")).toBeNull(),
      );
      expect(useWorkspaceStore.getState().currentNote).toBe(replacement);
      expect(
        Object.values(await readSurfaceOfflineDraftQueue(owner))[0].pendingWrite
          ?.body.id,
      ).toBe(acceptedId);
    } else {
      await waitFor(() => expect(mocks.error).toHaveBeenCalled());
      expect(
        Object.values(await readSurfaceOfflineDraftQueue(owner))[0].pendingWrite
          ?.body.id,
      ).toBe(acceptedId);
      act(() => {
        useWorkspaceStore
          .getState()
          .updateNoteContent(
            "Newer edit while the acknowledged identity is pending",
          );
      });
      await act(async () => {
        gate.resolve();
        await gate.promise;
      });
      delayed.mockRestore();
      reading.mockRestore();
      fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
      await act(async () => {
        await useWorkspaceStore.persist.rehydrate();
      });
      fireEvent.click(
        screen.getByRole("button", { name: "Reopen Quick Notes" }),
      );
      fireEvent.click(screen.getByRole("button", { name: /^(Save|Update)$/ }));
      await waitFor(() => expect(mocks.success).toHaveBeenCalled());
      expect(writes()[1][0]).toMatchObject({
        body: writes()[0][0].body,
        headers: writes()[0][0].headers,
      });
      expect(serverNotes.size).toBe(1);
      expect(useWorkspaceStore.getState().currentNote).toMatchObject({
        id: acceptedId,
        content: "Newer edit while the acknowledged identity is pending",
        isDirty: true,
      });
    }
    delayed.mockRestore();
    reading.mockRestore();
  },
);

it("keeps a migrated acknowledged new create recoverable until explicit previous-save confirmation without adopting its ID", async () => {
  localStorage.setItem(
    "tldw:research-workspace:migration:tombstone:workspace-a",
    JSON.stringify({
      legacyWorkspaceId: "workspace-a",
      serverWorkspaceId: "workspace-a",
      migrationId: "migration-a",
      contentRetained: false,
      deletedAt: "2026-10-08T00:00:00Z",
    }),
  );
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const serverSnapshot = buildWorkspaceSnapshot(useWorkspaceStore.getState());
  const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
  const serverNotes = new Map<string, Record<string, unknown>>();
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    const id = String(request.body.id);
    serverNotes.set(id, request.body);
    return { ...request.body, id, version: 1 };
  });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalled());
  expect(
    Object.values(await readSurfaceOfflineDraftQueue(owner))[0].pendingWrite
      ?.body.id,
  ).toBe(writes()[0][0].body.id);
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  act(() => {
    useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
    useWorkspaceStore.getState().clearCurrentNote();
    useWorkspaceStore
      .getState()
      .updateNoteContent("Unrelated new migrated draft");
  });
  const unrelated = useWorkspaceStore.getState().currentNote;
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.open).toHaveBeenCalled());
  const recovery = render(mocks.open.mock.calls.at(-1)![0].content);
  fireEvent.click(screen.getByRole("button", { name: "Retry previous save" }));
  await waitFor(async () => {
    const rows = Object.values(await readSurfaceOfflineDraftQueue(owner));
    expect(rows).toHaveLength(1);
    expect(rows[0]).toMatchObject({
      noteId: writes()[0][0].body.id,
      baseVersion: 1,
      metadata: {
        quickNotesAcceptedKey: writes()[0][0].headers["Idempotency-Key"],
        quickNotesDirty: false,
      },
    });
    expect(rows[0].pendingWrite).toBeUndefined();
    expect(rows[0].metadata?.quickNotesAuthorityId).toBeTruthy();
  });
  expect(writes()[1][0]).toMatchObject({
    body: writes()[0][0].body,
    headers: writes()[0][0].headers,
  });
  expect(serverNotes.size).toBe(1);
  expect(useWorkspaceStore.getState().currentNote).toBe(unrelated);
  recovery.unmount();
});

it("keeps the durable create operation through migrated in-memory UUID edits and acknowledgement before real rehydrate", async () => {
  localStorage.setItem(
    "tldw:research-workspace:migration:tombstone:workspace-a",
    JSON.stringify({
      legacyWorkspaceId: "workspace-a",
      serverWorkspaceId: "workspace-a",
      migrationId: "migration-a",
      contentRetained: false,
      deletedAt: "2026-10-08T00:00:00Z",
    }),
  );
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const serverSnapshot = buildWorkspaceSnapshot(useWorkspaceStore.getState());
  const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
  const serverNotes = new Map<string, Record<string, unknown>>();
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    const id = String(request.body.id);
    serverNotes.set(id, request.body);
    return { ...request.body, id, version: 1 };
  });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(1));
  const acceptedId = writes()[0][0].body.id;
  expect(useWorkspaceStore.getState().currentNote.id).toBe(acceptedId);
  act(() => {
    useWorkspaceStore.getState().updateNoteContent("Later migrated edit");
  });
  fireEvent.click(screen.getByRole("button", { name: "Update" }));
  await waitFor(() => expect(mocks.error).toHaveBeenCalledTimes(2));
  expect(writes()[1][0]).toMatchObject({
    method: "POST",
    body: writes()[0][0].body,
    headers: writes()[0][0].headers,
  });
  expect(
    Object.values(await readSurfaceOfflineDraftQueue(owner))[0].pendingWrite
      ?.body.id,
  ).toBe(acceptedId);
  expect(useWorkspaceStore.getState().currentNote.content).toBe(
    "Later migrated edit",
  );
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  act(() => {
    useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
    useWorkspaceStore
      .getState()
      .updateNoteContent("Next migrated edit after rehydrate");
  });
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(mocks.open).toHaveBeenCalled());
  expect(writes()).toHaveLength(2);
  expect(serverNotes.size).toBe(1);
  expect(
    Object.values(await readSurfaceOfflineDraftQueue(owner))[0].pendingWrite
      ?.body.id,
  ).toBe(acceptedId);
  const recovery = render(mocks.open.mock.calls.at(-1)![0].content);
  expect(
    screen.getByRole("button", { name: "Retry previous save" }),
  ).toBeTruthy();
  recovery.unmount();
});

afterEach(() => {
  Modal.destroyAll();
  vi.restoreAllMocks();
  vi.unstubAllGlobals();
});

it("recovers later local text after a tombstoned reload through explicit retry and dirty resume", async () => {
  localStorage.setItem(
    "tldw:research-workspace:migration:tombstone:workspace-a",
    JSON.stringify({
      legacyWorkspaceId: "workspace-a",
      serverWorkspaceId: "workspace-a",
      migrationId: "migration-a",
      contentRetained: false,
      deletedAt: "2026-10-08T00:00:00Z",
    }),
  );
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const serverSnapshot = buildWorkspaceSnapshot(useWorkspaceStore.getState());
  const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
  const gate = deferred();
  const serverNotes = new Map<string, unknown>();
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    if (request.method === "POST") {
      serverNotes.set(String(request.body.id), request.body);
      if (writes().length === 1) {
        await gate.promise;
        throw new Error("Lost response");
      }
      return { ...request.body, id: request.body.id, version: 2 };
    }
    if (request.method === "PUT") throw { status: 409 };
    return {
      id: [...serverNotes.keys()][0],
      title: "Remote newer",
      content: "Remote version 3",
      version: 3,
    };
  });
  render(<CollapsibleQuickNotes />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(writes()).toHaveLength(1));
  fireEvent.change(screen.getByLabelText("Note title"), {
    target: { value: "Later local title" },
  });
  fireEvent.change(screen.getByLabelText("Note content"), {
    target: { value: "Later local body" },
  });
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    gate.resolve();
    await gate.promise;
  });
  await waitFor(async () =>
    expect(
      Object.values(await readSurfaceOfflineDraftQueue(owner))[0],
    ).toMatchObject({
      title: "Later local title",
      content: "Later local body",
      pendingWrite: { body: writes()[0][0].body },
    }),
  );
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  act(() => {
    useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
    useWorkspaceStore.getState().clearCurrentNote();
  });
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  const retry = await screen.findByRole("button", {
    name: "Retry previous save",
  });
  expect(screen.queryByRole("button", { name: "Save" })).toBeNull();
  fireEvent.click(retry);
  const resume = await screen.findByRole("button", {
    name: "Resume local unsaved draft",
  });
  expect(useWorkspaceStore.getState().currentNote.content).toBe("");
  expect(writes()[1][0]).toMatchObject({
    method: "POST",
    body: writes()[0][0].body,
    headers: writes()[0][0].headers,
  });
  expect(serverNotes.size).toBe(1);
  fireEvent.click(resume);
  await waitFor(() =>
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      id: [...serverNotes.keys()][0],
      title: "Later local title",
      content: "Later local body",
      version: 2,
      isDirty: true,
    }),
  );
  fireEvent.click(screen.getByRole("button", { name: "Update" }));
  await waitFor(() => expect(writes()).toHaveLength(3));
  expect(writes()[2][0].headers["expected-version"]).toBe("2");
  await waitFor(() =>
    expect(
      mocks.open.mock.calls.some(
        ([entry]) => entry.key === "workspace-note-version-conflict",
      ),
    ).toBe(true),
  );
  const rejected = Object.values(await readSurfaceOfflineDraftQueue(owner))[0];
  expect(rejected).toMatchObject({
    content: "Later local body",
    noteId: [...serverNotes.keys()][0],
    baseVersion: 2,
    metadata: {
      quickNotesRejectedKey: writes()[2][0].headers["Idempotency-Key"],
      quickNotesDirty: true,
    },
  });
  expect(rejected.pendingWrite).toBeUndefined();
  fireEvent.click(screen.getByRole("button", { name: "Collapse" }));
  await act(async () => {
    await useWorkspaceStore.persist.rehydrate();
  });
  act(() => {
    useWorkspaceStore.getState().restoreServerWorkspace(serverSnapshot);
    useWorkspaceStore.getState().clearCurrentNote();
  });
  fireEvent.click(screen.getByRole("button", { name: "Reopen Quick Notes" }));
  fireEvent.click(
    await screen.findByRole("button", { name: "Resume local unsaved draft" }),
  );
  await waitFor(() =>
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      content: "Later local body",
      version: 2,
      isDirty: true,
    }),
  );
});

it("retains edits made during delayed ACK checkpoint readback without an absent-row window", async () => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
  const gate = deferred();
  let entered = false;
  let accepted = false;
  const get = PlasmoStorage.prototype.get;
  const delayed = vi
    .spyOn(PlasmoStorage.prototype, "get")
    .mockImplementation(async function (key: string) {
      const value = await get.call(this, key);
      if (
        accepted &&
        key.startsWith("tldw:notesOfflineDraftQueue:v1:") &&
        !entered
      ) {
        entered = true;
        await gate.promise;
      }
      return value;
    });
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/")) return [];
    accepted = true;
    return { ...request.body, id: request.body.id, version: 2 };
  });
  render(<QuickNotesSection />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() => expect(entered).toBe(true));
  fireEvent.change(screen.getByLabelText("Note content"), {
    target: { value: "Edited during ACK readback" },
  });
  await act(async () => {
    gate.resolve();
    await gate.promise;
  });
  await waitFor(() => expect(mocks.success).toHaveBeenCalled());
  delayed.mockRestore();
  await waitFor(async () =>
    expect(
      Object.values(await readSurfaceOfflineDraftQueue(owner))[0],
    ).toMatchObject({
      content: "Edited during ACK readback",
      noteId: writes()[0][0].body.id,
      baseVersion: 2,
    }),
  );
  expect(
    Object.values(await readSurfaceOfflineDraftQueue(owner))[0].pendingWrite,
  ).toBeUndefined();
});

it("confirms dirty saved-note selection before requesting or replacing the current draft", async () => {
  mocks.request.mockImplementation(async (request: BgRequestInit) =>
    request.path.includes("/search/")
      ? [
          {
            id: otherId,
            title: "Saved chip",
            content: "Canonical body",
            version: 3,
            keywords: ["workspace:a"],
          },
        ]
      : {
          id: otherId,
          title: "Saved chip",
          content: "Canonical body",
          version: 3,
        },
  );
  render(<QuickNotesSection />);
  fireEvent.click(await screen.findByRole("button", { name: "Saved chip" }));
  expect(useWorkspaceStore.getState().currentNote.content).toBe(
    "Original draft body",
  );
  expect(
    mocks.request.mock.calls.filter(
      ([request]) => request.path === `/api/v1/notes/${otherId}`,
    ),
  ).toHaveLength(0);
  const confirm = await screen.findByRole("dialog");
  expect(confirm.textContent).toContain("Unsaved Changes");
  fireEvent.click(screen.getByRole("button", { name: "OK" }));
  await waitFor(() =>
    expect(useWorkspaceStore.getState().currentNote.id).toBe(otherId),
  );
});

it.each([
  "server",
  "account",
  "workspace-return",
  "note-return",
  "clear-then-type",
  "edit",
])("rejects a late saved-note GET after %s changes", async (change) => {
  useWorkspaceStore.setState({ currentNote: { ...draft(), isDirty: false } });
  const gate = deferred();
  mocks.request.mockImplementation(async (request: BgRequestInit) => {
    if (request.path.includes("/search/"))
      return [{ id: otherId, title: "Saved chip", keywords: ["workspace:a"] }];
    await gate.promise;
    return {
      id: otherId,
      title: "Saved chip",
      content: "Canonical body",
      version: 3,
    };
  });
  render(<QuickNotesSection />);
  fireEvent.click(await screen.findByRole("button", { name: "Saved chip" }));
  await waitFor(() =>
    expect(
      mocks.request.mock.calls.some(
        ([request]) => request.path === `/api/v1/notes/${otherId}`,
      ),
    ).toBe(true),
  );
  act(() => {
    if (change === "edit")
      useWorkspaceStore.getState().updateNoteContent("Later text");
    else changeContext(change);
  });
  const expected = useWorkspaceStore.getState().currentNote;
  await act(async () => {
    gate.resolve();
    await gate.promise;
  });
  expect(useWorkspaceStore.getState().currentNote).toBe(expected);
});

it.each([
  "server",
  "account",
  "workspace-return",
  "note-return",
  "clear-then-type",
  "edit",
])(
  "rejects a late explicit conflict reload after %s changes",
  async (change) => {
    useWorkspaceStore.setState({ currentNote: { ...draft(), version: 2 } });
    const gate = deferred();
    let readCount = 0;
    mocks.request.mockImplementation(async (request: BgRequestInit) => {
      if (request.path.includes("/search/")) return [];
      if (request.method === "PUT")
        throw { status: 409, message: "version conflict" };
      if (++readCount > 1) await gate.promise;
      return { ...draft(), content: "Remote newer", version: 3 };
    });
    render(<QuickNotesSection />);
    fireEvent.click(screen.getByRole("button", { name: "Update" }));
    await waitFor(() =>
      expect(
        mocks.open.mock.calls.some(
          ([entry]) => entry.key === "workspace-note-version-conflict",
        ),
      ).toBe(true),
    );
    const message = render(
      mocks.open.mock.calls.find(
        ([entry]) => entry.key === "workspace-note-version-conflict",
      )![0].content,
    );
    fireEvent.click(screen.getByRole("button", { name: "Reload latest" }));
    await waitFor(() => expect(readCount).toBe(2));
    act(() => {
      if (change === "edit")
        useWorkspaceStore.getState().updateNoteContent("Later text");
      else changeContext(change);
    });
    const expected = useWorkspaceStore.getState().currentNote;
    await act(async () => {
      gate.resolve();
      await gate.promise;
    });
    expect(useWorkspaceStore.getState().currentNote).toBe(expected);
    message.unmount();
  },
);

it.each(["cancel", "workspace", "account", "clear", "confirm"])(
  "keeps explicit local draft resume separate from a dirty replacement through %s",
  async (boundary) => {
    const { servicePromptAuthorityKey } =
      await import("@/services/tldw/domains/service-prompts");
    const { loadServicePromptSnapshot } =
      await import("@/services/service-prompts");
    const { checkpointQuickNotesOfflineDraft } =
      await import("@/components/Notes/notes-manager-utils");
    const scope = await loadServicePromptSnapshot([]);
    const authorityId = servicePromptAuthorityKey(scope.requestScope);
    scope.release();
    const owner = createNotesGraphAuthorityScope(mocks.serverUrl, mocks.userId);
    const entry = {
      key: 'surface:quick-notes:["workspace-a","workspace:a"]:retained-a',
      noteId: null,
      baseVersion: null,
      title: "Retained title",
      content: "Later retained text",
      keywords: [],
      metadata: {
        quickNotesAuthorityId: authorityId,
        quickNotesWorkspaceId: "workspace-a",
        quickNotesWorkspaceTag: "workspace:a",
      },
      backlinkConversationId: null,
      backlinkMessageId: null,
      updatedAt: "2026-10-08T00:00:00Z",
      syncState: "queued" as const,
      lastError: null,
      pendingWrite: {
        authorityId,
        key: "retained-key",
        body: {
          id: noteId,
          title: "Old",
          content: "Old body",
          keywords: ["workspace:a"],
        },
        expectedVersion: null,
      },
    };
    await retainSurfaceOfflineDraft(owner, entry);
    await checkpointQuickNotesOfflineDraft(
      owner,
      entry.key,
      "retained-key",
      () => null,
      { id: noteId, version: 2 },
    );
    localStorage.setItem(
      "tldw:research-workspace:migration:tombstone:workspace-a",
      JSON.stringify({
        legacyWorkspaceId: "workspace-a",
        serverWorkspaceId: "workspace-a",
        migrationId: "migration-a",
        contentRetained: false,
        deletedAt: "2026-10-08T00:00:00Z",
      }),
    );
    useWorkspaceStore.getState().clearCurrentNote();
    mocks.request.mockResolvedValue([]);
    let confirmation: Parameters<typeof Modal.confirm>[0] | undefined;
    vi.spyOn(Modal, "confirm").mockImplementation((config) => {
      confirmation = config;
      return { destroy: vi.fn(), update: vi.fn() };
    });
    render(<QuickNotesSection />);
    await screen.findByRole("button", { name: "Resume local unsaved draft" });
    fireEvent.change(screen.getByLabelText("Note content"), {
      target: { value: "Unrelated replacement" },
    });
    expect(useWorkspaceStore.getState().currentNote).toMatchObject({
      content: "Unrelated replacement",
      isDirty: true,
    });
    fireEvent.click(
      screen.getByRole("button", { name: "Resume local unsaved draft" }),
    );
    expect(confirmation?.title).toBe("Unsaved Changes");
    if (
      boundary === "workspace" ||
      boundary === "account" ||
      boundary === "clear"
    )
      act(() => changeContext(boundary));
    const expected = useWorkspaceStore.getState().currentNote;
    if (boundary !== "cancel")
      await act(async () => {
        await confirmation?.onOk?.();
      });
    if (boundary === "confirm")
      expect(useWorkspaceStore.getState().currentNote).toMatchObject({
        id: noteId,
        version: 2,
        content: "Later retained text",
        isDirty: true,
      });
    else expect(useWorkspaceStore.getState().currentNote).toBe(expected);
    expect(writes()).toHaveLength(0);
  },
);

it("reports missing record-lock persistence and dispatches no unsafe save", async () => {
  vi.stubGlobal(
    "navigator",
    Object.create(window.navigator, { locks: { value: undefined } }),
  );
  useWorkspaceStore.setState({ currentNote: { ...draft(), id: undefined } });
  mocks.request.mockResolvedValue([]);
  render(<QuickNotesSection />);
  fireEvent.click(screen.getByRole("button", { name: "Save" }));
  await waitFor(() =>
    expect(mocks.error).toHaveBeenCalledWith(
      expect.stringContaining("Could not retain the current local note draft"),
    ),
  );
  expect(writes()).toHaveLength(0);
});
