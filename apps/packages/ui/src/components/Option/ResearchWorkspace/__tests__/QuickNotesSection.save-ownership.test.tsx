import React from "react"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"
import { useWorkspaceStore } from "@/store/workspace"
import type { WorkspaceNote } from "@/types/workspace"

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
  mocks.request.mockImplementation(async (request: any) => {
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
