import React from "react"
import {
  act,
  cleanup,
  fireEvent,
  render,
  screen,
  waitFor,
  within
} from "@testing-library/react"
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import { initialState, useWorkspaceStore } from "@/store/workspace"
import type { OwnedWorkspaceBundle } from "@/store/workspace-api"
import {
  createOwnedWorkspaceDraftStore,
  ownedWorkspaceDraftKey
} from "@/store/owned-workspace-state"
import { QuickNotesSection } from "../StudioPane/QuickNotesSection"

const mocks = vi.hoisted(() => ({
  generic: vi.fn(),
  keywords: vi.fn(),
  context: vi.fn(),
  list: vi.fn(),
  create: vi.fn(),
  update: vi.fn(),
  success: vi.fn(),
  error: vi.fn(),
  warning: vi.fn(),
  open: vi.fn(),
  destroy: vi.fn(),
  translate: vi.fn()
}))
vi.mock("@/services/background-proxy", () => ({ bgRequest: mocks.generic }))
vi.mock("@/services/note-keywords", () => ({ getNoteKeywords: mocks.keywords }))
vi.mock("@/services/owned-workspace-opening", () => ({
  createOwnedWorkspaceNotesContext: mocks.context
}))
vi.mock("react-i18next", async () => {
  const { createInstance } = await import("i18next")
  const { default: playground } =
    await import("@/assets/locale/en/playground.json")
  const i18n = createInstance()
  await i18n.init({ lng: "en", resources: { en: { playground } } })
  mocks.translate.mockImplementation(i18n.t.bind(i18n))
  return { useTranslation: () => ({ t: mocks.translate }) }
})
vi.mock("@/components/Common/MarkdownPreview", () => ({
  MarkdownPreview: ({ content }: { content: string }) => <div>{content}</div>
}))
vi.mock("antd", async () => {
  const actual = await vi.importActual<typeof import("antd")>("antd")
  return {
    ...actual,
    message: { useMessage: () => [mocks, null] },
    Modal: Object.assign(actual.Modal, {
      confirm: ({ onOk }: { onOk: () => void }) => onOk()
    })
  }
})

const state = () => useWorkspaceStore.getState()
const scope = {
  serverBase: "https://research.test",
  principalId: "2",
  organizationId: null
}
const note = {
  id: 3,
  workspace_id: "owned",
  title: "Associated note",
  content: "Canonical content",
  keywords_json: '["workspace:tag","physics"]',
  version: 2,
  created_at: "2026-09-13",
  last_modified: "2026-09-13"
}
const bundle: OwnedWorkspaceBundle = {
  workspace: {
    id: "owned",
    name: "Owned",
    archived: false,
    deleted: false,
    version: 1,
    created_at: "2026-09-13",
    last_modified: "2026-09-13",
    study_materials_policy: "general",
    workspace_profile: "research",
    banner_title: null,
    banner_subtitle: null,
    banner_color: null,
    audio_provider: null,
    audio_model: null,
    audio_voice: null,
    audio_speed: null
  },
  sources: [],
  artifacts: [],
  notes: [note]
}
const deferred = <T,>() => {
  let resolve!: (value: T) => void
  let reject!: (reason: unknown) => void
  const promise = new Promise<T>((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}
const select = () =>
  fireEvent.click(
    within(screen.getByTestId("workspace-notes-list")).getByRole("button", {
      name: note.title
    })
  )
const save = () =>
  fireEvent.click(
    screen.getByRole("button", { name: /^(loading)?(Save|Update)$/ })
  )
const edit = (content: string) =>
  fireEvent.change(screen.getByLabelText("Note content"), {
    target: { value: content }
  })
const recovery = () => {
  const call = mocks.open.mock.calls.at(-1)![0]
  render(call.content)
}

beforeEach(() => {
  vi.clearAllMocks()
  localStorage.clear()
  useWorkspaceStore.setState({
    ...initialState,
    storeHydrated: true,
    workspaceId: "owned",
    workspaceTag: "workspace:tag",
    activeWorkspaceOrigin: { kind: "server-owned", scope },
    ownedWorkspaceBundle: bundle
  })
  mocks.generic.mockResolvedValue({
    notes: [
      { id: 99, title: "Unrelated same-tag", keywords: ["workspace:tag"] }
    ]
  })
  mocks.keywords.mockResolvedValue(["global-secret"])
  mocks.context.mockResolvedValue({
    list: mocks.list,
    create: mocks.create,
    update: mocks.update
  })
  mocks.list.mockResolvedValue([note])
  mocks.update.mockResolvedValue({ ...note, version: 3 })
  mocks.create.mockResolvedValue({ ...note, id: 4, keywords_json: "[]" })
})
afterEach(cleanup)

describe("owned QuickNotes", () => {
  it("dispatches creation with a durable marker despite another workspace's pending draft", async () => {
    const pendingId = "pending-quick-note"
    const currentId = "durable-quick-note"
    const original = Storage.prototype.setItem
    const spy = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (
          key.startsWith(`${ownedWorkspaceDraftKey(scope, pendingId)}:revisions:`)
        )
          throw new Error("Quota")
        original.call(this, key, value)
      })
    const pendingBundle = {
      ...bundle,
      workspace: { ...bundle.workspace, id: pendingId },
      notes: []
    }
    try {
      const pendingAttempt = state().beginOwnedWorkspace(scope, pendingId)
      expect(state().activateOwnedWorkspace(pendingAttempt, pendingBundle)).toBe(
        "activated"
      )
      state().updateNoteContent("Unrelated unsaved draft")
      const attempt = state().beginOwnedWorkspace(scope, currentId)
      expect(
        state().activateOwnedWorkspace(attempt, {
          ...bundle,
          workspace: { ...bundle.workspace, id: currentId },
          notes: []
        })
      ).toBe("activated")
      spy.mockRestore()
      mocks.create.mockImplementation(async () => {
        const stored = createOwnedWorkspaceDraftStore(() => localStorage).load(
          scope,
          currentId
        )
        expect(stored.status).toBe("ready")
        if (stored.status !== "ready") throw new Error("Missing durable marker")
        expect(stored.durable).toBe(true)
        expect(stored.draft.currentNote.createUncertain).toBe(true)
        return { ...note, id: 4, workspace_id: currentId, keywords_json: "[]" }
      })
      render(<QuickNotesSection />)
      edit("Create in the current workspace")
      save()
      await waitFor(() => expect(state().currentNote.id).toBe(4))
      expect(mocks.create).toHaveBeenCalledTimes(1)
      expect(state().ownedWorkspaceDraftStatus).toBe("unavailable")
    } finally {
      spy.mockRestore()
      act(() => {
        const attempt = state().beginOwnedWorkspace(scope, pendingId)
        state().activateOwnedWorkspace(attempt, pendingBundle)
        state().updateNoteContent("Unrelated unsaved draft")
      })
    }
  })

  it("mirrors every owned-note recovery message into the extension locale", async () => {
    const { default: source } =
      await import("@/assets/locale/en/playground.json")
    const { default: extension } =
      await import("@/public/_locales/en/playground.json")
    const messages: Record<string, { message: string }> = extension
    for (const [key, message] of Object.entries(source.studio.ownedNotes)) {
      expect(messages[`studio_ownedNotes_${key}`], key).toEqual({ message })
    }
  })

  it("uses English locale resources for account recovery and refresh controls", async () => {
    mocks.context.mockRejectedValue({ status: 412 })
    render(<QuickNotesSection />)
    edit("Localized recovery")
    save()
    await waitFor(() => expect(mocks.error).toHaveBeenCalled())
    expect(mocks.translate).toHaveBeenCalledWith(
      "playground:studio.ownedNotes.accountChanged"
    )
    expect(mocks.error).toHaveBeenCalledWith(
      "Workspace account changed. Reopen the workspace before saving. Your local draft is unchanged."
    )
    fireEvent.click(screen.getByRole("button", { name: "Load note" }))
    expect(
      screen.getByRole("button", { name: "Refresh notes" })
    ).toBeInTheDocument()
    expect(mocks.translate).toHaveBeenCalledWith(
      "playground:studio.ownedNotes.refresh"
    )
  })

  it("uses canonical association without mount writes or generic reads, including no tag", async () => {
    useWorkspaceStore.setState({ workspaceTag: "" })
    render(<QuickNotesSection />)
    expect(screen.getByTestId("workspace-notes-list")).toHaveTextContent(
      note.title
    )
    select()
    expect(screen.getByLabelText("Note content")).toHaveValue(note.content)
    expect(state().currentNote.version).toBe(2)
    expect(state().currentNote.keywords).toEqual(["workspace:tag", "physics"])
    fireEvent.click(screen.getByRole("button", { name: "Load note" }))
    expect(screen.queryByText("Unrelated same-tag")).not.toBeInTheDocument()
    expect(mocks.generic).not.toHaveBeenCalled()
    expect(mocks.keywords).not.toHaveBeenCalled()
    expect(mocks.context).not.toHaveBeenCalled()
  })

  it("saves with required version and explicit empty keywords, preserving newer typing", async () => {
    const pending = deferred<typeof note>()
    mocks.update.mockReturnValue(pending.promise)
    render(<QuickNotesSection />)
    select()
    fireEvent.change(screen.getByLabelText("Note keywords"), {
      target: { value: "" }
    })
    edit("Submitted")
    save()
    save()
    await waitFor(() => expect(mocks.update).toHaveBeenCalledTimes(1))
    expect(mocks.update).toHaveBeenCalledWith(3, {
      title: note.title,
      content: "Submitted",
      keywords_json: "[]",
      version: 2
    })
    edit("Newer draft")
    await act(async () =>
      pending.resolve({
        ...note,
        content: "Submitted",
        keywords_json: "[]",
        version: 3
      })
    )
    expect(state().currentNote).toMatchObject({
      content: "Newer draft",
      isDirty: true,
      version: 3
    })
    expect(state().ownedWorkspaceBundle?.notes[0].content).toBe("Submitted")
    expect(mocks.success).not.toHaveBeenCalled()
  })

  it("creates a new note with empty keywords and no synthetic tag", async () => {
    render(<QuickNotesSection />)
    edit("New note")
    save()
    await waitFor(() =>
      expect(mocks.create).toHaveBeenCalledWith({
        title: "Untitled Note",
        content: "New note",
        keywords: []
      })
    )
    await waitFor(() => expect(state().currentNote.id).toBe(4))
    expect(state().currentNote.isDirty).toBe(false)
    expect(state().currentNote.createUncertain).toBeUndefined()
  })

  it("does not restore a cleared editor when an update completes", async () => {
    const pending = deferred<typeof note>()
    mocks.update.mockReturnValue(pending.promise)
    render(<QuickNotesSection />)
    select()
    edit("Saving")
    save()
    await waitFor(() => expect(mocks.update).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Clear current note" }))
    await act(async () => pending.resolve({ ...note, version: 3 }))
    expect(state().currentNote.content).toBe("")
    expect(state().ownedWorkspaceBundle?.notes[0].version).toBe(3)
    expect(mocks.success).not.toHaveBeenCalled()
  })

  it("aborts a same-ID A-B-A origin lifetime and ignores its late completion", async () => {
    const pending = deferred<typeof note>()
    mocks.update.mockReturnValue(pending.promise)
    render(<QuickNotesSection />)
    select()
    edit("Saving")
    save()
    await waitFor(() => expect(mocks.update).toHaveBeenCalled())
    const origin = state().activeWorkspaceOrigin
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: {
          kind: "server-owned",
          scope: { ...scope, principalId: "other" }
        }
      })
      useWorkspaceStore.setState({ activeWorkspaceOrigin: origin })
    })
    expect(mocks.context.mock.calls[0][2].aborted).toBe(true)
    await act(async () => pending.resolve({ ...note, version: 3 }))
    expect(state().currentNote.version).toBe(2)
    expect(mocks.success).not.toHaveBeenCalled()
  })

  it.each([409, 412])(
    "preserves the draft on %s and never reads generic notes",
    async (status) => {
      mocks.update.mockRejectedValue({ status })
      render(<QuickNotesSection />)
      select()
      edit("Local draft")
      save()
      await waitFor(() =>
        expect(
          mocks.open.mock.calls.length + mocks.error.mock.calls.length
        ).toBeGreaterThan(0)
      )
      expect(state().currentNote).toMatchObject({
        content: "Local draft",
        isDirty: true,
        version: 2
      })
      expect(mocks.generic).not.toHaveBeenCalled()
      expect(mocks.success).not.toHaveBeenCalled()
    }
  )

  it("recovers conflicts with a scoped list and retains edits made during recovery", async () => {
    mocks.update.mockRejectedValue({ status: 409 })
    const pending = deferred<(typeof note)[]>()
    mocks.list.mockReturnValue(pending.promise)
    render(<QuickNotesSection />)
    select()
    edit("Conflict draft")
    save()
    await waitFor(() => expect(mocks.open).toHaveBeenCalled())
    recovery()
    fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
    await waitFor(() => expect(mocks.list).toHaveBeenCalled())
    edit("  Latest local typing\n\n")
    await act(async () =>
      pending.resolve([{ ...note, content: "Remote", version: 4 }])
    )
    expect(state().currentNote).toMatchObject({ version: 4, isDirty: true })
    expect(state().currentNote.content).toContain("Remote")
    expect(state().currentNote.content).toContain(
      "## Local Draft (Unsaved)\n\n  Latest local typing\n\n"
    )
    save()
    await waitFor(() => expect(mocks.update).toHaveBeenCalledTimes(2))
    expect(mocks.update.mock.calls[1][1].content).toContain("Remote")
    expect(mocks.generic).not.toHaveBeenCalled()
  })

  it("blocks blind repeated creates after ambiguous failure and offers inspection", async () => {
    mocks.create.mockRejectedValue(new Error("Network disconnected"))
    render(<QuickNotesSection />)
    edit("Maybe saved")
    save()
    await waitFor(() => expect(mocks.open).toHaveBeenCalled())
    save()
    expect(mocks.create).toHaveBeenCalledTimes(1)
    recovery()
    fireEvent.click(screen.getByRole("button", { name: "Refresh notes" }))
    await waitFor(() => expect(mocks.list).toHaveBeenCalledTimes(1))
    expect(state().currentNote.content).toBe("Maybe saved")
  })

  it.each(["missing", "error"])(
    "reports %s conflict recovery without changing the draft",
    async (failure) => {
      mocks.update.mockRejectedValue({ status: 409 })
      if (failure === "missing") mocks.list.mockResolvedValue([])
      else mocks.list.mockRejectedValue(new Error("List failed"))
      render(<QuickNotesSection />)
      select()
      edit("Keep this draft")
      save()
      await waitFor(() => expect(mocks.open).toHaveBeenCalled())
      recovery()
      fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
      await waitFor(() => expect(mocks.error).toHaveBeenCalled())
      expect(state().currentNote).toMatchObject({
        content: "Keep this draft",
        version: 2,
        isDirty: true
      })
      expect(mocks.success).not.toHaveBeenCalled()
    }
  )

  it("does not revive an editor cleared during conflict recovery", async () => {
    mocks.update.mockRejectedValue({ status: 409 })
    const pending = deferred<(typeof note)[]>()
    mocks.list.mockReturnValue(pending.promise)
    render(<QuickNotesSection />)
    select()
    edit("Old draft")
    save()
    await waitFor(() => expect(mocks.open).toHaveBeenCalled())
    recovery()
    fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
    await waitFor(() => expect(mocks.list).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Clear current note" }))
    await act(async () => pending.resolve([{ ...note, version: 4 }]))
    expect(state().currentNote.content).toBe("")
    expect(state().currentNote.id).toBeUndefined()
  })

  it.each([false, true])(
    "scoped clear undo respects newer edits: %s",
    (newer) => {
      render(<QuickNotesSection />)
      select()
      edit("Restore me")
      fireEvent.click(
        screen.getByRole("button", { name: "Clear current note" })
      )
      recovery()
      if (newer) edit("New draft")
      fireEvent.click(screen.getByRole("button", { name: "Undo" }))
      expect(state().currentNote.content).toBe(
        newer ? "New draft" : "Restore me"
      )
      expect(mocks.generic).not.toHaveBeenCalled()
    }
  )

  it("ignores saved recovery handlers after switching accounts", async () => {
    mocks.update.mockRejectedValue({ status: 409 })
    render(<QuickNotesSection />)
    select()
    edit("Old account")
    save()
    await waitFor(() => expect(mocks.open).toHaveBeenCalled())
    recovery()
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: {
          kind: "server-owned",
          scope: { ...scope, principalId: "other" }
        }
      })
    })
    fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
    expect(mocks.list).not.toHaveBeenCalled()
    expect(mocks.context).toHaveBeenCalledTimes(1)
  })

  it("rejects an old recovery handler after same-scope origin A-B-A transitions", async () => {
    mocks.update.mockRejectedValue({ status: 409 })
    render(<QuickNotesSection />)
    select()
    edit("Old activation")
    save()
    await waitFor(() => expect(mocks.open).toHaveBeenCalled())
    recovery()
    const origin = state().activeWorkspaceOrigin
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: { kind: "server-owned", scope }
      })
    })
    act(() => {
      useWorkspaceStore.setState({ activeWorkspaceOrigin: origin })
    })
    fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
    await act(async () => {})
    expect(mocks.list).not.toHaveBeenCalled()
  })

  it("aborts on unmount and discards late success", async () => {
    const pending = deferred<typeof note>()
    mocks.update.mockReturnValue(pending.promise)
    const view = render(<QuickNotesSection />)
    select()
    edit("Pending")
    save()
    await waitFor(() => expect(mocks.update).toHaveBeenCalled())
    view.unmount()
    expect(mocks.context.mock.calls[0][2].aborted).toBe(true)
    await act(async () => pending.resolve({ ...note, version: 3 }))
    expect(state().currentNote.version).toBe(2)
    expect(mocks.success).not.toHaveBeenCalled()
  })

  it.each(["remount", "undo"])(
    "retains uncertain create protection after %s",
    async (action) => {
      mocks.create.mockRejectedValue(new Error("Network disconnected"))
      const view = render(<QuickNotesSection />)
      edit("Maybe created")
      save()
      await waitFor(() => expect(mocks.open).toHaveBeenCalled())
      if (action === "remount") {
        view.unmount()
        render(<QuickNotesSection />)
      } else {
        fireEvent.click(
          screen.getByRole("button", { name: "Clear current note" })
        )
        recovery()
        fireEvent.click(screen.getByRole("button", { name: "Undo" }))
      }
      save()
      await act(async () => {})
      expect(mocks.create).toHaveBeenCalledTimes(1)
    }
  )

  it("reports a pre-dispatch account failure without claiming a possible create", async () => {
    mocks.context.mockRejectedValue({ status: 412 })
    render(<QuickNotesSection />)
    edit("Not dispatched")
    save()
    await waitFor(() => expect(mocks.error).toHaveBeenCalled())
    expect(mocks.error).toHaveBeenCalledWith(
      expect.stringContaining("account changed")
    )
    expect(mocks.create).not.toHaveBeenCalled()
    expect(mocks.open).not.toHaveBeenCalled()
  })

  it("does not show an uncertain-create action for an editor that was cleared", async () => {
    const pending = deferred<typeof note>()
    mocks.create.mockReturnValue(pending.promise)
    render(<QuickNotesSection />)
    edit("Pending create")
    save()
    await waitFor(() => expect(mocks.create).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Clear current note" }))
    mocks.open.mockClear()
    await act(async () => pending.reject(new Error("Unknown outcome")))
    expect(mocks.open).not.toHaveBeenCalled()
    expect(state().currentNote.content).toBe("")
  })

  it("expires owned undo without restoring a cleared draft", () => {
    render(<QuickNotesSection />)
    select()
    fireEvent.click(screen.getByRole("button", { name: "Clear current note" }))
    recovery()
    const clock = vi.spyOn(Date, "now").mockReturnValue(Date.now() + 60_000)
    try {
      fireEvent.click(screen.getByRole("button", { name: "Undo" }))
      expect(state().currentNote.content).toBe("")
    } finally {
      clock.mockRestore()
    }
  })

  it("does not apply a late legacy save after entering an owned workspace", async () => {
    const pending = deferred<typeof note>()
    mocks.generic.mockImplementation((request: { method: string }) =>
      request.method === "POST"
        ? pending.promise
        : Promise.resolve({ notes: [] })
    )
    useWorkspaceStore.setState({
      activeWorkspaceOrigin: { kind: "legacy-local" },
      ownedWorkspaceBundle: null
    })
    render(<QuickNotesSection />)
    edit("Legacy save")
    save()
    await waitFor(() =>
      expect(mocks.generic).toHaveBeenCalledWith(
        expect.objectContaining({ method: "POST" })
      )
    )
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: { kind: "server-owned", scope },
        ownedWorkspaceBundle: bundle
      })
      state().setCurrentNote({
        title: "Owned draft",
        content: "Owned content",
        keywords: [],
        isDirty: true
      })
    })
    await act(async () => pending.resolve({ ...note, content: "Legacy save" }))
    expect(state().currentNote.content).toBe("Owned content")
    expect(mocks.success).not.toHaveBeenCalled()
  })

  it.each([400, 401, 403, 404, 409, 412, 422, 429])(
    "reports definite create rejection %s without an uncertainty lock",
    async (status) => {
      mocks.create.mockRejectedValue({ status })
      render(<QuickNotesSection />)
      edit("Rejected create")
      save()
      await waitFor(() => expect(mocks.error).toHaveBeenCalled())
      expect(mocks.open).not.toHaveBeenCalled()
      if (status === 412)
        expect(mocks.error).toHaveBeenCalledWith(
          expect.stringContaining("account changed")
        )
      save()
      await waitFor(() => expect(mocks.create).toHaveBeenCalledTimes(2))
    }
  )

  it.each(["select", "reload", "search"])(
    "fences late legacy %s completion and further requests",
    async (operation) => {
      const pending = deferred<unknown>()
      mocks.generic.mockImplementation(
        (request: { path: string; method: string }) => {
          if (request.method === "PUT") return Promise.reject({ status: 409 })
          if (request.path === "/api/v1/notes/3") return pending.promise
          if (operation === "search" && request.path.includes("/notes/?"))
            return pending.promise
          return Promise.resolve({ notes: [note] })
        }
      )
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: { kind: "legacy-local" },
        ownedWorkspaceBundle: null
      })
      render(<QuickNotesSection />)
      await screen.findByTestId("workspace-notes-list")
      if (operation === "select") select()
      else if (operation === "reload") {
        act(() => {
          state().loadNote({
            id: 3,
            title: "Legacy",
            content: "Legacy",
            keywords: [],
            version: 2
          })
        })
        edit("Legacy draft")
        save()
        await waitFor(() => expect(mocks.open).toHaveBeenCalled())
        recovery()
        fireEvent.click(screen.getByRole("button", { name: "Reload latest" }))
      } else fireEvent.click(screen.getByRole("button", { name: "Load note" }))
      const count = mocks.generic.mock.calls.length
      act(() => {
        useWorkspaceStore.setState({
          activeWorkspaceOrigin: { kind: "server-owned", scope },
          ownedWorkspaceBundle: bundle
        })
        state().setCurrentNote({
          title: "Owned draft",
          content: "Owned content",
          keywords: [],
          isDirty: true
        })
      })
      await act(async () =>
        pending.resolve(operation === "search" ? { notes: [note] } : note)
      )
      expect(state().currentNote.content).toBe("Owned content")
      expect(mocks.generic).toHaveBeenCalledTimes(count)
      expect(mocks.success).not.toHaveBeenCalled()
    }
  )

  it("persists uncertainty before dispatch and blocks a recovered draft in a new activation", async () => {
    const pending = deferred<typeof note>()
    let observed:
      | {
          marked?: boolean
          persisted: { currentNote?: { createUncertain?: boolean } } | null
        }
      | undefined
    mocks.create.mockImplementation(() => {
      const loaded = createOwnedWorkspaceDraftStore(() => localStorage).load(
        scope,
        "owned"
      )
      expect(loaded).toMatchObject({ status: "ready", durable: true })
      if (loaded.status !== "ready") throw new Error("Missing durable draft")
      observed = {
        marked: state().currentNote.createUncertain,
        persisted: loaded.draft
      }
      return pending.promise
    })
    const view = render(<QuickNotesSection />)
    edit("Unknown save")
    save()
    await waitFor(() => expect(mocks.create).toHaveBeenCalled())
    await act(async () => pending.reject(new Error("Disconnected")))
    expect(observed?.marked).toBe(true)
    expect(observed?.persisted?.currentNote?.createUncertain).toBe(true)
    const recovered = JSON.parse(JSON.stringify(state().currentNote))
    view.unmount()
    useWorkspaceStore.setState({
      ...initialState,
      storeHydrated: true,
      workspaceId: "owned",
      ownedWorkspaceBundle: bundle,
      activeWorkspaceOrigin: { kind: "server-owned", scope },
      currentNote: recovered
    })
    render(<QuickNotesSection />)
    save()
    await act(async () => {})
    expect(mocks.create).toHaveBeenCalledTimes(1)
  })

  it("does not dispatch create when uncertainty cannot be persisted", async () => {
    render(<QuickNotesSection />)
    edit("Quota protected")
    const original = Storage.prototype.setItem
    const storage = vi
      .spyOn(Storage.prototype, "setItem")
      .mockImplementation(function (key, value) {
        if (
          key.startsWith(`${ownedWorkspaceDraftKey(scope, "owned")}:revisions:`)
        )
          throw new Error("Quota")
        original.call(this, key, value)
      })
    try {
      save()
      await waitFor(() => expect(mocks.error).toHaveBeenCalled())
      expect(mocks.create).not.toHaveBeenCalled()
      expect(state().currentNote.content).toBe("Quota protected")
    } finally {
      storage.mockRestore()
    }
  })

  it("does not create a cleared editor after client setup finishes", async () => {
    const pending = deferred<{
      list: typeof mocks.list
      create: typeof mocks.create
      update: typeof mocks.update
    }>()
    mocks.context.mockReturnValue(pending.promise)
    render(<QuickNotesSection />)
    edit("Preparing create")
    save()
    await waitFor(() => expect(mocks.context).toHaveBeenCalled())
    fireEvent.click(screen.getByRole("button", { name: "Clear current note" }))
    await act(async () =>
      pending.resolve({
        list: mocks.list,
        create: mocks.create,
        update: mocks.update
      })
    )
    expect(mocks.create).not.toHaveBeenCalled()
    expect(state().currentNote.content).toBe("")
  })

  it("does not revive a legacy completion after origin A-B-A", async () => {
    const pending = deferred<typeof note>()
    mocks.generic.mockImplementation((request: { method: string }) =>
      request.method === "POST"
        ? pending.promise
        : Promise.resolve({ notes: [] })
    )
    const origin = { kind: "legacy-local" as const }
    useWorkspaceStore.setState({
      activeWorkspaceOrigin: origin,
      ownedWorkspaceBundle: null
    })
    render(<QuickNotesSection />)
    edit("Legacy save")
    save()
    await waitFor(() =>
      expect(mocks.generic).toHaveBeenCalledWith(
        expect.objectContaining({ method: "POST" })
      )
    )
    act(() => {
      useWorkspaceStore.setState({
        activeWorkspaceOrigin: { kind: "legacy-local" }
      })
    })
    act(() => {
      useWorkspaceStore.setState({ activeWorkspaceOrigin: origin })
    })
    edit("New draft")
    await act(async () => pending.resolve({ ...note, content: "Old save" }))
    expect(state().currentNote.content).toBe("New draft")
  })
})
