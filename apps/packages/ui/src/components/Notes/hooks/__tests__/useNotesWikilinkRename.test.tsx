import React from "react"
import { App, ConfigProvider } from "antd"
import { act, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import { useAntdMessage } from "@/hooks/useAntdMessage"
import { createNotesGraphAuthorityScope } from "../useNotesGraphAuthorityScope"
import {
  useNotesWikilinkRename,
  type NoteRenamedEvent,
  type UseNotesWikilinkRenameDeps
} from "../useNotesWikilinkRename"

// Renaming a note leaves [[Old title]] links in other notes unresolved
// (#3110). The owner chose to OFFER the update: nothing is rewritten unless
// the user confirms, and the rewrite can be undone.

const { mockBgRequest, mockGetCurrentUser } = vi.hoisted(() => ({
  mockBgRequest: vi.fn(),
  mockGetCurrentUser: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({ bgRequest: mockBgRequest }))
vi.mock("@/services/tldw/TldwAuth", () => ({
  tldwAuth: { getCurrentUser: mockGetCurrentUser }
}))
vi.mock("react-i18next", () => ({
  useTranslation: () => ({
    t: (key: string, fallback?: unknown) => (typeof fallback === "string" ? fallback : key)
  })
}))

const SERVER_URL = "https://notes.example.test"
const CONFIG = { serverUrl: SERVER_URL, authMode: "multi-user" as const, accessToken: "token" }
const SCOPE = createNotesGraphAuthorityScope(SERVER_URL, 1)
const REFERRERS = "/api/v1/notes/wikilinks/referrers"
const REWRITE = "/api/v1/notes/wikilinks/rewrite"
const UNDO = "/api/v1/notes/wikilinks/rewrite/undo"
const RENAME: NoteRenamedEvent = { noteId: "note-a", oldTitle: "Old title", newTitle: "New title" }

const translate = (key: string, options?: Record<string, unknown>) =>
  String(options?.defaultValue ?? key).replace(/\{\{\s*(\w+)\s*\}\}/g, (_match, name: string) =>
    String(options?.[name] ?? "")
  )

type ServerRequest = {
  path: string
  method?: string
  body?: Record<string, unknown>
  headers?: Record<string, string>
  servicePromptConfig?: Record<string, unknown>
}
type Handlers = Partial<Record<string, (request: ServerRequest) => unknown>>

const reloadSelectedNote = vi.fn()
const onLinksChanged = vi.fn()
const harness: { handleNoteRenamed: ((event: NoteRenamedEvent) => Promise<void>) | null } = {
  handleNoteRenamed: null
}

function Harness({ overrides }: { overrides?: Partial<UseNotesWikilinkRenameDeps> }) {
  const message = useAntdMessage()
  const rename = useNotesWikilinkRename({
    isOnline: true,
    authorityScope: SCOPE,
    connectionConfig: CONFIG,
    message,
    t: translate,
    selectedId: "note-a",
    isDirty: false,
    reloadSelectedNote,
    onLinksChanged,
    ...overrides
  })
  harness.handleNoteRenamed = rename.handleNoteRenamed
  return null
}

// Without motion a closed prompt leaves the DOM at once; jsdom never ends a CSS transition.
const tree = (overrides?: Partial<UseNotesWikilinkRenameDeps>) => (
  <ConfigProvider theme={{ token: { motion: false } }}>
    <App>
      <Harness overrides={overrides} />
    </App>
  </ConfigProvider>
)

const renderHarness = (overrides?: Partial<UseNotesWikilinkRenameDeps>) => render(tree(overrides))

const requestsTo = (path: string): ServerRequest[] =>
  mockBgRequest.mock.calls.map(([request]) => request as ServerRequest).filter((request) => request.path === path)

const twoReferrers = {
  title: "Old title",
  count: 2,
  notes: [
    { id: "linker-a", title: "Linker A", version: 4 },
    { id: "linker-b", title: "Linker B", version: 2 }
  ],
  next_after_note_id: null
}

const updatedResult = (id: string, title: string, version: number) => ({
  id,
  title,
  status: "updated",
  version,
  replaced_count: 1,
  replacements: [{ token_index: 0, original: "[[Old title]]" }]
})

const rewriteResponse = (results: Array<Record<string, unknown>>, overrides: Record<string, unknown> = {}) => ({
  old_title: "Old title",
  new_title: "New title",
  link_form: "title",
  replacement: "[[New title]]",
  new_title_shared: false,
  updated_count: results.filter((result) => result.status === "updated").length,
  skipped_count: results.filter((result) => result.status !== "updated").length,
  results,
  ...overrides
})

const useServer = (handlers: Handlers) => {
  mockBgRequest.mockImplementation(async (request: ServerRequest) => {
    const handler = handlers[request.path]
    if (!handler) throw new Error(`Unexpected request: ${request.path}`)
    return handler(request)
  })
}

const rename = async (event: NoteRenamedEvent = RENAME) => {
  await act(async () => {
    await harness.handleNoteRenamed?.(event)
  })
}

const confirmUpdate = async () => {
  fireEvent.click(await screen.findByRole("button", { name: "Update links" }))
}

describe("useNotesWikilinkRename", { timeout: 30_000 }, () => {
  beforeEach(() => {
    vi.clearAllMocks()
    harness.handleNoteRenamed = null
    mockGetCurrentUser.mockResolvedValue({ id: 1, is_active: true })
  })

  it("asks which notes the rename left with a broken link, as the verified owner", async () => {
    useServer({ [REFERRERS]: () => ({ title: "Old title", count: 0, notes: [], next_after_note_id: null }) })
    renderHarness()

    await rename()

    const [request] = requestsTo(REFERRERS)
    expect(request.method).toBe("POST")
    expect(request.body).toEqual({
      title: "Old title",
      exclude_note_id: "note-a",
      unresolved_only: true
    })
    expect(request.headers?.["X-TLDW-Expected-User-ID"]).toBe("1")
    expect(request.servicePromptConfig).toMatchObject({ serverUrl: SERVER_URL, expectedUserId: 1 })
  })

  it("shows no prompt when no other note links to the old title", async () => {
    useServer({ [REFERRERS]: () => ({ title: "Old title", count: 0, notes: [], next_after_note_id: null }) })
    renderHarness()

    await rename()

    expect(requestsTo(REFERRERS)).toHaveLength(1)
    expect(screen.queryByRole("button", { name: "Update links" })).not.toBeInTheDocument()
    expect(screen.queryByText(/link to "Old title"/)).not.toBeInTheDocument()
  })

  it("offers to update the links when other notes link to the old title", async () => {
    useServer({ [REFERRERS]: () => twoReferrers })
    renderHarness()

    await rename()

    expect(await screen.findByText('2 notes link to "Old title"')).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Update links" })).toBeInTheDocument()
    // Offering is all it does: nothing is rewritten until the user confirms.
    expect(requestsTo(REWRITE)).toHaveLength(0)
  })

  it("uses the singular for one linking note", async () => {
    useServer({
      [REFERRERS]: () => ({ ...twoReferrers, count: 1, notes: twoReferrers.notes.slice(0, 1) })
    })
    renderHarness()

    await rename()

    expect(await screen.findByText('1 note links to "Old title"')).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Update link" })).toBeInTheDocument()
  })

  it.each([
    ["the title only changed case or spacing", { ...RENAME, newTitle: "old   TITLE" }, {}],
    ["there was no previous title", { ...RENAME, oldTitle: "  " }, {}],
    ["the server is offline", RENAME, { isOnline: false }],
    ["the notes owner is not verified", RENAME, { authorityScope: null }]
  ])("does not ask when %s", async (_name, event, overrides) => {
    useServer({})
    renderHarness(overrides)

    await rename(event)

    expect(mockBgRequest).not.toHaveBeenCalled()
    expect(screen.queryByRole("button", { name: "Update links" })).not.toBeInTheDocument()
  })

  it("does not ask when the signed-in account is no longer the notes owner", async () => {
    mockGetCurrentUser.mockResolvedValue({ id: 2, is_active: true })
    useServer({})
    renderHarness()

    await rename()

    expect(mockBgRequest).not.toHaveBeenCalled()
  })

  it("rewrites the counted notes on confirm and shows Undo, naming the notes it skipped", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () =>
        rewriteResponse([
          updatedResult("linker-a", "Linker A", 5),
          { id: "linker-b", title: "Linker B", status: "skipped_conflict", version: 3, replaced_count: 0, replacements: [] }
        ])
    })
    renderHarness()
    await rename()

    await confirmUpdate()

    expect(await screen.findByText("Updated links in 1 note")).toBeInTheDocument()
    expect(screen.getByText(/Skipped: Linker B \(edited since\)/)).toBeInTheDocument()
    expect(screen.getByRole("button", { name: "Undo" })).toBeInTheDocument()
    const [request] = requestsTo(REWRITE)
    expect(request.method).toBe("POST")
    expect(request.body).toEqual({
      note_id: "note-a",
      old_title: "Old title",
      notes: [
        { id: "linker-a", expected_version: 4 },
        { id: "linker-b", expected_version: 2 }
      ]
    })
    expect(request.headers?.["X-TLDW-Expected-User-ID"]).toBe("1")
    expect(onLinksChanged).toHaveBeenCalledTimes(1)
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Update links" })).not.toBeInTheDocument()
    })
  })

  it("restores the previous text on Undo with the rewrite's own undo data", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () =>
        rewriteResponse([updatedResult("linker-a", "Linker A", 5), updatedResult("linker-b", "Linker B", 3)]),
      [UNDO]: () => ({
        restored_count: 1,
        skipped_count: 1,
        results: [
          { id: "linker-a", title: "Linker A", status: "restored", version: 6 },
          { id: "linker-b", title: "Linker B", status: "skipped_conflict", version: 4 }
        ]
      })
    })
    renderHarness()
    await rename()
    await confirmUpdate()

    fireEvent.click(await screen.findByRole("button", { name: "Undo" }))

    expect(await screen.findByText("Restored successfully")).toBeInTheDocument()
    expect(await screen.findByText(/Links were not restored in: Linker B \(edited since\)/)).toBeInTheDocument()
    const [request] = requestsTo(UNDO)
    expect(request.body).toEqual({
      old_title: "Old title",
      replacement: "[[New title]]",
      notes: [
        { id: "linker-a", expected_version: 5, replacements: [{ token_index: 0, original: "[[Old title]]" }] },
        { id: "linker-b", expected_version: 3, replacements: [{ token_index: 0, original: "[[Old title]]" }] }
      ]
    })
    expect(onLinksChanged).toHaveBeenCalledTimes(2)
  })

  it("reports a failed Undo instead of claiming the text was restored", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () => rewriteResponse([updatedResult("linker-a", "Linker A", 5)]),
      [UNDO]: () => ({
        restored_count: 0,
        skipped_count: 1,
        results: [{ id: "linker-a", title: "Linker A", status: "skipped_conflict", version: 6 }]
      })
    })
    renderHarness()
    await rename()
    await confirmUpdate()

    fireEvent.click(await screen.findByRole("button", { name: "Undo" }))

    expect(await screen.findByText("Failed to restore")).toBeInTheDocument()
    expect(screen.getByText(/Links were not restored in: Linker A \(edited since\)/)).toBeInTheDocument()
    expect(screen.queryByText("Restored successfully")).not.toBeInTheDocument()
  })

  it("leaves out a note that is open with unsaved edits, and says so", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () => rewriteResponse([updatedResult("linker-a", "Linker A", 5)])
    })
    renderHarness({ selectedId: "linker-b", isDirty: true })
    await rename()

    await confirmUpdate()

    expect(await screen.findByText("Updated links in 1 note")).toBeInTheDocument()
    expect(screen.getByText(/Skipped: Linker B \(open with unsaved changes\)/)).toBeInTheDocument()
    expect(requestsTo(REWRITE)[0].body?.notes).toEqual([{ id: "linker-a", expected_version: 4 }])
    expect(reloadSelectedNote).not.toHaveBeenCalled()
  })

  it("reloads the open note when its saved text was rewritten", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () =>
        rewriteResponse([updatedResult("linker-a", "Linker A", 5), updatedResult("linker-b", "Linker B", 3)])
    })
    renderHarness({ selectedId: "linker-b", isDirty: false })
    await rename()

    await confirmUpdate()

    expect(await screen.findByText("Updated links in 2 notes")).toBeInTheDocument()
    expect(reloadSelectedNote).toHaveBeenCalledTimes(1)
  })

  it("follows the cursor so every linking note is rewritten, and Undo covers them all", async () => {
    const laterPage = {
      title: "Old title",
      count: 3,
      notes: [{ id: "linker-c", title: "Linker C", version: 9 }],
      next_after_note_id: null
    }
    useServer({
      [REFERRERS]: (request) =>
        request.body?.after_note_id ? laterPage : { ...twoReferrers, count: 3, next_after_note_id: "linker-b" },
      [REWRITE]: (request) =>
        rewriteResponse(
          (request.body?.notes as Array<{ id: string; expected_version: number }>).map((note) =>
            updatedResult(note.id, note.id, note.expected_version + 1)
          )
        ),
      [UNDO]: (request) => ({
        restored_count: (request.body?.notes as unknown[]).length,
        skipped_count: 0,
        results: (request.body?.notes as Array<{ id: string }>).map((note) => ({
          id: note.id,
          title: note.id,
          status: "restored",
          version: 99
        }))
      })
    })
    renderHarness()
    await rename()
    expect(await screen.findByText('3 notes link to "Old title"')).toBeInTheDocument()

    await confirmUpdate()

    expect(await screen.findByText("Updated links in 3 notes")).toBeInTheDocument()
    expect(requestsTo(REFERRERS)[1].body).toEqual({
      title: "Old title",
      exclude_note_id: "note-a",
      unresolved_only: true,
      after_note_id: "linker-b"
    })
    expect(requestsTo(REWRITE).map((request) => request.body?.notes)).toEqual([
      [
        { id: "linker-a", expected_version: 4 },
        { id: "linker-b", expected_version: 2 }
      ],
      [{ id: "linker-c", expected_version: 9 }]
    ])

    fireEvent.click(screen.getByRole("button", { name: "Undo" }))

    expect(await screen.findByText("Restored successfully")).toBeInTheDocument()
    expect(
      requestsTo(UNDO).flatMap((request) => (request.body?.notes as Array<{ id: string }>).map((note) => note.id))
    ).toEqual(["linker-a", "linker-b", "linker-c"])
  })

  it("says when the links were written by id because the new title is shared", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () =>
        rewriteResponse([updatedResult("linker-a", "Linker A", 5), updatedResult("linker-b", "Linker B", 3)], {
          link_form: "id",
          replacement: "[[id:11111111-1111-4111-8111-111111111111]]",
          new_title_shared: true
        })
    })
    renderHarness()
    await rename()

    await confirmUpdate()

    expect(await screen.findByText("Updated links in 2 notes")).toBeInTheDocument()
    expect(
      screen.getByText(/Another note is also titled "New title", so the links use this note's id/)
    ).toBeInTheDocument()
  })

  it("says so, without Undo, when no note could be updated", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () =>
        rewriteResponse([
          { id: "linker-a", title: "Linker A", status: "skipped_conflict", version: 5, replaced_count: 0, replacements: [] },
          { id: "linker-b", title: "Linker B", status: "skipped_not_found", version: null, replaced_count: 0, replacements: [] }
        ])
    })
    renderHarness()
    await rename()

    await confirmUpdate()

    expect(
      await screen.findByText(
        /No links were updated\. Skipped: Linker A \(edited since\), Linker B \(no longer available\)/
      )
    ).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
    expect(onLinksChanged).not.toHaveBeenCalled()
  })

  it("reports a failed rewrite and offers no Undo", async () => {
    useServer({
      [REFERRERS]: () => twoReferrers,
      [REWRITE]: () => {
        throw new Error("Server unavailable")
      }
    })
    renderHarness()
    await rename()

    await confirmUpdate()

    expect(await screen.findByText(/Could not update links\. Server unavailable/)).toBeInTheDocument()
    expect(screen.queryByRole("button", { name: "Undo" })).not.toBeInTheDocument()
  })

  it("does not ask again for a rename the user dismissed", async () => {
    useServer({ [REFERRERS]: () => twoReferrers })
    renderHarness()
    await rename()

    fireEvent.click(await screen.findByRole("button", { name: /Close/i }))
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Update links" })).not.toBeInTheDocument()
    })
    await rename()
    await rename({ ...RENAME, oldTitle: "old  TITLE", newTitle: "NEW title" })

    expect(requestsTo(REFERRERS)).toHaveLength(1)
    expect(requestsTo(REWRITE)).toHaveLength(0)
    expect(screen.queryByRole("button", { name: "Update links" })).not.toBeInTheDocument()
  })

  it("still offers a different rename after one was dismissed", async () => {
    useServer({ [REFERRERS]: () => twoReferrers })
    renderHarness()
    await rename()
    fireEvent.click(await screen.findByRole("button", { name: /Close/i }))
    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Update links" })).not.toBeInTheDocument()
    })

    await rename({ noteId: "note-a", oldTitle: "New title", newTitle: "Newer title" })

    expect(await screen.findByText('2 notes link to "New title"')).toBeInTheDocument()
  })

  it("closes an open prompt when the notes owner changes", async () => {
    useServer({ [REFERRERS]: () => twoReferrers })
    const view = renderHarness()
    await rename()
    expect(await screen.findByRole("button", { name: "Update links" })).toBeInTheDocument()

    view.rerender(tree({ authorityScope: createNotesGraphAuthorityScope(SERVER_URL, 2) }))

    await waitFor(() => {
      expect(screen.queryByRole("button", { name: "Update links" })).not.toBeInTheDocument()
    })
  })
})
