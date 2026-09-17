import React from "react"
import {
  act,
  fireEvent,
  render,
  renderHook,
  screen,
  waitFor
} from "@testing-library/react"
import { beforeEach, describe, expect, it, vi } from "vitest"
import type {
  HistoryViewSelectionV1,
  HistorySelectionSnapshotV1
} from "@/types/history-selection"
import { resolveHistorySelection } from "@/utils/history-selection"

const mocks = vi.hoisted(() => ({
  bookmarks: new Map<string, any>(),
  capture: vi.fn(),
  confirm: vi.fn(),
  profile: vi.fn(),
  details: vi.fn(),
  localOwner: vi.fn(),
  link: vi.fn(),
  recoveries: vi.fn<(...args: any[]) => Promise<any[]>>(async () => []),
  dismissRecovery: vi.fn<(...args: any[]) => Promise<void>>(async () => {})
}))
vi.mock("@/db/dexie/history-selection", () => ({
  ensureLocalProfileId: () => mocks.profile(),
  loadHistoryTurnRecoveries: (...args: any[]) => mocks.recoveries(...args),
  dismissHistoryTurnRecovery: (...args: any[]) =>
    mocks.dismissRecovery(...args),
  loadHistoryBookmark: async (scope: any, owner: any) =>
    mocks.bookmarks.get(
      JSON.stringify([
        {
          profile_id: scope.profile_id,
          client_session_id: scope.client_session_id
        },
        { owner_key: owner.owner_key, conversation_id: owner.conversation_id }
      ])
    ) ?? null,
  saveHistoryBookmark: async (scope: any, view: any) =>
    mocks.bookmarks.set(
      JSON.stringify([
        scope,
        { owner_key: view.owner_key, conversation_id: view.conversation_id }
      ]),
      { ...scope, view }
    ),
  getLocalHistoryOwner: (id: string) => mocks.localOwner(id)
}))
vi.mock("@/services/chat-history-selection", () => ({
  captureHistorySnapshot: (...args: any[]) => mocks.capture(...args),
  confirmLegacyHistoryProjection: (...args: any[]) => mocks.confirm(...args)
}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getHistoryInfo = mocks.details
  }
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => ({ scopeKey: "scope" }),
  subscribeToServicePromptConfigChanges: () => () => {}
}))
vi.mock("@/db/dexie/server-chat-mirror", () => ({
  serverChatMirrorOwnerKey: () => "verified-scope",
  linkServerChatMirror: (...args: any[]) => mocks.link(...args)
}))
import {
  useHistorySelection,
  type HistorySelectionController
} from "../useHistorySelection"

const owner = {
  kind: "local" as const,
  profile_id: "profile",
  owner_key: "owner",
  conversation_id: "chat"
}
const snapshot: HistorySelectionSnapshotV1 = {
  version: 1,
  owner_key: "owner",
  conversation_id: "chat",
  source_digest: "source",
  storage_context_digest: "storage",
  fences: { conversation: "1", history: "1", settings: "1" },
  interpretation_status: { kind: "parent_graph_v1" },
  nodes: [
    {
      id: "u",
      revision: "u1",
      parent_id: null,
      role: "user",
      settled: true,
      preview: "Question"
    },
    {
      id: "a",
      revision: "a1",
      parent_id: "u",
      role: "assistant",
      settled: true,
      preview: "First answer"
    },
    {
      id: "b",
      revision: "b1",
      parent_id: "u",
      role: "assistant",
      settled: true,
      preview: "Other answer"
    }
  ]
}
let controllers: Record<string, HistorySelectionController>
function View({ label, reference }: { label: string; reference?: any }) {
  const controller = useHistorySelection()
  controllers[label] = controller
  React.useEffect(() => {
    void controller.open(owner, reference)
  }, [])
  return (
    <section aria-label={label}>
      <output data-testid={label}>
        {controller.capture?.status === "captured"
          ? controller.capture.selected_content
              .map((row) => row.message)
              .join(" / ")
          : controller.status}
      </output>
      <button
        onClick={() =>
          void controller.choose({ kind: "after_message", message_id: "a" })
        }
      >
        {label} first
      </button>
      <button onClick={() => void controller.choose({ kind: "empty" })}>
        {label} empty
      </button>
    </section>
  )
}
beforeEach(() => {
  controllers = {}
  mocks.bookmarks.clear()
  mocks.profile.mockReset().mockResolvedValue("profile")
  mocks.localOwner
    .mockReset()
    .mockImplementation(async (id) => ({ ...owner, conversation_id: id }))
  mocks.details.mockReset().mockResolvedValue(null)
  mocks.link.mockReset()
  mocks.capture.mockImplementation(async (_owner, view) => {
    const bound = { ...view, owner_key: "owner" }
    const result = resolveHistorySelection(snapshot, bound, "send", "")
    if (result.status !== "ready") return { ...result, snapshot, view: bound }
    return {
      status: "captured",
      snapshot,
      view: bound,
      rows: result.rows,
      selected_content: result.rows.map((row) => ({
        id: row.id,
        revision: row.revision,
        message: row.preview,
        images: []
      })),
      purpose: "send",
      storage_context_digest: "storage"
    }
  })
})
describe("mounted history selection", () => {
  it("keeps identical initialization independent and follows only the originating view revision", async () => {
    render(
      <>
        <View label="A" />
        <View label="B" />
      </>
    )
    await waitFor(() =>
      expect(screen.getByTestId("B").textContent).toBe(
        "Question / Other answer"
      )
    )
    const capturedB = controllers.B.view!
    fireEvent.click(screen.getByText("A first"))
    await waitFor(() =>
      expect(screen.getByTestId("A").textContent).toBe(
        "Question / First answer"
      )
    )
    expect(controllers.A.view!.view_session_id).not.toBe(
      controllers.B.view!.view_session_id
    )
    expect(controllers.A.bookmarkScope!.client_session_id).not.toBe(
      controllers.B.bookmarkScope!.client_session_id
    )
    await act(async () => {
      expect(await controllers.A.followResult(capturedB, "b")).toBe(false)
    })
    expect(screen.getByTestId("A").textContent).toBe("Question / First answer")
    const reference = controllers.A.reference!
    render(<View label="reopened" reference={reference} />)
    await waitFor(() =>
      expect(screen.getByTestId("reopened").textContent).toBe(
        "Question / First answer"
      )
    )
    expect(controllers.reopened.view!.view_session_id).not.toBe(
      controllers.A.view!.view_session_id
    )
    expect(controllers.reopened.bookmarkScope!.client_session_id).not.toBe(
      reference.client_session_id
    )
    fireEvent.click(screen.getByText("B empty"))
    await waitFor(() => expect(screen.getByTestId("B").textContent).toBe(""))
    await act(async () => {
      expect(await controllers.B.followResult(capturedB, "a")).toBe(false)
    })
    expect(controllers.B.view!.cursor).toEqual({ kind: "empty" })
  })
  it("does not replace an explicit choice when an older capture finishes", async () => {
    render(<View label="A" />)
    await waitFor(() => expect(controllers.A.status).toBe("ready"))
    const originalCapture = mocks.capture.getMockImplementation()!
    let release!: () => void
    mocks.capture.mockImplementationOnce(async (...args) => {
      await new Promise<void>((resolve) => {
        release = resolve
      })
      return originalCapture(...args)
    })
    let pending!: Promise<unknown>
    act(() => {
      pending = controllers.A.choose({ kind: "after_message", message_id: "a" })
    })
    fireEvent.click(screen.getByText("A empty"))
    await waitFor(() => expect(controllers.A.status).toBe("ready"))
    await act(async () => {
      release()
      await pending
    })
    expect(screen.getByTestId("A").textContent).toBe("")
    expect(controllers.A.view!.selection_revision).toBe(2)
  })
})

it("confirms complete source membership, retains failure for refresh, and fences a changed view", async () => {
  const legacy = {
    ...snapshot,
    interpretation_status: { kind: "legacy_review_required" }
  }
  mocks.capture.mockImplementation(async (_owner, view) => ({
    status: "legacy_review_required",
    code: "legacy_review_required",
    snapshot: legacy,
    view
  }))
  render(<View label="A" />)
  await waitFor(() =>
    expect(controllers.A.status).toBe("legacy_review_required")
  )
  mocks.confirm.mockRejectedValueOnce(new Error("source_changed"))
  await act(async () => {
    await controllers.A.confirm(["u", "a"], {
      kind: "before_message",
      message_id: "u"
    })
  })
  const confirmation = mocks.confirm.mock.calls.at(-1)![2]
  expect(confirmation.source_members).toEqual([
    { id: "u", revision: "u1" },
    { id: "a", revision: "a1" },
    { id: "b", revision: "b1" }
  ])
  expect(confirmation.ordered_path_ids).toEqual(["u", "a"])
  expect(controllers.A.status).toBe("stale_selection")
  let complete!: (value: any) => void
  mocks.confirm.mockImplementationOnce(
    async () =>
      new Promise((resolve) => {
        complete = resolve
      })
  )
  let confirmationPromise!: Promise<unknown>
  act(() => {
    confirmationPromise = controllers.A.confirm(["u", "a"], {
      kind: "after_message",
      message_id: "a"
    })
  })
  await act(async () => {
    await controllers.A.choose({ kind: "empty" })
  })
  await act(async () => {
    complete({
      projection_id: "p",
      cursor: { kind: "after_message", message_id: "a" }
    })
    await confirmationPromise
  })
  expect(controllers.A.view!.cursor).toEqual({ kind: "empty" })
  expect(controllers.A.pending).toBeNull()
})

it("preserves before-first through copied references while allocating independent writers", async () => {
  const origin = renderHook(() => useHistorySelection())
  await act(async () => {
    await origin.result.current.open(owner)
    await origin.result.current.choose({
      kind: "before_message",
      message_id: "u"
    })
  })
  const reference = origin.result.current.getReference()!
  render(
    <>
      <View label="copy-a" reference={reference} />
      <View label="copy-b" reference={reference} />
    </>
  )
  await waitFor(() => expect(controllers["copy-b"].status).toBe("ready"))
  expect(controllers["copy-a"].view!.cursor).toEqual({
    kind: "before_message",
    message_id: "u"
  })
  expect(screen.getByTestId("copy-a").textContent).toBe("")
  fireEvent.click(screen.getByText("copy-a first"))
  await waitFor(() =>
    expect(screen.getByTestId("copy-a").textContent).toBe(
      "Question / First answer"
    )
  )
  expect(screen.getByTestId("copy-b").textContent).toBe("")
  expect(controllers["copy-a"].bookmarkScope).not.toEqual(
    controllers["copy-b"].bookmarkScope
  )
})
it("keeps an unbound mirror readable without initializing local ownership, and binds only on explicit capture", async () => {
  mocks.details.mockResolvedValue({
    id: "mirror",
    server_chat_id: "chat",
    title: "Old conversation"
  })
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    await result.current.loadConversation({ historyId: "mirror" })
  })
  expect(result.current.error).toBe("unbound_server_mirror")
  expect(mocks.profile).not.toHaveBeenCalled()
  expect(mocks.localOwner).not.toHaveBeenCalled()
  expect(mocks.link).not.toHaveBeenCalled()
  await act(async () => {
    await result.current.loadConversation({
      historyId: "mirror",
      bindUnbound: true
    })
  })
  expect(result.current.status).toBe("ready")
  expect(mocks.link.mock.calls[0][0]).toMatchObject({
    ownerKey: "verified-scope",
    legacyHistoryId: "mirror",
    chatId: "chat"
  })
  expect(mocks.localOwner).not.toHaveBeenCalled()
})
it("does not create a durable profile for temporary history", async () => {
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    await result.current.loadConversation({ temporary: true })
  })
  expect(result.current.error).toBe("temporary_history_unavailable")
  expect(mocks.profile).not.toHaveBeenCalled()
  expect(result.current.reference).toBeNull()
})
it("keeps logical tab identities stable and refuses a result captured in another tab", async () => {
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    result.current.activate("A")
    await result.current.open(owner)
  })
  const first = result.current.view!
  await act(async () => {
    result.current.activate("B")
    await result.current.open(owner)
    await result.current.choose({ kind: "empty" })
  })
  const second = result.current.view!
  await act(async () => {
    result.current.activate("A")
  })
  expect(result.current.view).toEqual(first)
  await act(async () => {
    expect(await result.current.followResult(second, "a")).toBe(false)
  })
  expect(result.current.view).toEqual(first)
})

import {
  historySelectionExpansionPath,
  parseHistorySelectionHandoff
} from "../useHistorySelection"
it("rejects malformed or authority-bearing handoff data", () => {
  expect(() =>
    parseHistorySelectionHandoff("?historySelection=broken")
  ).toThrow("invalid_history_reference")
  expect(() =>
    parseHistorySelectionHandoff(
      "?historySelection=" +
        encodeURIComponent(
          JSON.stringify({
            profile_id: "p",
            client_session_id: "s",
            owner_key: "o",
            conversation_id: "c",
            owner_kind: "native",
            accessToken: "not-permitted"
          })
        )
    )
  ).toThrow("invalid_history_reference")
  expect(
    historySelectionExpansionPath({
      reference: null,
      owner: { kind: "unavailable", code: "temporary" }
    })
  ).toBe("/chat")
})

it("never reuses an old view revision after rehydrating the same conversation", async () => {
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    await result.current.open(owner)
    await result.current.choose({ kind: "after_message", message_id: "a" })
  })
  const origin = result.current.view!
  await act(async () => {
    await result.current.open(owner)
  })
  expect(result.current.view!.selection_revision).toBeGreaterThan(
    origin.selection_revision
  )
  await act(async () => {
    expect(await result.current.followResult(origin, "b")).toBe(false)
  })
})

it("exposes a missing bookmarked target instead of choosing the latest remaining row", async () => {
  const original = renderHook(() => useHistorySelection())
  await act(async () => {
    await original.result.current.open(owner)
  })
  const reference = original.result.current.getReference()!
  const changed = {
    ...snapshot,
    nodes: snapshot.nodes.filter((row) => row.id !== "b")
  }
  mocks.capture.mockImplementation(async (_owner, view) => ({
    ...resolveHistorySelection(changed, view, "send", ""),
    snapshot: changed,
    view
  }))
  render(<View label="missing" reference={reference} />)
  await waitFor(() =>
    expect(controllers.missing.status).toBe("stale_selection")
  )
  expect(controllers.missing.view!.cursor).toEqual({
    kind: "after_message",
    message_id: "b"
  })
  expect(
    controllers.missing.capture!.snapshot.nodes.map((row) => row.id)
  ).toEqual(["u", "a"])
})

it("retains the original uncertain confirmation through expansion and destination reload without automatic replay", async () => {
  const origin = renderHook(() => useHistorySelection())
  await act(async () => {
    await origin.result.current.open(owner)
  })
  const sourceReference = origin.result.current.getReference()!
  const sourceView = origin.result.current.view!
  const pending = {
    version: 1,
    projection_id: "uncertain-projection",
    owner_key: "owner",
    conversation_id: "chat",
    source_digest: "source",
    fences: snapshot.fences,
    source_members: snapshot.nodes.map(({ id, revision }) => ({
      id,
      revision
    })),
    ordered_path_ids: ["u", "a"],
    cursor: { kind: "before_message", message_id: "u" },
    selection_revision: sourceView.selection_revision
  }
  const key = JSON.stringify([
    {
      profile_id: "profile",
      client_session_id: sourceReference.client_session_id
    },
    { owner_key: "owner", conversation_id: "chat" }
  ])
  mocks.bookmarks.set(key, {
    ...sourceReference,
    view: sourceView,
    pending_confirmation: pending,
    pending_view_session_id: sourceView.view_session_id,
    pending_dispatch_started: true
  })
  const expanded = renderHook(() => useHistorySelection())
  await act(async () => {
    await expanded.result.current.open(owner, sourceReference)
  })
  let path!: string | null
  await act(async () => {
    path = await expanded.result.current.prepareExpansionPath()
  })
  const frozen = parseHistorySelectionHandoff(path!)!
  const destination = renderHook(() =>
    useHistorySelection({ storageKey: "h1-destination" })
  )
  await act(async () => {
    await destination.result.current.open(owner, frozen)
  })
  destination.unmount()
  const reopened = renderHook(() =>
    useHistorySelection({ storageKey: "h1-destination" })
  )
  const callsBefore = mocks.confirm.mock.calls.length
  await act(async () => {
    await reopened.result.current.open(owner)
  })
  expect(reopened.result.current.status).toBe("pending_unknown")
  expect(reopened.result.current.pending?.scope.client_session_id).toBe(
    sourceReference.client_session_id
  )
  expect(reopened.result.current.pending?.intent).toEqual(pending)
  expect(reopened.result.current.pending?.view.view_session_id).toBe(
    sourceView.view_session_id
  )
  expect(mocks.bookmarks.get(key).pending_dispatch_started).toBe(true)
  expect(mocks.confirm.mock.calls.length).toBe(callsBefore)
})

it("clears its reload reference when starting a new empty conversation", async () => {
  const hook = renderHook(() => useHistorySelection({ storageKey: "h1-reset" }))
  await act(async () => {
    await hook.result.current.open(owner)
  })
  await waitFor(() =>
    expect(hook.result.current.getStoredReference()).not.toBeNull()
  )
  act(() => hook.result.current.reset())
  expect(hook.result.current.getStoredReference()).toBeNull()
  expect(hook.result.current.view).toBeNull()
})

it("surfaces a failed explicit mirror binding after owner capture", async () => {
  mocks.details.mockResolvedValue({
    id: "mirror",
    server_chat_id: "chat",
    title: "Old conversation"
  })
  mocks.link.mockRejectedValue(new Error("mirror_binding_failed"))
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    await result.current.loadConversation({
      historyId: "mirror",
      bindUnbound: true
    })
  })
  expect(result.current.status).toBe("error")
  expect(result.current.error).toBe("mirror_binding_failed")
})


it("restores scoped pending operations from older views without making them selected rows", async () => {
  const entry: any = {
    scope: { profile_id: "profile", client_session_id: "old-view" },
    turn: {
      operation_id: "op",
      owner_key: "owner",
      conversation_id: "chat",
      state: "unknown",
      input_text: "pending",
      result_text: ""
    }
  }
  mocks.recoveries.mockResolvedValue([entry])
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    await result.current.open(owner, null)
  })
  await waitFor(() => expect(result.current.recoveries).toEqual([entry]))
  expect(result.current.capture?.snapshot.nodes.map((row) => row.id)).toEqual([
    "u",
    "a",
    "b"
  ])
  mocks.recoveries.mockResolvedValue([])
  await act(async () => {
    await result.current.dismissRecovery(entry)
  })
  expect(mocks.dismissRecovery).toHaveBeenCalledWith(
    entry.scope,
    expect.objectContaining({ owner_key: "owner", conversation_id: "chat" }),
    "op"
  )
  expect(result.current.recoveries).toEqual([])
})
