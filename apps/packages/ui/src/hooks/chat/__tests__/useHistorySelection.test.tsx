import { useSidepanelChatFreshness, FOCUS_CHECK_DEBOUNCE_MS, FOCUS_CHECK_MIN_INTERVAL_MS } from "@/routes/sidepanel-chat-freshness"
import React from "react"
import { useChatHistoryFreshness } from "../useChatHistoryFreshness"
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
  openButtons: [] as Array<() => void>,
  bookmarks: new Map<string, any>(),
  configChanged: null as null | (() => void),
  forkCandidate: vi.fn(async (..._args: any[]) => null as any),
  forkRecords: vi.fn(async (..._args: any[]) => [] as any[]),
  forkSettings: vi.fn(async (..._args: any[]) => null as any),
  updateForkSettings: vi.fn(async (..._args: any[]) => null as any),
  capture: vi.fn(),
  historyVersion: vi.fn(),
  beforeBookmark: vi.fn(async () => {}),
  verifyEdit: vi.fn(async (..._args: any[]) => {}),
  confirm: vi.fn(),
  profile: vi.fn(),
  details: vi.fn(),
  localOwner: vi.fn(),
  link: vi.fn(),
  recoveries: vi.fn<(...args: any[]) => Promise<any[]>>(async () => []),
  dismissRecovery: vi.fn<(...args: any[]) => Promise<boolean>>(async () => true)
}))
vi.mock("@/components/Common/Button", async (original) => {
  const { Button } =
    await original<typeof import("@/components/Common/Button")>()
  const React = await import("react")
  return {
    Button: React.forwardRef<HTMLButtonElement, any>((props, ref) => {
      if (props.children === "Open saved reply")
        mocks.openButtons.push(props.onClick)
      return React.createElement(Button, { ...props, ref })
    })
  }
})
vi.mock("@/db/dexie/fork-operations", () => ({
  findForkCandidate: (...args: any[]) => mocks.forkCandidate(...args),
  loadForkOperations: (...args: any[]) => mocks.forkRecords(...args),
  allowNewForkOperation: vi.fn()
}))
vi.mock("@/db/dexie/history-selection", async (original) => ({
  canonicalHistoryJson: (await original<any>()).canonicalHistoryJson,
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
  saveHistoryBookmark: async (scope: any, view: any) => {
    await mocks.beforeBookmark()
    return mocks.bookmarks.set(
      JSON.stringify([
        scope,
        { owner_key: view.owner_key, conversation_id: view.conversation_id }
      ]),
      { ...scope, view }
    )
  },
  getLocalHistoryOwner: (id: string) => mocks.localOwner(id)
}))
vi.mock("@/services/chat-history-selection", async (original) => ({
  readNativeHistoryVersion: (...args: unknown[]) => mocks.historyVersion(...args),
  qualifyNativeMessageEditCapture: (await original<any>()).qualifyNativeMessageEditCapture,
  verifyNativeMessageEditReceipt: (...args: any[]) => mocks.verifyEdit(...args),
  captureHistorySnapshot: (...args: any[]) => mocks.capture(...args),
  readNativeForkSettings: (...args: any[]) => mocks.forkSettings(...args),
  updateNativeForkSettings: (...args: any[]) => mocks.updateForkSettings(...args),
  confirmLegacyHistoryProjection: (...args: any[]) => mocks.confirm(...args)
}))
vi.mock("@/db/dexie/chat", () => ({
  PageAssistDatabase: class {
    getHistoryInfo = mocks.details
  }
}))
vi.mock("@/services/service-prompts", () => ({
  resolveServicePromptScope: async () => ({ scopeKey: "scope" }),
  subscribeToServicePromptConfigChanges: (changed: () => void) => { mocks.configChanged = changed; return () => {} }
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
  mocks.beforeBookmark.mockReset().mockResolvedValue(undefined)
  mocks.verifyEdit.mockReset().mockResolvedValue(undefined)
  mocks.forkCandidate.mockReset().mockResolvedValue(null)
  mocks.forkRecords.mockReset().mockResolvedValue([])
  mocks.forkSettings.mockReset().mockResolvedValue(null)
  mocks.updateForkSettings.mockReset().mockResolvedValue(null)
  mocks.confirm.mockClear()
  mocks.recoveries.mockReset().mockResolvedValue([])
  mocks.dismissRecovery.mockReset().mockResolvedValue(true)
  mocks.configChanged = null
  controllers = {}
  mocks.bookmarks.clear()
  mocks.profile.mockReset().mockResolvedValue("profile")
  mocks.localOwner
    .mockReset()
    .mockImplementation(async (id) => ({ ...owner, conversation_id: id }))
  mocks.details.mockReset().mockResolvedValue(null)
  mocks.link.mockReset()
  mocks.capture.mockReset().mockImplementation(async (_owner, view) => {
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

it("does not capture or publish a deliberate reopen cancelled before its account watcher installs", async () => {
  const mounted = renderHook(() => useHistorySelection())
  let current = true
  let pending!: Promise<boolean>
  act(() => {
    pending = mounted.result.current.loadConversation({ serverChatId: "chat", isCurrent: () => current })
    current = false
  })
  await act(async () => { expect(await pending).toBe(false) })
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(mounted.result.current.getCurrent().owner).toBeNull()
})

it("does not publish a late capture error after deliberate selection is cancelled", async () => {
  let reject!: (error: Error) => void
  mocks.capture.mockReturnValue(new Promise((_resolve, fail) => { reject = fail }))
  const mounted = renderHook(() => useHistorySelection())
  let current = true
  let pending!: Promise<boolean>
  act(() => {
    pending = mounted.result.current.loadConversation({ serverChatId: "chat", isCurrent: () => current })
  })
  await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
  current = false
  await act(async () => { reject(new Error("cancelled-owner-error")); expect(await pending).toBe(false) })
  expect(mounted.result.current.getCurrent().error).toBeNull()
  expect(mounted.result.current.canAutomaticallyLoad()).toBe(false)
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
it.each([undefined, "owner"])("rejects a foreign account-stamped local history before ownership or capture (local owner %s)", async (localOwner) => {
  mocks.details.mockResolvedValue({ id: "chat", server_scope_key: "foreign-account", local_owner_key: localOwner })
  const onCapture = vi.fn()
  const { result } = renderHook(() => useHistorySelection({ onCapture }))
  await act(async () => { await result.current.loadConversation({ historyId: "chat" }) })
  expect(result.current.error).toBe("owner_conversation_mismatch")
  expect(mocks.localOwner).not.toHaveBeenCalled()
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(onCapture).not.toHaveBeenCalled()
})

it.each([
  { server_scope_key: "verified-scope" },
  { local_owner_key: "owner" }
])("opens a verified account history or an intentional profile-owned local fork: %j", async (ownership) => {
  mocks.details.mockResolvedValue({ id: "chat", ...ownership })
  const onCapture = vi.fn()
  const { result } = renderHook(() => useHistorySelection({ onCapture }))
  await act(async () => { await result.current.loadConversation({ historyId: "chat" }) })
  expect(result.current.status).toBe("ready")
  expect(result.current.owner).toMatchObject({ kind: "local", conversation_id: "chat" })
  expect(onCapture).toHaveBeenCalled()
})

it("does not install ownership on an unowned local cache", async () => {
  mocks.details.mockResolvedValue({ id: "unowned" })
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => { await result.current.loadConversation({ historyId: "unowned" }) })
  expect(result.current.error).toBe("owner_conversation_mismatch")
  expect(mocks.localOwner).not.toHaveBeenCalled()
  expect(mocks.capture).not.toHaveBeenCalled()
})

it.each([
  { id: "cache" },
  { id: "cache", local_owner_key: "owner" },
  { id: "cache", server_chat_id: "other-chat" }
])("rejects unrelated local and native locators before capture: %j", async (details) => {
  mocks.details.mockResolvedValue(details)
  const onCapture = vi.fn()
  const { result } = renderHook(() => useHistorySelection({ onCapture }))
  await act(async () => {
    await result.current.loadConversation({ historyId: "cache", serverChatId: "chat" })
  })
  expect(result.current.error).toBe("owner_conversation_mismatch")
  expect(mocks.localOwner).not.toHaveBeenCalled()
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(onCapture).not.toHaveBeenCalled()
})

it.each([
  { server_scope_key: "verified-scope" },
  { local_owner_key: "owner" }
])("does not publish a selected local capture after its caller is cancelled: %j", async (ownership) => {
  mocks.details.mockResolvedValue({ id: "chat", ...ownership })
  const capture = mocks.capture.getMockImplementation()!
  let finish!: () => void
  const held = new Promise<void>((resolve) => { finish = resolve })
  mocks.capture.mockImplementationOnce(async (...args) => {
    await held
    return capture(...args)
  })
  let current = true
  const onCapture = vi.fn()
  const { result } = renderHook(() => useHistorySelection({ onCapture }))
  let pending!: Promise<boolean>
  act(() => {
    pending = result.current.loadConversation({ historyId: "chat", isCurrent: () => current })
  })
  await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
  await act(async () => {
    current = false
    finish()
    expect(await pending).toBe(false)
  })
  expect(onCapture).not.toHaveBeenCalled()
  expect(result.current.capture).toBeNull()
})

const changeAccount = (event: string) => {
  if (event === "cookie") {
    window.dispatchEvent(new StorageEvent("storage", {
      key: "tldwCookieSessionConfig",
      oldValue: JSON.stringify({ serverUrl: "http://old", authMode: "multi-user" }),
      newValue: null
    }))
  } else {
    window.dispatchEvent(new Event(`tldw:auth-${event}-changed`))
  }
}

it.each(
  ["local", "native"].flatMap((kind) =>
    ["principal", "credentials", "cookie"].flatMap((event) =>
      [false, true].map((held) => ({ kind, event, held }))
    )
  )
)("revokes $kind previews after $event boundary (held: $held)", async ({ kind, event, held }) => {
  mocks.details.mockResolvedValue({ id: "chat", server_scope_key: "verified-scope" })
  const capture = mocks.capture.getMockImplementation()!
  let finish!: () => void
  if (held) {
    const gate = new Promise<void>((resolve) => { finish = resolve })
    mocks.capture.mockImplementationOnce(async (...args) => { await gate; return capture(...args) })
  }
  const onCapture = vi.fn()
  const mounted = renderHook(() => useHistorySelection({ onCapture }))
  let pending!: Promise<boolean>
  act(() => {
    pending = mounted.result.current.loadConversation(kind === "local"
      ? { historyId: "chat" } : { serverChatId: "chat" })
  })
  if (held) await waitFor(() => expect(mocks.capture).toHaveBeenCalledTimes(1))
  else await act(async () => { await pending })
  const published = onCapture.mock.calls.length
  act(() => { changeAccount(event) })
  expect(mounted.result.current.getCurrent()).toMatchObject({
    view: null,
    capture: null,
    pending: null,
    error: "request_config_scope_changed"
  })
  expect(mounted.result.current.canAutomaticallyLoad()).toBe(false)
  const capturesBeforeActions = mocks.capture.mock.calls.length
  await act(async () => {
    expect(await mounted.result.current.choose({ kind: "empty" })).toBe(false)
    expect(await mounted.result.current.refresh()).toBe(false)
    if (held) { finish(); expect(await pending).toBe(false) }
  })
  expect(mocks.capture).toHaveBeenCalledTimes(capturesBeforeActions)
  expect(onCapture).toHaveBeenCalledTimes(published)
})

it("does not revive expired account previews when returning to a logical tab, while a profile fork reopens", async () => {
  const capture = mocks.capture.getMockImplementation()!
  mocks.capture.mockImplementation(async (owner, view) => {
    const result = await capture(owner, { ...view, conversation_id: "chat" })
    return {
      ...result,
      view: { ...result.view, conversation_id: owner.conversation_id },
      snapshot: { ...result.snapshot, conversation_id: owner.conversation_id }
    }
  })
  mocks.details.mockImplementation(async (id) => id === "chat"
    ? { id, server_scope_key: "verified-scope" }
    : { id, local_owner_key: "owner" })
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => { await result.current.loadConversation({ historyId: "chat" }) })
  expect(result.current.capture).not.toBeNull()
  act(() => { result.current.activate("profile-tab") })
  await act(async () => { await result.current.loadConversation({ historyId: "fork" }) })
  expect(result.current.owner).toMatchObject({ kind: "local", conversation_id: "fork" })
  expect(result.current.owner).not.toHaveProperty("validate_lease")
  act(() => { changeAccount("principal"); result.current.activate("default") })
  expect(result.current.capture).toBeNull()
  expect(result.current.view).toBeNull()
  expect(result.current.canAutomaticallyLoad()).toBe(false)
  act(() => { result.current.activate("profile-tab") })
  expect(result.current.canAutomaticallyLoad()).toBe(true)
  await act(async () => { await result.current.loadConversation({ historyId: "fork" }) })
  expect(result.current.status).toBe("ready")
})

it("invalidates an account-stamped local capture when the account changes during its read", async () => {
  mocks.details.mockResolvedValue({ id: "chat", server_scope_key: "verified-scope" })
  const capture = mocks.capture.getMockImplementation()!
  let entered!: () => void, finish!: () => void
  const started = new Promise<void>(resolve => { entered = resolve })
  const held = new Promise<void>(resolve => { finish = resolve })
  mocks.capture.mockImplementationOnce(async (...args) => { entered(); await held; return capture(...args) })
  const onCapture = vi.fn()
  const { result } = renderHook(() => useHistorySelection({ onCapture }))
  await act(async () => {
    const pending = result.current.loadConversation({ historyId: "chat" })
    await started
    mocks.configChanged!()
    finish()
    expect(await pending).toBe(false)
  })
  expect(onCapture).not.toHaveBeenCalled()
  expect(result.current.capture).toBeNull()
  expect(result.current.error).toBe("request_config_scope_changed")
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
it("keeps another logical tab's selection after reset and refuses its stale results", async () => {
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
  act(() => { result.current.reset() })
  expect(result.current.owner).toBeNull()
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

it("does not commit a held explicit mirror binding after caller cancellation", async () => {
  mocks.details.mockResolvedValue({ id: "mirror", server_chat_id: "chat" })
  let entered!: () => void, finish!: () => void
  const started = new Promise<void>((resolve) => { entered = resolve })
  const held = new Promise<void>((resolve) => { finish = resolve })
  const committed = vi.fn()
  mocks.link.mockImplementation(async (options) => {
    entered()
    await held
    if (options.validate_lease?.() === false)
      throw new Error("request_config_scope_changed")
    committed()
  })
  let current = true
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    const pending = result.current.loadConversation({ historyId: "mirror", bindUnbound: true, isCurrent: () => current })
    await started
    current = false
    finish()
    expect(await pending).toBe(false)
  })
  expect(committed).not.toHaveBeenCalled()
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
    "op", "dismissed", expect.objectContaining({ validate_lease: expect.any(Function) })
  )
  expect(result.current.recoveries).toEqual([])
})

it("a historyId-only bound native mirror supplies its own successful load receipt", async () => {
  mocks.details.mockResolvedValue({
    id: "mirror",
    server_chat_id: "chat",
    server_scope_key: "verified-scope",
    title: "Bound"
  })
  const receipt = vi.fn()
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    expect(
      await result.current.loadConversation(
        { historyId: "mirror" },
        null,
        receipt
      )
    ).toBe(true)
  })
  expect(receipt).toHaveBeenCalledOnce()
  expect(receipt.mock.calls[0][0]).toEqual({
    owner: result.current.owner,
    view: result.current.view
  })
  expect(receipt.mock.calls[0][0].owner).toMatchObject({
    kind: "native",
    conversation_id: "chat"
  })
  expect(mocks.localOwner).not.toHaveBeenCalled()
})

it("a superseded deferred owner load emits no receipt for the destination", async () => {
  let finish!: (value: any) => void, entered!: () => void
  const started = new Promise<void>((resolve) => {
    entered = resolve
  })
  mocks.details.mockImplementationOnce(
    () =>
      new Promise((resolve) => {
        finish = resolve
        entered()
      })
  )
  const receipt = vi.fn()
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    const pending = result.current.loadConversation(
      { historyId: "old" },
      null,
      receipt
    )
    await started
    await result.current.open(owner)
    finish({ id: "old" })
    expect(await pending).toBe(false)
  })
  expect(receipt).not.toHaveBeenCalled()
  expect(result.current.owner).toBe(owner)
})

it("a same-owner cursor change after open cannot become that load's receipt", async () => {
  mocks.details.mockResolvedValue({
    id: "mirror",
    server_chat_id: "chat",
    title: "Old"
  })
  const receipt = vi.fn()
  const { result } = renderHook(() => useHistorySelection())
  mocks.link.mockImplementationOnce(async () => {
    await result.current.choose({ kind: "empty" })
  })
  await act(async () => {
    expect(
      await result.current.loadConversation(
        { historyId: "mirror", bindUnbound: true },
        null,
        receipt
      )
    ).toBe(true)
  })
  expect(result.current.status).toBe("ready")
  expect(result.current.view?.cursor).toEqual({ kind: "empty" })
  expect(receipt).not.toHaveBeenCalled()
})

it("held fork candidate qualification exposes no ordinary settings fallback, then scoped settings edit/reopen works", async () => {
  const {result} = renderHook(() => useHistorySelection())
  const native: any = {kind: "native", owner_key: "owner", conversation_id: "chat", scope: {type: "workspace", workspaceId: "original"}, validate_lease: () => true}
  let release!: (value: any) => void
  mocks.forkCandidate.mockImplementationOnce(() => new Promise(done => {release = done}))
  mocks.forkSettings.mockResolvedValue({authorNote: "server only"})
  let opening!: Promise<boolean>
  act(() => { opening = result.current.open(native) })
  await waitFor(() => expect(mocks.forkCandidate).toHaveBeenCalled())
  expect(result.current.settingsMode("chat", native.scope)).toBe("pending")
  expect(mocks.forkSettings).not.toHaveBeenCalled()
  await act(async () => {release({owner_key: "owner", candidate_child_id: "chat"}); await opening})
  expect(result.current.settingsMode("chat", native.scope)).toBe("fork")
  expect(result.current.forkSettings).toEqual({authorNote: "server only"})
  mocks.updateForkSettings.mockResolvedValue({authorNote: "explicit edit"})
  await act(async () => {await result.current.updateForkSettings({authorNote: "explicit edit"})})
  expect(result.current.forkSettings).toEqual({authorNote: "explicit edit"})
  expect(result.current.settingsMode("chat", {type: "workspace", workspaceId: "elsewhere"})).toBe("pending")
  mocks.forkCandidate.mockResolvedValue({owner_key: "owner", candidate_child_id: "chat"})
  mocks.forkSettings.mockResolvedValue({authorNote: "legitimate later edit"})
  await act(async () => {await result.current.open(native)})
  expect(result.current.forkSettings).toEqual({authorNote: "legitimate later edit"})
})
it("fork operations refresh hides another owner/workspace and retains original source after selection changes", async () => {
  const {result} = renderHook(() => useHistorySelection())
  await act(async () => {await result.current.open(owner)})
  mocks.forkRecords.mockResolvedValue([{owner_key: "owner", operation_id: "op", state: "partial", candidate_child_id: "child"}])
  await act(async () => {await result.current.refreshForkOperations()})
  expect(result.current.forkOperations).toHaveLength(1)
  await act(async () => {result.current.reset()})
  expect(result.current.forkOperations).toHaveLength(0)
})
it("an old workspace fork control cannot release another namespace's active intent", async () => {
  const {result} = renderHook(() => useHistorySelection())
  const native: any = {kind: "native", owner_key: "owner", conversation_id: "chat", scope: {type: "workspace", workspaceId: "current"}, validate_lease: () => true}
  await act(async () => {await result.current.open(native)})
  const store = await import("@/db/dexie/fork-operations")
  vi.mocked(store.allowNewForkOperation).mockClear()
  await act(async () => {await result.current.allowNewFork({owner_key: "owner", conversation_id: "chat", context: {kind: "native", scope: {type: "workspace", workspaceId: "old"}}} as any)})
  expect(store.allowNewForkOperation).not.toHaveBeenCalled()
})

it.each(["published", "held-success", "held-error"])("config invalidation hides fork outcomes and fences %s reads without changing durable records", async mode => {
  const retained = [{owner_key: "owner", conversation_id: "chat", operation_id: "private-operation", state: "partial", candidate_child_id: "private-child", result: {state: "partial", code: "private-error"}, context: {kind: "native", scope: {type: "global"}}}]
  mocks.forkRecords.mockResolvedValue(retained)
  const hook = renderHook(() => useHistorySelection())
  await act(async () => {await hook.result.current.loadConversation({serverChatId: "chat"})})
  await waitFor(() => expect(hook.result.current.forkOperations).toHaveLength(1))
  await act(async () => {await hook.result.current.refresh()})
  await waitFor(() => expect(hook.result.current.forkOperations).toHaveLength(1))
  let finish!: () => void
  let reading: Promise<void> | undefined
  if (mode !== "published") {
    mocks.forkRecords.mockImplementationOnce(() => new Promise((resolve, reject) => {
      finish = () => mode === "held-success" ? resolve(retained) : reject(new Error("stale-record-error"))
    }))
    act(() => {reading = hook.result.current.refreshForkOperations()})
  }
  act(() => mocks.configChanged!())
  expect(hook.result.current.status).toBe("unsupported_history_capability")
  expect(hook.result.current.forkOperations).toEqual([])
  expect(hook.result.current.forkOperationsError).toBeNull()
  if (reading) await act(async () => {finish(); await reading})
  const reads = mocks.forkRecords.mock.calls.length
  await act(async () => {await hook.result.current.refreshForkOperations()})
  expect(mocks.forkRecords).toHaveBeenCalledTimes(reads)
  expect(hook.result.current.forkOperations).toEqual([])
  expect(hook.result.current.forkOperationsError).toBeNull()
  expect(hook.result.current.error).toBe("request_config_scope_changed")
  expect(await hook.result.current.inspectForkOperation(retained[0] as any)).toBe(false)
  expect(retained[0]).toMatchObject({operation_id: "private-operation", candidate_child_id: "private-child", state: "partial"})
  expect(hook.result.current.canAutomaticallyLoad()).toBe(false)
  await act(async () => {await hook.result.current.loadConversation({serverChatId: "chat"})})
  expect(hook.result.current.canAutomaticallyLoad()).toBe(true)
  await waitFor(() => expect(hook.result.current.forkOperations).toEqual(retained))
  expect(mocks.forkRecords.mock.calls.at(-1)?.[0]).toMatchObject({owner_key: "owner", conversation_id: "chat", scope: {type: "global"}})
})

it.each(["ordinary", "fork", "lookup-failed", "settings-failed"])("owner/settings qualification is independent of legacy ancestry readiness: %s", async kind => {
  mocks.capture.mockImplementation(async (_owner, view) => ({status: "legacy_review_required", code: "legacy_review_required", snapshot, view: {...view, owner_key: "owner"}}))
  let finish!: (value: any) => void
  mocks.forkCandidate.mockImplementationOnce(() => new Promise(resolve => {finish = resolve}))
  if (kind === "settings-failed") mocks.forkSettings.mockRejectedValueOnce(new Error("settings unavailable"))
  const hook = renderHook(() => useHistorySelection())
  let opening!: Promise<boolean>
  act(() => {opening = hook.result.current.loadConversation({serverChatId: "chat"})})
  await waitFor(() => expect(mocks.forkCandidate).toHaveBeenCalled())
  expect(hook.result.current.settingsMode("chat")).toBe("pending")
  if (kind === "lookup-failed") {
    // A failed fresh lookup after a prior ordinary qualification must close the old path too.
    await act(async () => {finish(null); await opening})
    mocks.forkCandidate.mockRejectedValueOnce(new Error("lookup unavailable"))
    await act(async () => {await hook.result.current.refresh()})
  } else await act(async () => {finish(kind === "ordinary" ? null : {candidate_child_id: "chat"}); await opening})
  expect(hook.result.current.settingsMode("chat")).toBe(kind.endsWith("failed") ? "pending" : kind)
  if (!kind.endsWith("failed")) expect(hook.result.current.status).toBe("legacy_review_required")
  expect(mocks.confirm).not.toHaveBeenCalled()
})

it.each(["published", "held-success", "held-error"])(
  "native recovery authority rejects invalidated %s presentation and actions",
  async (mode) => {
    const entry: Parameters<HistorySelectionController["dismissRecovery"]>[0] =
      {
        scope: { profile_id: "profile", client_session_id: "original-view" },
        turn: {
          origin_view: {
            view_session_id: "original",
            owner_key: "owner",
            conversation_id: "chat",
            interpretation: { kind: "parent_graph_v1" },
            cursor: { kind: "empty" },
            selection_revision: 0
          },
          selection_digest: "selection",
          request_context_digest: "request",
          created_at: 1,
          input_images: [],
          persistence: "server",
          operation_id: "retained-turn",
          owner_key: "owner",
          conversation_id: "chat",
          state: "generated_unsaved",
          input_text: "private input",
          result_text: "private result"
        }
      }
    const durable = [entry]
    mocks.recoveries.mockResolvedValue(durable)
    const hook = renderHook(() => useHistorySelection())
    await act(async () => {
      await hook.result.current.loadConversation({ serverChatId: "chat" })
    })
    await waitFor(() => expect(hook.result.current.recoveries).toEqual(durable))
    let settle!: () => void
    let reading: Promise<void> | undefined
    if (mode !== "published") {
      mocks.recoveries.mockImplementationOnce(
        () =>
          new Promise((resolve, reject) => {
            settle = () =>
              mode === "held-success"
                ? resolve(durable)
                : reject(new Error("private retained read error"))
          })
      )
      act(() => {
        reading = hook.result.current.refreshRecovery()
      })
    }
    const oldDismiss = hook.result.current.dismissRecovery
    act(() => mocks.configChanged!())
    const invalidOwner = hook.result.current.owner
    expect(invalidOwner?.kind).toBe("native")
    if (invalidOwner?.kind !== "native")
      throw new Error("expected native owner")
    expect(invalidOwner.validate_lease()).toBe(false)
    if (reading)
      await act(async () => {
        settle()
        await reading
      })
    expect(hook.result.current.recoveries).toEqual([])
    const reads = mocks.recoveries.mock.calls.length
    await act(async () => {
      await hook.result.current.refreshRecovery()
      await oldDismiss(entry)
    })
    expect(mocks.recoveries).toHaveBeenCalledTimes(reads)
    expect(mocks.dismissRecovery).not.toHaveBeenCalled()
    expect(hook.result.current.recoveries).toEqual([])
    expect(hook.result.current.recoveryError).toBeNull()
    expect(durable).toEqual([entry])
    await act(async () => {
      await hook.result.current.loadConversation({ serverChatId: "chat" })
    })
    await waitFor(() => expect(hook.result.current.recoveries).toEqual(durable))
    await act(async () => {
      await hook.result.current.dismissRecovery(entry)
    })
    expect(mocks.dismissRecovery).toHaveBeenCalledExactlyOnceWith(
      entry.scope,
      expect.objectContaining({ owner_key: "owner", conversation_id: "chat" }),
      "retained-turn", "dismissed", expect.objectContaining({ validate_lease: expect.any(Function) })
    )
  }
)

it("local recovery remains available after remote account invalidation but refuses another profile's dismissal", async () => {
  const entry: Parameters<HistorySelectionController["dismissRecovery"]>[0] = {
    scope: { profile_id: "profile", client_session_id: "old-local-view" },
    turn: {
      origin_view: {
        view_session_id: "original",
        owner_key: "owner",
        conversation_id: "chat",
        interpretation: { kind: "parent_graph_v1" },
        cursor: { kind: "empty" },
        selection_revision: 0
      },
      selection_digest: "selection",
      request_context_digest: "request",
      created_at: 1,
      input_images: [],
      persistence: "server",
      operation_id: "local-turn",
      owner_key: "owner",
      conversation_id: "chat",
      state: "unknown",
      input_text: "local input",
      result_text: ""
    }
  }
  const hook = renderHook(() => useHistorySelection())
  await act(async () => {
    await hook.result.current.loadConversation({ serverChatId: "remote" })
  })
  const oldConfigChanged = mocks.configChanged!
  await act(async () => {
    await hook.result.current.open(owner)
  })
  mocks.recoveries.mockResolvedValue([entry])
  act(() => oldConfigChanged())
  await act(async () => {
    await hook.result.current.refreshRecovery()
  })
  expect(hook.result.current.recoveries).toEqual([entry])
  await act(async () => {
    await hook.result.current.dismissRecovery({
      ...entry,
      scope: { ...entry.scope, profile_id: "foreign" }
    })
  })
  expect(mocks.dismissRecovery).not.toHaveBeenCalled()
  await act(async () => {
    await hook.result.current.dismissRecovery(entry)
  })
  expect(mocks.dismissRecovery).toHaveBeenCalledOnce()
})
it("does not re-commit consumers when a load starts with nothing to clear", () => {
  // A commit here re-runs every consumer effect; effects that start a load
  // (Playground's settings return) would then loop without end.
  let commits = 0
  const { result } = renderHook(() => {
    React.useEffect(() => {
      commits += 1
    })
    return useHistorySelection()
  })
  const before = commits
  act(() => {
    result.current.beginLoad()
    result.current.beginLoad()
  })
  expect(commits).toBe(before)
})


it("following an edit result refuses a retired editor before capture", async () => {
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    await result.current.open(owner)
  })
  const origin = structuredClone(result.current.view!)
  mocks.capture.mockClear()
  await act(async () => {
    const follow = result.current.followResult as (
      view: HistoryViewSelectionV1,
      id: string,
      isCurrent: () => boolean
    ) => Promise<boolean>
    expect(await follow(origin, "a", () => false)).toBe(false)
  })
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(result.current.view).toEqual(origin)
})

it("following an edit result keeps the original view when typing retires a held capture", async () => {
  const { result } = renderHook(() => useHistorySelection())
  await act(async () => {
    await result.current.open(owner)
  })
  const origin = structuredClone(result.current.view!)
  const capture = mocks.capture.getMockImplementation()!
  let release!: () => void
  const held = new Promise<void>(done => {release = done})
  mocks.capture.mockImplementationOnce(async (...args) => {
    await held
    return capture(...args)
  })
  let current = true
  let attempt!: Promise<boolean>
  act(() => {
    const follow = result.current.followResult as (
      view: HistoryViewSelectionV1,
      id: string,
      isCurrent: () => boolean
    ) => Promise<boolean>
    attempt = follow(origin, "a", () => current)
  })
  try {
    expect(result.current.view).toEqual(origin)
  } finally {
    current = false
    await act(async () => {
      release()
      await attempt
    })
  }
  expect(await attempt).toBe(false)
  expect(result.current.view).toEqual(origin)
})

it("a current edit result installs once and acknowledges its own editor retirement", async () => {
  let current = true
  const onCapture = vi.fn(() => {
    current = false
  })
  const hook = renderHook(() => useHistorySelection({ onCapture }))
  await act(async () => {
    await hook.result.current.open(owner)
  })
  current = true
  onCapture.mockClear()
  const origin = hook.result.current.view!
  await act(async () => {
    expect(
      await hook.result.current.followResult(origin, "a", () => current)
    ).toBe(true)
  })
  expect(hook.result.current.view).toMatchObject({
    cursor: {
      kind: "after_message",
      message_id: "a"
    },
    selection_revision: origin.selection_revision + 1
  })
  expect(onCapture).toHaveBeenCalledOnce()
  expect(hook.result.current.status).toBe("ready")
})

it("an edit result that cannot be captured preserves the original selected view", async () => {
  const hook = renderHook(() => useHistorySelection())
  await act(async () => {
    await hook.result.current.open(owner)
  })
  const origin = hook.result.current.view!
  mocks.capture.mockImplementationOnce(async (_owner, view) => ({
    status: "invalid_history",
    code: "missing_message",
    snapshot,
    view,
    purpose: "send"
  }))
  await act(async () => {
    expect(
      await hook.result.current.followResult(origin, "a", () => true)
    ).toBe(false)
  })
  expect(hook.result.current.view).toBe(origin)
  expect(hook.result.current.status).toBe("ready")
})

it.each(["new-draft", "same-id-restore", "source-intent"])(
  "actual %s retirement during edit result capture cannot install an obsolete branch",
  async (boundary) => {
    const { captureChatDraftSource } = await import("../chat-draft-transition")
    const { useStoreMessageOption } = await import("@/store/option")
    const { usePlaygroundSessionStore } = await import(
      "@/store/playground-session"
    )
    const hook = renderHook(() => useHistorySelection())
    await act(async () => {
      await hook.result.current.open(owner)
    })
    const origin = hook.result.current.view!
    const source = captureChatDraftSource()
    const capture = mocks.capture.getMockImplementation()!
    let release!: () => void
    const held = new Promise<void>((done) => {
      release = done
    })
    mocks.capture.mockImplementationOnce(async (...args) => {
      await held
      return capture(...args)
    })
    let attempt!: Promise<boolean>
    act(() => {
      attempt = hook.result.current.followResult(origin, "a", source.isCurrent)
    })
    if (boundary === "new-draft")
      useStoreMessageOption.getState().startNewChatDraft()
    else if (boundary === "same-id-restore")
      usePlaygroundSessionStore.setState({
        restoreRevision:
          usePlaygroundSessionStore.getState().restoreRevision + 1
      })
    else
      usePlaygroundSessionStore.setState({
        sourceSelectionRevision:
          usePlaygroundSessionStore.getState().sourceSelectionRevision + 1
      })
    await act(async () => {
      release()
      expect(await attempt).toBe(false)
    })
    expect(hook.result.current.view).toBe(origin)
    expect(hook.result.current.status).toBe("ready")
  }
)

const savedEditedRecovery = (view: HistoryViewSelectionV1) => ({
  scope: { profile_id: "profile", client_session_id: "originating-session" },
  turn: {
    operation_id: "edited-operation",
    action: "edit_resend" as const,
    origin_view: view,
    owner_key: "owner",
    conversation_id: "chat",
    persistence: "server" as const,
    input_id: "u",
    assistant_id: "b",
    settled_message_id: "b",
    created_at: 1,
    input_text: "Edited question",
    input_images: [],
    result_text: "Other answer",
    state: "generated_unsaved" as const,
    outcome: "complete" as const,
    selection_digest: "selection",
    request_context_digest: "request",
    admission: {
      version: 1 as const,
      owner_key: "owner",
      conversation_id: "chat",
      input_message_id: "u",
      input_message_revision: "u1",
      selection_digest: "selection"
    }
  }
})
const editedRecoveryOwner = {
  kind: "native" as const,
  owner_key: "owner",
  conversation_id: "chat",
  request_scope: {} as never,
  validate_lease: () => true
}

it("explicit recovery opens a freshly reread completed edit result and dismisses only after capture", async () => {
  const hook = renderHook(() => useHistorySelection())
  await act(async () => {
    await hook.result.current.open(editedRecoveryOwner)
  })
  const entry = savedEditedRecovery(hook.result.current.view!)
  mocks.recoveries.mockResolvedValue([entry])
  mocks.capture.mockClear()
  await act(async () => {
    expect(
      await hook.result.current.openRecoveryResult(entry, () => true)
    ).toBe(true)
  })
  expect(mocks.capture).toHaveBeenCalledOnce()
  expect(hook.result.current.view?.cursor).toEqual({
    kind: "after_message",
    message_id: "b"
  })
  expect(mocks.dismissRecovery).toHaveBeenCalledWith(
    entry.scope,
    expect.objectContaining({ owner_key: "owner", conversation_id: "chat" }),
    "edited-operation",
    "completed"
  )
})

it.each([
  "foreign profile",
  "foreign owner",
  "changed result",
  "missing receipt",
  "incomplete",
  "ordinary",
  "retired source"
])("explicit recovery does not follow or dismiss after %s", async (reason) => {
  const hook = renderHook(() => useHistorySelection())
  await act(async () => {
    await hook.result.current.open(editedRecoveryOwner)
  })
  const origin = hook.result.current.view!
  const entry = savedEditedRecovery(origin)
  mocks.recoveries.mockResolvedValue(
    reason === "missing receipt"
      ? []
      : [
          reason === "changed result"
            ? { ...entry, turn: { ...entry.turn, settled_message_id: "a" } }
            : reason === "incomplete"
              ? { ...entry, turn: { ...entry.turn, outcome: "interrupted" } }
              : reason === "ordinary"
                ? { ...entry, turn: { ...entry.turn, action: undefined } }
                : entry
        ]
  )
  const offered =
    reason === "foreign profile"
      ? { ...entry, scope: { ...entry.scope, profile_id: "foreign" } }
      : reason === "foreign owner"
        ? { ...entry, turn: { ...entry.turn, owner_key: "foreign" } }
        : entry
  mocks.capture.mockClear()
  await act(async () => {
    expect(
      await hook.result.current.openRecoveryResult(
        offered,
        () => reason !== "retired source"
      )
    ).toBe(false)
  })
  expect(hook.result.current.view).toBe(origin)
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(mocks.dismissRecovery).not.toHaveBeenCalled()
})

it("an editor opened during held recovery capture preserves the view and recovery receipt", async () => {
  const hook = renderHook(() => useHistorySelection())
  await act(async () => {
    await hook.result.current.open(editedRecoveryOwner)
  })
  const origin = hook.result.current.view!
  const entry = savedEditedRecovery(origin)
  mocks.recoveries.mockResolvedValue([entry])
  const capture = mocks.capture.getMockImplementation()!
  let entered!: () => void, release!: () => void
  const ready = new Promise<void>((done) => {
    entered = done
  })
  const held = new Promise<void>(done => {release = done})
  mocks.capture.mockImplementationOnce(async (...args) => { entered(); await held; return capture(...args) })
  let current = true,
    attempt!: Promise<boolean>
  await act(async () => {
    attempt = hook.result.current.openRecoveryResult(entry, () => current)
    await ready
  })
  current = false
  await act(async () => {
    release()
    expect(await attempt).toBe(false)
  })
  expect(hook.result.current.view).toBe(origin)
  expect(mocks.dismissRecovery).not.toHaveBeenCalled()
})

it("a retained rendered recovery callback refuses an actual same-ID H1 refresh before invocation", async () => {
  const { HistorySelectionReview } = await import(
    "@/components/Common/Playground/HistorySelectionReview"
  )
  let controller!: HistorySelectionController
  function RecoveryView() {
    controller = useHistorySelection()
    return <HistorySelectionReview selection={controller} />
  }
  render(<RecoveryView />)
  await act(async () => {
    await controller.open(editedRecoveryOwner)
  })
  const entry = savedEditedRecovery(controller.view!)
  mocks.recoveries.mockResolvedValue([entry])
  await act(async () => {
    await controller.refreshRecovery()
  })
  const retained = mocks.openButtons.at(-1)!
  expect(retained).toBeTypeOf("function")
  await act(async () => {
    expect(await controller.refresh()).toBe(true)
  })
  const refreshed = controller.view
  mocks.capture.mockClear()
  await act(async () => {
    retained()
    await Promise.resolve()
  })
  expect(mocks.capture).not.toHaveBeenCalled()
  expect(mocks.dismissRecovery).not.toHaveBeenCalled()
  expect(controller.view).toBe(refreshed)
  await act(async () => {
    mocks.openButtons.at(-1)!()
    await Promise.resolve()
  })
  await waitFor(() => expect(mocks.dismissRecovery).toHaveBeenCalledOnce())
})


const mountNativeEditRefresh = async () => {
  let saved = false
  const originalCapture = mocks.capture.getMockImplementation()!
  mocks.capture.mockImplementation(async (...args) => {
    const capture = await originalCapture(...args)
    capture.snapshot = structuredClone(capture.snapshot)
    capture.rows = structuredClone(capture.rows)
    capture.selected_content = structuredClone(capture.selected_content)
    for (const node of [...capture.snapshot.nodes, ...capture.rows])
      if (node.id === "u") node.revision = saved ? "opaque-2" : "opaque-1"
    for (const row of capture.selected_content)
      if (row.id === "u")
        Object.assign(row, {
          revision: saved ? "opaque-2" : "opaque-1",
          message: saved ? "Edited {{char}}  \n" : "Original"
        })
    return capture
  })
  const onCapture = vi.fn()
  const hook = renderHook(() => useHistorySelection({ onCapture }))
  const native = {
    kind: "native" as const,
    owner_key: "owner",
    conversation_id: "chat",
    validate_lease: () => true,
    request_scope: {
      config: {
        serverUrl: "https://original.test",
        authMode: "multi-user" as const
      },
      userId: "alice"
    }
  }
  await act(async () => {
    expect(await hook.result.current.open(native)).toBe(true)
  })
  const origin = hook.result.current.getCurrent().capture!
  onCapture.mockClear()
  saved = true
  return {
    hook,
    native,
    origin,
    onCapture,
    receipt: { message_id: "u", message: "Edited {{char}}  \n", version: 2 }
  }
}

it("native Save refresh preserves the original cursor and certifies canonical raw text before installation", async () => {
  const mounted = await mountNativeEditRefresh()
  try {
    await act(async () => {
      expect(
        await (mounted.hook.result.current as any).refreshEditedMessage(
          mounted.origin,
          mounted.receipt,
          () => true
        )
      ).toBe(true)
    })
    expect(mounted.hook.result.current.view?.cursor).toEqual(
      mounted.origin.view.cursor
    )
    expect(mounted.hook.result.current.view?.selection_revision).toBe(
      mounted.origin.view.selection_revision
    )
    expect(mounted.onCapture).toHaveBeenCalledOnce()
    expect(mounted.hook.result.current.status).toBe("ready")
  } finally {
    mounted.hook.unmount()
  }
})

it.each(["capture", "bookmark"])(
  "native Save refresh keeps an edited draft when source retires during held %s",
  async (stage) => {
    const mounted = await mountNativeEditRefresh()
    let current = true,
      reached = false,
      release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    if (stage === "bookmark")
      mocks.beforeBookmark.mockImplementationOnce(async () => {
        reached = true
        await held
      })
    else {
      const original = mocks.capture.getMockImplementation()!
      mocks.capture.mockImplementationOnce(async (...args) => {
        reached = true
        await held
        return original(...args)
      })
    }
    let pending!: Promise<boolean>
    try {
      act(() => {
        pending = (mounted.hook.result.current as any).refreshEditedMessage(
          mounted.origin,
          mounted.receipt,
          () => current
        )
      })
      await waitFor(() => expect(reached).toBe(true))
      expect(mounted.hook.result.current.getCurrent().capture).toBe(
        mounted.origin
      )
      expect(mounted.hook.result.current.status).toBe("ready")
      current = false
      await act(async () => {
        release()
        expect(await pending).toBe(false)
      })
      expect(mounted.onCapture).not.toHaveBeenCalled()
      expect(mounted.hook.result.current.getCurrent().capture).toBe(
        mounted.origin
      )
    } finally {
      release?.()
      await pending?.catch(() => {})
      mounted.hook.unmount()
    }
  }
)

it("native Save refresh refuses a newer canonical target before publication", async () => {
  const mounted = await mountNativeEditRefresh()
  mounted.receipt.message = "Different raw text"
  try {
    await act(async () => {
      expect(
        await (mounted.hook.result.current as any).refreshEditedMessage(
          mounted.origin,
          mounted.receipt,
          () => true
        )
      ).toBe(false)
    })
    expect(mounted.onCapture).not.toHaveBeenCalled()
    expect(mounted.hook.result.current.getCurrent().capture).toBe(
      mounted.origin
    )
    expect(mounted.hook.result.current.status).toBe("ready")
  } finally {
    mounted.hook.unmount()
  }
})


it.each(["complete", "interrupted", "stopped"] as const)("ordinary server %s recovery opens its exact acknowledged reply", async outcome => {
  const hook = renderHook(() => useHistorySelection())
  await act(async () => { await hook.result.current.open(editedRecoveryOwner) })
  const origin = hook.result.current.view!, saved = savedEditedRecovery(origin)
  const entry = { ...saved, turn: { ...saved.turn, action: undefined, outcome } }
  mocks.recoveries.mockResolvedValue([entry]); mocks.capture.mockClear()
  try {
    await act(async () => { expect(await hook.result.current.openRecoveryResult(entry, () => true)).toBe(true) })
    expect(hook.result.current.view?.cursor).toEqual({ kind: "after_message", message_id: "b" })
    expect(mocks.dismissRecovery).toHaveBeenCalledWith(entry.scope, origin, entry.turn.operation_id, "completed")
  } finally { hook.unmount() }
})

it("explicit regeneration recovery opens the actual alternative reply without treating original input admission as a new send", async () => {
  const hook = renderHook(() => useHistorySelection())
  await act(async () => { await hook.result.current.open(editedRecoveryOwner) })
  const origin = hook.result.current.view!, edited = savedEditedRecovery(origin)
  const entry = { ...edited, turn: { ...edited.turn, action: "regenerate" as const, selection_digest: "REGENERATION SELECTION",
    input_text: "Original question", admission: { ...edited.turn.admission, selection_digest: "ORIGINAL ADMISSION" } } }
  mocks.recoveries.mockResolvedValue([entry]); mocks.capture.mockClear()
  await act(async () => { expect(await hook.result.current.openRecoveryResult(entry, () => true)).toBe(true) })
  expect(mocks.capture).toHaveBeenCalledOnce()
  expect(hook.result.current.view?.cursor).toEqual({ kind: "after_message", message_id: "b" })
  expect(mocks.dismissRecovery).toHaveBeenCalledWith(entry.scope, origin, entry.turn.operation_id, "completed")
  hook.unmount()
})


describe("owned native stale-send choices", () => {
  const nativeOwner = {
    kind: "native" as const,
    owner_key: "owner",
    conversation_id: "chat",
    request_scope: {
      config: {
        serverUrl: "https://chat.test",
        authMode: "single-user" as const,
        apiKey: "synthetic"
      }
    },
    validate_lease: () => true
  }
  const mountedAtEarlierAnswer = async () => {
    const hook = renderHook(() => useHistorySelection())
    await act(async () => {
      await hook.result.current.open(nativeOwner)
      await hook.result.current.choose({
        kind: "after_message",
        message_id: "a"
      })
    })
    return hook
  }
  it.each(["latest", "branch"] as const)(
    "an accepted %s choice supplies a live lease after retiring its entry fence",
    async (choice) => {
      const hook = await mountedAtEarlierAnswer()
      const origin = hook.result.current.view!
      const entryFence = hook.result.current.fence()
      const accepted = vi.fn<(isCurrent: () => boolean) => void>()
      act(() => {
        hook.result.current.holdStaleSend(origin, () => true)
      })
      await act(async () => {
        const result =
          choice === "latest"
            ? await hook.result.current.loadLatest(origin, () => true, accepted)
            : hook.result.current.branchHere(origin, () => true, accepted)
        expect(result).toBe(true)
      })
      expect(accepted).toHaveBeenCalledOnce()
      expect(entryFence()).toBe(false)
      expect(accepted.mock.calls[0][0]()).toBe(true)
      act(() => {
        hook.result.current.beginLoad()
      })
      expect(accepted.mock.calls[0][0]()).toBe(false)
      hook.unmount()
    }
  )

  const mountedRecoveryComposer = async () => {
    const { HistorySelectionReview } = await import(
      "@/components/Common/Playground/HistorySelectionReview"
    )
    const { useComposerEvents } = await import("@/hooks/useComposerEvents")
    const { useStoreMessageOption } = await import("@/store/option")
    useStoreMessageOption.getState().startNewChatDraft()
    useStoreMessageOption.setState({
      streaming: false,
      isLoading: false,
      isProcessing: false,
      historyId: null,
      serverChatId: null,
      history: [],
      messages: []
    })
    let selection!: HistorySelectionController
    const focusRequests = vi.fn()
    function Composer() {
      const controller = useHistorySelection()
      React.useLayoutEffect(() => {
        selection = controller
      }, [controller])
      const input = React.useRef<HTMLTextAreaElement>(null)
      const onFocusComposer = React.useCallback(() => {
        focusRequests()
        input.current?.focus()
      }, [])
      useComposerEvents({ onFocusComposer })
      return (
        <>
          <textarea
            aria-label="Recovery composer"
            ref={input}
            defaultValue={"Exact input.  \nSecond line."}
          />
          <button>Other control</button>
          <HistorySelectionReview selection={controller} />
        </>
      )
    }
    const mounted = render(<Composer />)
    await act(async () => {
      await selection.open(nativeOwner)
      await selection.choose({ kind: "after_message", message_id: "a" })
      selection.holdStaleSend(selection.getCurrent().view!, () => true)
    })
    return { mounted, controller: () => selection, focusRequests }
  }

  it.each(["Load latest messages", "Branch here"])(
    "%s returns focus to the exact-input composer once after actual H1 acceptance",
    async (label) => {
      const probe = await mountedRecoveryComposer()
      const button = screen.getByRole("button", { name: label })
      button.focus()
      fireEvent.click(button)
      const input = screen.getByRole("textbox", { name: "Recovery composer" })
      await waitFor(() => {
        expect(input).toHaveFocus()
      })
      expect(probe.focusRequests).toHaveBeenCalledOnce()
      expect(input).toHaveValue("Exact input.  \nSecond line.")
      expect(probe.controller().status).toBe("ready")
      expect(screen.queryByRole("button", { name: label })).toBeNull()
      probe.mounted.unmount()
    }
  )

  it.each(["latest", "branch"] as const)(
    "a failed local %s acceptance callback cannot undo the installed choice",
    async (choice) => {
      const hook = await mountedAtEarlierAnswer()
      const origin = hook.result.current.view!
      act(() => {
        hook.result.current.holdStaleSend(origin, () => true)
      })
      const failedFocus = () => {
        throw new Error("local focus unavailable")
      }
      await act(async () => {
        expect(
          choice === "latest"
            ? await hook.result.current.loadLatest(
                origin,
                () => true,
                failedFocus
              )
            : hook.result.current.branchHere(origin, () => true, failedFocus)
        ).toBe(true)
      })
      expect(hook.result.current.status).toBe("ready")
      expect(hook.result.current.sendConflict).toBe(false)
      expect(hook.result.current.view?.cursor).toEqual({
        kind: "after_message",
        message_id: choice === "latest" ? "b" : "a"
      })
      hook.unmount()
    }
  )

  it.each(
    ["Load latest messages", "Branch here"].flatMap((label) =>
      [
        "source",
        "H1",
        "New",
        "account",
        "busy",
        "editor",
        "unmount",
        "focus ABA",
        "pointer",
        "hidden"
      ].map((boundary) => [label, boundary])
    )
  )(
    "%s does not steal focus after %s changes before its accepted frame",
    async (label, boundary) => {
      const frames: FrameRequestCallback[] = []
      const raf = vi
        .spyOn(globalThis, "requestAnimationFrame")
        .mockImplementation((callback) => frames.push(callback))
      const cancel = vi
        .spyOn(globalThis, "cancelAnimationFrame")
        .mockImplementation(() => {})
      const probe = await mountedRecoveryComposer()
      const { useStoreMessageOption } = await import("@/store/option")
      const button = screen.getByRole("button", { name: label })
      const other = screen.getByRole("button", { name: "Other control" })
      const editor = document.createElement("form")
      let lease: ReturnType<typeof vi.spyOn> | undefined
      let visibility: ReturnType<typeof vi.spyOn> | undefined
      try {
        button.focus()
        fireEvent.click(button)
        await waitFor(() => {
          expect(probe.controller().status).toBe("ready")
        })
        expect(frames).toHaveLength(1)
        act(() => {
          if (boundary === "source")
            useStoreMessageOption.getState().startNewChatDraft()
          if (boundary === "H1") probe.controller().beginLoad()
          if (boundary === "New") probe.controller().reset()
          if (boundary === "account")
            lease = vi.spyOn(nativeOwner, "validate_lease").mockReturnValue(false)
          if (boundary === "busy")
            useStoreMessageOption.setState({ isProcessing: true })
          if (boundary === "editor") {
            editor.dataset.chatMessageEditor = ""
            document.body.append(editor)
          }
          if (boundary === "unmount") probe.mounted.unmount()
          if (boundary === "focus ABA") {
            other.focus()
            other.blur()
          }
          if (boundary === "pointer") fireEvent.pointerDown(other)
          if (boundary === "hidden")
            visibility = vi
              .spyOn(document, "visibilityState", "get")
              .mockReturnValue("hidden")
          frames[0](performance.now())
        })
        expect(probe.focusRequests).not.toHaveBeenCalled()
        if (boundary !== "unmount")
          expect(
            screen.getByRole("textbox", { name: "Recovery composer" })
          ).toHaveValue("Exact input.  \nSecond line.")
      } finally {
        editor.remove()
        lease?.mockRestore()
        visibility?.mockRestore()
        useStoreMessageOption.setState({ isProcessing: false })
        probe.mounted.unmount()
        raf.mockRestore()
        cancel.mockRestore()
      }
    }
  )

  it("a held Latest never takes focus after the user moves away and back to BODY", async () => {
    const frames: FrameRequestCallback[] = []
    const raf = vi
      .spyOn(globalThis, "requestAnimationFrame")
      .mockImplementation((callback) => frames.push(callback))
    const cancel = vi
      .spyOn(globalThis, "cancelAnimationFrame")
      .mockImplementation(() => {})
    const probe = await mountedRecoveryComposer()
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    const capture = mocks.capture.getMockImplementation()!
    mocks.capture.mockImplementationOnce(async (...args) => {
      await held
      return capture(...args)
    })
    try {
      const button = screen.getByRole("button", { name: "Load latest messages" })
      button.focus()
      fireEvent.click(button)
      const other = screen.getByRole("button", { name: "Other control" })
      other.focus()
      other.blur()
      expect(document.activeElement).toBe(document.body)
      await act(async () => {
        release()
      })
      await waitFor(() => {
        expect(probe.controller().status).toBe("ready")
      })
      expect(frames).toHaveLength(1)
      act(() => {
        frames[0](performance.now())
      })
      expect(probe.focusRequests).not.toHaveBeenCalled()
    } finally {
      release()
      probe.mounted.unmount()
      raf.mockRestore()
      cancel.mockRestore()
    }
  })

  it("a refused Latest capture leaves focus quiet and retains exact composer input", async () => {
    const probe = await mountedRecoveryComposer()
    mocks.capture.mockRejectedValueOnce(new Error("capture unavailable"))
    const button = screen.getByRole("button", { name: "Load latest messages" })
    button.focus()
    fireEvent.click(button)
    await waitFor(() => {
      expect(probe.controller().error).toBe("capture unavailable")
    })
    expect(probe.controller().status).toBe("stale_selection")
    expect(probe.focusRequests).not.toHaveBeenCalled()
    expect(
      screen.getByRole("textbox", { name: "Recovery composer" })
    ).toHaveValue("Exact input.  \nSecond line.")
    probe.mounted.unmount()
  })

  it("an accepted branch focus lease never adopts a newer branch at the same view", async () => {
    const hook = await mountedAtEarlierAnswer()
    const origin = hook.result.current.view!
    const accepted = vi.fn<(isCurrent: () => boolean) => void>()
    act(() => {
      hook.result.current.holdStaleSend(origin, () => true)
      hook.result.current.branchHere(origin, () => true, accepted)
    })
    const firstLease = accepted.mock.calls[0][0]
    expect(firstLease()).toBe(true)
    act(() => {
      hook.result.current.holdStaleSend(origin, () => true)
      hook.result.current.branchHere(origin, () => true)
    })
    expect(hook.result.current.branchPending).toBe(true)
    expect(firstLease()).toBe(false)
    hook.unmount()
  })

  it("an old accepted Branch frame stays quiet after a newer branch at the same view", async () => {
    const frames: FrameRequestCallback[] = []
    const raf = vi
      .spyOn(globalThis, "requestAnimationFrame")
      .mockImplementation((callback) => frames.push(callback))
    const cancel = vi
      .spyOn(globalThis, "cancelAnimationFrame")
      .mockImplementation(() => {})
    const probe = await mountedRecoveryComposer()
    try {
      const button = screen.getByRole("button", { name: "Branch here" })
      const origin = probe.controller().view!
      button.focus()
      fireEvent.click(button)
      expect(frames).toHaveLength(1)
      act(() => {
        probe.controller().holdStaleSend(origin, () => true)
        probe.controller().branchHere(origin, () => true)
        frames[0](performance.now())
      })
      expect(probe.controller().branchPending).toBe(true)
      expect(probe.focusRequests).not.toHaveBeenCalled()
    } finally {
      probe.mounted.unmount()
      raf.mockRestore()
      cancel.mockRestore()
    }
  })

  it("holds the current graph without capture and grants exactly one explicit branch", async () => {
    const hook = await mountedAtEarlierAnswer()
    const origin = hook.result.current.getCurrent().view!
    const capture = hook.result.current.getCurrent().capture
    const calls = mocks.capture.mock.calls.length
    act(() => {
      expect(hook.result.current.holdStaleSend(origin, () => true)).toBe(true)
    })
    expect(hook.result.current.status).toBe("stale_selection")
    expect(hook.result.current.getCurrent().capture).toBe(capture)
    expect(mocks.capture).toHaveBeenCalledTimes(calls)
    act(() => {
      expect(hook.result.current.branchHere(origin, () => true)).toBe(true)
    })
    expect(hook.result.current.status).toBe("ready")
    expect(hook.result.current.branchPending).toBe(true)
    act(() => {
      expect(hook.result.current.takeBranchIntent(origin)).toBe(true)
    })
    expect(hook.result.current.takeBranchIntent(origin)).toBe(false)
    expect(hook.result.current.getCurrent().view).toBe(origin)
    hook.unmount()
  })
  it.each(["source", "H1", "New", "unmount"])(
    "an armed branch retires across %s",
    async (boundary) => {
      const hook = await mountedAtEarlierAnswer()
      const origin = hook.result.current.getCurrent().view!
      let current = true
      act(() => {
        hook.result.current.holdStaleSend(origin, () => current)
        hook.result.current.branchHere(origin, () => current)
      })
      act(() => {
        if (boundary === "source") current = false
        if (boundary === "H1") hook.result.current.beginLoad()
        if (boundary === "New") hook.result.current.reset()
        if (boundary === "unmount") hook.unmount()
      })
      expect(hook.result.current.takeBranchIntent(origin)).toBe(false)
      if (boundary !== "unmount") hook.unmount()
    }
  )
  it.each(["source", "H1"])(
    "an obsolete %s conflict cannot replace the current view",
    async (boundary) => {
      const hook = await mountedAtEarlierAnswer()
      const origin = hook.result.current.getCurrent().view!
      const fence = hook.result.current.fence()
      if (boundary === "H1")
        act(() => {
          hook.result.current.beginLoad()
        })
      act(() => {
        expect(
          hook.result.current.holdStaleSend(
            origin,
            boundary === "source" ? () => false : fence
          )
        ).toBe(false)
      })
      expect(hook.result.current.status).toBe("ready")
      hook.unmount()
    }
  )
  it("load latest derives its boundary from a new owned graph and clears branch intent", async () => {
    const hook = await mountedAtEarlierAnswer()
    const origin = hook.result.current.getCurrent().view!
    act(() => {
      hook.result.current.holdStaleSend(origin, () => true)
    })
    const fresh = {
      ...snapshot,
      nodes: [
        ...snapshot.nodes,
        {
          ...snapshot.nodes[1],
          id: "latest",
          parent_id: "a",
          revision: "latest-1"
        }
      ]
    }
    mocks.capture.mockImplementation(async (_owner, selected) => {
      const resolved = resolveHistorySelection(fresh, selected, "send", "")
      if (resolved.status !== "ready") throw new Error(resolved.code)
      return {
        status: "captured",
        purpose: "send",
        snapshot: fresh,
        view: selected,
        rows: resolved.rows,
        selected_content: resolved.rows.map((row) => ({
          id: row.id,
          revision: row.revision,
          message: row.preview,
          images: []
        })),
        storage_context_digest: "storage"
      }
    })
    await act(async () => {
      expect(await hook.result.current.loadLatest(origin, () => true)).toBe(
        true
      )
    })
    expect(hook.result.current.getCurrent().view?.cursor).toEqual({
      kind: "after_message",
      message_id: "latest"
    })
    expect(hook.result.current.status).toBe("ready")
    expect(hook.result.current.branchPending).toBe(false)
    expect(
      hook.result.current.takeBranchIntent(
        hook.result.current.getCurrent().view!
      )
    ).toBe(false)
    hook.unmount()
  })
  it.each(["source", "H1", "New", "unmount"])(
    "a held latest capture cannot install after %s",
    async (boundary) => {
      const hook = await mountedAtEarlierAnswer()
      const origin = hook.result.current.getCurrent().view!
      act(() => {
        hook.result.current.holdStaleSend(origin, () => true)
      })
      let current = true,
        release!: () => void
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      const actualCapture = mocks.capture.getMockImplementation()!
      mocks.capture.mockImplementationOnce(async (...args) => {
        await held
        return actualCapture(...args)
      })
      let loading!: Promise<boolean>
      act(() => {
        loading = hook.result.current.loadLatest(origin, () => current)
      })
      act(() => {
        if (boundary === "source") current = false
        if (boundary === "H1") hook.result.current.beginLoad()
        if (boundary === "New") hook.result.current.reset()
        if (boundary === "unmount") hook.unmount()
      })
      await act(async () => {
        release()
        expect(await loading).toBe(false)
      })
      expect(hook.result.current.getCurrent().view?.cursor).not.toEqual({
        kind: "after_message",
        message_id: "b"
      })
      if (boundary === "source")
        expect(hook.result.current.status).toBe("stale_selection")
      if (boundary !== "unmount") hook.unmount()
    }
  )
})

it("an actual H1 branch survives a held compact focus capture and subsequent focus checks", async () => {
  const nativeOwner = {
    kind: "native" as const,
    owner_key: "owner",
    conversation_id: "chat",
    request_scope: {
      config: {
        serverUrl: "https://chat.test",
        authMode: "single-user" as const,
        apiKey: "synthetic"
      }
    },
    validate_lease: () => true
  }
  const hook = renderHook(() => {
    const selection = useHistorySelection()
    const freshness = useSidepanelChatFreshness({
      historySelection: selection,
      checkableTabId: () => "source"
    })
    return { selection, freshness }
  })
  await act(async () => {
    await hook.result.current.selection.open(nativeOwner)
    await hook.result.current.selection.choose({
      kind: "after_message",
      message_id: "a"
    })
  })
  vi.useFakeTimers()
  try {
    const origin = hook.result.current.selection.getCurrent().view!
    const actual = mocks.capture.getMockImplementation()!
    let release!: () => void
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    mocks.capture.mockImplementationOnce(async (...args) => {
      await held
      return actual(...args)
    })
    await act(async () => {
      window.dispatchEvent(new Event("focus"))
      await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
    })
    const reads = mocks.capture.mock.calls.length
    act(() => {
      hook.result.current.selection.holdStaleSend(origin, () => true)
      hook.result.current.selection.branchHere(origin, () => true)
    })
    await act(async () => {
      release()
      await Promise.resolve()
      await vi.advanceTimersByTimeAsync(FOCUS_CHECK_MIN_INTERVAL_MS)
      window.dispatchEvent(new Event("focus"))
      await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
    })
    expect(hook.result.current.selection.getCurrent().view).toBe(origin)
    expect(hook.result.current.selection.branchPending).toBe(true)
    expect(mocks.capture).toHaveBeenCalledTimes(reads)
    expect(hook.result.current.freshness.notice).toBeNull()
  } finally {
    hook.unmount()
    vi.useRealTimers()
  }
})


describe("shared proactive native freshness", () => {
  const nativeOwner = {
    kind: "native" as const,
    owner_key: "owner",
    conversation_id: "chat",
    request_scope: {
      config: {
        serverUrl: "https://chat.test",
        authMode: "single-user" as const,
        apiKey: "synthetic"
      }
    },
    validate_lease: () => true
  }
  const mount = async () => {
    const { useStoreMessageOption } = await import("@/store/option")
    useStoreMessageOption.setState({
      isLoading: false,
      isProcessing: false,
      streaming: false,
      messages: [],
      chatDraftAddress: {
        kind: "native",
        id: "chat",
        ownerKey: "owner",
        scope: { type: "global" }
      }
    })
    mocks.historyVersion.mockReset().mockResolvedValue(1)
    const hook = renderHook(() => useHistorySelection())
    await act(async () => {
      await hook.result.current.open(nativeOwner)
      await hook.result.current.choose({
        kind: "after_message",
        message_id: "a"
      })
    })
    return hook
  }
  const changed = () => {
    const original = mocks.capture.getMockImplementation()!
    mocks.capture.mockImplementation(async (...args) => {
      const result = await original(...args)
      return {
        ...result,
        snapshot: {
          ...result.snapshot,
          fences: { ...result.snapshot.fences, history: "2" }
        }
      }
    })
    mocks.historyVersion.mockResolvedValue(2)
  }
  it("an unchanged detail requires no graph capture or display installation", async () => {
    const hook = await mount(),
      original = hook.result.current.getCurrent(),
      reads = mocks.capture.mock.calls.length
    await act(async () => {
      expect(await hook.result.current.checkFreshness(() => true)).toBe(false)
    })
    expect(mocks.capture).toHaveBeenCalledTimes(reads)
    expect(mocks.historyVersion).toHaveBeenCalledOnce()
    expect(hook.result.current.getCurrent()).toBe(original)
    hook.unmount()
  })
  it("same-ID history changes hold a passive choice without moving or saving the view", async () => {
    const hook = await mount(),
      origin = hook.result.current.getCurrent().view!,
      bookmarks = [...mocks.bookmarks.values()],
      reads = mocks.capture.mock.calls.length
    changed()
    await act(async () => {
      expect(await hook.result.current.checkFreshness(() => true)).toBe(true)
    })
    expect(hook.result.current.getCurrent().view).toBe(origin)
    expect(hook.result.current.freshnessConflict).toBe(true)
    expect(mocks.capture).toHaveBeenCalledTimes(reads + 1)
    expect([...mocks.bookmarks.values()]).toEqual(bookmarks)
    expect(mocks.dismissRecovery).not.toHaveBeenCalled()
    hook.unmount()
  })
  it.each(["GET", "manifest"])(
    "held %s cannot publish after original eligibility retires",
    async (boundary) => {
      const hook = await mount(),
        origin = hook.result.current.getCurrent(),
        reads = mocks.capture.mock.calls.length
      changed()
      let release!: () => void,
        current = true
      const held = new Promise<void>((resolve) => {
        release = resolve
      })
      if (boundary === "GET")
        mocks.historyVersion.mockImplementationOnce(async () => {
          await held
          return 2
        })
      else {
        const original = mocks.capture.getMockImplementation()!
        mocks.capture.mockImplementationOnce(async (...args) => {
          await held
          return original(...args)
        })
      }
      let pending!: Promise<boolean>
      act(() => {
        pending = hook.result.current.checkFreshness(() => current)
      })
      await act(async () => {
        await Promise.resolve()
      })
      current = false
      await act(async () => {
        release()
        expect(await pending).toBe(false)
      })
      expect(hook.result.current.getCurrent()).toBe(origin)
      if (boundary === "GET") expect(mocks.capture).toHaveBeenCalledTimes(reads)
      hook.unmount()
    }
  )
  it("a restored old native tip exposes the existing freshness choice even when its capture version is current", async () => {
    const { useStoreMessageOption } = await import("@/store/option")
    useStoreMessageOption.setState({
      isLoading: false, isProcessing: false, streaming: false, messages: [],
      chatDraftAddress: { kind: "native", id: "chat", ownerKey: "owner", scope: { type: "global" } }
    })
    const hook = renderHook(() => {
      const selection = useHistorySelection()
      const freshness = useSidepanelChatFreshness({ historySelection: selection, checkableTabId: () => "active" })
      return { selection, freshness }
    })
    try {
      await act(async () => {
        await hook.result.current.selection.open(nativeOwner)
        await hook.result.current.selection.choose({ kind: "after_message", message_id: "a" })
      })
      const origin = hook.result.current.selection.getCurrent().view!
      const capture = hook.result.current.selection.getCurrent().capture
      const reads = mocks.capture.mock.calls.length
      await act(async () => {
        hook.result.current.freshness.showTab("a")
        hook.result.current.freshness.checkShownTab()
      })
      expect(hook.result.current.selection.freshnessConflict).toBe(true)
      expect(hook.result.current.selection.getCurrent().view).toBe(origin)
      expect(hook.result.current.selection.getCurrent().capture).toBe(capture)
      expect(mocks.capture).toHaveBeenCalledTimes(reads)
      expect(hook.result.current.freshness.notice).toBeNull()
    } finally { hook.unmount() }
  })

  it("compact focus cannot automatically choose a new parent", async () => {
    vi.useFakeTimers()
    const hook = renderHook(() => {
      const selection = useHistorySelection()
      const freshness = useSidepanelChatFreshness({
        historySelection: selection,
        checkableTabId: () => "active"
      })
      return { selection, freshness }
    })
    try {
      const { useStoreMessageOption } = await import("@/store/option")
      useStoreMessageOption.setState({
        isLoading: false,
        isProcessing: false,
        streaming: false,
        messages: [],
        chatDraftAddress: {
          kind: "native",
          id: "chat",
          ownerKey: "owner",
          scope: { type: "global" }
        }
      })
      mocks.historyVersion.mockReset().mockResolvedValue(2)
      await act(async () => {
        await hook.result.current.selection.open(nativeOwner)
        await hook.result.current.selection.choose({
          kind: "after_message",
          message_id: "a"
        })
      })
      const origin = hook.result.current.selection.getCurrent().view!
      const next = {
        ...snapshot,
        fences: { ...snapshot.fences, history: "2" },
        nodes: [
          ...snapshot.nodes,
          { ...snapshot.nodes[1], id: "latest", parent_id: "a" }
        ]
      }
      mocks.capture.mockImplementation(async (_owner, view) => {
        const resolved = resolveHistorySelection(next, view, "send", "")
        if (resolved.status !== "ready") throw Error(resolved.code)
        return {
          status: "captured",
          snapshot: next,
          view,
          rows: resolved.rows,
          selected_content: resolved.rows.map((row) => ({
            id: row.id,
            revision: row.revision,
            message: row.preview,
            images: []
          })),
          purpose: "send",
          storage_context_digest: "storage"
        }
      })
      await act(async () => {
        window.dispatchEvent(new Event("focus"))
        await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
      })
      expect(hook.result.current.selection.getCurrent().view).toBe(origin)
      expect(hook.result.current.selection.freshnessConflict).toBe(true)
    } finally {
      hook.unmount()
      vi.useRealTimers()
    }
  })
  it("Load latest refuses before capture while a message edit is open", async () => {
    const hook = await mount(),
      origin = hook.result.current.getCurrent().view!,
      reads = mocks.capture.mock.calls.length
    act(() => {
      hook.result.current.holdStaleSend(origin, () => true)
    })
    const editor = document.createElement("form")
    editor.dataset.chatMessageEditor = ""
    document.body.append(editor)
    try {
      await act(async () => {
        expect(await hook.result.current.loadLatest(origin, () => true)).toBe(
          false
        )
      })
      expect(mocks.capture).toHaveBeenCalledTimes(reads)
      expect(hook.result.current.getCurrent().view).toBe(origin)
    } finally {
      editor.remove()
      hook.unmount()
    }
  })
  it("a message editor opened during Load latest keeps the original captured view", async () => {
    const hook = await mount(),
      original = hook.result.current.getCurrent(),
      origin = original.view!
    act(() => {
      hook.result.current.holdStaleSend(origin, () => true)
    })
    let release!: () => void
    const held = new Promise<void>((resolve) => {
        release = resolve
      }),
      capture = mocks.capture.getMockImplementation()!
    mocks.capture.mockImplementationOnce(async (...args) => {
      await held
      return capture(...args)
    })
    let loading!: Promise<boolean>
    act(() => {
      loading = hook.result.current.loadLatest(origin, () => true)
    })
    const editor = document.createElement("form")
    editor.dataset.chatMessageEditor = ""
    document.body.append(editor)
    try {
      await act(async () => {
        release()
        expect(await loading).toBe(false)
      })
      expect(hook.result.current.getCurrent().view).toBe(origin)
      expect(hook.result.current.getCurrent().capture).toBe(original.capture)
      expect(hook.result.current.status).toBe("stale_selection")
    } finally {
      editor.remove()
      hook.unmount()
    }
  })
  it.each(["New", "H1", "draft", "owner", "busy", "editor", "unmount"])(
    "a held detail cannot publish across %s",
    async (boundary) => {
      const hook = await mount(),
        reads = mocks.capture.mock.calls.length
      let release!: () => void,
        valid = true
      const held = new Promise<void>((resolve) => {
          release = resolve
        }),
        validate = nativeOwner.validate_lease
      nativeOwner.validate_lease = () => valid
      mocks.historyVersion.mockImplementationOnce(async () => {
        await held
        return 2
      })
      let pending!: Promise<boolean>
      act(() => {
        pending = hook.result.current.checkFreshness(() => true)
      })
      const { useStoreMessageOption } = await import("@/store/option")
      const editor = document.createElement("form")
      editor.dataset.chatMessageEditor = ""
      act(() => {
        if (boundary === "New") hook.result.current.reset()
        if (boundary === "H1") hook.result.current.beginLoad()
        if (boundary === "draft")
          useStoreMessageOption.getState().startNewChatDraft()
        if (boundary === "owner") valid = false
        if (boundary === "busy")
          useStoreMessageOption.setState({ isProcessing: true })
        if (boundary === "editor") document.body.append(editor)
        if (boundary === "unmount") hook.unmount()
      })
      try {
        await act(async () => {
          release()
          expect(await pending).toBe(false)
        })
        expect(mocks.capture).toHaveBeenCalledTimes(reads)
        expect(hook.result.current.freshnessConflict).toBe(false)
      } finally {
        nativeOwner.validate_lease = validate
        editor.remove()
        hook.unmount()
        useStoreMessageOption.setState({ isProcessing: false })
      }
    }
  )
  it("an armed explicit branch does not read or change its intent on focus checks", async () => {
    const hook = await mount(),
      origin = hook.result.current.getCurrent().view!
    act(() => {
      hook.result.current.holdStaleSend(origin, () => true)
      hook.result.current.branchHere(origin, () => true)
    })
    await act(async () => {
      expect(await hook.result.current.checkFreshness(() => true)).toBe(false)
    })
    expect(mocks.historyVersion).not.toHaveBeenCalled()
    expect(hook.result.current.branchPending).toBe(true)
    expect(hook.result.current.getCurrent().view).toBe(origin)
    hook.unmount()
  })
  it("unavailable detail leaves saved state and recovery untouched", async () => {
    const hook = await mount(),
      original = hook.result.current.getCurrent(),
      reads = mocks.capture.mock.calls.length
    mocks.historyVersion.mockRejectedValueOnce(
      Error("unsupported_history_version")
    )
    await act(async () => {
      expect(await hook.result.current.checkFreshness(() => true)).toBe(false)
    })
    expect(hook.result.current.getCurrent()).toBe(original)
    expect(mocks.capture).toHaveBeenCalledTimes(reads)
    expect(mocks.dismissRecovery).not.toHaveBeenCalled()
    hook.unmount()
  })

  it("the active full-view hook coalesces focus/visibility and skips unchanged graph reads", async () => {
    vi.useFakeTimers()
    const hook = renderHook(() => {
      const selection = useHistorySelection()
      useChatHistoryFreshness(selection)
      return selection
    })
    try {
      const { useStoreMessageOption } = await import("@/store/option")
      useStoreMessageOption.setState({
        isLoading: false,
        isProcessing: false,
        streaming: false,
        chatDraftAddress: {
          kind: "native",
          id: "chat",
          ownerKey: "owner",
          scope: { type: "global" }
        }
      })
      mocks.historyVersion.mockReset().mockResolvedValue(1)
      await act(async () => {
        await hook.result.current.open(nativeOwner)
        await hook.result.current.choose({
          kind: "after_message",
          message_id: "a"
        })
      })
      const reads = mocks.capture.mock.calls.length
      await act(async () => {
        window.dispatchEvent(new Event("focus"))
        document.dispatchEvent(new Event("visibilitychange"))
        await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
      })
      expect(mocks.historyVersion).toHaveBeenCalledOnce()
      await act(async () => {
        window.dispatchEvent(new Event("focus"))
        await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
      })
      expect(mocks.historyVersion).toHaveBeenCalledOnce()
      await act(async () => {
        await vi.advanceTimersByTimeAsync(FOCUS_CHECK_MIN_INTERVAL_MS)
        document.dispatchEvent(new Event("visibilitychange"))
        await vi.advanceTimersByTimeAsync(FOCUS_CHECK_DEBOUNCE_MS)
      })
      expect(mocks.historyVersion).toHaveBeenCalledTimes(2)
      expect(mocks.capture).toHaveBeenCalledTimes(reads)
    } finally {
      hook.unmount()
      vi.useRealTimers()
    }
  })
  it("an opening-only controller remains passive on focus", async () => {
    const hook = await mount()
    act(() => {
      window.dispatchEvent(new Event("focus"))
      document.dispatchEvent(new Event("visibilitychange"))
    })
    expect(mocks.historyVersion).not.toHaveBeenCalled()
    hook.unmount()
  })
})


describe("acknowledged recovery controller", () => {
  const mountRecovery = async () => {
    const hook = renderHook(() => useHistorySelection())
    await act(async () => {
      await hook.result.current.open(owner)
    })
    const entry: Parameters<HistorySelectionController["dismissRecovery"]>[0] =
      {
        scope: { profile_id: "profile", client_session_id: "old-view" },
        turn: {
          operation_id: "dismiss-ack",
          owner_key: "owner",
          conversation_id: "chat",
          origin_view: hook.result.current.view!,
          input_id: "input",
          created_at: 1,
          input_text: "  exact input\n  ",
          input_images: [],
          result_text: "  exact reply\n  ",
          state: "generated_unsaved",
          selection_digest: "selected",
          request_context_digest: "prepared"
        }
      }
    mocks.recoveries.mockResolvedValue([entry])
    await act(async () => {
      await hook.result.current.refreshRecovery()
    })
    return { hook, entry }
  }

  it.each(["refused", "failed"])(
    "retains recovery text and reports no acknowledgement when dismissal is %s",
    async (mode) => {
      const { hook, entry } = await mountRecovery()
      if (mode === "failed")
        mocks.dismissRecovery.mockRejectedValueOnce(
          new Error("journal unavailable")
        )
      else mocks.dismissRecovery.mockResolvedValueOnce(false)
      let receipt: unknown
      await act(async () => {
        receipt = await hook.result.current.dismissRecovery(entry)
      })
      expect(receipt).toBe(false)
      expect(hook.result.current.recoveries).toEqual([entry])
      expect(hook.result.current.recoveryError).toBe(
        mode === "failed"
          ? "journal unavailable"
          : "recovery_dismissal_not_committed"
      )
      hook.unmount()
    }
  )

  it.each(["caller", "same-ID reset", "owner lease", "unmount"])(
    "cannot acknowledge or remove an old recovery after %s during storage",
    async (boundary) => {
      const { hook, entry } = await mountRecovery()
      let callerCurrent = true
      let leaseCurrent = true
      let finish!: () => void
      let entered!: () => void
      const ready = new Promise<void>((resolve) => {
        entered = resolve
      })
      const leasedOwner = { ...owner, validate_lease: () => leaseCurrent }
      await act(async () => {
        await hook.result.current.open(leasedOwner)
        await hook.result.current.refreshRecovery()
      })
      const durable = [entry]
      mocks.dismissRecovery.mockImplementationOnce(
        async (_scope, _view, _id, _outcome, options) => {
          entered()
          await new Promise<void>((resolve) => {
            finish = resolve
          })
          if (options?.signal?.aborted || options?.validate_lease?.() === false)
            return false
          durable.splice(0)
          return true
        }
      )
      let pending!: Promise<boolean>
      act(() => {
        pending = hook.result.current.dismissRecovery(
          entry,
          () => callerCurrent
        )
      })
      await ready
      if (boundary === "caller") callerCurrent = false
      else if (boundary === "owner lease") leaseCurrent = false
      else if (boundary === "unmount") hook.unmount()
      else
        await act(async () => {
          hook.result.current.beginLoad()
          await hook.result.current.open(leasedOwner)
          await hook.result.current.refreshRecovery()
        })
      let receipt: unknown
      await act(async () => {
        finish()
        receipt = await pending
      })
      expect(receipt).toBe(false)
      expect(durable).toEqual([entry])
      if (boundary === "same-ID reset") {
        expect(hook.result.current.recoveries).toEqual([entry])
        expect(hook.result.current.recoveryError).toBeNull()
      }
      hook.unmount()
    }
  )

  it("refuses an old rendered caller before it can adopt a replacement same-ID epoch", async () => {
    const { hook, entry } = await mountRecovery()
    const old = hook.result.current.dismissRecovery
    const source = hook.result.current.fence()
    await act(async () => {
      hook.result.current.beginLoad()
      await hook.result.current.open(owner)
    })
    await act(async () => {
      expect(await old(entry, source)).toBe(false)
    })
    expect(mocks.dismissRecovery).not.toHaveBeenCalled()
    hook.unmount()
  })
})
