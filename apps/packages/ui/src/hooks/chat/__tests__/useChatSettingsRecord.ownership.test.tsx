import { act, renderHook, waitFor } from "@testing-library/react"
import { afterEach, beforeEach, expect, it, vi } from "vitest"
import { Storage } from "@plasmohq/storage"
import { useStoreMessageOption } from "@/store/option"
import { usePlaygroundSessionStore } from "@/store/playground-session"
import { freshChatDraftAddress } from "@/utils/chat-draft-address"
import type { NativeHistoryOwnerV1 } from "@/services/chat-history-selection"

type SettingsController = {
  settingsMode: () => "ordinary"
  getCurrent: () => { owner: NativeHistoryOwnerV1 | null }
  getSignal: () => undefined
  fence: () => () => boolean
}
const state = vi.hoisted(() => ({ owner: null as NativeHistoryOwnerV1 | null, controller: null as SettingsController | null, epoch: 0, write: vi.fn() }))
vi.mock("../useHistorySelection", async (importOriginal) => ({
  ...await importOriginal<typeof import("../useHistorySelection")>(),
  useHistorySelectionContext: () => state.controller
}))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: {} }))
vi.mock("@/services/chat-settings", async (importOriginal) => ({
  ...await importOriginal<typeof import("@/services/chat-settings")>(),
  applyChatSettingsPatch: (...args: unknown[]) => state.write(...args)
}))
import { getChatSettingsStorageKey, resolveChatSettingsKey } from "@/services/chat-settings"
import { useChatSettingsRecord } from "../useChatSettingsRecord"

const requestScope = { config: { serverUrl: "https://settings-test.example", authMode: "single-user" as const }, userId: 1 }
const owner = (id: string): NativeHistoryOwnerV1 => ({ kind: "native", conversation_id: id, owner_key: "verified-owner", request_scope: requestScope, validate_lease: () => true })
const key = (id: string) => getChatSettingsStorageKey(resolveChatSettingsKey({ historyId: null, serverChatId: id, requestScope }))
const settings = (note: string) => ({ schemaVersion: 2, updatedAt: "2026-10-08T00:00:00Z", authorNote: note })
const deferred = () => { let release!: () => void; const promise = new Promise<void>(resolve => { release = resolve }); return { promise, release } }

beforeEach(async () => {
  localStorage.clear(); sessionStorage.clear(); state.epoch = 0; state.write.mockReset()
  state.owner = owner("A")
  state.controller = { settingsMode: () => "ordinary", getCurrent: () => ({ owner: state.owner }), getSignal: () => undefined,
    fence: () => { const epoch = state.epoch; return () => epoch === state.epoch } }
  useStoreMessageOption.setState({ historyId: null, serverChatId: "A", chatDraftAddress: null, chatDraftRevision: 0 })
  await new Storage({ area: "local" }).set(key("A"), { value: settings("Alpha"), updatedAt: 1 })
})
afterEach(() => { vi.restoreAllMocks(); vi.unstubAllGlobals() })

it.each([null, settings("Beta")])("never presents a warm previous chat while the new key read is held, next=%j", async (next) => {
  await new Storage({ area: "local" }).set(key("B"), { value: next, updatedAt: 2 })
  const hook = renderHook(({ id }) => useChatSettingsRecord({ historyId: null, serverChatId: id }), { initialProps: { id: "A" } })
  await waitFor(() => expect(hook.result.current.settings?.authorNote).toBe("Alpha"))
  const entered = deferred(), held = deferred()
  const read = Storage.prototype.get
  vi.spyOn(Storage.prototype, "get").mockImplementation(async function(this: Storage, storageKey: string) {
    if (storageKey === key("B")) { entered.release(); await held.promise }
    return await read.call(this, storageKey)
  })
  act(() => { state.owner = owner("B"); state.epoch++; useStoreMessageOption.setState({ serverChatId: "B" }); hook.rerender({ id: "B" }) })
  await entered.promise
  expect(hook.result.current.settings).toBeNull()
  await act(async () => held.release())
  await waitFor(() => expect(hook.result.current.settings).toEqual(next))
})

it.each([false, true])("rejects a rendered callback after synchronous source replacement before React commits, same IDs=%s", async (sameIds) => {
  const address = freshChatDraftAddress(false)
  useStoreMessageOption.setState({ chatDraftAddress: address })
  const hook = renderHook(() => useChatSettingsRecord({ historyId: null, serverChatId: "A" }))
  await waitFor(() => expect(hook.result.current.settings?.authorNote).toBe("Alpha"))
  const staleWrite = hook.result.current.updateSettings
  await act(async () => {
    useStoreMessageOption.setState({ chatDraftAddress: freshChatDraftAddress(false), chatDraftRevision: 1, ...(sameIds ? {} : { serverChatId: "B" }) })
    await expect(staleWrite({ authorNote: "Must not reach A" })).rejects.toThrow("chat_settings_owner_unavailable")
  })
  expect(state.write).not.toHaveBeenCalled()
  expect(await new Storage({ area: "local" }).get(key("A"))).toMatchObject({ value: settings("Alpha") })
})

it.each(["restore", "selection"])("rejects a retained callback after synchronous %s retirement before React commits", async change => {
  const hook = renderHook(() => useChatSettingsRecord({ historyId: null, serverChatId: "A" }))
  await waitFor(() => expect(hook.result.current.settings?.authorNote).toBe("Alpha"))
  const staleWrite = hook.result.current.updateSettings
  await act(async () => {
    const session = usePlaygroundSessionStore.getState()
    if (change === "restore") session.cancelPendingRestore()
    else session.markSourceSelectionIntent()
    await expect(staleWrite({ authorNote: "Must not adopt a newer view" })).rejects.toThrow("chat_settings_owner_unavailable")
  })
  expect(state.write).not.toHaveBeenCalled()
})
