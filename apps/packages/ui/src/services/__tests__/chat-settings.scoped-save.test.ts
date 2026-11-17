import { afterEach, beforeEach, describe, expect, it, vi } from "vitest"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"

const state = vi.hoisted(() => ({ records: new Map<string, unknown>(), remote: {} as Record<string, unknown>, cacheFails: false }))
const transport = vi.hoisted(() => ({ initialize: vi.fn(), getChatSettings: vi.fn(), updateChatSettings: vi.fn() }))
vi.mock("@/utils/safe-storage", () => ({ createSafeStorage: () => ({
  hasPersistentBackend: true,
  get: async (key: string) => structuredClone(state.records.get(key)),
  set: async (key: string, value: unknown) => {
    if (state.cacheFails && key.startsWith("chatSettings:")) throw new Error("cache blocked")
    state.records.set(key, structuredClone(value))
  },
  remove: async (key: string) => { state.records.delete(key) }
}) }))
vi.mock("@/services/tldw/TldwApiClient", () => ({ tldwClient: transport }))
import { getChatSettingsRecovery, applyChatSettingsPatch, getChatSettingsStorageKey, normalizeChatSettingsRecord, resolveChatSettingsKey, syncChatSettingsForServerChat } from "../chat-settings"

const config = { serverUrl: "https://one.example", authMode: "single-user" as const, apiKey: "disposable-test-key" }
const requestScope: ServicePromptRequestScope = { config, userId: 1 }
const owner = { requestScope, validate_lease: () => true }
const params = { historyId: null, serverChatId: "same-id", ...owner }
const canonical = { schemaVersion: 2, updatedAt: "2026-10-08T00:00:00Z", authorNote: "Server", conversationContext: { world_book_ids: [8], chat_dictionary_ids: [7] }, chat_dictionary_ids: [7] }
const deferred = () => { let release!: () => void; const promise = new Promise<void>(resolve => { release = resolve }); return { promise, release } }

beforeEach(() => {
  state.records.clear(); state.remote = structuredClone(canonical); state.cacheFails = false
  vi.resetAllMocks()
  transport.initialize.mockResolvedValue(undefined)
  transport.getChatSettings.mockImplementation(async (id: string) => ({ conversation_id: id, settings: structuredClone(state.remote) }))
  transport.updateChatSettings.mockImplementation(async (id: string, patch: Record<string, unknown>) => {
    state.remote = { ...state.remote, ...structuredClone(patch) }
    return { conversation_id: id, settings: structuredClone(state.remote) }
  })
  const tails = new Map<string, Promise<unknown>>()
  vi.stubGlobal("navigator", Object.create(window.navigator, { locks: { value: {
    request: (key: string, operation: () => unknown) => { const next = (tails.get(key) ?? Promise.resolve()).then(operation); tails.set(key, next.catch(() => undefined)); return next }
  } } }))
})
afterEach(() => vi.unstubAllGlobals())

describe("ordinary settings captured owner and explicit reconciliation", () => {
  it.each([false, true])("read-only recovery never resends an unsupported retained change (already applied=%s)", async committed => {
    const patch = { authorNote: "Legacy unsupported note" }
    transport.updateChatSettings.mockImplementationOnce(async () => {
      if (committed) state.remote = { ...state.remote, ...patch }
      throw new Error("Lost response")
    })
    await expect(applyChatSettingsPatch({ ...params, patch })).rejects.toThrow("chat_settings_server_unconfirmed")
    const recovery = await getChatSettingsRecovery(params)
    expect(recovery).not.toBeNull()
    const verify = applyChatSettingsPatch({ ...params, patch, recoveryOperationId: recovery!.operationId, reconcileOnly: true })
    if (committed) {
      await expect(verify).resolves.toMatchObject(patch)
      expect(await getChatSettingsRecovery(params)).toBeNull()
    } else {
      await expect(verify).rejects.toThrow("chat_settings_recovery_requires_compatible_path")
      expect(await getChatSettingsRecovery(params)).toEqual(recovery)
    }
    expect(transport.updateChatSettings).toHaveBeenCalledTimes(1)
  })

  it("preserves canonical legacy generation and supported participant fields through normalization", () => {
    const supported = { generationOverrides: { temperature: 0, top_p: 0.8 }, turnTakingMode: "single", participantCharacterIds: [1] }
    expect(normalizeChatSettingsRecord({ ...canonical, ...supported, unknownControl: "unsupported" }))
      .toEqual({ ...canonical, ...supported })
  })

  it("acknowledges only the supported numeric participant and turn-taking intent", async () => {
    const patch = { participantCharacterIds: [1], turnTakingMode: "single" as const }
    expect(await applyChatSettingsPatch({ ...params, patch } as Parameters<typeof applyChatSettingsPatch>[0]))
      .toMatchObject(patch)
    expect(transport.updateChatSettings).toHaveBeenCalledWith("same-id", expect.objectContaining(patch), expect.any(Object))
  })

  it("preserves legacy sampling siblings when a partial edit creates the preferred override", async () => {
    state.remote = { ...canonical, generationOverrides: { temperature: 0.5, top_p: 0.9, repetition_penalty: 1.1, stop: ["END"] } }
    const saved = await applyChatSettingsPatch({ ...params, patch: { chatGenerationOverride: { temperature: 0.7 } } })
    expect(saved?.chatGenerationOverride).toEqual({ temperature: 0.7, top_p: 0.9, repetition_penalty: 1.1, stop: ["END"] })
  })

  it.each([{}, { enabled: false, top_p: null }])("keeps a preferred override authoritative over the legacy alias (%j)", async preferred => {
    state.remote = { ...canonical, chatGenerationOverride: preferred, generationOverrides: { top_p: 0.9, stop: ["LEGACY"] } }
    const saved = await applyChatSettingsPatch({ ...params, patch: { chatGenerationOverride: { temperature: 0 } } })
    expect(saved?.chatGenerationOverride).toEqual({ ...preferred, temperature: 0 })
  })

  it("does not promote malformed supported-field shapes or unknown controls", () => {
    expect(normalizeChatSettingsRecord({ ...canonical, generationOverrides: [], turnTakingMode: "unknown", participantCharacterIds: [{}], unknownControl: "unsupported" }))
      .toEqual(canonical)
  })

  it("keeps identical chat IDs separate across account, server and workspace and stores no credentials in keys", () => {
    const keys = [
      resolveChatSettingsKey(params),
      resolveChatSettingsKey({ ...params, requestScope: { ...requestScope, userId: 2 } }),
      resolveChatSettingsKey({ ...params, requestScope: { ...requestScope, config: { ...requestScope.config, serverUrl: "https://two.example" } } }),
      resolveChatSettingsKey({ ...params, scope: { type: "workspace", workspaceId: "w" } })
    ]
    expect(new Set(keys).size).toBe(4)
    expect(keys.join()).not.toContain(config.apiKey)
  })

  it("refuses a native write without a captured owner before any I/O", async () => {
    await expect(applyChatSettingsPatch({ historyId: null, serverChatId: "same-id", patch: { authorNote: "New" } })).rejects.toThrow("chat_settings_owner_unavailable")
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.records.size).toBe(0)
  })

  it("does not promote bare native caches on an authoritative empty GET", async () => {
    state.records.set("chatSettings:server:same-id", canonical)
    state.remote = {}
    expect(await syncChatSettingsForServerChat(params)).toBeNull()
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.records.get("chatSettings:server:same-id")).toEqual(canonical)
  })

  it("carries the original captured request scope, workspace and cancellation to GET and PUT", async () => {
    const signal = new AbortController().signal
    const scope = { type: "workspace", workspaceId: "w" } as const
    await applyChatSettingsPatch({ ...params, signal, scope, patch: { authorNote: "New" } })
    expect(transport.getChatSettings).toHaveBeenCalledWith("same-id", { requestScope, signal, scope })
    expect(transport.updateChatSettings).toHaveBeenCalledWith("same-id", expect.objectContaining({ authorNote: "New" }), { requestScope, signal, scope })
  })

  it("blocks a held GET after the captured view retires, including an ID ABA", async () => {
    const entered = deferred(), held = deferred()
    transport.getChatSettings.mockImplementationOnce(async () => { entered.release(); await held.promise; return { conversation_id: "same-id", settings: canonical } })
    let current = true
    const pending = syncChatSettingsForServerChat({ ...params, validate_lease: () => current })
    await entered.promise
    current = false
    held.release()
    await expect(pending).rejects.toMatchObject({ status: 412, details: { detail: { code: "request_config_scope_changed" } } })
    expect(state.records.size).toBe(0)
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
  })

  it("retains a pending exact patch when a held PUT retires without publishing a cache receipt", async () => {
    const entered = deferred(), held = deferred()
    transport.updateChatSettings.mockImplementationOnce(async (_id: string, patch: Record<string, unknown>) => { entered.release(); await held.promise; return { conversation_id: "same-id", settings: { ...canonical, ...patch } } })
    let current = true
    const pending = applyChatSettingsPatch({ ...params, validate_lease: () => current, patch: { authorNote: "New" } })
    await entered.promise
    current = false
    held.release()
    await expect(pending).rejects.toMatchObject({ status: 412, details: { detail: { code: "request_config_scope_changed" } } })
    expect(state.records.get(getChatSettingsStorageKey(resolveChatSettingsKey(params)))).toBeUndefined()
    expect([...state.records.values()]).toContainEqual(expect.objectContaining({ value: expect.objectContaining({ status: "pending", patch: expect.objectContaining({ authorNote: "New" }) }) }))
  })

  it("reads back an unknown accepted PUT before Retry and never resubmits an already applied patch", async () => {
    transport.updateChatSettings.mockImplementationOnce(async (_id: string, patch: Record<string, unknown>) => { state.remote = { ...state.remote, ...patch }; throw new Error("response lost") })
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).rejects.toThrow("chat_settings_server_unconfirmed")
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).toMatchObject({ authorNote: "New" })
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
    expect(transport.getChatSettings).toHaveBeenCalledTimes(2)
  })

  it("retries only device persistence after a matching acknowledged PUT", async () => {
    transport.updateChatSettings.mockImplementationOnce(async (id: string, patch: Record<string, unknown>) => { state.remote = { ...state.remote, ...patch }; state.cacheFails = true; return { conversation_id: id, settings: state.remote } })
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).rejects.toMatchObject({ message: "chat_settings_device_save_failed", serverSettings: expect.objectContaining({ authorNote: "New" }) })
    state.cacheFails = false
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).toMatchObject({ authorNote: "New" })
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
  })

  it.each([413, 422])("allows corrected input after a definitive prewrite validation refusal (%s)", async status => {
    transport.updateChatSettings.mockRejectedValueOnce(Object.assign(new Error("validation refused"), {
      status, details: { detail: status === 413 ? "authorNote exceeds 20000 characters" : "Invalid settings" }
    }))
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Invalid proposal" } })).rejects.toMatchObject({ message: "chat_settings_server_rejected" })
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "Corrected" } })).toMatchObject({ authorNote: "Corrected" })
    expect(transport.updateChatSettings).toHaveBeenCalledTimes(2)
  })

  it("allows a corrected proposal after a rejected write to a known-empty destination", async () => {
    state.remote = {}
    transport.updateChatSettings.mockRejectedValueOnce(Object.assign(new Error("validation refused"), { status: 413, details: { detail: "authorNote exceeds 20000 characters" } }))
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Invalid" } })).rejects.toThrow("chat_settings_server_rejected")
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "Corrected" } })).toMatchObject({ authorNote: "Corrected" })
  })

  it("keeps a server-error outcome unresolved before a different proposal", async () => {
    transport.updateChatSettings.mockRejectedValueOnce(Object.assign(new Error("server error"), { status: 500 }))
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "First" } })).rejects.toThrow("chat_settings_server_unconfirmed")
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Changed" } })).rejects.toThrow("chat_settings_pending_changed")
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
  })

  it("retains a different unresolved intent instead of replacing it with a newer edit", async () => {
    transport.updateChatSettings.mockRejectedValueOnce(new Error("response lost"))
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "First" } })).rejects.toThrow("chat_settings_server_unconfirmed")
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Newer" } })).rejects.toMatchObject({ message: "chat_settings_pending_changed", pendingPatch: { authorNote: "First" } })
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
    expect([...state.records.values()]).toContainEqual(expect.objectContaining({ value: expect.objectContaining({ status: "pending", intent: { authorNote: "First" } }) }))
  })

  it("reads an owned unresolved receipt without transport or sharing it with another account", async () => {
    transport.updateChatSettings.mockRejectedValueOnce(new Error("Response lost"))
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Original A" } })).rejects.toThrow("chat_settings_server_unconfirmed")
    vi.clearAllMocks()
    const recovery = await getChatSettingsRecovery(params)
    expect(recovery).toMatchObject({ status: "pending", patch: { authorNote: "Original A" }, operationId: expect.any(String) })
    recovery!.patch.authorNote = "Mutated projection"
    expect(await getChatSettingsRecovery(params)).toMatchObject({ patch: { authorNote: "Original A" } })
    expect(await getChatSettingsRecovery({ ...params, requestScope: { ...requestScope, userId: 2 } })).toBeNull()
    expect(transport.getChatSettings).not.toHaveBeenCalled()
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
  })

  it("does not manufacture absent recovery from a malformed owned receipt", async () => {
    const destination = resolveChatSettingsKey(params)
    state.records.set(`tldw:chatSettingsSaves:v1:${destination}`, { value: { version: 1, destination, operation_id: "operation", status: "pending", intent: [], patch: {} }, updatedAt: 1 })
    await expect(getChatSettingsRecovery(params)).rejects.toThrow("chat_settings_pending_invalid")
  })

  it("returns no unresolved receipt after successful reconciliation", async () => {
    transport.updateChatSettings.mockImplementationOnce(async (_id: string, patch: Record<string, unknown>) => { state.remote = { ...state.remote, ...patch }; throw new Error("Response lost") })
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Original A" } })).rejects.toThrow()
    const original = await getChatSettingsRecovery(params)
    await applyChatSettingsPatch({ ...params, patch: original!.patch })
    expect(await getChatSettingsRecovery(params)).toBeNull()
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
  })

  it("refuses to read recovery after the captured owner retires", async () => {
    await expect(getChatSettingsRecovery({ ...params, validate_lease: () => false })).rejects.toMatchObject({ status: 412 })
  })

  it("does not turn a replaced recovery operation into a new settings write", async () => {
    transport.updateChatSettings.mockImplementationOnce(async (_id: string, patch: Record<string, unknown>) => { state.remote = { ...state.remote, ...patch }; throw new Error("Response lost") })
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "Original A" } })).rejects.toThrow()
    const original = (await getChatSettingsRecovery(params))!
    await applyChatSettingsPatch({ ...params, patch: original.patch, recoveryOperationId: original.operationId } as Parameters<typeof applyChatSettingsPatch>[0])
    await applyChatSettingsPatch({ ...params, patch: { authorNote: "Newer B" } })
    vi.clearAllMocks()
    await expect(applyChatSettingsPatch({ ...params, patch: original.patch, recoveryOperationId: original.operationId } as Parameters<typeof applyChatSettingsPatch>[0])).rejects.toThrow("chat_settings_recovery_changed")
    expect(transport.getChatSettings).not.toHaveBeenCalled()
    expect(transport.updateChatSettings).not.toHaveBeenCalled()
    expect(state.remote.authorNote).toBe("Newer B")
  })

  it("retains the owning operation identity on an uncertain write for explicit Retry", async () => {
    transport.updateChatSettings.mockRejectedValueOnce(new Error("Response lost"))
    const failure = await applyChatSettingsPatch({ ...params, patch: { authorNote: "Original A" } }).catch(error => error)
    expect(failure.settingsOperationId).toBe((await getChatSettingsRecovery(params))!.operationId)
  })

  it("does not dispatch a second same-intent write while the first PUT is held", async () => {
    const entered = deferred(), held = deferred()
    transport.updateChatSettings.mockImplementationOnce(async (id: string, patch: Record<string, unknown>) => { entered.release(); await held.promise; state.remote = { ...state.remote, ...patch }; return { conversation_id: id, settings: state.remote } })
    const first = applyChatSettingsPatch({ ...params, patch: { authorNote: "First" } })
    await entered.promise
    const second = applyChatSettingsPatch({ ...params, patch: { authorNote: "First" } })
    await Promise.resolve(); await Promise.resolve(); await Promise.resolve()
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
    held.release()
    await Promise.all([first, second])
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
  })

  it("allows canonical legacy schema settings to be read and edited again", async () => {
    state.remote = { ...canonical, schemaVersion: 1 }
    expect(await syncChatSettingsForServerChat(params)).toMatchObject({ schemaVersion: 1 })
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).toMatchObject({ authorNote: "New" })
  })

  it("submits intentional server patches without client clock LWW timestamps", async () => {
    state.remote = { ...canonical, updatedAt: "2099-01-01T00:00:00Z" }
    transport.updateChatSettings.mockImplementationOnce(async (id: string, patch: Record<string, unknown>) => {
      // The existing backend gives equal/future server timestamps precedence.
      if (typeof patch.updatedAt !== "string") state.remote = { ...state.remote, ...patch }
      return { conversation_id: id, settings: state.remote }
    })
    expect(await applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).toMatchObject({ authorNote: "New" })
    expect(transport.updateChatSettings.mock.calls[0][1]).not.toHaveProperty("updatedAt")
  })

  it("rejects a canonical receipt that does not accept the edited fields", async () => {
    transport.updateChatSettings.mockResolvedValueOnce({ conversation_id: "same-id", settings: canonical })
    await expect(applyChatSettingsPatch({ ...params, patch: { authorNote: "New" } })).rejects.toMatchObject({ message: "chat_settings_ack_mismatch", serverSettings: canonical })
  })

  it.each([false, true])("recognizes a backend-removed null attachment on explicit Retry, cache failure=%s", async (cacheFailure) => {
    transport.updateChatSettings.mockImplementationOnce(async (id: string) => {
      delete state.remote.deepResearchAttachment
      if (!cacheFailure) throw new Error("response lost")
      state.cacheFails = true
      return { conversation_id: id, settings: state.remote }
    })
    await expect(applyChatSettingsPatch({ ...params, patch: { deepResearchAttachment: null } })).rejects.toThrow()
    state.cacheFails = false
    expect(await applyChatSettingsPatch({ ...params, patch: { deepResearchAttachment: null } })).not.toBeNull()
    expect(transport.updateChatSettings).toHaveBeenCalledOnce()
  })

  it("merges a nested edit against live remote context while preserving unrelated fields", async () => {
    state.records.set(getChatSettingsStorageKey(resolveChatSettingsKey(params)), { value: { ...canonical, conversationContext: { world_book_ids: [1], chat_dictionary_ids: [2] } }, updatedAt: Date.now() })
    const next = await applyChatSettingsPatch({ ...params, patch: { conversationContext: { chat_dictionary_ids: [9] } } })
    expect(next?.conversationContext).toEqual({ world_book_ids: [8], chat_dictionary_ids: [9] })
    expect(state.remote.authorNote).toBe("Server")
    expect(state.remote.chat_dictionary_ids).toEqual([9])
  })
})
