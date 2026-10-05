import { beforeEach, describe, expect, it, vi } from "vitest"
import { Storage } from "@plasmohq/storage"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { chatDraftAddressKey, type ChatDraftHandle } from "@/utils/chat-draft-address"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { copyChatCreationRecovery, createChatWithRecovery, listChatCreationRecovery, readChatCreationRecovery, markChatCreationBound, markChatCreationSettled } from "../pending-chat-promotion"

const snapshot = () => ({ requestScope: { config: { serverUrl: "https://server.test", authMode: "multi-user", apiKey: "NEVER-PERSIST-SECRET" }, userId: "alice" },
  scopeSignal: new AbortController().signal, scopeInvalidatedSignal: new AbortController().signal, release: vi.fn() }) as unknown as ServicePromptSnapshot
const handle = (id = "fresh-A"): ChatDraftHandle => ({ ownerHash: chatDraftOwnerKey(snapshot().requestScope), address: { kind: "fresh", id, temporary: false, scope: { type: "global" } } })
const args = (source = handle()) => ({ snapshot: snapshot(), handle: source, surface: "playground" as const,
  inputText: "ORIGINAL INPUT", setup: { characterId: "12", provider: "openai", model: "configured-model", temperature: 0, topP: 0 }, isCurrent: () => true })
beforeEach(() => {
  vi.restoreAllMocks(); localStorage.clear(); sessionStorage.clear()
  const tails = new Map<string, Promise<unknown>>()
  Object.defineProperty(navigator, "locks", { configurable: true, value: { request: (key: string, work: () => unknown) => {
    const next = (tails.get(key) ?? Promise.resolve()).then(work); tails.set(key, next.catch(() => undefined)); return next
  } } })
})
describe("actual private first-create record contract", () => {
  it("retains whitelisted original input before POST and admits only one concurrent same-source create", async () => {
    const create = vi.fn(async () => {
      const record = await readChatCreationRecovery(handle())
      expect(record).toMatchObject({ state: "unconfirmed", inputText: "ORIGINAL INPUT", temperature: 0, topP: 0 })
      expect(JSON.stringify(record)).not.toContain("NEVER-PERSIST-SECRET")
      expect(JSON.stringify(record)).not.toContain("requestScope")
      return { id: "created-once" }
    })
    const results = await Promise.allSettled([createChatWithRecovery({ ...args(), create }), createChatWithRecovery({ ...args(), create })])
    expect(results.filter(result => result.status === "fulfilled")).toHaveLength(1)
    expect(create).toHaveBeenCalledOnce()
    expect(await readChatCreationRecovery(handle())).toMatchObject({ state: "acknowledged", chatId: "created-once" })
  })
  it("keeps a missing acknowledgement blocked without replay", async () => {
    const create = vi.fn(async () => ({} as { id: string }))
    await expect(createChatWithRecovery({ ...args(), create })).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
    await expect(createChatWithRecovery({ ...args(), create })).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
    expect(create).toHaveBeenCalledOnce()
    expect(await readChatCreationRecovery(handle())).toMatchObject({ state: "unconfirmed" })
  })
  it("a copied fresh UUID follows the original late ACK and remains blocked after edit and cold read", async () => {
    let enter!: () => void, release!: () => void
    const started = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    const original = createChatWithRecovery({ ...args(), create: async () => { enter(); await held; return { id: "late-created" } } })
    await started
    const copied = handle("new-copied-UUID")
    expect(await copyChatCreationRecovery(handle(), copied)).toBe(true)
    expect(await readChatCreationRecovery(copied)).toMatchObject({ state: "unconfirmed", inputText: "ORIGINAL INPUT" })
    release(); await original
    expect(await readChatCreationRecovery(structuredClone(copied))).toMatchObject({ state: "acknowledged", chatId: "late-created" })
    const create = vi.fn()
    await expect(createChatWithRecovery({ ...args(copied), inputText: "edited after copy", create })).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
    expect(create).not.toHaveBeenCalled()
  })
  it.each(["owner", "workspace", "temporary"])("refuses %s mismatch before POST or copied provenance", async boundary => {
    const source = handle()
    if (boundary === "owner") source.ownerHash = "b".repeat(64)
    else if (source.address.kind === "fresh") {
      if (boundary === "workspace") source.address.scope = { type: "workspace", workspaceId: "foreign" }
      else source.address.temporary = true
    }
    const create = vi.fn()
    await expect(createChatWithRecovery({ ...args(source), create })).rejects.toThrow()
    expect(create).not.toHaveBeenCalled()
  })
  it("refuses a malformed typed record without replacing it", async () => {
    const source = handle()
    const storage = new Storage({ area: "local" })
    const key = "registry:chat-creation:" + source.ownerHash + ":" + chatDraftAddressKey(source.address)
    await storage.set(key, { value: { version: 99 }, updatedAt: 1 })
    const before = localStorage.getItem(key)
    const create = vi.fn()
    await expect(createChatWithRecovery({ ...args(), create })).rejects.toMatchObject({ code: "chat_creation_recovery_unavailable" })
    expect(create).not.toHaveBeenCalled()
    expect(localStorage.getItem(key)).toBe(before)
  })
  it("cancellation during reservation makes zero POST and allows a safe later attempt", async () => {
    const original = Storage.prototype.set
    let enter!: () => void, release!: () => void, current = true
    const started = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(Storage.prototype, "set").mockImplementation(async function (this: Storage, key, value) { await original.call(this, key, value); enter(); await held; return null })
    const create = vi.fn()
    const operation = createChatWithRecovery({ ...args(), isCurrent: () => current, create }).catch(error => error)
    await started; current = false; release()
    expect((await operation).message).toBe("stale_selection")
    expect(create).not.toHaveBeenCalled()
    expect(await readChatCreationRecovery(handle())).toMatchObject({ state: "prepared", inputText: "ORIGINAL INPUT" })
    current = true
    create.mockResolvedValue({ id: "safe-new-attempt" })
    await expect(createChatWithRecovery({ ...args(), create })).resolves.toMatchObject({ id: "safe-new-attempt" })
    expect(create).toHaveBeenCalledOnce()
  })
  it("keeps an empty attempted input reachable without applying the composer TTL", async () => {
    await createChatWithRecovery({ ...args(), inputText: "", create: async () => { throw new Error("lost") } }).catch(() => undefined)
    const now = Date.now()
    vi.spyOn(Date, "now").mockReturnValue(now + 366 * 86400000)
    expect((await listChatCreationRecovery(handle().ownerHash, { type: "global" })).attempts).toMatchObject([{ inputText: "", state: "unconfirmed" }])
  })
})


it("retains a bound chat's unsent original input until verified settlement", async () => {
  await createChatWithRecovery({ ...args(), create: async () => ({ id: "created-chat" }) })
  expect(await markChatCreationBound(handle(), "created-chat")).toBe(true)
  expect((await listChatCreationRecovery(handle().ownerHash, { type: "global" })).attempts).toMatchObject([{ state: "bound", inputText: "ORIGINAL INPUT" }])
  expect(await markChatCreationSettled(handle(), "other-chat")).toBe(false)
  expect((await readChatCreationRecovery(handle()))?.inputText).toBe("ORIGINAL INPUT")
  expect(await markChatCreationSettled(handle(), "created-chat")).toBe(true)
  expect((await listChatCreationRecovery(handle().ownerHash, { type: "global" })).attempts).toEqual([])
  expect(await readChatCreationRecovery(handle())).toMatchObject({ state: "settled", inputText: "", chatId: "created-chat" })
})
