import "@/hooks/chat/__tests__/native-plasmo-storage-fixture"
import { browser } from "../../../../../tldw-frontend/extension/shims/wxt-browser"
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
    const create = vi.fn(async (id: string) => {
      const record = await readChatCreationRecovery(handle())
      expect(record).toMatchObject({ state: "unconfirmed", inputText: "ORIGINAL INPUT", temperature: 0, topP: 0 })
      expect(JSON.stringify(record)).not.toContain("NEVER-PERSIST-SECRET")
      expect(JSON.stringify(record)).not.toContain("requestScope")
      return { id }
    })
    const results = await Promise.allSettled([createChatWithRecovery({ ...args(), create }), createChatWithRecovery({ ...args(), create })])
    expect(results.filter(result => result.status === "fulfilled")).toHaveLength(1)
    expect(create).toHaveBeenCalledOnce()
    expect(await readChatCreationRecovery(handle())).toMatchObject({ state: "acknowledged", chatId: create.mock.calls[0][0] })
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
    const original = createChatWithRecovery({ ...args(), create: async id => { enter(); await held; return { id } } })
    await started
    const copied = handle("new-copied-UUID")
    expect(await copyChatCreationRecovery(handle(), copied)).toBe(true)
    expect(await readChatCreationRecovery(copied)).toMatchObject({ state: "unconfirmed", inputText: "ORIGINAL INPUT" })
    release(); const created = await original
    expect(await readChatCreationRecovery(structuredClone(copied))).toMatchObject({ state: "acknowledged", chatId: created.id })
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
    const original = browser.storage.local.set
    let enter!: () => void, release!: () => void, current = true
    const started = new Promise<void>(resolve => { enter = resolve })
    const held = new Promise<void>(resolve => { release = resolve })
    vi.spyOn(browser.storage.local, "set").mockImplementation(async (items, callback) => {
      await original(items, callback)
      if (Object.keys(items).some(key => key.startsWith("registry:chat-creation:"))) { enter(); await held }
    })
    const create = vi.fn()
    const operation = createChatWithRecovery({ ...args(), isCurrent: () => current, create }).catch(error => error)
    await started; current = false; release()
    expect((await operation).message).toBe("stale_selection")
    expect(create).not.toHaveBeenCalled()
    expect(await readChatCreationRecovery(handle())).toMatchObject({ state: "prepared", inputText: "ORIGINAL INPUT" })
    current = true
    create.mockImplementation(async (id: string) => ({ id }))
    await expect(createChatWithRecovery({ ...args(), create })).resolves.toMatchObject({ id: (await readChatCreationRecovery(handle()))?.operationId })
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
  const created = await createChatWithRecovery({ ...args(), create: async id => ({ id }) })
  expect(await markChatCreationBound(handle(), created.id)).toBe(true)
  expect((await listChatCreationRecovery(handle().ownerHash, { type: "global" })).attempts).toMatchObject([{ state: "bound", inputText: "ORIGINAL INPUT" }])
  expect(await markChatCreationSettled(handle(), "other-chat")).toBe(false)
  expect((await readChatCreationRecovery(handle()))?.inputText).toBe("ORIGINAL INPUT")
  expect(await markChatCreationSettled(handle(), created.id)).toBe(true)
  expect((await listChatCreationRecovery(handle().ownerHash, { type: "global" })).attempts).toEqual([])
  expect(await readChatCreationRecovery(handle())).toMatchObject({ state: "settled", inputText: "", chatId: created.id })
})


describe("client UUID creation recovery", () => {
  it("persists the exact client UUID before the single POST and requires its acknowledgement", async () => {
    const create = vi.fn(async (id: string) => {
      const record = await readChatCreationRecovery(handle())
      expect(id).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/
      )
      expect(record).toMatchObject({
        state: "unconfirmed",
        operationId: id,
        clientChatId: id,
        inputText: "ORIGINAL INPUT"
      })
      return { id }
    })
    const results = await Promise.allSettled([
      createChatWithRecovery({ ...args(), create }),
      createChatWithRecovery({ ...args(), create })
    ])
    expect(
      results.filter((result) => result.status === "fulfilled")
    ).toHaveLength(1)
    expect(create).toHaveBeenCalledOnce()
    expect(await readChatCreationRecovery(handle())).toMatchObject({
      state: "acknowledged",
      chatId: create.mock.calls[0][0]
    })
  })

  it("retains uncertainty when the ACK names a different chat", async () => {
    const create = vi.fn(async () => ({ id: "different-chat" }))
    await expect(
      createChatWithRecovery({ ...args(), create })
    ).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
    expect(await readChatCreationRecovery(handle())).toMatchObject({
      state: "unconfirmed",
      inputText: "ORIGINAL INPUT"
    })
    await expect(
      createChatWithRecovery({ ...args(), create })
    ).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
    expect(create).toHaveBeenCalledOnce()
  })

  it.each(["legacy", "prepared"])(
    "preserves %s records without inferring a dispatched client ID",
    async (state) => {
      const source = handle()
      const operationId = "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14"
      const storage = new Storage({ area: "local" })
      await storage.set(
        "registry:chat-creation:" +
          source.ownerHash +
          ":" +
          chatDraftAddressKey(source.address),
        {
          value: {
            version: 1,
            handle: source,
            operationId,
            surface: "playground",
            createdAt: 1,
            state: state === "legacy" ? "unconfirmed" : "prepared",
            inputText: "OLD ORIGINAL"
          },
          updatedAt: 1
        }
      )
      const create = vi.fn(async (id: string) => ({ id }))
      if (state === "legacy") {
        await expect(
          createChatWithRecovery({ ...args(), create })
        ).rejects.toMatchObject({ code: "chat_creation_unconfirmed" })
        expect(create).not.toHaveBeenCalled()
        expect(await readChatCreationRecovery(source)).toMatchObject({
          operationId,
          inputText: "OLD ORIGINAL"
        })
        expect(
          (await readChatCreationRecovery(source))?.clientChatId
        ).toBeUndefined()
      } else {
        await createChatWithRecovery({ ...args(), create })
        expect(create).toHaveBeenCalledWith(operationId)
        expect(await readChatCreationRecovery(source)).toMatchObject({
          operationId,
          clientChatId: operationId,
          state: "acknowledged"
        })
      }
    }
  )

  it.each([
    "c".repeat(36),
    "00000000-0000-0000-0000-000000000000",
    "ffffffff-ffff-ffff-ffff-ffffffffffff",
    "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a15"
  ])(
    "refuses an invalid or mismatched persisted client UUID %s",
    async (clientChatId) => {
      const source = handle()
      const storage = new Storage({ area: "local" })
      const key =
        "registry:chat-creation:" +
        source.ownerHash +
        ":" +
        chatDraftAddressKey(source.address)
      await storage.set(key, {
        value: {
          version: 1,
          operationId: "6b0f8c1e-2d4a-4f3b-9a71-5c2e8d9f0a14",
          clientChatId,
          handle: source,
          surface: "playground",
          createdAt: 1,
          inputText: "KEPT",
          state: "unconfirmed"
        },
        updatedAt: 1
      })
      const before = localStorage.getItem(key)
      await expect(readChatCreationRecovery(source)).rejects.toThrow()
      expect(localStorage.getItem(key)).toBe(before)
    }
  )
})

it.each(["acknowledged", "bound", "settled"] as const)(
  "client UUID late ACK preserves the canonical alias's %s fact",
  async (state) => {
    const { acknowledgeChatCreation } =
      await import("../pending-chat-promotion")
    let entered!: () => void, release!: () => void
    const ready = new Promise<void>((resolve) => {
      entered = resolve
    })
    const held = new Promise<void>((resolve) => {
      release = resolve
    })
    const create = vi.fn(async (id: string) => {
      entered()
      await held
      return { id }
    })
    const original = createChatWithRecovery({ ...args(), create })
    await ready
    const alias = handle("copied-cold")
    expect(await copyChatCreationRecovery(handle(), alias)).toBe(true)
    const attempt = (await readChatCreationRecovery(structuredClone(alias)))!
    expect(await acknowledgeChatCreation(attempt, attempt.clientChatId!)).toBe(
      true
    )
    if (state !== "acknowledged")
      expect(await markChatCreationBound(alias, attempt.clientChatId!)).toBe(
        true
      )
    if (state === "settled")
      expect(await markChatCreationSettled(alias, attempt.clientChatId!)).toBe(
        true
      )
    release()
    await expect(original).resolves.toEqual({ id: attempt.clientChatId })
    expect(await readChatCreationRecovery(handle())).toMatchObject({
      state,
      chatId: attempt.clientChatId,
      inputText: state === "settled" ? "" : "ORIGINAL INPUT"
    })
    expect(await readChatCreationRecovery(alias)).toEqual(
      await readChatCreationRecovery(handle())
    )
    expect(create).toHaveBeenCalledOnce()
  }
)


it("keeps bound original input when a different operation tries to settle it", async () => {
  const created = await createChatWithRecovery({ ...args(), create: async id => ({ id }) })
  expect(await markChatCreationBound(handle(), created.id)).toBe(true)
  expect(await markChatCreationSettled(handle(), created.id, "foreign-operation")).toBe(false)
  const attempt = await readChatCreationRecovery(handle())
  expect(attempt).toMatchObject({ state: "bound", inputText: "ORIGINAL INPUT" })
  expect(await markChatCreationSettled(handle(), created.id, attempt!.operationId)).toBe(true)
})
