import { createSafeStorage } from "@/utils/safe-storage"
import { createLocalRegistryBucket } from "./settings/local-bucket"
import { chatDraftOwnerKey } from "@/hooks/useChatDraftOwner"
import { chatDraftAddressKey, isChatDraftHandle, type ChatDraftHandle } from "@/utils/chat-draft-address"
import type { ChatScope } from "@/types/chat-scope"
import type { ServicePromptSnapshot } from "@/services/service-prompts"

export type ChatPromotionPublication = Readonly<{
  historyId: string | null
  chatId: string
  scopeKey: string
  requestScope: ServicePromptSnapshot["requestScope"]
}>

/** A command retains this exact operation before any preparation await. */
export type CapturedChatPromotion = Readonly<{
  isCurrent: () => boolean
  getPublication: () => ChatPromotionPublication | undefined
}>

type PendingPromotion = {
  owner: object
  attempt: object
  historyId: string | null
  cancelled: boolean
  publication?: ChatPromotionPublication
  isSourceCurrent?: () => boolean
  scope: Promise<ServicePromptSnapshot>
  snapshot?: ServicePromptSnapshot
  completion: Promise<void>
  saved: Promise<void>
  finishWaiting: () => void
  failed: boolean
  error?: unknown
  promote: (
    snapshot: ServicePromptSnapshot,
    acknowledgePublication: (chatId: string) => void
  ) => Promise<void>
  abort: () => void
  onFailure: (error: unknown, retrying: boolean) => void
  onComplete: () => void
}

const pendingPromotions = new WeakMap<object, PendingPromotion>()
const promotionAttempts = new WeakMap<object, object>()
const capturedPromotions = new WeakMap<CapturedChatPromotion, PendingPromotion>()
const promotionIsCurrent = (pending: PendingPromotion) =>
  !pending.cancelled && promotionAttempts.get(pending.owner) === pending.attempt &&
  !pending.snapshot?.scopeSignal.aborted &&
  !pending.snapshot?.scopeInvalidatedSignal.aborted &&
  (pending.isSourceCurrent?.() ?? true)
const promotionChanged = () => new DOMException("The saving chat changed", "AbortError")

export const captureChatPromotion = (
  owner: object,
  historyId: string | null
): CapturedChatPromotion | null => {
  const pending = pendingPromotions.get(owner)
  if (!pending || pending.historyId !== historyId || !promotionIsCurrent(pending)) return null
  const capture = Object.freeze({
    isCurrent: () => promotionIsCurrent(pending),
    getPublication: () => promotionIsCurrent(pending) ? pending.publication : undefined
  })
  capturedPromotions.set(capture, pending)
  return capture
}
const INCOMPLETE = "chat_promotion_incomplete"
const incompleteError = () =>
  Object.assign(
    new Error(
      "Earlier messages are not fully saved. Retry saving chat before continuing."
    ),
    { code: INCOMPLETE }
  )
export const isChatPromotionIncompleteError = (error: unknown): boolean =>
  Boolean(
    error &&
    typeof error === "object" &&
    "code" in error &&
    error.code === INCOMPLETE
  )

export const clearChatPromotion = (owner: object, historyId: string | null) => {
  const pending = pendingPromotions.get(owner)
  if (!pending || pending.historyId !== historyId) return
  pendingPromotions.delete(owner)
  pending.cancelled = true
  if (promotionAttempts.get(owner) === pending.attempt) promotionAttempts.delete(owner)
  pending.abort()
  pending.finishWaiting()
  pending.snapshot?.release()
  pending.onComplete()
}

const runPromotion = (
  owner: object,
  pending: PendingPromotion
): Promise<void> => {
  pending.failed = false
  pending.completion = pending.scope
    .then(async (snapshot) => {
      pending.snapshot = snapshot
      snapshot.scopeSignal.throwIfAborted()
      if (!promotionIsCurrent(pending)) throw promotionChanged()
      await pending.promote(snapshot, (chatId) => {
        if (!chatId || !promotionIsCurrent(pending) ||
          (pending.publication && pending.publication.chatId !== chatId))
          throw promotionChanged()
        pending.publication = Object.freeze({
          historyId: pending.historyId, chatId, scopeKey: snapshot.scopeKey,
          requestScope: snapshot.requestScope
        })
      })
      snapshot.scopeSignal.throwIfAborted()
      if (!promotionIsCurrent(pending)) throw promotionChanged()
      if (pendingPromotions.get(owner) === pending)
        pendingPromotions.delete(owner)
      snapshot.release()
      pending.finishWaiting()
      pending.onComplete()
    })
    .catch((error) => {
      if (!pending.snapshot || pending.snapshot.scopeSignal.aborted || !promotionIsCurrent(pending)) {
        pending.cancelled = true
        if (promotionAttempts.get(owner) === pending.attempt) promotionAttempts.delete(owner)
        if (pendingPromotions.get(owner) === pending) {
          pendingPromotions.delete(owner)
          pending.onComplete()
        }
        pending.abort()
        pending.snapshot?.release()
        pending.finishWaiting()
        throw error
      }
      pending.failed = true
      pending.error = error
      pending.onFailure(error, false)
      throw incompleteError()
    })
  return pending.completion
}

export const trackChatPromotion = (
  owner: object,
  historyId: string | null,
  scope: Promise<ServicePromptSnapshot>,
  promote: PendingPromotion["promote"],
  options: Pick<PendingPromotion, "abort" | "onFailure" | "onComplete" | "isSourceCurrent">
): Promise<void> => {
  const previous = pendingPromotions.get(owner)
  if (previous) clearChatPromotion(owner, previous.historyId)
  let finishWaiting!: () => void
  const saved = new Promise<void>(resolve => { finishWaiting = resolve })
  const pending: PendingPromotion = {
    saved,
    finishWaiting,
    owner,
    attempt: {},
    historyId,
    cancelled: false,
    scope,
    promote,
    ...options,
    failed: false,
    completion: Promise.resolve()
  }
  pendingPromotions.set(owner, pending)
  promotionAttempts.set(owner, pending.attempt)
  void scope.then(
    (snapshot) => {
      snapshot.scopeSignal.addEventListener(
        "abort",
        () => {
          if (pendingPromotions.get(owner) === pending)
            clearChatPromotion(owner, historyId)
        },
        { once: true }
      )
    },
    () => undefined
  )
  return runPromotion(owner, pending)
}

export const retryChatPromotion = (
  owner: object,
  historyId: string | null
): Promise<void> | null => {
  const pending = pendingPromotions.get(owner)
  if (!pending || pending.historyId !== historyId) return null
  if (!pending.failed) return pending.completion
  pending.onFailure(pending.error, true)
  return runPromotion(owner, pending)
}

const waitWithoutCancellingOwner = <T>(
  work: Promise<T>,
  signal: AbortSignal
): Promise<T> => {
  signal.throwIfAborted()
  return new Promise<T>((resolve, reject) => {
    const abort = () => reject(signal.reason)
    signal.addEventListener("abort", abort, { once: true })
    work
      .then(resolve, reject)
      .finally(() => signal.removeEventListener("abort", abort))
  })
}

export const waitForChatPromotion = async (
  owner: object,
  historyId: string | null,
  snapshot: ServicePromptSnapshot,
  options?: { waitUntilSaved?: boolean; captured?: CapturedChatPromotion | null }
): Promise<ChatPromotionPublication | void> => {
  snapshot.scopeSignal.throwIfAborted()
  const captured = options?.captured
  const pending = captured ? capturedPromotions.get(captured) :
    options && "captured" in options ? undefined : pendingPromotions.get(owner)
  if (!pending || pending.historyId !== historyId || pending.owner !== owner) {
    if (captured) throw promotionChanged()
    return
  }
  if (captured && !promotionIsCurrent(pending)) throw promotionChanged()
  let ownerScope: ServicePromptSnapshot
  try {
    ownerScope = await waitWithoutCancellingOwner(
      pending.scope,
      snapshot.scopeSignal
    )
  } catch (error) {
    snapshot.scopeSignal.throwIfAborted()
    if (!captured && error instanceof Error && error.name === "AbortError") return
    throw error
  }
  if (captured && !promotionIsCurrent(pending)) throw promotionChanged()
  if (
    ownerScope.scopeInvalidatedSignal.aborted ||
    ownerScope.scopeKey !== snapshot.scopeKey
  ) {
    if (captured) throw promotionChanged()
    return
  }
  if (pending.failed && !options?.waitUntilSaved) pending.onFailure(pending.error, false)
  await waitWithoutCancellingOwner(options?.waitUntilSaved ? pending.saved : pending.completion, snapshot.scopeSignal)
  if (options?.waitUntilSaved) ownerScope.scopeSignal.throwIfAborted()
  snapshot.scopeSignal.throwIfAborted()
  if (captured) {
    if (!promotionIsCurrent(pending)) throw promotionChanged()
    return pending.publication
  }
}


/** Private recovery facts locate an attempt; they never grant conversation authority. */
export type ChatCreationRecovery = {
  version: 1
  operationId: string
  clientChatId?: string
  handle: ChatDraftHandle
  surface: "playground" | "sidepanel"
  createdAt: number
  state: "prepared" | "unconfirmed" | "acknowledged" | "bound" | "settled"
  inputText: string
  characterId?: string | number
  provider?: string
  model?: string
  temperature?: number
  topP?: number
  chatId?: string
}
// Unresolved creation must not inherit the composer's expiration or per-tab pin.
type CreationReference = { version: 1; handle: ChatDraftHandle; reference: ChatDraftHandle; operationId: string }
const creationRecovery = createLocalRegistryBucket<ChatCreationRecovery | CreationReference>({ prefix: "registry:chat-creation:" })
const creationKey = (handle: ChatDraftHandle) => `${handle.ownerHash}:${chatDraftAddressKey(handle.address)}`
const validCreationHandle = (handle: unknown): handle is ChatDraftHandle => isChatDraftHandle(handle) &&
  handle.address.kind === "fresh" && !handle.address.temporary
const validClientChatId = (value: unknown): value is string =>
  typeof value === "string" &&
  /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/.test(
    value
  ) &&
  value !== "00000000-0000-0000-0000-000000000000" &&
  value !== "ffffffff-ffff-ffff-ffff-ffffffffffff"
const validCreationRecord = (value: unknown): value is ChatCreationRecovery => {
  if (!value || typeof value !== "object") return false
  const record = value as ChatCreationRecovery
  return (
    record.version === 1 &&
    validCreationHandle(record.handle) &&
    typeof record.operationId === "string" &&
    /^[a-f0-9-]{36}$/.test(record.operationId) &&
    (record.clientChatId === undefined ||
      (validClientChatId(record.clientChatId) &&
        record.clientChatId === record.operationId &&
        (record.chatId === undefined ||
          record.chatId === record.clientChatId))) &&
    (record.surface === "playground" || record.surface === "sidepanel") &&
    typeof record.createdAt === "number" &&
    Number.isFinite(new Date(record.createdAt).getTime()) &&
    typeof record.inputText === "string" &&
    ["prepared", "unconfirmed", "acknowledged", "bound", "settled"].includes(
      record.state
    ) &&
    (["prepared", "unconfirmed"].includes(record.state) ||
      (typeof record.chatId === "string" && record.chatId.trim().length > 0))
  )
}

const sameCreationScope = (a: ChatDraftHandle, b: ChatDraftHandle) => a.ownerHash === b.ownerHash &&
  a.address.kind === "fresh" && b.address.kind === "fresh" && a.address.scope.type === b.address.scope.type &&
  (a.address.scope.type !== "workspace" || (b.address.scope.type === "workspace" && a.address.scope.workspaceId === b.address.scope.workspaceId))
const validCreationReference = (value: unknown): value is CreationReference => {
  if (!value || typeof value !== "object") return false
  const ref = value as CreationReference
  return ref.version === 1 && validCreationHandle(ref.handle) && validCreationHandle(ref.reference) &&
    sameCreationScope(ref.handle, ref.reference) && creationKey(ref.handle) !== creationKey(ref.reference) && typeof ref.operationId === "string"
}
export const readChatCreationRecovery = async (handle: ChatDraftHandle): Promise<ChatCreationRecovery | null> => {
  if (!validCreationHandle(handle)) throw new Error("chat_creation_source_invalid")
  const record = await creationRecovery.get(creationKey(handle), { failClosed: true })
  if (!record) return null
  const value = record.value
  if (validCreationReference(value) && creationKey(value.handle) === creationKey(handle)) {
    const canonical = await creationRecovery.get(creationKey(value.reference), { failClosed: true })
    if (canonical && validCreationRecord(canonical.value) && creationKey(canonical.value.handle) === creationKey(value.reference) && canonical.value.operationId === value.operationId) return canonical.value
    throw new Error("chat_creation_reference_invalid")
  }
  if (!validCreationRecord(value) || creationKey(value.handle) !== creationKey(handle)) throw new Error("chat_creation_record_invalid")
  return value
}
/** A copied draft keeps the original uncertainty even after autosave or a cold reload. */
export const copyChatCreationRecovery = async (source: ChatDraftHandle, destination: ChatDraftHandle): Promise<boolean> => {
  if (source.address.kind === "fresh" && source.address.temporary) return true
  try {
    if (!validCreationHandle(source) || !validCreationHandle(destination) || !sameCreationScope(source, destination)) return false
    const attempt = await readChatCreationRecovery(source)
    if (!attempt || creationKey(source) === creationKey(destination)) return true
    const existing = await readChatCreationRecovery(destination)
    if (existing) return existing.operationId === attempt.operationId
    const reference: CreationReference = { version: 1, handle: structuredClone(destination), reference: attempt.handle, operationId: attempt.operationId }
    const copied = await creationRecovery.set(creationKey(destination), reference, Date.now(), null)
    if (copied) notifyCreationRecovery()
    return copied
  } catch { return false }
}
const notifyCreationRecovery = () => {
  if (typeof window !== "undefined") window.dispatchEvent(new Event("tldw:chat-creation-recovery"))
}
export const watchChatCreationRecovery = (handles: ChatDraftHandle[], listener: () => void) => {
  const storage = createSafeStorage({ area: "local" })
  const watches = Object.fromEntries(handles.filter(validCreationHandle).map(handle => [creationRecovery.buildKey(creationKey(handle)), listener]))
  storage.watch(watches)
  window.addEventListener("tldw:chat-creation-recovery", listener)
  window.addEventListener("focus", listener)
  window.addEventListener("pageshow", listener)
  return () => {
    storage.unwatch(watches)
    window.removeEventListener("tldw:chat-creation-recovery", listener)
    window.removeEventListener("focus", listener)
    window.removeEventListener("pageshow", listener)
  }
}
export const listChatCreationRecovery = async (ownerHash: string, scope: ChatScope) => {
  const result = await creationRecovery.list()
  let unavailable = result.unavailable
  const attempts: ChatCreationRecovery[] = []
  for (const [key, record] of Object.entries(result.records)) {
    if (!key.startsWith(`${ownerHash}:`)) continue
    if (validCreationReference(record.value) && creationKey(record.value.handle) === key) continue
    if (!validCreationRecord(record.value) || creationKey(record.value.handle) !== key) { unavailable = true; continue }
    const address = record.value.handle.address
    if (address.kind !== "fresh" || address.scope.type !== scope.type ||
      (address.scope.type === "workspace" && scope.type === "workspace" && address.scope.workspaceId !== scope.workspaceId) ||
      record.value.state === "settled") continue
    attempts.push(record.value)
  }
  return { attempts: attempts.sort((a, b) => b.createdAt - a.createdAt), unavailable }
}
const uncertainCreationError = () => Object.assign(new Error(
  "Server save unconfirmed. The server may have created this chat. Review conversations or retained drafts before starting another chat, which may create a duplicate."
), { code: "chat_creation_unconfirmed" })
const unavailableCreationError = () => Object.assign(new Error(
  "Chat creation paused: recovery storage could not be verified. Your draft is kept. Check retained drafts or reconnect before starting another chat."
), { code: "chat_creation_recovery_unavailable" })

/** Record an owned creation fact without downgrading a matching later state. */
export const acknowledgeChatCreation = async (
  attempt: Pick<
    ChatCreationRecovery,
    "handle" | "operationId" | "clientChatId"
  >,
  chatId: string
): Promise<boolean> => {
  const matches = (value: unknown): value is ChatCreationRecovery =>
    validCreationRecord(value) &&
    creationKey(value.handle) === creationKey(attempt.handle) &&
    value.operationId === attempt.operationId &&
    value.clientChatId === attempt.clientChatId &&
    (value.clientChatId === undefined || value.clientChatId === chatId)
  const acknowledged = (value: unknown) =>
    matches(value) &&
    ["acknowledged", "bound", "settled"].includes(value.state) &&
    value.chatId === chatId
  try {
    const current = await readChatCreationRecovery(attempt.handle)
    if (!matches(current)) return false
    const key = creationKey(current.handle)
    const record = await creationRecovery.get(key, { failClosed: true })
    if (!record || !matches(record.value)) return false
    if (acknowledged(record.value)) return true
    if (record.value.state !== "unconfirmed" || !record.value.clientChatId)
      return false
    if (
      await creationRecovery.set(
        key,
        { ...record.value, state: "acknowledged", chatId },
        record.updatedAt,
        record
      )
    ) {
      notifyCreationRecovery()
      return true
    }
    // An explicit lookup or the original late response may have won the same CAS.
    return acknowledged(
      (await creationRecovery.get(key, { failClosed: true }))?.value
    )
  } catch {
    return false
  }
}

/** Persist the client UUID before a single create; never replay uncertain creation. */
export const createChatWithRecovery = async <T extends { id: string }>(params: {
  snapshot: ServicePromptSnapshot
  handle: ChatDraftHandle
  scope?: ChatScope
  surface: ChatCreationRecovery["surface"]
  inputText: string
  setup: Pick<ChatCreationRecovery, "characterId" | "provider" | "model" | "temperature" | "topP">
  isCurrent: () => boolean
  create: (clientChatId: string) => Promise<T>
}): Promise<T> => {
  const handle = structuredClone(params.handle)
  if (!validCreationHandle(handle) || handle.ownerHash !== chatDraftOwnerKey(params.snapshot.requestScope))
    throw unavailableCreationError()
  const scope = params.scope ?? { type: "global" }
  const address = handle.address
  if (address.kind !== "fresh" || address.scope.type !== scope.type ||
    (address.scope.type === "workspace" && scope.type === "workspace" && address.scope.workspaceId !== scope.workspaceId))
    throw unavailableCreationError()
  const current = () => params.isCurrent() && !params.snapshot.scopeSignal.aborted && !params.snapshot.scopeInvalidatedSignal.aborted
  if (!current()) throw new Error("stale_selection")
  let existing: ChatCreationRecovery | null
  try { existing = await readChatCreationRecovery(handle) } catch { throw unavailableCreationError() }
  if (existing && existing.state !== "prepared") throw uncertainCreationError()
  const canonical = existing?.handle ?? handle
  const key = creationKey(canonical)
  const expected = existing ? await creationRecovery.get(key, { failClosed: true }) : null
  if (existing && (!expected || !validCreationRecord(expected.value) || expected.value.operationId !== existing.operationId || expected.value.state !== "prepared")) throw uncertainCreationError()
  const now = Date.now()
  const clientChatId = existing?.operationId ?? crypto.randomUUID()
  if (!validClientChatId(clientChatId)) throw unavailableCreationError()
  const value: ChatCreationRecovery = {
    version: 1,
    operationId: clientChatId,
    clientChatId,
    handle: canonical,
    surface: params.surface,
    createdAt: existing?.createdAt ?? now,
    state: "prepared",
    inputText: params.inputText,
    characterId: params.setup.characterId,
    provider: params.setup.provider,
    model: params.setup.model,
    temperature: params.setup.temperature,
    topP: params.setup.topP
  }

  if (!current()) throw new Error("stale_selection")
  if (!(await creationRecovery.set(key, value, now, expected))) throw unavailableCreationError()
  notifyCreationRecovery()
  if (!current()) throw new Error("stale_selection")
  const prepared = { value, updatedAt: now }
  const dispatched = { ...value, state: "unconfirmed" as const }
  if (!(await creationRecovery.set(key, dispatched, now, prepared))) throw unavailableCreationError()
  notifyCreationRecovery()
  if (!current()) {
    // This callback has not been invoked: an exact conditional rollback is safe.
    await creationRecovery.set(key, value, now, { value: dispatched, updatedAt: now })
    notifyCreationRecovery()
    throw new Error("stale_selection")
  }
  let created: T
  try { created = await params.create(clientChatId) } catch (cause) { throw Object.assign(uncertainCreationError(), { cause }) }
  if (created?.id !== clientChatId) throw uncertainCreationError()
  // A late ACK belongs to this private attempt even if its originating view retired.
  if (!(await acknowledgeChatCreation(dispatched, created.id))) throw uncertainCreationError()
  return created
}
export const markChatCreationBound = async (
  handle: ChatDraftHandle,
  chatId: string,
  expectedOperationId?: string
) => {
  const attempt = await readChatCreationRecovery(handle)
  if (!attempt) return false
  const key = creationKey(attempt.handle)
  const record = await creationRecovery.get(key, { failClosed: true })
  if (
    !record ||
    !validCreationRecord(record.value) ||
    record.value.state !== "acknowledged" ||
    record.value.chatId !== chatId ||
    (expectedOperationId !== undefined &&
      record.value.operationId !== expectedOperationId)
  )
    return false
  const bound = await creationRecovery.set(
    key,
    { ...record.value, state: "bound" },
    record.updatedAt,
    record
  )
  if (bound) notifyCreationRecovery()
  return bound
}

/** Hide private input only after the native server settlement proves it was saved. */
export const markChatCreationSettled = async (handle: ChatDraftHandle, chatId: string, expectedOperationId?: string) => {
  try {
    const attempt = await readChatCreationRecovery(handle)
    if (!attempt) return false
    const key = creationKey(attempt.handle)
    const record = await creationRecovery.get(key, { failClosed: true })
    if (!record || !validCreationRecord(record.value) || record.value.chatId !== chatId || record.value.state !== "bound" ||
        (expectedOperationId !== undefined && record.value.operationId !== expectedOperationId)) return false
    const settled = await creationRecovery.set(key, { ...record.value, state: "settled", inputText: "" }, record.updatedAt, record)
    if (settled) notifyCreationRecovery()
    return settled
  } catch { return false } // Retain reachable text if private cleanup cannot be acknowledged.
}
