import type { ServicePromptSnapshot } from "@/services/service-prompts"

type PendingPromotion = {
  historyId: string | null
  scope: Promise<ServicePromptSnapshot>
  snapshot?: ServicePromptSnapshot
  completion: Promise<void>
  saved: Promise<void>
  finishWaiting: () => void
  failed: boolean
  error?: unknown
  promote: (snapshot: ServicePromptSnapshot) => Promise<void>
  abort: () => void
  onFailure: (error: unknown, retrying: boolean) => void
  onComplete: () => void
}

const pendingPromotions = new WeakMap<object, PendingPromotion>()
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
      await pending.promote(snapshot)
      snapshot.scopeSignal.throwIfAborted()
      if (pendingPromotions.get(owner) === pending)
        pendingPromotions.delete(owner)
      snapshot.release()
      pending.finishWaiting()
      pending.onComplete()
    })
    .catch((error) => {
      if (!pending.snapshot || pending.snapshot.scopeSignal.aborted) {
        if (pendingPromotions.get(owner) === pending)
          pendingPromotions.delete(owner)
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
  options: Pick<PendingPromotion, "abort" | "onFailure" | "onComplete">
): Promise<void> => {
  const previous = pendingPromotions.get(owner)
  if (previous) clearChatPromotion(owner, previous.historyId)
  let finishWaiting!: () => void
  const saved = new Promise<void>(resolve => { finishWaiting = resolve })
  const pending: PendingPromotion = {
    saved,
    finishWaiting,
    historyId,
    scope,
    promote,
    ...options,
    failed: false,
    completion: Promise.resolve()
  }
  pendingPromotions.set(owner, pending)
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
  options?: { waitUntilSaved?: boolean }
): Promise<void> => {
  snapshot.scopeSignal.throwIfAborted()
  const pending = pendingPromotions.get(owner)
  if (!pending || pending.historyId !== historyId) return
  let ownerScope: ServicePromptSnapshot
  try {
    ownerScope = await waitWithoutCancellingOwner(
      pending.scope,
      snapshot.scopeSignal
    )
  } catch (error) {
    snapshot.scopeSignal.throwIfAborted()
    if (error instanceof Error && error.name === "AbortError") return
    throw error
  }
  if (
    ownerScope.scopeInvalidatedSignal.aborted ||
    ownerScope.scopeKey !== snapshot.scopeKey
  )
    return
  if (pending.failed && !options?.waitUntilSaved) pending.onFailure(pending.error, false)
  await waitWithoutCancellingOwner(options?.waitUntilSaved ? pending.saved : pending.completion, snapshot.scopeSignal)
  if (options?.waitUntilSaved) ownerScope.scopeSignal.throwIfAborted()
  snapshot.scopeSignal.throwIfAborted()
}
