import { describe, expect, it, vi } from "vitest"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import { clearChatPromotion, retryChatPromotion, trackChatPromotion, waitForChatPromotion } from "../pending-chat-promotion"

const deferred = <T,>() => {
  let resolve!: (value: T) => void
  const promise = new Promise<T>(done => { resolve = done })
  return { promise, resolve }
}
const lease = (controller = new AbortController()) => ({
  scopeKey: "owner-A",
  scopeSignal: controller.signal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: vi.fn()
}) as unknown as ServicePromptSnapshot
const callbacks = (controller: AbortController) => ({
  abort: () => controller.abort(), onFailure: vi.fn(), onComplete: vi.fn()
})
const flush = async () => { for (let index = 0; index < 12; index++) await Promise.resolve() }

describe("saved loader promotion wait lifecycle", () => {
  it("rejects a loader wait when its owning lease resolves already aborted", async () => {
    const owner = {}
    const controller = new AbortController()
    controller.abort()
    const resolving = deferred<ServicePromptSnapshot>()
    const promote = vi.fn()
    const operation = trackChatPromotion(owner, "history-A", resolving.promise, promote, callbacks(controller)).catch(error => error)
    const loaderLease = lease()
    let outcome = "pending"
    const waiting = waitForChatPromotion(owner, "history-A", loaderLease, { waitUntilSaved: true }).then(() => { outcome = "resolved" }, () => { outcome = "rejected" })
    resolving.resolve(lease(controller))
    await operation
    await flush()
    expect(loaderLease.scopeSignal.aborted).toBe(false)
    expect(outcome).toBe("rejected")
    await waiting
    expect(promote).not.toHaveBeenCalled()
  })

  it.each(["history-A", "history-B"])("retires the previous loader waiter before replacing the owner with %s", async nextHistory => {
    const owner = {}
    const controller = new AbortController()
    const held = deferred<void>()
    const oldLease = lease(controller)
    const oldOperation = trackChatPromotion(owner, "history-A", Promise.resolve(oldLease), () => held.promise, callbacks(controller)).catch(error => error)
    await flush()
    let outcome = "pending"
    const waiting = waitForChatPromotion(owner, "history-A", lease(), { waitUntilSaved: true }).then(() => { outcome = "resolved" }, () => { outcome = "rejected" })
    await flush()
    const nextController = new AbortController()
    await trackChatPromotion(owner, nextHistory, Promise.resolve(lease(nextController)), async () => undefined, callbacks(nextController))
    await flush()
    try {
      expect(outcome).toBe("rejected")
      expect(controller.signal.aborted).toBe(true)
      await waiting
      expect(nextController.signal.aborted).toBe(false)
    } finally {
      held.resolve()
      await oldOperation
      clearChatPromotion(owner, nextHistory)
    }
  })

  it("keeps an incomplete save recoverable while send rejects and loader awaits explicit Retry", async () => {
    const owner = {}, controller = new AbortController()
    const promote = vi.fn().mockRejectedValueOnce(new Error("Ambiguous write")).mockResolvedValueOnce(undefined)
    await expect(trackChatPromotion(owner, "history-A", Promise.resolve(lease(controller)), promote, callbacks(controller))).rejects.toMatchObject({ code: "chat_promotion_incomplete" })
    let saved = false
    const waiting = waitForChatPromotion(owner, "history-A", lease(), { waitUntilSaved: true }).then(() => { saved = true })
    await expect(waitForChatPromotion(owner, "history-A", lease())).rejects.toMatchObject({ code: "chat_promotion_incomplete" })
    expect(saved).toBe(false)
    await retryChatPromotion(owner, "history-A")
    await waiting
    expect(saved).toBe(true)
    expect(promote).toHaveBeenCalledTimes(2)
  })

  it("cancels the loader without cancelling its owner and releases a later loader on clear", async () => {
    const owner = {}, controller = new AbortController(), loaderController = new AbortController()
    const held = deferred<void>()
    const operation = trackChatPromotion(owner, "history-A", Promise.resolve(lease(controller)), () => held.promise, callbacks(controller)).catch(error => error)
    const waiting = waitForChatPromotion(owner, "history-A", lease(loaderController), { waitUntilSaved: true }).catch(error => error)
    await flush()
    loaderController.abort()
    expect((await waiting).name).toBe("AbortError")
    expect(controller.signal.aborted).toBe(false)
    const later = waitForChatPromotion(owner, "history-A", lease(), { waitUntilSaved: true }).catch(error => error)
    await flush()
    clearChatPromotion(owner, "history-A")
    expect((await later).name).toBe("AbortError")
    held.resolve()
    await operation
  })
})
