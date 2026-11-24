// Keep WebUI bootstrap independent of the API facade and its feature domains.
let bootstrapRecovery: ((isCurrent: () => boolean) => Promise<void>) | null =
  null
let bootstrapRecoveryGeneration = 0
let bootstrapRecoveryInFlight: Promise<void> | null = null
// Every initialize() asks for recovery; space failed attempts 5 s, 10 s, ... up to 60 s
// on the monotonic clock, so a wall-clock change cannot stall or rush them.
const RECOVERY_RETRY_BASE_MS = 5_000
const RECOVERY_RETRY_MAX_MS = 60_000
let recoveryFailures = 0
let nextRecoveryAt = 0
let cookieNotKept = false
const cookieNotKeptListeners = new Set<() => void>()

/** A click is the user asking now; the backoff only spaces automatic retries. */
const allowRetryNow = (): void => {
  recoveryFailures = 0
  nextRecoveryAt = 0
}

/** Retire recovery before an explicit credential or account boundary. */
export const cancelRuntimeAuthBootstrapRecovery = (): void => {
  bootstrapRecoveryGeneration++
  bootstrapRecovery = null
  bootstrapRecoveryInFlight = null
  if (typeof window !== "undefined")
    window.removeEventListener("click", allowRetryNow, true)
}

/** The WebUI supplies its existing bounded cookie bootstrap after transient failure. */
export const setRuntimeAuthBootstrapRecovery = (
  recover: (isCurrent: () => boolean) => Promise<void>
): (() => boolean) => {
  cancelRuntimeAuthBootstrapRecovery()
  bootstrapRecovery = recover
  allowRetryNow()
  if (typeof window !== "undefined")
    window.addEventListener("click", allowRetryNow, true)
  const generation = bootstrapRecoveryGeneration
  return () =>
    generation === bootstrapRecoveryGeneration && recover === bootstrapRecovery
}

/** The sign-in cookie was not kept or not accepted; automatic sign-in stopped for this page. */
export const reportRuntimeAuthCookieNotKept = (): void => {
  if (cookieNotKept) return
  cookieNotKept = true
  cookieNotKeptListeners.forEach((listener) => listener())
}

export const isRuntimeAuthCookieNotKept = (): boolean => cookieNotKept

export const subscribeRuntimeAuthCookieNotKept = (
  listener: () => void
): (() => void) => {
  cookieNotKeptListeners.add(listener)
  return () => {
    cookieNotKeptListeners.delete(listener)
  }
}

/** Share the current eligible bootstrap attempt between initializing clients. */
export const recoverRuntimeAuthBootstrap = async (): Promise<void> => {
  const recover = bootstrapRecovery
  if (!recover) return
  if (bootstrapRecoveryInFlight) return await bootstrapRecoveryInFlight
  if (performance.now() < nextRecoveryAt) return
  const generation = bootstrapRecoveryGeneration
  const isCurrent = () =>
    generation === bootstrapRecoveryGeneration && recover === bootstrapRecovery
  const pending = Promise.resolve()
    .then(() => recover(isCurrent))
    .catch(() => undefined)
    .finally(() => {
      if (bootstrapRecoveryInFlight === pending)
        bootstrapRecoveryInFlight = null
      // Recovery retires itself on success or a final failure; still being current means "retry later".
      if (isCurrent())
        nextRecoveryAt =
          performance.now() +
          Math.min(
            RECOVERY_RETRY_BASE_MS * 2 ** recoveryFailures++,
            RECOVERY_RETRY_MAX_MS
          )
    })
  bootstrapRecoveryInFlight = pending
  await pending
}
