// Keep WebUI bootstrap independent of the API facade and its feature domains.
let bootstrapRecovery: ((isCurrent: () => boolean) => Promise<void>) | null =
  null
let bootstrapRecoveryGeneration = 0
let bootstrapRecoveryInFlight: Promise<void> | null = null
// Every initialize() asks for recovery; space failed attempts 5 s, 10 s, ... up to 60 s.
const RECOVERY_RETRY_BASE_MS = 5_000
const RECOVERY_RETRY_MAX_MS = 60_000
let recoveryFailures = 0
let nextRecoveryAt = 0
let cookieNotKept = false
const cookieNotKeptListeners = new Set<() => void>()

const setCookieNotKept = (value: boolean): void => {
  if (cookieNotKept === value) return
  cookieNotKept = value
  cookieNotKeptListeners.forEach((listener) => listener())
}

/** Retire recovery before an explicit credential or account boundary. */
export const cancelRuntimeAuthBootstrapRecovery = (): void => {
  bootstrapRecoveryGeneration++
  bootstrapRecovery = null
  bootstrapRecoveryInFlight = null
  setCookieNotKept(false)
}

/** The WebUI supplies its existing bounded cookie bootstrap after transient failure. */
export const setRuntimeAuthBootstrapRecovery = (
  recover: (isCurrent: () => boolean) => Promise<void>
): (() => boolean) => {
  cancelRuntimeAuthBootstrapRecovery()
  bootstrapRecovery = recover
  allowRuntimeAuthBootstrapRetry()
  const generation = bootstrapRecoveryGeneration
  return () =>
    generation === bootstrapRecoveryGeneration && recover === bootstrapRecovery
}

/** An explicit user action may retry now instead of waiting out the backoff. */
export const allowRuntimeAuthBootstrapRetry = (): void => {
  recoveryFailures = 0
  nextRecoveryAt = 0
}

/** The browser dropped the sign-in cookie, so automatic sign-in stopped for this page. */
export const reportRuntimeAuthCookieNotKept = (): void => setCookieNotKept(true)

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
  if (Date.now() < nextRecoveryAt) return
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
          Date.now() +
          Math.min(
            RECOVERY_RETRY_BASE_MS * 2 ** recoveryFailures++,
            RECOVERY_RETRY_MAX_MS
          )
    })
  bootstrapRecoveryInFlight = pending
  await pending
}
