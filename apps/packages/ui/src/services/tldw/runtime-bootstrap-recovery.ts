// Keep WebUI bootstrap independent of the API facade and its feature domains.
let bootstrapRecovery: ((isCurrent: () => boolean) => Promise<void>) | null =
  null
let bootstrapRecoveryGeneration = 0
let bootstrapRecoveryInFlight: Promise<void> | null = null

/** Retire recovery before an explicit credential or account boundary. */
export const cancelRuntimeAuthBootstrapRecovery = (): void => {
  bootstrapRecoveryGeneration++
  bootstrapRecovery = null
  bootstrapRecoveryInFlight = null
}

/** The WebUI supplies its existing bounded cookie bootstrap after transient failure. */
export const setRuntimeAuthBootstrapRecovery = (
  recover: (isCurrent: () => boolean) => Promise<void>
): (() => boolean) => {
  cancelRuntimeAuthBootstrapRecovery()
  bootstrapRecovery = recover
  const generation = bootstrapRecoveryGeneration
  return () =>
    generation === bootstrapRecoveryGeneration && recover === bootstrapRecovery
}

/** Share the current eligible bootstrap attempt between initializing clients. */
export const recoverRuntimeAuthBootstrap = async (): Promise<void> => {
  const recover = bootstrapRecovery
  if (!recover) return
  if (bootstrapRecoveryInFlight) return await bootstrapRecoveryInFlight
  const generation = bootstrapRecoveryGeneration
  const pending = Promise.resolve()
    .then(() =>
      recover(
        () =>
          generation === bootstrapRecoveryGeneration &&
          recover === bootstrapRecovery
      )
    )
    .catch(() => undefined)
    .finally(() => {
      if (bootstrapRecoveryInFlight === pending)
        bootstrapRecoveryInFlight = null
    })
  bootstrapRecoveryInFlight = pending
  await pending
}
