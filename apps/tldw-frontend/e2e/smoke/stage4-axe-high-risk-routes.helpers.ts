export type RedirectDisposition =
  | {
      shouldSkip: false
    }
  | {
      shouldSkip: true
      message: string
    }

type RedirectDispositionInput = {
  routePath: string
  finalPath: string
  mayRedirectWhenUnavailable?: boolean
  navigationObservedDuringScan?: boolean
}

export function getRedirectDispositionForA11yScan({
  routePath,
  finalPath,
  mayRedirectWhenUnavailable,
  navigationObservedDuringScan = false
}: RedirectDispositionInput): RedirectDisposition {
  if (!mayRedirectWhenUnavailable) {
    return { shouldSkip: false }
  }

  if (!navigationObservedDuringScan && finalPath === routePath) {
    return { shouldSkip: false }
  }

  if (finalPath !== routePath) {
    return {
      shouldSkip: true,
      message: `Route ${routePath} redirected to ${finalPath}; feature is unavailable in this runtime`
    }
  }

  return {
    shouldSkip: true,
    message: `Route ${routePath} reloaded during accessibility scan; feature is unavailable in this runtime`
  }
}
