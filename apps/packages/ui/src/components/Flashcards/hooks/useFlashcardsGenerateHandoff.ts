import React from "react"
import type { ServicePromptSnapshot } from "@/services/service-prompts"
import {
  consumeFlashcardsGenerateHandoff,
  readFlashcardsGenerateRoute,
  removeFlashcardsGenerateHandoff,
  type FlashcardsGenerateIntent
} from "@/services/tldw/flashcards-generate-handoff"
import { flashcardsHandoffAuthority, loadFlashcardsTransferSnapshot } from "@/services/tldw/flashcards-generate-transfer"

export const useFlashcardsGenerateHandoff = (
  location: { pathname?: string; search?: string; hash?: string },
  navigate: (route: string, options: { replace: boolean }) => void
) => {
  const { pathname, search, hash } = location
  const route = React.useMemo(() => readFlashcardsGenerateRoute({ pathname, search, hash }), [pathname, search, hash])
  const latest = React.useRef({ route, navigate })
  latest.current = { route, navigate }
  const [revision, setRevision] = React.useState(0)
  const [scope, setScope] = React.useState<ServicePromptSnapshot | null>(null)
  const [accepted, setAccepted] = React.useState<{ token: string; intent: FlashcardsGenerateIntent } | null>(null)
  const [error, setError] = React.useState<string | null>(null)

  React.useEffect(() => {
    const controller = new AbortController()
    let snapshot: ServicePromptSnapshot | undefined
    let invalidated = false
    const invalidate = () => {
      if (invalidated) return
      invalidated = true
      controller.abort()
      setScope(null)
      setAccepted(null)
      if (snapshot && latest.current.route.token) {
        void removeFlashcardsGenerateHandoff(latest.current.route.token).catch(() => {
          console.warn("Could not remove an invalidated Flashcards transfer.")
        })
        latest.current.navigate(latest.current.route.cleanRoute, { replace: true })
      }
      setRevision(value => value + 1)
    }
    const configChanged = (event: Event) => {
      const detail = (event as CustomEvent<{ authorityChanged?: boolean; refreshSessionInvalidated?: boolean }>).detail
      if (detail?.authorityChanged !== false || detail.refreshSessionInvalidated) invalidate()
    }
    window.addEventListener("tldw:auth-credentials-changed", invalidate)
    window.addEventListener("tldw:auth-principal-changed", invalidate)
    window.addEventListener("tldw:config-updated", configChanged)
    void loadFlashcardsTransferSnapshot(controller.signal).then(value => {
      if (controller.signal.aborted || value.scopeSignal.aborted) { value.release(); return }
      snapshot = value
      value.scopeSignal.addEventListener("abort", invalidate, { once: true })
      setScope(value)
    }).catch(() => {
      if (!controller.signal.aborted) setError("Sign in to the intended account, then reopen the transfer from its source.")
    })
    return () => {
      window.removeEventListener("tldw:auth-credentials-changed", invalidate)
      window.removeEventListener("tldw:auth-principal-changed", invalidate)
      window.removeEventListener("tldw:config-updated", configChanged)
      snapshot?.scopeSignal.removeEventListener("abort", invalidate)
      controller.abort()
      snapshot?.release()
    }
  }, [revision])

  React.useEffect(() => {
    if (route.legacy) {
      setAccepted(null)
      setError("This old link contains unbound source text. Reopen Flashcards from the original source while signed in to its account.")
      latest.current.navigate(latest.current.route.cleanRoute, { replace: true })
      return
    }
    if (!route.token || !scope || scope.scopeSignal.aborted) return
    const controller = new AbortController()
    const abort = () => controller.abort()
    scope.scopeSignal.addEventListener("abort", abort, { once: true })
    setAccepted(null)
    void consumeFlashcardsGenerateHandoff(route.token, flashcardsHandoffAuthority(scope), controller.signal).then(intent => {
      if (controller.signal.aborted || latest.current.route.token !== route.token) return
      setAccepted({ token: route.token!, intent })
      setError(null)
      latest.current.navigate(latest.current.route.cleanRoute, { replace: true })
    }).catch(error => {
      if (controller.signal.aborted || latest.current.route.token !== route.token) return
      setError(error instanceof Error ? error.message : "The transfer could not be opened. Reopen it from the source.")
      latest.current.navigate(latest.current.route.cleanRoute, { replace: true })
    })
    return () => {
      controller.abort()
      scope.scopeSignal.removeEventListener("abort", abort)
    }
  }, [route.token, route.legacy, scope])

  return {
    intent: scope?.scopeSignal.aborted ? null : accepted?.intent ?? null,
    scope: scope?.scopeSignal.aborted ? null : scope,
    generationKey: `${revision}:${accepted?.token ?? "manual"}`,
    authorityRevision: revision,
    hasRoute: Boolean(route.token || route.legacy),
    error
  }
}
