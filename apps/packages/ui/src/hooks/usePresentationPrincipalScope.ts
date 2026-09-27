import React from "react"

import { tldwClient } from "@/services/tldw/TldwApiClient"
import { tldwAuth } from "@/services/tldw/TldwAuth"
import {
  createPresentationPrincipalScope,
  type PresentationPrincipalScope
} from "@/components/Option/PresentationStudio/standalone-html-recovery"

export type PresentationPrincipalStatus = "loading" | "ready" | "guarded"
export type PresentationPrincipalBoundaryKind = "reauthenticate" | "switch" | "logout" | "mismatch"

export const usePresentationPrincipalScope = (
  options: { onBoundary?: (kind?: PresentationPrincipalBoundaryKind) => void } = {}
): {
  status: PresentationPrincipalStatus
  scope: PresentationPrincipalScope | null
  retry: () => Promise<void>
} => {
  const [status, setStatus] = React.useState<PresentationPrincipalStatus>("loading")
  const [scope, setScope] = React.useState<PresentationPrincipalScope | null>(null)
  const mountedRef = React.useRef(true)
  const epochRef = React.useRef(0)
  const callbackRef = React.useRef(options.onBoundary)
  const trustedScopeRef = React.useRef<PresentationPrincipalScope | null>(null)
  callbackRef.current = options.onBoundary

  const resolve = React.useCallback(async () => {
    const epoch = ++epochRef.current
    setScope(null)
    setStatus("loading")
    try {
      const [config, user] = await Promise.all([
        tldwClient.getConfig(),
        tldwAuth.getCurrentUser()
      ])
      if (!mountedRef.current || epoch !== epochRef.current) return
      const configured = typeof config?.serverUrl === "string" ? config.serverUrl.trim() : ""
      const fallback = typeof window !== "undefined" ? window.location.origin : ""
      const principal = String(user?.id ?? "").trim()
      if (!principal || !user?.is_active) throw new Error("principal unavailable")
      const nextScope = createPresentationPrincipalScope(configured || fallback, principal)
      const previousScope = trustedScopeRef.current
      if (previousScope && previousScope.principalScope !== nextScope.principalScope) {
        callbackRef.current?.("mismatch")
      }
      trustedScopeRef.current = nextScope
      setScope(nextScope)
      setStatus("ready")
    } catch {
      if (!mountedRef.current || epoch !== epochRef.current) return
      setScope(null)
      setStatus("guarded")
    }
  }, [])

  React.useEffect(() => {
    mountedRef.current = true
    void resolve()
    return () => {
      mountedRef.current = false
      epochRef.current += 1
    }
  }, [resolve])

  React.useEffect(() => {
    const invalidateAndResolve = () => {
      epochRef.current += 1
      callbackRef.current?.("reauthenticate")
      setScope(null)
      setStatus("loading")
      void resolve()
    }
    const authBoundary = (event: Event) => {
      epochRef.current += 1
      const kind = (event as CustomEvent<{ kind?: string }>).detail?.kind
      callbackRef.current?.(kind === "logout" ? "logout" : "switch")
      trustedScopeRef.current = null
      setScope(null)
      if (kind === "logout") {
        setStatus("guarded")
        return
      }
      setStatus("loading")
      void resolve()
    }
    const visible = () => {
      if (document.visibilityState === "visible") invalidateAndResolve()
    }
    const scopeMismatch = () => {
      epochRef.current += 1
      callbackRef.current?.("mismatch")
      trustedScopeRef.current = null
      setScope(null)
      setStatus("guarded")
    }

    window.addEventListener("tldw:auth-principal-changed", authBoundary)
    window.addEventListener("tldw:slides-scope-mismatch", scopeMismatch)
    window.addEventListener("tldw:config-updated", invalidateAndResolve)
    window.addEventListener("pageshow", invalidateAndResolve)
    window.addEventListener("focus", invalidateAndResolve)
    document.addEventListener("visibilitychange", visible)
    return () => {
      window.removeEventListener("tldw:auth-principal-changed", authBoundary)
      window.removeEventListener("tldw:slides-scope-mismatch", scopeMismatch)
      window.removeEventListener("tldw:config-updated", invalidateAndResolve)
      window.removeEventListener("pageshow", invalidateAndResolve)
      window.removeEventListener("focus", invalidateAndResolve)
      document.removeEventListener("visibilitychange", visible)
    }
  }, [resolve])

  return { status, scope, retry: resolve }
}
