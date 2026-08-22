import React from "react"

import { useServerOnline } from "@/hooks/useServerOnline"
import { tldwClient, type SlidesCapabilities } from "@/services/tldw/TldwApiClient"
import { tldwAuth } from "@/services/tldw/TldwAuth"

export type SlidesCapabilityStatus =
  | "loading"
  | "ready"
  | "generation_disabled"
  | "validator_unavailable"
  | "auth_required"
  | "forbidden"
  | "error"
  | "offline"

export type UseSlidesCapabilitiesResult = {
  capabilities: SlidesCapabilities | null
  status: SlidesCapabilityStatus
  reason: string | null
  canGenerate: boolean
  canReadStandalone: boolean
  canDraftStandalone: boolean
  canEditStandalone: boolean
  retry: () => Promise<void>
}

type CapabilityScope = { serverOrigin: string; principalId: string }

const sameScope = (left: CapabilityScope, right: CapabilityScope): boolean =>
  left.serverOrigin === right.serverOrigin && left.principalId === right.principalId

const resolveCapabilityScope = async (): Promise<CapabilityScope | null> => {
  try {
    const [config, user] = await Promise.all([tldwClient.getConfig(), tldwAuth.getCurrentUser()])
    const configured = typeof config?.serverUrl === "string" ? config.serverUrl.trim() : ""
    const fallback = typeof window !== "undefined" ? window.location.origin : ""
    const serverOrigin = new URL(configured || fallback, fallback || undefined).origin.toLowerCase()
    const principalId = String(user?.id ?? "").trim()
    return serverOrigin && principalId ? { serverOrigin, principalId } : null
  } catch {
    return null
  }
}

const errorStatus = (error: unknown): number | null => {
  const status = error && typeof error === "object" ? (error as { status?: unknown }).status : null
  return typeof status === "number" && Number.isFinite(status) ? status : null
}

export const useSlidesCapabilities = (): UseSlidesCapabilitiesResult => {
  const online = useServerOnline()
  const [capabilities, setCapabilities] = React.useState<SlidesCapabilities | null>(null)
  const [status, setStatus] = React.useState<SlidesCapabilityStatus>(
    online ? "loading" : "offline"
  )
  const [failureReason, setFailureReason] = React.useState<string | null>(null)
  const mountedRef = React.useRef(true)
  const requestIdRef = React.useRef(0)
  const abortRef = React.useRef<AbortController | null>(null)

  const invalidate = React.useCallback((nextStatus: SlidesCapabilityStatus) => {
    requestIdRef.current += 1
    abortRef.current?.abort()
    abortRef.current = null
    setCapabilities(null)
    setFailureReason(null)
    setStatus(nextStatus)
  }, [])

  const fetchCapabilities = React.useCallback(async () => {
    const requestId = ++requestIdRef.current
    abortRef.current?.abort()
    abortRef.current = null
    if (!online) {
      if (mountedRef.current && requestId === requestIdRef.current) {
        setCapabilities(null)
        setFailureReason(null)
        setStatus("offline")
      }
      return
    }

    setCapabilities(null)
    setFailureReason(null)
    setStatus("loading")
    const expectedScope = await resolveCapabilityScope()
    if (!mountedRef.current || requestId !== requestIdRef.current) return
    if (!expectedScope) {
      setStatus("error")
      return
    }

    const controller = new AbortController()
    abortRef.current = controller
    try {
      const next = await tldwClient.getSlidesCapabilities({ abortSignal: controller.signal })
      if (!mountedRef.current || requestId !== requestIdRef.current || controller.signal.aborted) return
      const confirmedScope = await resolveCapabilityScope()
      if (
        !mountedRef.current ||
        requestId !== requestIdRef.current ||
        controller.signal.aborted ||
        !confirmedScope ||
        !sameScope(expectedScope, confirmedScope)
      ) return
      setCapabilities(next)
      const htmlContent = next.content_kinds.standalone_html
      const generation = next.generation_modes.standalone_html
      if (htmlContent.reason === "validator_unavailable") {
        setStatus("validator_unavailable")
      } else if (!generation.enabled) {
        setStatus("generation_disabled")
      } else {
        setStatus("ready")
      }
    } catch (error) {
      if (!mountedRef.current || requestId !== requestIdRef.current || controller.signal.aborted) return
      setCapabilities(null)
      const nextStatus = errorStatus(error)
      if (nextStatus === 401) {
        setFailureReason("authentication_required")
        setStatus("auth_required")
      } else if (nextStatus === 403) {
        setFailureReason("permission_denied")
        setStatus("forbidden")
      } else {
        setStatus("error")
      }
    } finally {
      if (abortRef.current === controller) abortRef.current = null
    }
  }, [online])

  React.useEffect(() => {
    mountedRef.current = true
    void fetchCapabilities()
    return () => {
      mountedRef.current = false
      requestIdRef.current += 1
      abortRef.current?.abort()
      abortRef.current = null
    }
  }, [fetchCapabilities])

  React.useEffect(() => {
    const restore = () => { void fetchCapabilities() }
    const pagehide = () => invalidate(online ? "loading" : "offline")
    const visibility = () => {
      if (document.visibilityState === "visible") restore()
    }
    window.addEventListener("tldw:config-updated", restore)
    window.addEventListener("tldw:auth-principal-changed", restore)
    window.addEventListener("pagehide", pagehide)
    window.addEventListener("pageshow", restore)
    window.addEventListener("focus", restore)
    document.addEventListener("visibilitychange", visibility)
    return () => {
      window.removeEventListener("tldw:config-updated", restore)
      window.removeEventListener("tldw:auth-principal-changed", restore)
      window.removeEventListener("pagehide", pagehide)
      window.removeEventListener("pageshow", restore)
      window.removeEventListener("focus", restore)
      document.removeEventListener("visibilitychange", visibility)
    }
  }, [fetchCapabilities, invalidate, online])

  const htmlContent = capabilities?.content_kinds.standalone_html
  const generation = capabilities?.generation_modes.standalone_html

  return {
    capabilities,
    status,
    reason: failureReason ?? (generation && !generation.enabled ? generation.reason : htmlContent?.reason ?? null),
    canGenerate: status === "ready" && generation?.enabled === true,
    canReadStandalone: htmlContent?.read === true,
    canDraftStandalone: htmlContent?.draft_attachment === true,
    canEditStandalone: htmlContent?.edit === true,
    retry: fetchCapabilities
  }
}
