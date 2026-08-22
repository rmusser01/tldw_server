import React from "react"

import { useServerOnline } from "@/hooks/useServerOnline"
import { tldwClient, type SlidesCapabilities } from "@/services/tldw/TldwApiClient"

export type SlidesCapabilityStatus =
  | "loading"
  | "ready"
  | "generation_disabled"
  | "validator_unavailable"
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

export const useSlidesCapabilities = (): UseSlidesCapabilitiesResult => {
  const online = useServerOnline()
  const [capabilities, setCapabilities] = React.useState<SlidesCapabilities | null>(null)
  const [status, setStatus] = React.useState<SlidesCapabilityStatus>(
    online ? "loading" : "offline"
  )
  const mountedRef = React.useRef(true)

  React.useEffect(() => {
    mountedRef.current = true
    return () => {
      mountedRef.current = false
    }
  }, [])

  const fetchCapabilities = React.useCallback(async () => {
    if (!online) {
      if (mountedRef.current) {
        setCapabilities(null)
        setStatus("offline")
      }
      return
    }
    if (mountedRef.current) setStatus("loading")
    try {
      const next = await tldwClient.getSlidesCapabilities()
      if (!mountedRef.current) return
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
    } catch {
      if (!mountedRef.current) return
      setCapabilities(null)
      setStatus("error")
    }
  }, [online])

  React.useEffect(() => {
    void fetchCapabilities()
  }, [fetchCapabilities])

  const htmlContent = capabilities?.content_kinds.standalone_html
  const generation = capabilities?.generation_modes.standalone_html

  return {
    capabilities,
    status,
    reason: generation && !generation.enabled ? generation.reason : htmlContent?.reason ?? null,
    canGenerate: status === "ready" && generation?.enabled === true,
    canReadStandalone: htmlContent?.read === true,
    canDraftStandalone: htmlContent?.draft_attachment === true,
    canEditStandalone: htmlContent?.edit === true,
    retry: fetchCapabilities
  }
}
