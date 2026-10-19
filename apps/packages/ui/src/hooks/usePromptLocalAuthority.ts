import React from "react"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { deriveConnectionAuthorityId } from "@/services/chat-surface-scope"
import { resolvePromptOwnerId } from "@/services/prompt-local-owner"

/** Retire private local prompt state while a new connection/principal resolves. */
export function usePromptLocalAuthority() {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const boundaryKey = deriveConnectionAuthorityId(config)
  const hasAuthenticatedOwnership = config?.authMode === "multi-user" || config?.authSource === "cookie-session" || config?.credentialSource === "manual" || !!config?.apiKey || !!config?.accessToken
  const [revision, setRevision] = React.useState(0)
  const [resolved, setResolved] = React.useState<{ key: string; revision: number; ownerId: string | null } | null>(null)
  React.useEffect(() => {
    let current = true
    if (authorityLoading || !hasAuthenticatedOwnership) return
    void resolvePromptOwnerId().then(ownerId => {
      if (current) setResolved({ key: boundaryKey, revision, ownerId })
    }).catch(() => { if (current) setResolved(null) })
    return () => { current = false }
  }, [authorityLoading, boundaryKey, hasAuthenticatedOwnership, revision])
  React.useEffect(() => {
    const retire = () => { setResolved(null); setRevision(value => value + 1) }
    const configChanged = (event: Event) => {
      if ((event as CustomEvent<{ authorityChanged?: boolean }>).detail?.authorityChanged !== false) retire()
    }
    window.addEventListener("tldw:auth-principal-changed", retire)
    window.addEventListener("tldw:config-updated", configChanged)
    return () => {
      window.removeEventListener("tldw:auth-principal-changed", retire)
      window.removeEventListener("tldw:config-updated", configChanged)
    }
  }, [])
  const ready = !authorityLoading && resolved?.key === boundaryKey && resolved.revision === revision
  const localOnlyReady = !authorityLoading && !!config && !hasAuthenticatedOwnership
  return { ownerId: ready ? resolved.ownerId : null, ready, localOnlyReady, boundaryKey }
}
