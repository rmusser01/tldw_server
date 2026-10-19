import React from "react"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { deriveConnectionAuthorityId, deriveSingleUserApiKeyCredentialScope } from "@/services/chat-surface-scope"
import { servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error"
import { resolveServicePromptScope } from "@/services/service-prompts"
import type { ServicePromptRequestScope } from "@/services/tldw/domains/service-prompts"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"

export type PrivateWorkspaceScope = {
  key: string
  config: TldwConfig
  requestScope: ServicePromptRequestScope
  signal: AbortSignal
  isCurrent: () => boolean
}

/** Retire private state before re-verifying even an identical cookie connection. */
export function usePrivateWorkspaceScope(): { scope: PrivateWorkspaceScope | null; retry: () => void } {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const authority = deriveConnectionAuthorityId(config)
  const configRef = React.useRef(config)
  configRef.current = config
  const authorityRef = React.useRef(authority)
  authorityRef.current = authority
  const [scope, setScope] = React.useState<(PrivateWorkspaceScope & { authority: string }) | null>(null)
  const current = React.useRef<(PrivateWorkspaceScope & { authority: string }) | null>(null)
  const [revision, setRevision] = React.useState(0)
  const retry = React.useCallback(() => setRevision(value => value + 1), [])

  React.useLayoutEffect(() => {
    let mounted = true
    let generation = 0
    let pending: AbortController | null = null
    let lifetime: AbortController | null = null
    let verifiedIdentity: string | null = null
    const retire = () => {
      lifetime?.abort()
      current.current = null
      verifiedIdentity = null
      setScope(null)
    }
    if (authorityLoading || !configRef.current) {
      retire()
      return
    }
    const resolve = async (invalidated = false) => {
      const attempt = ++generation
      if (invalidated) retire()
      pending?.abort()
      const controller = new AbortController()
      pending = controller
      try {
        const verified = await resolveServicePromptScope({ signal: controller.signal })
        if (!mounted || generation !== attempt || controller.signal.aborted) return
        const captured = configRef.current
        if (!captured || !servicePromptTargetsMatch(captured, verified.config)) throw new Error("Workspace connection changed")
        // Cookie principals are verified by /me; they have no client JWT for legacy-cache verification.
        const cookiePrincipal = captured.authSource === "cookie-session" && verified.userId != null
        if (!verified.clientPrincipalVerified && !cookiePrincipal) throw new Error("Workspace principal unresolved")
        if (captured.authMode === "single-user" &&
          deriveSingleUserApiKeyCredentialScope(captured.authMode, captured.apiKey) !== verified.config.expectedSingleUserApiKeyScope) {
          throw new Error("Workspace credential changed")
        }
        const identity = JSON.stringify([verified.scopeKey, verified.userId])
        if (current.current && verifiedIdentity === identity) return
        retire()
        verifiedIdentity = identity
        lifetime = controller
        pending = null
        const next: PrivateWorkspaceScope & { authority: string } = {
          authority,
          key: JSON.stringify([authority, verified.scopeKey, attempt]),
          get config() { return configRef.current! },
          requestScope: { config: verified.config, userId: verified.userId },
          signal: controller.signal,
          isCurrent: () => current.current === next && !controller.signal.aborted && authorityRef.current === authority
        }
        current.current = next
        setScope(next)
      } catch {
        if (mounted && generation === attempt && !controller.signal.aborted) retire()
      }
    }
    const stop = watchChatAccountChanges((invalidated) => { void resolve(invalidated) })
    void resolve()
    return () => {
      mounted = false
      generation++
      stop()
      pending?.abort()
      lifetime?.abort()
      current.current = null
    }
  }, [authority, authorityLoading, revision])

  return { scope: !authorityLoading && scope?.authority === authority && scope.isCurrent() ? scope : null, retry }
}
