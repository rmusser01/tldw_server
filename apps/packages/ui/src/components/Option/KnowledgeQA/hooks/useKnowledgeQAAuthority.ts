import { useEffect, useMemo, useRef, useState } from "react"
import { useCanonicalConnectionConfig } from "@/hooks/useCanonicalConnectionConfig"
import { useConnectionStore } from "@/store/connection"
import { connectionAuthoritiesMatch } from "@/services/chat-surface-scope"
import { loadServicePromptSnapshot, type ServicePromptSnapshot } from "@/services/service-prompts"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { servicePromptPrincipalMatches, servicePromptSingleUserApiKeyScopeMatches, servicePromptTargetsMatch } from "@/services/tldw/service-prompt-scope-error"

/** One verified owner and cancellation lifetime for the active QA workspace. */
export function useKnowledgeQAAuthority() {
  const { config, authorityLoading } = useCanonicalConnectionConfig()
  const connected = useConnectionStore((store) => store.state.isConnected && store.state.mode !== "demo")
  const boundary = useRef({ config, ready: false, generation: 0 })
  const ready = connected && !authorityLoading && Boolean(config)
  if (boundary.current.ready !== ready || !connectionAuthoritiesMatch(boundary.current.config, config)) {
    boundary.current = { config, ready, generation: boundary.current.generation + 1 }
  }
  const generation = boundary.current.generation
  const [, rerender] = useState(0)
  const [resolved, setResolved] = useState<{ generation: number; snapshot: ServicePromptSnapshot } | null>(null)

  useEffect(() => {
    if (!ready) return
    const controller = new AbortController()
    let snapshot: ServicePromptSnapshot | null = null
    const invalidate = () => {
      if (boundary.current.generation !== generation) return
      boundary.current.generation += 1
      controller.abort()
      setResolved(null)
      rerender((value) => value + 1)
    }
    void loadServicePromptSnapshot([], { signal: controller.signal }).then((value) => {
      if (controller.signal.aborted || boundary.current.generation !== generation || value.scopeSignal.aborted) {
        value.release()
        return
      }
      snapshot = value
      value.scopeInvalidatedSignal.addEventListener("abort", invalidate, { once: true })
      setResolved({ generation, snapshot: value })
    }).catch(() => {
      if (!controller.signal.aborted && boundary.current.generation === generation) setResolved(null)
    })
    return () => {
      controller.abort()
      snapshot?.scopeInvalidatedSignal.removeEventListener("abort", invalidate)
      snapshot?.release()
    }
  }, [generation, ready])

  useEffect(() => {
    let disposed = false
    let checkGeneration = 0
    const invalidate = () => {
      checkGeneration += 1
      boundary.current.generation += 1
      setResolved(null)
      rerender((value) => value + 1)
    }
    const revalidateExpiryHint = async () => {
      const currentCheck = ++checkGeneration
      if (resolved?.generation !== generation) return
      const owner = resolved.snapshot
      const isCurrentCheck = () => !disposed && currentCheck === checkGeneration &&
        boundary.current.generation === generation && !owner.scopeSignal.aborted
      try {
        // An expiry hint can arrive after a successful rotation or newer login.
        // Check effective credentials, including immutable invalidation markers.
        const current = await tldwClient.ensureConfigForRequest(true)
        if (!isCurrentCheck()) return
        const expected = owner.requestScope
        const cookieSession = current.authSource === "cookie-session"
        if (!servicePromptTargetsMatch(current, expected.config) ||
          (!cookieSession && (
            (current.authMode === "multi-user" && !current.accessToken) ||
            !servicePromptPrincipalMatches(current, expected.userId) ||
            !servicePromptSingleUserApiKeyScopeMatches(current, expected.config.expectedSingleUserApiKeyScope)
          ))) invalidate()
      } catch {
        if (isCurrentCheck()) invalidate()
      }
    }
    const configChanged = (event: Event) => {
      const detail = (event as CustomEvent<{ authorityChanged?: boolean; refreshSessionInvalidated?: boolean }>).detail
      if (detail?.refreshSessionInvalidated) void revalidateExpiryHint()
      else if (detail?.authorityChanged !== false) invalidate()
      else checkGeneration += 1
    }
    window.addEventListener("tldw:auth-principal-changed", invalidate)
    window.addEventListener("tldw:config-updated", configChanged)
    return () => {
      disposed = true
      checkGeneration += 1
      window.removeEventListener("tldw:auth-principal-changed", invalidate)
      window.removeEventListener("tldw:config-updated", configChanged)
    }
  }, [config, generation, resolved])

  const snapshot = ready && resolved?.generation === generation && !resolved.snapshot.scopeSignal.aborted
    ? resolved.snapshot : null
  return useMemo(() => ({
    snapshot,
    key: `${generation}:${snapshot ? "verified" : "unverified"}`,
    isCurrent: () => boundary.current.generation === generation && (!snapshot || !snapshot.scopeSignal.aborted),
  }), [generation, snapshot])
}
