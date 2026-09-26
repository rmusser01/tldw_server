import React from "react"
import { sha256 } from "@noble/hashes/sha2.js"
import { bytesToHex, utf8ToBytes } from "@noble/hashes/utils.js"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { isServicePromptScopeUnresolvedError, resolveServicePromptScope } from "@/services/service-prompts"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"

type DraftOwner = { key: string }

/** Keep a composer and its pending storage work bound to a verified account. */
export const useChatDraftOwner = (clearComposer: () => void) => {
  const [owner, setOwner] = React.useState<DraftOwner | null>(null)
  const currentOwner = React.useRef<DraftOwner | null>(null)
  const clearRef = React.useRef(clearComposer)
  React.useLayoutEffect(() => { clearRef.current = clearComposer }, [clearComposer])

  React.useEffect(() => {
    let generation = 0
    let mounted = true
    const clear = () => {
      currentOwner.current = null
      setOwner(null)
      clearRef.current()
    }
    const resolve = async (invalidated = false) => {
      const check = ++generation
      if (invalidated) clear()
      try {
        const scope = await resolveServicePromptScope()
        if (!mounted || check !== generation) return
        // Include the full server identity and auth source; never put credentials in storage keys.
        const key = bytesToHex(sha256(utf8ToBytes(JSON.stringify([
          scope.config.serverUrl.replace(/\/+$/, ""), scope.config.authMode,
          scope.config.authSource || "manual", scope.config.orgId ?? null,
          scope.userId, scope.config.expectedSingleUserApiKeyScope ?? null
        ]))))
        if (currentOwner.current?.key === key) return
        if (currentOwner.current) clear()
        const nextOwner = { key }
        currentOwner.current = nextOwner
        setOwner(nextOwner)
      } catch (error) {
        // A temporary recheck failure does not revoke the verified draft owner.
        // Explicit account/credential changes already clear before the request.
        const authorityLost = isServicePromptScopeUnresolvedError(error) ||
          isRequestConfigScopeChangedError(error) ||
          (error && typeof error === "object" && (error as { status?: unknown }).status === 401)
        if (mounted && check === generation && currentOwner.current && authorityLost) clear()
      }
    }
    const stop = watchChatAccountChanges((invalidated) => { void resolve(invalidated) })
    const revalidate = () => { void resolve() }
    window.addEventListener("focus", revalidate)
    window.addEventListener("pageshow", revalidate)
    void resolve()
    return () => {
      mounted = false
      generation++
      currentOwner.current = null
      stop()
      window.removeEventListener("focus", revalidate)
      window.removeEventListener("pageshow", revalidate)
    }
  }, [])

  const isCurrent = React.useCallback(
    () => owner !== null && currentOwner.current === owner,
    [owner]
  )
  return { ownerKey: owner?.key ?? null, isCurrent }
}
