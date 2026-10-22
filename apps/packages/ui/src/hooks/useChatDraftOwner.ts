import React from "react"
import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import { chatDraftOwnerKey } from "@/utils/chat-draft-owner"
export { chatDraftOwnerKey } from "@/utils/chat-draft-owner"
import { Storage } from "@plasmohq/storage"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { isServicePromptScopeUnresolvedError, resolveServicePromptScope } from "@/services/service-prompts"
import type { ServicePromptScope } from "@/services/service-prompts"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { REFRESH_ROTATION_KEY } from "@/services/tldw/single-user-credential"
import { safeStorageSerde } from "@/utils/safe-storage"

type DraftOwner = { key: string; serverScopeKey: string; requestScope: ServicePromptScope }

/** Keep a composer and its pending storage work bound to a verified account. */
export const useChatDraftOwner = (
  clearComposer: (source?: Event) => void,
  onAccountChange?: (invalidated: boolean, source?: Event) => void,
  onInitialOwner?: () => void
) => {
  const [owner, setOwner] = React.useState<DraftOwner | null>(null)
  const currentOwner = React.useRef<DraftOwner | null>(null)
  const everOwned = React.useRef(false)
  const hasEverOwned = React.useCallback(() => everOwned.current, [])
  const initialOwnerRef = React.useRef(onInitialOwner)
  React.useLayoutEffect(() => { initialOwnerRef.current = onInitialOwner }, [onInitialOwner])
  const accountChangeRef = React.useRef(onAccountChange)
  React.useLayoutEffect(() => { accountChangeRef.current = onAccountChange }, [onAccountChange])
  const [verificationState, setVerificationState] = React.useState<"pending" | "ready" | "error">("pending")
  const retryRef = React.useRef<(() => void) | null>(null)
  const retryVerification = React.useCallback(() => retryRef.current?.(), [])
  const clearRef = React.useRef(clearComposer)
  React.useLayoutEffect(() => { clearRef.current = clearComposer }, [clearComposer])

  React.useEffect(() => {
    let generation = 0
    let mounted = true
    const clear = (source?: Event) => {
      currentOwner.current = null
      setOwner(null)
      clearRef.current(source)
    }
    const resolve = async (invalidated = false, source?: Event) => {
      const check = ++generation
      if (invalidated) clear(source)
      if (!currentOwner.current) setVerificationState("pending")
      try {
        const scope = await resolveServicePromptScope()
        if (!mounted || check !== generation) return
        // Include the full server identity and auth source; never put credentials in storage keys.
        setVerificationState("ready")
        const key = chatDraftOwnerKey(scope)
        const serverScopeKey = serverChatMirrorOwnerKey({ requestScope: scope })
        if (currentOwner.current?.key === key && currentOwner.current.serverScopeKey === serverScopeKey) return
        if (currentOwner.current) clear()
        if (!everOwned.current) initialOwnerRef.current?.()
        everOwned.current = true
        const nextOwner = { key, serverScopeKey, requestScope: scope }
        currentOwner.current = nextOwner
        setOwner(nextOwner)
      } catch (error) {
        // A temporary recheck failure does not revoke the verified draft owner.
        // Explicit account/credential changes already clear before the request.
        const authorityLost = isServicePromptScopeUnresolvedError(error) ||
          isRequestConfigScopeChangedError(error) ||
          (error && typeof error === "object" && (error as { status?: unknown }).status === 401)
        if (mounted && check === generation) {
          if (currentOwner.current && authorityLost) clear()
          setVerificationState(currentOwner.current ? "ready" : "error")
        }
      }
    }
    const stop = watchChatAccountChanges((invalidated, _config, source) => {
      accountChangeRef.current?.(invalidated, source)
      void resolve(invalidated, source)
    })
    const rotationStorage = new Storage({ area: "local", serde: safeStorageSerde })
    const rotationWatch = {
      [REFRESH_ROTATION_KEY]: () => {
        if (mounted && !currentOwner.current) void resolve()
      }
    }
    void rotationStorage.watch(rotationWatch)
    const revalidate = () => { void resolve() }
    retryRef.current = revalidate
    window.addEventListener("focus", revalidate)
    window.addEventListener("pageshow", revalidate)
    void resolve()
    return () => {
      mounted = false
      generation++
      currentOwner.current = null
      retryRef.current = null
      stop()
      void rotationStorage.unwatch(rotationWatch)
      window.removeEventListener("focus", revalidate)
      window.removeEventListener("pageshow", revalidate)
    }
  }, [])

  const isCurrent = React.useCallback(
    () => owner !== null && currentOwner.current === owner,
    [owner]
  )
  return { hasEverOwned, ownerKey: owner?.key ?? null, serverScopeKey: owner?.serverScopeKey ?? null, requestScope: owner?.requestScope ?? null, isCurrent, verificationState, retryVerification }
}
