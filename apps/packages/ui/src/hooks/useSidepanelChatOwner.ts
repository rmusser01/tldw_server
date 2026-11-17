import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import {
  type ServicePromptSnapshot,
  loadServicePromptSnapshot,
  isServicePromptScopeUnresolvedError
} from "@/services/service-prompts"
import { isRequestConfigScopeChangedError } from "@/services/tldw/service-prompt-scope-error"
import { useSidepanelChatTabsStore } from "@/store/sidepanel-chat-tabs"
import React from "react"

export type SidepanelChatOwner = {
  snapshot: ServicePromptSnapshot
  ownerKey: string
  revision: number
  isCurrent: () => boolean
}

/** Keep the route and its closures bound to the principal verified at mount. */
export const useSidepanelChatOwner = () => {
  const revision = useSidepanelChatTabsStore((state) => state.revision)
  const [owner, setOwner] = React.useState<SidepanelChatOwner | null>(null)
  const [verificationState, setVerificationState] = React.useState<"checking" | "retrying" | "unavailable" | "ready">("checking")
  const retryRef = React.useRef<(() => void) | null>(null)
  const retryVerification = React.useCallback(() => retryRef.current?.(), [])
  React.useEffect(() => {
    let mounted = true
    let generation = 0
    let current: SidepanelChatOwner | null = null
    const controller = new AbortController()
    const resolve = async () => {
      if (!mounted || useSidepanelChatTabsStore.getState().revision !== revision) return
      const check = ++generation
      setVerificationState(previous => previous === "unavailable" || previous === "retrying" ? "retrying" : "checking")
      let snapshot: ServicePromptSnapshot | undefined
      try {
        snapshot = await loadServicePromptSnapshot([], {
          signal: controller.signal
        })
        if (
          !mounted ||
          check !== generation ||
          snapshot.scopeSignal.aborted ||
          snapshot.scopeInvalidatedSignal.aborted ||
          useSidepanelChatTabsStore.getState().revision !== revision
        ) {
          snapshot.release()
          return
        }
        const ownerKey = serverChatMirrorOwnerKey(snapshot)
        const previousOwnerKey = useSidepanelChatTabsStore.getState().ownerKey
        if (previousOwnerKey && previousOwnerKey !== ownerKey) {
          // A cookie session can change outside this window. Announce the
          // verified boundary so transcript, queue and Prompt stores revoke
          // together before the replacement route takes its initial snapshot.
          snapshot.release()
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
          return
        }
        if (current?.ownerKey === ownerKey && current.isCurrent()) {
          setVerificationState("ready")
          snapshot.release()
          return
        }
        current?.snapshot.release()
        if (
          !useSidepanelChatTabsStore.getState().bindOwner(ownerKey, revision)
        ) {
          snapshot.release()
          return
        }
        const lease = snapshot
        const next: SidepanelChatOwner = {
          snapshot: lease,
          ownerKey,
          revision,
          isCurrent: () =>
            mounted &&
            current === next &&
            !lease.scopeSignal.aborted &&
            !lease.scopeInvalidatedSignal.aborted &&
            useSidepanelChatTabsStore.getState().revision === revision &&
            useSidepanelChatTabsStore.getState().ownerKey === ownerKey
        }
        current = next
        setOwner(next)
        setVerificationState("ready")
      } catch (error) {
        snapshot?.release()
        if (!mounted || check !== generation || controller.signal.aborted ||
          useSidepanelChatTabsStore.getState().revision !== revision) return
        setVerificationState("unavailable")
        const authorityLost = isServicePromptScopeUnresolvedError(error) ||
          isRequestConfigScopeChangedError(error) ||
          (error && typeof error === "object" && (error as { status?: unknown }).status === 401)
        if (authorityLost && useSidepanelChatTabsStore.getState().ownerKey) {
          current?.snapshot.release()
          current = null
          setOwner(null)
          window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed"))
        }
        // Temporary I/O failure retains a valid lease and all durable records.
      }
    }
    retryRef.current = resolve
    void resolve()
    window.addEventListener("focus", resolve)
    window.addEventListener("pageshow", resolve)
    return () => {
      mounted = false
      generation++
      controller.abort()
      if (retryRef.current === resolve) retryRef.current = null
      current?.snapshot.release()
      window.removeEventListener("focus", resolve)
      window.removeEventListener("pageshow", resolve)
    }
  }, [revision])
  return { owner: owner?.isCurrent() ? owner : null, verificationState, retryVerification }
}
