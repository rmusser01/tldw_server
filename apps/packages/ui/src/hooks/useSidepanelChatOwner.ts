import { serverChatMirrorOwnerKey } from "@/db/dexie/server-chat-mirror"
import {
  type ServicePromptSnapshot,
  loadServicePromptSnapshot
} from "@/services/service-prompts"
import { useSidepanelChatTabsStore } from "@/store/sidepanel-chat-tabs"
import React from "react"

export type SidepanelChatOwner = {
  snapshot: ServicePromptSnapshot
  ownerKey: string
  revision: number
  isCurrent: () => boolean
}

/** Keep the route and its closures bound to the principal verified at mount. */
export const useSidepanelChatOwner = (): SidepanelChatOwner | null => {
  const revision = useSidepanelChatTabsStore((state) => state.revision)
  const [owner, setOwner] = React.useState<SidepanelChatOwner | null>(null)
  React.useEffect(() => {
    let mounted = true
    let generation = 0
    let current: SidepanelChatOwner | null = null
    const controller = new AbortController()
    const resolve = async () => {
      const check = ++generation
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
      } catch {
        snapshot?.release()
        // No new owner can be established offline. Retain a still-valid lease
        // and all durable records when a same-owner recheck fails temporarily.
      }
    }
    void resolve()
    window.addEventListener("focus", resolve)
    window.addEventListener("pageshow", resolve)
    return () => {
      mounted = false
      generation++
      controller.abort()
      current?.snapshot.release()
      window.removeEventListener("focus", resolve)
      window.removeEventListener("pageshow", resolve)
    }
  }, [revision])
  return owner?.isCurrent() ? owner : null
}
