import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { tldwClient } from "@/services/tldw/TldwApiClient"

/**
 * One profile read for the chat composer's saved preferences (XP-16).
 *
 * The composer mounts useComposerEnabledPreference and
 * useComposerVariantPreference together, and each read
 * GET /api/v1/users/me/profile?sections=preferences on mount, so every chat
 * load read the (slow) profile twice. A read that starts while another is
 * still pending now shares it.
 *
 * Nothing is kept once the read settles. A pending read is not shared after
 * the account or server changes, or after a composer preference is changed,
 * so a later composer never sees an older answer.
 */
type ProfileWithPreferences = {
  preferences?: Record<string, unknown> | null
} | null

let pendingRead: Promise<ProfileWithPreferences> | null = null
let stopWatchingAccount: (() => void) | null = null

/** Stop sharing the pending read; the next caller starts a fresh one. */
export const forgetPendingComposerProfileRead = (): void => {
  pendingRead = null
}

const watchAccountOnce = () => {
  if (stopWatchingAccount) return
  stopWatchingAccount = watchChatAccountChanges((invalidated) => {
    if (invalidated) forgetPendingComposerProfileRead()
  })
}

export const readComposerProfilePreferences =
  (): Promise<ProfileWithPreferences> => {
    watchAccountOnce()
    if (pendingRead) return pendingRead
    let request: Promise<ProfileWithPreferences>
    try {
      request = Promise.resolve(
        tldwClient.getCurrentUserProfile({ sections: "preferences" })
      )
    } catch (error) {
      request = Promise.reject(error)
    }
    const read = request.finally(() => {
      if (pendingRead === read) pendingRead = null
    })
    pendingRead = read
    return read
  }

const hot = (
  import.meta as { hot?: { dispose: (callback: () => void) => void } }
).hot
hot?.dispose(() => {
  stopWatchingAccount?.()
  stopWatchingAccount = null
  pendingRead = null
})
