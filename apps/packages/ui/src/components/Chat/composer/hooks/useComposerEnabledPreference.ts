import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"

export const COMPOSER_ENABLED_PREFERENCE_KEY = "tldw:nextgenComposerEnabled"

/**
 * Catalog key registered in `Config_Files/user_profile_catalog.yaml` —
 * `PATCH /api/v1/users/me/profile` accepts
 * `{ updates: [{ key, value }] }` and `GET /me/profile` returns
 * `{ preferences: { [key]: value } }`. Parallel to the variant key.
 */
export const COMPOSER_ENABLED_PROFILE_KEY = "preferences.ui.composer_enabled"

/**
 * Hook for the user's "enable new composer" toggle — controls whether
 * `<ChatComposer variant=...>` mounts on /chat and the extension
 * sidepanel.
 *
 * Storage layers (in order of authority on initial render):
 *   1. **`localStorage`** keyed by `tldw:nextgenComposerEnabled` — read
 *      synchronously so the first render's flag check is correct
 *      without a flicker of the wrong composer.
 *   2. **Server profile** via `GET /api/v1/users/me/profile` — fetched
 *      asynchronously on mount; if it returns a different value, state
 *      updates and `localStorage` is overwritten so the user's choice
 *      from another device wins.
 *   3. **Storage event** — live cross-tab sync.
 *
 * `setEnabled`:
 *   - Updates state synchronously.
 *   - Writes to `localStorage`.
 *   - Fires `PATCH /me/profile` with the new value (best-effort).
 */

const isBool = (value: unknown): value is boolean => typeof value === "boolean"

const readStored = (): boolean => {
  if (typeof window === "undefined") return false
  try {
    return (
      window.localStorage.getItem(COMPOSER_ENABLED_PREFERENCE_KEY) === "1"
    )
  } catch {
    return false
  }
}

export interface UseComposerEnabledPreferenceOptions {
  /** Skip server fetch + PATCH. Useful for tests. Default false. */
  disableServerSync?: boolean
}

export type UseComposerEnabledPreferenceResult = [
  boolean,
  (next: boolean) => void,
]

export function useComposerEnabledPreference(
  options: UseComposerEnabledPreferenceOptions = {}
): UseComposerEnabledPreferenceResult {
  const disableServerSync = options.disableServerSync ?? false

  const [enabled, setEnabledState] = React.useState<boolean>(readStored)
  const localChangeVersionRef = React.useRef(0)

  // Server hydrate on mount.
  React.useEffect(() => {
    if (disableServerSync) return
    let cancelled = false
    const hydrateVersion = localChangeVersionRef.current
    const hydrate = async () => {
      try {
        const profile = await tldwClient.getCurrentUserProfile({
          sections: "preferences",
        })
        if (cancelled) return
        if (localChangeVersionRef.current !== hydrateVersion) return
        const serverValue = profile?.preferences?.[COMPOSER_ENABLED_PROFILE_KEY]
        if (!isBool(serverValue)) return
        setEnabledState((current) =>
          current === serverValue ? current : serverValue
        )
        try {
          window.localStorage.setItem(
            COMPOSER_ENABLED_PREFERENCE_KEY,
            serverValue ? "1" : "0"
          )
        } catch {
          /* ignore */
        }
      } catch {
        // Offline / unauthenticated / single-user mode without profile API.
      }
    }
    void hydrate()
    return () => {
      cancelled = true
    }
  }, [disableServerSync])

  // Cross-tab live sync.
  React.useEffect(() => {
    if (typeof window === "undefined") return
    const handler = (event: StorageEvent) => {
      if (event.key !== COMPOSER_ENABLED_PREFERENCE_KEY) return
      const next = event.newValue === "1"
      localChangeVersionRef.current += 1
      setEnabledState((current) => (current === next ? current : next))
    }
    window.addEventListener("storage", handler)
    return () => window.removeEventListener("storage", handler)
  }, [])

  const setEnabled = React.useCallback(
    (next: boolean) => {
      localChangeVersionRef.current += 1
      setEnabledState(next)
      if (typeof window !== "undefined") {
        try {
          window.localStorage.setItem(
            COMPOSER_ENABLED_PREFERENCE_KEY,
            next ? "1" : "0"
          )
        } catch {
          /* ignore */
        }
      }
      if (disableServerSync) return
      void tldwClient
        .updateCurrentUserProfile({
          updates: [{ key: COMPOSER_ENABLED_PROFILE_KEY, value: next }],
        })
        .catch(() => {
          /* ignore */
        })
    },
    [disableServerSync]
  )

  return [enabled, setEnabled]
}
