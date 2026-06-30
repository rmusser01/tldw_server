import React from "react"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import type { ChatComposerVariant } from "../types"

export const COMPOSER_VARIANT_PREFERENCE_KEY = "tldw:composerVariant"

/**
 * Catalog key registered in `Config_Files/user_profile_catalog.yaml` —
 * `PATCH /api/v1/users/me/profile` accepts `{ updates: [{ key, value }] }`
 * and `GET /me/profile` returns `{ preferences: { [key]: value } }`.
 */
export const COMPOSER_VARIANT_PROFILE_KEY = "preferences.ui.composer_variant"

const VALID_VARIANTS: ChatComposerVariant[] = ["v1", "v3", "v5"]

const isValidVariant = (
  value: unknown
): value is ChatComposerVariant =>
  typeof value === "string" && (VALID_VARIANTS as string[]).includes(value)

/**
 * Hook for the user's chat-composer variant preference — drives which of
 * V1 / V3 / V5 renders under `<ChatComposer />`.
 *
 * Storage layers (in order of authority on initial render):
 *   1. **`localStorage`** keyed by `tldw:composerVariant` — read
 *      synchronously so the first render shows the right variant
 *      immediately (no flicker).
 *   2. **Server profile** via `GET /api/v1/users/me/profile` — fetched
 *      asynchronously on mount; if it returns a different value, state
 *      updates and `localStorage` is overwritten so the user's choice
 *      from another device wins.
 *
 * `setVariant`:
 *   - Updates state synchronously (optimistic).
 *   - Writes to `localStorage` (offline + cross-tab fallback).
 *   - Fires `PATCH /me/profile` with the new value (best-effort —
 *     errors are swallowed; local choice persists either way).
 *
 * Defaults to `v1` for new users. Unknown server / stored values fall
 * back to the default. `setVariant` ignores unknown values at runtime.
 */

export interface UseComposerVariantPreferenceOptions {
  /** Fallback variant when nothing is stored or stored value is invalid. */
  defaultVariant?: ChatComposerVariant
  /**
   * Skip the async server fetch. Useful for tests that don't want to
   * mock the API client. Default false.
   */
  disableServerSync?: boolean
}

export type UseComposerVariantPreferenceResult = [
  ChatComposerVariant,
  (next: ChatComposerVariant) => void,
]

export function useComposerVariantPreference(
  options: UseComposerVariantPreferenceOptions = {}
): UseComposerVariantPreferenceResult {
  const defaultVariant = options.defaultVariant ?? "v1"
  const disableServerSync = options.disableServerSync ?? false

  const readInitial = React.useCallback((): ChatComposerVariant => {
    if (typeof window === "undefined") return defaultVariant
    try {
      const stored = window.localStorage.getItem(
        COMPOSER_VARIANT_PREFERENCE_KEY
      )
      return isValidVariant(stored) ? stored : defaultVariant
    } catch {
      return defaultVariant
    }
  }, [defaultVariant])

  const [variant, setVariantState] =
    React.useState<ChatComposerVariant>(readInitial)
  const localChangeVersionRef = React.useRef(0)

  // Hydrate from server on mount. Only one fetch per hook instance.
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
        const serverValue = profile?.preferences?.[COMPOSER_VARIANT_PROFILE_KEY]
        if (!isValidVariant(serverValue)) return
        setVariantState((current) =>
          current === serverValue ? current : serverValue
        )
        try {
          window.localStorage.setItem(
            COMPOSER_VARIANT_PREFERENCE_KEY,
            serverValue
          )
        } catch {
          /* ignore */
        }
      } catch {
        // Server unreachable, unauthenticated, single-user mode without
        // profile API — localStorage value remains authoritative.
      }
    }
    void hydrate()
    return () => {
      cancelled = true
    }
  }, [disableServerSync])

  // Cross-tab live sync: when the user switches variant in another tab,
  // the `storage` event fires here. Update local state so all open tabs
  // reflect the change without needing a navigate/reload. (Server PATCH
  // also propagates eventually, but it requires a refetch on the other
  // tab — `storage` is instant.)
  React.useEffect(() => {
    if (typeof window === "undefined") return
    const handler = (event: StorageEvent) => {
      if (event.key !== COMPOSER_VARIANT_PREFERENCE_KEY) return
      const next = event.newValue
      if (!isValidVariant(next)) return
      localChangeVersionRef.current += 1
      setVariantState((current) => (current === next ? current : next))
    }
    window.addEventListener("storage", handler)
    return () => window.removeEventListener("storage", handler)
  }, [])

  const setVariant = React.useCallback(
    (next: ChatComposerVariant) => {
      if (!isValidVariant(next)) return
      localChangeVersionRef.current += 1
      setVariantState(next)
      if (typeof window !== "undefined") {
        try {
          window.localStorage.setItem(COMPOSER_VARIANT_PREFERENCE_KEY, next)
        } catch {
          // Safari private mode / quota — session-only persist OK.
        }
      }
      if (disableServerSync) return
      // Fire-and-forget server update. Errors are swallowed: the user's
      // local choice already applied; a failed sync just means the
      // preference doesn't propagate to other devices this time.
      void tldwClient
        .updateCurrentUserProfile({
          updates: [{ key: COMPOSER_VARIANT_PROFILE_KEY, value: next }],
        })
        .catch(() => {
          /* ignore */
        })
    },
    [disableServerSync]
  )

  return [variant, setVariant]
}
