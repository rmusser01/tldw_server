import type { RecipePersistenceOwnerView } from "@/services/recipe-persistence-owner"
import { resolveRecipePersistenceOwnerView } from "@/services/recipe-persistence-uncertainty"
import { useCallback, useLayoutEffect, useRef, useState } from "react"

/** Recipe ownership is fresh per open and never inferred by the UI. */
export function useRecipePersistenceOwner(enabled: boolean): {
  owner: RecipePersistenceOwnerView | null
  loading: boolean
  refresh: () => Promise<RecipePersistenceOwnerView | null>
} {
  const [owner, setOwner] = useState<RecipePersistenceOwnerView | null>(null)
  const [loading, setLoading] = useState(enabled)
  const generation = useRef(0)
  const active = useRef(enabled)
  active.current = enabled

  const refresh = useCallback(async () => {
    if (!active.current) return null
    const current = ++generation.current
    setOwner(null)
    setLoading(true)
    let next: RecipePersistenceOwnerView | null = null
    try {
      next = await resolveRecipePersistenceOwnerView()
    } catch {
      // An unavailable authority must never restore the prior owner's access.
    }
    if (!active.current || current !== generation.current) return null
    // A new object also identifies this open when the two opaque fields match.
    const view = next
      ? {
          ownerId: next.ownerId,
          authorizationRevision: next.authorizationRevision
        }
      : null
    setOwner(view)
    setLoading(false)
    return view
  }, [])

  useLayoutEffect(() => {
    if (enabled) void refresh()
    else {
      setOwner(null)
      setLoading(false)
    }
    return () => {
      generation.current += 1
    }
  }, [enabled, refresh])

  return { owner: enabled ? owner : null, loading: enabled && loading, refresh }
}
