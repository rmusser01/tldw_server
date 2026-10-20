import React from 'react'

/** The account half of the editor's `[authorityScope, noteId]` request key. */
const authorityOf = (requestKey: string | undefined): string | null => {
  if (!requestKey) return null
  try {
    const parsed = JSON.parse(requestKey)
    return Array.isArray(parsed) && parsed[0] != null ? String(parsed[0]) : null
  } catch {
    return null
  }
}

/**
 * Keeps the on-demand Connections section open across note switches in the
 * same account (NE-10). Connections load only on request, so the section
 * used to fold up on every switch. If the user left it open, the next note
 * opens it too and requests that note's connections. The choice lives for
 * this page visit only and is forgotten when the account changes, so
 * coming back to an account asks again.
 */
export const useConnectionsDisclosure = (
  requestKey: string | undefined,
  onRequestNeighbors: (() => void) | undefined
) => {
  const authority = authorityOf(requestKey)
  const rememberedRef = React.useRef<{ authority: string | null; open: boolean } | null>(null)
  const onRequestRef = React.useRef(onRequestNeighbors)
  onRequestRef.current = onRequestNeighbors

  const remembered = rememberedRef.current
  const initiallyOpen =
    Boolean(onRequestNeighbors) &&
    authority != null &&
    remembered?.open === true &&
    remembered.authority === authority

  React.useEffect(() => {
    // Another account forgets the choice, so coming back asks again.
    if (authority != null && rememberedRef.current && rememberedRef.current.authority !== authority) {
      rememberedRef.current = null
    }
    if (initiallyOpen) onRequestRef.current?.()
    // Request once per note the section reopens for.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [requestKey])

  const onOpenChange = React.useCallback(
    (open: boolean) => {
      rememberedRef.current = { authority, open }
      if (open) onRequestRef.current?.()
    },
    [authority]
  )

  return { initiallyOpen, onOpenChange }
}
