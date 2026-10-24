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

/** The user's last open/closed choice for Connections, and for which account. */
export type ConnectionsMemory = {
  current: { authority: string | null; open: boolean } | null
}

/**
 * Page-level memory for useConnectionsDisclosure. The Notes page owns it so
 * the choice outlives the editor pane, which unmounts while the Graph view
 * is showing.
 */
export const useConnectionsMemory = (): ConnectionsMemory =>
  React.useRef<ConnectionsMemory['current']>(null)

/**
 * Keeps the on-demand Connections section open across note switches in the
 * same account (NE-10). Connections load only on request, so the section
 * used to fold up on every switch. If the user left it open, the next note
 * opens it too and requests that note's connections. With `memory` from
 * useConnectionsMemory the choice lasts while the Notes page is open
 * (including visits to the Graph view); without it, while this pane is
 * mounted. It is forgotten when the account changes, so coming back to an
 * account asks again.
 */
export const useConnectionsDisclosure = (
  requestKey: string | undefined,
  onRequestNeighbors: (() => void) | undefined,
  memory?: ConnectionsMemory
) => {
  const authority = authorityOf(requestKey)
  const ownMemory = React.useRef<ConnectionsMemory['current']>(null)
  const rememberedRef = memory ?? ownMemory
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
