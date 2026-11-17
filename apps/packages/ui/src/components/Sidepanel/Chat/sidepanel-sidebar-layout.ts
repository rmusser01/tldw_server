/**
 * Layout rules for the side panel's chat list (XS-04).
 *
 * Pro mode docks a 288px chat list beside the conversation. Below about
 * 640px that dock leaves the chat too narrow to read, so narrower panels use
 * the overlay drawer instead. When the user collapses the docked list, that
 * choice is remembered for the next time the panel opens.
 */

/** Narrowest panel, in CSS pixels, that keeps the chat list docked. */
export const SIDEPANEL_DOCKED_SIDEBAR_MIN_WIDTH_PX = 640

/** Matches panels too narrow to dock the chat list. */
export const SIDEPANEL_UNDOCKED_SIDEBAR_MEDIA_QUERY = `(max-width: ${
  SIDEPANEL_DOCKED_SIDEBAR_MIN_WIDTH_PX - 1
}px)`

/** Per-viewer layout preference; never synced or shared. */
export const PRO_SIDEBAR_COLLAPSED_STORAGE_KEY =
  "tldw:sidepanel:proSidebarCollapsed"

const getLocalStorage = (): Storage | null => {
  try {
    return typeof window === "undefined" ? null : window.localStorage
  } catch {
    return null
  }
}

/** True when the user last collapsed the docked Pro chat list. */
export const readProSidebarCollapsed = (): boolean => {
  try {
    return getLocalStorage()?.getItem(PRO_SIDEBAR_COLLAPSED_STORAGE_KEY) === "true"
  } catch {
    return false
  }
}

/** Remembers whether the docked Pro chat list is collapsed. */
export const writeProSidebarCollapsed = (collapsed: boolean): void => {
  try {
    const storage = getLocalStorage()
    if (!storage) return
    if (collapsed) storage.setItem(PRO_SIDEBAR_COLLAPSED_STORAGE_KEY, "true")
    else storage.removeItem(PRO_SIDEBAR_COLLAPSED_STORAGE_KEY)
  } catch {
    // Storage can be blocked (private windows, previews); the layout still works.
  }
}
