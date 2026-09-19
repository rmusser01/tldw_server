export type PlaygroundShortcutAction =
  | "toggle_artifacts"
  | "toggle_compare"
  | "toggle_modes"

type ShortcutEvent = {
  altKey?: boolean
  shiftKey?: boolean
  ctrlKey?: boolean
  metaKey?: boolean
  repeat?: boolean
  key?: string
  target?: EventTarget | null
}

export const isEditableTarget = (
  target: EventTarget | null | undefined
): boolean => {
  if (!target || !(target instanceof HTMLElement)) return false
  if (target.isContentEditable) return true
  const tagName = target.tagName.toLowerCase()
  return tagName === "input" || tagName === "textarea" || tagName === "select"
}

/**
 * Whether a keydown should open the playground shortcuts help panel.
 *
 * "?" is an ordinary typed character, so this must stay false while the caret
 * sits in the composer or any other editable target — otherwise every question
 * mark a user types is swallowed by `preventDefault()`. Modifier chords are
 * unreachable by typing and are deliberately not gated this way.
 */
export const shouldOpenShortcutsHelp = (event: ShortcutEvent): boolean => {
  if (event.altKey || event.ctrlKey || event.metaKey) return false
  if (!event.shiftKey) return false
  if (event.key !== "?") return false
  return !isEditableTarget(event.target)
}

export const resolvePlaygroundShortcutAction = (
  event: ShortcutEvent
): PlaygroundShortcutAction | null => {
  if (event.repeat) return null
  if (!event.altKey || !event.shiftKey) return null
  if (event.ctrlKey || event.metaKey) return null
  if (isEditableTarget(event.target)) return null

  const key = String(event.key || "").toLowerCase()
  if (key === "a") return "toggle_artifacts"
  if (key === "c") return "toggle_compare"
  if (key === "m") return "toggle_modes"
  return null
}
