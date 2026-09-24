import { isEditableTarget } from "@/utils/editable-target"

export type PlaygroundMessageShortcutAction =
  | "variant_prev"
  | "variant_next"
  | "new_branch"
  | "regenerate"

type ShortcutEvent = {
  altKey?: boolean
  shiftKey?: boolean
  ctrlKey?: boolean
  metaKey?: boolean
  repeat?: boolean
  key?: string
  target?: EventTarget | null
}

export const resolvePlaygroundMessageShortcutAction = (
  event: ShortcutEvent
): PlaygroundMessageShortcutAction | null => {
  if (event.repeat) return null
  if (!event.altKey || !event.shiftKey) return null
  if (event.ctrlKey || event.metaKey) return null
  if (isEditableTarget(event.target)) return null

  const key = String(event.key || "").toLowerCase()
  if (key === "arrowleft") return "variant_prev"
  if (key === "arrowright") return "variant_next"
  if (key === "b") return "new_branch"
  if (key === "r") return "regenerate"
  return null
}
