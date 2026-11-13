/**
 * Items of a `role="menu"`, in DOM order. Items marked aria-disabled stay
 * in the order, so assistive technology reads them and why they are off.
 */
const MENU_ITEM_SELECTOR = [
  "[role='menuitem']",
  "[role='menuitemcheckbox']",
  "[role='menuitemradio']"
]
  .map((role) => `${role}:not(:disabled)`)
  .join(", ")

/**
 * Arrow Up/Down (wrapping), Home and End for a menu of focusable menu items,
 * as the WAI-ARIA menu button pattern asks. Moves focus and returns true when
 * `key` is one of those keys; returns false for any other key.
 */
export const moveMenuFocus = (menu: HTMLElement, key: string): boolean => {
  const items = Array.from(
    menu.querySelectorAll<HTMLElement>(MENU_ITEM_SELECTOR)
  )
  if (items.length === 0) return false
  const current = items.indexOf(document.activeElement as HTMLElement)
  const last = items.length - 1
  let next: number
  switch (key) {
    case "ArrowDown":
      next = current < 0 || current === last ? 0 : current + 1
      break
    case "ArrowUp":
      next = current <= 0 ? last : current - 1
      break
    case "Home":
      next = 0
      break
    case "End":
      next = last
      break
    default:
      return false
  }
  items[next].focus()
  return true
}
