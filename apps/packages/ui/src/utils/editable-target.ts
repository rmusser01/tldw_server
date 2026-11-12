const EDITABLE_SELECTOR = [
  "input",
  "textarea",
  "select",
  "[contenteditable]:not([contenteditable='false'])",
  "[role='textbox']",
  "[role='combobox']",
  "[role='searchbox']"
].join(", ")

/**
 * Whether a keyboard event on `target` means the user is typing, so global
 * single-key shortcuts ("?", "/", "j", ...) must not fire.
 *
 * Covers input, textarea, select, contenteditable regions (including their
 * descendants) and ARIA textbox/combobox/searchbox widgets. The ancestor
 * check matters because jsdom does not implement `isContentEditable`, and
 * rich editors often focus a child of the editable root.
 */
export const isEditableTarget = (
  target: EventTarget | null | undefined
): boolean => {
  if (typeof Element === "undefined" || !(target instanceof Element)) {
    return false
  }
  if (target instanceof HTMLElement && target.isContentEditable) return true
  return target.closest(EDITABLE_SELECTOR) !== null
}
