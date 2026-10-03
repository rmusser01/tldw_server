import { isEditableTarget } from "@/utils/editable-target"

const INPUT_TAGS = new Set(["input", "textarea", "select", "option"])
const INTERACTIVE_TAGS = new Set(["button", "a", "summary"])

const SHORTCUT_SUPPRESSION_SELECTORS = [
  "[contenteditable='true']",
  "[role='dialog']",
  "[aria-modal='true']",
  "[aria-haspopup='menu']",
  "[role='menu']",
  "[role='menuitem']",
  "[data-shortcut-scope='suppress']",
  ".ant-dropdown",
  ".ant-dropdown-menu",
  ".ant-select-dropdown",
  ".ant-picker-dropdown",
  ".ant-modal-wrap",
  ".ant-modal-root"
].join(",")

const INTERACTIVE_SHORTCUT_SELECTORS = [
  "button",
  "a[href]",
  "[role='link']",
  "summary"
].join(",")

const toElement = (target: EventTarget | null): HTMLElement | null => {
  if (!target) return null
  if (target instanceof HTMLElement) return target
  if (target instanceof Element) return target as HTMLElement
  if (target instanceof Node) {
    return target.parentElement as HTMLElement | null
  }
  return null
}

export function shouldHandleGlobalShortcut(target: EventTarget | null): boolean {
  const element = toElement(target)
  if (!element) return true

  const tag = element.tagName.toLowerCase()
  if (INPUT_TAGS.has(tag)) return false
  if (INTERACTIVE_TAGS.has(tag)) return false
  if (element.getAttribute("role") === "button") return false
  if (isEditableTarget(element)) return false
  if (element.closest(INTERACTIVE_SHORTCUT_SELECTORS)) return false
  if (element.closest(SHORTCUT_SUPPRESSION_SELECTORS)) return false

  return true
}
