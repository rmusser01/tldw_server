export type InputSelectionTarget = {
  kind: "input"
  element: HTMLInputElement | HTMLTextAreaElement
  originalValue: string
  start: number
  end: number
}

export type ContentEditableTarget = {
  kind: "contenteditable"
  range: Range
  root: HTMLElement
  selectedText: string
}

export type SelectionTarget = InputSelectionTarget | ContentEditableTarget

export type SelectionContext = Readonly<{
  selectionText: string
  anchorRange: Range | null
  target: Readonly<SelectionTarget> | null
}>

const ALLOWED_INPUT_TYPES = new Set([
  "text",
  "search",
  "url",
  "email",
  "tel",
  "password"
])

const isEditableInput = (el: Element | null): el is HTMLInputElement => {
  if (!el || !(el instanceof HTMLInputElement)) return false
  const type = (el.type || "text").toLowerCase()
  return ALLOWED_INPUT_TYPES.has(type)
}

const isEditableTextarea = (
  el: Element | null
): el is HTMLTextAreaElement => {
  return Boolean(el && el instanceof HTMLTextAreaElement)
}

export const detectSelectionTarget = (
  selection: Selection | null
): SelectionTarget | null => {
  const activeEl = document.activeElement
  if (isEditableTextarea(activeEl) || isEditableInput(activeEl)) {
    const start = activeEl.selectionStart
    const end = activeEl.selectionEnd
    if (
      typeof start === "number" &&
      typeof end === "number" &&
      start !== end
    ) {
      return {
        kind: "input",
        element: activeEl,
        originalValue: activeEl.value,
        start,
        end
      }
    }
  }

  if (!selection || selection.rangeCount === 0) return null
  const range = selection.getRangeAt(0)
  if (range.collapsed) return null

  const container = range.commonAncestorContainer
  const containerEl =
    container instanceof HTMLElement ? container : container.parentElement
  if (!containerEl) return null

  if (!containerEl.isContentEditable) return null
  const root =
    containerEl.closest<HTMLElement>("[contenteditable='true'],[contenteditable='']") ||
    containerEl

  return {
    kind: "contenteditable",
    range: range.cloneRange(),
    root,
    selectedText: range.toString()
  }
}

export const isSelectionTargetValid = (target: SelectionTarget | null) => {
  if (!target) return false
  if (target.kind === "input") {
    if (
      !target.element.isConnected ||
      target.element.readOnly ||
      target.element.disabled ||
      target.element.value !== target.originalValue ||
      (target.element.tagName === "INPUT" &&
        !ALLOWED_INPUT_TYPES.has(target.element.type.toLowerCase()))
    ) {
      return false
    }
    const length = target.element.value.length
    return (
      target.start >= 0 &&
      target.end <= length &&
      target.start < target.end
    )
  }
  return Boolean(
    target.root?.isConnected &&
    target.root.isContentEditable &&
    !target.range.collapsed &&
    target.root.contains(target.range.startContainer) &&
    target.root.contains(target.range.endContainer) &&
    target.range.toString() === target.selectedText
  )
}

export const captureSelectionContext = (
  fallbackText?: string
): SelectionContext | null => {
  const selection = window.getSelection()
  const selectionText = (selection?.toString() || fallbackText || "").trim()
  if (!selectionText) return null

  const target = detectSelectionTarget(selection)
  return Object.freeze({
    selectionText,
    anchorRange:
      selection && selection.rangeCount > 0
        ? selection.getRangeAt(0).cloneRange()
        : null,
    target: target ? Object.freeze(target) : null
  })
}

export const replaceSelectionTarget = (
  target: SelectionTarget,
  replacement: string
): boolean => {
  try {
    if (target.kind === "input") {
      if (!isSelectionTargetValid(target)) return false
      target.element.focus()
      if (!isSelectionTargetValid(target)) return false
      target.element.setRangeText(replacement, target.start, target.end, "end")
      return true
    }

    if (!isSelectionTargetValid(target)) return false
    target.root.focus()
    if (!isSelectionTargetValid(target)) return false
    const range = target.range
    range.deleteContents()
    range.insertNode(document.createTextNode(replacement))
    range.collapse(false)
    return true
  } catch {
    return false
  }
}
