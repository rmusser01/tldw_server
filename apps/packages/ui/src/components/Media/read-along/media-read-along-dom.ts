import type { ReadAlongSelection } from './types'

const isNodeInside = (container: HTMLElement, node: Node | null): boolean => {
  if (!node) return false
  if (node === container) return true
  return container.contains(node)
}

const makeEmptyDomRect = (): DOMRect => {
  if (typeof DOMRect !== 'undefined') {
    return new DOMRect()
  }
  return {
    x: 0,
    y: 0,
    width: 0,
    height: 0,
    top: 0,
    right: 0,
    bottom: 0,
    left: 0,
    toJSON: () => ({})
  } as DOMRect
}

export const isRangeInsideContentBody = (
  range: Range | null | undefined,
  contentBody: HTMLElement | null | undefined
): boolean => {
  if (!range || !contentBody) return false
  return (
    isNodeInside(contentBody, range.commonAncestorContainer) &&
    isNodeInside(contentBody, range.startContainer) &&
    isNodeInside(contentBody, range.endContainer)
  )
}

export const getRangeAnchorRect = (range: Range | null | undefined): DOMRect => {
  if (!range) return makeEmptyDomRect()

  if (typeof range.getBoundingClientRect === 'function') {
    const rect = range.getBoundingClientRect()
    if (rect.width > 0 || rect.height > 0) {
      return rect
    }
  }

  if (typeof range.getClientRects === 'function') {
    const firstRect = range.getClientRects()[0]
    if (firstRect) {
      return firstRect as DOMRect
    }
  }

  return makeEmptyDomRect()
}

export const findNearestReadAlongSegmentId = (
  node: Node | null | undefined,
  contentBody?: HTMLElement | null
): string | undefined => {
  let current: Node | null = node ?? null
  while (current && current !== contentBody) {
    if (typeof HTMLElement !== 'undefined' && current instanceof HTMLElement) {
      const segmentId = current.dataset.readAlongSegmentId
      if (segmentId) return segmentId
    }
    current = current.parentNode
  }

  if (contentBody?.dataset.readAlongSegmentId) {
    return contentBody.dataset.readAlongSegmentId
  }

  return undefined
}

export const getContentSelectionFromDom = (
  contentBody: HTMLElement | null | undefined
): ReadAlongSelection | null => {
  if (
    !contentBody ||
    typeof window === 'undefined' ||
    typeof window.getSelection !== 'function'
  ) {
    return null
  }

  const selection = window.getSelection()
  if (!selection || selection.rangeCount === 0 || selection.isCollapsed) {
    return null
  }

  const range = selection.getRangeAt(0)
  if (!isRangeInsideContentBody(range, contentBody)) {
    return null
  }

  const selectedText = selection.toString().trim()
  if (!selectedText) return null

  const startSegmentId = findNearestReadAlongSegmentId(range.startContainer, contentBody)
  const endSegmentId = findNearestReadAlongSegmentId(range.endContainer, contentBody)

  return {
    selectedText,
    anchorRect: getRangeAnchorRect(range),
    startSegmentId,
    endSegmentId,
    mappingConfidence: startSegmentId || endSegmentId ? 'exact' : 'text-only'
  }
}
