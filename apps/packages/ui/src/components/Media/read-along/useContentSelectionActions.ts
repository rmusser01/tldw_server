import { useCallback, useState } from 'react'
import type { RefObject } from 'react'

import { getContentSelectionFromDom } from './media-read-along-dom'
import type { ReadAlongSelection } from './types'

export interface UseContentSelectionActionsOptions {
  contentBodyRef: RefObject<HTMLElement | null>
  onApplyAnnotationSelection: (selectionText: string, location?: string) => void
}

export interface ContentSelectionActionState extends ReadAlongSelection {
  location?: string
}

const buildSelectionLocation = (selection: ReadAlongSelection): string | undefined => {
  if (!selection.startSegmentId) return undefined
  if (!selection.endSegmentId || selection.endSegmentId === selection.startSegmentId) {
    return `read-along-segment:${selection.startSegmentId}`
  }
  return `read-along-segment:${selection.startSegmentId}..${selection.endSegmentId}`
}

export const useContentSelectionActions = ({
  contentBodyRef,
  onApplyAnnotationSelection
}: UseContentSelectionActionsOptions) => {
  const [selectionActionState, setSelectionActionState] =
    useState<ContentSelectionActionState | null>(null)

  const clearSelectionActions = useCallback(() => {
    setSelectionActionState(null)
  }, [])

  const handleContentSelectionEvent = useCallback(() => {
    const selection = getContentSelectionFromDom(contentBodyRef.current)
    if (!selection) {
      clearSelectionActions()
      return
    }

    setSelectionActionState({
      ...selection,
      location: buildSelectionLocation(selection)
    })
  }, [clearSelectionActions, contentBodyRef])

  const applyAnnotationSelection = useCallback(() => {
    if (!selectionActionState) return
    onApplyAnnotationSelection(
      selectionActionState.selectedText,
      selectionActionState.location
    )
    clearSelectionActions()
  }, [clearSelectionActions, onApplyAnnotationSelection, selectionActionState])

  return {
    selectionActionState,
    handleContentSelectionEvent,
    clearSelectionActions,
    applyAnnotationSelection
  }
}
