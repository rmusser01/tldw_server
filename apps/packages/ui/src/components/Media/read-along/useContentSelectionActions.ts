import { useCallback, useEffect, useState } from 'react'
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

const toActionState = (selection: ReadAlongSelection): ContentSelectionActionState => ({
  ...selection,
  location: buildSelectionLocation(selection)
})

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

    setSelectionActionState(toActionState(selection))
  }, [clearSelectionActions, contentBodyRef])

  const applyAnnotationSelection = useCallback(() => {
    if (!selectionActionState) return

    const currentSelection = getContentSelectionFromDom(contentBodyRef.current)
    if (!currentSelection) {
      clearSelectionActions()
      return
    }

    const currentActionState = toActionState(currentSelection)
    onApplyAnnotationSelection(currentActionState.selectedText, currentActionState.location)
    clearSelectionActions()
  }, [clearSelectionActions, contentBodyRef, onApplyAnnotationSelection, selectionActionState])

  useEffect(() => {
    if (!selectionActionState || typeof document === 'undefined') return

    const handleSelectionChange = () => {
      const selection = getContentSelectionFromDom(contentBodyRef.current)
      if (!selection) {
        clearSelectionActions()
        return
      }
      setSelectionActionState(toActionState(selection))
    }

    document.addEventListener('selectionchange', handleSelectionChange)
    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange)
    }
  }, [clearSelectionActions, contentBodyRef, selectionActionState])

  return {
    selectionActionState,
    handleContentSelectionEvent,
    clearSelectionActions,
    applyAnnotationSelection
  }
}
