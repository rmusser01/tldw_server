import { useCallback, useEffect, useState } from 'react'
import type { RefObject } from 'react'

import { getContentSelectionFromDom } from './media-read-along-dom'
import type { ReadAlongSelection } from './types'

export interface UseContentSelectionActionsOptions {
  contentBodyRef: RefObject<HTMLElement | null>
  contentIdentityKey: string
  onApplyAnnotationSelection: (selectionText: string, location?: string) => void
}

export interface ContentSelectionActionState extends ReadAlongSelection {
  contentIdentityKey: string
  location?: string
}

const buildSelectionLocation = (selection: ReadAlongSelection): string | undefined => {
  if (!selection.startSegmentId) return undefined
  if (!selection.endSegmentId || selection.endSegmentId === selection.startSegmentId) {
    return `read-along-segment:${selection.startSegmentId}`
  }
  return `read-along-segment:${selection.startSegmentId}..${selection.endSegmentId}`
}

const toActionState = (
  selection: ReadAlongSelection,
  contentIdentityKey: string
): ContentSelectionActionState => ({
  ...selection,
  contentIdentityKey,
  location: buildSelectionLocation(selection)
})

export const useContentSelectionActions = ({
  contentBodyRef,
  contentIdentityKey,
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

    setSelectionActionState(toActionState(selection, contentIdentityKey))
  }, [clearSelectionActions, contentBodyRef, contentIdentityKey])

  const applyAnnotationSelection = useCallback(() => {
    if (!selectionActionState) return
    if (selectionActionState.contentIdentityKey !== contentIdentityKey) {
      clearSelectionActions()
      return
    }

    const currentSelection = getContentSelectionFromDom(contentBodyRef.current)
    if (!currentSelection) {
      clearSelectionActions()
      return
    }

    const currentActionState = toActionState(currentSelection, contentIdentityKey)
    onApplyAnnotationSelection(currentActionState.selectedText, currentActionState.location)
    clearSelectionActions()
  }, [
    clearSelectionActions,
    contentBodyRef,
    contentIdentityKey,
    onApplyAnnotationSelection,
    selectionActionState
  ])

  useEffect(() => {
    if (!selectionActionState || typeof document === 'undefined') return

    const handleSelectionChange = () => {
      const selection = getContentSelectionFromDom(contentBodyRef.current)
      if (!selection) {
        clearSelectionActions()
        return
      }
      setSelectionActionState(toActionState(selection, contentIdentityKey))
    }

    document.addEventListener('selectionchange', handleSelectionChange)
    return () => {
      document.removeEventListener('selectionchange', handleSelectionChange)
    }
  }, [clearSelectionActions, contentBodyRef, contentIdentityKey, selectionActionState])

  useEffect(() => {
    if (
      selectionActionState &&
      selectionActionState.contentIdentityKey !== contentIdentityKey
    ) {
      clearSelectionActions()
    }
  }, [clearSelectionActions, contentIdentityKey, selectionActionState])

  const currentSelectionActionState =
    selectionActionState?.contentIdentityKey === contentIdentityKey
      ? selectionActionState
      : null

  return {
    selectionActionState: currentSelectionActionState,
    handleContentSelectionEvent,
    clearSelectionActions,
    applyAnnotationSelection
  }
}
