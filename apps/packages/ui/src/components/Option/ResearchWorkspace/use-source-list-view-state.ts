import React from "react"
import {
  DEFAULT_SOURCE_LIST_VIEW_STATE,
  type SourceListViewState
} from "./SourcesPane/source-list-view"

export const useSourceListViewState = () => {
  const [sourceListViewState, setSourceListViewState] =
    React.useState<SourceListViewState>(DEFAULT_SOURCE_LIST_VIEW_STATE)

  const patchSourceListViewState = React.useCallback(
    (patch: Partial<SourceListViewState>) => {
      setSourceListViewState((current) => ({ ...current, ...patch }))
    },
    []
  )

  const resetAdvancedSourceFilters = React.useCallback(() => {
    setSourceListViewState((current) => ({
      ...DEFAULT_SOURCE_LIST_VIEW_STATE,
      expanded: current.expanded
    }))
  }, [])

  return {
    sourceListViewState,
    patchSourceListViewState,
    resetAdvancedSourceFilters
  }
}
