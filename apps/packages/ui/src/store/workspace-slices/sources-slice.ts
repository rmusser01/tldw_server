import type { WorkspaceSlice } from './types'
import type { WorkspaceState } from '../workspace'
import type {
  WorkspaceSource,
  WorkspaceSourceFolder
} from '@/types/workspace'

// TODO: These helpers need to be exported from workspace.ts
import {
  generateWorkspaceId,
  getUniqueSourceFolderName,
  createWorkspaceOrganizationStateIndex,
  getWorkspaceSourceStatus,
  reviveDateOrNull
} from '../workspace'

import {
  collectDescendantFolderIds,
  deriveEffectiveSelectedSourceIds
} from '@/store/workspace-organization'

// Extract SourcesActions from workspace.ts (action interface only)
type SourcesActions = Pick<
  WorkspaceState,
  | 'createSourceFolder'
  | 'renameSourceFolder'
  | 'moveSourceFolder'
  | 'deleteSourceFolder'
  | 'assignSourceToFolders'
  | 'removeSourceFromFolder'
  | 'toggleSourceFolderSelection'
  | 'setActiveFolder'
  | 'addSource'
  | 'addSources'
  | 'removeSource'
  | 'removeSources'
  | 'reorderSource'
  | 'toggleSourceSelection'
  | 'selectAllSources'
  | 'deselectAllSources'
  | 'setSelectedSourceIds'
  | 'setSourceSearchQuery'
  | 'setSourceStatusById'
  | 'setSourceStatusByMediaId'
  | 'focusSourceById'
  | 'focusSourceByMediaId'
  | 'clearSourceFocusTarget'
  | 'setSourcesLoading'
  | 'setSourcesError'
  | 'restoreSource'
  | 'getSelectedSources'
  | 'getSelectedMediaIds'
  | 'getEffectiveSelectedSources'
  | 'getEffectiveSelectedMediaIds'
>

export const createSourcesSlice: WorkspaceSlice<SourcesActions> = (set, get) => ({
  createSourceFolder: (name, parentFolderId = null) => {
    const state = get()
    if (parentFolderId && !state.sourceFolders.some((folder) => folder.id === parentFolderId)) {
      throw new Error(`Cannot create folder under missing parent "${parentFolderId}"`)
    }

    const folder: WorkspaceSourceFolder = {
      id: generateWorkspaceId(),
      workspaceId: state.workspaceId,
      name: getUniqueSourceFolderName(
        state.sourceFolders,
        name,
        parentFolderId
      ),
      parentFolderId,
      createdAt: new Date(),
      updatedAt: new Date()
    }

    set((current) => ({
      sourceFolders: [...current.sourceFolders, folder]
    }))
    return folder
  },

  renameSourceFolder: (folderId, name) =>
    set((state) => {
      const folder = state.sourceFolders.find((entry) => entry.id === folderId)
      if (!folder) {
        return state
      }

      return {
        sourceFolders: state.sourceFolders.map((entry) =>
          entry.id === folderId
            ? {
                ...entry,
                name: getUniqueSourceFolderName(
                  state.sourceFolders,
                  name,
                  entry.parentFolderId,
                  entry.id
                ),
                updatedAt: new Date()
              }
            : entry
        )
      }
    }),

  moveSourceFolder: (folderId, parentFolderId) =>
    set((state) => {
      const folder = state.sourceFolders.find((entry) => entry.id === folderId)
      if (!folder) {
        return state
      }

      if (parentFolderId === folderId) {
        throw new Error("A folder cannot be moved under itself.")
      }

      if (
        parentFolderId &&
        !state.sourceFolders.some((entry) => entry.id === parentFolderId)
      ) {
        throw new Error(`Cannot move folder under missing parent "${parentFolderId}".`)
      }

      const organizationIndex = createWorkspaceOrganizationStateIndex(state)
      const descendantFolderIds = new Set(
        collectDescendantFolderIds(organizationIndex, folderId)
      )
      if (parentFolderId && descendantFolderIds.has(parentFolderId)) {
        throw new Error("A folder cannot be moved under one of its descendants.")
      }

      return {
        sourceFolders: state.sourceFolders.map((entry) =>
          entry.id === folderId
            ? {
                ...entry,
                parentFolderId,
                name: getUniqueSourceFolderName(
                  state.sourceFolders,
                  entry.name,
                  parentFolderId,
                  entry.id
                ),
                updatedAt: new Date()
              }
            : entry
        )
      }
    }),

  deleteSourceFolder: (folderId) =>
    set((state) => {
      const folderToDelete = state.sourceFolders.find(
        (folder) => folder.id === folderId
      )
      if (!folderToDelete) {
        return state
      }

      const reparentedSourceFolders = state.sourceFolders
        .filter((folder) => folder.id !== folderId)
        .reduce<WorkspaceSourceFolder[]>((accumulator, folder) => {
          if (folder.parentFolderId !== folderId) {
            accumulator.push(folder)
            return accumulator
          }

          accumulator.push({
            ...folder,
            parentFolderId: folderToDelete.parentFolderId,
            name: getUniqueSourceFolderName(
              accumulator,
              folder.name,
              folderToDelete.parentFolderId,
              folder.id
            ),
            updatedAt: new Date()
          })
          return accumulator
        }, [])

      return {
        sourceFolders: reparentedSourceFolders,
        sourceFolderMemberships: state.sourceFolderMemberships.filter(
          (membership) => membership.folderId !== folderId
        ),
        selectedSourceFolderIds: state.selectedSourceFolderIds.filter(
          (selectedFolderId) => selectedFolderId !== folderId
        ),
        activeFolderId:
          state.activeFolderId === folderId
            ? folderToDelete.parentFolderId
            : state.activeFolderId
      }
    }),

  assignSourceToFolders: (sourceId, folderIds) =>
    set((state) => {
      if (!state.sources.some((source) => source.id === sourceId)) {
        return state
      }

      const validFolderIds = [...new Set(folderIds)].filter((folderId) =>
        state.sourceFolders.some((folder) => folder.id === folderId)
      )

      return {
        sourceFolderMemberships: [
          ...state.sourceFolderMemberships.filter(
            (membership) => membership.sourceId !== sourceId
          ),
          ...validFolderIds.map((folderId) => ({
            folderId,
            sourceId
          }))
        ]
      }
    }),

  removeSourceFromFolder: (sourceId, folderId) =>
    set((state) => ({
      sourceFolderMemberships: state.sourceFolderMemberships.filter(
        (membership) =>
          !(
            membership.sourceId === sourceId &&
            membership.folderId === folderId
          )
      )
    })),

  toggleSourceFolderSelection: (folderId) =>
    set((state) => {
      if (!state.sourceFolders.some((folder) => folder.id === folderId)) {
        return state
      }

      const isSelected = state.selectedSourceFolderIds.includes(folderId)
      return {
        selectedSourceFolderIds: isSelected
          ? state.selectedSourceFolderIds.filter(
              (selectedFolderId) => selectedFolderId !== folderId
            )
          : [...state.selectedSourceFolderIds, folderId]
      }
    }),

  setActiveFolder: (folderId) =>
    set((state) => ({
      activeFolderId:
        folderId === null ||
        state.sourceFolders.some((folder) => folder.id === folderId)
          ? folderId
          : null
    })),

  addSource: (sourceData) => {
    const source: WorkspaceSource = {
      ...sourceData,
      status: sourceData.status || "ready",
      statusMessage: sourceData.statusMessage || undefined,
      id: generateWorkspaceId(),
      addedAt: new Date()
    }
    set((state) => {
      // Prevent duplicates by mediaId
      if (state.sources.some((s) => s.mediaId === source.mediaId)) {
        return state
      }
      return { sources: [...state.sources, source] }
    })
    return source
  },

  addSources: (sourcesData) => {
    const newSources: WorkspaceSource[] = sourcesData.map((s) => ({
      ...s,
      status: s.status || "ready",
      statusMessage: s.statusMessage || undefined,
      id: generateWorkspaceId(),
      addedAt: new Date()
    }))
    set((state) => {
      // Filter out duplicates by mediaId
      const existingMediaIds = new Set(state.sources.map((s) => s.mediaId))
      const uniqueNewSources = newSources.filter(
        (s) => !existingMediaIds.has(s.mediaId)
      )
      return { sources: [...state.sources, ...uniqueNewSources] }
    })
    return newSources
  },

  removeSource: (id) =>
    set((state) => ({
      sources: state.sources.filter((s) => s.id !== id),
      selectedSourceIds: state.selectedSourceIds.filter((sid) => sid !== id),
      sourceFolderMemberships: state.sourceFolderMemberships.filter(
        (membership) => membership.sourceId !== id
      )
    })),

  removeSources: (ids) =>
    set((state) => {
      const idsSet = new Set(ids)
      return {
        sources: state.sources.filter((s) => !idsSet.has(s.id)),
        selectedSourceIds: state.selectedSourceIds.filter(
          (sid) => !idsSet.has(sid)
        ),
        sourceFolderMemberships: state.sourceFolderMemberships.filter(
          (membership) => !idsSet.has(membership.sourceId)
        )
      }
    }),

  reorderSource: (sourceId, targetIndex) =>
    set((state) => {
      const currentIndex = state.sources.findIndex(
        (source) => source.id === sourceId
      )
      if (currentIndex < 0) return state

      const boundedTargetIndex = Math.max(
        0,
        Math.min(targetIndex, state.sources.length - 1)
      )
      if (boundedTargetIndex === currentIndex) return state

      const reorderedSources = [...state.sources]
      const [movedSource] = reorderedSources.splice(currentIndex, 1)
      reorderedSources.splice(boundedTargetIndex, 0, movedSource)

      return {
        sources: reorderedSources
      }
    }),

  toggleSourceSelection: (id) =>
    set((state) => {
      const source = state.sources.find((entry) => entry.id === id)
      if (!source || getWorkspaceSourceStatus(source) !== "ready") {
        return state
      }
      const isSelected = state.selectedSourceIds.includes(id)
      return {
        selectedSourceIds: isSelected
          ? state.selectedSourceIds.filter((sid) => sid !== id)
          : [...state.selectedSourceIds, id]
      }
    }),

  selectAllSources: () =>
    set((state) => ({
      selectedSourceIds: state.sources
        .filter((source) => getWorkspaceSourceStatus(source) === "ready")
        .map((source) => source.id)
    })),

  deselectAllSources: () => set({ selectedSourceIds: [] }),

  setSelectedSourceIds: (ids) =>
    set((state) => {
      const readySourceIds = new Set(
        state.sources
          .filter((source) => getWorkspaceSourceStatus(source) === "ready")
          .map((source) => source.id)
      )
      return {
        selectedSourceIds: ids.filter((id) => readySourceIds.has(id))
      }
    }),

  setSourceSearchQuery: (query) => set({ sourceSearchQuery: query }),

  setSourceStatusById: (sourceId, status, statusMessage, readiness) =>
    set((state) => {
      const nextSources = state.sources.map((source) =>
        source.id === sourceId
          ? {
              ...source,
              status,
              statusMessage: statusMessage || undefined,
              readiness: readiness ?? source.readiness
            }
          : source
      )
      return {
        sources: nextSources,
        selectedSourceIds:
          status === "error"
            ? state.selectedSourceIds.filter((id) => id !== sourceId)
            : state.selectedSourceIds
      }
    }),

  setSourceStatusByMediaId: (mediaId, status, statusMessage, readiness) =>
    set((state) => {
      const targetSource = state.sources.find(
        (source) => source.mediaId === mediaId
      )
      if (!targetSource) return state

      const nextSources = state.sources.map((source) =>
        source.mediaId === mediaId
          ? {
              ...source,
              status,
              statusMessage: statusMessage || undefined,
              readiness: readiness ?? source.readiness
            }
          : source
      )
      return {
        sources: nextSources,
        selectedSourceIds:
          status === "error"
            ? state.selectedSourceIds.filter((id) => id !== targetSource.id)
            : state.selectedSourceIds
      }
    }),

  focusSourceById: (id) => {
    const state = get()
    const sourceExists = state.sources.some((source) => source.id === id)
    if (!sourceExists) return false

    set((current) => ({
      sourceFocusTarget: {
        sourceId: id,
        token: (current.sourceFocusTarget?.token ?? 0) + 1
      }
    }))
    return true
  },

  focusSourceByMediaId: (mediaId) => {
    const state = get()
    const source = state.sources.find((entry) => entry.mediaId === mediaId)
    if (!source) return false

    set((current) => ({
      sourceFocusTarget: {
        sourceId: source.id,
        token: (current.sourceFocusTarget?.token ?? 0) + 1
      }
    }))
    return true
  },

  clearSourceFocusTarget: () => set({ sourceFocusTarget: null }),

  setSourcesLoading: (loading) => set({ sourcesLoading: loading }),

  setSourcesError: (error) => set({ sourcesError: error }),

  restoreSource: (source, options) =>
    set((state) => {
      const sourceExists = state.sources.some(
        (entry) => entry.id === source.id || entry.mediaId === source.mediaId
      )
      if (sourceExists) {
        return state
      }

      const nextSources = [...state.sources]
      const insertionIndex = Math.min(
        Math.max(options?.index ?? nextSources.length, 0),
        nextSources.length
      )
      nextSources.splice(insertionIndex, 0, {
        ...source,
        addedAt: reviveDateOrNull(source.addedAt) || new Date()
      })

      const shouldSelect =
        options?.select === true &&
        getWorkspaceSourceStatus(source) === "ready"

      return {
        sources: nextSources,
        selectedSourceIds: shouldSelect
          ? [...new Set([...state.selectedSourceIds, source.id])]
          : state.selectedSourceIds
      }
    }),

  getSelectedSources: () => {
    const state = get()
    const selectedSet = new Set(state.selectedSourceIds)
    return state.sources.filter(
      (source) =>
        selectedSet.has(source.id) && getWorkspaceSourceStatus(source) === "ready"
    )
  },

  getSelectedMediaIds: () => {
    const state = get()
    const selectedSet = new Set(state.selectedSourceIds)
    return state.sources
      .filter(
        (source) =>
          selectedSet.has(source.id) &&
          getWorkspaceSourceStatus(source) === "ready"
      )
      .map((s) => s.mediaId)
  },

  getEffectiveSelectedSources: () => {
    const state = get()
    const organizationIndex = createWorkspaceOrganizationStateIndex(state)
    const effectiveSelectedIds = new Set(
      deriveEffectiveSelectedSourceIds(
        organizationIndex,
        state.selectedSourceIds,
        state.selectedSourceFolderIds
      )
    )

    return state.sources.filter((source) => effectiveSelectedIds.has(source.id))
  },

  getEffectiveSelectedMediaIds: () =>
    get()
      .getEffectiveSelectedSources()
      .map((source) => source.mediaId)
})
