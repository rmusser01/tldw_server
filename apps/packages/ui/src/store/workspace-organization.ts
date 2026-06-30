import type {
  WorkspaceSource,
  WorkspaceSourceFolder,
  WorkspaceSourceFolderMembership
} from "@/types/workspace"
import { isWorkspaceSourceSelectable } from "./workspace-source-status"

export type FolderSelectionState = "unchecked" | "checked" | "indeterminate"
export type SourceSelectionOrigin = "none" | "direct" | "folder" | "both"

export interface WorkspaceOrganizationIndexInput {
  sources: WorkspaceSource[]
  sourceFolders: WorkspaceSourceFolder[]
  sourceFolderMemberships: WorkspaceSourceFolderMembership[]
}

export interface WorkspaceOrganizationIndex {
  sources: WorkspaceSource[]
  sourceIdsInOrder: string[]
  folderIdsInOrder: string[]
  readySourceIds: Set<string>
  folderById: Map<string, WorkspaceSourceFolder>
  sourceById: Map<string, WorkspaceSource>
  rootFolderIds: string[]
  childrenByFolderId: Map<string, string[]>
  sourceIdsByFolderId: Map<string, string[]>
  folderIdsBySourceId: Map<string, string[]>
}

export const createWorkspaceOrganizationIndex = ({
  sources,
  sourceFolders,
  sourceFolderMemberships
}: WorkspaceOrganizationIndexInput): WorkspaceOrganizationIndex => {
  const sourceById = new Map<string, WorkspaceSource>()
  const folderById = new Map<string, WorkspaceSourceFolder>()

  for (const source of sources) {
    if (!sourceById.has(source.id)) {
      sourceById.set(source.id, source)
    }
  }

  for (const folder of sourceFolders) {
    if (!folderById.has(folder.id)) {
      folderById.set(folder.id, folder)
    }
  }

  const sourceIdsInOrder = sources
    .map((source) => source.id)
    .filter((sourceId, index, list) => list.indexOf(sourceId) === index)
  const folderIdsInOrder = sourceFolders
    .map((folder) => folder.id)
    .filter((folderId, index, list) => list.indexOf(folderId) === index)

  const readySourceIds = new Set(
    sourceIdsInOrder.filter((sourceId) => {
      const source = sourceById.get(sourceId)
      return source !== undefined && isWorkspaceSourceSelectable(source)
    })
  )

  const childrenByFolderId = new Map<string, string[]>()
  const sourceIdsByFolderId = new Map<string, string[]>()
  const folderIdsBySourceId = new Map<string, string[]>()
  const rootFolderIds: string[] = []

  for (const folderId of folderIdsInOrder) {
    childrenByFolderId.set(folderId, [])
    sourceIdsByFolderId.set(folderId, [])
  }

  for (const sourceId of sourceIdsInOrder) {
    folderIdsBySourceId.set(sourceId, [])
  }

  for (const folderId of folderIdsInOrder) {
    const folder = folderById.get(folderId)
    if (!folder) continue

    if (
      folder.parentFolderId &&
      folder.parentFolderId !== folderId &&
      folderById.has(folder.parentFolderId)
    ) {
      childrenByFolderId.set(folder.parentFolderId, [
        ...(childrenByFolderId.get(folder.parentFolderId) || []),
        folderId
      ])
      continue
    }

    rootFolderIds.push(folderId)
  }

  const membershipKeys = new Set<string>()
  for (const membership of sourceFolderMemberships) {
    if (!folderById.has(membership.folderId) || !sourceById.has(membership.sourceId)) {
      continue
    }

    const membershipKey = `${membership.folderId}::${membership.sourceId}`
    if (membershipKeys.has(membershipKey)) {
      continue
    }
    membershipKeys.add(membershipKey)

    sourceIdsByFolderId.set(membership.folderId, [
      ...(sourceIdsByFolderId.get(membership.folderId) || []),
      membership.sourceId
    ])
    folderIdsBySourceId.set(membership.sourceId, [
      ...(folderIdsBySourceId.get(membership.sourceId) || []),
      membership.folderId
    ])
  }

  return {
    sources,
    sourceIdsInOrder,
    folderIdsInOrder,
    readySourceIds,
    folderById,
    sourceById,
    rootFolderIds,
    childrenByFolderId,
    sourceIdsByFolderId,
    folderIdsBySourceId
  }
}

export const collectDescendantFolderIds = (
  index: WorkspaceOrganizationIndex,
  folderId: string
): string[] => {
  if (!index.folderById.has(folderId)) {
    return []
  }

  const descendants: string[] = []
  const visited = new Set<string>()

  const visit = (candidateId: string) => {
    if (visited.has(candidateId) || !index.folderById.has(candidateId)) {
      return
    }

    visited.add(candidateId)
    descendants.push(candidateId)

    for (const childId of index.childrenByFolderId.get(candidateId) || []) {
      visit(childId)
    }
  }

  visit(folderId)
  return descendants
}

export const collectDescendantReadySourceIds = (
  index: WorkspaceOrganizationIndex,
  folderId: string
): string[] => {
  const descendantSourceIds = new Set(collectDescendantSourceIds(index, folderId))

  return index.sourceIdsInOrder.filter(
    (sourceId) =>
      descendantSourceIds.has(sourceId) && index.readySourceIds.has(sourceId)
  )
}

export const collectDescendantSourceIds = (
  index: WorkspaceOrganizationIndex,
  folderId: string
): string[] => {
  const descendantSourceIds = new Set<string>()

  for (const descendantFolderId of collectDescendantFolderIds(index, folderId)) {
    for (const sourceId of index.sourceIdsByFolderId.get(descendantFolderId) || []) {
      descendantSourceIds.add(sourceId)
    }
  }

  return index.sourceIdsInOrder.filter((sourceId) =>
    descendantSourceIds.has(sourceId)
  )
}

const collectSelectedFolderReadySourceIdSet = (
  index: WorkspaceOrganizationIndex,
  selectedFolderIds: string[]
): Set<string> => {
  const selectedSourceIds = new Set<string>()

  for (const folderId of selectedFolderIds) {
    for (const sourceId of collectDescendantReadySourceIds(index, folderId)) {
      selectedSourceIds.add(sourceId)
    }
  }

  return selectedSourceIds
}

export const deriveEffectiveSelectedSourceIds = (
  index: WorkspaceOrganizationIndex,
  directSelectedSourceIds: string[],
  selectedFolderIds: string[]
): string[] => {
  const effectiveSelectedSourceIds = new Set<string>()

  for (const sourceId of directSelectedSourceIds) {
    if (index.readySourceIds.has(sourceId)) {
      effectiveSelectedSourceIds.add(sourceId)
    }
  }

  for (const sourceId of collectSelectedFolderReadySourceIdSet(
    index,
    selectedFolderIds
  )) {
    effectiveSelectedSourceIds.add(sourceId)
  }

  return index.sourceIdsInOrder.filter((sourceId) =>
    effectiveSelectedSourceIds.has(sourceId)
  )
}

export const getFolderSelectionState = (
  index: WorkspaceOrganizationIndex,
  folderId: string,
  directSelectedSourceIds: string[],
  selectedFolderIds: string[]
): FolderSelectionState => {
  if (!index.folderById.has(folderId)) {
    return "unchecked"
  }

  const subtreeReadySourceIds = collectDescendantReadySourceIds(index, folderId)
  if (subtreeReadySourceIds.length === 0) {
    return selectedFolderIds.includes(folderId) ? "checked" : "unchecked"
  }

  const effectiveSelectedSourceIds = new Set(
    deriveEffectiveSelectedSourceIds(
      index,
      directSelectedSourceIds,
      selectedFolderIds
    )
  )
  const selectedCount = subtreeReadySourceIds.filter((sourceId) =>
    effectiveSelectedSourceIds.has(sourceId)
  ).length

  if (selectedCount === 0) {
    return "unchecked"
  }

  if (selectedCount === subtreeReadySourceIds.length) {
    return "checked"
  }

  return "indeterminate"
}

export const getSourceSelectionOrigin = (
  sourceId: string,
  directSelectedSourceIds: string[],
  selectedFolderIds: string[],
  index: WorkspaceOrganizationIndex
): SourceSelectionOrigin => {
  if (!index.readySourceIds.has(sourceId)) {
    return "none"
  }

  const isDirectSelected = directSelectedSourceIds.includes(sourceId)
  const isFolderSelected = collectSelectedFolderReadySourceIdSet(
    index,
    selectedFolderIds
  ).has(sourceId)

  if (isDirectSelected && isFolderSelected) {
    return "both"
  }

  if (isDirectSelected) {
    return "direct"
  }

  if (isFolderSelected) {
    return "folder"
  }

  return "none"
}
