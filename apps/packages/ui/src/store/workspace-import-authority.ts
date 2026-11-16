import type { WorkspaceExportBundle } from './workspace-bundle'
import { remapImportedWorkspaceChat } from '@/utils/workspace-chat-import'
import { readWorkspaceArtifactClientPayload } from './workspace-artifact-codec'
import type { GeneratedArtifact } from '@/types/workspace'

export type WorkspaceImportIdentity = { id: string; name: string; tag: string }
export type WorkspaceImportOptions = {
  identity: WorkspaceImportIdentity
  verifiedCanonicalScopeKey: string
}

/** Imported JSON is historical content, never authority to mutate its original server. */
export const prepareLegacyWorkspaceImport = (
  bundle: WorkspaceExportBundle,
  identity: WorkspaceImportIdentity,
  makeId: () => string = () => crypto.randomUUID()
): WorkspaceExportBundle => {
  const sourceIds = new Map((bundle.workspace.snapshot.sources ?? []).map(source => [source.id, makeId()]))
  const historicalReports = new WeakSet<object>()
  const output: WorkspaceExportBundle = JSON.parse(JSON.stringify(bundle, function (this: object, key: string, value: unknown) {
    // Receipt identities describe the original check, not destination navigation.
    if (['claim_verification', 'synthesis_verification'].includes(key) || historicalReports.has(this)) {
      if (value !== null && typeof value === 'object') historicalReports.add(value)
      return value
    }
    if (['sourceId', 'source_id'].includes(key) && typeof value === 'string') return sourceIds.get(value) ?? value
    if (['selectedSourceIds', 'source_ids'].includes(key) && Array.isArray(value))
      return value.map(id => sourceIds.get(id) ?? id)
    return value
  }))
  const snapshot = output.workspace.snapshot
  for (const source of snapshot.sources ?? []) source.id = sourceIds.get(source.id)!
  if (snapshot.sourceAnnotations) snapshot.sourceAnnotations = Object.fromEntries(
    Object.entries(snapshot.sourceAnnotations).map(([id, annotations]) => [sourceIds.get(id) ?? id, annotations])
  )
  const artifacts = snapshot.generatedArtifacts ?? []
  const artifactIds = new Map(artifacts.map(artifact => [artifact.id, makeId()]))
  snapshot.generatedArtifacts = artifacts.map(original => {
    const artifact: GeneratedArtifact = { ...readWorkspaceArtifactClientPayload(original.versionMetadata), ...original }
    artifact.id = artifactIds.get(original.id)!
    for (const key of ['workspaceArtifactId', 'workspaceArtifactVersion', 'serverId', 'ownerScope', 'ownerId', 'projectId', 'taskId', 'sourceCollectionId', 'audioUrl', 'presentationId', 'presentationVersion', 'exportRefs'] as const)
      delete artifact[key]
    for (const key of ['rootArtifactId', 'artifactVersionId', 'previousVersionId'] as const)
      if (artifact[key]) artifact[key] = artifactIds.get(artifact[key]!)
    artifact.producerMetadata = { origin: 'workspace_snapshot_import', importedProducer: original.producerMetadata ?? {} }
    artifact.versionMetadata = {
      ...artifact.versionMetadata,
      importedArtifactIdentity: {
        id: original.id,
        workspaceArtifactId: original.workspaceArtifactId,
        workspaceArtifactVersion: original.workspaceArtifactVersion,
        rootArtifactId: original.rootArtifactId,
        previousVersionId: original.previousVersionId
      }
    }
    delete artifact.versionMetadata.workspaceClient
    if (artifact.status === 'generating') {
      artifact.status = 'failed'
      artifact.errorMessage = 'Imported checkpoint. The original generation job is not connected to this workspace.'
    }
    return artifact
  })
  const note = snapshot.currentNote
  if (note) {
    for (const key of ['id', 'version', 'pendingNoteWriteKey', 'knowledge_provenance_state', 'knowledge_provenance_version', 'knowledge_provenance_hash', 'knowledge_provenance_reconciliation'] as const)
      delete note[key]
    note.isDirty = Boolean(note.title || note.content || note.isDirty)
    note.keywords = [...new Set([...(note.keywords ?? []).filter(key => !key.toLowerCase().startsWith('workspace:')), identity.tag, `workspace:${identity.id}`])]
  }
  output.workspace.name = snapshot.workspaceName = identity.name
  output.workspace.tag = snapshot.workspaceTag = identity.tag
  delete output.workspace.collectionId
  if (output.workspace.chatSession) output.workspace.chatSession = remapImportedWorkspaceChat(output.workspace.chatSession, makeId)
  return output
}
