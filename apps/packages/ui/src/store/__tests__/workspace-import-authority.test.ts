import { beforeEach, describe, expect, it } from 'vitest'
import { useWorkspaceStore, workspacePersistWriterInternals } from '../workspace'
import { bindWorkspaceArtifactOrigin, hasWorkspaceArtifactOrigin, hasWorkspaceArtifactOriginEvidence } from '../workspace-artifact-codec'
import type { GeneratedArtifact } from '@/types/workspace'
import { bindImportedWorkspaceChat } from '@/utils/workspace-chat-import'

const artifact = (): GeneratedArtifact => ({
  id: 'old-local', workspaceArtifactId: 'old-canonical', workspaceArtifactVersion: 9,
  type: 'summary', title: 'Reviewed summary', status: 'generating', createdAt: new Date(),
  content: 'Raw draft [E1.1.1]', data: { sectionAnalysis: { fingerprint: 'frozen', sections: [{ content: 'Original chapter' }] } },
  ownerScope: 'user', ownerId: 'old-owner', serverId: 4,
  producerMetadata: { origin: 'research_workspace_output_job', job_id: 4, settings: { max_tokens: 8192 } },
  reviewStatus: 'needs_revision', reviewMetadata: { sectionReview: { schemaVersion: 1, entries: [{ correction: 'Reviewed copy' }] } },
  versionMetadata: { workspaceClient: { schemaVersion: 1, serverId: 4 }, revisionReason: 'Source review' },
  exportRefs: [{ fileId: 17, format: 'markdown' }]
})

beforeEach(() => {
  workspacePersistWriterInternals.cancelPendingWrite()
  useWorkspaceStore.getState().reset()
  useWorkspaceStore.getState().initializeWorkspace('Original')
})

const bundleWith = (rows: GeneratedArtifact[]) => {
  useWorkspaceStore.setState({ generatedArtifacts: rows })
  return useWorkspaceStore.getState().exportWorkspaceBundle()!
}

describe('workspace import authority', () => {
  it('imports snapshots as fresh local drafts while preserving evidence, review and historical producer', () => {
    const original = bindWorkspaceArtifactOrigin(artifact(), 'old-account', useWorkspaceStore.getState().workspaceId)
    const bundle = bundleWith([original])
    const id = useWorkspaceStore.getState().importWorkspaceBundle(bundle)
    const imported = useWorkspaceStore.getState().generatedArtifacts[0]
    expect(id).toBeTruthy()
    expect(imported.id).not.toBe(original.id)
    expect(imported.workspaceArtifactId).toBeUndefined()
    expect(imported.workspaceArtifactVersion).toBeUndefined()
    expect(imported.ownerId).toBeUndefined()
    expect(imported.serverId).toBeUndefined()
    expect(imported.exportRefs).toBeUndefined()
    expect(hasWorkspaceArtifactOriginEvidence(imported)).toBe(false)
    expect(imported.status).toBe('failed')
    expect(imported.content).toBe(original.content)
    expect(imported.data).toEqual(original.data)
    expect(imported.reviewMetadata).toEqual(original.reviewMetadata)
    expect(imported.reviewStatus).toBe('needs_revision')
    expect(imported.producerMetadata).toMatchObject({ origin: 'workspace_snapshot_import', importedProducer: original.producerMetadata })
    expect(imported.versionMetadata?.workspaceClient).toBeUndefined()
    expect(bundle.workspace.snapshot.generatedArtifacts[0].workspaceArtifactId).toBe('old-canonical')
  })

  it('remaps local version links to the imported artifact identities', () => {
    const base = { ...artifact(), status: 'completed' as const }
    const newer = { ...base, id: 'newer', previousVersionId: base.id, rootArtifactId: base.id }
    useWorkspaceStore.getState().importWorkspaceBundle(bundleWith([base, newer]))
    const [first, second] = useWorkspaceStore.getState().generatedArtifacts
    expect(first.id).not.toBe(base.id)
    expect(second.previousVersionId).toBe(first.id)
    expect(second.rootArtifactId).toBe(first.id)
  })

  it('regenerates source links consistently without changing Media identity or exact evidence', () => {
    const source = useWorkspaceStore.getState().addSource({ mediaId: 17, title: 'Book', type: 'pdf' })
    const original = artifact()
    original.sourceLineage = [{ sourceId: source.id, mediaId: 17, title: 'Book' }]
    original.data = { sectionAnalysis: { fingerprint: 'frozen', sections: [{ source_id: source.id, media_id: 17, content: 'Exact text' }] } }
    const bundle = bundleWith([original])
    bundle.workspace.snapshot.selectedSourceIds = [source.id]
    bundle.workspace.snapshot.sourceAnnotations = { [source.id]: [{ id: 'annotation', quote: 'Exact text', note: 'Review', createdAt: 1, updatedAt: 1 }] }
    useWorkspaceStore.getState().importWorkspaceBundle(bundle)
    const state = useWorkspaceStore.getState()
    const newId = state.sources[0].id
    expect(newId).not.toBe(source.id)
    expect(state.sources[0].mediaId).toBe(17)
    expect(state.selectedSourceIds).toEqual([newId])
    expect(state.generatedArtifacts[0].sourceLineage?.[0].sourceId).toBe(newId)
    expect(state.generatedArtifacts[0].data?.sectionAnalysis).toEqual({ fingerprint: 'frozen', sections: [{ source_id: newId, media_id: 17, content: 'Exact text' }] })
    expect(JSON.parse(localStorage.getItem(`tldw:research-workspace:source-annotations:v1:${state.workspaceId}`)!)).toEqual({ [newId]: bundle.workspace.snapshot.sourceAnnotations[source.id] })
  })

  it('adopts only runtime-verified destination artifacts without dropping their origin', () => {
    const identity = { id: 'new-destination', name: 'Restored', tag: 'workspace:restored' }
    const row = bindWorkspaceArtifactOrigin({ ...artifact(), status: 'completed' }, 'destination-account', identity.id)
    const bundle = bundleWith([row])
    const id = useWorkspaceStore.getState().importWorkspaceBundle(bundle, { identity, verifiedCanonicalScopeKey: 'destination-account' })
    const imported = useWorkspaceStore.getState().generatedArtifacts[0]
    expect(id).toBe(identity.id)
    expect(imported.workspaceArtifactId).toBe('old-canonical')
    expect(hasWorkspaceArtifactOrigin(imported, 'destination-account', identity.id)).toBe(true)
  })

  it('retains the acknowledged destination chat when adopting the enrolled workspace', () => {
    const identity = { id: 'chat-destination', name: 'Restored', tag: 'workspace:restored' }
    const bundle = bundleWith([bindWorkspaceArtifactOrigin({ ...artifact(), status: 'completed' }, 'destination-account', identity.id)])
    bundle.workspace.chatSession = { historyId: null, serverChatId: 'destination-chat', history: [{ role: 'assistant', content: 'Saved answer' }],
      messages: [{ id: 'destination-message', serverMessageId: 'destination-message', serverMessageVersion: 1, isBot: true, role: 'assistant', name: 'AI', message: 'Saved answer', sources: [] }] }
    bindImportedWorkspaceChat(bundle.workspace.chatSession, 'destination-account', identity.id)
    const id = useWorkspaceStore.getState().importWorkspaceBundle(bundle, { identity, verifiedCanonicalScopeKey: 'destination-account' })
    expect(id).toBe(identity.id)
    expect(useWorkspaceStore.getState().getWorkspaceChatSession(identity.id)?.serverChatId).toBe('destination-chat')
  })

  it.each(['unverified', 'serialized', 'other-account', 'changed-payload'])('refuses %s destination chat authority without changing the current workspace', kind => {
    const identity = { id: 'chat-destination', name: 'Restored', tag: 'workspace:restored' }
    const bundle = bundleWith([bindWorkspaceArtifactOrigin({ ...artifact(), status: 'completed' }, 'destination-account', identity.id)])
    bundle.workspace.chatSession = { historyId: null, serverChatId: 'destination-chat', history: [], messages: [] }
    if (kind !== 'unverified') bindImportedWorkspaceChat(bundle.workspace.chatSession, kind === 'other-account' ? 'foreign' : 'destination-account', identity.id)
    if (kind === 'serialized') bundle.workspace.chatSession = JSON.parse(JSON.stringify(bundle.workspace.chatSession))
    if (kind === 'changed-payload') bundle.workspace.chatSession.serverChatId = 'substituted-chat'
    const before = useWorkspaceStore.getState().workspaceId
    expect(useWorkspaceStore.getState().importWorkspaceBundle(bundle, { identity, verifiedCanonicalScopeKey: 'destination-account' })).toBeNull()
    expect(useWorkspaceStore.getState().workspaceId).toBe(before)
  })

  it.each(['serialized', 'foreign-account', 'foreign-workspace', 'existing-workspace'])('rejects %s authority before changing local state', (kind) => {
    const current = useWorkspaceStore.getState().workspaceId
    const identity = { id: kind === 'existing-workspace' ? current : 'new-destination', name: 'Restored', tag: 'workspace:restored' }
    const row = bindWorkspaceArtifactOrigin({ ...artifact(), status: 'completed' }, kind === 'foreign-account' ? 'other-account' : 'destination-account', kind === 'foreign-workspace' ? 'other-workspace' : identity.id)
    const initial = bundleWith([row])
    const bundle = kind === 'serialized' ? JSON.parse(JSON.stringify(initial)) : initial
    const before = useWorkspaceStore.getState()
    expect(useWorkspaceStore.getState().importWorkspaceBundle(bundle, { identity, verifiedCanonicalScopeKey: 'destination-account' })).toBeNull()
    expect(useWorkspaceStore.getState()).toBe(before)
  })
})
