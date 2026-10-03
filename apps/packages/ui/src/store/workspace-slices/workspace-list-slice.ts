import type { WorkspaceSlice } from './types'
import type {
  WorkspaceState,
  WorkspaceChatSession,
  WorkspaceSourceTransferExecutionResult,
  WorkspaceSourceTransferRequest,
  WorkspaceUndoSnapshot
} from '../workspace'
import type {
  SavedWorkspace,
  WorkspaceCollection,
  WorkspaceConfig,
  WorkspaceSourceTransferSnapshot
} from '@/types/workspace'
import { DEFAULT_AUDIO_SETTINGS, DEFAULT_WORKSPACE_NOTE } from '@/types/workspace'
import type { WorkspaceExportBundle } from '@/store/workspace-bundle'
import {
  LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT,
  WORKSPACE_EXPORT_BUNDLE_FORMAT,
  WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION,
  sanitizeImportedChatSession
} from '@/store/workspace-bundle'
import {
  buildWorkspaceChatSessionKey,
  extractWorkspaceIdFromChatSessionKey,
  isWorkspaceChatSessionKeyForWorkspace
} from '@/store/workspace-chat-session-key'
import { applyWorkspaceSourceTransfer } from '../workspace-source-transfer'
import { normalizeWorkspaceAssistantDefaults } from '@/types/workspace-assistant-defaults'
import { buildResearchWorkspaceServerSourceSignature } from '../workspace-api'
import { parseHistorySelectionHandoff } from '@/hooks/chat/useHistorySelection'

// TODO: These helpers need to be exported from workspace.ts
import {
  generateWorkspaceId,
  createSlug,
  buildWorkspaceSnapshot,
  applyWorkspaceSnapshot,
  createEmptyWorkspaceSnapshot,
  createSavedWorkspaceEntry,
  upsertSavedWorkspace,
  getSavedWorkspaceCollectionId,
  getUniqueWorkspaceCollectionName,
  upsertArchivedWorkspace,
  sortByLastAccessedDesc,
  cloneWorkspaceChatSession,
  buildWorkspaceUndoSnapshot,
  cloneWorkspaceValue,
  hasRetainedWorkspaceContent,
  reviveDateOrNull,
  reviveSources,
  reviveSourceFolders,
  reviveSourceFolderMemberships,
  reviveArtifacts,
  reviveSavedWorkspace,
  reviveWorkspaceCollections,
  reviveWorkspaceSnapshot,
  coerceWorkspaceAssistantDefaultsForRehydrate,
  coerceWorkspaceBannerForRehydrate,
  createFallbackWorkspaceSnapshot,
  duplicateWorkspaceSnapshot,
  buildWorkspaceBundleSnapshot,
  cloneWorkspaceBundleChatSession,
  hydrateWorkspaceBundleSnapshot,
  initialSourcesState,
  initialStudioState
} from '../workspace'

// Combined type for all actions in this slice:
// WorkspaceIdentityActions + AudioSettingsActions + WorkspaceListActions + UndoActions + ResetActions
type WorkspaceListSliceActions = Pick<
  WorkspaceState,
  // Workspace Identity Actions
  | 'initializeWorkspace'
  | 'setWorkspaceName'
  | 'loadWorkspace'
  // Audio Settings Actions
  | 'setAudioSettings'
  | 'resetAudioSettings'
  // Workspace List Actions (collections)
  | 'createWorkspaceCollection'
  | 'renameWorkspaceCollection'
  | 'deleteWorkspaceCollection'
  | 'assignWorkspaceToCollection'
  // Workspace List Actions (save/switch/create/duplicate/archive/delete)
  | 'saveCurrentWorkspace'
  | 'exportWorkspaceBundle'
  | 'importWorkspaceBundle'
  | 'switchWorkspace'
  | 'installServerWorkspace'
  | 'createNewWorkspace'
  | 'duplicateWorkspace'
  | 'transferSourcesBetweenWorkspaces'
  | 'archiveWorkspace'
  | 'restoreArchivedWorkspace'
  | 'deleteWorkspace'
  | 'getSavedWorkspaces'
  | 'getArchivedWorkspaces'
  // Chat session management
  | 'saveWorkspaceChatSession'
  | 'getWorkspaceChatSession'
  | 'clearWorkspaceChatSession'
  // Undo/Restore Actions
  | 'captureUndoSnapshot'
  | 'restoreUndoSnapshot'
  // Reset Actions
  | 'reset'
  | 'resetSources'
  | 'resetStudio'
>

// TODO: initialState needs to be exported from workspace.ts
import { initialState } from '../workspace'

type WorkspaceSnapshotRecord = WorkspaceState['workspaceSnapshots'][string]

const toTransferSnapshot = (
  snapshot: WorkspaceSnapshotRecord
): WorkspaceSourceTransferSnapshot => ({
  workspaceId: snapshot.workspaceId,
  sources: snapshot.sources.map((source) => ({ ...source })),
  sourceFolders: snapshot.sourceFolders.map((folder) => ({ ...folder })),
  sourceFolderMemberships: snapshot.sourceFolderMemberships.map((membership) => ({
    ...membership
  }))
})

const reconcileSelectedSourceIds = (
  selectedSourceIds: string[],
  sources: WorkspaceSnapshotRecord['sources']
): string[] => {
  const sourceIdSet = new Set(sources.map((source) => source.id))
  return selectedSourceIds.filter((sourceId) => sourceIdSet.has(sourceId))
}

const mergeTransferredSnapshot = (
  snapshot: WorkspaceSnapshotRecord,
  transferSnapshot: WorkspaceSourceTransferSnapshot,
  nextSelectedSourceIds: string[]
): WorkspaceSnapshotRecord => {
  const sourceFolders = transferSnapshot.sourceFolders.map((folder) => ({ ...folder }))
  const folderIdSet = new Set(sourceFolders.map((folder) => folder.id))

  return {
    ...snapshot,
    sources: transferSnapshot.sources.map((source) => ({ ...source })),
    selectedSourceIds: reconcileSelectedSourceIds(
      nextSelectedSourceIds,
      transferSnapshot.sources
    ),
    sourceFolders,
    sourceFolderMemberships: transferSnapshot.sourceFolderMemberships.map(
      (membership) => ({ ...membership })
    ),
    selectedSourceFolderIds: snapshot.selectedSourceFolderIds.filter((folderId) =>
      folderIdSet.has(folderId)
    ),
    activeFolderId:
      snapshot.activeFolderId && folderIdSet.has(snapshot.activeFolderId)
        ? snapshot.activeFolderId
        : null
  }
}

export const createWorkspaceListSlice: WorkspaceSlice<WorkspaceListSliceActions> = (set, get) => ({
  // ─────────────────────────────────────────────────────────────────────────
  // Workspace Identity Actions
  // ─────────────────────────────────────────────────────────────────────────

  initializeWorkspace: (name = "New Research") => {
    const id = generateWorkspaceId()
    const slug = createSlug(name) || id.slice(0, 8)
    const createdAt = new Date()
    const tag = `workspace:${slug}`
    const snapshot = createEmptyWorkspaceSnapshot({
      id,
      name,
      tag,
      createdAt
    })

    set((state) => ({
      ...applyWorkspaceSnapshot(snapshot),
      savedWorkspaces: upsertSavedWorkspace(
        state.savedWorkspaces,
        createSavedWorkspaceEntry(snapshot, createdAt)
      ),
      archivedWorkspaces: state.archivedWorkspaces.filter(
        (workspace) => workspace.id !== id
      ),
      workspaceSnapshots: {
        ...state.workspaceSnapshots,
        [id]: snapshot
      }
    }))

    return id
  },

  setWorkspaceName: (name) => {
    set((state) => {
      const fallbackId = state.workspaceId || ""
      const slug = createSlug(name) || fallbackId.slice(0, 8)
      const nextTag = `workspace:${slug}`

      if (!state.workspaceId) {
        return {
          workspaceName: name,
          workspaceTag: nextTag
        }
      }

      const updatedSnapshot = {
        ...buildWorkspaceSnapshot(state),
        workspaceName: name,
        workspaceTag: nextTag
      }

      return {
        workspaceName: name,
        workspaceTag: nextTag,
        savedWorkspaces: state.savedWorkspaces.map((workspace) =>
          workspace.id === state.workspaceId
            ? { ...workspace, name, tag: nextTag }
            : workspace
        ),
        workspaceSnapshots: {
          ...state.workspaceSnapshots,
          [state.workspaceId]: updatedSnapshot
        }
      }
    })
  },

  loadWorkspace: (config) => {
    set((state) => {
      const existing = state.workspaceSnapshots[config.id]
      const snapshot =
        existing ??
        createEmptyWorkspaceSnapshot({
          id: config.id,
          name: config.name,
          tag: config.tag,
          createdAt: config.createdAt
        })

      const hydratedSnapshot = {
        ...snapshot,
        workspaceId: config.id,
        workspaceName: config.name,
        workspaceTag: config.tag,
        workspaceCreatedAt: config.createdAt,
        workspaceChatReferenceId:
          snapshot.workspaceChatReferenceId || config.id
      }

      return {
        ...applyWorkspaceSnapshot(hydratedSnapshot),
        savedWorkspaces: upsertSavedWorkspace(
          state.savedWorkspaces,
          createSavedWorkspaceEntry(
            hydratedSnapshot,
            new Date(),
            getSavedWorkspaceCollectionId(
              state.savedWorkspaces,
              state.archivedWorkspaces,
              hydratedSnapshot.workspaceId
            )
          )
        ),
        archivedWorkspaces: state.archivedWorkspaces.filter(
          (workspace) => workspace.id !== config.id
        ),
        workspaceSnapshots: {
          ...state.workspaceSnapshots,
          [config.id]: hydratedSnapshot
        }
      }
    })
  },

  // ─────────────────────────────────────────────────────────────────────────
  // Audio Settings Actions
  // ─────────────────────────────────────────────────────────────────────────

  setAudioSettings: (settings) =>
    set((state) => ({
      audioSettings: { ...state.audioSettings, ...settings }
    })),

  resetAudioSettings: () =>
    set({ audioSettings: { ...DEFAULT_AUDIO_SETTINGS } }),

  // ─────────────────────────────────────────────────────────────────────────
  // Workspace List Actions
  // ─────────────────────────────────────────────────────────────────────────

  createWorkspaceCollection: (name, description = null) => {
    const collection: WorkspaceCollection = {
      id: generateWorkspaceId(),
      name: "",
      description: description || null,
      createdAt: new Date(),
      updatedAt: new Date()
    }

    set((state) => {
      collection.name = getUniqueWorkspaceCollectionName(
        state.workspaceCollections,
        name
      )

      return {
        workspaceCollections: [...state.workspaceCollections, collection]
      }
    })

    return collection
  },

  renameWorkspaceCollection: (collectionId, name, description = null) =>
    set((state) => ({
      workspaceCollections: state.workspaceCollections.map((collection) =>
        collection.id === collectionId
          ? {
              ...collection,
              name: getUniqueWorkspaceCollectionName(
                state.workspaceCollections,
                name,
                collection.id
              ),
              description: description || null,
              updatedAt: new Date()
            }
          : collection
      )
    })),

  deleteWorkspaceCollection: (collectionId) =>
    set((state) => ({
      workspaceCollections: state.workspaceCollections.filter(
        (collection) => collection.id !== collectionId
      ),
      savedWorkspaces: state.savedWorkspaces.map((workspace) =>
        workspace.collectionId === collectionId
          ? { ...workspace, collectionId: null }
          : workspace
      ),
      archivedWorkspaces: state.archivedWorkspaces.map((workspace) =>
        workspace.collectionId === collectionId
          ? { ...workspace, collectionId: null }
          : workspace
      )
    })),

  assignWorkspaceToCollection: (workspaceId, collectionId) =>
    set((state) => {
      if (
        collectionId !== null &&
        !state.workspaceCollections.some(
          (collection) => collection.id === collectionId
        )
      ) {
        throw new Error(`Cannot assign workspace to missing collection "${collectionId}".`)
      }

      return {
        savedWorkspaces: state.savedWorkspaces.map((workspace) =>
          workspace.id === workspaceId
            ? { ...workspace, collectionId }
            : workspace
        ),
        archivedWorkspaces: state.archivedWorkspaces.map((workspace) =>
          workspace.id === workspaceId
            ? { ...workspace, collectionId }
            : workspace
        )
      }
    }),

  saveCurrentWorkspace: () => {
    const state = get()
    // Don't save if workspace has no ID (uninitialized)
    if (!state.workspaceId) return

    const snapshot = buildWorkspaceSnapshot(state)
    const savedWorkspace = createSavedWorkspaceEntry(
      snapshot,
      new Date(),
      getSavedWorkspaceCollectionId(
        state.savedWorkspaces,
        state.archivedWorkspaces,
        snapshot.workspaceId
      )
    )

    set((s) => {
      return {
        savedWorkspaces: upsertSavedWorkspace(
          s.savedWorkspaces,
          savedWorkspace
        ),
        archivedWorkspaces: s.archivedWorkspaces.filter(
          (workspace) => workspace.id !== savedWorkspace.id
        ),
        workspaceSnapshots: {
          ...s.workspaceSnapshots,
          [snapshot.workspaceId]: snapshot
        }
      }
    })
  },

  exportWorkspaceBundle: (id) => {
    const state = get()
    const targetWorkspaceId = id || state.workspaceId
    if (!targetWorkspaceId) return null

    const snapshot =
      targetWorkspaceId === state.workspaceId
        ? buildWorkspaceSnapshot(state)
        : state.workspaceSnapshots[targetWorkspaceId]
    if (!snapshot) return null

    const savedWorkspace =
      state.savedWorkspaces.find((workspace) => workspace.id === targetWorkspaceId) ||
      state.archivedWorkspaces.find(
        (workspace) => workspace.id === targetWorkspaceId
      ) ||
      null

    const activeChatSessionKey = buildWorkspaceChatSessionKey(
      targetWorkspaceId,
      snapshot.workspaceChatReferenceId || targetWorkspaceId
    )
    const chatSession =
      state.workspaceChatSessions[activeChatSessionKey] ||
      state.workspaceChatSessions[targetWorkspaceId]

    return {
      format: WORKSPACE_EXPORT_BUNDLE_FORMAT,
      schemaVersion: WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION,
      exportedAt: new Date().toISOString(),
      workspace: {
        name: snapshot.workspaceName || savedWorkspace?.name || "Untitled Workspace",
        tag: snapshot.workspaceTag || savedWorkspace?.tag || "",
        createdAt:
          snapshot.workspaceCreatedAt ||
          savedWorkspace?.createdAt ||
          null,
        studyMaterialsPolicy: snapshot.studyMaterialsPolicy ?? null,
        snapshot: buildWorkspaceBundleSnapshot(snapshot),
        ...(chatSession
          ? {
              chatSession: cloneWorkspaceBundleChatSession({
                messages: chatSession.messages,
                history: chatSession.history,
                historyId: chatSession.historyId,
                serverChatId: chatSession.serverChatId
              })
            }
          : {})
      }
    }
  },

  importWorkspaceBundle: (bundle) => {
    if (
      !bundle ||
      (bundle.format !== WORKSPACE_EXPORT_BUNDLE_FORMAT &&
        bundle.format !== LEGACY_WORKSPACE_EXPORT_BUNDLE_FORMAT) ||
      bundle.schemaVersion !== WORKSPACE_EXPORT_BUNDLE_SCHEMA_VERSION
    ) {
      return null
    }

    const snapshotPayload = bundle.workspace?.snapshot
    if (!snapshotPayload) return null

    const state = get()
    const now = new Date()
    const currentSnapshot = state.workspaceId
      ? buildWorkspaceSnapshot(state)
      : null

    const baseName =
      (typeof snapshotPayload.workspaceName === "string" &&
      snapshotPayload.workspaceName.trim()
        ? snapshotPayload.workspaceName.trim()
        : typeof bundle.workspace?.name === "string"
          ? bundle.workspace.name.trim()
          : "") || "Imported Workspace"
    const importedName = `${baseName} (Imported)`
    const importedId = generateWorkspaceId()
    const importedSlug = createSlug(importedName) || importedId.slice(0, 8)
    const importedTag = `workspace:${importedSlug}`

    const importedSnapshot = hydrateWorkspaceBundleSnapshot(
      snapshotPayload,
      importedId,
      importedName,
      importedTag
    )

    const nextSnapshots: Record<string, unknown> = {
      ...state.workspaceSnapshots,
      [importedSnapshot.workspaceId]: importedSnapshot
    }
    let nextSavedWorkspaces = state.savedWorkspaces

    if (currentSnapshot?.workspaceId) {
      nextSnapshots[currentSnapshot.workspaceId] = currentSnapshot
      nextSavedWorkspaces = upsertSavedWorkspace(
        nextSavedWorkspaces,
        createSavedWorkspaceEntry(
          currentSnapshot,
          now,
          getSavedWorkspaceCollectionId(
            state.savedWorkspaces,
            state.archivedWorkspaces,
            currentSnapshot.workspaceId
          )
        )
      )
    }

    nextSavedWorkspaces = upsertSavedWorkspace(
      nextSavedWorkspaces,
      createSavedWorkspaceEntry(importedSnapshot, now, null)
    )

    const importedChatSession =
      bundle.workspace?.chatSession &&
      sanitizeImportedChatSession(
        cloneWorkspaceBundleChatSession(bundle.workspace.chatSession)
      )
    const importedChatSessionKey = buildWorkspaceChatSessionKey(
      importedSnapshot.workspaceId,
      importedSnapshot.workspaceChatReferenceId
    )
    const nextWorkspaceChatSessions = importedChatSession
      ? {
          ...state.workspaceChatSessions,
          [importedChatSessionKey]: {
            messages: importedChatSession.messages,
            history: importedChatSession.history,
            historyId: importedChatSession.historyId,
            serverChatId: importedChatSession.serverChatId
          }
        }
      : state.workspaceChatSessions

    set({
      ...applyWorkspaceSnapshot(importedSnapshot),
      savedWorkspaces: nextSavedWorkspaces,
      archivedWorkspaces: state.archivedWorkspaces.filter(
        (workspace) => workspace.id !== importedSnapshot.workspaceId
      ),
      workspaceSnapshots: nextSnapshots,
      workspaceChatSessions: nextWorkspaceChatSessions
    } as Partial<WorkspaceState>)

    return importedSnapshot.workspaceId
  },

  installServerWorkspace: (workspace, { scopeKey, expectedWorkspaceId }) => {
    const state = get()
    if (state.workspaceId !== expectedWorkspaceId || !workspace.metadata) return false
    if (!state.workspaceId && (hasRetainedWorkspaceContent(state) || state.workspaceChatReferenceId)) return false
    const metadata = workspace.metadata
    const now = new Date()
    const outgoing = state.workspaceId ? buildWorkspaceSnapshot(state) : null
    const cached = workspace.id === state.workspaceId ? outgoing : state.workspaceSnapshots[workspace.id]
    const foreignOwner = cached?.serverWorkspace && cached.serverWorkspace.scopeKey !== scopeKey
    if (foreignOwner) return false
    const hasLegacyChat = Object.entries(state.workspaceChatSessions).some(([key, session]) =>
      isWorkspaceChatSessionKeyForWorkspace(key, workspace.id) &&
      (session.messages.length > 0 || session.history.length > 0 || session.serverChatId || session.historyId))
    // Older snapshots have no principal provenance. Do not adopt or discard their content.
    if (cached && !cached.serverWorkspace && (hasRetainedWorkspaceContent(cached) || hasLegacyChat ||
        cached.workspaceChatReferenceId !== workspace.id)) return false
    const previous = cached?.serverWorkspace ? cached : null
    if (previous?.currentNote.isDirty && previous.currentNote.serverScopeKey && previous.currentNote.serverScopeKey !== scopeKey) return false
    const snapshot = {
      ...createEmptyWorkspaceSnapshot({
        id: workspace.id,
        name: workspace.name || "Untitled Workspace",
        tag: `workspace:${createSlug(workspace.name) || workspace.id.slice(0, 8)}`,
        createdAt: reviveDateOrNull(metadata.created_at) || now,
        studyMaterialsPolicy: metadata.study_materials_policy,
        assistantDefaults: normalizeWorkspaceAssistantDefaults(metadata.assistantDefaults ?? metadata.assistant_defaults)
      }),
      ...(previous ? {
        workspaceTag: previous.workspaceTag,
        workspaceChatReferenceId: previous.workspaceChatReferenceId,
        sourceFolders: previous.sourceFolders,
        sourceFolderMemberships: previous.sourceFolderMemberships.filter(item => workspace.sources.some(source => source.id === item.sourceId)),
        selectedSourceFolderIds: previous.selectedSourceFolderIds,
        activeFolderId: previous.activeFolderId,
        leftPaneCollapsed: previous.leftPaneCollapsed,
        rightPaneCollapsed: previous.rightPaneCollapsed
      } : {}),
      sources: workspace.sources,
      selectedSourceIds: workspace.selectedSourceIds,
      generatedArtifacts: workspace.artifacts,
      workspaceBanner: {
        title: metadata.banner_title || "", subtitle: metadata.banner_subtitle || "", image: previous?.workspaceBanner.image || null
      },
      audioSettings: {
        ...DEFAULT_AUDIO_SETTINGS,
        ...(previous?.audioSettings || {}),
        ...(metadata.audio_provider && ["browser", "elevenlabs", "openai", "tldw"].includes(metadata.audio_provider)
          ? { provider: metadata.audio_provider as typeof DEFAULT_AUDIO_SETTINGS.provider } : {}),
        model: metadata.audio_model ?? DEFAULT_AUDIO_SETTINGS.model,
        voice: metadata.audio_voice ?? DEFAULT_AUDIO_SETTINGS.voice,
        speed: metadata.audio_speed ?? DEFAULT_AUDIO_SETTINGS.speed
      },
      currentNote: previous?.currentNote.isDirty ? { ...previous.currentNote } : { ...DEFAULT_WORKSPACE_NOTE },
      notes: previous?.notes || "",
      serverWorkspace: {
        scopeKey,
        sourceSignature: buildResearchWorkspaceServerSourceSignature(workspace.sources),
        selectedSourceSignature: workspace.selectedSourceIds.join("|"),
        metadata: Object.fromEntries(Object.entries(metadata).filter(([key]) =>
          key !== "effectiveAssistantDefault" && key !== "effective_assistant_default")) as typeof metadata,
        notes: workspace.notes.map(note => ({ ...note }))
      }
    }
    const snapshots = { ...state.workspaceSnapshots, ...(outgoing ? { [outgoing.workspaceId]: outgoing } : {}), [workspace.id]: snapshot }
    let saved = state.savedWorkspaces
    for (const item of [outgoing, snapshot]) {
      if (item) saved = upsertSavedWorkspace(saved, createSavedWorkspaceEntry(item, now,
        getSavedWorkspaceCollectionId(state.savedWorkspaces, state.archivedWorkspaces, item.workspaceId)))
    }
    set({ ...applyWorkspaceSnapshot(snapshot), workspaceSnapshots: snapshots, savedWorkspaces: saved,
      archivedWorkspaces: state.archivedWorkspaces.filter(item => item.id !== workspace.id) })
    return true
  },

  switchWorkspace: (id) => {
    const state = get()
    const targetWorkspace =
      state.savedWorkspaces.find((workspace) => workspace.id === id) || null
    const targetSnapshotFromState = state.workspaceSnapshots[id]
    if (!targetWorkspace && !targetSnapshotFromState) return

    const now = new Date()
    const currentSnapshot = state.workspaceId
      ? buildWorkspaceSnapshot(state)
      : null
    const targetSnapshot =
      targetSnapshotFromState ||
      createEmptyWorkspaceSnapshot({
        id,
        name: targetWorkspace?.name || "Untitled Workspace",
        tag: targetWorkspace?.tag || `workspace:${id.slice(0, 8)}`,
        createdAt: targetWorkspace?.createdAt || now
      })

    const nextSnapshots: Record<string, unknown> = {
      ...state.workspaceSnapshots,
      [targetSnapshot.workspaceId]: targetSnapshot
    }

    let nextSavedWorkspaces = state.savedWorkspaces
    if (currentSnapshot?.workspaceId) {
      nextSnapshots[currentSnapshot.workspaceId] = currentSnapshot
      nextSavedWorkspaces = upsertSavedWorkspace(
        nextSavedWorkspaces,
        createSavedWorkspaceEntry(
          currentSnapshot,
          now,
          getSavedWorkspaceCollectionId(
            state.savedWorkspaces,
            state.archivedWorkspaces,
            currentSnapshot.workspaceId
          )
        )
      )
    }

    nextSavedWorkspaces = upsertSavedWorkspace(
      nextSavedWorkspaces,
      createSavedWorkspaceEntry(
        targetSnapshot,
        now,
        getSavedWorkspaceCollectionId(
          state.savedWorkspaces,
          state.archivedWorkspaces,
          targetSnapshot.workspaceId
        )
      )
    )

    set({
      ...applyWorkspaceSnapshot(targetSnapshot),
      savedWorkspaces: nextSavedWorkspaces,
      archivedWorkspaces: state.archivedWorkspaces.filter(
        (workspace) => workspace.id !== targetSnapshot.workspaceId
      ),
      workspaceSnapshots: nextSnapshots
    } as Partial<WorkspaceState>)
  },

  createNewWorkspace: (name = "New Research") => {
    const state = get()
    const newId = generateWorkspaceId()
    const slug = createSlug(name) || newId.slice(0, 8)
    const createdAt = new Date()
    const tag = `workspace:${slug}`

    const newWorkspaceSnapshot = createEmptyWorkspaceSnapshot({
      id: newId,
      name,
      tag,
      createdAt
    })
    const currentSnapshot = state.workspaceId
      ? buildWorkspaceSnapshot(state)
      : null

    const nextSnapshots: Record<string, unknown> = {
      ...state.workspaceSnapshots,
      [newId]: newWorkspaceSnapshot
    }
    let nextSavedWorkspaces = state.savedWorkspaces

    if (currentSnapshot?.workspaceId) {
      nextSnapshots[currentSnapshot.workspaceId] = currentSnapshot
      nextSavedWorkspaces = upsertSavedWorkspace(
        nextSavedWorkspaces,
        createSavedWorkspaceEntry(
          currentSnapshot,
          createdAt,
          getSavedWorkspaceCollectionId(
            state.savedWorkspaces,
            state.archivedWorkspaces,
            currentSnapshot.workspaceId
          )
        )
      )
    }

    nextSavedWorkspaces = upsertSavedWorkspace(
      nextSavedWorkspaces,
      createSavedWorkspaceEntry(newWorkspaceSnapshot, createdAt)
    )

    set({
      ...applyWorkspaceSnapshot(newWorkspaceSnapshot),
      ...initialSourcesState,
      ...initialStudioState,
      savedWorkspaces: nextSavedWorkspaces,
      archivedWorkspaces: state.archivedWorkspaces.filter(
        (workspace) => workspace.id !== newWorkspaceSnapshot.workspaceId
      ),
      workspaceSnapshots: nextSnapshots
    } as Partial<WorkspaceState>)
  },

  duplicateWorkspace: (id) => {
    const state = get()
    const sourceWorkspaceId = id || state.workspaceId
    if (!sourceWorkspaceId) return null

    const currentSnapshot = state.workspaceId
      ? buildWorkspaceSnapshot(state)
      : null
    const sourceSnapshot =
      sourceWorkspaceId === state.workspaceId
        ? currentSnapshot
        : state.workspaceSnapshots[sourceWorkspaceId]
    if (!sourceSnapshot) return null

    const duplicatedSnapshot = duplicateWorkspaceSnapshot(sourceSnapshot)
    const now = new Date()
    const nextSnapshots: Record<string, unknown> = {
      ...state.workspaceSnapshots,
      [duplicatedSnapshot.workspaceId]: duplicatedSnapshot
    }
    let nextSavedWorkspaces = state.savedWorkspaces

    if (currentSnapshot?.workspaceId) {
      nextSnapshots[currentSnapshot.workspaceId] = currentSnapshot
      nextSavedWorkspaces = upsertSavedWorkspace(
        nextSavedWorkspaces,
        createSavedWorkspaceEntry(
          currentSnapshot,
          now,
          getSavedWorkspaceCollectionId(
            state.savedWorkspaces,
            state.archivedWorkspaces,
            currentSnapshot.workspaceId
          )
        )
      )
    }

    nextSavedWorkspaces = upsertSavedWorkspace(
      nextSavedWorkspaces,
      createSavedWorkspaceEntry(
        duplicatedSnapshot,
        now,
        getSavedWorkspaceCollectionId(
          state.savedWorkspaces,
          state.archivedWorkspaces,
          sourceWorkspaceId
        )
      )
    )

    set({
      ...applyWorkspaceSnapshot(duplicatedSnapshot),
      savedWorkspaces: nextSavedWorkspaces,
      archivedWorkspaces: state.archivedWorkspaces.filter(
        (workspace) => workspace.id !== duplicatedSnapshot.workspaceId
      ),
      workspaceSnapshots: nextSnapshots
    } as Partial<WorkspaceState>)

    return duplicatedSnapshot.workspaceId
  },

  transferSourcesBetweenWorkspaces: (
    request
  ): WorkspaceSourceTransferExecutionResult | null => {
    const state = get()
    if (!state.workspaceId) {
      return null
    }

    const originSnapshot = buildWorkspaceSnapshot(state)
    const originWorkspaceId = originSnapshot.workspaceId
    const now = new Date()
    const originCollectionId = getSavedWorkspaceCollectionId(
      state.savedWorkspaces,
      state.archivedWorkspaces,
      originWorkspaceId
    )

    let destinationSnapshot: WorkspaceSnapshotRecord
    let destinationWorkspaceId: string
    let destinationCollectionId: string | null
    let destinationWasCreated = false

    if (request.destination.kind === 'existing') {
      destinationWorkspaceId = request.destination.workspaceId.trim()
      if (!destinationWorkspaceId || destinationWorkspaceId === originWorkspaceId) {
        throw new Error('The transfer destination must be a different workspace.')
      }

      const archivedWorkspace = state.archivedWorkspaces.find(
        (workspace) => workspace.id === destinationWorkspaceId
      )
      if (archivedWorkspace) {
        throw new Error('Archived workspaces cannot be transfer targets in v1.')
      }

      const savedWorkspace = state.savedWorkspaces.find(
        (workspace) => workspace.id === destinationWorkspaceId
      )
      if (!savedWorkspace) {
        throw new Error(
          `Cannot transfer sources into missing workspace "${destinationWorkspaceId}".`
        )
      }

      destinationSnapshot =
        state.workspaceSnapshots[destinationWorkspaceId] ||
        createEmptyWorkspaceSnapshot({
          id: savedWorkspace.id,
          name: savedWorkspace.name,
          tag: savedWorkspace.tag,
          createdAt: savedWorkspace.createdAt
        })
      destinationCollectionId = savedWorkspace.collectionId ?? null
    } else {
      const destinationName = request.destination.name.trim() || 'New Research'
      destinationWorkspaceId = generateWorkspaceId()
      const destinationSlug =
        createSlug(destinationName) || destinationWorkspaceId.slice(0, 8)
      destinationSnapshot = createEmptyWorkspaceSnapshot({
        id: destinationWorkspaceId,
        name: destinationName,
        tag: `workspace:${destinationSlug}`,
        createdAt: now
      })
      destinationCollectionId = originCollectionId
      destinationWasCreated = true
    }

    const transferResult = applyWorkspaceSourceTransfer({
      mode: request.mode,
      originSnapshot: toTransferSnapshot(originSnapshot),
      destinationSnapshot: toTransferSnapshot(destinationSnapshot),
      selectedSourceIds: request.selectedSourceIds,
      conflictResolutions: request.conflictResolutions || {},
      emptyFolderPolicy: request.emptyFolderPolicy || 'keep',
      sourceFolderFallbackName: request.sourceFolderFallbackName || 'Untitled Folder',
      generateId: () => generateWorkspaceId()
    })

    const nextOriginSnapshot = mergeTransferredSnapshot(
      originSnapshot,
      transferResult.originSnapshot,
      originSnapshot.selectedSourceIds.filter(
        (sourceId) => !transferResult.removedOriginSourceIds.includes(sourceId)
      )
    )
    const nextDestinationSnapshot = mergeTransferredSnapshot(
      destinationSnapshot,
      transferResult.destinationSnapshot,
      destinationWasCreated
        ? transferResult.transferredDestinationSourceIds
        : destinationSnapshot.selectedSourceIds
    )

    let nextSavedWorkspaces = upsertSavedWorkspace(
      state.savedWorkspaces,
      createSavedWorkspaceEntry(nextOriginSnapshot, now, originCollectionId)
    )
    nextSavedWorkspaces = upsertSavedWorkspace(
      nextSavedWorkspaces,
      createSavedWorkspaceEntry(
        nextDestinationSnapshot,
        now,
        destinationCollectionId
      )
    )

    const nextArchivedWorkspaces = state.archivedWorkspaces.filter(
      (workspace) => workspace.id !== destinationWorkspaceId
    )
    const nextWorkspaceSnapshots = {
      ...state.workspaceSnapshots,
      [originWorkspaceId]: nextOriginSnapshot,
      [destinationWorkspaceId]: nextDestinationSnapshot
    }
    const activeSnapshot = request.switchToDestinationOnComplete
      ? nextDestinationSnapshot
      : nextOriginSnapshot

    set({
      ...applyWorkspaceSnapshot(activeSnapshot),
      savedWorkspaces: nextSavedWorkspaces,
      archivedWorkspaces: nextArchivedWorkspaces,
      workspaceSnapshots: nextWorkspaceSnapshots
    } as Partial<WorkspaceState>)

    return {
      ...transferResult,
      originWorkspaceId,
      destinationWorkspaceId,
      destinationWasCreated
    }
  },

  archiveWorkspace: (id) => {
    set((state) => {
      const now = new Date()
      const currentSnapshot = state.workspaceId
        ? buildWorkspaceSnapshot(state)
        : null
      const snapshotToArchive =
        id === state.workspaceId
          ? currentSnapshot
          : state.workspaceSnapshots[id]
      const savedEntry =
        state.savedWorkspaces.find((workspace) => workspace.id === id) ||
        state.archivedWorkspaces.find((workspace) => workspace.id === id) ||
        (snapshotToArchive
          ? createSavedWorkspaceEntry(
              snapshotToArchive,
              now,
              getSavedWorkspaceCollectionId(
                state.savedWorkspaces,
                state.archivedWorkspaces,
                id
              )
            )
          : null)

      if (!savedEntry) {
        return state
      }

      const nextSnapshots = { ...state.workspaceSnapshots }
      if (snapshotToArchive) {
        nextSnapshots[id] = snapshotToArchive
      }

      const nextSavedWorkspaces = state.savedWorkspaces.filter(
        (workspace) => workspace.id !== id
      )
      const nextArchivedWorkspaces = upsertArchivedWorkspace(
        state.archivedWorkspaces,
        {
          ...savedEntry,
          lastAccessedAt: now,
          sourceCount: snapshotToArchive
            ? snapshotToArchive.sources.length
            : savedEntry.sourceCount
        }
      )

      if (state.workspaceId !== id) {
        return {
          savedWorkspaces: nextSavedWorkspaces,
          archivedWorkspaces: nextArchivedWorkspaces,
          workspaceSnapshots: nextSnapshots
        }
      }

      if (nextSavedWorkspaces.length > 0) {
        const fallbackWorkspace = nextSavedWorkspaces[0]
        const fallbackSnapshot =
          nextSnapshots[fallbackWorkspace.id] ||
          createEmptyWorkspaceSnapshot({
            id: fallbackWorkspace.id,
            name: fallbackWorkspace.name,
            tag: fallbackWorkspace.tag,
            createdAt: fallbackWorkspace.createdAt
          })

        return {
          ...applyWorkspaceSnapshot(fallbackSnapshot),
          savedWorkspaces: upsertSavedWorkspace(
            nextSavedWorkspaces,
            createSavedWorkspaceEntry(
              fallbackSnapshot,
              now,
              fallbackWorkspace.collectionId
            )
          ),
          archivedWorkspaces: nextArchivedWorkspaces,
          workspaceSnapshots: {
            ...nextSnapshots,
            [fallbackSnapshot.workspaceId]: fallbackSnapshot
          }
        }
      }

      const replacementSnapshot = createFallbackWorkspaceSnapshot()
      return {
        ...applyWorkspaceSnapshot(replacementSnapshot),
        savedWorkspaces: [
          createSavedWorkspaceEntry(replacementSnapshot, now, null)
        ],
        archivedWorkspaces: nextArchivedWorkspaces,
        workspaceSnapshots: {
          ...nextSnapshots,
          [replacementSnapshot.workspaceId]: replacementSnapshot
        }
      }
    })
  },

  restoreArchivedWorkspace: (id) => {
    set((state) => {
      const archivedWorkspace = state.archivedWorkspaces.find(
        (workspace) => workspace.id === id
      )
      if (!archivedWorkspace) {
        return state
      }

      const snapshot =
        state.workspaceSnapshots[id] ||
        createEmptyWorkspaceSnapshot({
          id: archivedWorkspace.id,
          name: archivedWorkspace.name,
          tag: archivedWorkspace.tag,
          createdAt: archivedWorkspace.createdAt
        })
      const now = new Date()

      return {
        savedWorkspaces: upsertSavedWorkspace(
          state.savedWorkspaces,
          createSavedWorkspaceEntry(
            snapshot,
            now,
            archivedWorkspace.collectionId
          )
        ),
        archivedWorkspaces: state.archivedWorkspaces.filter(
          (workspace) => workspace.id !== id
        ),
        workspaceSnapshots: {
          ...state.workspaceSnapshots,
          [snapshot.workspaceId]: snapshot
        }
      }
    })
  },

  deleteWorkspace: (id) => {
    set((state) => {
      const nextSavedWorkspaces = state.savedWorkspaces.filter(
        (workspace) => workspace.id !== id
      )
      const nextArchivedWorkspaces = state.archivedWorkspaces.filter(
        (workspace) => workspace.id !== id
      )
      const { [id]: _removedWorkspace, ...remainingSnapshots } =
        state.workspaceSnapshots
      const remainingChatSessions = Object.fromEntries(
        Object.entries(state.workspaceChatSessions).filter(
          ([sessionKey]) => !isWorkspaceChatSessionKeyForWorkspace(sessionKey, id)
        )
      )

      if (state.workspaceId !== id) {
        return {
          savedWorkspaces: nextSavedWorkspaces,
          archivedWorkspaces: nextArchivedWorkspaces,
          workspaceSnapshots: remainingSnapshots,
          workspaceChatSessions: remainingChatSessions
        }
      }

      if (nextSavedWorkspaces.length > 0) {
        const fallbackWorkspace = nextSavedWorkspaces[0]
        const fallbackSnapshot =
          remainingSnapshots[fallbackWorkspace.id] ||
          createEmptyWorkspaceSnapshot({
            id: fallbackWorkspace.id,
            name: fallbackWorkspace.name,
            tag: fallbackWorkspace.tag,
            createdAt: fallbackWorkspace.createdAt
          })

        return {
          ...applyWorkspaceSnapshot(fallbackSnapshot),
          savedWorkspaces: upsertSavedWorkspace(
            nextSavedWorkspaces,
            createSavedWorkspaceEntry(
              fallbackSnapshot,
              new Date(),
              fallbackWorkspace.collectionId
            )
          ),
          archivedWorkspaces: nextArchivedWorkspaces,
          workspaceSnapshots: {
            ...remainingSnapshots,
            [fallbackSnapshot.workspaceId]: fallbackSnapshot
          },
          workspaceChatSessions: remainingChatSessions
        }
      }

      const replacementSnapshot = createFallbackWorkspaceSnapshot()

      return {
        ...applyWorkspaceSnapshot(replacementSnapshot),
        savedWorkspaces: [createSavedWorkspaceEntry(replacementSnapshot, new Date(), null)],
        archivedWorkspaces: nextArchivedWorkspaces,
        workspaceSnapshots: {
          ...remainingSnapshots,
          [replacementSnapshot.workspaceId]: replacementSnapshot
        },
        workspaceChatSessions: remainingChatSessions
      }
    })
  },

  getSavedWorkspaces: () => {
    const state = get()
    return sortByLastAccessedDesc(state.savedWorkspaces)
  },

  getArchivedWorkspaces: () => {
    const state = get()
    return sortByLastAccessedDesc(state.archivedWorkspaces)
  },

  saveWorkspaceChatSession: (workspaceSessionKey, session) => {
    const normalizedSessionKey = workspaceSessionKey.trim()
    if (!normalizedSessionKey) return
    set((state) => ({
      workspaceChatSessions: {
        ...state.workspaceChatSessions,
        [normalizedSessionKey]: cloneWorkspaceChatSession(session)
      }
    }))
  },

  getWorkspaceChatSession: (workspaceSessionKey, qualification) => {
    const state = get()
    const normalizedSessionKey = workspaceSessionKey.trim()
    if (!normalizedSessionKey) return null
    let session = state.workspaceChatSessions[normalizedSessionKey]
    if (qualification) {
      const checkpoint = session?.checkpoint
      if (
        !checkpoint || checkpoint.version !== 1 ||
        typeof checkpoint.draft !== 'string' ||
        typeof qualification.ownerKey !== 'string' || !qualification.ownerKey.trim() ||
        typeof qualification.workspaceId !== 'string' || !qualification.workspaceId.trim() ||
        typeof qualification.referenceId !== 'string' || !qualification.referenceId.trim() ||
        checkpoint.ownerKey !== qualification.ownerKey ||
        checkpoint.workspaceId !== qualification.workspaceId ||
        checkpoint.referenceId !== qualification.referenceId ||
        normalizedSessionKey !== buildWorkspaceChatSessionKey(
          qualification.workspaceId, qualification.referenceId
        ) ||
        [session.historyId, session.serverChatId].some(
          (id) => id !== null && (typeof id !== 'string' || !id.trim())
        )
      ) return null

      const reference = checkpoint.historySelectionReference
      if (reference === null) {
        if (
          session.messages.length || session.history.length ||
          session.historyId !== null || session.serverChatId !== null
        ) return null
      } else {
        try {
          if ('owner_kind' in reference) return null
          const handoff = parseHistorySelectionHandoff(
            `?historySelection=${encodeURIComponent(JSON.stringify({
              ...reference,
              owner_kind: session.serverChatId !== null ? 'native' : 'local'
            }))}`
          )
          if (
            !handoff ||
            handoff.conversation_id !== (session.serverChatId ?? session.historyId)
          ) return null
        } catch {
          return null
        }
      }
    } else {
      session = session ?? state.workspaceChatSessions[
        extractWorkspaceIdFromChatSessionKey(normalizedSessionKey)
      ]
    }
    return session ? cloneWorkspaceChatSession(session) : null
  },

  clearWorkspaceChatSession: (workspaceSessionKey) => {
    const normalizedSessionKey = workspaceSessionKey.trim()
    if (!normalizedSessionKey) return
    set((state) => {
      const { [normalizedSessionKey]: _removedSession, ...remainingSessions } =
        state.workspaceChatSessions
      return {
        workspaceChatSessions: remainingSessions
      }
    })
  },

  captureUndoSnapshot: () => {
    const state = get()
    return buildWorkspaceUndoSnapshot(state)
  },

  restoreUndoSnapshot: (snapshot) => {
    const clonedSnapshot = cloneWorkspaceValue(snapshot)
    const restoredSnapshots = Object.fromEntries(
      Object.entries(clonedSnapshot.workspaceSnapshots || {}).map(
        ([workspaceId, workspaceSnapshot]) => [
          workspaceId,
          reviveWorkspaceSnapshot(workspaceId, workspaceSnapshot)
        ]
      )
    )
    const restoredActiveSnapshot = restoredSnapshots[clonedSnapshot.workspaceId]
    const restoredSources = reviveSources(clonedSnapshot.sources || [])
    const restoredSourceIdSet = new Set(
      restoredSources.map((source) => source.id)
    )
    const restoredSourceFolders = reviveSourceFolders(
      clonedSnapshot.sourceFolders || [],
      clonedSnapshot.workspaceId
    )
    const restoredFolderIdSet = new Set(
      restoredSourceFolders.map((folder) => folder.id)
    )
    set({
      workspaceId: clonedSnapshot.workspaceId,
      workspaceName: clonedSnapshot.workspaceName,
      workspaceTag: clonedSnapshot.workspaceTag,
      studyMaterialsPolicy: clonedSnapshot.studyMaterialsPolicy ?? null,
      assistantDefaults: coerceWorkspaceAssistantDefaultsForRehydrate(
        clonedSnapshot.assistantDefaults
      ),
      effectiveAssistantDefault: clonedSnapshot.effectiveAssistantDefault ?? null,
      workspaceCreatedAt: reviveDateOrNull(clonedSnapshot.workspaceCreatedAt),
      workspaceChatReferenceId:
        clonedSnapshot.workspaceChatReferenceId ||
        clonedSnapshot.workspaceId,
      sources: restoredSources,
      selectedSourceIds: clonedSnapshot.selectedSourceIds || [],
      sourceFolders: restoredSourceFolders,
      sourceFolderMemberships: reviveSourceFolderMemberships(
        clonedSnapshot.sourceFolderMemberships || [],
        restoredSourceIdSet,
        restoredFolderIdSet
      ),
      selectedSourceFolderIds: (
        clonedSnapshot.selectedSourceFolderIds || []
      ).filter((id) => restoredFolderIdSet.has(id)),
      activeFolderId:
        clonedSnapshot.activeFolderId &&
        restoredFolderIdSet.has(clonedSnapshot.activeFolderId)
          ? clonedSnapshot.activeFolderId
          : null,
      generatedArtifacts: reviveArtifacts(clonedSnapshot.generatedArtifacts || []),
      notes: clonedSnapshot.notes || "",
      currentNote: clonedSnapshot.currentNote || { ...DEFAULT_WORKSPACE_NOTE },
      serverWorkspace: restoredActiveSnapshot?.serverWorkspace ?? null,
      workspaceBanner: coerceWorkspaceBannerForRehydrate(
        clonedSnapshot.workspaceBanner
      ),
      leftPaneCollapsed: Boolean(clonedSnapshot.leftPaneCollapsed),
      rightPaneCollapsed: Boolean(clonedSnapshot.rightPaneCollapsed),
      audioSettings:
        clonedSnapshot.audioSettings || { ...DEFAULT_AUDIO_SETTINGS },
      savedWorkspaces: (clonedSnapshot.savedWorkspaces || []).map(
        reviveSavedWorkspace
      ),
      archivedWorkspaces: (clonedSnapshot.archivedWorkspaces || []).map(
        reviveSavedWorkspace
      ),
      workspaceCollections: reviveWorkspaceCollections(
        clonedSnapshot.workspaceCollections || []
      ),
      workspaceSnapshots: restoredSnapshots,
      workspaceChatSessions: clonedSnapshot.workspaceChatSessions || {}
    })
  },

  // ─────────────────────────────────────────────────────────────────────────
  // Reset Actions
  // ─────────────────────────────────────────────────────────────────────────

  reset: () => set(initialState),

  resetSources: () =>
    set({
      ...initialSourcesState
    }),

  resetStudio: () =>
    set({
      ...initialStudioState
    })
})
