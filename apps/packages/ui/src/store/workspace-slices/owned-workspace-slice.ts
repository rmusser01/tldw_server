import type { WorkspaceSlice } from "./types"
import type { WorkspaceState } from "../workspace"
import {
  applyWorkspaceSnapshot,
  buildWorkspaceSnapshot,
  createEmptyWorkspaceSnapshot,
  createSavedWorkspaceEntry,
  getSavedWorkspaceCollectionId,
  upsertSavedWorkspace
} from "../workspace"
import {
  createOwnedWorkspaceDraftStore,
  ownedWorkspaceDraftKey,
  ownedWorkspaceRenameDraftSchema,
  ownedWorkspaceAssistantDraftSchema,
  ownedWorkspaceBannerDraftSchema,
  prepareOwnedWorkspaceActivation,
  type OwnedWorkspaceActivationResult,
  type OwnedWorkspaceAttempt,
  type OwnedWorkspaceDraft,
  type OwnedWorkspaceRenameDraft,
  type OwnedWorkspaceAssistantDraft,
  type OwnedWorkspaceBannerDraft,
  type OwnedWorkspaceScope
} from "../owned-workspace-state"
import {
  mapServerArtifactToLocal,
  mapServerSourceToLocal,
  validateOwnedWorkspaceNotes,
  validateOwnedWorkspaceMetadata,
  validateOwnedWorkspaceRecord,
  type OwnedWorkspaceBundle
} from "../workspace-api"
import {
  DEFAULT_AUDIO_SETTINGS,
  type WorkspaceNote
} from "../../types/workspace"
import {
  normalizeWorkspaceApiResponse,
  type WorkspaceApiResponse,
  type WorkspaceNoteApiResponse
} from "@/services/tldw/domains/workspace-api"

type Snapshot = ReturnType<typeof buildWorkspaceSnapshot>
type DraftStore = ReturnType<typeof createOwnedWorkspaceDraftStore>
type RenameReceiptResult = "stale" | "saved" | "newer-draft" | "editor-changed"
type RenameReceiptInput = {
  origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
  workspaceId: string
  expectedWorkspace: WorkspaceApiResponse
  submittedDraft: OwnedWorkspaceRenameDraft
  editorSession: number
  workspace: WorkspaceApiResponse
}
type AssistantReceiptInput = Omit<RenameReceiptInput, "submittedDraft"> & {
  submittedDraft: OwnedWorkspaceAssistantDraft
}
type BannerReceiptInput = Omit<RenameReceiptInput, "submittedDraft"> & {
  submittedDraft: OwnedWorkspaceBannerDraft
}
export type OwnedWorkspaceState = {
  activeWorkspaceOrigin:
    | { kind: "legacy-local" }
    | { kind: "server-owned"; scope: OwnedWorkspaceScope }
  ownedWorkspaceAttempt: OwnedWorkspaceAttempt | null
  ownedWorkspaceBundle: OwnedWorkspaceBundle | null
  ownedWorkspaceBaseline: Snapshot | null
  ownedWorkspaceComposer: string
  ownedWorkspaceRenameDraft: OwnedWorkspaceRenameDraft | null
  ownedWorkspaceRenameSession: number
  ownedWorkspaceAssistantDraft: OwnedWorkspaceAssistantDraft | null
  ownedWorkspaceAssistantSession: number
  ownedWorkspaceBannerDraft: OwnedWorkspaceBannerDraft | null
  ownedWorkspaceBannerSession: number
  ownedNoteEditorSession: number
  ownedWorkspaceDraftStatus:
    | "saved"
    | "unavailable"
    | "invalid"
    | "conflict"
    | null
  ownedWorkspaceConflict:
    | "server-edits"
    | "note-changed"
    | "note-missing"
    | "concurrent-edits"
    | "invalid"
    | "unavailable"
    | null
}
export type OwnedWorkspaceActions = {
  beginOwnedWorkspace(
    scope: OwnedWorkspaceScope,
    id: string
  ): OwnedWorkspaceAttempt
  activateOwnedWorkspace(
    attempt: OwnedWorkspaceAttempt,
    bundle: OwnedWorkspaceBundle
  ): OwnedWorkspaceActivationResult
  invalidateOwnedWorkspace(): void
  prepareOwnedWorkspaceArchive(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
  }): boolean
  completeOwnedWorkspaceArchive(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
    workspace: WorkspaceApiResponse
  }): boolean
  setOwnedWorkspaceComposer(value: string): void
  setOwnedWorkspaceRenameDraft(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
    expectedDraft: OwnedWorkspaceRenameDraft | null
    draft: OwnedWorkspaceRenameDraft | null
  }): boolean
  applyOwnedWorkspaceRenameReceipt(
    input: RenameReceiptInput
  ): RenameReceiptResult
  acceptOwnedWorkspaceRenameReview(
    input: RenameReceiptInput
  ): RenameReceiptResult
  setOwnedWorkspaceAssistantDraft(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
    expectedDraft: OwnedWorkspaceAssistantDraft | null
    draft: OwnedWorkspaceAssistantDraft | null
  }): boolean
  applyOwnedWorkspaceAssistantReceipt(
    input: AssistantReceiptInput
  ): RenameReceiptResult
  acceptOwnedWorkspaceAssistantReview(
    input: AssistantReceiptInput
  ): RenameReceiptResult
  setOwnedWorkspaceBannerDraft(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
    expectedDraft: OwnedWorkspaceBannerDraft | null
    draft: OwnedWorkspaceBannerDraft | null
  }): boolean
  applyOwnedWorkspaceBannerReceipt(
    input: BannerReceiptInput
  ): RenameReceiptResult
  acceptOwnedWorkspaceBannerReview(
    input: BannerReceiptInput
  ): RenameReceiptResult
  setOwnedNoteCreateUncertain(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
    editorSession: number
    value: boolean
  }): WorkspaceNote | null
  applyOwnedWorkspaceNoteReceipt(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
    editorSession: number
    submitted: WorkspaceNote
    note: WorkspaceNoteApiResponse
  }): "stale" | "saved" | "newer-draft" | "editor-changed"
  replaceOwnedWorkspaceNotes(input: {
    origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
    workspaceId: string
    expectedBundle: OwnedWorkspaceBundle
    notes: WorkspaceNoteApiResponse[]
  }): boolean
}
export const initialOwnedWorkspaceState: OwnedWorkspaceState = {
  activeWorkspaceOrigin: { kind: "legacy-local" },
  ownedWorkspaceAttempt: null,
  ownedWorkspaceBundle: null,
  ownedWorkspaceBaseline: null,
  ownedWorkspaceComposer: "",
  ownedWorkspaceRenameDraft: null,
  ownedWorkspaceRenameSession: 0,
  ownedWorkspaceAssistantDraft: null,
  ownedWorkspaceAssistantSession: 0,
  ownedWorkspaceBannerDraft: null,
  ownedWorkspaceBannerSession: 0,
  ownedNoteEditorSession: 0,
  ownedWorkspaceDraftStatus: null,
  ownedWorkspaceConflict: null
}

const clearedTransientState = {
  sourcesLoading: false,
  sourcesError: null,
  sourceSearchQuery: "",
  sourceFocusTarget: null,
  isGeneratingOutput: false,
  generatingOutputType: null,
  addSourceModalOpen: false,
  addSourceProcessing: false,
  addSourceError: null,
  chatFocusTarget: null,
  noteFocusTarget: null
}
function clearedOwnedView() {
  return {
    ...applyWorkspaceSnapshot(
      createEmptyWorkspaceSnapshot({
        id: "",
        name: "",
        tag: "",
        createdAt: new Date()
      })
    ),
    ...clearedTransientState
  }
}

function sourceEdits(sources: Snapshot["sources"]) {
  return sources.map(
    ({
      status: _status,
      statusMessage: _message,
      statusDetails: _details,
      readiness: _readiness,
      ...source
    }) => source
  )
}

function metadataProjection(metadata: WorkspaceApiResponse) {
  const provider = metadata.audio_provider
  const supported =
    provider === "browser" ||
    provider === "elevenlabs" ||
    provider === "openai" ||
    provider === "tldw"
  return {
    workspaceName: metadata.name ?? "Untitled Workspace",
    studyMaterialsPolicy: metadata.study_materials_policy ?? null,
    assistantDefaults: metadata.assistantDefaults ?? null,
    workspaceBanner: {
      title: metadata.banner_title ?? "",
      subtitle: metadata.banner_subtitle ?? "",
      image: null
    },
    audioSettings: {
      ...DEFAULT_AUDIO_SETTINGS,
      provider: supported ? provider : "tldw",
      backend: supported ? "" : (provider ?? ""),
      model: metadata.audio_model ?? "",
      voice: metadata.audio_voice ?? "",
      speed: metadata.audio_speed ?? 1,
      allowFallback: false
    }
  } satisfies Partial<Snapshot>
}

export function captureOwnedWorkspaceDraft(
  state: WorkspaceState
): OwnedWorkspaceDraft | null {
  if (state.activeWorkspaceOrigin.kind !== "server-owned") return null
  const snapshot = buildWorkspaceSnapshot(state)
  const pendingChanges: OwnedWorkspaceDraft["pendingChanges"] = {}
  // Canonical mutations are retained for recovery, never replayed as server truth.
  for (const key of [
    "workspaceName",
    "studyMaterialsPolicy",
    "assistantDefaults",
    "selectedSourceIds",
    "generatedArtifacts",
    "workspaceBanner",
    "audioSettings"
  ] as const) {
    if (
      JSON.stringify(snapshot[key]) !==
      JSON.stringify(state.ownedWorkspaceBaseline?.[key])
    ) {
      pendingChanges[key] = JSON.parse(JSON.stringify(snapshot[key]))
    }
  }
  const sources = sourceEdits(snapshot.sources)
  if (
    JSON.stringify(sources) !==
    JSON.stringify(sourceEdits(state.ownedWorkspaceBaseline?.sources ?? []))
  ) {
    pendingChanges.sources = JSON.parse(JSON.stringify(sources))
  }
  return {
    schemaVersion: 1,
    scope: state.activeWorkspaceOrigin.scope,
    workspaceId: state.workspaceId,
    notes: state.notes,
    composer: state.ownedWorkspaceComposer,
    renameDraft: state.ownedWorkspaceRenameDraft,
    assistantDraft: state.ownedWorkspaceAssistantDraft,
    bannerDraft: state.ownedWorkspaceBannerDraft,
    currentNote: {
      ...state.currentNote,
      keywords: [...state.currentNote.keywords]
    },
    sourceFolders: state.sourceFolders.map((folder) => ({
      ...folder,
      createdAt: folder.createdAt.toISOString(),
      updatedAt: folder.updatedAt.toISOString()
    })),
    sourceFolderMemberships: state.sourceFolderMemberships,
    selectedSourceFolderIds: state.selectedSourceFolderIds,
    activeFolderId: state.activeFolderId,
    leftPaneCollapsed: state.leftPaneCollapsed,
    rightPaneCollapsed: state.rightPaneCollapsed,
    pendingChanges
  }
}

/** Save at action boundaries, including the outgoing state on a scope/target change. */
export function preserveOwnedWorkspaceDrafts(
  before: WorkspaceState,
  after: WorkspaceState,
  drafts: DraftStore
): WorkspaceState {
  let status = after.ownedWorkspaceDraftStatus
  const outgoing = captureOwnedWorkspaceDraft(before)
  const incoming = captureOwnedWorkspaceDraft(after)
  if (
    outgoing &&
    drafts.hasPendingWrite(outgoing.scope, outgoing.workspaceId) &&
    (!incoming ||
      ownedWorkspaceDraftKey(outgoing.scope, outgoing.workspaceId) !==
        ownedWorkspaceDraftKey(incoming.scope, incoming.workspaceId))
  ) {
    status = drafts.save(outgoing).status
    if (status === "invalid")
      throw new Error("Cannot leave an invalid server-owned workspace draft")
  }
  if (
    incoming &&
    (status !== "saved" ||
      JSON.stringify(incoming) !== JSON.stringify(outgoing))
  ) {
    status = drafts.save(incoming).status
    if (status === "invalid")
      throw new Error("Cannot apply an invalid server-owned workspace draft")
  }
  return {
    ...after,
    ownedWorkspaceDraftStatus: drafts.hasPendingWrites()
      ? "unavailable"
      : status
  }
}

export function assertLegacyWorkspace(state: WorkspaceState): void {
  if (state.activeWorkspaceOrigin.kind === "server-owned") {
    throw new Error(
      "This legacy action is not available for a server-owned workspace"
    )
  }
}

export function createOwnedWorkspaceSlice(
  set: Parameters<WorkspaceSlice<OwnedWorkspaceActions>>[0],
  get: () => WorkspaceState,
  drafts: DraftStore
): OwnedWorkspaceActions {
  let generation = 0
  const isCurrent = (
    state: WorkspaceState,
    input: {
      origin: OwnedWorkspaceState["activeWorkspaceOrigin"]
      workspaceId: string
    }
  ) =>
    state.activeWorkspaceOrigin.kind === "server-owned" &&
    state.activeWorkspaceOrigin === input.origin &&
    state.workspaceId === input.workspaceId &&
    state.ownedWorkspaceBundle?.workspace.id === input.workspaceId &&
    !state.ownedWorkspaceAttempt
  const metadataReceiptPatch = (
    state: WorkspaceState,
    input: RenameReceiptInput | AssistantReceiptInput | BannerReceiptInput,
    inspected: boolean
  ) => {
    if (
      !isCurrent(state, input) ||
      state.ownedWorkspaceBundle?.workspace !== input.expectedWorkspace ||
      !state.ownedWorkspaceBaseline
    )
      return null
    validateOwnedWorkspaceMetadata(input.workspace, input.workspaceId)
    const baseVersion = input.submittedDraft.baseVersion
    if (!Number.isSafeInteger(baseVersion) || baseVersion < 1)
      throw new Error("Invalid workspace editor base version")
    const minimumVersion = Math.max(
      baseVersion,
      input.expectedWorkspace.version
    )
    // An explicit GET review may confirm an unchanged version; PATCH must advance it.
    if (
      input.workspace.version < minimumVersion ||
      (!inspected && input.workspace.version === minimumVersion)
    )
      throw new Error("Invalid workspace metadata receipt version")
    const metadata = normalizeWorkspaceApiResponse(
      structuredClone(input.workspace)
    )
    const projection = metadataProjection(metadata)
    // Refresh clean projections while retaining unrelated unsaved local edits.
    const clean = Object.fromEntries(
      Object.entries(projection).filter(
        ([key]) =>
          JSON.stringify(state[key as keyof typeof projection]) ===
          JSON.stringify(
            state.ownedWorkspaceBaseline![key as keyof typeof projection]
          )
      )
    ) as Partial<typeof projection>
    return {
      ...clean,
      ownedWorkspaceBundle: {
        ...state.ownedWorkspaceBundle!,
        workspace: metadata
      },
      ownedWorkspaceBaseline: {
        ...state.ownedWorkspaceBaseline,
        ...projection
      },
      effectiveAssistantDefault: metadata.effectiveAssistantDefault ?? null
    }
  }
  const applyRenameReceipt = (
    input: RenameReceiptInput,
    inspected = false
  ): RenameReceiptResult => {
    let result: RenameReceiptResult = "stale"
    set((state) => {
      const patch = metadataReceiptPatch(state, input, inspected)
      if (!patch) return state
      ownedWorkspaceRenameDraftSchema.parse(input.submittedDraft)
      const draft = state.ownedWorkspaceRenameDraft
      const sameSession =
        state.ownedWorkspaceRenameSession === input.editorSession
      result =
        !sameSession || !draft
          ? "editor-changed"
          : draft === input.submittedDraft
            ? "saved"
            : "newer-draft"
      return {
        ...patch,
        ownedWorkspaceRenameDraft:
          result === "saved"
            ? null
            : result === "newer-draft"
              ? { ...draft!, baseVersion: input.workspace.version }
              : draft,
        ownedWorkspaceRenameSession:
          state.ownedWorkspaceRenameSession + (result === "saved" ? 1 : 0)
      }
    })
    return result
  }
  const applyAssistantReceipt = (
    input: AssistantReceiptInput,
    inspected = false
  ): RenameReceiptResult => {
    let result: RenameReceiptResult = "stale"
    set((state) => {
      const patch = metadataReceiptPatch(state, input, inspected)
      if (!patch) return state
      ownedWorkspaceAssistantDraftSchema.parse(input.submittedDraft)
      const draft = state.ownedWorkspaceAssistantDraft
      result =
        state.ownedWorkspaceAssistantSession !== input.editorSession || !draft
          ? "editor-changed"
          : draft === input.submittedDraft
            ? "saved"
            : "newer-draft"
      return {
        ...patch,
        ownedWorkspaceAssistantDraft:
          result === "saved"
            ? null
            : result === "newer-draft"
              ? { ...draft!, baseVersion: input.workspace.version }
              : draft,
        ownedWorkspaceAssistantSession:
          state.ownedWorkspaceAssistantSession + (result === "saved" ? 1 : 0)
      }
    })
    return result
  }
  const applyBannerReceipt = (
    input: BannerReceiptInput,
    inspected = false
  ): RenameReceiptResult => {
    let result: RenameReceiptResult = "stale"
    set((state) => {
      const patch = metadataReceiptPatch(state, input, inspected)
      if (!patch) return state
      ownedWorkspaceBannerDraftSchema.parse(input.submittedDraft)
      const draft = state.ownedWorkspaceBannerDraft
      result =
        state.ownedWorkspaceBannerSession !== input.editorSession || !draft
          ? "editor-changed"
          : draft === input.submittedDraft
            ? "saved"
            : "newer-draft"
      return {
        ...patch,
        ownedWorkspaceBannerDraft:
          result === "saved"
            ? null
            : result === "newer-draft"
              ? { ...draft!, baseVersion: input.workspace.version }
              : draft,
        ownedWorkspaceBannerSession:
          state.ownedWorkspaceBannerSession + (result === "saved" ? 1 : 0)
      }
    })
    return result
  }
  return {
    applyOwnedWorkspaceBannerReceipt: (input) => applyBannerReceipt(input),
    acceptOwnedWorkspaceBannerReview: (input) =>
      applyBannerReceipt(input, true),
    setOwnedWorkspaceBannerDraft(input) {
      let applied = false
      set((state) => {
        if (
          !isCurrent(state, input) ||
          state.ownedWorkspaceBannerDraft !== input.expectedDraft
        )
          return state
        if (input.draft) ownedWorkspaceBannerDraftSchema.parse(input.draft)
        applied = true
        return {
          ownedWorkspaceBannerDraft: input.draft,
          ownedWorkspaceBannerSession:
            state.ownedWorkspaceBannerSession +
            (Boolean(input.draft) !== Boolean(input.expectedDraft) ? 1 : 0)
        }
      })
      return applied
    },
    applyOwnedWorkspaceAssistantReceipt: (input) =>
      applyAssistantReceipt(input),
    acceptOwnedWorkspaceAssistantReview: (input) =>
      applyAssistantReceipt(input, true),
    setOwnedWorkspaceAssistantDraft(input) {
      let applied = false
      set((state) => {
        if (
          !isCurrent(state, input) ||
          state.ownedWorkspaceAssistantDraft !== input.expectedDraft
        )
          return state
        if (input.draft) ownedWorkspaceAssistantDraftSchema.parse(input.draft)
        applied = true
        return {
          ownedWorkspaceAssistantDraft: input.draft,
          ownedWorkspaceAssistantSession:
            state.ownedWorkspaceAssistantSession +
            (Boolean(input.draft) !== Boolean(input.expectedDraft) ? 1 : 0)
        }
      })
      return applied
    },
    applyOwnedWorkspaceRenameReceipt: (input) => applyRenameReceipt(input),
    acceptOwnedWorkspaceRenameReview: (input) =>
      applyRenameReceipt(input, true),
    setOwnedWorkspaceRenameDraft(input) {
      let applied = false
      set((state) => {
        if (
          !isCurrent(state, input) ||
          state.ownedWorkspaceRenameDraft !== input.expectedDraft
        )
          return state
        if (input.draft) ownedWorkspaceRenameDraftSchema.parse(input.draft)
        applied = true
        return {
          ownedWorkspaceRenameDraft: input.draft,
          ownedWorkspaceRenameSession:
            state.ownedWorkspaceRenameSession +
            (Boolean(input.draft) !== Boolean(input.expectedDraft) ? 1 : 0)
        }
      })
      return applied
    },
    setOwnedNoteCreateUncertain(input) {
      const previous = get().currentNote
      let marked: WorkspaceNote | null = null
      set((state) => {
        if (
          !isCurrent(state, input) ||
          state.ownedNoteEditorSession !== input.editorSession ||
          state.currentNote.id != null
        )
          return state
        const { createUncertain: _previous, ...draft } = state.currentNote
        marked = input.value ? { ...draft, createUncertain: true } : draft
        return { currentNote: marked }
      })
      const marker =
        marked && input.value ? captureOwnedWorkspaceDraft(get()) : null
      // Other targets can retain failed writes; only this marker gates creation.
      if (
        marked &&
        input.value &&
        (!marker || drafts.save(marker).status !== "saved")
      ) {
        set((state) =>
          isCurrent(state, input) &&
          state.currentNote === marked &&
          state.ownedNoteEditorSession === input.editorSession
            ? { currentNote: previous }
            : state
        )
        throw new Error(
          "Draft recovery storage is unavailable. Restore local storage before saving."
        )
      }
      return marked
    },
    applyOwnedWorkspaceNoteReceipt(input) {
      let result: ReturnType<
        OwnedWorkspaceActions["applyOwnedWorkspaceNoteReceipt"]
      > = "stale"
      set((state) => {
        if (!isCurrent(state, input)) return state
        validateOwnedWorkspaceNotes([input.note], input.workspaceId)
        const existing = state.ownedWorkspaceBundle!.notes.find(
          (note) => note.id === input.note.id
        )
        if (
          (existing && existing.version > input.note.version) ||
          (existing?.version === input.note.version &&
            (existing.title !== input.note.title ||
              existing.content !== input.note.content ||
              existing.keywords_json !== input.note.keywords_json)) ||
          (input.submitted.id != null &&
            (!existing ||
              input.note.id !== input.submitted.id ||
              input.submitted.version == null ||
              input.note.version <= input.submitted.version)) ||
          (state.ownedNoteEditorSession === input.editorSession &&
            (state.currentNote.id !== input.submitted.id ||
              state.currentNote.version !== input.submitted.version))
        )
          return state
        const keywords: unknown = JSON.parse(input.note.keywords_json)
        if (
          !Array.isArray(keywords) ||
          keywords.some((keyword) => typeof keyword !== "string")
        )
          throw new Error("Invalid canonical note keywords")
        const notes = existing
          ? state.ownedWorkspaceBundle!.notes.map((note) =>
              note.id === input.note.id ? input.note : note
            )
          : [...state.ownedWorkspaceBundle!.notes, input.note]
        const bundle = { ...state.ownedWorkspaceBundle!, notes }
        if (state.ownedNoteEditorSession !== input.editorSession) {
          result = "editor-changed"
          return { ownedWorkspaceBundle: bundle }
        }
        const unchanged = state.currentNote === input.submitted
        const { createUncertain: _uncertain, ...currentDraft } =
          state.currentNote
        result = unchanged ? "saved" : "newer-draft"
        return {
          ownedWorkspaceBundle: bundle,
          currentNote: unchanged
            ? {
                id: input.note.id,
                title: input.note.title,
                content: input.note.content,
                keywords: keywords as string[],
                version: input.note.version,
                isDirty: false
              }
            : {
                ...currentDraft,
                id: input.note.id,
                version: input.note.version,
                isDirty: true
              }
        }
      })
      return result
    },
    replaceOwnedWorkspaceNotes(input) {
      let applied = false
      set((state) => {
        if (
          !isCurrent(state, input) ||
          state.ownedWorkspaceBundle !== input.expectedBundle
        )
          return state
        validateOwnedWorkspaceNotes(input.notes, input.workspaceId)
        applied = true
        return {
          ownedWorkspaceBundle: { ...input.expectedBundle, notes: input.notes }
        }
      })
      return applied
    },
    beginOwnedWorkspace(scope, id) {
      if (!get().storeHydrated)
        throw new Error("Wait for workspace storage hydration")
      ownedWorkspaceDraftKey(scope, id)
      const attempt = {
        scope: structuredClone(scope),
        workspaceId: id,
        generation: ++generation
      }
      set((state) => ({
        ...(state.activeWorkspaceOrigin.kind === "server-owned" &&
        ownedWorkspaceDraftKey(state.activeWorkspaceOrigin.scope, id) !==
          ownedWorkspaceDraftKey(scope, id)
          ? clearedOwnedView()
          : {}),
        ownedWorkspaceAttempt: attempt,
        ownedWorkspaceConflict: null
      }))
      return structuredClone(attempt)
    },
    activateOwnedWorkspace(attempt, bundle) {
      let result: OwnedWorkspaceActivationResult = "stale"
      set((state) => {
        if (
          !state.storeHydrated ||
          prepareOwnedWorkspaceActivation(
            attempt,
            state.ownedWorkspaceAttempt,
            bundle
          ).status === "stale"
        )
          return state
        let recovered = drafts.load(attempt.scope, attempt.workspaceId)
        const current = captureOwnedWorkspaceDraft(state)
        if (
          recovered.status === "missing" &&
          !recovered.deleted &&
          current &&
          ownedWorkspaceDraftKey(current.scope, current.workspaceId) ===
            ownedWorkspaceDraftKey(attempt.scope, attempt.workspaceId)
        ) {
          const saved = drafts.save(current)
          recovered = {
            status: "ready",
            draft: current,
            durable: saved.status === "saved"
          }
        }
        if (recovered.status === "conflict") {
          result = "draft-conflict"
          return { ownedWorkspaceConflict: "concurrent-edits" }
        }
        if (
          recovered.status === "invalid" ||
          recovered.status === "unavailable"
        ) {
          result = "draft-conflict"
          return { ownedWorkspaceConflict: recovered.status }
        }
        const prepared = prepareOwnedWorkspaceActivation(
          attempt,
          state.ownedWorkspaceAttempt,
          bundle,
          recovered.status === "ready" ? recovered.draft : undefined
        )
        result = prepared.status
        if (prepared.status === "stale") return state
        if (prepared.status === "draft-conflict")
          return { ownedWorkspaceConflict: prepared.reason }
        const metadata = prepared.bundle.workspace
        const snapshot = createEmptyWorkspaceSnapshot({
          id: metadata.id,
          name: metadata.name ?? "Untitled Workspace",
          tag: "",
          createdAt: new Date(metadata.created_at),
          studyMaterialsPolicy: metadata.study_materials_policy,
          assistantDefaults: metadata.assistantDefaults ?? null
        })
        snapshot.sources = prepared.bundle.sources.map(mapServerSourceToLocal)
        snapshot.selectedSourceIds = prepared.bundle.sources
          .filter((source) => source.selected)
          .map((source) => source.id)
        snapshot.generatedArtifacts = prepared.bundle.artifacts.map(
          mapServerArtifactToLocal
        )
        Object.assign(snapshot, metadataProjection(metadata))
        const baseline = structuredClone(snapshot)
        const draft = prepared.draft
        if (draft) {
          snapshot.notes = draft.notes
          snapshot.currentNote = draft.currentNote
          snapshot.sourceFolders = draft.sourceFolders.map((folder) => ({
            ...folder,
            parentFolderId: folder.parentFolderId ?? null,
            createdAt: new Date(folder.createdAt),
            updatedAt: new Date(folder.updatedAt)
          }))
          snapshot.sourceFolderMemberships = draft.sourceFolderMemberships
          snapshot.selectedSourceFolderIds = draft.selectedSourceFolderIds
          snapshot.activeFolderId = draft.activeFolderId
          snapshot.leftPaneCollapsed = draft.leftPaneCollapsed
          snapshot.rightPaneCollapsed = draft.rightPaneCollapsed
        }
        const outgoing =
          state.activeWorkspaceOrigin.kind === "legacy-local" &&
          state.workspaceId
            ? buildWorkspaceSnapshot(state)
            : null
        return {
          ...applyWorkspaceSnapshot(snapshot),
          activeWorkspaceOrigin: {
            kind: "server-owned",
            scope: structuredClone(attempt.scope)
          },
          ownedWorkspaceBundle: prepared.bundle,
          ownedWorkspaceBaseline: baseline,
          ownedWorkspaceAttempt: null,
          ownedWorkspaceComposer: draft?.composer ?? "",
          ownedWorkspaceRenameDraft: draft?.renameDraft ?? null,
          ownedWorkspaceRenameSession: state.ownedWorkspaceRenameSession + 1,
          ownedWorkspaceAssistantDraft: draft?.assistantDraft ?? null,
          ownedWorkspaceAssistantSession:
            state.ownedWorkspaceAssistantSession + 1,
          ownedWorkspaceBannerDraft: draft?.bannerDraft ?? null,
          ownedWorkspaceBannerSession: state.ownedWorkspaceBannerSession + 1,
          ownedWorkspaceConflict: null,
          effectiveAssistantDefault: metadata.effectiveAssistantDefault ?? null,
          ...clearedTransientState,
          ...(outgoing
            ? {
                workspaceSnapshots: {
                  ...state.workspaceSnapshots,
                  [outgoing.workspaceId]: outgoing
                },
                savedWorkspaces: upsertSavedWorkspace(
                  state.savedWorkspaces,
                  createSavedWorkspaceEntry(
                    outgoing,
                    new Date(),
                    getSavedWorkspaceCollectionId(
                      state.savedWorkspaces,
                      state.archivedWorkspaces,
                      outgoing.workspaceId
                    )
                  )
                )
              }
            : {})
        }
      })
      return result
    },
    prepareOwnedWorkspaceArchive(input) {
      const state = get()
      if (!isCurrent(state, input)) return false
      const draft = captureOwnedWorkspaceDraft(state)
      if (!draft) return false
      // Archive admission requires a fresh durable write, even without new typing.
      const saved = drafts.save(draft, true)
      set({ ownedWorkspaceDraftStatus: saved.status })
      return saved.status === "saved"
    },
    completeOwnedWorkspaceArchive(input) {
      const state = get()
      if (!isCurrent(state, input)) return false
      validateOwnedWorkspaceRecord(input.workspace, input.workspaceId)
      if (!input.workspace.archived)
        throw new Error("Archive completion requires an archived workspace")
      if (
        input.workspace.version < state.ownedWorkspaceBundle!.workspace.version
      )
        return false
      if (!get().prepareOwnedWorkspaceArchive(input)) return false
      // The wrapped setter saves the outgoing draft before clearing the active view.
      set(clearedOwnedView())
      return true
    },
    invalidateOwnedWorkspace() {
      const current = get()
      if (
        current.activeWorkspaceOrigin.kind === "legacy-local" &&
        !current.ownedWorkspaceAttempt &&
        !current.ownedWorkspaceConflict
      )
        return
      set((state) =>
        state.activeWorkspaceOrigin.kind === "server-owned"
          ? clearedOwnedView()
          : { ownedWorkspaceAttempt: null, ownedWorkspaceConflict: null }
      )
    },
    setOwnedWorkspaceComposer(value) {
      if (get().activeWorkspaceOrigin.kind !== "server-owned")
        throw new Error("No active server-owned workspace")
      set({ ownedWorkspaceComposer: value })
    }
  }
}
