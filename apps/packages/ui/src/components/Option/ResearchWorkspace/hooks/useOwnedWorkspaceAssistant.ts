import { useEffect, useMemo, useReducer } from "react"
import { useWorkspaceStore } from "@/store/workspace"
import type { OwnedWorkspaceAssistantDraft } from "@/store/owned-workspace-state"
import {
  createOwnedWorkspaceMetadataContext,
  type OwnedWorkspacePersonaOption
} from "@/services/owned-workspace-opening"
import type { WorkspaceApiResponse } from "@/services/tldw/domains/workspace-api"

type AssistantError =
  | "conflict"
  | "accountChanged"
  | "loadFailed"
  | "saveFailed"
  | "reviewFailed"
type Review = {
  workspace: WorkspaceApiResponse
  expectedWorkspace: WorkspaceApiResponse
}
type Editor = {
  session: number
  pending: "load" | "save" | "review" | null
  error: AssistantError | null
  workspace: WorkspaceApiResponse | null
  personas: OwnedWorkspacePersonaOption[]
  catalogError: boolean
  review: Review | null
  consent: OwnedWorkspaceAssistantDraft | null
}

/** Persist settings, but keep catalog, consent and request identity within this modal. */
export function useOwnedWorkspaceAssistant() {
  const origin = useWorkspaceStore((s) => s.activeWorkspaceOrigin)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const attempt = useWorkspaceStore((s) => s.ownedWorkspaceAttempt)
  const draft = useWorkspaceStore((s) => s.ownedWorkspaceAssistantDraft)
  const session = useWorkspaceStore((s) => s.ownedWorkspaceAssistantSession)
  const storageStatus = useWorkspaceStore((s) => s.ownedWorkspaceDraftStatus)
  const enabled = origin?.kind === "server-owned"
  const [, render] = useReducer((n: number) => n + 1, 0)
  const lifetime = useMemo(
    () => ({
      origin,
      workspaceId,
      attempt,
      controller: new AbortController(),
      editor: null as Editor | null
    }),
    [origin, workspaceId, attempt]
  )

  useEffect(() => {
    if (!enabled) return
    if (lifetime.controller.signal.aborted)
      lifetime.controller = new AbortController()
    const controller = lifetime.controller
    const unsubscribe = useWorkspaceStore.subscribe((state) => {
      if (
        state.activeWorkspaceOrigin !== origin ||
        state.workspaceId !== workspaceId ||
        state.ownedWorkspaceAttempt !== attempt
      ) {
        controller.abort()
        lifetime.editor = null
      }
    })
    return () => {
      controller.abort()
      lifetime.editor = null
      unsubscribe()
    }
  }, [enabled, lifetime, origin, workspaceId, attempt])

  const current = () => {
    const state = useWorkspaceStore.getState()
    return (
      enabled &&
      !lifetime.controller.signal.aborted &&
      !lifetime.attempt &&
      state.activeWorkspaceOrigin === origin &&
      state.workspaceId === workspaceId &&
      state.ownedWorkspaceAttempt === attempt
    )
  }
  const editor = lifetime.editor
  const editorCurrent = (candidate: Editor | null): candidate is Editor =>
    !!candidate &&
    current() &&
    lifetime.editor === candidate &&
    useWorkspaceStore.getState().ownedWorkspaceAssistantSession ===
      candidate.session
  const updateDraft = (
    candidate: Editor,
    next: OwnedWorkspaceAssistantDraft | null
  ) => {
    if (!editorCurrent(candidate)) return false
    const state = useWorkspaceStore.getState()
    return state.setOwnedWorkspaceAssistantDraft({
      origin,
      workspaceId,
      expectedDraft: state.ownedWorkspaceAssistantDraft,
      draft: next
    })
  }
  const report = (candidate: Editor, error: AssistantError | null) => {
    if (!editorCurrent(candidate)) return
    candidate.error = error
    if (error) candidate.review = null
    candidate.consent = null
    render()
  }
  const close = (candidate: Editor) => {
    if (lifetime.editor !== candidate) return
    candidate.consent = null
    lifetime.editor = null
    render()
  }

  const load = async (candidate: Editor, inspect: boolean) => {
    if (
      !editorCurrent(candidate) ||
      candidate.pending ||
      origin.kind !== "server-owned"
    )
      return
    const expectedWorkspace =
      useWorkspaceStore.getState().ownedWorkspaceBundle?.workspace
    if (!expectedWorkspace) return
    candidate.pending = inspect ? "review" : "load"
    candidate.consent = null
    candidate.review = null
    render()
    try {
      const context = await createOwnedWorkspaceMetadataContext(
        workspaceId,
        origin.scope,
        lifetime.controller.signal
      )
      if (!editorCurrent(candidate)) return
      const workspace = await context.get()
      if (!editorCurrent(candidate)) return
      const state = useWorkspaceStore.getState()
      const savedDraft = state.ownedWorkspaceAssistantDraft
      if (
        state.ownedWorkspaceBundle?.workspace !== expectedWorkspace ||
        workspace.version < expectedWorkspace.version ||
        (savedDraft && workspace.version < savedDraft.baseVersion)
      ) {
        report(candidate, "reviewFailed")
        return
      }
      candidate.workspace = workspace
      candidate.personas = []
      if (!savedDraft) {
        if (
          !updateDraft(candidate, {
            assistantId: workspace.assistantDefaults?.assistantId ?? "",
            personaMemoryMode:
              workspace.assistantDefaults?.personaMemoryMode ?? "read_only",
            baseVersion: workspace.version
          })
        )
          return
        candidate.session =
          useWorkspaceStore.getState().ownedWorkspaceAssistantSession
        candidate.error = null
      } else {
        // A restored editor keeps its original base; even equal-version recovery is explicit.
        candidate.review = { workspace, expectedWorkspace }
        candidate.error =
          workspace.version > savedDraft.baseVersion ? "conflict" : null
      }
      try {
        const personas = await context.listPersonas()
        if (!editorCurrent(candidate)) return
        candidate.personas = personas
        candidate.catalogError = false
      } catch (error) {
        if (!editorCurrent(candidate)) return
        candidate.catalogError = true
        if ((error as { status?: number })?.status === 412) {
          candidate.review = null
          report(candidate, "accountChanged")
        }
      }
      render()
    } catch (error) {
      report(
        candidate,
        (error as { status?: number })?.status === 412
          ? "accountChanged"
          : inspect
            ? "reviewFailed"
            : "loadFailed"
      )
    } finally {
      candidate.pending = null
      if (editorCurrent(candidate)) render()
    }
  }

  const open = async () => {
    if (!current() || editorCurrent(lifetime.editor)) return
    const candidate: Editor = {
      session: useWorkspaceStore.getState().ownedWorkspaceAssistantSession,
      pending: null,
      error: null,
      workspace: null,
      personas: [],
      catalogError: false,
      review: null,
      consent: null
    }
    lifetime.editor = candidate
    render()
    await load(candidate, false)
  }
  const cancel = () => {
    if (!editorCurrent(editor)) return
    if (updateDraft(editor, null)) close(editor)
  }
  const change = (
    fields: Partial<
      Pick<OwnedWorkspaceAssistantDraft, "assistantId" | "personaMemoryMode">
    >
  ) => {
    if (!editorCurrent(editor)) return
    const draft = useWorkspaceStore.getState().ownedWorkspaceAssistantDraft
    if (!draft) return
    editor.consent = null
    updateDraft(editor, { ...draft, ...fields })
    render()
  }
  const confirm = (checked: boolean) => {
    if (
      !editorCurrent(editor) ||
      editor.pending ||
      editor.review ||
      editor.error
    )
      return
    const draft = useWorkspaceStore.getState().ownedWorkspaceAssistantDraft
    editor.consent =
      checked && draft?.personaMemoryMode === "read_write" ? draft : null
    render()
  }
  const maySave = (
    candidate: Editor | null,
    draft: OwnedWorkspaceAssistantDraft | null,
    clear: boolean
  ) =>
    editorCurrent(candidate) &&
    !!draft &&
    !!candidate.workspace &&
    !candidate.pending &&
    !candidate.error &&
    !candidate.review &&
    (clear ||
      (!candidate.catalogError &&
        candidate.personas.some((p) => p.id === draft.assistantId) &&
        !!draft.assistantId.trim() &&
        (draft.personaMemoryMode !== "read_write" ||
          candidate.consent === draft)))

  const save = async (clear = false) => {
    const state = useWorkspaceStore.getState()
    const submittedDraft = state.ownedWorkspaceAssistantDraft
    const expectedWorkspace = state.ownedWorkspaceBundle?.workspace
    if (
      !editor ||
      !submittedDraft ||
      !expectedWorkspace ||
      !maySave(editor, submittedDraft, clear) ||
      origin.kind !== "server-owned"
    )
      return
    const editorSession = editor.session
    editor.pending = "save"
    editor.consent = null
    render()
    try {
      const context = await createOwnedWorkspaceMetadataContext(
        workspaceId,
        origin.scope,
        lifetime.controller.signal
      )
      if (
        !editorCurrent(editor) ||
        useWorkspaceStore.getState().ownedWorkspaceAssistantDraft !==
          submittedDraft
      )
        return
      const workspace = await context.patch({
        version: submittedDraft.baseVersion,
        assistantDefaults: clear
          ? null
          : {
              assistantKind: "persona",
              assistantId: submittedDraft.assistantId,
              personaMemoryMode: submittedDraft.personaMemoryMode,
              voice: null,
              style: null,
              toolPolicyProfileId: null
            },
        ...(!clear && submittedDraft.personaMemoryMode === "read_write"
          ? { confirmReadWriteAssistantDefault: true }
          : {})
      })
      if (!current()) return
      // Receipt acceptance is separate from modal identity: Cancel cannot undo a dispatched write.
      const result = useWorkspaceStore
        .getState()
        .applyOwnedWorkspaceAssistantReceipt({
          origin,
          workspaceId,
          expectedWorkspace,
          submittedDraft,
          editorSession,
          workspace
        })
      if (result === "saved") close(editor)
      else if (result === "stale") report(editor, "saveFailed")
      else if (editorCurrent(editor)) {
        editor.workspace = workspace
        render()
      }
    } catch (error) {
      const status = (error as { status?: number })?.status
      report(
        editor,
        status === 409
          ? "conflict"
          : status === 412
            ? "accountChanged"
            : "saveFailed"
      )
    } finally {
      editor.pending = null
      if (editorCurrent(editor)) render()
    }
  }

  const review = async () => {
    if (editorCurrent(editor) && editor.error !== "accountChanged")
      await load(editor, true)
  }
  const keepMine = () => {
    if (!editorCurrent(editor) || editor.pending || !editor.review) return
    const state = useWorkspaceStore.getState()
    const draft = state.ownedWorkspaceAssistantDraft
    if (
      !draft ||
      state.ownedWorkspaceBundle?.workspace !== editor.review.expectedWorkspace
    ) {
      report(editor, "reviewFailed")
      return
    }
    if (
      updateDraft(editor, {
        ...draft,
        baseVersion: editor.review.workspace.version
      })
    ) {
      editor.review = null
      report(editor, null)
    }
  }
  const useServer = () => {
    if (!editorCurrent(editor) || editor.pending || !editor.review) return
    const state = useWorkspaceStore.getState()
    const submittedDraft = state.ownedWorkspaceAssistantDraft
    if (!submittedDraft) return
    editor.consent = null
    try {
      const result = state.acceptOwnedWorkspaceAssistantReview({
        origin,
        workspaceId,
        expectedWorkspace: editor.review.expectedWorkspace,
        submittedDraft,
        editorSession: editor.session,
        workspace: editor.review.workspace
      })
      if (result === "saved") close(editor)
      else report(editor, "reviewFailed")
    } catch {
      report(editor, "reviewFailed")
    }
  }

  const visible =
    editor?.session === session && editorCurrent(editor) ? editor : null
  return {
    enabled,
    draft,
    isOpen: !!visible,
    loading: visible?.pending === "load" || visible?.pending === "review",
    saving: visible?.pending === "save",
    pending: !!visible?.pending,
    error: visible?.error ?? null,
    remote: visible?.review?.workspace ?? null,
    workspace: visible?.workspace ?? null,
    personas: visible?.personas ?? [],
    catalogError: visible?.catalogError ?? false,
    confirmed: !!draft && visible?.consent === draft,
    canSave: maySave(visible, draft, false),
    canClear: maySave(visible, draft, true),
    storageUnavailable:
      !!draft &&
      (storageStatus === "unavailable" || storageStatus === "invalid"),
    open,
    cancel,
    changePersona: (assistantId: string) => change({ assistantId }),
    changeMode: (
      personaMemoryMode: OwnedWorkspaceAssistantDraft["personaMemoryMode"]
    ) => change({ personaMemoryMode }),
    confirm,
    save,
    review,
    keepMine,
    useServer
  }
}
