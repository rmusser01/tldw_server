import { useCallback, useEffect, useMemo, useRef, useState } from "react"
import { useWorkspaceStore } from "@/store/workspace"
import { createOwnedWorkspaceNotesContext } from "@/services/owned-workspace-opening"
import type { WorkspaceNoteApiResponse } from "@/services/tldw/domains/workspace-api"
import { WORKSPACE_UNDO_WINDOW_MS } from "../../undo-manager"

type ActivationOperations = { locked: boolean }
// Survive panel collapse/remount without retaining a released account activation.
const activationOperations = new WeakMap<
  object,
  Map<string, ActivationOperations>
>()

export const ownedNoteForEditor = (note: WorkspaceNoteApiResponse) => ({
  id: note.id,
  title: note.title,
  content: note.content,
  keywords: JSON.parse(note.keywords_json) as string[],
  version: note.version
})

/** Explicit owned operations share one abortable activation lifetime. */
export function useOwnedQuickNotes() {
  const origin = useWorkspaceStore((s) => s.activeWorkspaceOrigin)
  const workspaceId = useWorkspaceStore((s) => s.workspaceId)
  const bundle = useWorkspaceStore((s) => s.ownedWorkspaceBundle)
  const editorSession = useWorkspaceStore((s) => s.ownedNoteEditorSession)
  const enabled = origin?.kind === "server-owned"
  const lifetime = useRef<AbortController | null>(null)
  const operations = useMemo(() => {
    if (!enabled) return null
    let workspaces = activationOperations.get(origin)
    if (!workspaces) {
      workspaces = new Map()
      activationOperations.set(origin, workspaces)
    }
    let current = workspaces.get(workspaceId)
    if (!current) {
      current = { locked: false }
      workspaces.set(workspaceId, current)
    }
    return current
  }, [enabled, origin, workspaceId])
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    const controller = new AbortController()
    lifetime.current = controller
    const unsubscribe = useWorkspaceStore.subscribe((state) => {
      if (
        state.activeWorkspaceOrigin !== origin ||
        state.workspaceId !== workspaceId ||
        state.ownedWorkspaceAttempt
      )
        controller.abort()
    })
    return () => {
      controller.abort()
      unsubscribe()
    }
  }, [origin, workspaceId])

  const isCurrent = useCallback(() => {
    const state = useWorkspaceStore.getState()
    return (
      !!lifetime.current &&
      !lifetime.current.signal.aborted &&
      state.activeWorkspaceOrigin === origin &&
      state.workspaceId === workspaceId &&
      !state.ownedWorkspaceAttempt
    )
  }, [origin, workspaceId])

  const client = async () => {
    if (!isCurrent() || origin?.kind !== "server-owned")
      throw new DOMException("Workspace changed", "AbortError")
    const context = await createOwnedWorkspaceNotesContext(
      workspaceId,
      origin.scope,
      lifetime.current!.signal
    )
    if (!isCurrent()) throw new DOMException("Workspace changed", "AbortError")
    return context
  }

  const refresh = async () => {
    if (!isCurrent()) return null
    const expectedBundle = useWorkspaceStore.getState().ownedWorkspaceBundle
    if (!expectedBundle) return null
    const context = await client()
    const notes = await context.list()
    if (!isCurrent()) return null
    if (
      !useWorkspaceStore.getState().replaceOwnedWorkspaceNotes({
        origin,
        workspaceId,
        expectedBundle,
        notes
      })
    )
      throw new Error(
        "Notes changed during refresh. Refresh again to inspect the latest notes."
      )
    return notes
  }

  const save = async () => {
    if (!isCurrent() || !operations || operations.locked)
      return "stale" as const
    let submitted = useWorkspaceStore.getState().currentNote
    const session = useWorkspaceStore.getState().ownedNoteEditorSession
    if (!submitted.id && submitted.createUncertain) return "uncertain" as const
    operations.locked = true
    setSaving(true)
    let createDispatched = false
    try {
      const context = await client()
      if (
        submitted.id &&
        (!Number.isSafeInteger(submitted.version) || submitted.version! < 1)
      )
        throw new Error(
          "The note version is missing. Reload the latest note before saving."
        )
      if (!submitted.id) {
        const marked = useWorkspaceStore
          .getState()
          .setOwnedNoteCreateUncertain({
            origin,
            workspaceId,
            editorSession: session,
            value: true
          })
        if (!marked || !isCurrent()) return "stale" as const
        submitted = marked
      }
      const fields = {
        title: submitted.title || "Untitled Note",
        content: submitted.content
      }
      createDispatched = !submitted.id
      const note = submitted.id
        ? await context.update(submitted.id, {
            ...fields,
            keywords_json: JSON.stringify(submitted.keywords),
            version: submitted.version!
          })
        : await context.create({ ...fields, keywords: [...submitted.keywords] })
      if (!isCurrent()) return "stale" as const
      return useWorkspaceStore.getState().applyOwnedWorkspaceNoteReceipt({
        origin,
        workspaceId,
        editorSession: session,
        submitted,
        note
      })
    } catch (error) {
      if (createDispatched) {
        const status = (error as { status?: number })?.status
        if (
          [400, 401, 403, 404, 405, 409, 412, 413, 415, 422, 429].includes(
            status
          )
        ) {
          if (isCurrent())
            useWorkspaceStore.getState().setOwnedNoteCreateUncertain({
              origin,
              workspaceId,
              editorSession: session,
              value: false
            })
        } else return "uncertain" as const
      }
      throw error
    } finally {
      operations.locked = false
      if (isCurrent()) setSaving(false)
    }
  }

  const recover = async (noteId: number, session: number) => {
    if (
      !isCurrent() ||
      useWorkspaceStore.getState().ownedNoteEditorSession !== session
    )
      return false
    const notes = await refresh()
    if (!notes || !isCurrent()) return false
    const state = useWorkspaceStore.getState()
    if (
      state.ownedNoteEditorSession !== session ||
      state.currentNote.id !== noteId
    )
      return false
    const latest = notes.find((note) => note.id === noteId)
    if (!latest)
      throw new Error(
        "This note is no longer associated with the workspace. Your local draft is unchanged."
      )
    const local = state.currentNote
    const remote = ownedNoteForEditor(latest)
    const titleChanged = local.title !== remote.title
    const remoteContent = titleChanged
      ? `# ${remote.title}\n\n${remote.content}`
      : remote.content
    const localContent = titleChanged
      ? `# ${local.title}\n\n${local.content}`
      : local.content
    // Preserve both revisions verbatim for explicit review, including empty/whitespace edits.
    const content =
      remoteContent === localContent
        ? local.content
        : `${remoteContent}\n\n---\n\n## Local Draft (Unsaved)\n\n${localContent}`
    state.setCurrentNote({
      ...local,
      content,
      keywords: [...new Set([...remote.keywords, ...local.keywords])],
      version: latest.version,
      isDirty: true
    })
    return true
  }

  const clear = () => {
    if (!isCurrent()) return null
    const previous = useWorkspaceStore.getState().currentNote
    useWorkspaceStore.getState().clearCurrentNote()
    const cleared = useWorkspaceStore.getState().currentNote
    const session = useWorkspaceStore.getState().ownedNoteEditorSession
    const expires = Date.now() + WORKSPACE_UNDO_WINDOW_MS
    let used = false
    return () => {
      const state = useWorkspaceStore.getState()
      if (
        used ||
        Date.now() >= expires ||
        !isCurrent() ||
        state.ownedNoteEditorSession !== session ||
        state.currentNote !== cleared
      )
        return false
      used = true
      state.setCurrentNote(previous)
      return true
    }
  }

  return {
    enabled,
    bundle,
    editorSession,
    saving,
    isCurrent,
    save,
    refresh,
    recover,
    clear
  }
}
