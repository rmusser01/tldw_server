import { useCallback, useEffect, useRef, useState } from "react"
import { createWorkspaceStorage, useWorkspaceStore } from "@/store/workspace"
import { isWorkspaceSourceSelectable } from "@/store/workspace-source-status"
import { WORKSPACE_STORAGE_KEY } from "@/store/workspace-events"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { resolveServicePromptScope } from "@/services/service-prompts"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import { buildChatSurfaceScopeKeyFromConfig } from "@/services/chat-surface-scope"
import { extractCompletedIngestJobMediaId } from "@/services/tldw/ingest-job-results"
import {
  buildKnowledgeQaSeedNote,
  consumeResearchWorkspacePrefill,
  getResearchWorkspaceOwner,
  saveResearchWorkspacePrefill,
  type ResearchWorkspacePrefill,
  type WorkspaceKnowledgeQaPrefillSource,
} from "./research-workspace-prefill"

const emptyStatus = {
  attached: 0,
  pending: 0,
  failed: 0,
  importing: false,
  error: null as string | null,
}

/** Retain the handoff and successful snapshot IDs until every source can attach. */
export function useResearchWorkspacePrefill(
  workspaceId: string | null,
  hydrated: boolean,
) {
  const [status, setStatus] = useState(emptyStatus)
  const [attempt, setAttempt] = useState(0)
  const retained = useRef<ResearchWorkspacePrefill | null>(null)
  const retry = useCallback(async () => {
    setAttempt((value) => value + 1)
  }, [])

  useEffect(() => {
    if (!workspaceId || !hydrated) return
    let active = true
    let stopSelection = () => {}
    const controller = new AbortController()
    const isCurrent = () =>
      active &&
      !controller.signal.aborted &&
      useWorkspaceStore.getState().workspaceId === workspaceId
    const assertCurrent = () => {
      if (!isCurrent()) throw new Error("Import destination changed")
    }
    const stop = watchChatAccountChanges((changed) => {
      if (!changed) return
      active = false
      controller.abort()
      retained.current = null
      setStatus(emptyStatus)
      setAttempt((value) => value + 1)
    })
    setStatus(emptyStatus)
    const run = async () => {
      try {
        const owner = await getResearchWorkspaceOwner()
        assertCurrent()
        const cached = retained.current
        const payload =
          cached?.ownerScope === owner && !cached.completed
            ? cached
            : await consumeResearchWorkspacePrefill(owner, true)
        assertCurrent()
        if (!payload) return
        if (payload.workspaceId && payload.workspaceId !== workspaceId) {
          if (
            payload.draftRetained &&
            payload.sources.every(
              (source) =>
                !source.importError &&
                (source.mediaId != null || source.snapshotMediaId != null),
            )
          )
            return
          setStatus({
            ...emptyStatus,
            error:
              "Unfinished Knowledge imports belong to another workspace. Reopen that workspace to retry.",
          })
          return
        }
        retained.current = payload
        payload.workspaceId = workspaceId
        if (!payload.completed) {
          // Clear both direct and folder-expanded scope, retaining the library.
          const state = useWorkspaceStore.getState()
          state.setSelectedSourceIds([])
          for (const id of state.selectedSourceFolderIds)
            state.toggleSourceFolderSelection(id)
          payload.selectionIntent = { mediaIds: [], selectedSourceIds: [] }
        }
        const checkpointSelection = () => {
          void saveResearchWorkspacePrefill(payload).catch(() => {
            if (isCurrent())
              setStatus((previous) => ({
                ...previous,
                error:
                  "Research selection could not be saved. Retry unfinished imports.",
              }))
          })
        }
        const syncImportedSelection = () => {
          const intent = payload.selectionIntent
          if (!isCurrent() || !intent) return
          const current = useWorkspaceStore.getState()
          // Direct and folder choices both supersede unfinished automatic selection.
          if (
            current.selectedSourceFolderIds.length > 0 ||
            current.selectedSourceIds.length !==
              intent.selectedSourceIds.length ||
            current.selectedSourceIds.some(
              (id, index) => id !== intent.selectedSourceIds[index],
            )
          ) {
            payload.selectionIntent = null
            checkpointSelection()
            return
          }
          const imported = current.sources.filter((source) =>
            intent.mediaIds.includes(source.mediaId),
          )
          const readyIds = imported
            .filter(isWorkspaceSourceSelectable)
            .map((source) => source.id)
          const selectionChanged =
            readyIds.length !== intent.selectedSourceIds.length ||
            readyIds.some((id, index) => id !== intent.selectedSourceIds[index])
          if (selectionChanged) {
            intent.selectedSourceIds = readyIds
            current.setSelectedSourceIds(readyIds)
          }
          // Missing sources were deliberately removed; never restore them here.
          if (
            payload.completed &&
            imported.every(isWorkspaceSourceSelectable)
          ) {
            payload.selectionIntent = null
            checkpointSelection()
          } else if (selectionChanged) checkpointSelection()
        }
        // Resume only the unfinished selection after completion; attachments and the
        // retained draft must never replay on remount or workspace switches.
        stopSelection = useWorkspaceStore.subscribe(syncImportedSelection)
        syncImportedSelection()
        if (payload.completed) return
        await saveResearchWorkspacePrefill(payload)
        assertCurrent()
        const requestScope = await resolveServicePromptScope({
          signal: controller.signal,
        })
        assertCurrent()
        if (
          buildChatSurfaceScopeKeyFromConfig(
            {
              ...requestScope.config,
              apiKey: undefined,
            },
            { userId: requestScope.userId },
          ) !== owner
        )
          throw new Error("Import account changed")
        const groups = new Map<string, WorkspaceKnowledgeQaPrefillSource[]>()
        for (const source of payload.sources) {
          const key = JSON.stringify([
            source.sourceType,
            source.mediaId ?? source.originalId,
            source.url,
          ])
          const group = groups.get(key) || []
          group.push(source)
          groups.set(key, group)
        }
        let attached = 0
        let failed = 0
        setStatus({ ...emptyStatus, pending: groups.size, importing: true })
        for (const sources of groups.values()) {
          assertCurrent()
          const source = sources[0]
          let mediaId = source.mediaId ?? source.snapshotMediaId
          if (mediaId == null) {
            try {
              if (!sources.some((item) => item.excerpt.trim()))
                throw new Error("No retrieved excerpt to import")
              const text = buildKnowledgeQaSeedNote({
                ...payload,
                answer: null,
                sources,
              })
              const file = new File(
                [text],
                `${source.title} - retrieved excerpts.txt`,
                { type: "text/plain" },
              )
              const result = await tldwClient.uploadMedia(
                file,
                {
                  media_type: "document",
                  title: `${source.title} — retrieved excerpts`,
                  overwrite: false,
                  perform_analysis: false,
                  perform_chunking: true,
                  generate_embeddings: true,
                  embedding_dispatch_mode: "background",
                },
                { requestScope, signal: controller.signal, assertCurrent },
              )
              const id = Number(
                extractCompletedIngestJobMediaId(result) ?? result?.id,
              )
              if (!Number.isSafeInteger(id) || id <= 0)
                throw new Error("No saved snapshot returned")
              mediaId = id
              for (const item of sources) {
                item.snapshotMediaId = id
                delete item.importError
              }
              // Checkpoint under the original owner/destination even if navigation just
              // changed; the completed upload must not be repeated on return.
              await saveResearchWorkspacePrefill(payload)
              assertCurrent()
            } catch (error) {
              assertCurrent()
              failed += 1
              for (const item of sources)
                item.importError =
                  "Could not import retrieved excerpts. Retry when the server is available."
              await saveResearchWorkspacePrefill(payload)
              assertCurrent()
              setStatus({
                attached,
                failed,
                pending: groups.size - attached - failed,
                importing: true,
                error: null,
              })
              continue
            }
          }
          assertCurrent()
          const state = useWorkspaceStore.getState()
          if (!state.sources.some((item) => item.mediaId === mediaId)) {
            state.addSources([
              {
                mediaId,
                title:
                  source.mediaId == null
                    ? `${source.title} — retrieved excerpts`
                    : source.title,
                type: source.mediaId == null ? "text" : source.type,
                ...(source.mediaId == null
                  ? { status: "processing" as const }
                  : {}),
                url: source.url,
                knowledgeQaEvidence: {
                  importId: payload.id,
                  threadId: payload.threadId,
                  sources,
                  trustState: payload.answerTrustState,
                  trustReasonCodes: payload.answerTrustReasonCodes,
                  evidenceOrigin: payload.answerEvidenceOrigin,
                  scope: payload.scope,
                  snapshot: source.mediaId == null,
                },
              },
            ])
          }
          if (
            payload.selectionIntent &&
            !payload.selectionIntent.mediaIds.includes(mediaId)
          ) {
            payload.selectionIntent.mediaIds.push(mediaId)
          }
          syncImportedSelection()
          attached += 1
          setStatus({
            attached,
            failed,
            pending: groups.size - attached - failed,
            importing: true,
            error: null,
          })
        }
        assertCurrent()
        const state = useWorkspaceStore.getState()
        if (
          !payload.draftRetained &&
          !state.currentNote.content.includes(`Import reference: ${payload.id}`)
        ) {
          state.captureToCurrentNote({
            title:
              payload.threadId || payload.query || payload.answer
                ? `Knowledge QA: ${payload.query.slice(0, 80) || "import"}`
                : "Reviewed sources",
            content: buildKnowledgeQaSeedNote(payload),
            mode: "append",
          })
        }
        const stored = await createWorkspaceStorage().getItem(
          WORKSPACE_STORAGE_KEY,
        )
        assertCurrent()
        const snapshot = stored
          ? JSON.parse(stored).state?.workspaceSnapshots?.[workspaceId]
          : null
        const draftPersisted =
          payload.draftRetained ||
          snapshot?.currentNote?.content?.includes(
            `Import reference: ${payload.id}`,
          )
        const sourcesPersisted = useWorkspaceStore
          .getState()
          .sources.every(
            (source) =>
              source.knowledgeQaEvidence?.importId !== payload.id ||
              snapshot?.sources?.some(
                (saved: { mediaId: number }) =>
                  saved.mediaId === source.mediaId,
              ),
          )
        if (!draftPersisted || !sourcesPersisted)
          throw new Error("Workspace persistence is unavailable")
        payload.draftRetained = true
        payload.completed = failed === 0
        syncImportedSelection()
        await saveResearchWorkspacePrefill(payload)
        assertCurrent()
        setStatus({
          attached,
          failed,
          pending: 0,
          importing: false,
          error: null,
        })
      } catch {
        if (isCurrent())
          setStatus((previous) => ({
            ...previous,
            importing: false,
            error:
              "Knowledge import could not be saved. Your pending evidence is retained; retry unfinished imports.",
          }))
      }
    }
    void run()
    return () => {
      active = false
      controller.abort()
      stopSelection()
      stop()
    }
  }, [workspaceId, hydrated, attempt])
  return { ...status, retry }
}
