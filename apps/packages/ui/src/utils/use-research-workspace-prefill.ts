import { normalizeNoteKeyword } from "@/services/note-keywords"
import { bgRequest } from "@/services/background-proxy"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import {
  retainKnowledgeNoteProvenance,
  resolveKnowledgeNoteProvenance,
  knowledgeNoteHead,
  knowledgeNoteProvenanceMatches,
  knowledgeNoteWriteFields,
  type KnowledgeNoteHead,
  validateKnowledgeNoteProvenance,
  stripKnowledgeNoteProvenance,
} from "./knowledge-note-provenance"
import { useCallback, useEffect, useRef, useState } from "react"
import {
  createWorkspaceStorage,
  hasResearchWorkspaceMigrationTombstone,
  useWorkspaceStore,
} from "@/store/workspace"
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
  serverBacked = false,
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
              const isNote = /note/i.test(source.sourceType || "")
              let fullNote: { content: string; version: number } | null = null
              if (isNote) {
                if (source.originalId == null)
                  throw new Error("Missing original note identity")
                const note = await bgRequest<{
                  id: string
                  content: string
                  version: number
                  deleted?: boolean
                }>({
                  ...requestScopeFields(requestScope),
                  abortSignal: controller.signal,
                  path: `/api/v1/notes/${encodeURIComponent(String(source.originalId))}`,
                  method: "GET",
                })
                assertCurrent()
                if (
                  note?.id !== String(source.originalId) ||
                  note.deleted === true ||
                  typeof note.content !== "string" ||
                  !note.content.length ||
                  note.content.length > 5_000_000 ||
                  !Number.isSafeInteger(note.version) ||
                  note.version <= 0
                )
                  throw new Error(
                    "Original note content or revision is unavailable",
                  )
                fullNote = {
                  content: stripKnowledgeNoteProvenance(note.content),
                  version: note.version,
                }
              } else if (!sources.some((item) => item.excerpt.trim())) {
                throw new Error("No retrieved excerpt to import")
              }
              const snapshotLabel = fullNote
                ? `full note snapshot (v${fullNote.version})`
                : "retrieved excerpts"
              const evidence = buildKnowledgeQaSeedNote({
                ...payload,
                answer: null,
                sources: fullNote
                  ? sources.map((item) => ({
                      ...item,
                      originalVersion: fullNote.version,
                    }))
                  : sources,
              })
              const text = fullNote
                ? `${evidence}\n\nFull note content (version ${fullNote.version}):\n${fullNote.content}`
                : evidence
              const file = new File(
                [text],
                `${source.title} - ${snapshotLabel}.txt`,
                {
                  type: "text/plain",
                },
              )
              const result = await tldwClient.uploadMedia(
                file,
                {
                  media_type: "document",
                  title: `${source.title} — ${snapshotLabel}`,
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
                if (fullNote) item.originalVersion = fullNote.version
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
                item.importError = /note/i.test(item.sourceType || "")
                  ? "Could not import the full note. Retry when the server is available."
                  : "Could not import retrieved excerpts. Retry when the server is available."
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
          // Snapshot ingestion can reuse media also returned by retrieval.
          // ponytail: scans each handoff per attachment; index if large batches slow it.
          const evidenceSources = payload.sources.filter(
            (item) => (item.mediaId ?? item.snapshotMediaId) === mediaId,
          )
          const knowledgeQaEvidence = {
            importId: payload.id,
            threadId: payload.threadId,
            sources: evidenceSources,
            trustState: payload.answerTrustState,
            trustReasonCodes: payload.answerTrustReasonCodes,
            evidenceOrigin: payload.answerEvidenceOrigin,
            scope: payload.scope,
            snapshot: evidenceSources.some((item) => item.mediaId == null),
          }
          if (!state.sources.some((item) => item.mediaId === mediaId)) {
            state.addSources([
              {
                mediaId,
                title:
                  source.mediaId == null
                    ? source.originalVersion != null
                      ? `${source.title} — full note snapshot (v${source.originalVersion})`
                      : `${source.title} — retrieved excerpts`
                    : source.title,
                type: source.mediaId == null ? "text" : source.type,
                ...(source.mediaId == null
                  ? { status: "processing" as const }
                  : {}),
                url: source.url,
                knowledgeQaEvidence,
              },
            ])
          } else {
            useWorkspaceStore.setState((current) => ({
              sources: current.sources.map((item) =>
                item.mediaId === mediaId
                  ? { ...item, knowledgeQaEvidence }
                  : item,
              ),
            }))
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
        const draftWasDirty = state.currentNote.isDirty
        if (
          !payload.draftRetained &&
          (!payload.canonicalNoteId ||
            (!draftWasDirty && state.currentNote.id == null)) &&
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
        if (
          serverBacked ||
          hasResearchWorkspaceMigrationTombstone(workspaceId)
        ) {
          const current = useWorkspaceStore.getState()
          const draft = current.currentNote
          // Retain the original legacy identity across a lost response and remount.
          if (
            !payload.canonicalNoteId &&
            typeof draft.id === "number" &&
            Number.isSafeInteger(draft.id) &&
            draft.id > 0
          )
            payload.legacyNoteId = draft.id
          // Client-supplied canonical UUID makes a lost create response retryable.
          payload.canonicalNoteId ||=
            typeof draft.id === "string" ? draft.id : payload.id
          await saveResearchWorkspacePrefill(payload)
          assertCurrent()
          const convertingLegacyDraft =
            !payload.draftRetained &&
            typeof payload.legacyNoteId === "number" &&
            Number.isSafeInteger(payload.legacyNoteId) &&
            payload.legacyNoteId > 0 &&
            draft.id === payload.legacyNoteId
          let draftDiscarded = Boolean(
            !convertingLegacyDraft &&
            (payload.draftRetained || draft.id != null) &&
            draft.id !== payload.canonicalNoteId,
          )
          const stopWatchingDraft = useWorkspaceStore.subscribe((next) => {
            if (
              !next.currentNote.id &&
              !next.currentNote.title &&
              !next.currentNote.content
            )
              draftDiscarded = true
          })
          try {
            const request = {
              ...requestScopeFields(requestScope),
              abortSignal: controller.signal,
            }
            const path =
              `/api/v1/notes/${encodeURIComponent(payload.canonicalNoteId)}` as const
            type CanonicalNote = KnowledgeNoteHead & {
              id: string
              title: string
              content: string
              keywords?: unknown[]
              version: number
            }
            let existing: CanonicalNote | null = null
            try {
              existing = await bgRequest<CanonicalNote>({
                ...request,
                path,
                method: "GET",
              })
            } catch (error) {
              if ((error as { status?: number })?.status !== 404) throw error
            }
            assertCurrent()
            if (existing && existing.id !== payload.canonicalNoteId)
              throw new Error("Canonical note identity changed")
            const alreadyRetained =
              existing?.content.includes(`Import reference: ${payload.id}`) ||
              resolveKnowledgeNoteProvenance(existing).provenance?.research
                ?.import_id === payload.id
            // The owned dirty draft is authoritative even when a prior attempt
            // reached the server. Capture-to-note itself must not count as an edit.
            const preferDraft =
              !alreadyRetained || (!draftDiscarded && draftWasDirty)
            const body = preferDraft ? draft.content : existing!.content
            if (existing?.knowledge_provenance_state === "deleted")
              throw new Error("Source history was removed. Restore it explicitly before retaining this import.")
            const requiredProvenance = validateKnowledgeNoteProvenance({
              origin:
                payload.threadId || payload.query || payload.answer
                  ? "knowledge_qa"
                  : "reviewed_sources",
              trust_state: payload.answerTrustState,
              evidence_origin: payload.answerEvidenceOrigin,
              thread_id: payload.threadId,
              question: payload.query,
              scope: payload.scope,
              trust_reason_codes: payload.answerTrustReasonCodes,
              sources: payload.sources.map(({ importError: _error, ...source }) => source),
              research: {
                workspace_id: workspaceId,
                import_id: payload.id,
                sources: current.sources
                  .filter((source) => source.knowledgeQaEvidence)
                  .map((source) => ({
                    mediaId: source.mediaId,
                    evidence: source.knowledgeQaEvidence,
                  })),
              },
            })
            // A required checkpoint cannot fall back to an older note marker.
            if (!requiredProvenance)
              throw new Error("Current research provenance is invalid")
            // Old servers have no receipt API; their owned exact GET acknowledges the stable client UUID.
            if (existing && alreadyRetained && !["active", "absent", "deleted"].includes(existing.knowledge_provenance_state || "")) delete payload.pendingNoteWrite
            const content = retainKnowledgeNoteProvenance(
              body,
              requiredProvenance,
            )
            if (!payload.pendingNoteWrite) payload.pendingNoteWrite = {
              idempotencyKey: crypto.randomUUID(),
              method: existing ? "PUT" : "POST",
              expectedVersion: existing?.version,
              body: {
                ...(existing ? {} : { id: payload.canonicalNoteId }),
                ...knowledgeNoteWriteFields(content, existing, { create: !existing, replacement: requiredProvenance }),
                title:
                  (preferDraft ? draft.title : existing?.title) ||
                  "Knowledge research",
                content,
                keywords: [
                  ...new Set(
                    [
                      ...(preferDraft
                        ? draft.keywords
                        : existing?.keywords || []
                      )
                        .map(normalizeNoteKeyword)
                        .filter(
                          (keyword): keyword is string => keyword !== null,
                        ),
                      current.workspaceTag,
                      `workspace:${workspaceId}`,
                    ].filter(Boolean),
                  ),
                ],
                ...(payload.threadId && !payload.threadId.startsWith("shared-")
                  ? { conversation_id: payload.threadId }
                  : {}),
              },
            }
            await saveResearchWorkspacePrefill(payload)
            assertCurrent()
            const pending = payload.pendingNoteWrite
            let saved: CanonicalNote
            try {
              saved = await bgRequest<CanonicalNote>({
                ...request, path: pending.method === "POST" ? "/api/v1/notes/" : path,
                method: pending.method,
                headers: { ...request.headers, "Idempotency-Key": pending.idempotencyKey,
                  ...(pending.expectedVersion != null ? { "expected-version": String(pending.expectedVersion) } : {}),
                },
                body: pending.body,
              })
            } catch (error) {
              if ((error as { status?: number })?.status === 409) {
                delete payload.pendingNoteWrite
                await saveResearchWorkspacePrefill(payload)
              }
              throw error
            }
            assertCurrent()
            const confirmed = await bgRequest<CanonicalNote>({
              ...request,
              path,
              method: "GET",
            })
            assertCurrent()
            const keywords = (confirmed.keywords || [])
              .map(normalizeNoteKeyword)
              .filter((keyword): keyword is string => keyword !== null)
            const provenance = resolveKnowledgeNoteProvenance(confirmed).provenance
            if (
              saved.id !== payload.canonicalNoteId ||
              confirmed.id !== payload.canonicalNoteId ||
              (confirmed.version <= saved.version && stripKnowledgeNoteProvenance(confirmed.content) !== stripKnowledgeNoteProvenance(String(pending.body.content))) ||
              !keywords.includes(`workspace:${workspaceId}`) ||
              provenance?.research?.import_id !== payload.id
            )
              throw new Error("Canonical research note was not retained")
            delete payload.pendingNoteWrite
            await saveResearchWorkspacePrefill(payload)
            assertCurrent()
            // A receipt only acknowledges its original source snapshot. Preserve the
            // unfinished handoff so a deliberate retry can save newly attached sources.
            if (!knowledgeNoteProvenanceMatches(provenance, requiredProvenance))
              throw new Error("Earlier save confirmed; retry to retain the remaining source history.")
            // Do not replace edits made while canonical persistence was pending.
            const latest = useWorkspaceStore.getState().currentNote
            if (!draftDiscarded && latest === draft && (!draftWasDirty || (draft.title === pending.body.title && stripKnowledgeNoteProvenance(draft.content) === stripKnowledgeNoteProvenance(String(pending.body.content)))))
              current.loadNote({
                ...confirmed,
                keywords: keywords.filter(
                  (keyword) =>
                    keyword !== current.workspaceTag &&
                    keyword !== `workspace:${workspaceId}`,
                ),
              })
            else if (!draftDiscarded && latest.id === draft.id)
              current.setCurrentNote({
                ...latest,
                id: confirmed.id,
                version: latest.id === confirmed.id ? Math.max(latest.version || 0, confirmed.version) : confirmed.version,
                ...knowledgeNoteHead(confirmed),
                content: retainKnowledgeNoteProvenance(
                  latest.content,
                  confirmed,
                ),
              })
          } finally {
            stopWatchingDraft()
          }
        } else {
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
        }
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
  }, [workspaceId, hydrated, serverBacked, attempt])
  return { ...status, retry }
}
