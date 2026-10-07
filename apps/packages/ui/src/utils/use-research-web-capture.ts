import { useEffect, useRef, useState } from "react"
import { useWorkspaceStore } from "@/store/workspace"
import { mapServerSourceReviewFields } from "@/store/workspace-api"
import { isWorkspaceSourceSelectable } from "@/store/workspace-source-status"
import type { WorkspaceSource } from "@/types/workspace"
import { tldwMedia } from "@/services/tldw/TldwMedia"
import { tldwClient } from "@/services/tldw/TldwApiClient"
import { loadServicePromptSnapshot } from "@/services/service-prompts"
import { watchChatAccountChanges } from "@/services/chat-account-boundary"
import {
  getResearchWorkspaceOwner,
  readResearchWebCaptures,
  saveResearchWebCapture,
  type ResearchWebCapture
} from "./research-workspace-prefill"
import {
  prepareWebCaptureAcceptance,
  confirmWebCaptureAcceptance
} from "./research-web-capture"

type Preview = { title: string; text: string; capturedAt: string }
type Session = {
  source: WorkspaceSource
  controller: AbortController
  ready: Promise<void>
  scope?: Awaited<ReturnType<typeof loadServicePromptSnapshot>>
  owner?: string
  pending?: ResearchWebCapture
  preview?: Preview
  busy: boolean
  selection: string
  manualSelection: boolean
  stop: () => void
}
const selectionKey = () => {
  const state = useWorkspaceStore.getState()
  return JSON.stringify([
    state.selectedSourceIds,
    state.selectedSourceFolderIds
  ])
}

/** Page-owned explicit capture. Effects only retire work; acquisition always follows a click. */
export function useResearchWebCapture(workspaceId: string | null) {
  const [source, setSource] = useState<WorkspaceSource | null>(null)
  const [preview, setPreview] = useState<Preview | null>(null)
  const [pending, setPending] = useState<ResearchWebCapture | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [notice, setNotice] = useState<string | null>(null)
  const active = useRef<Session | null>(null)
  const retire = () => {
    const session = active.current
    active.current = null
    session?.controller.abort()
    session?.scope?.release()
    session?.stop()
  }
  useEffect(() => {
    const stop = watchChatAccountChanges((changed) => {
      if (changed) {
        retire()
        setSource(null)
        setPreview(null)
        setPending(null)
        setBusy(false)
      }
    })
    return () => {
      retire()
      stop()
    }
  }, [workspaceId])
  const assertCurrent = (session: Session) => {
    const state = useWorkspaceStore.getState()
    if (
      active.current !== session ||
      session.controller.signal.aborted ||
      session.scope?.scopeSignal.aborted ||
      session.scope?.scopeInvalidatedSignal.aborted ||
      state.workspaceId !== workspaceId ||
      !state.sources.some(
        (item) =>
          item.id === session.source.id &&
          item.mediaId === session.source.mediaId
      )
    )
      throw new Error("Capture destination changed")
  }
  const cancel = () => {
    retire()
    setSource(null)
    setPreview(null)
    setPending(null)
    setBusy(false)
    setError(null)
    setNotice(null)
  }
  const open = (item: WorkspaceSource) => {
    cancel()
    if (!workspaceId) return
    const session: Session = {
      source: item,
      controller: new AbortController(),
      ready: Promise.resolve(),
      busy: false,
      selection: selectionKey(),
      manualSelection: false,
      stop: () => {}
    }
    active.current = session
    setSource(item)
    session.stop = useWorkspaceStore.subscribe(() => {
      if (selectionKey() !== session.selection) session.manualSelection = true
      try {
        assertCurrent(session)
      } catch {
        retire()
        setSource(null)
        setPreview(null)
        setPending(null)
        setBusy(false)
      }
    })
    session.ready = (async () => {
      session.owner = await getResearchWorkspaceOwner()
      assertCurrent(session)
      session.scope = await loadServicePromptSnapshot([], {
        signal: session.controller.signal
      })
      try {
        assertCurrent(session)
      } catch (reason) {
        session.scope.release()
        throw reason
      }
      if (session.scope.scopeKey !== session.owner)
        throw new Error("Capture account changed")
      const records = await readResearchWebCaptures(session.owner, workspaceId)
      assertCurrent(session)
      session.pending = records.find(
        (record) => !record.attached && record.sourceId === item.id
      )
      setPending(session.pending || null)
    })().catch((reason) => {
      if (active.current === session)
        setError(
          reason instanceof Error ? reason.message : "Capture unavailable"
        )
      throw reason
    })
    // The explicit next action also awaits this promise; opening reports failures without an unhandled rejection.
    void session.ready.catch(() => {})
  }
  const extract = async () => {
    const session = active.current
    if (!session || session.busy) return
    session.busy = true
    setBusy(true)
    setError(null)
    setNotice(null)
    try {
      await session.ready
      assertCurrent(session)
      if (session.pending) throw new Error("Retry the accepted capture first")
      const response = await tldwMedia.extractPublicArticle(
        session.source.webCapture?.requestedUrl || session.source.url || "",
        {
          requestScope: session.scope!.requestScope,
          signal: session.scope!.scopeSignal
        }
      )
      assertCurrent(session)
      const article = response.results?.[0]
      if (
        response.results?.length !== 1 ||
        article.extraction_successful === false ||
        article.error ||
        typeof article.content !== "string" ||
        !article.content.trim()
      )
        throw new Error(
          article?.error || "No readable article text was returned"
        )
      session.preview = {
        title: article.title || session.source.title,
        text: article.content,
        capturedAt: new Date().toISOString()
      }
      setPreview(session.preview)
    } catch (reason) {
      if (active.current === session)
        setError(reason instanceof Error ? reason.message : "Capture failed")
    } finally {
      session.busy = false
      if (active.current === session) setBusy(false)
    }
  }
  const save = async () => {
    const session = active.current
    if (!session || session.busy) return
    session.busy = true
    setBusy(true)
    setError(null)
    try {
      await session.ready
      assertCurrent(session)
      if (!session.pending) {
        if (!session.preview) throw new Error("Capture article first")
        const body = await prepareWebCaptureAcceptance({
          url:
            session.source.webCapture?.requestedUrl || session.source.url || "",
          title: session.preview.title,
          text: session.preview.text,
          capturedAt: session.preview.capturedAt,
          workspaceId: workspaceId!,
          refreshOf: session.source.webCapture?.clipId
        })
        assertCurrent(session)
        if (
          session.source.webCapture?.contentSha256 ===
          body.capture_metadata?.web_capture_v1?.content_sha256
        ) {
          setNotice("Text unchanged")
          return
        }
        session.pending = {
          ownerScope: session.owner!,
          workspaceId: workspaceId!,
          sourceId: session.source.id,
          body
        }
      }
      // Durable exact acceptance precedes every mutation, including retries and late replies.
      await saveResearchWebCapture(session.pending)
      assertCurrent(session)
      setPending(session.pending)
      const options = {
        requestScope: session.scope!.requestScope,
        signal: session.scope!.scopeSignal
      }
      if (!session.pending.pin)
        await tldwClient.saveWebClip(session.pending.body, options)
      assertCurrent(session)
      const confirmed = await confirmWebCaptureAcceptance(
        session.pending.body,
        options,
        () => assertCurrent(session)
      )
      const record = { ...session.pending, pin: confirmed.pin }
      await saveResearchWebCapture(record)
      assertCurrent(session)
      const added: WorkspaceSource = {
        id: confirmed.source.id,
        mediaId: confirmed.source.media_id,
        title: confirmed.source.title,
        type: "website",
        url: confirmed.source.url || undefined,
        addedAt: new Date(confirmed.source.added_at),
        status: "ready",
        ...mapServerSourceReviewFields(confirmed.source),
        webCapture: confirmed.pin,
        knowledgeQaEvidence: session.source.knowledgeQaEvidence
      }
      const select =
        !session.manualSelection &&
        selectionKey() === session.selection &&
        isWorkspaceSourceSelectable(added)
      useWorkspaceStore.setState((state) => ({
        sources: state.sources.some((item) => item.id === added.id)
          ? state.sources
          : [...state.sources, added],
        ...(select
          ? {
              selectedSourceIds: [
                ...new Set([...state.selectedSourceIds, added.id])
              ]
            }
          : {})
      }))
      await saveResearchWebCapture({ ...record, attached: true })
      assertCurrent(session)
      cancel()
    } catch (reason) {
      if (active.current === session)
        setError(
          reason instanceof Error
            ? reason.message
            : "Capture could not be confirmed; retry capture"
        )
    } finally {
      session.busy = false
      if (active.current === session) setBusy(false)
    }
  }
  return {
    source,
    preview,
    pending,
    busy,
    error,
    notice,
    open,
    extract,
    save,
    cancel
  }
}
