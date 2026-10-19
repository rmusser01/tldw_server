import React from "react"
import { Button } from "@/components/Common/Button"
import { Alert } from "@/components/ui/primitives/Alert"
import { usePrivateWorkspaceScope, type PrivateWorkspaceScope } from "@/hooks/usePrivateWorkspaceScope"
import { requestScopeFields } from "@/services/tldw/domains/service-prompts"
import type { TldwConfig } from "@/services/tldw/TldwApiClient"
import {
  bulkReviewClaims, getClaimForReview, getClaimReviewHistory, getClaimSource,
  listClaimsForReview, reviewClaim,
  type ClaimHistoryItem, type ClaimRequestScope, type ClaimReviewItem, type ClaimReviewStatus
} from "@/services/claims-review"

const statuses: ClaimReviewStatus[] = ["pending", "approved", "rejected", "flagged", "reassigned"]
const transitions: Record<ClaimReviewStatus, ClaimReviewStatus[]> = {
  pending: ["approved", "rejected", "flagged"],
  reassigned: ["pending", "approved", "rejected", "flagged"],
  flagged: ["pending", "approved", "rejected"],
  approved: ["pending"], rejected: ["pending"]
}
const errorStatus = (error: unknown): number | undefined => {
  if (!error || typeof error !== "object") return undefined
  const value = error as { status?: number; response?: { status?: number } }
  return value.status ?? value.response?.status
}
const fieldClass = "w-full rounded-md border border-border bg-surface2 px-3 py-2 text-sm text-text focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus"
const optionalFields = (notes: string, reason: string) => ({
  ...(notes.trim() ? { notes: notes.trim() } : {}),
  ...(reason.trim() ? { reason_code: reason.trim() } : {})
})

/** Remount every private state holder when canonical server/account authority changes. */
export const ClaimsReviewShell = () => {
  const { scope, retry } = usePrivateWorkspaceScope()
  if (!scope) return <div><p role="status">Resolving Claims review connection…</p><Button onClick={retry}>Retry connection</Button></div>
  return <ClaimsReviewWorkspace key={scope.key} config={scope.config} scope={scope} />
}

const ClaimsReviewWorkspace = ({ config, scope }: { config: TldwConfig; scope: PrivateWorkspaceScope }) => {
  const [filter, setFilter] = React.useState<ClaimReviewStatus>("pending")
  const [items, setItems] = React.useState<ClaimReviewItem[]>([])
  const [nextOffset, setNextOffset] = React.useState<number | null>(null)
  const [loading, setLoading] = React.useState(true)
  const [busy, setBusy] = React.useState(false)
  const [dirtyQueue, setDirtyQueue] = React.useState(false)
  const [selected, setSelected] = React.useState<number[]>([])
  const [active, setActive] = React.useState<ClaimReviewItem | null>(null)
  const [history, setHistory] = React.useState<ClaimHistoryItem[] | null>(null)
  const [historyError, setHistoryError] = React.useState(false)
  const [notes, setNotes] = React.useState("")
  const [reason, setReason] = React.useState("")
  const [correction, setCorrection] = React.useState("")
  const [batchNotes, setBatchNotes] = React.useState("")
  const [batchReason, setBatchReason] = React.useState("")
  const [conflict, setConflict] = React.useState(false)
  const [error, setError] = React.useState("")
  const [notice, setNotice] = React.useState("")
  const [denied, setDenied] = React.useState(false)
  const [source, setSource] = React.useState<string | null>(null)
  const [sourceLoading, setSourceLoading] = React.useState(false)
  const [sourceError, setSourceError] = React.useState("")
  const configRef = React.useRef(config)
  configRef.current = config
  const generation = React.useRef(0)
  const requests = React.useRef(new Set<AbortController>())
  const queueSequence = React.useRef(0)
  const historySequence = React.useRef(0)
  const sourceSequence = React.useRef(0)
  const decisionLock = React.useRef(false)

  React.useEffect(() => () => {
    generation.current += 1
    for (const controller of requests.current) controller.abort()
    requests.current.clear()
  }, [])

  const request = React.useCallback(async <T,>(run: (scope: ClaimRequestScope) => Promise<T>) => {
    if (!scope.isCurrent()) return { current: false } as const
    const controller = new AbortController()
    const abort = () => controller.abort()
    scope.signal.addEventListener("abort", abort, { once: true })
    requests.current.add(controller)
    const owner = generation.current
    try {
      const result = await run({ configSnapshot: { ...configRef.current }, abortSignal: controller.signal,
        headers: requestScopeFields(scope.requestScope).headers })
      return { result, current: owner === generation.current && !controller.signal.aborted && scope.isCurrent() }
    } catch (cause) {
      if (owner !== generation.current || controller.signal.aborted || !scope.isCurrent()) return { current: false } as const
      throw cause
    } finally {
      requests.current.delete(controller)
      scope.signal.removeEventListener("abort", abort)
    }
  }, [scope])

  const handleError = React.useCallback((cause: unknown, fallback: string) => {
    if (errorStatus(cause) === 403 || errorStatus(cause) === 401) {
      generation.current += 1
      queueSequence.current += 1
      historySequence.current += 1
      sourceSequence.current += 1
      for (const controller of requests.current) controller.abort()
      requests.current.clear()
      setLoading(false)
      setSourceLoading(false)
      setSourceError("")
      setNextOffset(null)
      setDirtyQueue(false)
      setNotice("")
      setNotes("")
      setReason("")
      setCorrection("")
      setBatchNotes("")
      setBatchReason("")
      setDenied(true)
      setItems([])
      setActive(null)
      setHistory(null)
      setSource(null)
      setSelected([])
      setError("You do not have permission to review Claims. Admin permission is required for batch decisions.")
    } else setError(fallback)
  }, [])

  const loadQueue = React.useCallback(async (offset = 0) => {
    const sequence = ++queueSequence.current
    setLoading(true)
    setError("")
    if (offset === 0) {
      setItems([])
      setSelected([])
      setActive(null)
      setHistory(null)
      setSource(null)
      setConflict(false)
      setNotes("")
      setReason("")
      setCorrection("")
    }
    try {
      const response = await request((scope) => listClaimsForReview(filter, offset, scope))
      if (!response.current || sequence !== queueSequence.current) return
      const page = response.result!
      setItems((previous) => offset === 0 ? page.items : [...previous, ...page.items.filter((item) => !previous.some((old) => old.id === item.id))])
      setNextOffset(page.has_more ? page.next_offset : null)
      setDirtyQueue(false)
      setDenied(false)
    } catch (cause) {
      if (sequence === queueSequence.current) handleError(cause, "Could not load Claims. Refresh the queue to try again.")
    } finally {
      if (sequence === queueSequence.current) setLoading(false)
    }
  }, [filter, handleError, request])

  React.useEffect(() => { void loadQueue() }, [loadQueue])

  const loadHistory = React.useCallback(async (id: number) => {
    const sequence = ++historySequence.current
    setHistory(null)
    setHistoryError(false)
    try {
      const response = await request((scope) => getClaimReviewHistory(id, scope))
      if (response.current && sequence === historySequence.current) setHistory(response.result!)
    } catch (cause) {
      if (sequence !== historySequence.current) return
      if (errorStatus(cause) === 403 || errorStatus(cause) === 401) handleError(cause, "")
      else setHistoryError(true)
    }
  }, [handleError, request])

  const activeId = active?.id
  React.useEffect(() => {
    if (activeId !== undefined) void loadHistory(activeId)
    else historySequence.current += 1
  }, [activeId, loadHistory])

  const inspect = (item: ClaimReviewItem) => {
    sourceSequence.current += 1
    setActive(item)
    setConflict(false)
    setNotes(item.review_notes || "")
    setReason(item.review_reason_code || "")
    setCorrection("")
    setSource(null)
    setSourceError("")
    setSourceLoading(false)
  }

  const refreshClaim = async () => {
    if (!active || decisionLock.current) return
    decisionLock.current = true
    setBusy(true)
    try {
      const response = await request((scope) => getClaimForReview(active.id, scope))
      if (!response.current) return
      setActive(response.result!)
      setConflict(false)
      setError("")
      await loadHistory(active.id)
    } catch (cause) { handleError(cause, "Could not load the latest claim. Your draft is preserved.") }
    finally { decisionLock.current = false; setBusy(false) }
  }

  const decide = async (status: ClaimReviewStatus) => {
    if (!active || conflict || decisionLock.current || loading || denied || error ||
        !transitions[active.review_status]?.includes(status) || !Number.isInteger(active.review_version) || active.review_version < 1) return
    decisionLock.current = true
    setBusy(true)
    setError("")
    try {
      const response = await request((scope) => reviewClaim(active.id, {
        status, review_version: active.review_version,
        ...optionalFields(notes, reason),
        ...(correction.trim() ? { corrected_text: correction.trim() } : {})
      }, scope))
      if (!response.current) return
      const saved = response.result!
      setActive(saved)
      setItems((previous) => previous.filter((item) => item.id !== saved.id || saved.review_status === filter).map((item) => item.id === saved.id ? saved : item))
      setSelected((previous) => previous.filter((id) => id !== saved.id))
      setDirtyQueue(true)
      setNotice(`Claim ${saved.id} saved as ${saved.review_status}.`)
      setCorrection("")
      await loadHistory(saved.id)
    } catch (cause) {
      if (errorStatus(cause) === 409) {
        setConflict(true)
        setError("This claim changed since it was loaded. Load the latest claim before deciding again; your draft is preserved.")
      } else handleError(cause, "Decision was not saved. Your draft is preserved; inspect the latest claim before retrying.")
    } finally { decisionLock.current = false; setBusy(false) }
  }

  const decideBatch = async (status: ClaimReviewStatus) => {
    const ids = items.filter((item) => selected.includes(item.id)).map((item) => item.id)
    if (!ids.length || decisionLock.current || loading || denied || error) return
    decisionLock.current = true
    setBusy(true)
    setError("")
    try {
      const response = await request((scope) => bulkReviewClaims({
        claim_ids: ids, status, ...optionalFields(batchNotes, batchReason)
      }, scope))
      if (!response.current) return
      const result = response.result!
      const succeeded = result.updated.filter((id) => ids.includes(id))
      setItems((previous) => previous.filter((item) => !succeeded.includes(item.id)))
      setSelected((previous) => previous.filter((id) => !succeeded.includes(id)))
      if (active && succeeded.includes(active.id)) {
        setActive(null)
        setHistory(null)
        setSource(null)
      }
      setDirtyQueue(succeeded.length > 0)
      setNotice(`${succeeded.length} saved; ${result.conflicts.length} conflict; ${result.missing.length} missing; ${result.invalid.length} invalid. Failed selections are retained. Inspect saved decisions using their status filter.`)
    } catch (cause) { handleError(cause, "Batch decisions were not confirmed. Refresh before trying again; selections are preserved.") }
    finally { decisionLock.current = false; setBusy(false) }
  }

  const loadSource = async () => {
    if (!active) return
    const sequence = ++sourceSequence.current
    setSourceLoading(true)
    setSourceError("")
    try {
      const response = await request((scope) => getClaimSource(active.media_id, scope))
      if (response.current && sequence === sourceSequence.current) setSource(response.result?.content?.text || "")
    } catch (cause) {
      if (sequence !== sourceSequence.current) return
      if (errorStatus(cause) === 401) handleError(cause, "")
      else setSourceError("Current source text is unavailable. Check source access or open it in Media.")
    } finally { if (sequence === sourceSequence.current) setSourceLoading(false) }
  }

  const blocked = busy || loading || denied
  return <section className="space-y-5 text-text" aria-labelledby="claims-review-title" data-testid="claims-review-shell">
    <header>
      <h1 id="claims-review-title" className="text-2xl font-semibold">Claims Review</h1>
      <p className="mt-1 max-w-3xl text-sm text-text-muted">Inspect extracted claims against their source, then save a decision and reason. Saved history records each decision and reviewer.</p>
    </header>
    <div className="flex flex-wrap items-end gap-3">
      <label className="text-sm">Review status
        <select className={fieldClass} aria-label="Review status" value={filter} disabled={busy}
          onChange={(event) => {
            queueSequence.current += 1
            historySequence.current += 1
            sourceSequence.current += 1
            setLoading(true)
            setActive(null)
            setSelected([])
            setNotice("")
            setFilter(event.target.value as ClaimReviewStatus)
          }}>
          {statuses.map((status) => <option key={status} value={status}>{status}</option>)}
        </select>
      </label>
      <Button variant="outline" disabled={busy} onClick={() => void loadQueue()}>Refresh queue</Button>
      <p className="py-2 text-sm text-text-muted" role="status">{items.length} claims loaded</p>
    </div>
    {error && <Alert variant="error" role="alert">{error}</Alert>}
    {notice && <Alert variant="success" role="status">{notice}</Alert>}
    {loading && <p role="status">Loading Claims queue…</p>}
    {!loading && !denied && !error && items.length === 0 && <p>No claims in this status. Choose another status or refresh after extraction.</p>}
    <div className="grid gap-6 lg:grid-cols-2">
      <div className="min-w-0 space-y-3">
        <ul className="divide-y divide-border rounded-lg border border-border bg-surface">
          {items.map((item) => <li key={item.id} className="flex items-start gap-3 p-3">
            <input type="checkbox" className="mt-1 h-4 w-4 accent-primary" aria-label={`Select claim ${item.id}`}
              checked={selected.includes(item.id)} disabled={blocked || Boolean(error)}
              onChange={() => setSelected((previous) => previous.includes(item.id) ? previous.filter((id) => id !== item.id) : [...previous, item.id])} />
            <button type="button" aria-label={`Inspect claim ${item.id}`} disabled={blocked}
              className="min-w-0 flex-1 rounded-md text-left focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-focus disabled:opacity-50"
              onClick={() => inspect(item)}>
              <span className="block whitespace-pre-wrap break-words text-sm">{item.claim_text}</span>
              <span className="mt-1 block text-xs text-text-muted">{item.media_title || `Media ${item.media_id}`} · #{item.id} · {item.review_status}</span>
            </button>
          </li>)}
        </ul>
        {nextOffset !== null && <Button variant="outline" disabled={blocked || dirtyQueue || Boolean(error)} onClick={() => void loadQueue(nextOffset)}>Load more claims</Button>}
        {dirtyQueue && <p className="text-sm text-text-muted">Refresh the queue before loading more after a decision.</p>}
        <fieldset className="space-y-3 rounded-lg border border-border p-4" disabled={blocked || Boolean(error)}>
          <legend className="px-1 text-sm font-medium">Admin batch decisions · {selected.length} selected</legend>
          <label className="block text-sm">Batch reason code<input className={fieldClass} aria-label="Batch reason code" value={batchReason} onChange={(e) => setBatchReason(e.target.value)} /></label>
          <label className="block text-sm">Batch notes<textarea className={fieldClass} aria-label="Batch notes" value={batchNotes} onChange={(e) => setBatchNotes(e.target.value)} /></label>
          <div className="flex flex-wrap gap-2">
            <Button variant="primary" disabled={!selected.length || blocked || Boolean(error)} onClick={() => void decideBatch("approved")}>Approve selected claims</Button>
            <Button variant="outline" disabled={!selected.length || blocked || Boolean(error)} onClick={() => void decideBatch("rejected")}>Reject selected claims</Button>
          </div>
        </fieldset>
      </div>
      <div className="min-w-0">
        {active ? <article className="space-y-4 rounded-lg border border-border bg-surface p-4" aria-label={`Claim ${active.id} detail`}>
          <div>
            <h2 className="text-lg font-semibold">Claim {active.id} · {active.review_status}</h2>
            <p className="mt-2 max-w-prose whitespace-pre-wrap break-words">{active.claim_text}</p>
            <p className="mt-2 text-xs text-text-muted">Media {active.media_id} · chunk {active.chunk_index} · extraction span {active.span_start ?? "unknown"}–{active.span_end ?? "unknown"} · review version {active.review_version}</p>
          </div>
          <Button variant="outline" disabled={sourceLoading} onClick={() => void loadSource()}>Load current source text</Button>
          {sourceLoading && <p role="status">Loading source…</p>}
          {sourceError && <Alert variant="warning">{sourceError}</Alert>}
          {source !== null && <div className="space-y-2">
            <h3 className="text-sm font-medium">Current source text</h3>
            <p className="text-xs text-text-muted">Extraction offsets refer to the original chunk; current source content may have changed.</p>
            <pre className="max-h-64 overflow-auto whitespace-pre-wrap break-words rounded-md bg-surface2 p-3 font-sans text-sm">{source || "No source text returned."}</pre>
          </div>}
          <label className="block text-sm">Review notes<textarea className={fieldClass} aria-label="Review notes" disabled={busy} value={notes} onChange={(e) => setNotes(e.target.value)} /></label>
          <label className="block text-sm">Reason code<input className={fieldClass} aria-label="Reason code" disabled={busy} value={reason} onChange={(e) => setReason(e.target.value)} /></label>
          <label className="block text-sm">Corrected claim text<textarea className={fieldClass} aria-label="Corrected claim text" disabled={busy} value={correction} placeholder="Optional correction, saved with the decision" onChange={(e) => setCorrection(e.target.value)} /></label>
          <div className="flex flex-wrap gap-2">
            {(transitions[active.review_status] || []).map((status) => <Button key={status} variant={status === "approved" ? "primary" : "outline"}
              disabled={blocked || conflict || Boolean(error) || !Number.isInteger(active.review_version) || active.review_version < 1}
              onClick={() => void decide(status)}>
              {status === "approved" ? "Approve claim" : status === "rejected" ? "Reject claim" : status === "flagged" ? "Flag claim" : "Return claim to pending"}
            </Button>)}
            <Button variant="outline" disabled={blocked} onClick={() => void refreshClaim()}>Load latest claim</Button>
          </div>
          <section className="space-y-2 border-t border-border pt-4" aria-label="Saved review history">
            <h3 className="text-sm font-semibold">Saved review history</h3>
            {history === null && !historyError && <p role="status">Loading saved history…</p>}
            {historyError && <Alert variant="warning" action={{ label: "Retry history", onClick: () => void loadHistory(active.id) }}>Decision history could not be loaded.</Alert>}
            {history?.length === 0 && <p className="text-sm text-text-muted">No saved decisions yet.</p>}
            <ol className="space-y-3">
              {history?.map((event) => <li key={event.id} className="text-sm">
                <p>{event.old_status || "unreviewed"} → {event.new_status} · <time>{event.created_at}</time></p>
                <p className="text-text-muted">{event.reviewer_id === null || event.reviewer_id === undefined ? "Reviewer not recorded" : `Reviewer ${event.reviewer_id}`}</p>
                {event.reason_code && <p>Reason: {event.reason_code}</p>}
                {event.notes && <p className="whitespace-pre-wrap break-words">{event.notes}</p>}
                {event.new_text && event.new_text !== event.old_text && <p className="whitespace-pre-wrap break-words">Saved text: {event.new_text}</p>}
              </li>)}
            </ol>
          </section>
        </article> : <p className="text-sm text-text-muted">Inspect a claim to view its source, save a decision, and read its history.</p>}
      </div>
    </div>
  </section>
}
