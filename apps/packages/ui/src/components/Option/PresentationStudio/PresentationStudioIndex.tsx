import React from "react"
import { useNavigate } from "react-router-dom"

import { Button } from "@/components/Common/Button"
import { PageShell } from "@/components/Common/PageShell"
import { Badge, LoadingState, StatePanel } from "@/components/ui"
import { useServerOnline } from "@/hooks/useServerOnline"
import {
  tldwClient,
  type PresentationSummary
} from "@/services/tldw/TldwApiClient"

const PAGE_SIZE = 25
const AUTHORITY_EVENTS = [
  "tldw:config-updated",
  "tldw:auth-principal-changed",
  "tldw:slides-scope-mismatch"
] as const

const formatBytes = (bytes: number): string =>
  `${(Math.max(0, bytes) / 1024).toFixed(1)} KB`

const summaryMeta = (presentation: PresentationSummary): string[] => {
  if (presentation.content_kind === "structured_slides") {
    return [`${presentation.slide_count} slides`]
  }
  if (presentation.content_kind === "standalone_html") {
    return [
      `${presentation.html_slide_count} HTML slides`,
      formatBytes(presentation.html_bytes)
    ]
  }
  return ["Read only"]
}

const kindLabel = (presentation: PresentationSummary): string => {
  if (presentation.content_kind === "structured_slides") return "Structured slides"
  if (presentation.content_kind === "standalone_html") return "Standalone HTML + JavaScript"
  return "Unsupported kind"
}

export const PresentationStudioIndex: React.FC = () => {
  const navigate = useNavigate()
  const online = useServerOnline()
  const [presentations, setPresentations] = React.useState<PresentationSummary[]>([])
  const [loading, setLoading] = React.useState(online)
  const [loadingMore, setLoadingMore] = React.useState(false)
  const [error, setError] = React.useState<"initial" | "pagination" | null>(null)
  const [nextOffset, setNextOffset] = React.useState<number | null>(0)
  const [authorityEpoch, setAuthorityEpoch] = React.useState(0)
  const requestIdRef = React.useRef(0)

  React.useEffect(() => {
    const invalidate = () => {
      requestIdRef.current += 1
      setPresentations([])
      setNextOffset(0)
      setError(null)
      setAuthorityEpoch((current) => current + 1)
    }
    for (const eventName of AUTHORITY_EVENTS) {
      window.addEventListener(eventName, invalidate)
    }
    return () => {
      requestIdRef.current += 1
      for (const eventName of AUTHORITY_EVENTS) {
        window.removeEventListener(eventName, invalidate)
      }
    }
  }, [])

  const load = React.useCallback(async (offset: number, append: boolean) => {
    if (!online) return
    const requestId = ++requestIdRef.current
    append ? setLoadingMore(true) : setLoading(true)
    setError(null)
    try {
      const result = await tldwClient.listPresentations({ limit: PAGE_SIZE, offset })
      if (requestId !== requestIdRef.current) return
      const candidateOffset = result.pagination.next_offset
      if (
        result.pagination.has_more &&
        (
          typeof candidateOffset !== "number" ||
          !Number.isFinite(candidateOffset) ||
          !Number.isInteger(candidateOffset) ||
          candidateOffset <= offset
        )
      ) {
        setNextOffset(null)
        setError("pagination")
        return
      }
      setPresentations((current) => {
        const merged = append ? [...current, ...result.presentations] : result.presentations
        const byId = new Map<string, PresentationSummary>()
        for (const item of merged) byId.set(item.id, item)
        return Array.from(byId.values())
      })
      setNextOffset(result.pagination.has_more ? result.pagination.next_offset : null)
    } catch {
      if (requestId !== requestIdRef.current) return
      setError(append ? "pagination" : "initial")
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false)
        setLoadingMore(false)
      }
    }
  }, [online])

  React.useEffect(() => {
    if (!online) {
      setLoading(false)
      return
    }
    void load(0, false)
  }, [authorityEpoch, load, online])

  if (!online) {
    return (
      <PageShell className="py-6">
        <StatePanel
          state="unavailable"
          title="Presentation Studio is offline"
          message="Reconnect to load your presentations."
        />
      </PageShell>
    )
  }

  return (
    <PageShell className="space-y-6 py-6" maxWidthClassName="max-w-5xl">
      <header className="flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <h1 className="text-2xl font-semibold text-text">Presentation Studio</h1>
          <p className="max-w-[70ch] text-sm text-text-muted">
            Create and return to durable presentation projects.
          </p>
        </div>
        <Button variant="primary" size="lg" onClick={() => navigate("/presentation-studio/new")}>
          New presentation
        </Button>
      </header>

      {loading ? (
        <div role="status" aria-label="Loading presentations" className="rounded-lg border border-border bg-surface p-4">
          <LoadingState mode="skeleton" rows={4} />
        </div>
      ) : null}

      {!loading && error ? (
        <StatePanel
          state="error"
          title={error === "pagination" ? "Presentation pages could not continue" : "Presentations could not load"}
          message={error === "pagination" ? "The server returned an invalid next page. Retry from the beginning." : "Check the server connection, then try again."}
          primaryAction={{ label: "Retry", onClick: () => void load(0, false) }}
          role="alert"
        />
      ) : null}

      {!loading && error === null && presentations.length === 0 ? (
        <StatePanel
          state="empty"
          title="No presentations yet"
          message="Start with direct material or create a structured deck."
          primaryAction={{ label: "New presentation", onClick: () => navigate("/presentation-studio/new") }}
        />
      ) : null}

      {presentations.length > 0 ? (
        <section aria-labelledby="presentation-list-heading" className="space-y-3">
          <h2 id="presentation-list-heading" className="text-base font-semibold text-text">Recent projects</h2>
          <ul className="divide-y divide-border overflow-hidden rounded-lg border border-border bg-surface">
            {presentations.map((presentation) => (
              <li key={presentation.id} className="flex flex-col gap-3 p-4 sm:flex-row sm:items-center sm:justify-between">
                <div className="min-w-0 space-y-2">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3 className="truncate text-base font-semibold text-text">{presentation.title}</h3>
                    <Badge variant={presentation.content_kind === "unsupported" ? "warning" : "secondary"}>
                      {kindLabel(presentation)}
                    </Badge>
                  </div>
                  <p className="flex flex-wrap gap-x-3 gap-y-1 text-sm text-text-muted">
                    {summaryMeta(presentation).map((value) => <span key={value}>{value}</span>)}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="lg"
                  ariaLabel={`Open ${presentation.title}`}
                  onClick={() => navigate(`/presentation-studio/${presentation.id}`)}
                >
                  Open
                </Button>
              </li>
            ))}
          </ul>
          {nextOffset !== null ? (
            <div className="flex justify-center">
              <Button variant="outline" size="lg" loading={loadingMore} onClick={() => void load(nextOffset, true)}>
                Load more
              </Button>
            </div>
          ) : null}
        </section>
      ) : null}
    </PageShell>
  )
}
