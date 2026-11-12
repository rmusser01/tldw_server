/**
 * Request budget helpers for Playwright specs.
 *
 * trackRequests() records every API request a page *starts* (the `request`
 * event, so aborted, failed and still-pending requests count too).
 * analyseRequests() turns the recording into the numbers the ux-regression
 * request-budget ratchet compares with its baseline: total API calls,
 * per-key counts, duplicate GETs within a short window, and endpoints that
 * keep firing once the page is idle (polling).
 */
import type { Page, Request } from "@playwright/test"
import type { RatchetObservation } from "./ratchet"

export type TrackedRequest = {
  method: string
  url: string
  /** Method plus normalised path, e.g. "GET /api/v1/notes/:id". */
  key: string
  /** Method plus exact path and query: what "identical request" means for duplicates. */
  exactKey: string
  /** Milliseconds since the tracker started. */
  at: number
}

export type RequestTracker = {
  /** Milliseconds since the tracker started. */
  elapsed: () => number
  /** Requests recorded so far. */
  requests: () => TrackedRequest[]
  /** Stop recording and return everything recorded. */
  stop: () => TrackedRequest[]
}

const UUID_SEGMENT = /^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$/i
const NUMERIC_SEGMENT = /^\d+$/
const HEX_SEGMENT = /^[0-9a-f]{16,}$/i

/** Replace id-like path segments (numbers, UUIDs, long hex) with ":id". */
export function normaliseRequestPath(pathname: string): string {
  return pathname
    .split("/")
    .map((segment) =>
      NUMERIC_SEGMENT.test(segment) || UUID_SEGMENT.test(segment) || HEX_SEGMENT.test(segment) ? ":id" : segment
    )
    .join("/")
}

export function requestKey(method: string, url: string): string {
  return `${method.toUpperCase()} ${normaliseRequestPath(new URL(url).pathname)}`
}

/** Backend and Next API routes; static assets, HMR and pages are ignored. */
export const isApiUrl = (url: URL): boolean => url.pathname.startsWith("/api/")

export function trackRequests(
  page: Page,
  { include = isApiUrl }: { include?: (url: URL) => boolean } = {}
): RequestTracker {
  const startedAt = Date.now()
  const tracked: TrackedRequest[] = []
  const onRequest = (request: Request) => {
    let url: URL
    try {
      url = new URL(request.url())
    } catch {
      return
    }
    if (!include(url)) return
    const method = request.method().toUpperCase()
    tracked.push({
      method,
      url: request.url(),
      key: requestKey(method, request.url()),
      exactKey: `${method} ${url.pathname}${url.search}`,
      at: Date.now() - startedAt,
    })
  }
  page.on("request", onRequest)
  return {
    elapsed: () => Date.now() - startedAt,
    requests: () => [...tracked],
    stop: () => {
      page.off("request", onRequest)
      return [...tracked]
    },
  }
}

export type RequestBudgetSettings = {
  /** Only requests started within this many ms of the tracker start count. */
  windowMs: number
  /** A GET repeated with the same path and query within this many ms is a duplicate. */
  duplicateWindowMs: number
  /** Requests after this offset count as idle traffic. */
  idleFromMs: number
  /** An endpoint hit at least this many times while idle is polling. */
  pollingMinHits: number
}

export type RequestBudgetReport = {
  settings: RequestBudgetSettings
  total: number
  perKey: Record<string, number>
  /** Normalised key -> number of repeats within duplicateWindowMs. */
  duplicateGets: Map<string, RatchetObservation>
  /** Normalised key -> hits after idleFromMs, for keys with at least pollingMinHits. */
  polling: Map<string, RatchetObservation>
}

const formatOffset = (ms: number): string => `${(ms / 1000).toFixed(1)}s`

export function analyseRequests(
  recorded: readonly TrackedRequest[],
  settings: RequestBudgetSettings
): RequestBudgetReport {
  const requests = recorded.filter((request) => request.at <= settings.windowMs).sort((a, b) => a.at - b.at)

  const perKey: Record<string, number> = {}
  for (const request of requests) perKey[request.key] = (perKey[request.key] ?? 0) + 1

  const lastSeen = new Map<string, number>()
  const duplicateGets = new Map<string, RatchetObservation>()
  for (const request of requests) {
    if (request.method !== "GET") continue
    const previous = lastSeen.get(request.exactKey)
    lastSeen.set(request.exactKey, request.at)
    if (previous === undefined || request.at - previous > settings.duplicateWindowMs) continue
    const entry = duplicateGets.get(request.key) ?? { count: 0, details: [] }
    entry.count += 1
    entry.details?.push(`${request.exactKey} at ${formatOffset(request.at)} (previous ${formatOffset(previous)})`)
    duplicateGets.set(request.key, entry)
  }

  const idleHits = new Map<string, TrackedRequest[]>()
  for (const request of requests) {
    if (request.at < settings.idleFromMs) continue
    idleHits.set(request.key, [...(idleHits.get(request.key) ?? []), request])
  }
  const polling = new Map<string, RatchetObservation>()
  for (const [key, hits] of idleHits) {
    if (hits.length < settings.pollingMinHits) continue
    polling.set(key, {
      count: hits.length,
      details: [`idle hits at ${hits.map((hit) => formatOffset(hit.at)).join(", ")}`],
    })
  }

  return { settings, total: requests.length, perKey, duplicateGets, polling }
}

/** A compact, JSON-friendly copy of a report for test attachments. */
export function serialiseRequestReport(report: RequestBudgetReport, recorded: readonly TrackedRequest[]) {
  return {
    settings: report.settings,
    total: report.total,
    perKey: Object.fromEntries(Object.entries(report.perKey).sort(([a], [b]) => a.localeCompare(b))),
    duplicateGets: Object.fromEntries(report.duplicateGets),
    polling: Object.fromEntries(report.polling),
    timeline: recorded.map((request) => `${formatOffset(request.at)} ${request.exactKey}`),
  }
}
