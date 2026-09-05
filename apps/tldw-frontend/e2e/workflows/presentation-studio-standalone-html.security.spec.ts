import { createHash, randomUUID } from "node:crypto"
import http from "node:http"
import type { AddressInfo } from "node:net"

import {
  expect,
  test,
  type BrowserContext,
  type Page,
  type Route,
  type TestInfo
} from "@playwright/test"

const TEST_API_KEY = "TASK17-NONSECRET-TEST-KEY"
const RECOVERY_PREFIX = "tldw:presentation-studio:html:draft:v1:workspace:"

type SecurityEvent = {
  kind: string
  at: number
  surface?: string
  url?: string
  contentType?: string
  sourceCorrelated?: boolean
  dataOwnedAnchor?: boolean
  download?: string
  target?: string
  rel?: string
  method?: string
  workerName?: string
  workerType?: string
}

type ProtocolRequest = {
  method: string
  path: string
  origin: string | null
  contentType: string | null
  bodySourceCorrelated: boolean
  bodyBytes: number
  bodySha256: string | null
  ifMatch: string | null
  acceptedContentKinds: string | null
  apiKey: string | null
  authorization: string | null
  idempotencyKey: string | null
  requestedHeaders: string | null
  requestedMethod: string | null
}

type ProtocolResponse = {
  path: string
  status: number
  contentType: string
}

type SecurityProtocolState = {
  presentationId: string
  principalId: number
  ownerPrincipalId: number
  source: string
  digest: string
  initialSource: string
  initialDigest: string
  etag: string
  corruptDigest: boolean
  authenticated: boolean
  generationCompletionEnabled: boolean
  generationSubmitted: boolean
  generationPollCount: number
  authRequestCount: number
  authHeldCount: number
  authPendingCount: number
  authHoldAfterRequestCount: number | null
  requests: ProtocolRequest[]
  responses: ProtocolResponse[]
  unexpectedPaths: string[]
  overflowCount: number
}

type SecurityProtocolServer = {
  origin: string
  state: SecurityProtocolState
  deferAuthAfter: (additionalResponses: number) => void
  releaseAuth: () => void
  close: () => Promise<void>
}

type ContextObservations = {
  requests: Array<{
    url: string
    method: string
    sourceCorrelated: boolean
    contentType: string | null
  }>
  newPages: string[]
  workers: string[]
  serviceWorkers: string[]
  errors: string[]
  navigations: string[]
  childNavigations: string[]
  overflowCount: number
}

type SecurityAggregate = {
  events: SecurityEvent[]
  overflowCount: number
}

type SecurityFixture = {
  marker: string
  generatedSource: string
  safeDirtySource: string
  editorSource: string
  corruptSource: string
  sentinelOrigin: string
  presentationId: string
  webOrigin: string
  generationJobId: string
}

const sha256 = (value: string) => createHash("sha256").update(value).digest("hex")

const securityFixture = (testInfo: TestInfo): SecurityFixture => {
  const entropy = `${testInfo.project.name}-${randomUUID()}`.replace(/[^a-zA-Z0-9-]/g, "-")
  const marker = `TASK17-SOURCE-${entropy}`
  const sentinelOrigin = `https://task17-source-${entropy}.invalid`
  const presentationId = `task17-security-${entropy}`
  const generationJobId = randomUUID()
  const configuredBaseUrl = String(testInfo.project.use.baseURL ?? "http://localhost:8080")
  const webOrigin = new URL(configuredBaseUrl).origin
  const generatedSource = `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task17 generated source</title><style>.slide{color:#111}</style></head>
<body><section class="slide"><h1>Task17 generated source</h1><p>${marker}</p></section>
<script>globalThis.__TASK17_EXECUTED__="${marker}";</script></body></html>`
  const editorSource = `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Task17 exact-digest editor source</title>
<style>.slide{background-image:url("${sentinelOrigin}/style/${marker}")}</style></head>
<body><section class="slide"><h1>Task17 exact-digest editor source</h1><p>Trusted editor outline</p>
<a href="${sentinelOrigin}/link/${marker}">external destination</a>
<img src="${sentinelOrigin}/image/${marker}" alt="external image">
<p>${sentinelOrigin}/plain/${marker}</p>
<p>data:text/html,%3Cscript%3EglobalThis.__TASK17_EXECUTED__%3D%22${marker}%22%3C/script%3E</p></section>
<script>globalThis.__TASK17_EXECUTED__="${marker}";</script></body></html>`
  const safeDirtySource = generatedSource
    .replace("Task17 generated source", "Task17 safe dirty source")
    .replace(`<p>${marker}</p>`, `<p>${marker} safe recovery text</p>`)
  const corruptSource = `<!doctype html><html><head><title>Corrupt URL detail</title></head>
<body><section class="slide"><h1>Corrupt URL detail</h1><a href="${sentinelOrigin}/${marker}">bad</a></section>
<script>globalThis.__TASK17_EXECUTED__="${marker}";</script></body></html>`
  return {
    marker,
    generatedSource,
    safeDirtySource,
    editorSource,
    corruptSource,
    sentinelOrigin,
    presentationId,
    webOrigin,
    generationJobId
  }
}

const capabilities = {
  schema_version: 1,
  content_kind_request_header: "X-Slides-Accept-Content-Kinds",
  content_kinds: {
    structured_slides: { read: true, edit: true },
    standalone_html: {
      read: true,
      edit: true,
      export_attachment: true,
      draft_attachment: true,
      reason: null,
      limits: {
        max_document_bytes: 1_048_576,
        max_source_write_bytes: 1_048_576,
        max_draft_attachment_bytes: 1_048_576,
        max_slides: 30,
        max_nesting_depth: 128
      }
    }
  },
  generation_modes: {
    structured_slides: { enabled: true, transport: "existing_source_endpoints" },
    standalone_html: {
      enabled: true,
      reason: null,
      transport: "slides_generation_job",
      source_kinds: ["prompt", "chat", "media", "notes", "rag"],
      provider: "task17-provider",
      model: "task17-model",
      adapter_id: "openai_official_chat_v1",
      endpoint_identity: "https://provider.example.test/v1/chat/completions",
      generation_config_revision: `sha256:${"8".repeat(64)}`,
      input_limits: {
        max_request_bytes: 4_194_304,
        max_source_chars: 200_000,
        max_source_tokens: 50_000,
        max_audience_chars: 500,
        max_source_identifier_bytes: 256,
        max_note_ids: 100,
        max_rag_query_chars: 20_000,
        max_rag_top_k: 100
      },
      output_limits: {
        max_provider_response_bytes: 8_388_608,
        max_document_bytes: 1_048_576
      }
    }
  }
}

const readRequestBody = (request: http.IncomingMessage): Promise<string> =>
  new Promise((resolve, reject) => {
    let body = ""
    request.setEncoding("utf8")
    request.on("data", (chunk: string) => {
      body += chunk
    })
    request.on("end", () => resolve(body))
    request.on("error", reject)
  })

const startSecurityProtocolServer = async (
  fixture: SecurityFixture,
  options: {
    source?: string
    corruptDigest?: boolean
    generationCompletionEnabled?: boolean
  } = {}
): Promise<SecurityProtocolServer> => {
  const source = options.source ?? fixture.generatedSource
  const state: SecurityProtocolState = {
    presentationId: fixture.presentationId,
    principalId: 42,
    ownerPrincipalId: 42,
    source,
    digest: sha256(source),
    initialSource: source,
    initialDigest: sha256(source),
    etag: '"v1"',
    corruptDigest: options.corruptDigest ?? false,
    authenticated: true,
    generationCompletionEnabled: options.generationCompletionEnabled ?? true,
    generationSubmitted: false,
    generationPollCount: 0,
    authRequestCount: 0,
    authHeldCount: 0,
    authPendingCount: 0,
    authHoldAfterRequestCount: null,
    requests: [],
    responses: [],
    unexpectedPaths: [],
    overflowCount: 0
  }
  const authWaiters = new Set<() => void>()
  const releaseAuth = () => {
    state.authHoldAfterRequestCount = null
    for (const release of authWaiters) release()
    authWaiters.clear()
    state.authPendingCount = 0
  }

  const server = http.createServer(async (request, response) => {
    const method = (request.method ?? "GET").toUpperCase()
    const url = new URL(request.url ?? "/", "http://127.0.0.1")
    const origin = typeof request.headers.origin === "string" ? request.headers.origin : null
    const body = method === "GET" || method === "HEAD" || method === "OPTIONS"
      ? ""
      : await readRequestBody(request)
    const protocolRequest: ProtocolRequest = {
      method,
      path: `${url.pathname}${url.search}`,
      origin,
      contentType: typeof request.headers["content-type"] === "string"
        ? request.headers["content-type"]
        : null,
      bodySourceCorrelated: body.includes(fixture.marker),
      bodyBytes: Buffer.byteLength(body),
      bodySha256: body ? sha256(body) : null,
      ifMatch: typeof request.headers["if-match"] === "string" ? request.headers["if-match"] : null,
      acceptedContentKinds: typeof request.headers["x-slides-accept-content-kinds"] === "string"
        ? request.headers["x-slides-accept-content-kinds"]
        : null,
      apiKey: typeof request.headers["x-api-key"] === "string" ? request.headers["x-api-key"] : null,
      authorization: typeof request.headers.authorization === "string"
        ? request.headers.authorization
        : null,
      idempotencyKey: typeof request.headers["idempotency-key"] === "string"
        ? request.headers["idempotency-key"]
        : null,
      requestedHeaders: typeof request.headers["access-control-request-headers"] === "string"
        ? request.headers["access-control-request-headers"]
        : null,
      requestedMethod: typeof request.headers["access-control-request-method"] === "string"
        ? request.headers["access-control-request-method"]
        : null
    }
    if (state.requests.length < 2_000) state.requests.push(protocolRequest)
    else state.overflowCount += 1

    const corsHeaders: Record<string, string> = { Vary: "Origin" }
    if (origin === fixture.webOrigin) {
      Object.assign(corsHeaders, {
        "Access-Control-Allow-Origin": fixture.webOrigin,
        "Access-Control-Allow-Credentials": "true",
        "Access-Control-Expose-Headers": [
          "ETag",
          "Retry-After",
          "Last-Modified",
          "Content-Length",
          "X-Request-ID",
          "Traceparent",
          "Content-Disposition",
          "X-Content-Type-Options",
          "X-Download-Options",
          "Cache-Control",
          "Referrer-Policy",
          "Cross-Origin-Resource-Policy"
        ].join(", ")
      })
    }

    const send = (
      status: number,
      contentType: string,
      payload: string | Buffer,
      headers: Record<string, string> = {}
    ) => {
      if (state.responses.length < 2_000) {
        state.responses.push({ path: url.pathname, status, contentType })
      } else state.overflowCount += 1
      const contentLength = typeof payload === "string"
        ? Buffer.byteLength(payload)
        : payload.byteLength
      response.writeHead(status, {
        ...corsHeaders,
        "Content-Type": contentType,
        "Content-Length": String(contentLength),
        "Last-Modified": "Fri, 22 Aug 2026 12:00:00 GMT",
        "X-Request-ID": fixture.generationJobId,
        Traceparent: `00-${"a".repeat(32)}-${"b".repeat(16)}-01`,
        ...headers
      })
      response.end(payload)
    }
    const sendJson = (
      status: number,
      payload: unknown,
      headers: Record<string, string> = {}
    ) => send(status, "application/json", JSON.stringify(payload), headers)

    if (method === "OPTIONS") {
      response.writeHead(204, {
        ...corsHeaders,
        "Access-Control-Allow-Headers": "accept, authorization, cache-control, content-type, if-match, idempotency-key, traceparent, x-api-key, x-request-id, x-slides-accept-content-kinds",
        "Access-Control-Allow-Methods": "GET, HEAD, PUT, POST, OPTIONS"
      })
      response.end()
      return
    }

    const stateTitle = state.source.match(/<title>([^<]+)<\/title>/)?.[1] ?? "Standalone HTML"
    const summary = {
      id: state.presentationId,
      title: stateTitle,
      description: null,
      theme: "black",
      created_at: "2026-08-22T12:00:00Z",
      last_modified: "2026-08-22T12:00:00Z",
      deleted: false,
      version: Number(state.etag.slice(2, -1)),
      provenance: {
        source_kind: "prompt",
        provider: "task17-provider",
        model: "task17-model"
      },
      content_kind: "standalone_html",
      html_slide_count: 1,
      html_bytes: Buffer.byteLength(state.source)
    }
    const detail = {
      ...summary,
      client_id: String(state.principalId),
      source_type: "prompt",
      source_ref: null,
      source_query: null,
      html_document: state.source,
      html_sha256: state.corruptDigest ? "0".repeat(64) : state.digest,
      generation_job_uuid: fixture.generationJobId,
      generation_provenance: {
        schema_version: 1,
        source_kind: "prompt",
        source_ref: null,
        source_snapshot_hmac_sha256: "a".repeat(64),
        digest_key_id: "task17-key",
        source_bytes: 24,
        provider: "task17-provider",
        model: "task17-model",
        adapter_id: "openai_official_chat_v1",
        endpoint_identity: "https://provider.example.test/v1/chat/completions",
        prompt_sha256: "b".repeat(64)
      }
    }
    const presentationBase = `/api/v1/slides/presentations/${encodeURIComponent(
      state.presentationId
    )}`

    if (url.pathname === "/api/v1/health" || url.pathname === "/api/v1/health/live") {
      sendJson(200, { status: "ok" })
      return
    }
    if (url.pathname === "/openapi.json") {
      sendJson(200, {
        info: { version: "task17" },
        paths: {
          "/api/v1/slides/presentations": {},
          "/api/v1/slides/presentations/{presentation_id}": {},
          "/api/v1/slides/presentations/{presentation_id}/export": {}
        }
      })
      return
    }
    if (url.pathname === "/api/v1/auth/me") {
      state.authRequestCount += 1
      if (
        state.authHoldAfterRequestCount !== null &&
        state.authRequestCount > state.authHoldAfterRequestCount
      ) {
        state.authHeldCount += 1
        await new Promise<void>((resolve) => {
          const release = () => {
            authWaiters.delete(release)
            state.authPendingCount = authWaiters.size
            resolve()
          }
          authWaiters.add(release)
          state.authPendingCount = authWaiters.size
        })
      }
      if (!state.authenticated) {
        sendJson(401, { detail: "not_authenticated" }, {
          "Cache-Control": "private, no-store"
        })
        return
      }
      sendJson(200, {
        id: state.principalId,
        username: `task17-owner-${state.principalId}`,
        is_active: true
      }, { "Cache-Control": "private, no-store" })
      return
    }
    if (url.pathname === "/api/v1/config/docs-info") {
      sendJson(200, { capabilities: {} })
      return
    }
    if (url.pathname === "/api/v1/slides/capabilities") {
      if (url.searchParams.get("task17_drain") === "1") {
        sendJson(503, { detail: "service_draining" }, { "Retry-After": "3" })
        return
      }
      sendJson(200, capabilities, { "Cache-Control": "private, no-store" })
      return
    }
    if (url.pathname === "/api/v1/slides/styles") {
      sendJson(200, { styles: [], total: 0 })
      return
    }
    if (url.pathname === "/api/v1/slides/generations" && method === "POST") {
      const idempotencyKey = request.headers["idempotency-key"]
      if (typeof idempotencyKey !== "string" || idempotencyKey.length < 16) {
        sendJson(400, { detail: "invalid_idempotency_key" })
        return
      }
      state.generationSubmitted = true
      state.generationPollCount = 0
      sendJson(202, {
        generation_id: fixture.generationJobId,
        status: "queued",
        status_url: `/api/v1/slides/generations/${fixture.generationJobId}`,
        presentation_id: null
      }, {
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (
      url.pathname === `/api/v1/slides/generations/${fixture.generationJobId}` &&
      method === "GET" &&
      state.generationSubmitted
    ) {
      state.generationPollCount += 1
      const completed = state.generationCompletionEnabled && state.generationPollCount >= 2
      sendJson(200, completed
        ? {
            generation_id: fixture.generationJobId,
            status: "completed",
            status_url: `/api/v1/slides/generations/${fixture.generationJobId}`,
            presentation_id: fixture.presentationId,
            content_kind: "standalone_html"
          }
        : {
            generation_id: fixture.generationJobId,
            status: "running",
            status_url: `/api/v1/slides/generations/${fixture.generationJobId}`,
            presentation_id: null,
            progress_text: "Validating generated document"
          }, {
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (
      state.principalId !== state.ownerPrincipalId &&
      url.pathname.startsWith(presentationBase)
    ) {
      sendJson(404, { detail: "presentation_not_found" })
      return
    }
    if (url.pathname === `${presentationBase}/metadata`) {
      sendJson(200, summary, {
        ETag: state.etag,
        "Cache-Control": "private, no-store"
      })
      return
    }
    if (url.pathname === presentationBase && method === "GET") {
      sendJson(200, detail, {
        ETag: state.etag,
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (url.pathname === `${presentationBase}/html-source` && method === "PUT") {
      const requestContentType = typeof request.headers["content-type"] === "string"
        ? request.headers["content-type"].trim().toLowerCase()
        : ""
      if (requestContentType !== "application/octet-stream") {
        sendJson(415, { detail: "unsupported_media_type" })
        return
      }
      if (request.headers["if-match"] !== state.etag) {
        sendJson(412, { detail: "presentation_version_conflict", etag: state.etag })
        return
      }
      if (body.includes(fixture.sentinelOrigin)) {
        sendJson(422, { detail: "standalone_html_invalid_document" })
        return
      }
      state.source = body
      state.digest = sha256(body)
      state.corruptDigest = false
      const nextVersion = Number(state.etag.slice(2, -1)) + 1
      state.etag = `"v${nextVersion}"`
      sendJson(200, {
        ...detail,
        version: nextVersion,
        html_document: body,
        html_sha256: state.digest,
        html_bytes: Buffer.byteLength(body)
      }, {
        ETag: state.etag,
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (url.pathname === `${presentationBase}/draft-attachment` && method === "POST") {
      send(200, "application/octet-stream", Buffer.from(body), {
        "Content-Disposition": 'attachment; filename="presentation.html"',
        "X-Content-Type-Options": "nosniff",
        "X-Download-Options": "noopen",
        "Cache-Control": "private, no-store",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Resource-Policy": "same-origin"
      })
      return
    }
    if (url.pathname === `${presentationBase}/versions` && method === "GET") {
      sendJson(200, {
        versions: [{
          version: Number(state.etag.slice(2, -1)),
          created_at: "2026-08-22T12:00:00Z",
          title: stateTitle,
          html_sha256: state.digest,
          html_bytes: Buffer.byteLength(state.source)
        }],
        total: 1
      })
      return
    }
    if (url.pathname === `${presentationBase}/versions/1` && method === "GET") {
      sendJson(200, {
        ...detail,
        version: 1,
        html_document: state.initialSource,
        html_sha256: state.initialDigest,
        html_bytes: Buffer.byteLength(state.initialSource)
      }, {
        ETag: '"v1"',
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (url.pathname === `${presentationBase}/versions/1/restore` && method === "POST") {
      if (request.headers["if-match"] !== state.etag) {
        sendJson(412, { detail: "presentation_version_conflict", etag: state.etag })
        return
      }
      const nextVersion = Number(state.etag.slice(2, -1)) + 1
      state.source = state.initialSource
      state.digest = state.initialDigest
      state.etag = `"v${nextVersion}"`
      sendJson(200, {
        ...detail,
        version: nextVersion,
        html_document: state.initialSource,
        html_sha256: state.initialDigest,
        html_bytes: Buffer.byteLength(state.initialSource)
      }, {
        ETag: state.etag,
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (url.pathname === `${presentationBase}/export` && method === "GET") {
      if (url.searchParams.get("format") === "json") {
        sendJson(200, detail, {
          ETag: state.etag,
          "Cache-Control": "private, no-store",
          "X-Content-Type-Options": "nosniff",
          "X-Download-Options": "noopen",
          "Content-Disposition": 'attachment; filename="presentation.json"',
          "Referrer-Policy": "no-referrer",
          "Cross-Origin-Resource-Policy": "same-origin"
        })
        return
      }
      if (url.searchParams.get("format") !== "html") {
        sendJson(400, { detail: "unsupported_export_format" })
        return
      }
      send(200, "application/octet-stream", Buffer.from(state.source), {
        "Content-Disposition": 'attachment; filename="presentation.html"',
        "X-Content-Type-Options": "nosniff",
        "X-Download-Options": "noopen",
        "Cache-Control": "private, no-store",
        "Referrer-Policy": "no-referrer",
        "Cross-Origin-Resource-Policy": "same-origin"
      })
      return
    }
    if (url.pathname === "/api/v1/slides/presentations" && method === "GET") {
      sendJson(200, {
        presentations: state.principalId === state.ownerPrincipalId ? [summary] : [],
        total: state.principalId === state.ownerPrincipalId ? 1 : 0,
        limit: 50,
        offset: 0,
        pagination: {
          mode: "offset",
          limit: 50,
          offset: 0,
          total: state.principalId === state.ownerPrincipalId ? 1 : 0,
          has_more: false,
          next_offset: null
        },
        has_more: false,
        next_offset: null
      })
      return
    }
    if (url.pathname.startsWith("/api/v1/notifications")) {
      sendJson(200, url.pathname.endsWith("unread-count")
        ? { unread_count: 0 }
        : { items: [], total: 0 })
      return
    }
    if (url.pathname === "/api/v1/persona/profiles" && method === "GET") {
      sendJson(200, { items: [], total: 0 })
      return
    }
    if (url.pathname === "/api/v1/rag/health" && method === "GET") {
      sendJson(200, { status: "healthy" })
      return
    }
    if (state.unexpectedPaths.length < 2_000) {
      state.unexpectedPaths.push(`${method} ${url.pathname}${url.search}`)
    } else state.overflowCount += 1
    sendJson(404, { detail: "not found" })
  })

  await new Promise<void>((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject)
      resolve()
    })
  })
  const address = server.address() as AddressInfo
  return {
    origin: `http://127.0.0.1:${address.port}`,
    state,
    deferAuthAfter: (additionalResponses) => {
      state.authHeldCount = 0
      state.authHoldAfterRequestCount = state.authRequestCount + additionalResponses
    },
    releaseAuth,
    close: () => new Promise<void>((resolve, reject) => {
      releaseAuth()
      server.close((error) => error ? reject(error) : resolve())
    })
  }
}

const installContextObservability = (
  context: BrowserContext,
  fixture: SecurityFixture
): ContextObservations => {
  const observations: ContextObservations = {
    requests: [],
    newPages: [],
    workers: [],
    serviceWorkers: [],
    errors: [],
    navigations: [],
    childNavigations: [],
    overflowCount: 0
  }
  const pushBounded = (target: string[], value: string) => {
    if (target.length < 2_000) target.push(value.slice(0, 4_096))
    else observations.overflowCount += 1
  }
  const observePage = (page: Page) => {
    page.on("worker", (worker) => pushBounded(observations.workers, worker.url()))
    page.on("framenavigated", (frame) => {
      if (frame === page.mainFrame()) pushBounded(observations.navigations, frame.url())
      else pushBounded(observations.childNavigations, frame.url())
    })
  }
  context.pages().forEach(observePage)
  context.on("request", (request) => {
    const body = request.postDataBuffer()
    const observation = {
      url: request.url().slice(0, 4_096),
      method: request.method(),
      sourceCorrelated: body?.includes(Buffer.from(fixture.marker)) ?? false,
      contentType: request.headers()["content-type"] ?? null
    }
    if (observations.requests.length < 2_000) observations.requests.push(observation)
    else observations.overflowCount += 1
  })
  context.on("page", (page) => {
    pushBounded(observations.newPages, page.url())
    observePage(page)
  })
  context.on("serviceworker", (worker) => pushBounded(observations.serviceWorkers, worker.url()))
  context.on("weberror", (error) => pushBounded(observations.errors, error.error().message))
  context.on("requestfailed", (request) => {
    if (request.url().includes(fixture.marker)) pushBounded(observations.errors, request.url())
  })
  return observations
}

const installPageSecurityInstrumentation = async (
  context: BrowserContext,
  fixture: SecurityFixture,
  apiOrigin: string,
  options: {
    authMode?: "single-user" | "multi-user"
    accessToken?: string
    calibrateBlobUrlSinks?: boolean
  } = {}
): Promise<SecurityAggregate> => {
  const aggregate: SecurityAggregate = { events: [], overflowCount: 0 }
  await context.exposeBinding(
    "__task17RecordSecurityEvent",
    (_source, event: SecurityEvent) => {
      if (aggregate.events.length < 10_000) {
        aggregate.events.push({
          ...event,
          surface: event.surface?.slice(0, 4_096),
          url: event.url?.slice(0, 4_096),
          contentType: event.contentType?.slice(0, 256),
          download: event.download?.slice(0, 512),
          target: event.target?.slice(0, 128),
          rel: event.rel?.slice(0, 256),
          method: event.method?.slice(0, 32),
          workerName: event.workerName?.slice(0, 128),
          workerType: event.workerType?.slice(0, 32)
        })
      } else {
        aggregate.overflowCount += 1
      }
    }
  )
  await context.addInitScript(
    ({
      marker,
      serverUrl,
      apiKey,
      authMode,
      accessToken,
      calibrateBlobUrlSinks
    }) => {
      type EventRecord = SecurityEvent
      type SecurityWindow = typeof window & {
        __task17SecurityEvents?: EventRecord[]
        __task17SecurityOverflow?: number
        __task17RecordSecurityEvent?: (event: EventRecord) => Promise<void>
        __task17FlushSecurityEvents?: () => Promise<void>
        __TASK17_EXECUTED__?: string
        chrome?: Record<string, unknown>
        browser?: Record<string, unknown>
      }
      const scope = window as SecurityWindow
      const events: EventRecord[] = []
      const pendingSecurityWork = new Set<Promise<unknown>>()
      scope.__task17SecurityEvents = events
      scope.__task17SecurityOverflow = 0
      const retainedLifecycleKinds = new Set([
        "storage-set",
        "storage-remove",
        "logical-storage-set",
        "logical-storage-remove",
        "logical-storage-clear",
        "storage-calibration",
        "storage-shim-calibration",
        "runtime-calibration",
        "url-sink-calibration",
        "html-sink-calibration",
        "message-sink-calibration",
        "service-worker-calibration",
        "runtime-message",
        "service-worker-post",
        "websocket-send",
        "send-beacon",
        "history-state",
        "worker-constructor",
        "worker-post",
        "worker-terminate",
        "blob-create",
        "blob-revoke",
        "anchor-click",
        "anchor-remove"
      ])
      const trackSecurityWork = <T,>(operation: Promise<T>) => {
        const tracked = operation.catch(() => {
          scope.__task17SecurityOverflow = (scope.__task17SecurityOverflow ?? 0) + 1
        })
        pendingSecurityWork.add(tracked)
        void tracked.finally(() => pendingSecurityWork.delete(tracked))
      }
      const mirror = (event: EventRecord) => {
        const binding = scope.__task17RecordSecurityEvent
        trackSecurityWork(binding
          ? binding(event)
          : Promise.reject(new Error("Task17 security recorder binding is unavailable")))
      }
      const record = (event: Omit<EventRecord, "at">) => {
        const retainedBlobUrlAssignment = event.kind === "dom-url" &&
          typeof event.url === "string" &&
          event.url.includes("blob:")
        if (
          !event.sourceCorrelated &&
          !retainedLifecycleKinds.has(event.kind) &&
          !retainedBlobUrlAssignment
        ) return
        const stamped = { ...event, at: performance.now() }
        if (events.length < 2_000) events.push(stamped)
        else scope.__task17SecurityOverflow = (scope.__task17SecurityOverflow ?? 0) + 1
        mirror(stamped)
      }
      const sourceCorrelated = (root: unknown): boolean => {
        const visited = new WeakSet<object>()
        let remaining = 500
        const scan = (value: unknown, depth: number): boolean => {
          remaining -= 1
          if (remaining < 0 || depth > 4 || value == null) return false
          if (typeof value === "string") return value.slice(0, 1_048_576).includes(marker)
          if (value instanceof ArrayBuffer) {
            return new TextDecoder().decode(value.slice(0, 1_048_576)).includes(marker)
          }
          if (ArrayBuffer.isView(value)) {
            const bytes = new Uint8Array(
              value.buffer,
              value.byteOffset,
              Math.min(value.byteLength, 1_048_576)
            )
            return new TextDecoder().decode(bytes).includes(marker)
          }
          if (typeof value !== "object") return false
          if (visited.has(value)) return false
          visited.add(value)
          if (Array.isArray(value)) {
            return value.slice(0, 100).some((item) => scan(item, depth + 1))
          }
          try {
            return Object.keys(value as Record<string, unknown>)
              .slice(0, 100)
              .some((key) => {
                const descriptor = Object.getOwnPropertyDescriptor(value, key)
                return descriptor && "value" in descriptor
                  ? scan(descriptor.value, depth + 1)
                  : false
              })
          } catch {
            return false
          }
        }
        return scan(root, 0)
      }

      const blobCorrelation = new WeakMap<Blob, boolean>()
      const NativeBlob = window.Blob
      window.Blob = new Proxy(NativeBlob, {
        construct(target, args) {
          const blob = Reflect.construct(target, args) as Blob
          blobCorrelation.set(blob, sourceCorrelated(args[0]))
          return blob
        }
      }) as typeof Blob

      let task17ExecutedValue = scope.__TASK17_EXECUTED__
      Object.defineProperty(scope, "__TASK17_EXECUTED__", {
        configurable: true,
        get: () => task17ExecutedValue,
        set: (value: string | undefined) => {
          task17ExecutedValue = value
          record({
            kind: "sentinel-execution",
            sourceCorrelated: true
          })
        }
      })
      scope.__task17FlushSecurityEvents = async () => {
        while (pendingSecurityWork.size > 0) {
          await Promise.all([...pendingSecurityWork])
        }
        const binding = scope.__task17RecordSecurityEvent
        if (!binding) {
          scope.__task17SecurityOverflow = (scope.__task17SecurityOverflow ?? 0) + 1
          return
        }
        await binding({
          kind: "aggregate-barrier",
          at: performance.now(),
          sourceCorrelated: false
        }).catch(() => {
          scope.__task17SecurityOverflow = (scope.__task17SecurityOverflow ?? 0) + 1
        })
      }

      const local = window.localStorage
      const session = window.sessionStorage
      const nativeSetItem = Storage.prototype.setItem
      Storage.prototype.setItem = function setItem(key: string, value: string) {
        record({
          kind: "storage-set",
          surface: this === local ? "localStorage" : this === session ? "sessionStorage" : "Storage",
          sourceCorrelated: sourceCorrelated(value),
          url: key
        })
        return nativeSetItem.call(this, key, value)
      }
      const nativeRemoveItem = Storage.prototype.removeItem
      Storage.prototype.removeItem = function removeItem(key: string) {
        record({
          kind: "storage-remove",
          surface: this === local ? "localStorage" : this === session ? "sessionStorage" : "Storage",
          url: key
        })
        return nativeRemoveItem.call(this, key)
      }

      type StorageKeys = string | string[] | Record<string, unknown> | null
      type StorageResult = Record<string, unknown>
      type ExistingStorageArea = {
        get?: (...args: unknown[]) => unknown
        set?: (...args: unknown[]) => unknown
        remove?: (...args: unknown[]) => unknown
        clear?: (...args: unknown[]) => unknown
      }
      const createStorageArea = (
        surface: string,
        existing: ExistingStorageArea | undefined,
        preferCallbackDelegate: boolean
      ) => {
        const values = new Map<string, unknown>()
        const usesLocalStorageFallback = surface.endsWith(".local")
        const readFallbackValue = (key: string): { found: boolean; value?: unknown } => {
          if (values.has(key)) return { found: true, value: values.get(key) }
          if (!usesLocalStorageFallback) return { found: false }
          const raw = localStorage.getItem(key)
          if (raw == null) return { found: false }
          try {
            return { found: true, value: JSON.parse(raw) }
          } catch {
            return { found: true, value: raw }
          }
        }
        const delegate = <T,>(
          method: keyof ExistingStorageArea,
          args: unknown[],
          fallback: () => T | Promise<T>
        ): Promise<T> => {
          const candidate = existing?.[method]
          if (typeof candidate !== "function") return Promise.resolve(fallback())
          if (!preferCallbackDelegate) {
            try {
              return Promise.resolve(candidate.apply(existing, args) as T | Promise<T>)
            } catch (error) {
              return Promise.reject(error)
            }
          }
          return new Promise<T>((resolve, reject) => {
            let settled = false
            const settle = (value: T) => {
              if (settled) return
              settled = true
              resolve(value)
            }
            try {
              const returned = candidate.apply(existing, [...args, settle])
              if (returned && typeof (returned as PromiseLike<T>).then === "function") {
                void Promise.resolve(returned as PromiseLike<T>).then(settle, reject)
              } else if (returned !== undefined) {
                settle(returned as T)
              }
            } catch (error) {
              reject(error)
            }
          })
        }
        const withCallback = <T,>(promise: Promise<T>, callback?: (value: T) => void) => {
          if (callback) void promise.then(callback)
          return promise
        }
        const fallbackGet = (keys: StorageKeys | undefined): StorageResult => {
          if (keys == null) {
            if (!usesLocalStorageFallback) return Object.fromEntries(values.entries())
            const result: StorageResult = {}
            for (let index = 0; index < localStorage.length; index += 1) {
              const key = localStorage.key(index)
              if (!key) continue
              const entry = readFallbackValue(key)
              if (entry.found) result[key] = entry.value
            }
            return result
          }
          if (typeof keys === "string") {
            const entry = readFallbackValue(keys)
            return entry.found ? { [keys]: entry.value } : {}
          }
          if (Array.isArray(keys)) {
            const result: StorageResult = {}
            for (const key of keys) {
              const entry = readFallbackValue(key)
              if (entry.found) result[key] = entry.value
            }
            return result
          }
          return Object.fromEntries(Object.entries(keys).map(([key, fallback]) => {
            const entry = readFallbackValue(key)
            return [key, entry.found ? entry.value : fallback]
          }))
        }
        const area = {
          get: (
            keysOrCallback?: StorageKeys | ((value: StorageResult) => void),
            maybeCallback?: (value: StorageResult) => void
          ) => {
            const callback = typeof keysOrCallback === "function"
              ? keysOrCallback
              : maybeCallback
            const keys = typeof keysOrCallback === "function" ? null : keysOrCallback
            return withCallback(
              delegate<StorageResult>("get", [keys ?? null], () => fallbackGet(keys)),
              callback
            )
          },
          set: (items: Record<string, unknown>, callback?: () => void) => {
            record({
              kind: "logical-storage-set",
              surface,
              sourceCorrelated: sourceCorrelated(items)
            })
            const operation = delegate<void>("set", [items], () => {
              Object.entries(items).forEach(([key, value]) => {
                values.set(key, value)
                if (usesLocalStorageFallback) {
                  localStorage.setItem(key, JSON.stringify(value))
                }
              })
            })
            if (callback) void operation.then(callback)
            return operation
          },
          remove: (keys: string | string[], callback?: () => void) => {
            record({ kind: "logical-storage-remove", surface })
            const operation = delegate<void>("remove", [keys], () => {
              ;(Array.isArray(keys) ? keys : [keys]).forEach((key) => {
                values.delete(key)
                if (usesLocalStorageFallback) localStorage.removeItem(key)
              })
            })
            if (callback) void operation.then(callback)
            return operation
          },
          clear: (callback?: () => void) => {
            record({ kind: "logical-storage-clear", surface })
            const operation = delegate<void>("clear", [], () => {
              values.clear()
              if (usesLocalStorageFallback) localStorage.clear()
            })
            if (callback) void operation.then(callback)
            return operation
          }
        }
        return Object.assign(Object.create(existing ?? null), area) as typeof area
      }
      type ExistingRuntime = {
        sendMessage?: (...args: unknown[]) => unknown
      }
      const createRuntime = (
        surface: string,
        existing: ExistingRuntime | undefined,
        preferCallbackDelegate: boolean
      ) => Object.assign(Object.create(existing ?? null), {
        sendMessage: (...inputArgs: unknown[]) => {
          const args = [...inputArgs]
          const callback = typeof args.at(-1) === "function"
            ? args.pop() as (value: unknown) => void
            : undefined
          record({
            kind: "runtime-message",
            surface,
            sourceCorrelated: sourceCorrelated(args)
          })
          const nativeSend = existing?.sendMessage
          let operation: Promise<unknown>
          if (typeof nativeSend !== "function") {
            operation = Promise.resolve(undefined)
          } else if (preferCallbackDelegate) {
            operation = new Promise((resolve, reject) => {
              let settled = false
              const settle = (value: unknown) => {
                if (settled) return
                settled = true
                resolve(value)
              }
              try {
                const returned = nativeSend.apply(existing, [...args, settle])
                if (returned && typeof (returned as PromiseLike<unknown>).then === "function") {
                  void Promise.resolve(returned as PromiseLike<unknown>).then(settle, reject)
                } else if (returned !== undefined) {
                  settle(returned)
                }
              } catch (error) {
                reject(error)
              }
            })
          } else {
            try {
              operation = Promise.resolve(nativeSend.apply(existing, args))
            } catch (error) {
              operation = Promise.reject(error)
            }
          }
          if (callback) void operation.then(callback)
          return operation
        }
      })
      const chromeLike: Record<string, unknown> = scope.chrome ?? {}
      const existingChromeStorage = chromeLike.storage as Record<string, ExistingStorageArea> | undefined
      chromeLike.storage = Object.assign({}, existingChromeStorage, {
        local: createStorageArea("chrome.local", existingChromeStorage?.local, true),
        sync: createStorageArea("chrome.sync", existingChromeStorage?.sync, true),
        session: createStorageArea("chrome.session", existingChromeStorage?.session, true)
      })
      const existingChromeRuntime = chromeLike.runtime as ExistingRuntime | undefined
      chromeLike.runtime = createRuntime("chrome.runtime", existingChromeRuntime, true)
      scope.chrome = chromeLike as SecurityWindow["chrome"]
      const browserLike: Record<string, unknown> = scope.browser ?? {}
      const existingBrowserStorage = browserLike.storage as Record<string, ExistingStorageArea> | undefined
      browserLike.storage = Object.assign({}, existingBrowserStorage, {
        local: createStorageArea("browser.local", existingBrowserStorage?.local, false),
        sync: createStorageArea("browser.sync", existingBrowserStorage?.sync, false),
        session: createStorageArea("browser.session", existingBrowserStorage?.session, false)
      })
      const existingBrowserRuntime = browserLike.runtime as ExistingRuntime | undefined
      browserLike.runtime = createRuntime("browser.runtime", existingBrowserRuntime, false)
      scope.browser = browserLike

      const wrapWorker = () => {
        const NativeWorker = window.Worker
        if (typeof NativeWorker !== "function") return
        const WorkerProxy = new Proxy(NativeWorker, {
          construct(target, args) {
            const workerUrl = String(args[0])
            const workerOptions = (args[1] ?? {}) as WorkerOptions
            const workerName = typeof workerOptions.name === "string"
              ? workerOptions.name.slice(0, 128)
              : undefined
            const workerType = workerOptions.type?.slice(0, 32) ?? "classic"
            const correlated = sourceCorrelated(args)
            record({
              kind: "worker-constructor",
              surface: "Worker",
              url: workerUrl,
              workerName,
              workerType,
              sourceCorrelated: correlated
            })
            const worker = Reflect.construct(target, args) as Worker
            const nativePostMessage = worker.postMessage.bind(worker)
            worker.postMessage = ((...postArgs: Parameters<Worker["postMessage"]>) => {
              const message = postArgs[0] as { type?: unknown } | null
              record({
                kind: "worker-post",
                surface: message && message.type === "extract"
                  ? "StandaloneHtmlOutlineWorker"
                  : "Worker",
                url: workerUrl,
                workerName,
                workerType,
                sourceCorrelated: sourceCorrelated(postArgs)
              })
              return nativePostMessage(...postArgs)
            }) as Worker["postMessage"]
            const nativeTerminate = worker.terminate.bind(worker)
            worker.terminate = (() => {
              record({
                kind: "worker-terminate",
                surface: "Worker",
                url: workerUrl,
                workerName,
                workerType,
                sourceCorrelated: false
              })
              return nativeTerminate()
            }) as Worker["terminate"]
            return worker
          }
        })
        Object.defineProperty(window, "Worker", {
          configurable: true,
          writable: true,
          value: WorkerProxy
        })
      }
      const wrapSharedWorker = () => {
        const NativeSharedWorker = window.SharedWorker
        if (typeof NativeSharedWorker !== "function") return
        const SharedWorkerProxy = new Proxy(NativeSharedWorker, {
          construct(target, args) {
            const workerUrl = String(args[0])
            const workerOptions = (args[1] ?? {}) as WorkerOptions
            const workerName = typeof workerOptions.name === "string"
              ? workerOptions.name.slice(0, 128)
              : undefined
            const workerType = workerOptions.type?.slice(0, 32) ?? "classic"
            record({
              kind: "worker-constructor",
              surface: "SharedWorker",
              url: workerUrl,
              workerName,
              workerType,
              sourceCorrelated: sourceCorrelated(args)
            })
            const worker = Reflect.construct(target, args) as SharedWorker
            const nativePostMessage = worker.port.postMessage.bind(worker.port)
            worker.port.postMessage = ((...postArgs: Parameters<MessagePort["postMessage"]>) => {
              record({
                kind: "worker-post",
                surface: "SharedWorker",
                url: workerUrl,
                workerName,
                workerType,
                sourceCorrelated: sourceCorrelated(postArgs)
              })
              return nativePostMessage(...postArgs)
            }) as MessagePort["postMessage"]
            return worker
          }
        })
        Object.defineProperty(window, "SharedWorker", {
          configurable: true,
          writable: true,
          value: SharedWorkerProxy
        })
      }
      wrapWorker()
      wrapSharedWorker()

      if (typeof ServiceWorker !== "undefined") {
        const nativeServiceWorkerPostMessage = ServiceWorker.prototype.postMessage
        ServiceWorker.prototype.postMessage = function postMessage(message, transfer) {
          record({
            kind: "service-worker-post",
            surface: "ServiceWorker",
            sourceCorrelated: sourceCorrelated([message, transfer])
          })
          return nativeServiceWorkerPostMessage.call(this, message, transfer)
        }
        record({ kind: "service-worker-calibration", surface: "prototype-postMessage" })
      }

      const nativeWebSocketSend = WebSocket.prototype.send
      WebSocket.prototype.send = function send(data) {
        record({
          kind: "websocket-send",
          surface: this.url,
          sourceCorrelated: sourceCorrelated(data)
        })
        return nativeWebSocketSend.call(this, data)
      }
      record({ kind: "message-sink-calibration", surface: "websocket-send" })

      if (typeof navigator.sendBeacon === "function") {
        const nativeSendBeacon = navigator.sendBeacon.bind(navigator)
        Object.defineProperty(navigator, "sendBeacon", {
          configurable: true,
          value: (url: string | URL, data?: BodyInit | null) => {
            record({
              kind: "send-beacon",
              url: String(url).slice(0, 4_096),
              sourceCorrelated: sourceCorrelated(data)
            })
            return nativeSendBeacon(url, data)
          }
        })
        record({ kind: "message-sink-calibration", surface: "sendBeacon" })
      }

      for (const method of ["pushState", "replaceState"] as const) {
        const nativeHistoryMethod = history[method].bind(history)
        history[method] = ((state: unknown, unused: string, url?: string | URL | null) => {
          record({
            kind: "history-state",
            surface: method,
            url: url == null ? "" : String(url).slice(0, 4_096),
            sourceCorrelated: sourceCorrelated(state)
          })
          return nativeHistoryMethod(state, unused, url)
        }) as History[typeof method]
      }
      record({ kind: "message-sink-calibration", surface: "history-state" })

      const innerHtml = Object.getOwnPropertyDescriptor(Element.prototype, "innerHTML")
      if (innerHtml?.get && innerHtml.set) {
        Object.defineProperty(Element.prototype, "innerHTML", {
          configurable: innerHtml.configurable,
          enumerable: innerHtml.enumerable,
          get: innerHtml.get,
          set(value: string) {
            const element = this as Element
            record({
              kind: "dom-html",
              surface: element.closest?.(".monaco-editor") ? "monaco" : "application",
              sourceCorrelated: sourceCorrelated(value)
            })
            innerHtml.set?.call(this, value)
          }
        })
      }
      const urlAttributes = new Set([
        "action",
        "data",
        "formaction",
        "href",
        "poster",
        "src",
        "srcset"
      ])
      const nativeSetAttribute = Element.prototype.setAttribute
      Element.prototype.setAttribute = function setAttribute(name, value) {
        const normalizedName = name.toLowerCase()
        if (normalizedName === "srcdoc") {
          record({
            kind: "dom-html",
            surface: `${this.tagName.toLowerCase()}.srcdoc`,
            sourceCorrelated: sourceCorrelated(value)
          })
        } else if (urlAttributes.has(normalizedName)) {
          const anchor = this instanceof HTMLAnchorElement ? this : null
          record({
            kind: "dom-url",
            surface: `${this.tagName.toLowerCase()}.${normalizedName}`,
            url: String(value).slice(0, 4_096),
            sourceCorrelated: sourceCorrelated(value),
            dataOwnedAnchor: anchor
              ? Object.prototype.hasOwnProperty.call(
                  anchor.dataset,
                  "standaloneHtmlDownload"
                )
              : undefined,
            download: anchor?.download,
            target: anchor?.target,
            rel: anchor?.rel
          })
        }
        return nativeSetAttribute.call(this, name, value)
      }
      const wrapUrlProperty = (
        prototype: object,
        property: string,
        surface: string
      ) => {
        const descriptor = Object.getOwnPropertyDescriptor(prototype, property)
        if (!descriptor?.get || !descriptor.set || descriptor.configurable === false) return
        Object.defineProperty(prototype, property, {
          ...descriptor,
          get: descriptor.get,
          set(this: object, value: unknown) {
            const anchor = this instanceof HTMLAnchorElement ? this : null
            record({
              kind: "dom-url",
              surface,
              url: String(value).slice(0, 4_096),
              sourceCorrelated: sourceCorrelated(value),
              dataOwnedAnchor: anchor
                ? Object.prototype.hasOwnProperty.call(
                    anchor.dataset,
                    "standaloneHtmlDownload"
                  )
                : undefined,
              download: anchor?.download,
              target: anchor?.target,
              rel: anchor?.rel
            })
            descriptor.set?.call(this, value)
          }
        })
      }
      for (const [prototype, property, surface] of [
        [HTMLAnchorElement.prototype, "href", "anchor.href"],
        [HTMLAreaElement.prototype, "href", "area.href"],
        [HTMLImageElement.prototype, "src", "image.src"],
        [HTMLImageElement.prototype, "srcset", "image.srcset"],
        [HTMLScriptElement.prototype, "src", "script.src"],
        [HTMLIFrameElement.prototype, "src", "iframe.src"],
        [HTMLLinkElement.prototype, "href", "link.href"],
        [HTMLSourceElement.prototype, "src", "source.src"],
        [HTMLSourceElement.prototype, "srcset", "source.srcset"],
        [HTMLFormElement.prototype, "action", "form.action"],
        [HTMLInputElement.prototype, "src", "input.src"],
        [HTMLInputElement.prototype, "formAction", "input.formAction"],
        [HTMLButtonElement.prototype, "formAction", "button.formAction"],
        [HTMLMediaElement.prototype, "src", "media.src"],
        [HTMLVideoElement.prototype, "poster", "video.poster"],
        [HTMLObjectElement.prototype, "data", "object.data"]
      ] as Array<[object, string, string]>) {
        wrapUrlProperty(prototype, property, surface)
      }
      const iframeSrcdoc = Object.getOwnPropertyDescriptor(
        HTMLIFrameElement.prototype,
        "srcdoc"
      )
      if (iframeSrcdoc?.get && iframeSrcdoc.set && iframeSrcdoc.configurable !== false) {
        Object.defineProperty(HTMLIFrameElement.prototype, "srcdoc", {
          ...iframeSrcdoc,
          get: iframeSrcdoc.get,
          set(this: HTMLIFrameElement, value: string) {
            record({
              kind: "dom-html",
              surface: "iframe.srcdoc",
              sourceCorrelated: sourceCorrelated(value)
            })
            iframeSrcdoc.set?.call(this, value)
          }
        })
      }
      const nativeStyleSetProperty = CSSStyleDeclaration.prototype.setProperty
      CSSStyleDeclaration.prototype.setProperty = function setProperty(
        property,
        value,
        priority
      ) {
        record({
          kind: "dom-url",
          surface: `style.${property}`,
          url: String(value).slice(0, 4_096),
          sourceCorrelated: sourceCorrelated(value)
        })
        return nativeStyleSetProperty.call(this, property, value, priority)
      }
      const calibrationAnchor = document.createElement("a")
      const calibrationUrl = "https://calibration.invalid/国際化"
      calibrationAnchor.setAttribute("href", calibrationUrl)
      if (calibrationAnchor.getAttribute("href") === calibrationUrl) {
        record({ kind: "url-sink-calibration", surface: "setAttribute" })
      }
      calibrationAnchor.href = `${calibrationUrl}/property`
      if (calibrationAnchor.href.startsWith("https://calibration.invalid/")) {
        record({ kind: "url-sink-calibration", surface: "href-property" })
      }
      const calibrationFrame = document.createElement("iframe")
      const calibrationMarkup = "<p>国際化 srcdoc calibration</p>"
      calibrationFrame.setAttribute("srcdoc", calibrationMarkup)
      if (calibrationFrame.getAttribute("srcdoc") === calibrationMarkup) {
        record({ kind: "html-sink-calibration", surface: "srcdoc-attribute" })
      }
      calibrationFrame.srcdoc = `${calibrationMarkup}<p>property</p>`
      if (calibrationFrame.srcdoc.includes("property")) {
        record({ kind: "html-sink-calibration", surface: "srcdoc-property" })
      }
      const nativeInsertAdjacentHtml = Element.prototype.insertAdjacentHTML
      Element.prototype.insertAdjacentHTML = function insertAdjacentHTML(position, text) {
        record({ kind: "dom-html", sourceCorrelated: sourceCorrelated(text) })
        return nativeInsertAdjacentHtml.call(this, position, text)
      }
      const nativeWrite = Document.prototype.write
      Document.prototype.write = function write(...values: string[]) {
        record({ kind: "document-write", sourceCorrelated: sourceCorrelated(values) })
        return nativeWrite.apply(this, values)
      }
      const nativeParseFromString = DOMParser.prototype.parseFromString
      DOMParser.prototype.parseFromString = function parseFromString(input, mimeType) {
        record({
          kind: "dom-parser",
          contentType: mimeType,
          sourceCorrelated: sourceCorrelated(input)
        })
        return nativeParseFromString.call(this, input, mimeType)
      }

      const NativeFunction = window.Function
      window.Function = new Proxy(NativeFunction, {
        apply(target, thisArg, args) {
          record({ kind: "dynamic-code", sourceCorrelated: sourceCorrelated(args) })
          return Reflect.apply(target, thisArg, args)
        },
        construct(target, args) {
          record({ kind: "dynamic-code", sourceCorrelated: sourceCorrelated(args) })
          return Reflect.construct(target, args)
        }
      }) as FunctionConstructor
      const nativeOpen = window.open
      window.open = ((...args: Parameters<typeof window.open>) => {
        record({
          kind: "window-open",
          url: args[0] == null ? "" : String(args[0]),
          sourceCorrelated: sourceCorrelated(args)
        })
        return nativeOpen.apply(window, args)
      }) as typeof window.open

      const nativeCreateObjectUrl = URL.createObjectURL.bind(URL)
      URL.createObjectURL = ((object: Blob | MediaSource) => {
        const objectUrl = nativeCreateObjectUrl(object)
        const event: EventRecord = {
          kind: "blob-create",
          at: performance.now(),
          url: objectUrl,
          contentType: object instanceof Blob ? object.type : "media-source",
          sourceCorrelated: false
        }
        if (events.length < 2_000) events.push(event)
        else scope.__task17SecurityOverflow = (scope.__task17SecurityOverflow ?? 0) + 1
        if (object instanceof NativeBlob) {
          const knownCorrelation = blobCorrelation.get(object)
          if (knownCorrelation !== undefined) {
            event.sourceCorrelated = knownCorrelation
            mirror(event)
          } else {
            const classification = object.text().then((text) => {
              event.sourceCorrelated = sourceCorrelated(text)
              mirror(event)
            })
            trackSecurityWork(classification)
          }
        } else {
          mirror(event)
        }
        return objectUrl
      }) as typeof URL.createObjectURL
      const nativeRevokeObjectUrl = URL.revokeObjectURL.bind(URL)
      URL.revokeObjectURL = ((objectUrl: string) => {
        record({ kind: "blob-revoke", url: objectUrl })
        return nativeRevokeObjectUrl(objectUrl)
      }) as typeof URL.revokeObjectURL
      if (calibrateBlobUrlSinks) {
        const blobAttributeCalibrationUrl = URL.createObjectURL(new Blob([
          "Task17 benign international Blob URL calibration: 章/節"
        ], { type: "text/plain" }))
        const blobAttributeCalibrationAnchor = document.createElement("a")
        blobAttributeCalibrationAnchor.setAttribute("href", blobAttributeCalibrationUrl)
        URL.revokeObjectURL(blobAttributeCalibrationUrl)
        const blobPropertyCalibrationUrl = URL.createObjectURL(new Blob([
          "Task17 benign international Blob property calibration: 版本 2.1"
        ], { type: "text/plain" }))
        const blobPropertyCalibrationArea = document.createElement("area")
        blobPropertyCalibrationArea.href = blobPropertyCalibrationUrl
        URL.revokeObjectURL(blobPropertyCalibrationUrl)
        const blobCompositeCalibrationUrl = URL.createObjectURL(new Blob([
          "Task17 benign composite Blob URL calibration"
        ], { type: "text/plain" }))
        const blobCompositeCalibrationElement = document.createElement("div")
        blobCompositeCalibrationElement.style.setProperty(
          "background-image",
          `url("${blobCompositeCalibrationUrl}")`
        )
        URL.revokeObjectURL(blobCompositeCalibrationUrl)
      }
      const nativeAnchorClick = HTMLAnchorElement.prototype.click
      HTMLAnchorElement.prototype.click = function click() {
        record({
          kind: "anchor-click",
          url: this.href,
          dataOwnedAnchor: Object.prototype.hasOwnProperty.call(
            this.dataset,
            "standaloneHtmlDownload"
          ),
          download: this.download,
          target: this.target,
          rel: this.rel
        })
        return nativeAnchorClick.call(this)
      }
      const nativeRemove = Element.prototype.remove
      Element.prototype.remove = function remove() {
        if (this instanceof HTMLAnchorElement) {
          record({
            kind: "anchor-remove",
            url: this.href,
            dataOwnedAnchor: Object.prototype.hasOwnProperty.call(
              this.dataset,
              "standaloneHtmlDownload"
            )
          })
        }
        return nativeRemove.call(this)
      }

      for (const level of ["log", "info", "warn", "error"] as const) {
        const native = console[level].bind(console)
        console[level] = (...args: unknown[]) => {
          record({
            kind: "console",
            surface: level,
            sourceCorrelated: sourceCorrelated(args)
          })
          native(...args)
        }
      }

      const config = {
        serverUrl,
        authMode,
        ...(authMode === "multi-user" ? { accessToken } : { apiKey })
      }
      if (!localStorage.getItem("tldwConfig")) {
        localStorage.setItem("tldwConfig", JSON.stringify(config))
        localStorage.setItem("serverUrl", serverUrl)
        localStorage.setItem("tldwServerUrl", serverUrl)
        localStorage.setItem("tldw-api-host", serverUrl)
        localStorage.setItem("authMode", authMode)
        if (authMode === "multi-user") {
          localStorage.setItem("accessToken", accessToken)
          localStorage.removeItem("apiKey")
        } else {
          localStorage.setItem("apiKey", apiKey)
          localStorage.removeItem("accessToken")
        }
      }
      localStorage.setItem("isMigrated", "true")
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("assistant_setup_dismissed", "true")
      localStorage.setItem("__tldw_test_bypass", "true")

      const calibration = "国際化 storage calibration"
      localStorage.setItem("__task17_local_probe__", calibration)
      localStorage.removeItem("__task17_local_probe__")
      sessionStorage.setItem("__task17_session_probe__", calibration)
      sessionStorage.removeItem("__task17_session_probe__")
      type CalibrationArea = {
        get: (
          key: string,
          callback?: (value: Record<string, unknown>) => void
        ) => Promise<Record<string, unknown>>
        set: (value: Record<string, unknown>, callback?: () => void) => Promise<void>
        remove: (key: string, callback?: () => void) => Promise<void>
      }
      const storageSurfaces: Array<{ surface: string; area: CalibrationArea }> = [
        { surface: "chrome.local", area: (chromeLike.storage as { local: CalibrationArea }).local },
        { surface: "chrome.sync", area: (chromeLike.storage as { sync: CalibrationArea }).sync },
        { surface: "chrome.session", area: (chromeLike.storage as { session: CalibrationArea }).session },
        { surface: "browser.local", area: (browserLike.storage as { local: CalibrationArea }).local },
        { surface: "browser.sync", area: (browserLike.storage as { sync: CalibrationArea }).sync },
        { surface: "browser.session", area: (browserLike.storage as { session: CalibrationArea }).session }
      ]
      for (const { surface, area } of storageSurfaces) {
        const promiseKey = `__task17_promise_probe_${surface}__`
        void area.set({ [promiseKey]: calibration })
          .then(() => area.get(promiseKey))
          .then((result) => {
            if (result[promiseKey] === calibration) {
              record({ kind: "storage-calibration", surface: `${surface}:promise` })
            }
            return area.remove(promiseKey)
          })
        const callbackKey = `__task17_callback_probe_${surface}__`
        void area.set({ [callbackKey]: calibration }, () => {
          void area.get(callbackKey, (result) => {
            if (result[callbackKey] === calibration) {
              record({ kind: "storage-calibration", surface: `${surface}:callback` })
            }
            void area.remove(callbackKey)
          })
        })
      }
      const chromeLocal = (chromeLike.storage as { local: CalibrationArea }).local
      void chromeLocal.get("__tldw_first_run_complete", (result) => {
        if (result.__tldw_first_run_complete === true) {
          record({
            kind: "storage-shim-calibration",
            surface: "preseeded-localStorage-value"
          })
        }
      })
      void chromeLocal.get("__task17_missing_storage_key__", (result) => {
        if (!Object.prototype.hasOwnProperty.call(result, "__task17_missing_storage_key__")) {
          record({ kind: "storage-shim-calibration", surface: "missing-key-omitted" })
        }
      })
      type CalibrationRuntime = {
        sendMessage: (
          value: object,
          callback?: (result: unknown) => void
        ) => Promise<unknown>
      }
      for (const { surface, runtimeArea } of [
        {
          surface: "chrome.runtime",
          runtimeArea: chromeLike.runtime as CalibrationRuntime
        },
        {
          surface: "browser.runtime",
          runtimeArea: browserLike.runtime as CalibrationRuntime
        }
      ]) {
        void runtimeArea.sendMessage({ __task17_promise_probe__: calibration }).then(() => {
          record({ kind: "runtime-calibration", surface: `${surface}:promise` })
        })
        void runtimeArea.sendMessage(
          { __task17_callback_probe__: calibration },
          () => record({ kind: "runtime-calibration", surface: `${surface}:callback` })
        )
      }
    },
    {
      marker: fixture.marker,
      serverUrl: apiOrigin,
      apiKey: TEST_API_KEY,
      authMode: options.authMode ?? "single-user",
      accessToken: options.accessToken ?? "task17-owner-42-token",
      calibrateBlobUrlSinks: options.calibrateBlobUrlSinks ?? false
    }
  )
  return aggregate
}

const installDeferredDigestAndPageShowProbe = async (
  context: BrowserContext,
  deferredMarker: string,
  sourceMarker: string
): Promise<void> => {
  await context.addInitScript(({ digestMarker, visibleMarker }) => {
    type Task17Window = typeof window & {
      __task17DigestDeferred?: boolean
      __task17ReleaseDigest?: () => void
      __task17PageShows?: Array<{
        persisted: boolean
        sourceVisible: boolean
        bodySourceVisible: boolean
        monacoSourceVisible: boolean
        formSourceVisible: boolean
        historyStateSourceVisible: boolean
      }>
      monaco?: { editor?: { getModels?: () => Array<{ getValue: () => string }> } }
    }
    const scope = window as Task17Window
    scope.__task17PageShows = []
    window.addEventListener("pageshow", (event) => {
      const bodySourceVisible = (document.body?.innerText ?? "").includes(visibleMarker)
      const monacoSourceVisible = (scope.monaco?.editor?.getModels?.() ?? []).some((model) =>
        model.getValue().includes(visibleMarker)
      )
      const formSourceVisible = Array.from(document.querySelectorAll("input, textarea")).some(
        (element) => (element as HTMLInputElement | HTMLTextAreaElement).value.includes(visibleMarker)
      )
      let historyStateSourceVisible = false
      try {
        historyStateSourceVisible = JSON.stringify(history.state)?.slice(0, 1_048_576)
          .includes(visibleMarker) ?? false
      } catch {
        historyStateSourceVisible = true
      }
      scope.__task17PageShows?.push({
        persisted: event.persisted,
        sourceVisible: bodySourceVisible || monacoSourceVisible || formSourceVisible ||
          historyStateSourceVisible,
        bodySourceVisible,
        monacoSourceVisible,
        formSourceVisible,
        historyStateSourceVisible
      })
    })
    const nativeDigest = crypto.subtle.digest.bind(crypto.subtle)
    let hasDeferred = false
    crypto.subtle.digest = ((algorithm: AlgorithmIdentifier, data: BufferSource) => {
      let text = ""
      try {
        const bytes = data instanceof ArrayBuffer
          ? new Uint8Array(data)
          : new Uint8Array(data.buffer, data.byteOffset, data.byteLength)
        text = new TextDecoder().decode(bytes)
      } catch {
        text = ""
      }
      if (hasDeferred || !text.includes(digestMarker)) return nativeDigest(algorithm, data)
      hasDeferred = true
      scope.__task17DigestDeferred = true
      return new Promise<ArrayBuffer>((resolve, reject) => {
        scope.__task17ReleaseDigest = () => {
          scope.__task17ReleaseDigest = undefined
          void nativeDigest(algorithm, data).then(resolve, reject)
        }
      })
    }) as SubtleCrypto["digest"]
  }, {
    digestMarker: deferredMarker,
    visibleMarker: sourceMarker
  })
}

type SourcePresenceProbe = {
  persisted: boolean
  sourceVisible: boolean
  bodySourceVisible: boolean
  monacoSourceVisible: boolean
  formSourceVisible: boolean
  historyStateSourceVisible: boolean
}

const installPageShowSourceProbe = async (
  context: BrowserContext,
  sourceMarker: string
): Promise<void> => {
  await context.addInitScript(({ visibleMarker }) => {
    type ProbeWindow = typeof window & {
      __task17PageShows?: SourcePresenceProbe[]
      __task17VisibilityStates?: string[]
      monaco?: { editor?: { getModels?: () => Array<{ getValue: () => string }> } }
    }
    const scope = window as ProbeWindow
    scope.__task17PageShows = []
    scope.__task17VisibilityStates = [document.visibilityState]
    document.addEventListener("visibilitychange", () => {
      scope.__task17VisibilityStates?.push(document.visibilityState)
    })
    window.addEventListener("pageshow", (event) => {
      const bodySourceVisible = (document.body?.innerText ?? "").includes(visibleMarker)
      const monacoSourceVisible = (scope.monaco?.editor?.getModels?.() ?? []).some((model) =>
        model.getValue().includes(visibleMarker)
      )
      const formSourceVisible = Array.from(document.querySelectorAll("input, textarea")).some(
        (element) => (element as HTMLInputElement | HTMLTextAreaElement).value.includes(visibleMarker)
      )
      let historyStateSourceVisible = false
      try {
        historyStateSourceVisible = JSON.stringify(history.state)?.slice(0, 1_048_576)
          .includes(visibleMarker) ?? false
      } catch {
        historyStateSourceVisible = true
      }
      scope.__task17PageShows?.push({
        persisted: event.persisted,
        sourceVisible: bodySourceVisible || monacoSourceVisible || formSourceVisible ||
          historyStateSourceVisible,
        bodySourceVisible,
        monacoSourceVisible,
        formSourceVisible,
        historyStateSourceVisible
      })
    })
  }, { visibleMarker: sourceMarker })
}

const logoutThenLoginAsPrincipal = async (
  page: Page,
  protocol: SecurityProtocolServer,
  principalId: number
) => {
  protocol.state.authenticated = false
  await page.evaluate(() => {
    const raw = localStorage.getItem("tldwConfig")
    const config = raw ? JSON.parse(raw) as Record<string, unknown> : {}
    config.authMode = "multi-user"
    delete config.accessToken
    delete config.refreshToken
    localStorage.removeItem("accessToken")
    localStorage.setItem("tldwConfig", JSON.stringify(config))
    const prefixes = [
      "tldw:presentation-studio:html:draft:v1:",
      "tldw:presentation-studio:html:resume:v1:"
    ]
    for (let index = sessionStorage.length - 1; index >= 0; index -= 1) {
      const key = sessionStorage.key(index)
      if (key && prefixes.some((prefix) => key.startsWith(prefix))) {
        sessionStorage.removeItem(key)
      }
    }
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "logout" }
    }))
  })

  protocol.state.principalId = principalId
  protocol.state.authenticated = true
  await page.evaluate(({ nextPrincipal }) => {
    const raw = localStorage.getItem("tldwConfig")
    const config = raw ? JSON.parse(raw) as Record<string, unknown> : {}
    const token = `task17-owner-${nextPrincipal}-token`
    config.authMode = "multi-user"
    config.accessToken = token
    delete config.refreshToken
    localStorage.setItem("accessToken", token)
    localStorage.setItem("tldwConfig", JSON.stringify(config))
    window.dispatchEvent(new CustomEvent("tldw:auth-principal-changed", {
      detail: { kind: "login" }
    }))
  }, { nextPrincipal: principalId })
}

const assertPersistedProbeSourceFree = async (
  page: Page,
  testInfo: TestInfo,
  surface: string
) => {
  const pageShows = await page.evaluate(() => (
    window as typeof window & { __task17PageShows?: SourcePresenceProbe[] }
  ).__task17PageShows ?? [])
  const persisted = pageShows.filter((entry) => entry.persisted)
  if (persisted.length === 0) {
    testInfo.annotations.push({
      type: "engine-limitation",
      description: `${testInfo.project.name} did not restore the clean ${surface} history entry from bfcache.`
    })
    return
  }
  expect(persisted.every((entry) =>
    entry.sourceVisible === false &&
    entry.bodySourceVisible === false &&
    entry.monacoSourceVisible === false &&
    entry.formSourceVisible === false &&
    entry.historyStateSourceVisible === false
  )).toBe(true)
  const visibilityStates = await page.evaluate(() => (
    window as typeof window & { __task17VisibilityStates?: string[] }
  ).__task17VisibilityStates ?? [])
  const hiddenIndex = visibilityStates.indexOf("hidden")
  expect(hiddenIndex).toBeGreaterThanOrEqual(0)
  expect(visibilityStates.slice(hiddenIndex + 1)).toContain("visible")
}

const assertCurrentSourceAbsent = async (page: Page, sourceMarker: string) => {
  const presence = await page.evaluate((marker) => {
    const scope = window as typeof window & {
      monaco?: { editor?: { getModels?: () => Array<{ getValue: () => string }> } }
    }
    let historyStateSourceVisible = false
    try {
      historyStateSourceVisible = JSON.stringify(history.state)?.slice(0, 1_048_576)
        .includes(marker) ?? false
    } catch {
      historyStateSourceVisible = true
    }
    return {
      body: (document.body?.innerText ?? "").includes(marker),
      monaco: (scope.monaco?.editor?.getModels?.() ?? []).some((model) =>
        model.getValue().includes(marker)
      ),
      form: Array.from(document.querySelectorAll("input, textarea")).some((element) => (
        element as HTMLInputElement | HTMLTextAreaElement
      ).value.includes(marker)),
      history: historyStateSourceVisible
    }
  }, sourceMarker)
  expect(presence).toEqual({ body: false, monaco: false, form: false, history: false })
}

const pageSecurityEvents = (page: Page): Promise<SecurityEvent[]> =>
  page.evaluate(() => (
    window as typeof window & { __task17SecurityEvents?: SecurityEvent[] }
  ).__task17SecurityEvents ?? [])

const pageSecurityOverflow = (page: Page): Promise<number> =>
  page.evaluate(() => (
    window as typeof window & { __task17SecurityOverflow?: number }
  ).__task17SecurityOverflow ?? 0)

const flushPageSecurityAggregate = (page: Page): Promise<void> =>
  page.evaluate(async () => {
    const flush = (window as typeof window & {
      __task17FlushSecurityEvents?: () => Promise<void>
    }).__task17FlushSecurityEvents
    if (!flush) throw new Error("Task17 security aggregate flush is unavailable")
    await flush()
  })

const recoveryKey = (
  apiOrigin: string,
  principalId: string | number,
  presentationId: string
): string => `${RECOVERY_PREFIX}${encodeURIComponent(apiOrigin)}:${encodeURIComponent(
  String(principalId)
)}:${encodeURIComponent(presentationId)}`

const generationRecoveryKeys = (apiOrigin: string, principalId: string | number) => {
  const namespace = `${encodeURIComponent(apiOrigin)}:${encodeURIComponent(String(principalId))}`
  return {
    draft: `tldw:presentation-studio:html:draft:v1:${namespace}`,
    resume: `tldw:presentation-studio:html:resume:v1:${namespace}`
  }
}

const assertInstrumentationReady = async (
  page: Page,
  options: { expectBlobUrlSinkCalibration?: boolean } = {}
): Promise<SecurityEvent[]> => {
  await expect.poll(async () => page.evaluate(() => Array.isArray(
    (window as typeof window & { __task17SecurityEvents?: unknown }).__task17SecurityEvents
  ))).toBe(true)
  await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
    event.kind === "storage-calibration"
  ).length).toBe(12)
  await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
    event.kind === "runtime-calibration"
  ).length).toBe(4)
  await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
    event.kind === "storage-shim-calibration"
  ).length).toBe(2)
  await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
    event.kind === "url-sink-calibration"
  ).length).toBe(2)
  await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
    event.kind === "html-sink-calibration"
  ).length).toBe(2)
  if (options.expectBlobUrlSinkCalibration) {
    await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
      event.kind === "dom-url" &&
      event.url?.includes("blob:") &&
      event.sourceCorrelated === false
    ).map((event) => event.surface)).toEqual(expect.arrayContaining([
      "a.href",
      "area.href",
      "style.background-image"
    ]))
  }
  await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
    event.kind === "message-sink-calibration"
  ).length).toBeGreaterThanOrEqual(2)
  const events = await pageSecurityEvents(page)
  expect(events.filter((event) => event.kind === "storage-calibration").map((event) =>
    event.surface
  )).toEqual(expect.arrayContaining([
    "chrome.local:promise",
    "chrome.local:callback",
    "chrome.sync:promise",
    "chrome.sync:callback",
    "chrome.session:promise",
    "chrome.session:callback",
    "browser.local:promise",
    "browser.local:callback",
    "browser.sync:promise",
    "browser.sync:callback",
    "browser.session:promise",
    "browser.session:callback"
  ]))
  const logicalSurfaces = events
    .filter((event) => event.kind === "logical-storage-set")
    .map((event) => event.surface)
  expect(logicalSurfaces).toEqual(expect.arrayContaining([
    "chrome.local",
    "chrome.sync",
    "chrome.session",
    "browser.local",
    "browser.sync",
    "browser.session"
  ]))
  expect(events.filter((event) => event.kind === "runtime-message").map((event) => event.surface))
    .toEqual(expect.arrayContaining(["chrome.runtime", "browser.runtime"]))
  expect(events.filter((event) => event.kind === "runtime-calibration").map((event) =>
    event.surface
  )).toEqual(expect.arrayContaining([
    "chrome.runtime:promise",
    "chrome.runtime:callback",
    "browser.runtime:promise",
    "browser.runtime:callback"
  ]))
  expect(events.filter((event) => event.kind === "storage-shim-calibration").map((event) =>
    event.surface
  )).toEqual(expect.arrayContaining([
    "preseeded-localStorage-value",
    "missing-key-omitted"
  ]))
  expect(events.filter((event) => event.kind === "url-sink-calibration").map((event) =>
    event.surface
  )).toEqual(expect.arrayContaining(["setAttribute", "href-property"]))
  expect(events.filter((event) => event.kind === "html-sink-calibration").map((event) =>
    event.surface
  )).toEqual(expect.arrayContaining(["srcdoc-attribute", "srcdoc-property"]))
  expect(await page.evaluate(() => typeof Object.getOwnPropertyDescriptor(
    window,
    "__TASK17_EXECUTED__"
  )?.set)).toBe("function")
  expect(events.some((event) => event.kind === "storage-set" && event.surface === "localStorage"))
    .toBe(true)
  expect(events.some((event) => event.kind === "storage-set" && event.surface === "sessionStorage"))
    .toBe(true)
  expect(await pageSecurityOverflow(page)).toBe(0)
  return events
}

const navigationHeaders = async (response: Awaited<ReturnType<Page["goto"]>>) =>
  response ? response.allHeaders() : {}

const browserPrimaryModifier = (page: Page): Promise<"Meta" | "Control"> =>
  page.evaluate(() =>
    /Macintosh|Mac OS X|iPhone|iPad|iPod/i.test(navigator.userAgent)
      ? "Meta"
      : "Control"
  )

const openSecurityWorkspace = async (input: {
  page: Page
  context: BrowserContext
  fixture: SecurityFixture
  protocol: SecurityProtocolServer
  authMode?: "single-user" | "multi-user"
  beforeMainNavigation?: () => Promise<void>
}) => {
  const aggregate = await installPageSecurityInstrumentation(
    input.context,
    input.fixture,
    input.protocol.origin,
    { authMode: input.authMode }
  )
  const studioProbe = await input.context.newPage()
  let studioHeaders: Record<string, string> = {}
  let studioCspMeta: string | null = null
  try {
    await studioProbe.route(`${input.protocol.origin}/api/v1/**`, (route) => route.abort())
    const studioResponse = await studioProbe.goto(
      `/presentation-studio/${encodeURIComponent(input.fixture.presentationId)}`,
      { waitUntil: "domcontentloaded" }
    )
    studioHeaders = await navigationHeaders(studioResponse)
    studioCspMeta = await studioProbe.evaluate(() =>
      document.querySelector('meta[http-equiv="Content-Security-Policy" i]')
        ?.getAttribute("content") ?? null
    )
    await flushPageSecurityAggregate(studioProbe)
  } finally {
    await studioProbe.close()
  }

  await input.beforeMainNavigation?.()
  const observations = installContextObservability(input.context, input.fixture)
  const controlResponse = await input.page.goto("/presentation-studio", {
    waitUntil: "domcontentloaded"
  })
  const controlHeaders = await navigationHeaders(controlResponse)
  const controlCspMeta = await input.page.evaluate(() =>
    document.querySelector('meta[http-equiv="Content-Security-Policy" i]')
      ?.getAttribute("content") ?? null
  )
  const detailApiResponsePromise = input.page.waitForResponse((response) => {
    const url = new URL(response.url())
    return url.origin === input.protocol.origin &&
      url.pathname === `/api/v1/slides/presentations/${input.fixture.presentationId}` &&
      response.request().method() === "GET"
  })
  const currentTitle = input.protocol.state.source.match(/<title>([^<]+)<\/title>/)?.[1] ??
    "Standalone HTML"
  await input.page.getByRole("button", { name: `Open ${currentTitle}` }).click()
  await expect(input.page).toHaveURL(new RegExp(
    `/presentation-studio/${input.fixture.presentationId}$`
  ))
  const detailApiResponse = await detailApiResponsePromise
  await assertInstrumentationReady(input.page)
  return {
    aggregate,
    observations,
    detailApiResponse,
    controlCsp: controlHeaders["content-security-policy"] ?? null,
    studioCsp: studioHeaders["content-security-policy"] ?? null,
    controlCspMeta,
    studioCspMeta
  }
}

const replaceMonacoSource = async (page: Page, source: string) => {
  const editor = page.locator(".monaco-editor").first()
  await expect(editor).toBeVisible()
  const focusedVisibleEditor = await editor.evaluate((element) => {
    const monaco = (window as typeof window & {
      monaco?: {
        editor?: {
          getEditors?: () => Array<{
            focus: () => void
            getDomNode: () => HTMLElement | null
          }>
        }
      }
    }).monaco
    const visibleEditor = monaco?.editor?.getEditors?.().find((candidate) => {
      const node = candidate.getDomNode()
      return node === element || Boolean(node && (
        node.contains(element) || element.contains(node)
      ))
    })
    visibleEditor?.focus()
    return Boolean(visibleEditor && element.contains(document.activeElement))
  })
  expect(focusedVisibleEditor).toBe(true)
  const modifier = await browserPrimaryModifier(page)
  await page.keyboard.press(`${modifier}+A`)
  const selection = await page.evaluate(() => {
    const monaco = (window as typeof window & {
      monaco?: {
        editor?: {
          getEditors?: () => Array<{
            hasTextFocus: () => boolean
            getModel: () => { getFullModelRange: () => Record<string, number> } | null
            getSelection: () => Record<string, number> | null
          }>
        }
      }
    }).monaco
    const activeEditor = monaco?.editor?.getEditors?.().find((candidate) =>
      candidate.hasTextFocus()
    )
    const selected = activeEditor?.getSelection()
    const full = activeEditor?.getModel()?.getFullModelRange()
    if (!selected || !full) return null
    return {
      selected: {
        startLineNumber: selected.startLineNumber,
        startColumn: selected.startColumn,
        endLineNumber: selected.endLineNumber,
        endColumn: selected.endColumn
      },
      full: {
        startLineNumber: full.startLineNumber,
        startColumn: full.startColumn,
        endLineNumber: full.endLineNumber,
        endColumn: full.endColumn
      }
    }
  })
  expect(selection?.selected).toEqual(selection?.full)
  const replacedSource = await editor.evaluate((element, input) => {
    const monaco = (window as typeof window & {
      monaco?: {
        editor?: {
          getEditors?: () => Array<{
            executeEdits: (
              sourceId: string,
              edits: Array<{ range: Record<string, number>; text: string }>
            ) => boolean
            getDomNode: () => HTMLElement | null
            getModel: () => { getFullModelRange: () => Record<string, number> } | null
            getValue: () => string
          }>
        }
      }
    }).monaco
    const visibleEditor = monaco?.editor?.getEditors?.().find((candidate) => {
      const node = candidate.getDomNode()
      return node === element || Boolean(node && (
        node.contains(element) || element.contains(node)
      ))
    })
    const model = visibleEditor?.getModel()
    if (!visibleEditor || !model) return null
    visibleEditor.executeEdits("task17-replace-source", [{
      range: model.getFullModelRange(),
      text: input
    }])
    return visibleEditor.getValue()
  }, source)
  expect(replacedSource).toBe(source)
  await expect.poll(async () => editor.evaluate((element) => {
    const monaco = (window as typeof window & {
      monaco?: {
        editor?: {
          getEditors?: () => Array<{
            getDomNode: () => HTMLElement | null
            getValue: () => string
          }>
        }
      }
    }).monaco
    return monaco?.editor?.getEditors?.().find((candidate) => {
      const node = candidate.getDomNode()
      return node === element || Boolean(node && (
        node.contains(element) || element.contains(node)
      ))
    })?.getValue()
  })).toBe(source)
}

const isApprovedOutlineWorkerEvent = (
  event: SecurityEvent,
  applicationWorkerOrigin: string
): boolean => {
  if (
    event.workerName !== "StandaloneHtmlOutlineWorker" ||
    event.workerType !== "classic" ||
    (event.kind === "worker-post" && event.surface !== "StandaloneHtmlOutlineWorker") ||
    !event.url
  ) return false
  if (event.url.startsWith(`blob:${applicationWorkerOrigin}`)) return true
  try {
    const workerUrl = new URL(event.url)
    if (workerUrl.origin !== applicationWorkerOrigin) return false
    return workerUrl.pathname === "/_next/static/chunks/StandaloneHtmlOutlineWorker.js"
  } catch {
    return false
  }
}

const sourceBearingForbiddenEvents = (
  events: SecurityEvent[],
  expectedRecoveryKey?: string | string[],
  expectedDownloadBlobUrl?: string,
  applicationWorkerOrigin?: string
) => events.filter((event) => {
  if (!event.sourceCorrelated) return false
  if (event.kind === "worker-post") {
    if (!event.url || !applicationWorkerOrigin) return true
    if (isApprovedOutlineWorkerEvent(event, applicationWorkerOrigin)) return false
    let isApplicationOwned = false
    let isApplicationBlob = false
    try {
      isApplicationBlob = event.url.startsWith(`blob:${applicationWorkerOrigin}`)
      isApplicationOwned = isApplicationBlob ||
        new URL(event.url).origin === applicationWorkerOrigin
    } catch {
      return true
    }
    if (!isApplicationOwned) return true
    if (event.workerType !== "module" && !(event.workerType === "classic" && isApplicationBlob)) {
      return true
    }
    return event.workerName !== "editorWorkerService"
  }
  if (
    event.kind === "storage-set" &&
    event.surface === "sessionStorage" &&
    (Array.isArray(expectedRecoveryKey)
      ? expectedRecoveryKey.includes(event.url ?? "")
      : event.url === expectedRecoveryKey)
  ) return false
  if (
    event.kind === "blob-create" &&
    event.url === expectedDownloadBlobUrl &&
    event.contentType === "application/octet-stream"
  ) return false
  return [
    "dom-html",
    "dom-parser",
    "dom-url",
    "document-write",
    "dynamic-code",
    "window-open",
    "console",
    "storage-set",
    "logical-storage-set",
    "runtime-message",
    "blob-create",
    "worker-constructor",
    "service-worker-post",
    "websocket-send",
    "send-beacon",
    "history-state"
  ].includes(event.kind)
})

const assertTerminalSecurityAudit = async (input: {
  page: Page
  aggregate: SecurityAggregate
  observations: ContextObservations
  protocol: SecurityProtocolServer
  fixture: SecurityFixture
  expectedRecoveryKey?: string | string[]
  expectedDownloadBlobUrl?: string
}) => {
  await flushPageSecurityAggregate(input.page)
  expect(await pageSecurityOverflow(input.page)).toBe(0)
  expect(input.aggregate.overflowCount).toBe(0)
  expect(input.aggregate.events.filter((event) => event.kind === "sentinel-execution"))
    .toEqual([])
  expect(sourceBearingForbiddenEvents(
    input.aggregate.events,
    input.expectedRecoveryKey,
    input.expectedDownloadBlobUrl,
    input.fixture.webOrigin
  )).toEqual([])
  expect(input.observations.overflowCount).toBe(0)
  expect(input.protocol.state.overflowCount).toBe(0)
  expect(input.observations.childNavigations).toEqual([])
  expect(input.observations.errors.filter((value) =>
    value.includes(input.fixture.marker) || value.includes(input.fixture.sentinelOrigin)
  )).toEqual([])
  const presentationBase = `${input.protocol.origin}/api/v1/slides/presentations/${encodeURIComponent(
    input.fixture.presentationId
  )}`
  expect(input.observations.requests.filter((request) => {
    if (!request.sourceCorrelated) return false
    return !(
      (request.method === "PUT" && request.url === `${presentationBase}/html-source`) ||
      (request.method === "POST" && request.url === `${presentationBase}/draft-attachment`) ||
      (
        request.method === "POST" &&
        request.url === `${input.protocol.origin}/api/v1/slides/generations` &&
        request.contentType?.startsWith("application/json")
      )
    )
  })).toEqual([])
}

test.describe("Standalone HTML Presentation Studio browser security", () => {
  test.describe.configure({ retries: 0 })
  test("observes bounded Blob URL assignments without treating benign Blob data as source", async ({
    context,
    page
  }, testInfo) => {
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    const probePath = "/__task17_blob_url_probe__"
    const probeRoute = async (route: Route) => route.fulfill({
      status: 200,
      contentType: "text/html; charset=utf-8",
      body: "<!doctype html><html><head><title>Task17 Blob URL probe</title></head><body><main>Source-free probe</main></body></html>"
    })
    try {
      await installPageSecurityInstrumentation(context, fixture, protocol.origin, {
        calibrateBlobUrlSinks: true
      })
      await context.route(`**${probePath}`, probeRoute)
      await page.goto(probePath, { waitUntil: "domcontentloaded" })
      const events = await assertInstrumentationReady(page, {
        expectBlobUrlSinkCalibration: true
      })
      const benignBlobAssignments = events.filter((event) =>
        event.kind === "dom-url" &&
        event.url?.includes("blob:") &&
        event.sourceCorrelated === false
      )
      expect(benignBlobAssignments.map((event) => event.surface)).toEqual(
        expect.arrayContaining(["a.href", "area.href", "style.background-image"])
      )
      expect(sourceBearingForbiddenEvents(
        events,
        undefined,
        undefined,
        fixture.webOrigin
      )).toEqual([])
    } finally {
      await context.unroute(`**${probePath}`, probeRoute)
      await protocol.close()
    }
  })

  test("uses real CORS while generated and editor URL-bearing source stay inert", async ({
    context,
    page
  }, testInfo) => {
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    try {
      const opened = await openSecurityWorkspace({ page, context, fixture, protocol })
      await expect(page.getByRole("heading", {
        level: 1,
        name: "Task17 generated source"
      })).toBeVisible()
      const editor = page.locator(".monaco-editor").first()
      await expect(editor).toBeVisible()
      await expect(page.getByRole("heading", {
        level: 2,
        name: "Safe outline: text only; code never runs in Studio"
      })).toBeVisible()
      await expect(page.getByText(fixture.marker, { exact: true })).toBeVisible()
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()

      await replaceMonacoSource(page, fixture.editorSource)
      await expect(page.getByText("Trusted editor outline", { exact: true })).toBeVisible()
      const outlineRegion = page.getByRole("heading", {
        level: 2,
        name: "Safe outline: text only; code never runs in Studio"
      }).locator("xpath=ancestor::section[1]")
      await expect(outlineRegion.locator(
        "script, style, svg, math, template, a, img, iframe, object, embed, form, input, button"
      )).toHaveCount(0)
      await expect(outlineRegion).not.toContainText(fixture.sentinelOrigin)
      await expect(outlineRegion).not.toContainText("data:text/")
      const outlineDomAudit = await outlineRegion.evaluate((root, markers) => {
        const allowedAttribute = (name: string) =>
          name === "class" || name === "dir" || name === "id" || name === "role" ||
          name.startsWith("aria-")
        const elements = [root, ...Array.from(root.querySelectorAll("*"))]
        return {
          disallowedAttributes: elements.flatMap((element) =>
            Array.from(element.attributes)
              .filter((attribute) => !allowedAttribute(attribute.name))
              .map((attribute) => `${element.tagName.toLowerCase()}.${attribute.name}`)
          ),
          correlatedAttributes: elements.flatMap((element) =>
            Array.from(element.attributes)
              .filter((attribute) => markers.some((marker) => attribute.value.includes(marker)))
              .map((attribute) => `${element.tagName.toLowerCase()}.${attribute.name}`)
          )
        }
      }, [fixture.marker, fixture.sentinelOrigin])
      expect(outlineDomAudit).toEqual({
        disallowedAttributes: [],
        correlatedAttributes: []
      })

      const modifier = await browserPrimaryModifier(page)
      const sentinelCursor = await editor.evaluate((element, needle) => {
        const monaco = (window as typeof window & {
          monaco?: {
            editor?: {
              getEditors?: () => Array<{
                focus: () => void
                getDomNode: () => HTMLElement | null
                getModel: () => {
                  getPositionAt: (offset: number) => Record<string, number>
                  getValue: () => string
                } | null
                getValue: () => string
                revealPositionInCenter: (position: Record<string, number>) => void
                setPosition: (position: Record<string, number>) => void
              }>
            }
          }
        }).monaco
        const visibleEditor = monaco?.editor?.getEditors?.().find((candidate) => {
          const node = candidate.getDomNode()
          return node === element || Boolean(node && (
            node.contains(element) || element.contains(node)
          ))
        })
        const model = visibleEditor?.getModel()
        if (!visibleEditor || !model) return null
        const offset = visibleEditor.getValue().indexOf(needle)
        if (offset < 0) return null
        const position = model.getPositionAt(offset + Math.floor(needle.length / 2))
        visibleEditor.setPosition(position)
        visibleEditor.revealPositionInCenter(position)
        visibleEditor.focus()
        return {
          matched: model.getValue().slice(offset, offset + needle.length) === needle,
          position
        }
      }, fixture.sentinelOrigin)
      expect(sentinelCursor?.matched).toBe(true)
      await page.waitForTimeout(50)
      const gesturePoint = await editor.evaluate((element) => {
        const monaco = (window as typeof window & {
          monaco?: {
            editor?: {
              getEditors?: () => Array<{
                getDomNode: () => HTMLElement | null
                getPosition: () => Record<string, number> | null
                getScrolledVisiblePosition: (
                  position: Record<string, number>
                ) => { top: number; left: number; height: number } | null
              }>
            }
          }
        }).monaco
        const visibleEditor = monaco?.editor?.getEditors?.().find((candidate) => {
          const node = candidate.getDomNode()
          return node === element || Boolean(node && (
            node.contains(element) || element.contains(node)
          ))
        })
        const position = visibleEditor?.getPosition()
        const visible = position
          ? visibleEditor?.getScrolledVisiblePosition(position)
          : null
        return visible
          ? {
              x: Math.max(1, Math.min(element.clientWidth - 1, visible.left + 1)),
              y: Math.max(1, Math.min(element.clientHeight - 1, visible.top + visible.height / 2))
            }
          : null
      })
      expect(gesturePoint).not.toBeNull()
      await editor.hover({ position: gesturePoint! })
      await editor.click({ button: "right", position: gesturePoint! })
      await page.keyboard.press("Escape")
      await editor.click({ button: "middle", position: gesturePoint! })
      await editor.click({ modifiers: [modifier], position: gesturePoint! })
      await page.keyboard.press("F12")
      await page.keyboard.press("Escape")
      await page.keyboard.press(`${modifier}+Enter`)
      await page.keyboard.press("Enter")
      await page.waitForTimeout(100)

      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()
      expect(opened.controlCsp).toBe(opened.studioCsp)
      expect(opened.controlCspMeta).toBe(opened.studioCspMeta)
      if (
        opened.controlCsp === null &&
        opened.studioCsp === null &&
        opened.controlCspMeta === null &&
        opened.studioCspMeta === null
      ) {
        testInfo.annotations.push({
          type: "csp-baseline",
          description: "Control and Studio documents both expose no CSP header or meta policy."
        })
      }
      const detailHeaders = await opened.detailApiResponse.allHeaders()
      expect(detailHeaders["access-control-allow-origin"]).toBe(fixture.webOrigin)
      expect(detailHeaders["access-control-allow-credentials"]).toBe("true")
      expect(detailHeaders["access-control-expose-headers"].toLowerCase()).toContain("retry-after")
      expect(detailHeaders["access-control-expose-headers"].toLowerCase()).toContain("last-modified")
      expect(detailHeaders["access-control-expose-headers"].toLowerCase()).toContain("content-length")
      expect(detailHeaders["access-control-expose-headers"].toLowerCase()).toContain("x-request-id")

      const generationIdempotencyKey = `task17-${testInfo.project.name}-${randomUUID()}`
      const browserCors = await page.evaluate(async ({ origin, apiKey, idempotencyKey }) => {
        const healthy = await fetch(`${origin}/api/v1/slides/capabilities`)
        const draining = await fetch(`${origin}/api/v1/slides/capabilities?task17_drain=1`)
        const generation = await fetch(`${origin}/api/v1/slides/generations`, {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
            "Idempotency-Key": idempotencyKey,
            "X-API-KEY": apiKey,
            "X-Slides-Accept-Content-Kinds": "structured_slides,standalone_html"
          },
          body: JSON.stringify({
            content_kind: "standalone_html",
            source_kind: "prompt",
            source: "Source-free protocol preflight characterization"
          })
        })
        return {
          healthyStatus: healthy.status,
          healthyOrigin: healthy.headers.get("access-control-allow-origin"),
          healthyRequestId: healthy.headers.get("x-request-id"),
          drainingStatus: draining.status,
          drainingOrigin: draining.headers.get("access-control-allow-origin"),
          retryAfter: draining.headers.get("retry-after"),
          generationStatus: generation.status,
          generationOrigin: generation.headers.get("access-control-allow-origin"),
          generationNosniff: generation.headers.get("x-content-type-options")
        }
      }, {
        origin: protocol.origin,
        apiKey: TEST_API_KEY,
        idempotencyKey: generationIdempotencyKey
      })
      expect(browserCors).toMatchObject({
        healthyStatus: 200,
        healthyOrigin: null,
        healthyRequestId: fixture.generationJobId,
        drainingStatus: 503,
        drainingOrigin: null,
        retryAfter: "3",
        generationStatus: 202,
        generationOrigin: null,
        generationNosniff: "nosniff"
      })
      const hostile = await fetch(`${protocol.origin}/api/v1/slides/capabilities`, {
        headers: { Origin: "https://hostile-origin.invalid" }
      })
      expect(hostile.headers.get("access-control-allow-origin")).toBeNull()
      const fixedPreflight = await fetch(`${protocol.origin}/api/v1/slides/capabilities`, {
        method: "OPTIONS",
        headers: {
          Origin: fixture.webOrigin,
          "Access-Control-Request-Method": "DELETE",
          "Access-Control-Request-Headers": "x-hostile-header, x-api-key"
        }
      })
      expect(fixedPreflight.headers.get("access-control-allow-methods"))
        .toBe("GET, HEAD, PUT, POST, OPTIONS")
      expect(fixedPreflight.headers.get("access-control-allow-methods")).not.toContain("DELETE")
      expect(fixedPreflight.headers.get("access-control-allow-headers"))
        .not.toContain("x-hostile-header")

      const detailRequest = protocol.state.requests.find((request) =>
        request.method === "GET" &&
        request.path === `/api/v1/slides/presentations/${fixture.presentationId}`
      )
      expect(detailRequest).toMatchObject({
        apiKey: TEST_API_KEY,
        acceptedContentKinds: "structured_slides,standalone_html"
      })
      expect(protocol.state.requests.some((request) =>
        request.method === "OPTIONS" &&
        request.origin === fixture.webOrigin &&
        request.requestedHeaders?.toLowerCase().includes("x-api-key") &&
        request.requestedMethod === "GET"
      )).toBe(true)
      expect(protocol.state.requests.some((request) =>
        request.method === "POST" &&
        request.path === "/api/v1/slides/generations" &&
        request.idempotencyKey === generationIdempotencyKey &&
        request.apiKey === TEST_API_KEY &&
        request.acceptedContentKinds === "structured_slides,standalone_html"
      )).toBe(true)
      expect(protocol.state.requests.some((request) =>
        request.method === "OPTIONS" &&
        request.origin === fixture.webOrigin &&
        request.requestedMethod === "POST" &&
        request.requestedHeaders?.toLowerCase().includes("idempotency-key")
      )).toBe(true)
      expect(protocol.state.responses.every((response) => response.contentType !== "text/html"))
        .toBe(true)
      expect(protocol.state.unexpectedPaths).toEqual([])

      expect(sourceBearingForbiddenEvents(
        opened.aggregate.events,
        recoveryKey(protocol.origin, 42, fixture.presentationId),
        undefined,
        fixture.webOrigin
      )).toEqual([])
      const sourceWorkerPosts = opened.aggregate.events.filter((event) =>
        event.kind === "worker-post" && event.sourceCorrelated
      )
      expect(sourceWorkerPosts.length).toBeGreaterThan(0)
      expect(sourceWorkerPosts.some((event) =>
        isApprovedOutlineWorkerEvent(event, fixture.webOrigin)
      )).toBe(true)
      expect(sourceWorkerPosts.every((event) => {
        if (isApprovedOutlineWorkerEvent(event, fixture.webOrigin)) return true
        const namedWorker = event.workerName === "editorWorkerService"
        const typeAndOriginAreConfined = event.workerType === "module"
          ? Boolean(event.url && new URL(event.url).origin === fixture.webOrigin)
          : event.workerType === "classic" &&
            Boolean(event.url?.startsWith(`blob:${fixture.webOrigin}`))
        return namedWorker && typeAndOriginAreConfined
      })).toBe(true)
      const arbitraryWorkerPost: SecurityEvent = {
        kind: "worker-post",
        at: 2,
        surface: "Worker",
        url: `${fixture.webOrigin}/_next/static/chunks/arbitrary.worker.js`,
        workerName: "ArbitrarySameOriginWorker",
        workerType: "module",
        sourceCorrelated: true
      }
      const exactPathArbitraryNamePost: SecurityEvent = {
        kind: "worker-post",
        at: 3,
        surface: "StandaloneHtmlOutlineWorker",
        url: `${fixture.webOrigin}/_next/static/chunks/StandaloneHtmlOutlineWorker.js`,
        workerName: "ArbitrarySameOriginWorker",
        workerType: "classic",
        sourceCorrelated: true
      }
      const nearMatchModulePost: SecurityEvent = {
        kind: "worker-post",
        at: 4,
        surface: "StandaloneHtmlOutlineWorker",
        url: `${fixture.webOrigin}/_next/static/chunks/standalone-html-outline.worker-near-match.js`,
        workerName: "StandaloneHtmlOutlineWorker",
        workerType: "module",
        sourceCorrelated: true
      }
      expect(sourceBearingForbiddenEvents([
        {
          kind: "worker-constructor",
          at: 1,
          surface: "Worker",
          url: arbitraryWorkerPost.url,
          workerName: arbitraryWorkerPost.workerName,
          workerType: arbitraryWorkerPost.workerType,
          sourceCorrelated: false
        },
        arbitraryWorkerPost,
        exactPathArbitraryNamePost,
        nearMatchModulePost
      ], undefined, undefined, fixture.webOrigin)).toEqual([
        arbitraryWorkerPost,
        exactPathArbitraryNamePost,
        nearMatchModulePost
      ])
      expect(opened.aggregate.events.filter((event) =>
        event.kind === "worker-constructor"
      ).every((event) => {
        if (!event.url || event.url.includes(fixture.marker)) return false
        if (event.url.startsWith("blob:")) return event.url.startsWith(`blob:${fixture.webOrigin}`)
        return new URL(event.url).origin === fixture.webOrigin
      })).toBe(true)
      expect(opened.observations.requests.some(({ url }) => url.startsWith(fixture.sentinelOrigin)))
        .toBe(false)
      expect(opened.observations.newPages).toEqual([])
      expect(opened.observations.workers.every((url) =>
        !url.includes(fixture.marker) &&
        (url.startsWith(`blob:${fixture.webOrigin}`) || new URL(url).origin === fixture.webOrigin)
      )).toBe(true)
      expect(opened.observations.serviceWorkers.every((url) => !url.includes(fixture.marker)))
        .toBe(true)
      expect(opened.observations.navigations.every((url) => !url.includes(fixture.marker)))
        .toBe(true)
      expect(opened.observations.errors.filter((message) => message.includes(fixture.marker)))
        .toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture,
        expectedRecoveryKey: recoveryKey(protocol.origin, 42, fixture.presentationId)
      })
    } finally {
      await protocol.close()
    }
  })

  test("keeps a real generation polling handoff inert in every security browser", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(120_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    const observations = installContextObservability(context, fixture)
    const aggregate = await installPageSecurityInstrumentation(
      context,
      fixture,
      protocol.origin
    )
    try {
      await page.goto("/presentation-studio/new", { waitUntil: "domcontentloaded" })
      await assertInstrumentationReady(page)
      const htmlMode = page.getByRole("radio", { name: /Standalone HTML \+ JavaScript/ })
      await expect(htmlMode).toBeEnabled()
      await htmlMode.check()
      await page.getByLabel("Subject and material").fill(
        `Cross-engine handoff ${testInfo.project.name} ${fixture.generationJobId}`
      )
      await page.getByLabel("Audience").fill("Task17 browser security reviewers")
      await page.getByRole("button", { name: "Generate standalone presentation" }).click()
      await expect(page.getByRole("heading", { name: "Submitted request" })).toBeVisible()
      await expect(page.getByText("Validating generated document")).toBeVisible()
      await expect(page).toHaveURL(new RegExp(
        `/presentation-studio/${fixture.presentationId}$`
      ), { timeout: 30_000 })
      await expect(page.locator(".monaco-editor").first()).toBeVisible()
      await expect(page.getByText(fixture.marker, { exact: true })).toBeVisible()
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()

      const generationRequest = protocol.state.requests.find((request) =>
        request.method === "POST" && request.path === "/api/v1/slides/generations"
      )
      expect(generationRequest?.idempotencyKey).toMatch(/^[A-Za-z0-9._~-]{16,200}$/)
      expect(generationRequest).toMatchObject({
        apiKey: TEST_API_KEY,
        acceptedContentKinds: null
      })
      expect(protocol.state.requests.filter((request) =>
        request.method === "GET" &&
        request.path === `/api/v1/slides/generations/${fixture.generationJobId}`
      ).length).toBeGreaterThanOrEqual(2)
      expect(protocol.state.requests.some((request) =>
        request.method === "OPTIONS" &&
        request.requestedMethod === "POST" &&
        request.requestedHeaders?.toLowerCase().includes("idempotency-key")
      )).toBe(true)
      expect(observations.requests.some(({ url }) => url.startsWith(fixture.sentinelOrigin)))
        .toBe(false)
      expect(sourceBearingForbiddenEvents(
        aggregate.events,
        recoveryKey(protocol.origin, 42, fixture.presentationId),
        undefined,
        fixture.webOrigin
      )).toEqual([])
      expect(protocol.state.unexpectedPaths).toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate,
        observations,
        protocol,
        fixture,
        expectedRecoveryKey: recoveryKey(protocol.origin, 42, fixture.presentationId)
      })
    } finally {
      await protocol.close()
    }
  })

  test("rejects a corrupt source-bearing detail before editor, worker, or DOM adoption", async ({
    context,
    page
  }, testInfo) => {
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture, {
      source: fixture.corruptSource,
      corruptDigest: true
    })
    try {
      const opened = await openSecurityWorkspace({ page, context, fixture, protocol })
      await expect(page.getByRole("heading", {
        level: 1,
        name: "Standalone HTML presentation"
      })).toBeVisible()
      await expect(page.locator(".monaco-editor")).toHaveCount(0)
      await expect(page.locator("body")).not.toContainText(fixture.marker)
      const events = await pageSecurityEvents(page)
      expect(events.some((event) => event.kind === "worker-post" && event.sourceCorrelated))
        .toBe(false)
      expect(sourceBearingForbiddenEvents(
        opened.aggregate.events,
        undefined,
        undefined,
        fixture.webOrigin
      )).toEqual([])
      expect(opened.observations.requests.some(({ url }) => url.startsWith(fixture.sentinelOrigin)))
        .toBe(false)
      expect(opened.observations.newPages).toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture
      })
    } finally {
      await protocol.close()
    }
  })

  test("fails a malformed outline closed while retaining the inert editor source", async ({
    context,
    page
  }, testInfo) => {
    const fixture = securityFixture(testInfo)
    const malformedMarker = `${fixture.marker}-MALFORMED-CANDIDATE`
    const nested = "<div>".repeat(129)
    const closed = "</div>".repeat(129)
    const complexSource = `<!doctype html><html><head><title>Complex outline</title></head><body>${nested}<section class="slide"><h1>${malformedMarker}</h1></section>${closed}<script>globalThis.__TASK17_EXECUTED__="${fixture.marker}";</script></body></html>`
    const protocol = await startSecurityProtocolServer(fixture)
    try {
      const opened = await openSecurityWorkspace({ page, context, fixture, protocol })
      await expect(page.locator(".monaco-editor").first()).toBeVisible()
      await replaceMonacoSource(page, complexSource)
      await expect(page.getByText("Outline unavailable", { exact: true })).toBeVisible()
      expect(await page.evaluate(() => {
        const monaco = (window as typeof window & {
          monaco?: { editor?: { getModels?: () => Array<{ getValue: () => string }> } }
        }).monaco
        return monaco?.editor?.getModels?.()[0]?.getValue()
      })).toBe(complexSource)
      const outlineRegion = page.getByRole("heading", {
        level: 2,
        name: "Safe outline: text only; code never runs in Studio"
      }).locator("xpath=ancestor::section[1]")
      await expect(outlineRegion).not.toContainText(malformedMarker)
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()
      expect(sourceBearingForbiddenEvents(
        opened.aggregate.events,
        recoveryKey(protocol.origin, 42, fixture.presentationId),
        undefined,
        fixture.webOrigin
      )).toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture,
        expectedRecoveryKey: recoveryKey(protocol.origin, 42, fixture.presentationId)
      })
    } finally {
      await protocol.close()
    }
  })

  test("times out only the first real outline worker request and recovers with a native replacement", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(45_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    let outlineRequests = 0
    const outlineWorkerPattern = /\/_next\/static\/chunks\/StandaloneHtmlOutlineWorker\.js(?:\?.*)?$/
    const outlineWorkerRoute = async (route: Route) => {
      outlineRequests += 1
      if (outlineRequests === 1) {
        await new Promise((resolve) => setTimeout(resolve, 11_000))
        await route.abort("failed").catch(() => undefined)
        return
      }
      await route.continue()
    }
    try {
      const opened = await openSecurityWorkspace({
        page,
        context,
        fixture,
        protocol,
        beforeMainNavigation: () => context.route(outlineWorkerPattern, outlineWorkerRoute)
      })
      const editor = page.locator(".monaco-editor").first()
      await expect(editor).toBeVisible()
      await expect(page.getByText("Outline unavailable", { exact: true }))
        .toBeVisible({ timeout: 15_000 })
      await expect(editor).toBeVisible()

      await replaceMonacoSource(page, fixture.editorSource)
      await expect(page.getByText("Trusted editor outline", { exact: true }))
        .toBeVisible({ timeout: 15_000 })
      expect(outlineRequests).toBeGreaterThanOrEqual(2)
      const events = await pageSecurityEvents(page)
      const outlineConstructors = events.filter((event) =>
        event.kind === "worker-constructor" &&
        event.workerName === "StandaloneHtmlOutlineWorker"
      )
      expect(outlineConstructors.length).toBeGreaterThanOrEqual(2)
      expect(outlineConstructors.every((event) =>
        event.sourceCorrelated === false &&
        isApprovedOutlineWorkerEvent(event, fixture.webOrigin)
      )).toBe(true)
      const outlineTerminations = events.filter((event) =>
        event.kind === "worker-terminate" &&
        event.url === outlineConstructors[0]?.url
      )
      expect(outlineTerminations.some((event) =>
        event.sourceCorrelated === false &&
        event.at <= (outlineConstructors[1]?.at ?? Number.NEGATIVE_INFINITY)
      )).toBe(true)
      expect(events.some((event) =>
        event.kind === "blob-create" && event.sourceCorrelated
      )).toBe(false)
      expect(sourceBearingForbiddenEvents(
        opened.aggregate.events,
        recoveryKey(protocol.origin, 42, fixture.presentationId),
        undefined,
        fixture.webOrigin
      )).toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture,
        expectedRecoveryKey: recoveryKey(protocol.origin, 42, fixture.presentationId)
      })
    } finally {
      await context.unroute(outlineWorkerPattern, outlineWorkerRoute)
      await protocol.close()
    }
  })

  test("confines draft download Blob use and recovery to the exact scoped session record", async ({
    context,
    page
  }, testInfo) => {
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    const expectedRecoveryKey = recoveryKey(protocol.origin, 42, fixture.presentationId)
    try {
      const opened = await openSecurityWorkspace({ page, context, fixture, protocol })
      await expect(page.locator(".monaco-editor").first()).toBeVisible()
      let events = await pageSecurityEvents(page)
      expect(events.some((event) => event.sourceCorrelated && [
        "storage-set",
        "logical-storage-set",
        "runtime-message"
      ].includes(event.kind))).toBe(false)

      await replaceMonacoSource(page, fixture.editorSource)
      const attachmentResponsePromise = page.waitForResponse((response) => {
        const url = new URL(response.url())
        return url.origin === protocol.origin &&
          url.pathname.endsWith("/draft-attachment") &&
          response.request().method() === "POST"
      })
      const downloadPromise = page.waitForEvent("download")
      await page.getByRole("button", { name: "Download current draft" }).click()
      const [attachmentResponse, download] = await Promise.all([
        attachmentResponsePromise,
        downloadPromise
      ])
      expect(download.suggestedFilename()).toBe("presentation.html")
      const downloadStream = await download.createReadStream()
      const downloadChunks: Buffer[] = []
      for await (const chunk of downloadStream) {
        downloadChunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
      }
      const downloadedBytes = Buffer.concat(downloadChunks)
      expect(downloadedBytes.byteLength).toBe(Buffer.byteLength(fixture.editorSource))
      expect(createHash("sha256").update(downloadedBytes).digest("hex"))
        .toBe(sha256(fixture.editorSource))
      await download.cancel()
      const attachmentHeaders = await attachmentResponse.allHeaders()
      expect(attachmentHeaders).toMatchObject({
        "content-type": "application/octet-stream",
        "content-disposition": 'attachment; filename="presentation.html"',
        "x-content-type-options": "nosniff",
        "x-download-options": "noopen",
        "cache-control": "private, no-store",
        "referrer-policy": "no-referrer",
        "cross-origin-resource-policy": "same-origin",
        "access-control-allow-origin": fixture.webOrigin
      })
      await expect.poll(async () => (await pageSecurityEvents(page)).filter((event) =>
        event.kind === "blob-create" && event.sourceCorrelated
      ).length).toBe(1)
      events = await pageSecurityEvents(page)
      const blobCreates = events.filter((event) =>
        event.kind === "blob-create" && event.sourceCorrelated
      )
      expect(blobCreates).toHaveLength(1)
      expect(blobCreates[0]).toMatchObject({
        contentType: "application/octet-stream",
        sourceCorrelated: true
      })
      expect(blobCreates[0].url).toMatch(/^blob:/)
      const downloadUrl = blobCreates[0].url as string
      await expect.poll(async () => (await pageSecurityEvents(page)).some((event) =>
        event.kind === "blob-revoke" && event.url === downloadUrl
      )).toBe(true)
      events = await pageSecurityEvents(page)
      const revoke = events.find((event) =>
        event.kind === "blob-revoke" && event.url === downloadUrl
      )
      expect(revoke).toBeDefined()
      expect((revoke?.at ?? Number.POSITIVE_INFINITY) - blobCreates[0].at)
        .toBeLessThanOrEqual(1_000)
      expect(events.filter((event) => event.kind === "anchor-click")).toEqual([
        expect.objectContaining({
          url: downloadUrl,
          dataOwnedAnchor: true,
          download: "presentation.html",
          target: "",
          rel: ""
        })
      ])
      expect(events.some((event) =>
        event.kind === "anchor-remove" &&
        event.url === downloadUrl &&
        event.dataOwnedAnchor
      )).toBe(true)
      expect(events.filter((event) =>
        event.kind === "anchor-click" && event.url === downloadUrl
      )).toHaveLength(1)
      expect(events.filter((event) =>
        event.kind === "dom-url" && event.url?.includes(downloadUrl)
      )).toEqual([
        expect.objectContaining({
          surface: "anchor.href",
          url: downloadUrl,
          dataOwnedAnchor: true,
          download: "presentation.html",
          target: "",
          rel: ""
        })
      ])
      expect(events.some((event) =>
        event.url === downloadUrl &&
        ["window-open", "worker-constructor", "worker-post"].includes(event.kind)
      )).toBe(false)
      expect(await page.locator("a[data-standalone-html-download]").count()).toBe(0)
      expect(context.pages().some((candidate) => candidate.url() === downloadUrl))
        .toBe(false)
      expect(protocol.state.requests.some((request) => request.path.includes(downloadUrl)))
        .toBe(false)
      expect([
        ...opened.observations.requests.map(({ url }) => url),
        ...opened.observations.navigations,
        ...opened.observations.newPages,
        ...opened.observations.workers,
        ...opened.observations.serviceWorkers
      ].some((value) => value === downloadUrl || value.includes(downloadUrl))).toBe(false)
      const attachmentRequest = protocol.state.requests.find((request) =>
        request.method === "POST" && request.path.endsWith("/draft-attachment")
      )
      expect(attachmentRequest).toMatchObject({
        apiKey: TEST_API_KEY,
        acceptedContentKinds: "structured_slides,standalone_html",
        contentType: "application/octet-stream",
        bodySourceCorrelated: true,
        bodyBytes: Buffer.byteLength(fixture.editorSource),
        bodySha256: sha256(fixture.editorSource)
      })
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()

      await replaceMonacoSource(page, fixture.safeDirtySource)
      await page.evaluate(() => window.dispatchEvent(
        new PageTransitionEvent("pagehide", { persisted: false })
      ))
      const rawRecovery = await page.evaluate(
        (key) => sessionStorage.getItem(key),
        expectedRecoveryKey
      )
      expect(rawRecovery).not.toBeNull()
      const record = JSON.parse(rawRecovery ?? "{}") as Record<string, unknown>
      expect(Object.keys(record).sort()).toEqual([
        "baseDigest",
        "baseEtag",
        "presentationId",
        "principalScope",
        "schemaVersion",
        "source",
        "updatedAt"
      ])
      expect(record).toMatchObject({
        schemaVersion: 1,
        principalScope: `${protocol.origin}|42`,
        presentationId: fixture.presentationId,
        baseEtag: '"v1"',
        baseDigest: sha256(fixture.generatedSource),
        source: fixture.safeDirtySource
      })
      expect(typeof record.updatedAt).toBe("number")
      expect(Date.now() - Number(record.updatedAt)).toBeGreaterThanOrEqual(0)
      expect(Date.now() - Number(record.updatedAt)).toBeLessThanOrEqual(86_400_000)

      await page.reload({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("heading", { name: "Recovered draft" })).toBeVisible()
      await page.getByRole("button", { name: "Restore recovered draft" }).click()
      await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Not saved")
      await page.getByRole("button", { name: "Save" }).click()
      await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Saved")
      expect(await page.evaluate(
        (key) => sessionStorage.getItem(key),
        expectedRecoveryKey
      )).toBeNull()
      const saveRequest = protocol.state.requests.find((request) =>
        request.method === "PUT" && request.path.endsWith("/html-source")
      )
      expect(saveRequest).toMatchObject({
        ifMatch: '"v1"',
        contentType: "application/octet-stream",
        acceptedContentKinds: "structured_slides,standalone_html",
        apiKey: TEST_API_KEY,
        bodySourceCorrelated: true
      })
      expect(protocol.state.requests.some((request) =>
        request.method === "OPTIONS" &&
        request.origin === fixture.webOrigin &&
        request.requestedMethod === "PUT" &&
        request.requestedHeaders?.toLowerCase().includes("if-match")
      )).toBe(true)
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()

      const expiredSource = fixture.safeDirtySource.replace(
        "Task17 safe dirty source",
        "Task17 expired private source"
      )
      await page.evaluate(({ key, value }) => {
        sessionStorage.setItem(key, JSON.stringify(value))
      }, {
        key: expectedRecoveryKey,
        value: {
          schemaVersion: 1,
          principalScope: `${protocol.origin}|42`,
          presentationId: fixture.presentationId,
          baseEtag: protocol.state.etag,
          baseDigest: protocol.state.digest,
          source: expiredSource,
          updatedAt: Date.now() - 86_400_001
        }
      })
      await page.reload({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("heading", { name: "Recovered draft" })).toHaveCount(0)
      expect(await page.evaluate(
        (key) => sessionStorage.getItem(key),
        expectedRecoveryKey
      )).toBeNull()

      const privateSource = fixture.safeDirtySource.replace(
        "Task17 safe dirty source",
        "Other-principal private outline"
      )
      await replaceMonacoSource(page, privateSource)
      expect(await page.evaluate(
        (key) => sessionStorage.getItem(key),
        expectedRecoveryKey
      )).not.toBeNull()
      protocol.state.principalId = 84
      await page.evaluate(() => window.dispatchEvent(new CustomEvent(
        "tldw:auth-principal-changed",
        { detail: { kind: "switch" } }
      )))
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        expectedRecoveryKey
      )).toBeNull()
      await expect(page.locator("body")).not.toContainText("Other-principal private outline")
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()

      events = await pageSecurityEvents(page)
      expect(sourceBearingForbiddenEvents(
        opened.aggregate.events,
        expectedRecoveryKey,
        downloadUrl,
        fixture.webOrigin
      )).toEqual([])
      expect(events.filter((event) =>
        event.sourceCorrelated &&
        event.kind === "storage-set" &&
        event.surface === "sessionStorage"
      ).every((event) => event.url === expectedRecoveryKey)).toBe(true)
      expect(events.some((event) =>
        event.sourceCorrelated &&
        (event.surface === "localStorage" || event.kind === "logical-storage-set" ||
          event.kind === "runtime-message")
      )).toBe(false)
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture,
        expectedRecoveryKey,
        expectedDownloadBlobUrl: downloadUrl
      })
    } finally {
      await protocol.close()
    }
  })

  test("scrubs a multi-user creation draft across logout, history restore, and visibility restore", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(120_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    const creationSource = `Task17 creation draft ${fixture.marker}`
    const ownerKeys = generationRecoveryKeys(protocol.origin, 42)
    await installPageShowSourceProbe(context, creationSource)
    const observations = installContextObservability(context, fixture)
    const aggregate = await installPageSecurityInstrumentation(context, fixture, protocol.origin, {
      authMode: "multi-user",
      accessToken: "task17-owner-42-token"
    })
    try {
      await page.goto("/presentation-studio/new", { waitUntil: "domcontentloaded" })
      await assertInstrumentationReady(page)
      const htmlMode = page.getByRole("radio", { name: /Standalone HTML \+ JavaScript/ })
      await expect(htmlMode).toBeEnabled()
      await htmlMode.check()
      await page.getByLabel("Subject and material").fill(creationSource)
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        ownerKeys.draft
      )).not.toBeNull()
      const draftRecord = JSON.parse(await page.evaluate(
        (key) => sessionStorage.getItem(key) ?? "{}",
        ownerKeys.draft
      )) as Record<string, unknown>
      expect(Object.keys(draftRecord).sort()).toEqual([
        "generationConfigRevision",
        "schemaVersion",
        "timestamp",
        "values"
      ])
      expect(draftRecord).toMatchObject({
        schemaVersion: 1,
        values: expect.objectContaining({ source: creationSource })
      })

      await page.goto(`/?task17-creation-away=${encodeURIComponent(testInfo.project.name)}`, {
        waitUntil: "domcontentloaded"
      })
      await logoutThenLoginAsPrincipal(page, protocol, 84)
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("heading", { level: 1, name: "New presentation" }))
        .toBeVisible()
      await assertCurrentSourceAbsent(page, creationSource)
      const restoredSource = page.getByLabel("Subject and material")
      if (await restoredSource.count() === 0) {
        const restoredHtmlMode = page.getByRole("radio", { name: /Standalone HTML \+ JavaScript/ })
        await expect(restoredHtmlMode).toBeEnabled()
        await restoredHtmlMode.check()
      }
      await expect(restoredSource).toHaveValue("")
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        ownerKeys.draft
      )).toBeNull()
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        ownerKeys.resume
      )).toBeNull()
      await page.goForward({ waitUntil: "domcontentloaded" })
      await expect(page).toHaveURL(new RegExp(`task17-creation-away=${encodeURIComponent(testInfo.project.name)}`))
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("heading", { level: 1, name: "New presentation" }))
        .toBeVisible()
      await assertCurrentSourceAbsent(page, creationSource)
      await assertPersistedProbeSourceFree(page, testInfo, "creation")
      expect(sourceBearingForbiddenEvents(
        aggregate.events,
        [ownerKeys.draft, ownerKeys.resume],
        undefined,
        fixture.webOrigin
      )).toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate,
        observations,
        protocol,
        fixture,
        expectedRecoveryKey: [ownerKeys.draft, ownerKeys.resume]
      })
    } finally {
      await protocol.close()
    }
  })

  test("restores a same-principal creation draft only after history authority settles", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(120_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    const stalePrefix = `Task17 same-principal native form value ${fixture.marker}`
    const finalSource = `${stalePrefix} exact-final-keystroke`
    const ownerKeys = generationRecoveryKeys(protocol.origin, 42)
    await installPageShowSourceProbe(context, fixture.marker)
    const observations = installContextObservability(context, fixture)
    const aggregate = await installPageSecurityInstrumentation(context, fixture, protocol.origin, {
      authMode: "multi-user",
      accessToken: "task17-owner-42-token"
    })
    await context.route(`${protocol.origin}/api/v1/auth/me`, (route) => route.continue())
    const expectGuardedForm = async () => {
      await expect(page.getByRole("heading", { level: 1, name: "New presentation" }))
        .toBeVisible()
      await expect.poll(() => protocol.state.authPendingCount).toBeGreaterThan(0)
      await expect(page.getByLabel("Subject and material")).toHaveCount(0)
      await assertCurrentSourceAbsent(page, fixture.marker)
    }
    try {
      await page.goto("/presentation-studio/new", { waitUntil: "domcontentloaded" })
      await assertInstrumentationReady(page)
      const htmlMode = page.getByRole("radio", { name: /Standalone HTML \+ JavaScript/ })
      await expect(htmlMode).toBeEnabled()
      await htmlMode.check()
      const sourceField = page.getByLabel("Subject and material")
      await expect(sourceField).toBeEnabled()
      await sourceField.fill(stalePrefix)
      await sourceField.pressSequentially(" exact-final-keystroke")
      await expect(sourceField).toHaveValue(finalSource)

      protocol.deferAuthAfter(0)
      await page.goto(`/?task17-same-principal-creation-away=${encodeURIComponent(
        testInfo.project.name
      )}`, { waitUntil: "domcontentloaded" })
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expectGuardedForm()

      protocol.releaseAuth()
      await expect(page.getByLabel("Subject and material")).toHaveValue(finalSource)
      const firstRestoredRecord = JSON.parse(await page.evaluate(
        (key) => sessionStorage.getItem(key) ?? "{}",
        ownerKeys.draft
      )) as { values?: { source?: string } }
      expect(firstRestoredRecord.values?.source).toBe(finalSource)

      await page.goForward({ waitUntil: "domcontentloaded" })
      await expect(page).toHaveURL(new RegExp(
        `task17-same-principal-creation-away=${encodeURIComponent(testInfo.project.name)}`
      ))
      protocol.deferAuthAfter(0)
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expectGuardedForm()

      protocol.releaseAuth()
      await expect(page.getByLabel("Subject and material")).toHaveValue(finalSource)
      const secondRestoredRecord = JSON.parse(await page.evaluate(
        (key) => sessionStorage.getItem(key) ?? "{}",
        ownerKeys.draft
      )) as { values?: { source?: string } }
      expect(secondRestoredRecord.values?.source).toBe(finalSource)
      expect(sourceBearingForbiddenEvents(
        aggregate.events,
        [ownerKeys.draft, ownerKeys.resume],
        undefined,
        fixture.webOrigin
      )).toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate,
        observations,
        protocol,
        fixture,
        expectedRecoveryKey: [ownerKeys.draft, ownerKeys.resume]
      })
    } finally {
      protocol.releaseAuth()
      await protocol.close()
    }
  })

  test("retires a multi-user submitted request before account-switch history and visibility restore", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(120_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture, {
      generationCompletionEnabled: false
    })
    const submittedSource = `Task17 submitted immutable request ${fixture.marker}`
    const ownerKeys = generationRecoveryKeys(protocol.origin, 42)
    await installPageShowSourceProbe(context, submittedSource)
    const observations = installContextObservability(context, fixture)
    const aggregate = await installPageSecurityInstrumentation(context, fixture, protocol.origin, {
      authMode: "multi-user",
      accessToken: "task17-owner-42-token"
    })
    try {
      await page.goto("/presentation-studio/new", { waitUntil: "domcontentloaded" })
      await assertInstrumentationReady(page)
      const htmlMode = page.getByRole("radio", { name: /Standalone HTML \+ JavaScript/ })
      await expect(htmlMode).toBeEnabled()
      await htmlMode.check()
      await page.getByLabel("Subject and material").fill(submittedSource)
      await page.getByLabel("Audience").fill("Task17 multi-user isolation reviewers")
      await page.getByRole("button", { name: "Generate standalone presentation" }).click()
      await expect(page.getByRole("heading", { name: "Submitted request" })).toBeVisible()
      await expect(page.getByLabel("Submitted request").getByText(submittedSource, { exact: true }))
        .toBeVisible()
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        ownerKeys.resume
      )).not.toBeNull()

      await page.goto(`/?task17-submitted-away=${encodeURIComponent(testInfo.project.name)}`, {
        waitUntil: "domcontentloaded"
      })
      const pollsAfterLeave = protocol.state.generationPollCount
      await logoutThenLoginAsPrincipal(page, protocol, 84)
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("heading", { level: 1, name: "New presentation" }))
        .toBeVisible()
      await expect(page.getByRole("heading", { name: "Submitted request" })).toHaveCount(0)
      await assertCurrentSourceAbsent(page, submittedSource)
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        ownerKeys.draft
      )).toBeNull()
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        ownerKeys.resume
      )).toBeNull()
      await page.waitForTimeout(1_100)
      expect(protocol.state.generationPollCount).toBe(pollsAfterLeave)
      await assertCurrentSourceAbsent(page, submittedSource)
      await page.goForward({ waitUntil: "domcontentloaded" })
      await expect(page).toHaveURL(new RegExp(`task17-submitted-away=${encodeURIComponent(testInfo.project.name)}`))
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("heading", { name: "Submitted request" })).toHaveCount(0)
      await assertCurrentSourceAbsent(page, submittedSource)
      expect(protocol.state.generationPollCount).toBe(pollsAfterLeave)
      await assertPersistedProbeSourceFree(page, testInfo, "submitted-request")
      expect(sourceBearingForbiddenEvents(
        aggregate.events,
        [ownerKeys.draft, ownerKeys.resume],
        undefined,
        fixture.webOrigin
      )).toEqual([])
      const submitRequest = protocol.state.requests.find((request) =>
        request.method === "POST" && request.path === "/api/v1/slides/generations"
      )
      expect(submitRequest).toMatchObject({
        authorization: "Bearer task17-owner-42-token",
        bodySourceCorrelated: true,
        idempotencyKey: expect.stringMatching(/^[A-Za-z0-9._~-]{16,200}$/)
      })
      await assertTerminalSecurityAudit({
        page,
        aggregate,
        observations,
        protocol,
        fixture,
        expectedRecoveryKey: [ownerKeys.draft, ownerKeys.resume]
      })
    } finally {
      await protocol.close()
    }
  })

  test("guards a multi-user workspace before account-switch history and visibility restore", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(120_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    const oldRecoveryKey = recoveryKey(protocol.origin, 42, fixture.presentationId)
    await installPageShowSourceProbe(context, fixture.marker)
    try {
      const initialMetadataResponse = page.waitForResponse((response) =>
        response.url() === `${protocol.origin}/api/v1/slides/presentations/${fixture.presentationId}/metadata`
      )
      const opened = await openSecurityWorkspace({
        page,
        context,
        fixture,
        protocol,
        authMode: "multi-user"
      })
      expect((await initialMetadataResponse).headers()["cache-control"]).toBe("private, no-store")
      await expect(page.locator(".monaco-editor").first()).toBeVisible()
      await expect(page.getByText(fixture.marker, { exact: true })).toBeVisible()
      const ownerDetail = protocol.state.requests.find((request) =>
        request.method === "GET" &&
        request.path === `/api/v1/slides/presentations/${fixture.presentationId}`
      )
      expect(ownerDetail?.authorization).toBe("Bearer task17-owner-42-token")
      await page.evaluate(({ key, origin, presentationId, source, digest, etag }) => {
        sessionStorage.setItem(key, JSON.stringify({
          schemaVersion: 1,
          principalScope: `${origin}|42`,
          presentationId,
          baseEtag: etag,
          baseDigest: digest,
          source,
          updatedAt: Date.now()
        }))
      }, {
        key: oldRecoveryKey,
        origin: protocol.origin,
        presentationId: fixture.presentationId,
        source: fixture.safeDirtySource,
        digest: protocol.state.digest,
        etag: protocol.state.etag
      })

      const ownerPresentationReads = protocol.state.requests.filter((request) =>
        request.method === "GET" &&
        request.path.startsWith(`/api/v1/slides/presentations/${fixture.presentationId}`)
      ).length
      await page.goto(`/?task17-workspace-away=${encodeURIComponent(testInfo.project.name)}`, {
        waitUntil: "domcontentloaded"
      })
      await logoutThenLoginAsPrincipal(page, protocol, 84)
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect.poll(() => protocol.state.requests.filter((request) =>
        request.method === "GET" &&
        request.path.startsWith(`/api/v1/slides/presentations/${fixture.presentationId}`)
      ).length).toBeGreaterThan(ownerPresentationReads)
      const switchedPresentationReads = protocol.state.requests.filter((request) =>
        request.method === "GET" &&
        request.path.startsWith(`/api/v1/slides/presentations/${fixture.presentationId}`)
      ).slice(ownerPresentationReads)
      expect(switchedPresentationReads[0]).toMatchObject({
        path: `/api/v1/slides/presentations/${fixture.presentationId}/metadata`,
        authorization: "Bearer task17-owner-84-token"
      })
      await expect(page.getByRole("heading", {
        level: 1,
        name: "Presentation metadata is unavailable"
      }))
        .toBeVisible()
      await assertCurrentSourceAbsent(page, fixture.marker)
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        oldRecoveryKey
      )).toBeNull()
      await page.goForward({ waitUntil: "domcontentloaded" })
      await expect(page).toHaveURL(new RegExp(`task17-workspace-away=${encodeURIComponent(testInfo.project.name)}`))
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page.getByRole("heading", {
        level: 1,
        name: "Presentation metadata is unavailable"
      }))
        .toBeVisible()
      await assertCurrentSourceAbsent(page, fixture.marker)
      const allSwitchedPresentationReads = protocol.state.requests.filter((request) =>
        request.method === "GET" &&
        request.authorization === "Bearer task17-owner-84-token" &&
        request.path.startsWith(`/api/v1/slides/presentations/${fixture.presentationId}`)
      )
      expect(allSwitchedPresentationReads.length).toBeGreaterThan(0)
      expect(allSwitchedPresentationReads.every((request) =>
        request.path === `/api/v1/slides/presentations/${fixture.presentationId}/metadata`
      )).toBe(true)
      await assertPersistedProbeSourceFree(page, testInfo, "workspace")
      const switchedMetadata = switchedPresentationReads.find((request) =>
        request.path === `/api/v1/slides/presentations/${fixture.presentationId}/metadata`
      )
      expect(switchedMetadata?.authorization).toBe("Bearer task17-owner-84-token")
      expect(sourceBearingForbiddenEvents(
        opened.aggregate.events,
        oldRecoveryKey,
        undefined,
        fixture.webOrigin
      )).toEqual([])
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture,
        expectedRecoveryKey: oldRecoveryKey
      })
    } finally {
      await protocol.close()
    }
  })

  test("preserves the last pending keystroke across real history and scrubs a clean bfcache account switch", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(120_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    const pendingLabel = "Task17 cross-engine pending last keystroke"
    const pendingSource = fixture.safeDirtySource
      .replace("Task17 safe dirty source", pendingLabel)
    const oldRecoveryKey = recoveryKey(protocol.origin, 42, fixture.presentationId)
    await installDeferredDigestAndPageShowProbe(context, pendingLabel, fixture.marker)
    try {
      const opened = await openSecurityWorkspace({ page, context, fixture, protocol })
      await expect(page.locator(".monaco-editor").first()).toBeVisible()

      await replaceMonacoSource(page, pendingSource)
      expect(await page.evaluate(() => Boolean((
        window as typeof window & { __task17DigestDeferred?: boolean }
      ).__task17DigestDeferred))).toBe(true)
      let pendingLeavePrompts = 0
      page.once("dialog", async (dialog) => {
        pendingLeavePrompts += 1
        expect(dialog.type()).toBe("confirm")
        await dialog.accept()
      })
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page).toHaveURL(/\/presentation-studio$/)
      expect(pendingLeavePrompts).toBe(1)
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        oldRecoveryKey
      )).not.toBeNull()
      const pendingRecord = JSON.parse(await page.evaluate(
        (key) => sessionStorage.getItem(key) ?? "{}",
        oldRecoveryKey
      )) as { source?: string }
      expect(pendingRecord.source).toBe(pendingSource)

      await page.goForward({ waitUntil: "domcontentloaded" })
      await expect.poll(async () => page.evaluate(() => typeof (
        window as typeof window & { __task17ReleaseDigest?: () => void }
      ).__task17ReleaseDigest === "function")).toBe(true)
      await page.evaluate(() => (
        window as typeof window & { __task17ReleaseDigest?: () => void }
      ).__task17ReleaseDigest?.())
      await expect(page.getByRole("heading", { name: "Recovered draft" })).toBeVisible()
      await page.getByRole("button", { name: "Restore recovered draft" }).click()
      await expect(page.locator(".monaco-editor .view-lines").first()).toContainText(pendingLabel)
      await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Not saved")
      await page.getByRole("button", { name: "Save", exact: true }).click()
      await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Saved")
      expect(await page.evaluate(
        (key) => sessionStorage.getItem(key),
        oldRecoveryKey
      )).toBeNull()
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()

      await page.evaluate(({ key, origin, presentationId, source, etag, digest }) => {
        sessionStorage.setItem(key, JSON.stringify({
          schemaVersion: 1,
          principalScope: `${origin}|42`,
          presentationId,
          baseEtag: etag,
          baseDigest: digest,
          source,
          updatedAt: Date.now()
        }))
      }, {
        key: oldRecoveryKey,
        origin: protocol.origin,
        presentationId: fixture.presentationId,
        source: pendingSource,
        etag: protocol.state.etag,
        digest: protocol.state.digest
      })

      let cleanLeavePrompts = 0
      page.on("dialog", async (dialog) => {
        cleanLeavePrompts += 1
        await dialog.dismiss()
      })
      const ownerReadsBeforeSwitch = protocol.state.requests.filter((request) =>
        request.method === "GET" &&
        request.path === `/api/v1/slides/presentations/${fixture.presentationId}/metadata`
      ).length
      await page.goto(`/?task17-away=${encodeURIComponent(testInfo.project.name)}`, {
        waitUntil: "domcontentloaded"
      })
      expect(cleanLeavePrompts).toBe(0)
      protocol.state.principalId = 84
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect.poll(() => protocol.state.requests.filter((request) =>
        request.method === "GET" &&
        request.path === `/api/v1/slides/presentations/${fixture.presentationId}/metadata`
      ).length).toBeGreaterThan(ownerReadsBeforeSwitch)
      await expect(page.getByRole("heading", {
        level: 1,
        name: "Presentation metadata is unavailable"
      }))
        .toBeVisible()
      await expect(page.locator("body")).not.toContainText(pendingLabel)
      await expect(page.locator("body")).not.toContainText(fixture.marker)
      await expect.poll(async () => page.evaluate(
        (key) => sessionStorage.getItem(key),
        oldRecoveryKey
      )).toBeNull()
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()

      const pageShows = await page.evaluate(() => (
        window as typeof window & {
          __task17PageShows?: Array<{ persisted: boolean; sourceVisible: boolean }>
        }
      ).__task17PageShows ?? [])
      if (pageShows.some((entry) => entry.persisted)) {
        expect(pageShows.filter((entry) => entry.persisted).every((entry) =>
          entry.sourceVisible === false
        )).toBe(true)
      } else {
        testInfo.annotations.push({
          type: "engine-limitation",
          description: `${testInfo.project.name} did not restore the clean account-switch navigation from bfcache.`
        })
      }
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture,
        expectedRecoveryKey: oldRecoveryKey
      })
    } finally {
      await protocol.close()
    }
  })

  test("keeps direct detail, version, restore, and attachment routes inert across navigation", async ({
    context,
    page
  }, testInfo) => {
    test.setTimeout(60_000)
    const fixture = securityFixture(testInfo)
    const protocol = await startSecurityProtocolServer(fixture)
    try {
      const opened = await openSecurityWorkspace({ page, context, fixture, protocol })
      await expect(page.locator(".monaco-editor").first()).toBeVisible()

      await page.goto("/presentation-studio", { waitUntil: "domcontentloaded" })
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page.locator(".monaco-editor").first()).toBeVisible()
      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()
      await page.goForward({ waitUntil: "domcontentloaded" })
      await expect(page).toHaveURL(/\/presentation-studio$/)
      await page.goBack({ waitUntil: "domcontentloaded" })
      await expect(page.locator(".monaco-editor").first()).toBeVisible()

      const routeEvidence = await page.evaluate(async ({
        origin,
        presentationId,
        apiKey,
        draft
      }) => {
        const headers = {
          "X-API-KEY": apiKey,
          "X-Slides-Accept-Content-Kinds": "structured_slides,standalone_html"
        }
        const fingerprint = async (bytes: ArrayBuffer) => {
          const digest = await crypto.subtle.digest("SHA-256", bytes)
          return Array.from(new Uint8Array(digest))
            .map((value) => value.toString(16).padStart(2, "0"))
            .join("")
        }
        const jsonEvidence = async (path: string, init?: RequestInit) => {
          const response = await fetch(`${origin}${path}`, {
            ...init,
            headers: { ...headers, ...(init?.headers ?? {}) }
          })
          const text = await response.text()
          const parsed = JSON.parse(text) as Record<string, unknown>
          const source = typeof parsed.html_document === "string" ? parsed.html_document : null
          return {
            status: response.status,
            contentType: response.headers.get("content-type"),
            nosniff: response.headers.get("x-content-type-options"),
            noopen: response.headers.get("x-download-options"),
            disposition: response.headers.get("content-disposition"),
            cacheControl: response.headers.get("cache-control"),
            referrerPolicy: response.headers.get("referrer-policy"),
            corp: response.headers.get("cross-origin-resource-policy"),
            etag: response.headers.get("etag"),
            hasSource: source !== null,
            sourceBytes: source === null ? 0 : new TextEncoder().encode(source).byteLength,
            sourceDigest: source === null
              ? null
              : await fingerprint(new TextEncoder().encode(source).buffer),
            containsMarker: text.includes("TASK17-SOURCE-")
          }
        }
        const base = `/api/v1/slides/presentations/${encodeURIComponent(presentationId)}`
        const detail = await jsonEvidence(base)
        const versions = await jsonEvidence(`${base}/versions`)
        const version = await jsonEvidence(`${base}/versions/1`)
        const jsonExport = await jsonEvidence(`${base}/export?format=json`)
        const htmlExportResponse = await fetch(`${origin}${base}/export?format=html`, { headers })
        const htmlExportBytes = await htmlExportResponse.arrayBuffer()
        const draftResponse = await fetch(`${origin}${base}/draft-attachment`, {
          method: "POST",
          headers: {
            ...headers,
            Accept: "application/octet-stream",
            "Content-Type": "application/octet-stream"
          },
          body: draft
        })
        const draftBytes = await draftResponse.arrayBuffer()
        const restore = await jsonEvidence(`${base}/versions/1/restore`, {
          method: "POST",
          headers: { "If-Match": '"v1"' }
        })
        return {
          detail,
          versions,
          version,
          jsonExport,
          restore,
          htmlExport: {
            status: htmlExportResponse.status,
            contentType: htmlExportResponse.headers.get("content-type"),
            disposition: htmlExportResponse.headers.get("content-disposition"),
            nosniff: htmlExportResponse.headers.get("x-content-type-options"),
            bytes: htmlExportBytes.byteLength,
            digest: await fingerprint(htmlExportBytes)
          },
          draftAttachment: {
            status: draftResponse.status,
            contentType: draftResponse.headers.get("content-type"),
            disposition: draftResponse.headers.get("content-disposition"),
            nosniff: draftResponse.headers.get("x-content-type-options"),
            bytes: draftBytes.byteLength,
            digest: await fingerprint(draftBytes)
          }
        }
      }, {
        origin: protocol.origin,
        presentationId: fixture.presentationId,
        apiKey: TEST_API_KEY,
        draft: fixture.editorSource
      })

      expect(routeEvidence.detail).toMatchObject({
        status: 200,
        contentType: expect.stringContaining("application/json"),
        nosniff: "nosniff",
        hasSource: true,
        sourceBytes: Buffer.byteLength(fixture.generatedSource),
        sourceDigest: sha256(fixture.generatedSource),
        containsMarker: true
      })
      expect(routeEvidence.versions).toMatchObject({
        status: 200,
        contentType: expect.stringContaining("application/json"),
        hasSource: false,
        containsMarker: false
      })
      expect(routeEvidence.version).toMatchObject({
        status: 200,
        contentType: expect.stringContaining("application/json"),
        nosniff: "nosniff",
        hasSource: true,
        sourceDigest: sha256(fixture.generatedSource)
      })
      expect(routeEvidence.jsonExport).toMatchObject({
        status: 200,
        contentType: expect.stringContaining("application/json"),
        nosniff: "nosniff",
        noopen: "noopen",
        disposition: 'attachment; filename="presentation.json"',
        cacheControl: "private, no-store",
        referrerPolicy: "no-referrer",
        corp: "same-origin",
        hasSource: true,
        sourceDigest: sha256(fixture.generatedSource)
      })
      expect(routeEvidence.restore).toMatchObject({
        status: 200,
        contentType: expect.stringContaining("application/json"),
        nosniff: "nosniff",
        etag: '"v2"',
        hasSource: true,
        sourceDigest: sha256(fixture.generatedSource)
      })
      expect(routeEvidence.htmlExport).toEqual({
        status: 200,
        contentType: "application/octet-stream",
        disposition: 'attachment; filename="presentation.html"',
        nosniff: "nosniff",
        bytes: Buffer.byteLength(fixture.generatedSource),
        digest: sha256(fixture.generatedSource)
      })
      expect(routeEvidence.draftAttachment).toEqual({
        status: 200,
        contentType: "application/octet-stream",
        disposition: 'attachment; filename="presentation.html"',
        nosniff: "nosniff",
        bytes: Buffer.byteLength(fixture.editorSource),
        digest: sha256(fixture.editorSource)
      })

      await context.setExtraHTTPHeaders({
        "X-API-KEY": TEST_API_KEY,
        "X-Slides-Accept-Content-Kinds": "structured_slides,standalone_html"
      })
      const directJsonPage = await context.newPage()
      const directJsonResponse = await directJsonPage.goto(
        `${protocol.origin}/api/v1/slides/presentations/${fixture.presentationId}`,
        { waitUntil: "domcontentloaded" }
      )
      expect((await directJsonResponse?.allHeaders())?.["content-type"])
        .toContain("application/json")
      await expect(directJsonPage.locator("body")).toContainText(fixture.marker)
      expect(await directJsonPage.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()
      await flushPageSecurityAggregate(directJsonPage)
      expect(await pageSecurityOverflow(directJsonPage)).toBe(0)
      await directJsonPage.close()

      const directVersionPage = await context.newPage()
      const directVersionResponse = await directVersionPage.goto(
        `${protocol.origin}/api/v1/slides/presentations/${fixture.presentationId}/versions/1`,
        { waitUntil: "domcontentloaded" }
      )
      expect((await directVersionResponse?.allHeaders())?.["content-type"])
        .toContain("application/json")
      expect((await directVersionResponse?.allHeaders())?.["x-content-type-options"])
        .toBe("nosniff")
      await expect(directVersionPage.locator("body")).toContainText(fixture.marker)
      expect(await directVersionPage.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()
      await flushPageSecurityAggregate(directVersionPage)
      expect(await pageSecurityOverflow(directVersionPage)).toBe(0)
      await directVersionPage.close()

      const directJsonExportPage = await context.newPage()
      const jsonExportUrl = `${protocol.origin}/api/v1/slides/presentations/${fixture.presentationId}/export?format=json`
      const directJsonExportResponsePromise = directJsonExportPage.waitForResponse((response) =>
        response.url() === jsonExportUrl
      )
      const directJsonExportPromise = directJsonExportPage.waitForEvent("download")
      const jsonExportNavigationPromise = directJsonExportPage.goto(jsonExportUrl).catch(() => null)
      const [directJsonExportResponse, directJsonExport] = await Promise.all([
        directJsonExportResponsePromise,
        directJsonExportPromise
      ])
      await jsonExportNavigationPromise
      expect(directJsonExport.suggestedFilename()).toBe("presentation.json")
      expect(await directJsonExportResponse.allHeaders()).toMatchObject({
        "content-type": "application/json",
        "content-disposition": 'attachment; filename="presentation.json"',
        "x-content-type-options": "nosniff",
        "x-download-options": "noopen",
        "cache-control": "private, no-store",
        "referrer-policy": "no-referrer",
        "cross-origin-resource-policy": "same-origin"
      })
      const jsonExportStream = await directJsonExport.createReadStream()
      const jsonExportChunks: Buffer[] = []
      for await (const chunk of jsonExportStream) {
        jsonExportChunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
      }
      const jsonExportBytes = Buffer.concat(jsonExportChunks)
      const jsonExportDocument = JSON.parse(jsonExportBytes.toString("utf8")) as {
        html_document?: string
        html_sha256?: string
      }
      expect(jsonExportDocument).toMatchObject({
        html_document: fixture.generatedSource,
        html_sha256: sha256(fixture.generatedSource)
      })
      expect(await directJsonExportPage.locator("body").textContent())
        .not.toContain(fixture.marker)
      expect(await directJsonExportPage.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()
      await flushPageSecurityAggregate(directJsonExportPage)
      expect(await pageSecurityOverflow(directJsonExportPage)).toBe(0)
      await directJsonExport.cancel()
      await directJsonExportPage.close()

      const directDownloadPage = await context.newPage()
      const htmlExportUrl = `${protocol.origin}/api/v1/slides/presentations/${fixture.presentationId}/export?format=html`
      const directDownloadResponsePromise = directDownloadPage.waitForResponse((response) =>
        response.url() === htmlExportUrl
      )
      const directDownloadPromise = directDownloadPage.waitForEvent("download")
      const navigationPromise = directDownloadPage.goto(htmlExportUrl).catch(() => null)
      const [directDownloadResponse, directDownload] = await Promise.all([
        directDownloadResponsePromise,
        directDownloadPromise
      ])
      await navigationPromise
      expect(directDownload.suggestedFilename()).toBe("presentation.html")
      expect(await directDownloadResponse.allHeaders()).toMatchObject({
        "content-type": "application/octet-stream",
        "content-disposition": 'attachment; filename="presentation.html"',
        "x-content-type-options": "nosniff",
        "x-download-options": "noopen",
        "cache-control": "private, no-store",
        "referrer-policy": "no-referrer",
        "cross-origin-resource-policy": "same-origin"
      })
      const stream = await directDownload.createReadStream()
      const chunks: Buffer[] = []
      for await (const chunk of stream) {
        chunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
      }
      const directBytes = Buffer.concat(chunks)
      expect(directBytes.byteLength).toBe(Buffer.byteLength(fixture.generatedSource))
      expect(createHash("sha256").update(directBytes).digest("hex"))
        .toBe(sha256(fixture.generatedSource))
      expect(await directDownloadPage.locator("body").textContent()).not.toContain(fixture.marker)
      await flushPageSecurityAggregate(directDownloadPage)
      expect(await pageSecurityOverflow(directDownloadPage)).toBe(0)
      await directDownload.cancel()
      await directDownloadPage.close()
      await context.setExtraHTTPHeaders({})

      expect(await page.evaluate(() => (
        window as typeof window & { __TASK17_EXECUTED__?: string }
      ).__TASK17_EXECUTED__)).toBeUndefined()
      expect(opened.observations.requests.some(({ url }) => url.startsWith(fixture.sentinelOrigin)))
        .toBe(false)
      expect(protocol.state.responses.every((response) => response.contentType !== "text/html"))
        .toBe(true)
      expect(protocol.state.unexpectedPaths).toEqual([])
      expect(protocol.state.requests.some((request) =>
        request.method === "POST" &&
        request.path.endsWith("/versions/1/restore") &&
        request.ifMatch === '"v1"' &&
        request.acceptedContentKinds === "structured_slides,standalone_html"
      )).toBe(true)
      await assertTerminalSecurityAudit({
        page,
        aggregate: opened.aggregate,
        observations: opened.observations,
        protocol,
        fixture
      })
    } finally {
      await context.setExtraHTTPHeaders({})
      await protocol.close()
    }
  })
})
