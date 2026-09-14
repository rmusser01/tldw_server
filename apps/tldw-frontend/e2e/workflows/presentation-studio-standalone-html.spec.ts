import { createHash, randomUUID } from "node:crypto"

import { expect, test, type Locator, type Page, type Route } from "@playwright/test"

const API_ORIGIN = "http://127.0.0.1:17991"
const REVISION = `sha256:${"7".repeat(64)}`
const WEB_ORIGIN = "http://localhost:8080"
const CORS_HEADERS = {
  "Access-Control-Allow-Origin": WEB_ORIGIN,
  "Access-Control-Allow-Credentials": "true",
  "Access-Control-Expose-Headers": [
    "ETag",
    "Content-Disposition",
    "X-Content-Type-Options",
    "X-Download-Options",
    "Cache-Control",
    "Referrer-Policy",
    "Cross-Origin-Resource-Policy"
  ].join(", ")
}

const sourceDocument = (label: string) => `<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>${label}</title><style>.slide{color:#111}</style></head>
<body><section class="slide"><h1>${label}</h1><p>Trusted outline text</p><div class="notes"><p>Speaker context</p></div></section><script>document.addEventListener('keydown', () => {});</script></body></html>`

const BASE_SOURCE = sourceDocument("Task17 base slide")
const LOST_RESPONSE_SOURCE = sourceDocument("Task17 saved after lost response")
const OVERWRITE_SOURCE = sourceDocument("Task17 edited C")
const SERVER_CONFLICT_SOURCE = sourceDocument("Task17 server conflict D")
const DISCARD_CANDIDATE = sourceDocument("Task17 local discard E")
const DISCARD_SERVER_SOURCE = sourceDocument("Task17 fresh server F")
const RECOVERY_SOURCE = sourceDocument("Task17 pagehide recovery")
const PENDING_DIGEST_MARKER = "Task17 pending Back last keystroke"
const PENDING_BACK_SOURCE = sourceDocument("Task17 pending Back candidate").replace(
  "</section>",
  `<p>${PENDING_DIGEST_MARKER}</p></section>`
)

const sha256 = (value: string) => createHash("sha256").update(value).digest("hex")

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
      generation_config_revision: REVISION,
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

type MockState = {
  presentationId: string
  generationId: string
  generationJobId: string
  source: string
  title: string
  etag: string
  version: number
  generationCompleted: boolean
  saveBehavior: "normal" | "lost" | "conflict"
  principalId: number
  presentationAvailable: boolean
  requests: Array<{ method: string; path: string; body: string | null; headers: Record<string, string> }>
}

const createState = (projectName: string): MockState => {
  const entropy = `${projectName}-${randomUUID()}`.replace(/[^a-zA-Z0-9-]/g, "-")
  return {
    presentationId: `task17-html-${entropy}`,
    generationId: randomUUID(),
    generationJobId: randomUUID(),
    source: BASE_SOURCE,
    title: "Task17 base slide",
    etag: '"v1"',
    version: 1,
    generationCompleted: false,
    saveBehavior: "normal",
    principalId: 42,
    presentationAvailable: true,
    requests: []
  }
}

const summary = (state: MockState) => ({
  id: state.presentationId,
  title: state.title,
  description: null,
  theme: "black",
  created_at: "2026-08-22T12:00:00Z",
  last_modified: `2026-08-22T12:00:0${Math.min(state.version, 9)}Z`,
  deleted: false,
  version: state.version,
  provenance: {
    source_kind: "prompt",
    provider: "task17-provider",
    model: "task17-model"
  },
  content_kind: "standalone_html",
  html_slide_count: 1,
  html_bytes: Buffer.byteLength(state.source)
})

const detail = (state: MockState) => ({
  ...summary(state),
  client_id: String(state.principalId),
  source_type: "prompt",
  source_ref: null,
  source_query: null,
  html_document: state.source,
  html_sha256: sha256(state.source),
  generation_job_uuid: state.generationJobId,
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
})

const fulfillJson = async (
  route: Route,
  data: unknown,
  status = 200,
  headers: Record<string, string> = {}
) => {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: { ...CORS_HEADERS, ...headers },
    body: JSON.stringify(data)
  })
}

const installWebUiConfig = async (page: Page, state: MockState) => {
  await page.addInitScript(
    ({ serverUrl, principalId }) => {
      const config = {
        serverUrl,
        authMode: "single-user",
        apiKey: "TASK17-NONSECRET-TEST-KEY"
      }
      localStorage.setItem("tldwConfig", JSON.stringify(config))
      localStorage.setItem("serverUrl", serverUrl)
      localStorage.setItem("tldwServerUrl", serverUrl)
      localStorage.setItem("tldw-api-host", serverUrl)
      localStorage.setItem("authMode", "single-user")
      localStorage.setItem("apiKey", "TASK17-NONSECRET-TEST-KEY")
      localStorage.setItem("isMigrated", "true")
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("assistant_setup_dismissed", "true")
      localStorage.setItem("__tldw_test_bypass", "true")
      ;(window as typeof window & { __task17PrincipalId?: number }).__task17PrincipalId = principalId
    },
    { serverUrl: API_ORIGIN, principalId: state.principalId }
  )
}

const installDeferredDigestAndPageShowProbe = async (page: Page, marker: string) => {
  await page.addInitScript(({ deferredMarker }) => {
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
      const bodySourceVisible = (document.body?.innerText ?? "").includes(deferredMarker)
      const monacoSourceVisible = (scope.monaco?.editor?.getModels?.() ?? []).some((model) =>
        model.getValue().includes(deferredMarker)
      )
      const formSourceVisible = Array.from(document.querySelectorAll("input, textarea")).some(
        (element) => (element as HTMLInputElement | HTMLTextAreaElement).value
          .includes(deferredMarker)
      )
      let historyStateSourceVisible = false
      try {
        historyStateSourceVisible = JSON.stringify(history.state)?.slice(0, 1_048_576)
          .includes(deferredMarker) ?? false
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
      if (hasDeferred || !text.includes(deferredMarker)) return nativeDigest(algorithm, data)
      hasDeferred = true
      scope.__task17DigestDeferred = true
      return new Promise<ArrayBuffer>((resolve, reject) => {
        scope.__task17ReleaseDigest = () => {
          scope.__task17ReleaseDigest = undefined
          void nativeDigest(algorithm, data).then(resolve, reject)
        }
      })
    }) as SubtleCrypto["digest"]
  }, { deferredMarker: marker })
}

const installPageShowProbe = async (page: Page, sourceMarker: string) => {
  await page.addInitScript(({ marker }) => {
    type Task17Window = typeof window & {
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
      const bodySourceVisible = (document.body?.innerText ?? "").includes(marker)
      const monacoSourceVisible = (scope.monaco?.editor?.getModels?.() ?? []).some((model) =>
        model.getValue().includes(marker)
      )
      const formSourceVisible = Array.from(document.querySelectorAll("input, textarea")).some(
        (element) => (element as HTMLInputElement | HTMLTextAreaElement).value.includes(marker)
      )
      let historyStateSourceVisible = false
      try {
        historyStateSourceVisible = JSON.stringify(history.state)?.slice(0, 1_048_576)
          .includes(marker) ?? false
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
  }, { marker: sourceMarker })
}

const installRoutes = async (page: Page, state: MockState) => {
  await page.route("**/openapi.json", async (route) => {
    await fulfillJson(route, {
      info: { version: "task17" },
      paths: {
        "/api/v1/slides/presentations": {},
        "/api/v1/slides/presentations/{presentation_id}": {},
        "/api/v1/slides/presentations/{presentation_id}/export": {}
      }
    })
  })
  await page.route("**/api/v1/**", async (route) => {
    const request = route.request()
    const url = new URL(request.url())
    const method = request.method().toUpperCase()
    const path = url.pathname
    const body = request.postData()
    state.requests.push({ method, path, body, headers: request.headers() })

    if (path === "/api/v1/health") {
      await fulfillJson(route, { status: "ok" })
      return
    }
    if (path === "/api/v1/auth/me") {
      await fulfillJson(route, {
        id: state.principalId,
        username: `task17-owner-${state.principalId}`,
        is_active: true
      })
      return
    }
    if (path === "/api/v1/config/docs-info") {
      await fulfillJson(route, { capabilities: {} })
      return
    }
    if (path === "/api/v1/slides/capabilities") {
      await fulfillJson(route, capabilities, 200, { "Cache-Control": "private, no-store" })
      return
    }
    if (path === "/api/v1/slides/styles") {
      await fulfillJson(route, { styles: [], total: 0 })
      return
    }
    if (path === "/api/v1/slides/generations" && method === "POST") {
      await fulfillJson(route, {
        generation_id: state.generationId,
        status: "queued",
        status_url: `/api/v1/slides/generations/${state.generationId}`,
        presentation_id: null
      }, 202)
      return
    }
    if (path === `/api/v1/slides/generations/${state.generationId}`) {
      await fulfillJson(route, state.generationCompleted
        ? {
            generation_id: state.generationId,
            status: "completed",
            status_url: `/api/v1/slides/generations/${state.generationId}`,
            presentation_id: state.presentationId,
            content_kind: "standalone_html"
          }
        : {
            generation_id: state.generationId,
            status: "running",
            status_url: `/api/v1/slides/generations/${state.generationId}`,
            presentation_id: null,
            progress_text: "Validating generated document"
          })
      return
    }
    if (
      !state.presentationAvailable &&
      path.startsWith(`/api/v1/slides/presentations/${state.presentationId}`)
    ) {
      await fulfillJson(route, { detail: "presentation_not_found" }, 404)
      return
    }
    if (path === `/api/v1/slides/presentations/${state.presentationId}/metadata`) {
      await fulfillJson(route, summary(state), 200, { ETag: state.etag })
      return
    }
    if (path === `/api/v1/slides/presentations/${state.presentationId}` && method === "GET") {
      await fulfillJson(route, detail(state), 200, {
        ETag: state.etag,
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (path === `/api/v1/slides/presentations/${state.presentationId}/html-source` && method === "PUT") {
      if (state.saveBehavior === "conflict") {
        state.saveBehavior = "normal"
        await fulfillJson(route, {
          detail: "presentation_version_conflict",
          current_version: state.version,
          etag: state.etag
        }, 412)
        return
      }
      state.source = body ?? ""
      state.title = state.source.match(/<title>([^<]+)<\/title>/)?.[1] ?? state.title
      state.version += 1
      state.etag = `"v${state.version}"`
      if (state.saveBehavior === "lost") {
        state.saveBehavior = "normal"
        await route.abort("failed")
        return
      }
      await fulfillJson(route, detail(state), 200, {
        ETag: state.etag,
        "Cache-Control": "private, no-store",
        "X-Content-Type-Options": "nosniff"
      })
      return
    }
    if (path === `/api/v1/slides/presentations/${state.presentationId}/draft-attachment` && method === "POST") {
      await route.fulfill({
        status: 200,
        headers: {
          ...CORS_HEADERS,
          "Content-Type": "application/octet-stream",
          "Content-Disposition": 'attachment; filename="presentation.html"',
          "X-Content-Type-Options": "nosniff",
          "X-Download-Options": "noopen",
          "Cache-Control": "private, no-store",
          "Referrer-Policy": "no-referrer",
          "Cross-Origin-Resource-Policy": "same-origin"
        },
        body: Buffer.from(body ?? "")
      })
      return
    }
    if (path === "/api/v1/slides/presentations" && method === "GET") {
      await fulfillJson(route, {
        presentations: state.presentationAvailable ? [summary(state)] : [],
        total: state.presentationAvailable ? 1 : 0,
        limit: 50,
        offset: 0,
        pagination: {
          mode: "offset",
          limit: 50,
          offset: 0,
          total: state.presentationAvailable ? 1 : 0,
          has_more: false,
          next_offset: null
        },
        has_more: false,
        next_offset: null
      })
      return
    }
    if (path.startsWith("/api/v1/notifications")) {
      await fulfillJson(route, path.endsWith("unread-count") ? { unread_count: 0 } : { items: [], total: 0 })
      return
    }
    await fulfillJson(route, {})
  })
}

const openDetail = async (page: Page, state: MockState) => {
  await installWebUiConfig(page, state)
  await installRoutes(page, state)
  await page.goto("/presentation-studio", { waitUntil: "domcontentloaded" })
  await page.getByRole("button", { name: `Open ${state.title}` }).click()
  await expect(page).toHaveURL(new RegExp(`/presentation-studio/${state.presentationId}$`))
  await expect(page.getByRole("heading", { level: 1, name: "Task17 base slide" })).toBeVisible()
}

const browserPrimaryModifier = (page: Page): Promise<"Meta" | "Control"> =>
  page.evaluate(() =>
    /Macintosh|Mac OS X|iPhone|iPad|iPod/i.test(navigator.userAgent)
      ? "Meta"
      : "Control"
  )

const replaceMonacoSource = async (page: Page, source: string) => {
  const editor = page.locator(".monaco-editor").first()
  await expect(editor).toBeVisible()
  const outlineRegion = page.getByRole("region", {
    name: "Safe outline: text only; code never runs in Studio",
    includeHidden: true
  })
  await expect(outlineRegion.getByRole("status", { includeHidden: true })).toHaveText("Current")
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
            getModel: () => {
              getFullModelRange: () => Record<string, number>
            } | null
            getSelection: () => Record<string, number> | null
          }>
        }
      }
    }).monaco
    const activeEditor = monaco?.editor?.getEditors?.().find((candidate) =>
      candidate.hasTextFocus()
    )
    const model = activeEditor?.getModel()
    const selected = activeEditor?.getSelection()
    const full = model?.getFullModelRange()
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

const replaceMonacoSourceWithPendingLastKeystroke = async (
  page: Page,
  source: string,
  marker: string
) => {
  const editor = page.locator(".monaco-editor").first()
  await expect(editor).toBeVisible()
  const markerPrefix = marker.slice(0, -1)
  const prefixSource = source.replace(marker, markerPrefix)
  const insertionOffset = prefixSource.indexOf(markerPrefix) + markerPrefix.length
  expect(insertionOffset).toBeGreaterThan(markerPrefix.length)

  const preparedSource = await editor.evaluate((element, input) => {
    const monaco = (window as typeof window & {
      monaco?: {
        editor?: {
          getEditors?: () => Array<{
            executeEdits: (
              sourceId: string,
              edits: Array<{ range: Record<string, number>; text: string }>
            ) => boolean
            focus: () => void
            getDomNode: () => HTMLElement | null
            getModel: () => {
              getFullModelRange: () => Record<string, number>
              getPositionAt: (offset: number) => Record<string, number>
            } | null
            getValue: () => string
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
    visibleEditor.executeEdits("task17-pending-prefix", [{
      range: model.getFullModelRange(),
      text: input.prefix
    }])
    visibleEditor.setPosition(model.getPositionAt(input.offset))
    visibleEditor.focus()
    return visibleEditor.getValue()
  }, { prefix: prefixSource, offset: insertionOffset })
  expect(preparedSource).toBe(prefixSource)
  expect(await page.evaluate(() => Boolean((
    window as typeof window & { __task17DigestDeferred?: boolean }
  ).__task17DigestDeferred))).toBe(false)

  await page.keyboard.insertText(marker.slice(-1))
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
  await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Not saved")
}

const expectMonacoText = async (page: Page, text: string) => {
  await expect(page.locator(".monaco-editor .view-lines").first()).toContainText(text)
}

const focusByKeyboard = async (page: Page, locator: Locator) => {
  await page.locator("body").click({ position: { x: 2, y: 2 } })
  for (let index = 0; index < 40; index += 1) {
    await page.keyboard.press("Tab")
    if (await locator.evaluate((element) => document.activeElement === element)) return
  }
  throw new Error("keyboard_focus_target_not_reached")
}

test.describe("Standalone HTML Presentation Studio workflow", () => {
  test("generates, resumes, saves after response loss, resolves conflict choices, and reopens", async ({ page }, testInfo) => {
    test.setTimeout(120_000)
    const state = createState(testInfo.project.name)
    await installWebUiConfig(page, state)
    await installRoutes(page, state)

    await page.goto("/presentation-studio/new", { waitUntil: "domcontentloaded" })
    const htmlMode = page.getByRole("radio", { name: /Standalone HTML \+ JavaScript/ })
    await expect(htmlMode).toBeEnabled()
    await htmlMode.check()
    await page.getByLabel("Subject and material").fill("Task17 direct release material")
    await page.getByLabel("Audience").fill("Release reviewers")
    await page.getByRole("button", { name: "Generate standalone presentation" }).click()
    await expect(page.getByRole("heading", { name: "Submitted request" })).toBeVisible()
    await expect(page.getByText("Validating generated document")).toBeVisible()

    await page.getByRole("button", { name: "Stop waiting" }).click()
    await expect(page).toHaveURL(/\/presentation-studio$/)
    state.generationCompleted = true
    await page.goto("/presentation-studio/new")
    await expect(page.getByRole("button", { name: "Resume" })).toBeVisible()
    await page.getByRole("button", { name: "Resume" }).click()
    await expect(page).toHaveURL(new RegExp(`/presentation-studio/${state.presentationId}$`))
    await expect(page.locator(".monaco-editor")).toBeVisible()
    await expect(page.getByText("Trusted outline text", { exact: true })).toBeVisible()
    await expect(page.getByText("Speaker notes")).toBeVisible()

    await replaceMonacoSource(page, LOST_RESPONSE_SOURCE)
    state.saveBehavior = "lost"
    await page.getByRole("button", { name: "Save", exact: true }).click()
    await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Saved")
    expect(state.source).toBe(LOST_RESPONSE_SOURCE)

    await replaceMonacoSource(page, OVERWRITE_SOURCE)
    state.source = SERVER_CONFLICT_SOURCE
    state.title = "Task17 server conflict D"
    state.version = 3
    state.etag = '"v3"'
    state.saveBehavior = "conflict"
    await page.getByRole("button", { name: "Save", exact: true }).click()
    await expect(page.getByRole("heading", { name: "Conflict" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Discard my changes and load server version" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Overwrite server with my draft" })).toBeVisible()
    await expect(page.getByRole("button", { name: "Download my draft" })).toBeVisible()

    const downloadPromise = page.waitForEvent("download")
    await page.getByRole("button", { name: "Download my draft" }).click()
    const download = await downloadPromise
    expect(download.suggestedFilename()).toBe("presentation.html")

    await page.getByRole("button", { name: "Overwrite server with my draft" }).click()
    await expect(page.getByRole("button", { name: "Confirm overwrite" })).toBeVisible()
    await page.getByRole("button", { name: "Confirm overwrite" }).click()
    await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Saved")
    expect(state.source).toBe(OVERWRITE_SOURCE)

    await page.reload({ waitUntil: "domcontentloaded" })
    await expect(page.locator(".monaco-editor")).toBeVisible()
    await expectMonacoText(page, "Task17 edited C")

    await replaceMonacoSource(page, DISCARD_CANDIDATE)
    state.source = DISCARD_SERVER_SOURCE
    state.title = "Task17 fresh server F"
    state.version = 5
    state.etag = '"v5"'
    state.saveBehavior = "conflict"
    await page.getByRole("button", { name: "Save", exact: true }).click()
    await expect(page.getByRole("heading", { name: "Conflict" })).toBeVisible()
    await page.getByRole("button", { name: "Discard my changes and load server version" }).click()
    await page.getByRole("button", { name: "Confirm discard and load server version" }).click()
    await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Saved")
    await expectMonacoText(page, "Task17 fresh server F")

    const generationSubmissions = state.requests.filter(
      (request) => request.method === "POST" && request.path === "/api/v1/slides/generations"
    )
    expect(generationSubmissions).toHaveLength(1)
    expect(generationSubmissions[0].headers["idempotency-key"]).toMatch(/^[A-Za-z0-9._~-]{16,200}$/)
  })

  test("supports keyboard and mobile layout plus scoped pagehide recovery and account fencing", async ({ page }, testInfo) => {
    test.setTimeout(120_000)
    const state = createState(testInfo.project.name)
    await page.setViewportSize({ width: 390, height: 844 })
    await installDeferredDigestAndPageShowProbe(page, PENDING_DIGEST_MARKER)
    await installWebUiConfig(page, state)
    await installRoutes(page, state)
    await page.goto("/presentation-studio", { waitUntil: "domcontentloaded" })
    await page.getByRole("button", { name: `Open ${state.title}` }).click()
    await expect(page).toHaveURL(new RegExp(`/presentation-studio/${state.presentationId}$`))
    await expect(page.getByRole("heading", { level: 1, name: "Task17 base slide" }))
      .toBeVisible()

    await expect(page.getByRole("heading", { level: 1 })).toHaveCount(1)
    const codeTab = page.getByRole("tab", { name: "Code" })
    const outlineTab = page.getByRole("tab", { name: "Outline" })
    for (const tab of [codeTab, outlineTab]) {
      const box = await tab.boundingBox()
      expect(box?.height).toBeGreaterThanOrEqual(44)
      expect(box?.width).toBeGreaterThanOrEqual(44)
    }
    await focusByKeyboard(page, codeTab)
    expect(await codeTab.evaluate((element) => element.matches(":focus-visible"))).toBe(true)
    expect(await codeTab.evaluate((element) => {
      const style = getComputedStyle(element)
      return style.outlineStyle !== "none" || style.boxShadow !== "none"
    })).toBe(true)
    await page.keyboard.press("ArrowRight")
    expect(await outlineTab.evaluate((element) => document.activeElement === element)).toBe(true)
    expect(await outlineTab.evaluate((element) => element.matches(":focus-visible"))).toBe(true)
    await expect(outlineTab).toHaveAttribute("aria-selected", "true")
    await page.keyboard.press("ArrowLeft")
    expect(await codeTab.evaluate((element) => document.activeElement === element)).toBe(true)
    await expect(codeTab).toHaveAttribute("aria-selected", "true")
    expect(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1)).toBe(true)

    await replaceMonacoSourceWithPendingLastKeystroke(
      page,
      PENDING_BACK_SOURCE,
      PENDING_DIGEST_MARKER
    )
    const recoveryKey = `tldw:presentation-studio:html:draft:v1:workspace:${encodeURIComponent(API_ORIGIN)}:${encodeURIComponent("42")}:${encodeURIComponent(state.presentationId)}`
    expect(await page.evaluate(() => (
      window as typeof window & { __task17DigestDeferred?: boolean }
    ).__task17DigestDeferred)).toBe(true)
    let backPrompts = 0
    page.once("dialog", async (dialog) => {
      backPrompts += 1
      expect(dialog.type()).toBe("confirm")
      await dialog.accept()
    })
    await page.goBack({ waitUntil: "domcontentloaded" })
    await expect(page).toHaveURL(/\/presentation-studio$/)
    expect(backPrompts).toBe(1)
    await expect.poll(async () => page.evaluate((key) => {
      const recoveryRecord = sessionStorage.getItem(key)
      return recoveryRecord ? JSON.parse(recoveryRecord).source : null
    }, recoveryKey)).toBe(PENDING_BACK_SOURCE)

    await page.goForward({ waitUntil: "domcontentloaded" })
    await expect.poll(async () => page.evaluate(() => Boolean((
      window as typeof window & { __task17ReleaseDigest?: () => void }
    ).__task17ReleaseDigest))).toBe(true)
    await page.evaluate(() => (
      window as typeof window & { __task17ReleaseDigest?: () => void }
    ).__task17ReleaseDigest?.())
    await expect(page.getByRole("heading", { name: "Recovered draft" })).toBeVisible()
    await page.getByRole("button", { name: "Restore recovered draft" }).click()
    await expectMonacoText(page, "Task17 pending Back candidate")
    const firstHistoryDisposition = await page.evaluate(() => (
      window as typeof window & {
        __task17PageShows?: Array<{ persisted: boolean; sourceVisible: boolean }>
      }
    ).__task17PageShows ?? [])
    if (firstHistoryDisposition.some((entry) => entry.persisted)) {
      expect(firstHistoryDisposition.filter((entry) => entry.persisted).every((entry) =>
        entry.sourceVisible === false
      )).toBe(true)
    } else {
      testInfo.annotations.push({
        type: "engine-limitation",
        description: `${testInfo.project.name} did not use bfcache for pending Back/Forward.`
      })
    }

    await replaceMonacoSource(page, BASE_SOURCE)
    await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Saved")
    await page.evaluate(
      ({ key, origin, presentationId }) => {
        sessionStorage.setItem(key, JSON.stringify({
          schemaVersion: 1,
          principalScope: `${origin}|42`,
          presentationId,
          baseEtag: '"v1"',
          baseDigest: "0".repeat(64),
          source: "expired private source",
          updatedAt: Date.now() - 86_400_001
        }))
      },
      { key: recoveryKey, origin: API_ORIGIN, presentationId: state.presentationId }
    )
    await page.reload({ waitUntil: "domcontentloaded" })
    await expect(page.getByRole("heading", { name: "Recovered draft" })).toHaveCount(0)
    expect(await page.evaluate((key) => sessionStorage.getItem(key), recoveryKey)).toBeNull()

  })

  test("scrubs a clean mounted source and old recovery across a real account-switch history round trip", async ({ page }, testInfo) => {
    test.setTimeout(120_000)
    const state = createState(testInfo.project.name)
    await installPageShowProbe(page, "Task17 base slide")
    await installWebUiConfig(page, state)
    await installRoutes(page, state)
    await page.goto("/presentation-studio", { waitUntil: "domcontentloaded" })
    await page.getByRole("button", { name: `Open ${state.title}` }).click()
    await expect(page).toHaveURL(new RegExp(`/presentation-studio/${state.presentationId}$`))
    await expect(page.locator(".monaco-editor")).toBeVisible()

    const oldRecoveryKey = `tldw:presentation-studio:html:draft:v1:workspace:${encodeURIComponent(API_ORIGIN)}:${encodeURIComponent("42")}:${encodeURIComponent(state.presentationId)}`
    await page.evaluate(({ key, origin, presentationId, source }) => {
      sessionStorage.setItem(key, JSON.stringify({
        schemaVersion: 1,
        principalScope: `${origin}|42`,
        presentationId,
        baseEtag: '"v1"',
        baseDigest: "0".repeat(64),
        source,
        updatedAt: Date.now()
      }))
    }, {
      key: oldRecoveryKey,
      origin: API_ORIGIN,
      presentationId: state.presentationId,
      source: RECOVERY_SOURCE
    })

    let leavePrompts = 0
    page.on("dialog", async (dialog) => {
      leavePrompts += 1
      await dialog.dismiss()
    })
    const metadataReadsBeforeSwitch = state.requests.filter((request) =>
      request.method === "GET" &&
      request.path === `/api/v1/slides/presentations/${state.presentationId}/metadata`
    ).length
    await page.goto("/", { waitUntil: "domcontentloaded" })
    expect(leavePrompts).toBe(0)
    state.principalId = 84
    state.presentationAvailable = false
    await page.goBack({ waitUntil: "domcontentloaded" })
    await expect.poll(() => state.requests.filter((request) =>
      request.method === "GET" &&
      request.path === `/api/v1/slides/presentations/${state.presentationId}/metadata`
    ).length).toBeGreaterThan(metadataReadsBeforeSwitch)
    await expect(page.getByRole("heading", {
      level: 1,
      name: "Presentation metadata is unavailable"
    }))
      .toBeVisible()
    await expect(page.locator("body")).not.toContainText("Task17 base slide")
    await expect(page.locator("body")).not.toContainText("Task17 pagehide recovery")
    await expect.poll(async () => page.evaluate(
      (key) => sessionStorage.getItem(key),
      oldRecoveryKey
    )).toBeNull()

    const disposition = await page.evaluate(() => (
      window as typeof window & {
        __task17PageShows?: Array<{ persisted: boolean; sourceVisible: boolean }>
      }
    ).__task17PageShows ?? [])
    if (disposition.some((entry) => entry.persisted)) {
      expect(disposition.filter((entry) => entry.persisted).every((entry) =>
        entry.sourceVisible === false
      )).toBe(true)
    } else {
      testInfo.annotations.push({
        type: "engine-limitation",
        description: `${testInfo.project.name} did not use bfcache for the clean account-switch Back round trip.`
      })
    }
  })

  test("retains a controlled textarea fallback when the Monaco lazy surface cannot load", async ({ page }, testInfo) => {
    const state = createState(testInfo.project.name)
    let blockedMonacoChunks = 0
    await page.route(/monaco-editor|monaco_editor|monaco\.editor/i, async (route) => {
      blockedMonacoChunks += 1
      await route.abort("failed")
    })
    await openDetail(page, state)

    const fallback = page.locator('textarea[aria-label="HTML source"]')
    await expect(fallback).toBeVisible()
    expect(blockedMonacoChunks).toBeGreaterThan(0)
    await fallback.fill(RECOVERY_SOURCE)
    await expect(page.getByTestId("standalone-html-save-status")).toHaveText("Not saved")
    await expect(page.getByText("Task17 pagehide recovery", { exact: true })).toBeVisible()
  })
})
