import { chromium } from "playwright"

const frontendUrl = process.env.TLDW_WEB_URL ?? "http://127.0.0.1:18109"
const apiOrigin = "http://127.0.0.1:8000"

const source = {
  source_id: "source-ready",
  title: "Queryable report",
  source_type: "document",
  origin_url: "https://example.test/report",
  origin_host: "example.test",
  state: "ready",
  reason_code: null,
  citation_ready: true,
  retrieval_ready: true,
  position: 1,
  added_at: "2026-08-21T10:00:00Z"
}

const bootstrap = {
  schema_version: 1,
  generated_at: "2026-08-22T09:00:00Z",
  share: {
    share_id: 42,
    access_level: "view_chat_add",
    allow_clone: false,
    owner_display_name: "Avery Owner",
    shared_at: "2026-08-20T12:00:00Z"
  },
  workspace: {
    workspace_id: "workspace-shared",
    name: "Election evidence review",
    description: "Shared evidence for recipient review."
  },
  allowed_actions: {
    inspect_sources: { allowed: true, reason_code: null },
    ask_grounded_questions: { allowed: true, reason_code: null },
    add_sources: { allowed: false, reason_code: "recipient_mutation_disabled" },
    edit_workspace: { allowed: false, reason_code: "recipient_mutation_disabled" },
    clone_workspace: { allowed: false, reason_code: "owner_disabled" }
  },
  generation_default: {
    provider: "anthropic",
    model: "claude-shared",
    ready: true,
    reason_code: null
  },
  source_summary: { total: 1, queryable: 1, processing: 0, failed: 0 },
  sources: {
    items: [source],
    pagination: { offset: 0, limit: 50, total: 1, has_more: false }
  },
  conversation: {
    conversation_id: "conversation-1",
    messages: [
      {
        message_id: "message-existing",
        role: "assistant",
        content: "Existing **grounded** answer with `inert code`.",
        created_at: "2026-08-21T11:30:00Z",
        citations: []
      }
    ],
    next_before: null
  },
  partial_errors: []
}

const preview = {
  source_id: source.source_id,
  title: source.title,
  source_type: source.source_type,
  origin_url: source.origin_url,
  origin_host: source.origin_host,
  state: source.state,
  reason_code: null,
  content_available: true,
  preview_mode: "content_excerpt",
  unavailable_reason: null,
  text_preview: "Focused source preview",
  text_total_chars: 22,
  text_truncated: false,
  snippets: [],
  generated_at: "2026-08-22T09:00:01Z"
}

const chatResponse = {
  schema_version: 1,
  request_id: "00000000-0000-4000-8000-000000000042",
  conversation_id: "conversation-1",
  turn: {
    user_message: {
      message_id: "message-user-new",
      role: "user",
      content: "Summarize this source",
      created_at: "2026-08-22T09:00:02Z"
    },
    assistant_message: {
      message_id: "message-assistant-new",
      role: "assistant",
      content: "The source supports one conclusion.",
      created_at: "2026-08-22T09:00:03Z"
    }
  },
  citations: [
    {
      citation_id: "citation-1",
      source_id: source.source_id,
      source_title: source.title,
      locator: { chunk: 1, start_char: 0, end_char: 10 },
      quote: "Focused evidence",
      score: 0.9
    }
  ],
  generation: { provider: "anthropic", model: "claude-shared" },
  source_scope: { mode: "all", effective_source_count: 1 },
  replay: { replayed: false }
}

const assert = (condition, message) => {
  if (!condition) throw new Error(message)
}

const body64 = (value) =>
  Buffer.from(JSON.stringify(value), "utf8").toString("base64")

const evaluate = async (cdp, expression) => {
  const response = await cdp.send("Runtime.evaluate", {
    expression,
    awaitPromise: true,
    returnByValue: true
  })
  if (response.exceptionDetails) {
    throw new Error(response.exceptionDetails.text)
  }
  return response.result.value
}

const waitFor = async (cdp, expression, timeoutMs = 15_000) => {
  const started = Date.now()
  while (Date.now() - started < timeoutMs) {
    if (await evaluate(cdp, expression)) return
    await new Promise((resolve) => setTimeout(resolve, 50))
  }
  throw new Error(`Timed out waiting for: ${expression}`)
}

const fulfill = (cdp, requestId, value, responseCode = 200) =>
  cdp.send("Fetch.fulfillRequest", {
    requestId,
    responseCode,
    responseHeaders: [
      { name: "Content-Type", value: "application/json" },
      { name: "Access-Control-Allow-Origin", value: frontendUrl },
      {
        name: "Access-Control-Allow-Headers",
        value: "Authorization, Content-Type, X-API-KEY"
      },
      { name: "Access-Control-Allow-Methods", value: "GET, POST, OPTIONS" }
    ],
    body: body64(value)
  })

const installApiFixtures = async (cdp) => {
  const held = { chat: null, preview: null }
  await cdp.send("Fetch.enable", {
    patterns: [{ urlPattern: `${apiOrigin}/*`, requestStage: "Request" }]
  })
  cdp.on("Fetch.requestPaused", async (event) => {
    const url = new URL(event.request.url)
    if (event.request.method === "OPTIONS") {
      await fulfill(cdp, event.requestId, {}, 204)
      return
    }
    if (url.pathname.endsWith("/shared-with-me/42/workspace")) {
      await fulfill(cdp, event.requestId, bootstrap)
      return
    }
    if (url.pathname.endsWith("/sources/source-ready/preview")) {
      held.preview = event.requestId
      return
    }
    if (
      url.pathname.endsWith("/shared-with-me/42/chat") &&
      event.request.method === "POST"
    ) {
      const request = JSON.parse(event.request.postData ?? "{}")
      held.chat = {
        requestId: event.requestId,
        response: { ...chatResponse, request_id: request.request_id }
      }
      return
    }
    if (url.pathname.includes("/models")) {
      await fulfill(cdp, event.requestId, [])
      return
    }
    await fulfill(cdp, event.requestId, {})
  })
  return held
}

const commonMetricsExpression = `(() => {
  const rect = (element) => {
    if (!element) return null
    const value = element.getBoundingClientRect()
    return { left: value.left, right: value.right, top: value.top, bottom: value.bottom, width: value.width, height: value.height }
  }
  const visible = (element) => {
    const value = element.getBoundingClientRect()
    const style = getComputedStyle(element)
    return value.width > 0 && value.height > 0 && style.visibility !== "hidden" && style.display !== "none"
  }
  const controls = Array.from(document.querySelectorAll("button,input,select,textarea,a"))
    .filter(visible)
    .map((element) => ({ label: element.getAttribute("aria-label") || element.textContent.trim().slice(0, 40), ...rect(element) }))
  const shell = document.querySelector('[data-testid="shared-workspace-shell"]')
  const sources = document.querySelector('[data-testid="shared-workspace-sources-pane"]')
  const chat = document.querySelector('[data-testid="shared-workspace-chat-pane"]')
  return {
    viewport: { width: innerWidth, height: innerHeight },
    documentOverflowX: document.documentElement.scrollWidth - document.documentElement.clientWidth,
    bodyOverflowX: document.body.scrollWidth - document.body.clientWidth,
    shell: rect(shell),
    shellOverflowX: shell ? shell.scrollWidth - shell.clientWidth : null,
    sources: rect(sources),
    sourcesOverflowX: sources ? sources.scrollWidth - sources.clientWidth : null,
    chat: rect(chat),
    chatOverflowX: chat ? chat.scrollWidth - chat.clientWidth : null,
    outOfBoundsControls: controls.filter((control) => control.left < -0.5 || control.right > innerWidth + 0.5 || control.top < -0.5 || control.bottom > innerHeight + 0.5)
  }
})()`

const createSession = async (browser, width, height) => {
  const context = await browser.newContext()
  const page = await context.newPage()
  const cdp = await context.newCDPSession(page)
  await cdp.send("Page.enable")
  await cdp.send("Runtime.enable")
  await cdp.send("Emulation.setDeviceMetricsOverride", {
    width,
    height,
    deviceScaleFactor: 1,
    mobile: width < 768
  })
  await cdp.send("Page.addScriptToEvaluateOnNewDocument", {
    source: `(() => {
      const config = { serverUrl: "${apiOrigin}", apiKey: "task9-test-key", authMode: "single-user" }
      localStorage.setItem("tldwConfig", JSON.stringify(config))
      localStorage.setItem("serverUrl", config.serverUrl)
      localStorage.setItem("tldwServerUrl", config.serverUrl)
      localStorage.setItem("tldw-api-host", config.serverUrl)
      localStorage.setItem("apiKey", config.apiKey)
      localStorage.setItem("authMode", config.authMode)
      localStorage.setItem("isMigrated", "true")
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("assistant_setup_dismissed", "true")
      localStorage.setItem("__tldw_test_bypass", "true")
    })()`
  })
  const held = await installApiFixtures(cdp)
  await cdp.send("Page.navigate", {
    url: `${frontendUrl}/research-workspace?shared=42`
  })
  try {
    await waitFor(
      cdp,
      `Boolean(document.querySelector('[data-testid="shared-workspace-shell"]'))`
    )
  } catch (error) {
    const diagnosis = await evaluate(
      cdp,
      `({ href: location.href, text: document.body?.innerText?.slice(0, 1200), html: document.body?.innerHTML?.slice(0, 2000) })`
    )
    throw new Error(`${error.message}\n${JSON.stringify(diagnosis, null, 2)}`)
  }
  return { cdp, context, held }
}

const runMobile = async (browser) => {
  const { cdp, context, held } = await createSession(browser, 390, 844)
  const base = await evaluate(cdp, commonMetricsExpression)
  assert(base.viewport.width === 390 && base.viewport.height === 844, "Mobile viewport mismatch")
  assert(base.documentOverflowX === 0 && base.bodyOverflowX === 0, "Mobile root overflow")
  assert(base.shellOverflowX === 0 && base.sourcesOverflowX === 0, "Mobile pane overflow")
  assert(base.outOfBoundsControls.length === 0, "Mobile controls exceed viewport")
  assert(
    base.sources?.width > 0 && (!base.chat || base.chat.width === 0),
    "Mobile Sources tab must be the only visible pane"
  )

  await evaluate(
    cdp,
    `document.querySelector('button[aria-label="Preview Queryable report"]').click()`
  )
  await waitFor(cdp, `Boolean(document.querySelector('[role="dialog"]'))`)
  await waitFor(cdp, `document.body.textContent.includes("Loading source preview")`)
  const loading = await evaluate(
    cdp,
    `(() => {
      const dialog = document.querySelector('[role="dialog"]')
      const wrapper = dialog?.closest('.ant-drawer-content-wrapper')
      const rect = wrapper?.getBoundingClientRect()
      return {
        label: dialog?.getAttribute('aria-label') || dialog?.textContent,
        rect: rect ? { top: rect.top, bottom: rect.bottom, width: rect.width, height: rect.height } : null,
        bodyOverflowX: document.body.scrollWidth - document.body.clientWidth
      }
    })()`
  )
  assert(held.preview, "Mobile preview request was not held")
  assert(loading.label?.includes("Loading source preview"), "Mobile loading label missing")
  assert(loading.rect?.top === 0 && loading.rect?.bottom === 844, "Mobile sheet is not full height")
  assert(loading.rect?.width === 390, "Mobile sheet is not full width")
  assert(loading.bodyOverflowX === 0, "Mobile preview introduced overflow")
  await fulfill(cdp, held.preview, preview)
  await waitFor(cdp, `document.body.textContent.includes("Focused source preview")`)
  const loadedLabel = await evaluate(
    cdp,
    `document.querySelector('[role="dialog"]')?.textContent.includes("Source preview")`
  )
  assert(loadedLabel, "Mobile loaded preview label missing")
  await context.close()
  return { base, loading, loadedLabel }
}

const runDesktop = async (browser) => {
  const { cdp, context, held } = await createSession(browser, 1440, 900)
  const base = await evaluate(cdp, commonMetricsExpression)
  assert(base.viewport.width === 1440 && base.viewport.height === 900, "Desktop viewport mismatch")
  assert(base.documentOverflowX === 0 && base.bodyOverflowX === 0, "Desktop root overflow")
  assert(base.shellOverflowX === 0, "Desktop shell overflow")
  assert(base.sourcesOverflowX === 0 && base.chatOverflowX === 0, "Desktop pane overflow")
  assert(base.outOfBoundsControls.length === 0, "Desktop controls exceed viewport")
  assert(base.sources && base.chat && base.sources.right <= base.chat.left + 1, "Desktop panes overlap")

  await waitFor(
    cdp,
    `!document.querySelector('button[aria-label="Ask shared workspace"]').disabled`
  ).catch(async () => {
    await evaluate(
      cdp,
      `(() => {
        const input = document.querySelector('textarea[aria-label="Ask about shared sources"]')
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
        setter.call(input, 'Summarize this source')
        input.dispatchEvent(new Event('input', { bubbles: true }))
      })()`
    )
    await waitFor(
      cdp,
      `!document.querySelector('button[aria-label="Ask shared workspace"]').disabled`
    )
  })
  const before = await evaluate(
    cdp,
    `(() => {
      const input = document.querySelector('textarea[aria-label="Ask about shared sources"]')
      if (!input.value) {
        const setter = Object.getOwnPropertyDescriptor(HTMLTextAreaElement.prototype, 'value').set
        setter.call(input, 'Summarize this source')
        input.dispatchEvent(new Event('input', { bubbles: true }))
      }
      const button = document.querySelector('button[aria-label="Ask shared workspace"]')
      const rect = button.getBoundingClientRect()
      return { width: rect.width, height: rect.height, label: button.getAttribute('aria-label') }
    })()`
  )
  await waitFor(
    cdp,
    `!document.querySelector('button[aria-label="Ask shared workspace"]').disabled`
  )
  await evaluate(
    cdp,
    `document.querySelector('button[aria-label="Ask shared workspace"]').click()`
  )
  await waitFor(
    cdp,
    `Boolean(document.querySelector('button[aria-label="Asking shared workspace"]'))`
  )
  const during = await evaluate(
    cdp,
    `(() => {
      const button = document.querySelector('button[aria-label="Asking shared workspace"]')
      const rect = button.getBoundingClientRect()
      return { width: rect.width, height: rect.height, label: button.getAttribute('aria-label'), disabled: button.disabled }
    })()`
  )
  assert(held.chat, "Desktop chat request was not held")
  assert(during.disabled && during.label === "Asking shared workspace", "Desktop loading label missing")
  assert(before.width === during.width && before.height === during.height, "Submit control shifted while loading")
  await fulfill(cdp, held.chat.requestId, held.chat.response)
  await waitFor(cdp, `document.body.textContent.includes("Answer added")`)
  await context.close()
  return { base, before, during }
}

const browser = await chromium.launch({
  headless: true,
  executablePath:
    process.env.CHROME_PATH ??
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
})
try {
  const mobile = await runMobile(browser)
  const desktop = await runDesktop(browser)
  process.stdout.write(`${JSON.stringify({ mobile, desktop }, null, 2)}\n`)
} finally {
  await browser.close()
}
