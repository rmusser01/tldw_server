import {
  type BrowserContext,
  type Locator,
  type Page,
  chromium,
  expect,
  test
} from "@playwright/test"
import fs from "node:fs"
import http from "node:http"
import { AddressInfo } from "node:net"
import os from "node:os"
import path from "node:path"

import {
  forceConnected,
  setSelectedModel,
  waitForConnectionStore
} from "./utils/connection"
import { launchWithExtension } from "./utils/extension"
import { resolveExtensionHeadlessMode } from "./utils/extension-common"
import { grantHostPermission } from "./utils/permissions"

const EXT_PATH = path.resolve(
  process.env.TLDW_E2E_EXTENSION_PATH || ".output/chrome-mv3"
)
const MODEL_ID = "prompt-improvement-model"
const MODEL_KEY = `tldw:${MODEL_ID}`
const TEMPLATE_ID = "e2e-prompt-improvement-template"
const TEMPLATE_TITLE = "E2E selected system template"
const SYSTEM_COUNTERPART = "SYSTEM_COUNTERPART_SENTINEL"
const USER_COUNTERPART = "USER_COUNTERPART_SENTINEL"
const HISTORY_SENTINEL = "HISTORY_SENTINEL"
const PAGE_CONTEXT_SENTINEL = "PAGE_CONTEXT_SENTINEL"
const RAG_SENTINEL = "RAG_SENTINEL"
const TOOL_SENTINEL = "TOOL_SENTINEL"
const EXCLUDED_SENTINELS = [
  SYSTEM_COUNTERPART,
  USER_COUNTERPART,
  HISTORY_SENTINEL,
  PAGE_CONTEXT_SENTINEL,
  RAG_SENTINEL,
  TOOL_SENTINEL
] as const
const REQUEST_KEYS = [
  "model_selection",
  "operation_id",
  "protected_tokens",
  "target",
  "text"
]
const AXE_SOURCE_PATH = [
  path.resolve("../packages/ui/node_modules/axe-core/axe.min.js"),
  path.resolve("packages/ui/node_modules/axe-core/axe.min.js"),
  path.resolve("apps/packages/ui/node_modules/axe-core/axe.min.js")
].find((candidate) => fs.existsSync(candidate))
if (!AXE_SOURCE_PATH) {
  throw new Error("Could not resolve the workspace axe-core browser bundle")
}
const AXE_SOURCE = fs.readFileSync(AXE_SOURCE_PATH, "utf8")

const LIMITS = {
  max_request_bytes: 64_000,
  max_draft_chars: 24_000,
  max_candidate_chars: 24_000,
  max_raw_output_chars: 32_000,
  max_findings: 5,
  max_finding_text_chars: 500,
  max_provider_chars: 100,
  max_model_chars: 500,
  max_meta_prompt_version_chars: 100,
  max_warning_chars: 100,
  max_warnings: 16,
  max_protected_tokens: 64,
  max_protected_token_kind_chars: 50,
  max_protected_token_chars: 500,
  max_protected_token_occurrences: 100,
  max_protected_token_total_chars: 4_000
}

type CapabilityMode = "supported" | "false" | "404" | "offline"

type PromptMockOptions = {
  failFirstImprovement?: boolean
}

type RecordedRequest = {
  method: string
  url: string
  body: string
  json: Record<string, unknown> | null
}

type PromptMockServer = {
  server: http.Server
  baseUrl: string
  requests: RecordedRequest[]
  improveRequests: () => RecordedRequest[]
  releaseNextDeferred: () => boolean
}

const readBody = (req: http.IncomingMessage) =>
  new Promise<string>((resolve) => {
    let body = ""
    req.on("data", (chunk) => {
      body += chunk
    })
    req.on("end", () => resolve(body))
  })

const parseBody = (body: string): Record<string, unknown> | null => {
  if (!body) return null
  try {
    const value = JSON.parse(body)
    return value && typeof value === "object" && !Array.isArray(value)
      ? value
      : null
  } catch {
    return null
  }
}

const startPromptMockServer = async (
  capabilityMode: CapabilityMode = "supported",
  options: PromptMockOptions = {}
): Promise<PromptMockServer> => {
  const requests: RecordedRequest[] = []
  const deferredResolvers: Array<() => void> = []
  let providerFailureAttempts = 0

  const server = http.createServer(async (req, res) => {
    const method = String(req.method || "GET").toUpperCase()
    const url = req.url || "/"

    const sendJson = (status: number, payload: unknown) => {
      res.writeHead(status, {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers":
          "content-type, x-api-key, authorization",
        "access-control-allow-methods": "GET, POST, PATCH, OPTIONS"
      })
      res.end(JSON.stringify(payload))
    }

    if (method === "OPTIONS") {
      res.writeHead(204, {
        "access-control-allow-origin": "*",
        "access-control-allow-headers":
          "content-type, x-api-key, authorization",
        "access-control-allow-methods": "GET, POST, PATCH, OPTIONS"
      })
      return res.end()
    }

    const body = ["POST", "PUT", "PATCH"].includes(method)
      ? await readBody(req)
      : ""
    requests.push({ method, url, body, json: parseBody(body) })

    if (url === "/api/v1/health" && method === "GET") {
      return sendJson(200, { status: "ok" })
    }
    if (url.startsWith("/api/v1/llm/models/metadata") && method === "GET") {
      return sendJson(200, [
        {
          id: MODEL_ID,
          name: "Prompt Improvement Model",
          provider: "mock",
          context_length: 4096,
          capabilities: ["chat"]
        }
      ])
    }
    if (url === "/api/v1/llm/models" && method === "GET") {
      return sendJson(200, [MODEL_ID])
    }
    if (url.startsWith("/api/v1/users/me/profile") && method === "GET") {
      return sendJson(200, { preferences: {} })
    }
    if (url === "/api/v1/users/me/profile" && method === "PATCH") {
      return sendJson(200, { preferences: {} })
    }
    if (url === "/openapi.json" && method === "GET") {
      return sendJson(200, {
        openapi: "3.0.0",
        info: { version: "prompt-improvement-e2e" },
        paths: {
          "/api/v1/health": {},
          "/api/v1/llm/models": {},
          "/api/v1/llm/models/metadata": {},
          "/api/v1/prompts/capabilities": {},
          "/api/v1/prompts/improve": {}
        }
      })
    }
    if (url === "/api/v1/prompts/capabilities" && method === "GET") {
      if (capabilityMode === "offline") {
        req.socket.destroy()
        return
      }
      if (capabilityMode === "404") {
        return sendJson(404, { detail: "not found" })
      }
      return sendJson(200, {
        prompt_improvement_v1: {
          supported: capabilityMode === "supported",
          limits: LIMITS
        },
        single_text_recipe_v2: { supported: false }
      })
    }
    if (url === "/api/v1/prompts/improve" && method === "POST") {
      const payload = parseBody(body) || {}
      const text = String(payload.text || "")
      if (options.failFirstImprovement) {
        providerFailureAttempts += 1
        if (providerFailureAttempts === 1) {
          return sendJson(503, {
            code: "provider_unavailable",
            message: "sanitized provider unavailable",
            retryable: true,
            request_id: "prompt-e2e-provider-failure"
          })
        }
      }
      if (text.includes("[DEFER_STALE]") || text.includes("[DEFER_CONFIRM]")) {
        await new Promise<void>((resolve) => deferredResolvers.push(resolve))
      }
      const target = payload.target === "system" ? "system" : "user_message"
      const improvedText =
        target === "system"
          ? "Improved system instruction for {{topic}}."
          : text.includes("[DEFER_STALE]")
            ? "Deferred stale candidate."
            : text.includes("[DEFER_CONFIRM]")
              ? "Deferred confirmed replacement."
              : "Improved user request for {{topic}}."
      return sendJson(200, {
        schema_version: 1,
        operation_id: payload.operation_id,
        status: "improved",
        improved_text: improvedText,
        findings: [
          {
            category: "clarity",
            issue: "The request was ambiguous.",
            change: "Clarified the requested outcome."
          }
        ],
        review_required: false,
        warnings: [],
        resolved_model: {
          provider: "mock",
          model: MODEL_ID,
          display_name: "Prompt Improvement Model"
        },
        meta_prompt_version: "prompt-improvement-v1"
      })
    }
    if (url === "/api/v1/chat/completions" && method === "POST") {
      return sendJson(200, {
        choices: [
          {
            message: { role: "assistant", content: "Prompt E2E chat reply" }
          }
        ]
      })
    }
    return sendJson(404, { detail: "not found" })
  })

  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  const address = server.address() as AddressInfo
  return {
    server,
    baseUrl: `http://127.0.0.1:${address.port}`,
    requests,
    improveRequests: () =>
      requests.filter(
        (request) =>
          request.method === "POST" && request.url === "/api/v1/prompts/improve"
      ),
    releaseNextDeferred: () => {
      const resolve = deferredResolvers.shift()
      if (!resolve) return false
      resolve()
      return true
    }
  }
}

const stopPromptMockServer = async (mock: PromptMockServer) => {
  while (mock.releaseNextDeferred()) {
    // Drain any deliberately deferred provider completions before shutdown.
  }
  mock.server.closeAllConnections?.()
  await new Promise<void>((resolve) => mock.server.close(() => resolve()))
}

const MOCK_API_KEY = "prompt-improvement-e2e-key"

const buildSeedConfig = (baseUrl: string, apiKey = MOCK_API_KEY) => ({
  __tldw_first_run_complete: true,
  __tldw_allow_offline: true,
  "tldw:seenHints": {
    "knowledge-search": true,
    "more-tools": true
  },
  tldwConfig: {
    serverUrl: baseUrl,
    authMode: "single-user",
    apiKey
  },
  tldw_skip_landing_hub: true
})

const buildRealServerHarnessConfig = (serverUrl: string, apiKey: string) => ({
  capabilityUrl: `${serverUrl.replace(/\/$/, "")}/api/v1/prompts/capabilities`,
  capabilityHeaders: { "X-API-KEY": apiKey },
  seedConfig: buildSeedConfig(serverUrl, apiKey)
})

const uiModeStorage = (mode: "casual" | "pro") =>
  JSON.stringify({ state: { mode }, version: 0 })

const ensureChatInput = async (page: Page) => {
  const startButton = page.getByRole("button", { name: /Start chatting/i })
  if (
    (await startButton.count()) > 0 &&
    (await startButton.first().isVisible())
  ) {
    await startButton.first().click()
  }
  const input = page
    .getByTestId("chat-input")
    .or(
      page.getByRole("textbox", {
        name: /^(Message|Message input)$/i
      })
    )
    .filter({ visible: true })
  await expect(input).toHaveCount(1, { timeout: 20_000 })
  await expect(input).toBeVisible({ timeout: 20_000 })
  await expect(input).toBeEditable()
  return input
}

const seedPromptTemplate = async (page: Page) => {
  await page.evaluate(
    ({ id, title, content }) =>
      new Promise<void>((resolve, reject) => {
        const openRequest = indexedDB.open("PageAssistDatabase")
        openRequest.onerror = () => reject(openRequest.error)
        openRequest.onsuccess = () => {
          const database = openRequest.result
          const transaction = database.transaction("prompts", "readwrite")
          transaction.objectStore("prompts").put({
            id,
            title,
            content,
            is_system: true,
            favorite: false,
            createdBy: "e2e",
            createdAt: Date.now(),
            updatedAt: Date.now(),
            deletedAt: null,
            syncStatus: "local",
            sourceSystem: "workspace"
          })
          transaction.oncomplete = () => {
            database.close()
            resolve()
          }
          transaction.onerror = () => reject(transaction.error)
        }
      }),
    {
      id: TEMPLATE_ID,
      title: TEMPLATE_TITLE,
      content: `${SYSTEM_COUNTERPART} Keep {{topic}} literal.`
    }
  )
}

const seedExcludedContext = async (page: Page, selectedModel = MODEL_KEY) => {
  await page.evaluate(
    ({ model, history, pageContext, rag, tool }) => {
      const store = (window as any).__tldw_useStoreMessageOption
      if (!store?.setState)
        throw new Error("Message option store is unavailable")
      store.setState({
        selectedModel: model,
        history: [{ role: "user", content: history }],
        messages: [
          {
            id: "prompt-e2e-history",
            isBot: false,
            name: "You",
            role: "user",
            message: history,
            sources: []
          }
        ],
        documentContext: [
          { title: pageContext, type: "tab", url: "https://example.invalid" }
        ],
        contextFiles: [
          {
            id: "prompt-e2e-context",
            filename: pageContext,
            type: "text/plain",
            content: pageContext,
            size: pageContext.length,
            uploadedAt: Date.now(),
            processed: true
          }
        ],
        selectedKnowledge: {
          id: "prompt-e2e-knowledge",
          title: rag,
          body: rag
        },
        ragPinnedResults: [{ id: "prompt-e2e-rag", title: rag, snippet: rag }],
        toolChoice: "required",
        actionInfo: tool
      })
    },
    {
      model: selectedModel,
      history: HISTORY_SENTINEL,
      pageContext: PAGE_CONTEXT_SENTINEL,
      rag: RAG_SENTINEL,
      tool: TOOL_SENTINEL
    }
  )
}

type SurfaceLaunch = {
  context: BrowserContext
  bootstrapPage: Page
  chatPage: Page
  extensionId: string
}

const launchChatSurface = async (
  mock: PromptMockServer,
  surface: "sidepanel" | "options",
  options: {
    withModel?: boolean
    mode?: "casual" | "pro"
    nextgen?: boolean
    variant?: "v1" | "v3" | "v5"
    viewport?: { width: number; height: number }
    seedTemplate?: boolean
  } = {}
): Promise<SurfaceLaunch> => {
  const {
    withModel = true,
    mode = "casual",
    nextgen = false,
    variant = "v1",
    viewport,
    seedTemplate = false
  } = options
  const launched = await launchWithExtension(EXT_PATH, {
    seedConfig: buildSeedConfig(mock.baseUrl),
    seedLocalStorage: {
      "tldw-ui-mode": uiModeStorage(mode),
      "tldw:nextgenComposerEnabled": nextgen ? "1" : "0",
      "tldw:composerVariant": variant
    }
  })
  const { context, page, openSidepanel, extensionId, optionsUrl } = launched
  try {
    const permission = await grantHostPermission(
      context,
      extensionId,
      `${new URL(mock.baseUrl).origin}/*`
    )
    expect(
      permission,
      "Packaged extension must receive local mock host access"
    ).toBe(true)
    if (seedTemplate) await seedPromptTemplate(page)
    if (withModel) await setSelectedModel(page, MODEL_KEY)

    const chatPage =
      surface === "sidepanel" ? await openSidepanel("/chat") : page
    if (surface === "options") {
      await chatPage.goto(`${optionsUrl}#/chat`, {
        waitUntil: "domcontentloaded"
      })
    }
    if (viewport) await chatPage.setViewportSize(viewport)
    await waitForConnectionStore(chatPage, `prompt-improvement:${surface}`)
    await forceConnected(
      chatPage,
      { serverUrl: mock.baseUrl },
      `prompt-improvement:${surface}:connected`
    )
    await ensureChatInput(chatPage)
    if (withModel) await seedExcludedContext(chatPage)
    return { context, bootstrapPage: page, chatPage, extensionId }
  } catch (setupError) {
    try {
      await context.close()
    } catch (closeError) {
      throw new AggregateError(
        [setupError, closeError],
        "Extension setup failed and its browser context did not close cleanly"
      )
    }
    throw setupError
  }
}

const openPromptActions = async (page: Page, scope: Page | Locator = page) => {
  const trigger = scope.getByRole("button", { name: "Improve prompt" })
  await expect(trigger).toBeVisible({ timeout: 20_000 })
  await trigger.click()
  const actions = page.getByRole("group", {
    name: "Prompt improvement actions"
  })
  await expect(actions).toBeVisible()
  await expect(actions).not.toContainText(/Build from recipe/i)
  return { trigger, actions }
}

const assertPromptRequest = (
  request: RecordedRequest,
  expected: { target: "system" | "user_message"; text: string }
) => {
  expect(request.json).not.toBeNull()
  const payload = request.json as Record<string, unknown>
  expect(Object.keys(payload).sort()).toEqual([...REQUEST_KEYS].sort())
  expect(payload.operation_id).toMatch(
    /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
  )
  expect(payload.target).toBe(expected.target)
  expect(payload.text).toBe(expected.text)
  expect(payload.model_selection).toMatchObject({ selected_model: MODEL_KEY })
  expect(payload.protected_tokens).toEqual(
    expected.text.includes("{{topic}}")
      ? [{ kind: "template_variable", value: "{{topic}}", occurrences: 1 }]
      : []
  )
  const serialized = JSON.stringify(payload)
  for (const sentinel of EXCLUDED_SENTINELS) {
    if (expected.text.includes(sentinel)) continue
    expect(serialized).not.toContain(sentinel)
  }
}

const selectTemplate = async (page: Page) => {
  const promptTrigger = page.getByTestId("chat-prompt-select")
  await expect(promptTrigger).toBeVisible({ timeout: 20_000 })
  await promptTrigger.click()
  const templateItem = page.getByRole("menuitem", {
    name: new RegExp(TEMPLATE_TITLE)
  })
  await expect(templateItem).toBeVisible({ timeout: 20_000 })
  await templateItem.click()
  await expect(promptTrigger).toContainText(TEMPLATE_TITLE)
  return promptTrigger
}

const openSystemEditor = async (page: Page) => {
  const promptTrigger = page.getByTestId("chat-prompt-select")
  await promptTrigger.click()
  await page.getByRole("menuitem", { name: /Edit system prompt/i }).click()
  const editor = page.getByRole("textbox", { name: "Enter system prompt" })
  await expect(editor).toBeVisible()
  return editor
}

const requirePromptImprovementRealServerConfig = () => {
  const serverUrl = String(process.env.TLDW_E2E_SERVER_URL || "")
    .trim()
    .replace(/\/$/, "")
  const apiKey = String(process.env.TLDW_E2E_API_KEY || "").trim()
  if (!serverUrl || !apiKey) {
    throw new Error(
      "Prompt-improvement E2E requires TLDW_E2E_SERVER_URL and TLDW_E2E_API_KEY."
    )
  }
  return { serverUrl, apiKey }
}

const probePackagedRuntime = async () => {
  const profileDir = fs.mkdtempSync(
    path.join(os.tmpdir(), "tldw-prompt-improvement-preflight-")
  )
  const configuredLaunchTimeout = Number.parseInt(
    String(process.env.TLDW_E2E_EXTENSION_LAUNCH_TIMEOUT_MS || ""),
    10
  )
  const launchTimeout =
    Number.isFinite(configuredLaunchTimeout) && configuredLaunchTimeout > 0
      ? configuredLaunchTimeout
      : 30_000
  let context: BrowserContext | null = null
  try {
    context = await chromium.launchPersistentContext(profileDir, {
      timeout: launchTimeout,
      headless: resolveExtensionHeadlessMode(),
      executablePath:
        process.env.PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH ||
        chromium.executablePath(),
      ignoreDefaultArgs: ["--disable-extensions"],
      args: [
        `--disable-extensions-except=${EXT_PATH}`,
        `--load-extension=${EXT_PATH}`,
        "--no-crashpad",
        "--disable-crash-reporter",
        "--crash-dumps-dir=/tmp"
      ]
    })
    const targetWait = Number.parseInt(
      String(process.env.TLDW_E2E_EXTENSION_TARGET_WAIT_MS || "30000"),
      10
    )
    if (!context.serviceWorkers().length && !context.backgroundPages().length) {
      const foundTarget = await Promise.race([
        context.waitForEvent("serviceworker").then(() => true),
        context.waitForEvent("backgroundpage").then(() => true),
        new Promise<false>((resolve) =>
          setTimeout(() => resolve(false), targetWait)
        )
      ])
      if (!foundTarget) {
        throw new Error(
          "Could not determine extension id from [no extension targets]"
        )
      }
    }
  } finally {
    await context?.close()
    fs.rmSync(profileDir, { recursive: true, force: true })
  }
}

test("real-server harness uses the configured API key for fetch and extension storage", () => {
  const apiKey = "live-api-key-sentinel"
  const realServer = buildRealServerHarnessConfig(
    "http://127.0.0.1:8000/",
    apiKey
  )

  expect(realServer.capabilityHeaders["X-API-KEY"]).toBe(apiKey)
  expect(realServer.seedConfig.tldwConfig.apiKey).toBe(apiKey)
  expect(buildSeedConfig("http://127.0.0.1:8000").tldwConfig.apiKey).toBe(
    MOCK_API_KEY
  )
})

test("real-server harness fails closed when required configuration is missing", () => {
  const previousServerUrl = process.env.TLDW_E2E_SERVER_URL
  const previousApiKey = process.env.TLDW_E2E_API_KEY
  try {
    delete process.env.TLDW_E2E_SERVER_URL
    delete process.env.TLDW_E2E_API_KEY
    expect(requirePromptImprovementRealServerConfig).toThrow(
      "Prompt-improvement E2E requires TLDW_E2E_SERVER_URL and TLDW_E2E_API_KEY."
    )
  } finally {
    if (previousServerUrl === undefined) delete process.env.TLDW_E2E_SERVER_URL
    else process.env.TLDW_E2E_SERVER_URL = previousServerUrl
    if (previousApiKey === undefined) delete process.env.TLDW_E2E_API_KEY
    else process.env.TLDW_E2E_API_KEY = previousApiKey
  }
})

test.describe("Packaged extension prompt improvement parity", () => {
  test.describe.configure({ mode: "serial" })

  test.beforeAll(async () => {
    test.setTimeout(90_000)
    await probePackagedRuntime()
  })

  test("sidepanel improves a selected system template and restores the exact draft with Undo", async () => {
    test.setTimeout(120_000)
    const mock = await startPromptMockServer()
    const originalSystemDraft = `${SYSTEM_COUNTERPART} Keep {{topic}} literal.`
    let context: BrowserContext | null = null
    try {
      const launched = await launchChatSurface(mock, "sidepanel", {
        mode: "pro",
        viewport: { width: 1000, height: 850 },
        seedTemplate: true
      })
      context = launched.context
      const page = launched.chatPage
      const composer = await ensureChatInput(page)
      await composer.fill(USER_COUNTERPART)
      const promptTrigger = await selectTemplate(page)
      const editor = await openSystemEditor(page)
      await expect(editor).toHaveValue(originalSystemDraft)

      const { actions } = await openPromptActions(
        page,
        page.getByRole("dialog", { name: "Edit system prompt" })
      )
      const improveNow = actions.getByRole("button", { name: /Improve now/ })
      await expect(improveNow).toBeEnabled()
      await improveNow.click()
      await expect(editor).toHaveValue(
        "Improved system instruction for {{topic}}."
      )
      const undo = page.getByRole("button", { name: "Undo improvement" })
      await expect(undo).toBeVisible()
      await undo.click()
      await expect(editor).toHaveValue(originalSystemDraft)
      await page.getByRole("button", { name: "Cancel", exact: true }).click()
      await expect(promptTrigger).toContainText(TEMPLATE_TITLE)

      await expect.poll(() => mock.improveRequests().length).toBe(1)
      assertPromptRequest(mock.improveRequests()[0], {
        target: "system",
        text: originalSystemDraft
      })
    } finally {
      await context?.close()
      await stopPromptMockServer(mock)
    }
  })

  test("narrow options chat reviews, edits, applies, restores focus, and exposes an accessible full-width sheet", async ({}, testInfo) => {
    test.setTimeout(120_000)
    const mock = await startPromptMockServer()
    const originalDraft = "Clarify {{topic}} for a new reader."
    let context: BrowserContext | null = null
    try {
      const launched = await launchChatSurface(mock, "options", {
        viewport: { width: 390, height: 780 },
        seedTemplate: true
      })
      context = launched.context
      const page = launched.chatPage
      await seedExcludedContext(page)
      const storeTemplateSelected = await page.evaluate((templateId) => {
        const store = (window as any).__tldw_useStoreMessageOption
        store.setState({ selectedSystemPrompt: templateId })
        return store.getState().selectedSystemPrompt
      }, TEMPLATE_ID)
      expect(storeTemplateSelected).toBe(TEMPLATE_ID)
      const input = await ensureChatInput(page)
      await input.fill(originalDraft)

      const firstMenu = await openPromptActions(page)
      await firstMenu.trigger.press("Escape")
      await expect(firstMenu.actions).not.toBeVisible()
      await expect(firstMenu.trigger).toBeFocused()

      const secondMenu = await openPromptActions(page)
      await secondMenu.actions
        .getByRole("button", { name: /Review changes/ })
        .click()
      const dialog = page.getByRole("dialog", { name: "Prompt improvement" })
      await expect(dialog).toBeVisible()
      const candidate = page.getByRole("textbox", {
        name: "Improved prompt candidate"
      })
      await expect(candidate).toHaveValue(
        "Improved user request for {{topic}}."
      )

      const readLayout = () =>
        dialog.evaluate((element) => {
          const drawer =
            element.closest(".ant-drawer-content-wrapper") ?? element
          const rect = drawer.getBoundingClientRect()
          return {
            left: Math.round(rect.left),
            right: Math.round(rect.right),
            width: Math.round(rect.width),
            viewport: window.innerWidth,
            documentWidth: document.documentElement.scrollWidth
          }
        })
      await expect.poll(async () => (await readLayout()).left).toBe(0)
      const layout = await readLayout()
      expect(layout.left).toBe(0)
      expect(layout.right).toBeLessThanOrEqual(layout.viewport)
      expect(layout.width).toBe(layout.viewport)
      expect(layout.documentWidth).toBeLessThanOrEqual(layout.viewport)

      await page.evaluate(AXE_SOURCE)
      const axeResults = await dialog.evaluate(async (root) => {
        const axe = (window as any).axe as {
          run: (
            context: Element,
            options: Record<string, unknown>
          ) => Promise<{ violations: unknown[] }>
        }
        if (!axe?.run)
          throw new Error("axe-core did not load in the extension page")
        return axe.run(root, {
          resultTypes: ["violations"],
          runOnly: {
            type: "tag",
            values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]
          }
        })
      })
      expect(
        axeResults.violations,
        JSON.stringify(axeResults.violations, null, 2)
      ).toEqual([])
      fs.writeFileSync(
        testInfo.outputPath("prompt-improvement-review-axe.json"),
        JSON.stringify(axeResults, null, 2),
        "utf8"
      )
      const snapshot = await dialog.ariaSnapshot()
      expect(snapshot).toContain("Review improved prompt")
      expect(snapshot).toContain("Improved prompt candidate")
      fs.writeFileSync(
        testInfo.outputPath("prompt-improvement-review-a11y.aria.yml"),
        snapshot,
        "utf8"
      )
      await page.screenshot({
        path: testInfo.outputPath("prompt-improvement-review-narrow.png"),
        fullPage: true
      })

      await candidate.press("Escape")
      await expect(dialog).not.toBeVisible()
      await expect(input).toBeFocused()

      const thirdMenu = await openPromptActions(page)
      await thirdMenu.actions
        .getByRole("button", { name: /Review changes/ })
        .click()
      const editableCandidate = page.getByRole("textbox", {
        name: "Improved prompt candidate"
      })
      await editableCandidate.fill("Edited candidate for {{topic}}.")
      await page.getByRole("button", { name: "Apply to draft" }).click()
      await expect(input).toHaveValue("Edited candidate for {{topic}}.")
      await expect(input).toBeFocused()
      expect(
        await page.evaluate(
          () =>
            (window as any).__tldw_useStoreMessageOption.getState()
              .selectedSystemPrompt
        )
      ).toBe(TEMPLATE_ID)

      await expect.poll(() => mock.improveRequests().length).toBe(2)
      for (const request of mock.improveRequests()) {
        assertPromptRequest(request, {
          target: "user_message",
          text: originalDraft
        })
      }
    } finally {
      await context?.close()
      await stopPromptMockServer(mock)
    }
  })

  test("sidepanel never overwrites typing committed while Improve now is pending", async () => {
    test.setTimeout(120_000)
    const mock = await startPromptMockServer()
    const requestedDraft = "[DEFER_STALE] Improve this draft."
    const liveDraft = "Typing committed while the provider is pending."
    let context: BrowserContext | null = null
    try {
      const launched = await launchChatSurface(mock, "sidepanel")
      context = launched.context
      const page = launched.chatPage
      const input = await ensureChatInput(page)
      await input.fill(requestedDraft)
      const { actions } = await openPromptActions(page)
      await actions.getByRole("button", { name: /Improve now/ }).click()
      await expect.poll(() => mock.improveRequests().length).toBe(1)
      await input.fill(liveDraft)
      mock.releaseNextDeferred()

      await expect(input).toHaveValue(liveDraft)
      await expect(
        page.getByText(/draft changed while this result was open/i)
      ).toBeVisible()
      await expect(
        page.getByRole("textbox", { name: "Improved prompt candidate" })
      ).toHaveValue("Deferred stale candidate.")
      assertPromptRequest(mock.improveRequests()[0], {
        target: "user_message",
        text: requestedDraft
      })
    } finally {
      await context?.close()
      await stopPromptMockServer(mock)
    }
  })

  test("sidepanel requires explicit confirmation before replacing a changed live draft", async () => {
    test.setTimeout(120_000)
    const mock = await startPromptMockServer()
    const requestedDraft = "[DEFER_CONFIRM] Review this draft."
    const liveDraft = "Newer live draft must survive normal Apply."
    let context: BrowserContext | null = null
    try {
      const launched = await launchChatSurface(mock, "sidepanel")
      context = launched.context
      const page = launched.chatPage
      const input = await ensureChatInput(page)
      await input.fill(requestedDraft)
      const { actions } = await openPromptActions(page)
      await actions.getByRole("button", { name: /Review changes/ }).click()
      await expect.poll(() => mock.improveRequests().length).toBe(1)
      await input.fill(liveDraft)
      mock.releaseNextDeferred()

      await expect(input).toHaveValue(liveDraft)
      const replace = page.getByRole("button", {
        name: "Replace current draft"
      })
      await expect(replace).toBeVisible()
      await replace.click({ timeout: 20_000 })
      const confirmReplace = page.getByRole("button", {
        name: "Confirm replace"
      })
      await expect(confirmReplace).toBeVisible({ timeout: 20_000 })
      await confirmReplace.click({ timeout: 20_000 })
      await expect(input).toHaveValue("Deferred confirmed replacement.")
      assertPromptRequest(mock.improveRequests()[0], {
        target: "user_message",
        text: requestedDraft
      })
    } finally {
      await context?.close()
      await stopPromptMockServer(mock)
    }
  })

  test("missing model offers recovery without sending an improvement request", async () => {
    test.setTimeout(120_000)
    const mock = await startPromptMockServer()
    let context: BrowserContext | null = null
    try {
      const launched = await launchChatSurface(mock, "sidepanel", {
        withModel: false
      })
      context = launched.context
      const page = launched.chatPage
      const input = await ensureChatInput(page)
      await input.fill("Draft without a selected route.")
      const { actions } = await openPromptActions(page)
      await expect(actions).toContainText(
        "Select a chat model to improve this draft."
      )
      await expect(
        actions.getByRole("button", { name: /Improve now/ })
      ).toBeDisabled()
      const selectModel = actions.getByRole("button", { name: "Select model" })
      await expect(selectModel).toBeVisible()
      await selectModel.click()
      await expect(
        page.getByRole("dialog", {
          name: /Current Chat Model Settings|currentChatModelSettings/i
        })
      ).toBeVisible({ timeout: 20_000 })
      expect(mock.improveRequests()).toHaveLength(0)
    } finally {
      await context?.close()
      await stopPromptMockServer(mock)
    }
  })

  test("structured provider failure preserves the draft and Retry succeeds with a new operation", async () => {
    test.setTimeout(120_000)
    const mock = await startPromptMockServer("supported", {
      failFirstImprovement: true
    })
    const draft = "Keep this {{topic}} draft intact."
    let context: BrowserContext | null = null
    try {
      const launched = await launchChatSurface(mock, "sidepanel")
      context = launched.context
      const page = launched.chatPage
      const input = await ensureChatInput(page)
      await input.fill(draft)
      const { actions } = await openPromptActions(page)
      await actions.getByRole("button", { name: /Improve now/ }).click()

      await expect(
        page.getByRole("alert").filter({
          hasText: "prompt improvement service is unavailable"
        })
      ).toBeVisible()
      const retry = page.getByRole("button", { name: "Retry" })
      await expect(retry).toBeVisible()
      await expect(input).toHaveValue(draft)
      await expect.poll(() => mock.improveRequests().length).toBe(1)
      const firstRequest = mock.improveRequests()[0]
      assertPromptRequest(firstRequest, {
        target: "user_message",
        text: draft
      })

      await retry.click()
      await expect.poll(() => mock.improveRequests().length).toBe(2)
      const secondRequest = mock.improveRequests()[1]
      const firstOperationId = firstRequest.json?.operation_id
      const secondOperationId = secondRequest.json?.operation_id
      expect(secondOperationId).not.toBe(firstOperationId)
      assertPromptRequest(secondRequest, {
        target: "user_message",
        text: draft
      })
      await expect(input).toHaveValue("Improved user request for {{topic}}.")
      const appliedStatus = page
        .getByTestId("chat-messages")
        .getByRole("status")
        .filter({ hasText: "Improvement applied." })
      await expect(appliedStatus).toBeVisible()
      await expect(
        appliedStatus
          .locator("..")
          .getByRole("button", { name: "Undo improvement" })
      ).toBeVisible()
    } finally {
      await context?.close()
      await stopPromptMockServer(mock)
    }
  })

  for (const capabilityMode of ["false", "404", "offline"] as const) {
    test(`capability ${capabilityMode} fails closed and exposes no recipe action`, async () => {
      test.setTimeout(120_000)
      const mock = await startPromptMockServer(capabilityMode)
      let context: BrowserContext | null = null
      try {
        const launched = await launchChatSurface(mock, "sidepanel")
        context = launched.context
        const page = launched.chatPage
        const input = await ensureChatInput(page)
        await input.fill(`Capability ${capabilityMode} draft.`)
        const { actions } = await openPromptActions(page)
        await expect(actions).toContainText(
          "Prompt improvement requires a newer server version."
        )
        await expect(
          actions.getByRole("button", { name: /Improve now/ })
        ).toBeDisabled()
        await expect(
          actions.getByRole("button", { name: /Review changes/ })
        ).toBeDisabled()
        await expect(actions).not.toContainText(/Build from recipe/i)
        expect(mock.improveRequests()).toHaveLength(0)
      } finally {
        await context?.close()
        await stopPromptMockServer(mock)
      }
    })
  }

  test("casual and pro legacy, V1, V3, and V5 composers render exactly one actionable entry", async () => {
    test.setTimeout(180_000)
    const mock = await startPromptMockServer()
    let context: BrowserContext | null = null
    try {
      const launched = await launchChatSurface(mock, "sidepanel")
      context = launched.context
      const page = launched.chatPage
      const combinations = [
        { mode: "casual", variant: "legacy", enabled: false },
        { mode: "pro", variant: "legacy", enabled: false },
        { mode: "casual", variant: "v1", enabled: true },
        { mode: "pro", variant: "v1", enabled: true },
        { mode: "casual", variant: "v3", enabled: true },
        { mode: "pro", variant: "v3", enabled: true },
        { mode: "casual", variant: "v5", enabled: true },
        { mode: "pro", variant: "v5", enabled: true }
      ] as const

      for (const combination of combinations) {
        await page.evaluate((settings) => {
          localStorage.setItem(
            "tldw-ui-mode",
            JSON.stringify({ state: { mode: settings.mode }, version: 0 })
          )
          localStorage.setItem(
            "tldw:nextgenComposerEnabled",
            settings.enabled ? "1" : "0"
          )
          if (settings.variant !== "legacy") {
            localStorage.setItem("tldw:composerVariant", settings.variant)
          }
        }, combination)
        await page.reload({ waitUntil: "domcontentloaded" })
        await waitForConnectionStore(
          page,
          `prompt-improvement:${combination.mode}:${combination.variant}`
        )
        await forceConnected(page, { serverUrl: mock.baseUrl })
        await ensureChatInput(page)
        await seedExcludedContext(page)
        const actions = page.getByRole("button", { name: "Improve prompt" })
        await expect(actions).toHaveCount(1)
        await expect(actions).toBeVisible()
        await actions.click()
        await expect(
          page.getByRole("group", { name: "Prompt improvement actions" })
        ).toBeVisible()
        await actions.press("Escape")
        await expect(actions).toBeFocused()
      }
      expect(mock.improveRequests()).toHaveLength(0)
    } finally {
      await context?.close()
      await stopPromptMockServer(mock)
    }
  })

  test("configured real local server smoke matches the advertised capability state", async () => {
    test.setTimeout(120_000)
    const { serverUrl, apiKey } = requirePromptImprovementRealServerConfig()
    const realServer = buildRealServerHarnessConfig(serverUrl, apiKey)
    const capabilityResponse = await fetch(realServer.capabilityUrl, {
      headers: realServer.capabilityHeaders
    })
    expect(capabilityResponse.ok).toBe(true)
    const capability = await capabilityResponse.json()
    const supported = capability.prompt_improvement_v1?.supported
    expect(typeof supported).toBe("boolean")

    const { context, page, openSidepanel, extensionId } =
      await launchWithExtension(EXT_PATH, {
        seedConfig: realServer.seedConfig
      })
    try {
      const granted = await grantHostPermission(
        context,
        extensionId,
        `${new URL(serverUrl).origin}/*`
      )
      expect(granted).toBe(true)
      await setSelectedModel(page, MODEL_KEY)
      const sidepanel = await openSidepanel("/chat")
      await waitForConnectionStore(sidepanel, "prompt-improvement:real-local")
      const input = await ensureChatInput(sidepanel)
      await input.fill("Real local capability gate smoke.")
      const { actions } = await openPromptActions(sidepanel)
      const improveNow = actions.getByRole("button", { name: /Improve now/ })
      const reviewChanges = actions.getByRole("button", {
        name: /Review changes/
      })
      if (supported) {
        await expect(improveNow).toBeEnabled()
        await expect(reviewChanges).toBeEnabled()
      } else {
        await expect(actions).toContainText(
          "Prompt improvement requires a newer server version."
        )
        await expect(improveNow).toBeDisabled()
        await expect(reviewChanges).toBeDisabled()
      }
      await expect(actions).not.toContainText(/Build from recipe/i)
    } finally {
      await context.close()
    }
  })
})
