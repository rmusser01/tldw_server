import AxeBuilder from "@axe-core/playwright"
import {
  type BrowserContext,
  type Page,
  type Route,
  expect,
  test
} from "@playwright/test"
import { readFileSync } from "node:fs"
import http from "node:http"
import type { AddressInfo } from "node:net"
import path from "node:path"

import { launchWithExtension } from "./utils/extension"
import { grantHostPermission } from "./utils/permissions"

const EXTENSION_PATH = path.resolve(".output/chrome-mv3")
const PROMPT_ID = "chat.rag.answer"
const FAKE_API_KEY = "THIS-IS-A-SECURE-KEY-123-FAKE-KEY"
const CORRUPT_REVISION = "11111111-1111-4111-8111-111111111111"
const TEST_MODEL = "openai:service-prompts-e2e-model"

type ServicePromptDetail = {
  default_parts: Record<string, string>
  effective_parts: Record<string, string>
  id: string
  revision: string | null
  saved_parts: Record<string, string> | null
  source: "packaged" | "user"
}

type GateConfig = {
  apiKey: string
  serverUrl: string
  webUrl: string
}

type RecordedRequest = {
  method: string
  path: string
  search: string
}

type TestHarnessWindow = Window & {
  __tldw_useStoreMessageOption?: {
    getState?: () => { chatMode?: string }
    setState?: (state: Record<string, unknown>) => void
  }
  __tldw_useStoreChatModelSettings?: {
    setState?: (state: Record<string, unknown>) => void
  }
}

const fixture = JSON.parse(
  readFileSync(
    path.resolve(
      "../packages/ui/src/utils/__fixtures__/service-prompt-rendering.json"
    ),
    "utf8"
  )
) as { defaults: Record<string, Record<string, string>> }
const packagedTemplate = fixture.defaults[PROMPT_ID]?.template
if (!packagedTemplate) {
  throw new Error(`Missing packaged ${PROMPT_ID} E2E fixture.`)
}

const normalizeBaseUrl = (value: string): string => value.replace(/\/+$/, "")

const assertLoopbackUrl = (rawUrl: string, label: string): string => {
  let parsed: URL
  try {
    parsed = new URL(rawUrl)
  } catch {
    throw new Error(`${label} must be an absolute HTTP URL.`)
  }
  if (
    !["127.0.0.1", "localhost", "::1"].includes(parsed.hostname) ||
    !["http:", "https:"].includes(parsed.protocol)
  ) {
    throw new Error(`${label} must target a disposable loopback server.`)
  }
  return normalizeBaseUrl(parsed.toString())
}

const assertHealthy = async (
  url: string,
  label: string,
  headers?: Record<string, string>
) => {
  const response = await fetch(url, { headers }).catch((error) => {
    throw new Error(`${label} is unavailable at ${url}: ${String(error)}`)
  })
  if (!response.ok) {
    const body = await response.text().catch(() => "")
    throw new Error(
      `${label} failed at ${url} (HTTP ${response.status}): ${body.slice(0, 500)}`
    )
  }
}

const requireGateConfig = async (): Promise<GateConfig> => {
  const serverUrl = assertLoopbackUrl(
    String(process.env.TLDW_E2E_SERVER_URL || "").trim(),
    "TLDW_E2E_SERVER_URL"
  )
  const webUrl = assertLoopbackUrl(
    String(process.env.TLDW_WEB_URL || "").trim(),
    "TLDW_WEB_URL"
  )
  const apiKey = String(process.env.TLDW_E2E_API_KEY || "").trim()
  if (apiKey !== FAKE_API_KEY) {
    throw new Error(
      "This mutating release gate requires the documented disposable fake API key."
    )
  }
  await assertHealthy(`${serverUrl}/api/v1/health`, "Disposable tldw server", {
    "X-API-KEY": apiKey
  })
  await assertHealthy(webUrl, "WebUI")
  return { apiKey, serverUrl, webUrl }
}

const servicePromptHeaders = (apiKey: string) => ({
  "Content-Type": "application/json",
  "X-API-KEY": apiKey
})

const corsHeaders = {
  "Access-Control-Allow-Origin": "*",
  "Access-Control-Allow-Headers": "*",
  "Access-Control-Allow-Methods": "GET,POST,OPTIONS"
}

const fulfillRouteJson = async (route: Route, body: unknown) => {
  if (route.request().method() === "OPTIONS") {
    await route.fulfill({ status: 204, headers: corsHeaders })
    return
  }
  await route.fulfill({
    status: 200,
    contentType: "application/json",
    headers: corsHeaders,
    body: JSON.stringify(body)
  })
}

const readJsonResponse = async <T>(response: Response): Promise<T> => {
  const body = await response.text()
  try {
    return JSON.parse(body) as T
  } catch {
    throw new Error(
      `${response.url} returned non-JSON (HTTP ${response.status}): ${body.slice(0, 500)}`
    )
  }
}

const getLiveDetail = async (
  target: GateConfig
): Promise<ServicePromptDetail> => {
  const response = await fetch(
    `${target.serverUrl}/api/v1/service-prompts/${PROMPT_ID}`,
    { headers: servicePromptHeaders(target.apiKey) }
  )
  const body = await readJsonResponse<unknown>(response)
  if (!response.ok) {
    throw new Error(
      `GET ${PROMPT_ID} failed (HTTP ${response.status}): ${JSON.stringify(body)}`
    )
  }
  return body as ServicePromptDetail
}

const conditionalCleanup = async (target: GateConfig) => {
  const detailResponse = await fetch(
    `${target.serverUrl}/api/v1/service-prompts/${PROMPT_ID}`,
    { headers: servicePromptHeaders(target.apiKey) }
  )
  const detail = await readJsonResponse<
    ServicePromptDetail | { detail?: { revision?: unknown } }
  >(detailResponse)
  if (!detailResponse.ok) {
    const errorDetail = (
      detail as {
        detail?: { code?: unknown; revision?: unknown }
      }
    ).detail
    if (
      errorDetail?.code !== "service_prompt_corrupt_override" ||
      typeof errorDetail.revision !== "string" ||
      !errorDetail.revision
    ) {
      throw new Error(
        `Conditional ${PROMPT_ID} cleanup detail read failed (HTTP ${detailResponse.status}).`
      )
    }
  }
  const revision = detailResponse.ok
    ? (detail as ServicePromptDetail).revision
    : (detail as { detail: { revision: string } }).detail.revision
  if (revision === null) return
  if (typeof revision !== "string" || !revision) {
    throw new Error(
      `Conditional ${PROMPT_ID} cleanup returned an invalid revision.`
    )
  }

  const resetUrl = new URL(
    `${target.serverUrl}/api/v1/service-prompts/${PROMPT_ID}`
  )
  resetUrl.searchParams.set("expected_revision", revision)
  const resetResponse = await fetch(resetUrl, {
    method: "DELETE",
    headers: servicePromptHeaders(target.apiKey)
  })
  if (!resetResponse.ok) {
    const body = await resetResponse.text().catch(() => "")
    throw new Error(
      `Conditional ${PROMPT_ID} cleanup failed (HTTP ${resetResponse.status}): ${body.slice(0, 500)}`
    )
  }
}

const cleanupTargets = new Map<string, GateConfig>()

test.afterEach(async ({ browserName: _browserName }, testInfo) => {
  const target = cleanupTargets.get(testInfo.testId)
  cleanupTargets.delete(testInfo.testId)
  if (target) await conditionalCleanup(target)
})

const extensionConfig = (serverUrl: string, apiKey: string) => ({
  __tldw_first_run_complete: true,
  assistant_setup_dismissed: true,
  tldw_skip_landing_hub: true,
  tldwConfig: {
    apiKey,
    authMode: "single-user" as const,
    serverUrl
  }
})

const seedWebUi = async (page: Page, target: GateConfig) => {
  await page.addInitScript(
    ({ apiKey, model, serverUrl }) => {
      const config = {
        apiKey,
        authMode: "single-user",
        serverUrl
      }
      localStorage.setItem("tldwConfig", JSON.stringify(config))
      localStorage.setItem("serverUrl", serverUrl)
      localStorage.setItem("tldwServerUrl", serverUrl)
      localStorage.setItem("tldw-api-host", serverUrl)
      localStorage.setItem("authMode", "single-user")
      localStorage.setItem("apiKey", apiKey)
      localStorage.setItem("selectedModel", JSON.stringify(model))
      localStorage.setItem("isMigrated", "true")
      localStorage.setItem("__tldw_first_run_complete", "true")
      localStorage.setItem("assistant_setup_dismissed", "true")
      localStorage.setItem("__tldw_test_bypass", "true")
      localStorage.setItem("playgroundChatContextRailVisible", "true")
      localStorage.setItem("playgroundChatRuntimeRailVisible", "false")
      localStorage.setItem(
        "plasmo-sync:playgroundChatContextRailVisible",
        "true"
      )
      localStorage.setItem(
        "plasmo-sync:playgroundChatRuntimeRailVisible",
        "false"
      )
    },
    { apiKey: target.apiKey, model: TEST_MODEL, serverUrl: target.serverUrl }
  )
}

const openPromptEditor = async (page: Page, url: string) => {
  await page.goto(url, { waitUntil: "domcontentloaded" })
  await expect(
    page.getByRole("region", { name: "RAG answer editor" })
  ).toBeVisible({ timeout: 30_000 })
  await expect(page.getByRole("textbox", { name: "Template" })).toBeVisible()
}

const writeExtensionConnection = async (
  page: Page,
  config: Record<string, unknown>
) => {
  await page.evaluate(async (nextConfig) => {
    const write = (area: typeof chrome.storage.local) =>
      new Promise<void>((resolve, reject) => {
        area.set({ tldwConfig: nextConfig }, () => {
          const error = chrome.runtime.lastError
          if (error) reject(new Error(error.message))
          else resolve()
        })
      })
    await Promise.all([write(chrome.storage.local), write(chrome.storage.sync)])
    window.dispatchEvent(new CustomEvent("tldw:config-updated"))
  }, config)
}

const grantOrigin = async (
  context: BrowserContext,
  extensionId: string,
  baseUrl: string
) => {
  const granted = await grantHostPermission(
    context,
    extensionId,
    `${new URL(baseUrl).origin}/*`
  )
  expect(granted, `Host permission for ${baseUrl}`).toBe(true)
}

const closeServer = async (server: http.Server) => {
  await new Promise<void>((resolve) => {
    let finished = false
    const done = () => {
      if (finished) return
      finished = true
      resolve()
    }
    server.close(done)
    server.closeAllConnections?.()
    const fallback = setTimeout(done, 1_000)
    fallback.unref?.()
  })
}

const listen = async (server: http.Server): Promise<string> => {
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", () => {
      server.off("error", reject)
      resolve()
    })
  })
  const address = server.address() as AddressInfo
  return `http://127.0.0.1:${address.port}`
}

const sendJson = (
  request: http.IncomingMessage,
  response: http.ServerResponse,
  status: number,
  body: unknown
) => {
  const origin = request.headers.origin || "*"
  response.writeHead(status, {
    "access-control-allow-credentials": "true",
    "access-control-allow-origin": origin,
    "content-type": "application/json"
  })
  response.end(JSON.stringify(body))
}

const startUnresolvedScopeServer = async () => {
  const server = http.createServer((request, response) => {
    const url = new URL(request.url || "/", "http://127.0.0.1")
    if (request.method === "OPTIONS") {
      response.writeHead(204, {
        "access-control-allow-headers":
          "authorization, content-type, x-api-key",
        "access-control-allow-methods": "GET, PUT, DELETE, OPTIONS",
        "access-control-allow-origin": request.headers.origin || "*"
      })
      return response.end()
    }
    if (
      url.pathname === "/api/v1/health" ||
      url.pathname === "/api/v1/health/live"
    ) {
      return sendJson(request, response, 200, { status: "ok" })
    }
    return sendJson(request, response, 401, {
      detail: "Authenticated test scope is intentionally unresolved."
    })
  })
  return { baseUrl: await listen(server), server }
}

const promptDefinition = {
  affected_workflows: [{ id: "chat.main.rag", label: "Main chat RAG" }],
  description:
    "Controls how retrieved context and the current question are presented to the model.",
  id: PROMPT_ID,
  label: "RAG answer",
  parts: [
    {
      key: "template",
      label: "Template",
      mode: "template",
      required_variables: ["context", "question"]
    }
  ]
}

const packagedDetail = (): ServicePromptDetail & typeof promptDefinition => ({
  ...promptDefinition,
  default_parts: { template: packagedTemplate },
  effective_parts: { template: packagedTemplate },
  revision: null,
  saved_parts: null,
  source: "packaged"
})

const startCorruptServer = async () => {
  let corrupt = true
  const requests: RecordedRequest[] = []
  const server = http.createServer(async (request, response) => {
    const method = (request.method || "GET").toUpperCase()
    const url = new URL(request.url || "/", "http://127.0.0.1")
    if (method === "OPTIONS") {
      response.writeHead(204, {
        "access-control-allow-headers":
          "authorization, content-type, x-api-key",
        "access-control-allow-methods": "GET, PUT, DELETE, OPTIONS",
        "access-control-allow-origin": request.headers.origin || "*"
      })
      return response.end()
    }
    const recordedRequest: RecordedRequest = {
      method,
      path: url.pathname,
      search: url.search
    }
    requests.push(recordedRequest)

    if (
      ["/api/v1/health", "/api/v1/health/live"].includes(url.pathname) &&
      method === "GET"
    ) {
      return sendJson(request, response, 200, { status: "ok" })
    }
    if (url.pathname === "/api/v1/service-prompts" && method === "GET") {
      return sendJson(request, response, 200, [promptDefinition])
    }
    if (
      url.pathname === `/api/v1/service-prompts/${PROMPT_ID}` &&
      method === "GET"
    ) {
      if (corrupt) {
        return sendJson(request, response, 500, {
          detail: {
            can_reset: true,
            code: "service_prompt_corrupt_override",
            message: "The saved workflow prompt cannot be read safely.",
            revision: CORRUPT_REVISION
          }
        })
      }
      return sendJson(request, response, 200, packagedDetail())
    }
    if (
      url.pathname === `/api/v1/service-prompts/${PROMPT_ID}` &&
      method === "DELETE"
    ) {
      if (url.searchParams.get("expected_revision") !== CORRUPT_REVISION) {
        return sendJson(request, response, 409, {
          detail: {
            code: "service_prompt_revision_conflict",
            current_revision: CORRUPT_REVISION,
            message: "Revision mismatch."
          }
        })
      }
      corrupt = false
      return sendJson(request, response, 200, packagedDetail())
    }
    return sendJson(request, response, 404, { detail: "not found" })
  })
  return { baseUrl: await listen(server), requests, server }
}

const assertNoSeriousAxeViolations = async (page: Page) => {
  const results = await new AxeBuilder({ page }).analyze()
  type AxeViolation = (typeof results.violations)[number]
  const violations = results.violations.filter(
    (violation: AxeViolation) =>
      violation.impact === "critical" || violation.impact === "serious"
  )
  expect(
    violations,
    violations
      .map(
        (violation: AxeViolation) =>
          `${violation.id}: ${violation.help} (${violation.nodes.length} nodes)`
      )
      .join("\n")
  ).toEqual([])
}

const contentText = (content: unknown): string => {
  if (typeof content === "string") return content
  if (!Array.isArray(content)) return ""
  return content
    .map((part) =>
      part && typeof part === "object" && "text" in part
        ? String((part as { text?: unknown }).text || "")
        : ""
    )
    .join("")
}

test.describe("Workflow prompts cross-host release gate", () => {
  test("shares a live saved prompt across WebUI and extension, isolates scope, and resets Main RAG", async ({
    browserName: _browserName
  }, testInfo) => {
    test.setTimeout(180_000)
    const target = await requireGateConfig()
    cleanupTargets.set(testInfo.testId, target)
    await conditionalCleanup(target)
    const initialDetail = await getLiveDetail(target)
    expect(initialDetail.source).toBe("packaged")

    const marker = `E2E_DELETED_RAG_PROMPT_${Date.now()}`
    const customTemplate = `${marker}\nContext: {context}\nQuestion: {question}`
    const launch = await launchWithExtension(EXTENSION_PATH, {
      seedConfig: extensionConfig(target.serverUrl, target.apiKey)
    })
    const { context, extensionId, optionsUrl, page: extensionPage } = launch
    const unresolved = await startUnresolvedScopeServer()

    try {
      await grantOrigin(context, extensionId, target.serverUrl)
      await grantOrigin(context, extensionId, unresolved.baseUrl)

      const webPage = await context.newPage()
      await seedWebUi(webPage, target)
      await openPromptEditor(
        webPage,
        `${target.webUrl}/settings/prompt?prompt=${PROMPT_ID}`
      )
      const webTemplate = webPage.getByRole("textbox", { name: "Template" })
      await webTemplate.fill(customTemplate)
      await webPage.getByRole("button", { name: "Save changes" }).click()
      await expect(
        webPage.getByText("Customized", { exact: true })
      ).toBeVisible()

      await expect
        .poll(() => getLiveDetail(target), { timeout: 15_000 })
        .toMatchObject({
          effective_parts: { template: customTemplate },
          source: "user"
        })
      await openPromptEditor(
        extensionPage,
        `${optionsUrl}#/settings/prompt?prompt=${PROMPT_ID}`
      )
      await expect(
        extensionPage.getByRole("textbox", { name: "Template" })
      ).toHaveValue(customTemplate)

      await writeExtensionConnection(extensionPage, {
        accessToken: "unresolved-e2e-token",
        authMode: "multi-user",
        serverUrl: unresolved.baseUrl
      })
      await extensionPage.goto(
        `${optionsUrl}#/settings/prompt?prompt=${PROMPT_ID}`,
        { waitUntil: "domcontentloaded" }
      )
      await expect(
        extensionPage.getByText("Server or account changed", { exact: true })
      ).toBeVisible({ timeout: 30_000 })
      await expect(
        extensionPage.getByText("Unable to load workflow prompts", {
          exact: true
        })
      ).toBeVisible({ timeout: 30_000 })
      await expect(extensionPage.locator("body")).not.toContainText(marker)
      await expect(
        extensionPage.getByRole("button", { name: "Save changes" })
      ).toHaveCount(0)

      await writeExtensionConnection(extensionPage, {
        apiKey: target.apiKey,
        authMode: "single-user",
        serverUrl: target.serverUrl
      })
      await openPromptEditor(
        extensionPage,
        `${optionsUrl}#/settings/prompt?prompt=${PROMPT_ID}`
      )
      await expect(
        extensionPage.getByRole("textbox", { name: "Template" })
      ).toHaveValue(customTemplate)

      await extensionPage
        .getByRole("button", { name: "Reset to default" })
        .click()
      const resetDialog = extensionPage
        .getByRole("dialog")
        .filter({ hasText: "Reset RAG answer?" })
      await expect(resetDialog).toBeVisible()
      await resetDialog
        .getByRole("button", { name: "Reset", exact: true })
        .click()
      await expect(
        extensionPage.getByText("Server default", { exact: true })
      ).toBeVisible()
      await expect(
        extensionPage.getByRole("textbox", { name: "Template" })
      ).toHaveValue(initialDetail.default_parts.template)

      await webPage.reload({ waitUntil: "domcontentloaded" })
      await expect(
        webPage.getByText("Server default", { exact: true })
      ).toBeVisible()
      await expect(webTemplate).toHaveValue(
        initialDetail.default_parts.template
      )
      await webPage.close()

      const chatPage = await context.newPage()
      await seedWebUi(chatPage, target)

      const ragContent =
        "A deterministic source says packaged prompts survive reset."
      const question = "What does the deterministic source say?"
      const ragRequests: Record<string, unknown>[] = []
      const providerRequests: Record<string, unknown>[] = []
      await chatPage.route("**/api/v1/llm/models/metadata**", (route) =>
        fulfillRouteJson(route, {
          models: [
            {
              id: TEST_MODEL,
              model: TEST_MODEL,
              name: "Service Prompt E2E model",
              provider: "openai",
              apiProvider: "custom-openai-api",
              capabilities: ["chat"],
              configured: true,
              available: true
            }
          ]
        })
      )
      await chatPage.route("**/api/v1/llm/providers**", (route) =>
        fulfillRouteJson(route, {
          providers: [
            {
              id: "openai",
              name: "OpenAI-compatible E2E",
              apiProvider: "custom-openai-api",
              models: [TEST_MODEL],
              configured: true,
              available: true
            }
          ]
        })
      )
      await chatPage.route("**/api/v1/rag/search", async (route) => {
        if (route.request().method() === "OPTIONS") {
          await fulfillRouteJson(route, {})
          return
        }
        ragRequests.push(
          (route.request().postDataJSON() as Record<string, unknown>) || {}
        )
        await fulfillRouteJson(route, {
          results: [
            {
              content: ragContent,
              metadata: {
                source: "service-prompts-e2e",
                title: "E2E source"
              },
              score: 1
            }
          ]
        })
      })
      await chatPage.route("**/api/v1/chat/completions", async (route) => {
        if (route.request().method() === "OPTIONS") {
          await route.fulfill({ status: 204, headers: corsHeaders })
          return
        }
        providerRequests.push(
          (route.request().postDataJSON() as Record<string, unknown>) || {}
        )
        await route.fulfill({
          status: 200,
          contentType: "text/event-stream",
          headers: {
            ...corsHeaders,
            "Cache-Control": "no-cache",
            Connection: "keep-alive"
          },
          body:
            `data: ${JSON.stringify({ choices: [{ delta: { content: "Packaged RAG reply" } }] })}\n\n` +
            "data: [DONE]\n\n"
        })
      })

      await chatPage.goto(`${target.webUrl}/chat`, {
        waitUntil: "domcontentloaded"
      })
      const chatInput = chatPage.getByTestId("chat-input").first()
      await expect(chatInput).toBeVisible({ timeout: 30_000 })
      await chatPage.waitForFunction(
        () =>
          Boolean(
            (window as TestHarnessWindow).__tldw_useStoreMessageOption
              ?.setState
          ),
        undefined,
        { timeout: 30_000 }
      )
      await chatPage.evaluate((model) => {
        const store = (window as TestHarnessWindow)
          .__tldw_useStoreMessageOption
        store?.setState?.({
          selectedModel: model,
          temporaryChat: true,
          historyId: "temp",
          serverChatId: null,
          messages: [],
          history: [],
          streaming: false,
          isProcessing: false,
          chatMode: "rag",
          webSearch: false,
          contextFiles: [],
          documentContext: null,
          fileRetrievalEnabled: false,
          ragMediaIds: null,
          selectedKnowledge: {
            id: "service-prompts-e2e-knowledge",
            title: "Service Prompts E2E knowledge"
          },
          compareMode: false,
          compareSelectedModels: [],
          uploadedFiles: []
        })
        ;(
          window as TestHarnessWindow
        ).__tldw_useStoreChatModelSettings?.setState?.({
          apiProvider: "custom-openai-api"
        })
      }, TEST_MODEL)
      await expect
        .poll(() =>
          chatPage.evaluate(
            () =>
              (
                window as TestHarnessWindow
              ).__tldw_useStoreMessageOption?.getState?.().chatMode
          )
        )
        .toBe("rag")
      const rail = chatPage
        .getByTestId("playground-cockpit-left-rail")
        .getByTestId("playground-context-rail")
      await expect(rail).toContainText("Service Prompts E2E knowledge", {
        timeout: 30_000
      })
      await chatInput.fill(question)
      await chatPage
        .getByTestId("composer-inline-send-control")
        .getByRole("button")
        .first()
        .click()
      await expect
        .poll(() => providerRequests.length, { timeout: 30_000 })
        .toBe(1)
      expect(ragRequests).toHaveLength(1)

      const expectedProviderPrompt = initialDetail.default_parts.template
        .replace("{context}", `<doc id='0'>${ragContent}</doc>`)
        .replace("{question}", question)
      const providerMessages = (providerRequests[0]?.messages || []) as Array<{
        content?: unknown
      }>
      expect(
        providerMessages.some(
          (message) => contentText(message.content) === expectedProviderPrompt
        )
      ).toBe(true)
      expect(JSON.stringify(providerRequests[0])).not.toContain(marker)
    } finally {
      await context.close()
      await closeServer(unresolved.server)
    }
  })

  test("preserves a corrupt revision through built-extension reset and recovers packaged state", async () => {
    test.setTimeout(120_000)
    const corruptServer = await startCorruptServer()
    let context: BrowserContext | null = null

    try {
      const launch = await launchWithExtension(EXTENSION_PATH, {
        seedConfig: extensionConfig(corruptServer.baseUrl, FAKE_API_KEY)
      })
      context = launch.context
      const { extensionId, optionsUrl, page } = launch
      await assertHealthy(
        `${corruptServer.baseUrl}/api/v1/health`,
        "Corrupt transport server"
      )
      await grantOrigin(context, extensionId, corruptServer.baseUrl)
      await page.setViewportSize({ width: 1280, height: 900 })
      await page.goto(`${optionsUrl}#/settings/prompt?prompt=${PROMPT_ID}`, {
        waitUntil: "domcontentloaded"
      })
      await expect(
        page.getByText("Saved customization is unavailable")
      ).toBeVisible({ timeout: 30_000 })
      await expect(
        page.getByRole("button", { name: "Reset corrupt customization" })
      ).toBeVisible()
      await assertNoSeriousAxeViolations(page)

      await page
        .getByRole("button", { name: "Reset corrupt customization" })
        .click()
      const resetDialog = page
        .getByRole("dialog")
        .filter({ hasText: "Reset RAG answer?" })
      await expect(resetDialog).toBeVisible()
      await resetDialog
        .getByRole("button", { name: "Reset", exact: true })
        .click()

      await expect
        .poll(
          () =>
            corruptServer.requests.find(
              (request) =>
                request.method === "DELETE" &&
                request.path === `/api/v1/service-prompts/${PROMPT_ID}`
            ),
          { timeout: 15_000 }
        )
        .toBeDefined()
      const resetRequest = corruptServer.requests.find(
        (request) =>
          request.method === "DELETE" &&
          request.path === `/api/v1/service-prompts/${PROMPT_ID}`
      )
      expect(
        new URLSearchParams(resetRequest?.search).get("expected_revision")
      ).toBe(CORRUPT_REVISION)
      await expect(
        page.getByText("Server default", { exact: true })
      ).toBeVisible({ timeout: 30_000 })
      await expect(page.getByRole("textbox", { name: "Template" })).toHaveValue(
        packagedTemplate
      )

      const previewRequestStart = corruptServer.requests.length
      await page.getByRole("button", { name: "Preview" }).click()
      await expect(
        page.getByRole("region", { name: "Prompt preview" })
      ).toBeVisible()
      await page.evaluate(
        () =>
          new Promise<void>((resolve) => {
            requestAnimationFrame(() =>
              requestAnimationFrame(() => resolve())
            )
          })
      )
      expect(
        corruptServer.requests.slice(previewRequestStart),
        "Preview must remain local-only"
      ).toEqual([])

      await page.setViewportSize({ width: 390, height: 844 })
      await expect(
        page.getByRole("textbox", { name: "Template" })
      ).toBeVisible()
      await assertNoSeriousAxeViolations(page)
    } finally {
      await context?.close()
      await closeServer(corruptServer.server)
    }
  })
})
