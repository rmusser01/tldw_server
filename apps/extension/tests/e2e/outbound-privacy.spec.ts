import { cp, mkdir, readFile, writeFile } from "node:fs/promises"
import { createServer, type RequestListener } from "node:http"
import path from "node:path"

import { chromium, expect, test, type BrowserContext, type Worker } from "@playwright/test"

import {
  chromiumFixtureOptions,
  fixtureSetting,
  loopbackOrigin,
  observeOutbound
} from "../../../tldw-frontend/e2e/utils/outbound-observer"

const apiOrigin = loopbackOrigin(18083)
const pageOrigin = loopbackOrigin(18084)
const apiKey = "synthetic-outbound-extension-fixture-key"
const modelId = "outbound-fixture-model"
const prompt = "Reply using the configured local fixture."
const reply = "Outbound fixture worker reply"
const manifestFile = "manifest.json"
const fixturePage = '<!doctype html><title>Local fixture</title><link rel="icon" href="data:,"><textarea></textarea>'

async function listen(port: number, handler: RequestListener) {
  const server = createServer(handler)
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject)
    server.listen(port, "127.0.0.1", resolve)
  })
  return async () => {
    server.closeAllConnections()
    await new Promise<void>((resolve, reject) =>
      server.close(error => error ? reject(error) : resolve())
    )
  }
}

async function launch(profileRoot: string, apiFixture: boolean) {
  const built = path.resolve("build/chrome-mv3")
  const manifest = JSON.parse(await readFile(path.join(built, manifestFile), "utf8"))
  expect(manifest.manifest_version).toBe(3)
  expect(manifest.background?.service_worker).toMatch(/\.js$/)
  expect(manifest.host_permissions).toEqual([fixtureSetting("OUTBOUND_MANIFEST_HOST_PERMISSION")])
  expect(manifest.optional_host_permissions ?? []).toEqual([])
  expect(manifest.externally_connectable).toBeUndefined()
  const copy = path.join(profileRoot, "extension")
  await mkdir(profileRoot, { recursive: true })
  await cp(built, copy, { recursive: true })
  if (apiFixture) {
    // Production manifest checks above use the pristine artifact. Extra access is only
    // granted in the isolated fixture copy.
    manifest.host_permissions.push(`${apiOrigin}/*`)
    await writeFile(path.join(copy, manifestFile), JSON.stringify(manifest))
  }

  const launchOptions = chromiumFixtureOptions()
  const context = await chromium.launchPersistentContext(path.join(profileRoot, "profile"), {
    ...launchOptions,
    channel: "chromium",
    headless: true,
    locale: "en-US",
    ignoreDefaultArgs: ["--disable-extensions"],
    args: [
      ...launchOptions.args,
      `--disable-extensions-except=${copy}`,
      `--load-extension=${copy}`,
      "--disable-crash-reporter",
      "--no-crashpad"
    ]
  })
  const expectedWorker = (url: string) => {
    const parsed = new URL(url)
    return parsed.protocol === "chrome-extension:" &&
      parsed.pathname === `/${manifest.background.service_worker}`
  }
  try {
    const recorder = await observeOutbound(
      context,
      [apiFixture ? apiOrigin : pageOrigin],
      apiFixture ? `${apiOrigin}/api/v1/chat/completions` : undefined
    )
    const worker = context.serviceWorkers().find(candidate => expectedWorker(candidate.url())) ??
      await context.waitForEvent("serviceworker", worker => expectedWorker(worker.url()))
    const extensionOrigin = new URL(worker.url()).origin
    // WHATWG URL does not assign a tuple origin to chrome-extension.
    const extensionBase = extensionOrigin === "null"
      ? "chrome-extension://" + new URL(worker.url()).host
      : extensionOrigin
    return { context, recorder, worker, extensionBase }
  } catch (error) {
    await context.close()
    throw error
  }
}

async function openFixturePopup(context: BrowserContext, worker: Worker, url: string, text: string) {
  const page = await context.newPage()
  await page.goto(url)
  await expect(page.locator("html")).toHaveAttribute("data-tldw-copilot-popup-ready", "true")
  const selection = page.getByRole("textbox")
  await selection.fill(text)
  await selection.selectText()
  await worker.evaluate(async ({ url, text }) => {
    const tab = (await chrome.tabs.query({})).find(candidate => candidate.url === url)
    if (tab?.id === undefined) throw new Error("Local fixture tab is missing")
    await chrome.tabs.sendMessage(tab.id, {
      type: "tldw:popup:open",
      payload: { selectionText: text, pageUrl: url, pageTitle: "Local fixture" }
    })
  }, { url, text })
  return page.locator("#tldw-copilot-popup-host .tldw-popup")
}

test("packaged worker uses the configured fixture", async ({}, testInfo) => {
  let authenticatedChatPosts = 0
  const close = await listen(18083, (request, response) => {
    const pathname = new URL(request.url ?? "/", apiOrigin).pathname
    if (pathname === "/fixture" && request.method === "GET") {
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" })
      response.end(fixturePage)
      return
    }
    response.setHeader("Access-Control-Allow-Origin", request.headers.origin ?? "*")
    response.setHeader("Access-Control-Allow-Credentials", "true")
    if (request.method === "OPTIONS") {
      response.setHeader("Access-Control-Allow-Headers", "content-type, x-api-key, authorization")
      response.setHeader("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
      response.writeHead(204)
      response.end()
      return
    }
    const send = (body: unknown, status = 200) => {
      response.writeHead(status, { "Content-Type": "application/json" })
      response.end(JSON.stringify(body))
    }
    if (pathname === "/health" || pathname === "/api/v1/health") {
      send({ status: "ok" })
      return
    }
    if (pathname === "/openapi.json") {
      send({
        openapi: "3.0.0",
        info: { version: "fixture" },
        paths: {
          "/api/v1/health": { get: {} },
          "/api/v1/chat/completions": { post: {} },
          "/api/v1/llm/models": { get: {} },
          "/api/v1/llm/models/metadata": { get: {} }
        }
      })
      return
    }
    if (pathname === "/api/v1/llm/models/metadata") {
      send({ models: [{
        id: modelId,
        model: modelId,
        name: "Outbound Fixture Model",
        provider: "openai",
        type: "chat",
        is_configured: true,
        provider_is_configured: true,
        provider_enabled: true,
        availability: "available",
        context_length: 4096,
        capabilities: ["chat"]
      }], total: 1 })
      return
    }
    if (pathname === "/api/v1/llm/providers") {
      send({ default_provider: "openai", providers: [{
        name: "openai", provider: "openai", is_configured: true, enabled: true, default_model: modelId
      }] })
      return
    }
    if (pathname === "/api/v1/llm/models") {
      send([modelId])
      return
    }
    if (pathname === "/api/v1/chat/completions" && request.method === "POST") {
      if (request.headers["x-api-key"] !== apiKey) {
        send({ detail: "invalid fixture key" }, 401)
        return
      }
      request.setEncoding("utf8")
      let body = ""
      request.on("data", chunk => { body += chunk })
      request.once("end", () => {
        if (!body.includes(prompt)) {
          send({ detail: "fixture prompt missing" }, 400)
          return
        }
        authenticatedChatPosts++
        if (JSON.parse(body).stream) {
          response.writeHead(200, { "Content-Type": "text/event-stream" })
          const token = JSON.stringify({ choices: [{ delta: { content: reply } }] })
          response.end(`data: ${token}\n\ndata: [DONE]\n\n`)
        } else {
          send({ choices: [{ message: { role: "assistant", content: reply } }] })
        }
      })
      return
    }
    send({ detail: "fixture route not found" }, 404)
  })
  try {
    const { context, recorder, worker, extensionBase } =
      await launch(testInfo.outputPath("worker"), true)
    try {
      await worker.evaluate(async ({ model, serverUrl, key }) => {
        const stored = {
          __tldw_first_run_complete: true,
          __tldw_allow_offline: true,
          tldw_skip_landing_hub: true,
          tldwConfig: JSON.stringify({ authMode: "single-user", serverUrl, apiKey: key }),
          selectedModel: JSON.stringify(model)
        }
        await chrome.storage.local.set(stored)
        await chrome.storage.sync.set(stored)
      }, { model: `tldw:${modelId}`, serverUrl: apiOrigin, key: apiKey })

      const options = await context.newPage()
      await options.goto(`${extensionBase}/options.html#/settings/tldw`)
      await expect(options.getByLabel("Server URL")).toHaveValue(apiOrigin)

      const sidepanel = await context.newPage()
      await sidepanel.goto(`${extensionBase}/sidepanel.html#/chat`)
      await expect(sidepanel.locator("#root > *").first()).toBeAttached()

      const popup = await openFixturePopup(context, worker, `${apiOrigin}/fixture`, prompt)
      await expect(popup.locator('[data-role="response"]')).toHaveText(reply, { timeout: 20_000 })
      expect(authenticatedChatPosts).toBeGreaterThan(0)
      expect(recorder.sawWorkerRequest()).toBe(true)
    } finally {
      await recorder.finish()
    }
  } finally {
    await close()
  }
})

test("content script activates on a local page", async ({}, testInfo) => {
  let documentRequests = 0
  let otherRequests = 0
  const close = await listen(18084, (request, response) => {
    if (request.method === "GET" && request.url === "/fixture") {
      documentRequests++
      response.writeHead(200, { "Content-Type": "text/html; charset=utf-8" })
      response.end(fixturePage)
      return
    }
    otherRequests++
    response.writeHead(404)
    response.end()
  })
  try {
    const { context, recorder, worker } = await launch(testInfo.outputPath("content"), false)
    try {
      const popup = await openFixturePopup(context, worker, `${pageOrigin}/fixture`, "Local fixture text")
      await expect(popup).toBeVisible()
    } finally {
      await recorder.finish()
    }
    expect(documentRequests).toBe(1)
    expect(otherRequests).toBe(0)
  } finally {
    await close()
  }
})
