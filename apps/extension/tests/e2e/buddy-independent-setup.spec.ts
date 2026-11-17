import {
  type BrowserContext,
  type Locator,
  type Page,
  expect,
  test
} from "@playwright/test"
import { createHash } from "node:crypto"
import { cpSync, readFileSync, readdirSync, writeFileSync } from "node:fs"
import { createServer } from "node:http"
import path from "node:path"

import { buildPersonaVisualPackFixture } from "../../../tldw-frontend/e2e/fixtures/persona-visual-packs"
import { seedAuth } from "../../../tldw-frontend/e2e/utils/helpers"
import { launchWithExtension } from "./utils/extension"
import { grantHostPermission } from "./utils/permissions"

const KEY = "buddy-qualification-synthetic-key"
const WORKSPACE = "buddy-qualification-workspace"
const CONVERSATION = "buddy-qualification-conversation"
const POSITION = { x: 240, y: 160 }

async function startBuddyFixture(upgraded: boolean) {
  const pack = buildPersonaVisualPackFixture("qualification")
  const buddy = {
    id: "fixture-buddy",
    name: "Fixture Buddy",
    version: upgraded ? 7 : 1,
    optional_persona_id: null,
    optional_persona_available: false,
    display_mode: "dynamic",
    manifest: pack.manifest,
    assets: Object.values(pack.assets_by_id).map(({ url, ...asset }) => ({
      ...asset,
      content_url: url
    })),
    attribution: {}
  }
  const workspace = {
    id: WORKSPACE,
    name: "Qualification workspace",
    version: upgraded ? 4 : 1,
    deleted: false,
    archived: false,
    config: {},
    assistant_defaults: upgraded
      ? {
          assistant_kind: "persona",
          assistant_id: "research_assistant",
          persona_memory_mode: "read_only"
        }
      : null
  }
  const conversation = {
    id: CONVERSATION,
    title: "Qualification conversation",
    workspace_id: null,
    scope_type: "global",
    assistant_kind: null,
    assistant_id: null,
    created_at: "2026-09-06T12:00:00.000Z",
    version: 1
  }
  const attachment = {
    client_slot: "default",
    version: upgraded ? 3 : 0,
    attachment: upgraded
      ? {
          buddy_id: buddy.id,
          scope_type: "conversation",
          scope_id: CONVERSATION
        }
      : null,
    target: upgraded ? { title: conversation.title, workspace_id: null } : null
  }
  const writes: Array<{ method: string; path: string; body: unknown }> = []
  const server = createServer(async (request, response) => {
    response.setHeader(
      "Access-Control-Allow-Origin",
      String(request.headers.origin || "*")
    )
    response.setHeader(
      "Access-Control-Allow-Methods",
      "GET, PATCH, PUT, POST, OPTIONS"
    )
    response.setHeader(
      "Access-Control-Allow-Headers",
      "Content-Type, X-API-KEY, Authorization"
    )
    if (request.method === "OPTIONS") {
      response.writeHead(204)
      response.end()
      return
    }
    const pathname = new URL(request.url || "/", "http://127.0.0.1").pathname
    const send = (body: unknown, status = 200) => {
      response.writeHead(status, { "Content-Type": "application/json" })
      response.end(JSON.stringify(body))
    }
    if (
      request.headers["x-api-key"] !== KEY &&
      ![
        "/health",
        "/api/v1/health",
        "/api/v1/health/live",
        "/api/v1/rag/health",
        "/openapi.json",
        "/api/v1/config/docs-info"
      ].includes(pathname)
    ) {
      send({ detail: "Invalid synthetic key" }, 401)
      return
    }
    let raw = ""
    for await (const chunk of request) raw += chunk
    const body = raw ? JSON.parse(raw) : null
    if (request.method !== "GET")
      writes.push({ method: request.method!, path: pathname, body })
    if (pathname === "/openapi.json") {
      send({
        openapi: "3.1.0",
        info: { title: "Buddy synthetic fixture", version: "1" },
        paths: { "/api/v1/media/": { get: {} } }
      })
      return
    }
    if (pathname === "/api/v1/users/me/profile") {
      send({ id: 1, username: "buddy-fixture" })
      return
    }
    if (pathname === "/api/v1/config/docs-info") {
      send({
        capabilities: { persona: true },
        supported_features: { persona: true }
      })
      return
    }
    if (pathname === "/api/v1/buddies") {
      send({ buddies: [buddy] })
      return
    }
    if (pathname === `/api/v1/buddies/${buddy.id}`) {
      if (request.method === "PATCH") {
        expect(body.expected_version).toBe(buddy.version)
        buddy.display_mode = body.display_mode
        buddy.version += 1
      }
      send(buddy)
      return
    }
    if (pathname === "/api/v1/buddies/attachment") {
      if (request.method === "PUT") {
        expect(body.expected_version).toBe(attachment.version)
        attachment.attachment = {
          buddy_id: body.buddy_id,
          scope_type: body.scope_type,
          scope_id: body.scope_id
        }
        attachment.version += 1
        attachment.target = {
          title:
            body.scope_type === "workspace"
              ? workspace.name
              : conversation.title,
          workspace_id: body.scope_type === "workspace" ? WORKSPACE : null
        }
      }
      send(attachment)
      return
    }
    if (pathname === "/api/v1/buddies/attachment/conversations") {
      send({ conversations: attachment.attachment ? [conversation] : [] })
      return
    }
    if (pathname === "/api/v1/buddies/attachment/activity") {
      send({ items: [] })
      return
    }
    if (pathname === "/api/v1/buddies/turns") {
      send({ turns: [] })
      return
    }
    if (pathname.startsWith("/api/v1/buddies/conversation-targets/")) {
      send(
        pathname.endsWith("/reply-settings")
          ? { provider: null, model: null }
          : conversation
      )
      return
    }
    if (pathname === "/api/v1/workspaces/") {
      send({ items: [workspace] })
      return
    }
    if (pathname === `/api/v1/workspaces/${WORKSPACE}`) {
      if (request.method === "PATCH") {
        expect(body.version).toBe(workspace.version)
        workspace.assistant_defaults = body.assistant_defaults
        workspace.version += 1
      }
      send(workspace)
      return
    }
    if (pathname === "/api/v1/persona/catalog") {
      send([{ id: "research_assistant", name: "Research Assistant" }])
      return
    }
    if (pathname === "/api/v1/persona/visual-starter-packs") {
      send({ starter_packs: [] })
      return
    }
    if (pathname.startsWith("/api/v1/chats")) {
      send(
        pathname.includes("/messages")
          ? []
          : pathname.endsWith(CONVERSATION)
            ? conversation
            : [conversation]
      )
      return
    }
    if (pathname.includes("/llm/")) {
      send({ models: [], providers: [] })
      return
    }
    if (pathname.startsWith("/api/v1/media")) {
      send({ items: [], pagination: { total_items: 0 } })
      return
    }
    send({ status: "healthy", ok: true, unread_count: 0 })
  })
  await new Promise<void>((resolve, reject) => {
    server.once("error", reject)
    server.listen(0, "127.0.0.1", resolve)
  })
  const port = (server.address() as { port: number }).port
  return {
    url: `http://127.0.0.1:${port}`,
    buddy,
    workspace,
    attachment,
    writes,
    close: () =>
      new Promise<void>((resolve, reject) =>
        server.close((error) => (error ? reject(error) : resolve()))
      )
  }
}

async function openBuddyOptions(controls: Locator) {
  const summary = controls.getByText("Buddy options", { exact: true })
  if ((await summary.locator("..").getAttribute("open")) === null)
    await summary.click()
}

async function exerciseBuddy(
  page: Page,
  fixture: Awaited<ReturnType<typeof startBuddyFixture>>,
  upgraded: boolean
) {
  if (upgraded) {
    await expect(page.getByTestId("independent-buddy")).toBeVisible()
    await expect
      .poll(() =>
        page.evaluate(
          () =>
            JSON.parse(
              localStorage.getItem("tldw-persona-buddy-shell") || "null"
            )?.version
        )
      )
      .toBe(1)
    const box = await page.getByTestId("independent-buddy").boundingBox()
    expect(box?.x).toBe(POSITION.x)
    expect(box?.y).toBe(POSITION.y)
  }
  await page
    .getByRole("button", { name: "Buddy & Persona", exact: true })
    .first()
    .click()
  const management = page.getByRole("dialog", {
    name: "Buddy & Persona Management"
  })
  await expect(
    management.getByRole("button", { name: "Use Fixture Buddy" })
  ).toBeVisible()
  await management.getByRole("button", { name: "Use Fixture Buddy" }).click()
  await management
    .getByLabel("Scope", { exact: true })
    .selectOption("conversation")
  await management
    .getByLabel("Conversation location", { exact: true })
    .selectOption("")
  await management
    .getByLabel("Conversation", { exact: true })
    .selectOption(CONVERSATION)
  await management.getByRole("button", { name: "Apply", exact: true }).click()
  await expect(management).toHaveCount(0)
  await expect(page.getByTestId("independent-buddy")).toBeVisible()
  expect(fixture.attachment.attachment).toEqual({
    buddy_id: fixture.buddy.id,
    scope_type: "conversation",
    scope_id: CONVERSATION
  })
  expect(fixture.buddy.optional_persona_id).toBeNull()
  await page
    .getByRole("button", {
      name: "Open Fixture Buddy — Qualification conversation",
      exact: true
    })
    .click()
  const controls = page.getByRole("dialog", {
    name: "Fixture Buddy — Qualification conversation"
  })
  await openBuddyOptions(controls)
  await controls
    .getByLabel("Expressions", { exact: true })
    .selectOption("static")
  await expect.poll(() => fixture.buddy.display_mode).toBe("static")
  await page.screenshot({
    path: test.info().outputPath("static-conversation.png")
  })
  await controls.getByRole("button", { name: "Close", exact: true }).click()
  await page
    .getByRole("button", { name: "Buddy & Persona", exact: true })
    .first()
    .click()
  await management
    .getByLabel("Scope", { exact: true })
    .selectOption("workspace")
  await management
    .getByLabel("Workspace", { exact: true })
    .selectOption(WORKSPACE)
  const defaults = management.getByLabel(
    "Default Persona for new conversations",
    { exact: true }
  )
  await expect(defaults).toHaveValue(upgraded ? "research_assistant" : "")
  await defaults.selectOption(upgraded ? "" : "research_assistant")
  await management.getByRole("button", { name: "Cancel", exact: true }).click()
  expect(fixture.workspace.assistant_defaults?.assistant_id ?? "").toBe(
    upgraded ? "research_assistant" : ""
  )
  await page
    .getByRole("button", { name: "Buddy & Persona", exact: true })
    .first()
    .click()
  await management
    .getByLabel("Scope", { exact: true })
    .selectOption("workspace")
  await management
    .getByLabel("Workspace", { exact: true })
    .selectOption(WORKSPACE)
  await defaults.selectOption("research_assistant")
  await management.getByRole("button", { name: "Apply", exact: true }).click()
  await expect(management).toHaveCount(0)
  expect(fixture.attachment.attachment).toEqual({
    buddy_id: fixture.buddy.id,
    scope_type: "workspace",
    scope_id: WORKSPACE
  })
  expect(fixture.workspace.assistant_defaults).toEqual({
    assistant_kind: "persona",
    assistant_id: "research_assistant",
    persona_memory_mode: "read_only",
    style: null,
    voice: null,
    tool_policy_profile_id: null
  })
  await page
    .getByRole("button", {
      name: "Open Fixture Buddy — Qualification workspace",
      exact: true
    })
    .click()
  const workspaceControls = page.getByRole("dialog", {
    name: "Fixture Buddy — Qualification workspace"
  })
  await openBuddyOptions(workspaceControls)
  await workspaceControls
    .getByLabel("Expressions", { exact: true })
    .selectOption("dynamic")
  await expect.poll(() => fixture.buddy.display_mode).toBe("dynamic")
  await workspaceControls
    .getByRole("button", { name: "Close", exact: true })
    .click()
  await page.reload({ waitUntil: "domcontentloaded" })
  await expect(
    page.getByRole("button", {
      name: "Open Fixture Buddy — Qualification workspace",
      exact: true
    })
  ).toBeVisible()
  await page
    .getByRole("button", {
      name: "Open Fixture Buddy — Qualification workspace",
      exact: true
    })
    .click()
  await openBuddyOptions(workspaceControls)
  await expect(
    workspaceControls.getByLabel("Expressions", { exact: true })
  ).toHaveValue("dynamic")
  await expect(
    page.getByTestId("independent-buddy").getByTestId("persona-visual-frame")
  ).toBeVisible()
  await page.screenshot({
    path: test.info().outputPath("dynamic-workspace-after-reload.png")
  })
  await test.info().attach("state-and-writes.json", {
    body: JSON.stringify(
      {
        buddy: fixture.buddy,
        workspace: fixture.workspace,
        attachment: fixture.attachment,
        writes: fixture.writes
      },
      null,
      2
    ),
    contentType: "application/json"
  })
}

for (const surface of ["extension", "webui"] as const) {
  for (const upgraded of [false, true]) {
    test(`independent Buddy ${surface} ${upgraded ? "upgraded" : "fresh"} profile setup`, async ({
      browser
    }) => {
      test.skip(
        surface === "webui" && !process.env.TLDW_BUDDY_WEB_URL,
        "Set TLDW_BUDDY_WEB_URL for the real WebUI surface"
      )
      const fixture = await startBuddyFixture(upgraded)
      let context: BrowserContext | undefined
      try {
        const flags = {
          __tldw_first_run_complete: true,
          tldw_skip_landing_hub: true
        }
        const config = {
          serverUrl: fixture.url,
          apiKey: KEY,
          authMode: "single-user",
          authSource: "manual",
          credentialSource: "manual",
          apiKeyPersistence: "device",
          apiKeyServerOrigin: fixture.url
        }
        let page: Page
        let url: string
        if (surface === "extension") {
          const extensionPath = test.info().outputPath("extension")
          cpSync(path.resolve(".output/chrome-mv3"), extensionPath, {
            recursive: true
          })
          const manifestPath = path.join(extensionPath, "manifest.json")
          const manifest = JSON.parse(readFileSync(manifestPath, "utf8"))
          manifest.host_permissions = [
            ...new Set([
              ...(manifest.host_permissions || []),
              `${fixture.url}/*`
            ])
          ]
          writeFileSync(manifestPath, JSON.stringify(manifest))
          const launched = await launchWithExtension(extensionPath, {
            seedConfig: {
              ...flags,
              serverUrl: fixture.url,
              apiKey: KEY,
              authMode: "single-user",
              tldwConfig: config
            }
          })
          context = launched.context
          page = launched.page
          url = `${launched.optionsUrl}#/workspaces`
          const buildPath = path.resolve(".output/chrome-mv3")
          const hash = (file: string) =>
            createHash("sha256").update(readFileSync(file)).digest("hex")
          const scripts = (
            readdirSync(buildPath, { recursive: true }) as string[]
          ).filter((file) => /\.m?js$/.test(file))
          const scriptHashes = Object.fromEntries(
            scripts.map((file) => [file, hash(path.join(buildPath, file))])
          )
          expect(
            Object.fromEntries(
              scripts.map((file) => [
                file,
                hash(path.join(extensionPath, file))
              ])
            )
          ).toEqual(scriptHashes)
          const loadedManifest = await page.evaluate(() =>
            chrome.runtime.getManifest()
          )
          expect(loadedManifest.host_permissions).toContain(`${fixture.url}/*`)
          await test.info().attach("artifact-receipt.json", {
            body: JSON.stringify(
              {
                buildPath,
                copiedArtifactPath: extensionPath,
                originalManifestSha256: hash(
                  path.join(buildPath, "manifest.json")
                ),
                copiedManifestSha256: hash(manifestPath),
                loadedManifest,
                scriptHashes,
                userAgent: await page.evaluate(() => navigator.userAgent),
                transport:
                  "native extension background HTTP; no request API shim"
              },
              null,
              2
            ),
            contentType: "application/json"
          })
          expect(
            await grantHostPermission(
              context,
              launched.extensionId,
              `${fixture.url}/*`
            )
          ).toBe(true)
        } else {
          context = await browser.newContext()
          page = await context.newPage()
          url = `${process.env.TLDW_BUDDY_WEB_URL}/workspaces`
          await seedAuth(page, {
            serverUrl: fixture.url,
            apiKey: KEY,
            allowOffline: false
          })
        }
        if (upgraded) {
          await context.addInitScript((position) => {
            if (!localStorage.getItem("__buddyUpgradeSeeded")) {
              localStorage.setItem(
                "tldw-persona-buddy-shell",
                JSON.stringify({
                  version: 0,
                  state: {
                    firstUseHintDismissed: true,
                    positions: {
                      "web-desktop": position,
                      "sidepanel-desktop": position
                    }
                  }
                })
              )
              localStorage.setItem("__buddyUpgradeSeeded", "true")
            }
          }, POSITION)
        }
        await page.goto(url, { waitUntil: "domcontentloaded" })
        if (upgraded) await page.reload({ waitUntil: "domcontentloaded" })
        await exerciseBuddy(page, fixture, upgraded)
      } finally {
        await context?.close()
        await fixture.close()
      }
    })
  }
}
