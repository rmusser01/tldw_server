import {
  type BrowserContext,
  type Locator,
  type Page,
  expect,
  test
} from "@playwright/test"
import fs from "node:fs"
import http from "node:http"
import { AddressInfo } from "node:net"
import path from "node:path"

import {
  forceConnected,
  forceConnectionState,
  waitForConnectionStore
} from "./utils/connection"
import { launchWithExtension } from "./utils/extension"
import { grantHostPermission } from "./utils/permissions"

const EXT_PATH = path.resolve(
  process.env.TLDW_E2E_EXTENSION_PATH || ".output/chrome-mv3"
)
const PROJECT_ID = 42
const RUNTIME_SENTINEL = "EXTENSION_RUNTIME_ONLY_DO_NOT_PERSIST"
const ORIGINAL_DRAFT = "Extension unsent draft, exactly."
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

type CapabilityMode = "supported" | "old" | "unknown"
type JsonObject = Record<string, unknown>
type RecordedRequest = {
  method: string
  url: string
  json: JsonObject | null
}
type RecipeMock = {
  server: http.Server
  baseUrl: string
  requests: RecordedRequest[]
  creates: () => RecordedRequest[]
  updates: () => RecordedRequest[]
}

const readBody = (request: http.IncomingMessage) =>
  new Promise<string>((resolve) => {
    let body = ""
    request.on("data", (chunk) => {
      body += chunk
    })
    request.on("end", () => resolve(body))
  })

const parseJson = (body: string): JsonObject | null => {
  try {
    const parsed = JSON.parse(body)
    return parsed && typeof parsed === "object" && !Array.isArray(parsed)
      ? parsed
      : null
  } catch {
    return null
  }
}

const serverPrompt = (body: JsonObject, id: number) => ({
  id,
  project_id: PROJECT_ID,
  uuid: `extension-recipe-${id}`,
  name: String(body.name || "Untitled recipe"),
  system_prompt: String(body.system_prompt || ""),
  user_prompt: String(body.user_prompt || ""),
  prompt_format: body.prompt_format,
  prompt_schema_version: body.prompt_schema_version,
  prompt_definition: body.prompt_definition,
  few_shot_examples: null,
  modules_config: null,
  version_number: 1,
  change_description: "extension recipe E2E",
  parent_version_id: null,
  updated_at: "2026-09-11T12:00:00Z"
})

async function startRecipeMock(mode: CapabilityMode = "supported") {
  const requests: RecordedRequest[] = []
  let nextId = 800
  const server = http.createServer(async (request, response) => {
    const method = String(request.method || "GET").toUpperCase()
    const url = request.url || "/"
    const send = (status: number, body: unknown) => {
      response.writeHead(status, {
        "content-type": "application/json",
        "access-control-allow-origin": "*",
        "access-control-allow-headers":
          "content-type, x-api-key, authorization, x-tldw-recipe-owner, x-tldw-recipe-operation",
        "access-control-expose-headers":
          "x-tldw-recipe-owner, x-tldw-recipe-operation",
        "access-control-allow-methods": "GET, POST, PUT, PATCH, OPTIONS"
      })
      response.end(JSON.stringify(body))
    }
    if (method === "OPTIONS") return send(204, {})
    const body = ["POST", "PUT", "PATCH"].includes(method)
      ? await readBody(request)
      : ""
    const entry = { method, url, json: body ? parseJson(body) : null }
    requests.push(entry)

    if (url === "/api/v1/health" || url === "/api/v1/health/live") {
      return send(200, { status: "ok" })
    }
    if (url.startsWith("/api/v1/llm/models/metadata")) {
      return send(200, { models: [] })
    }
    if (url === "/api/v1/llm/models") return send(200, [])
    if (url === "/api/v1/llm/providers") return send(200, { providers: [] })
    if (url === "/api/v1/config/docs-info") return send(200, {})
    if (url === "/api/v1/notifications/unread-count") {
      return send(200, { count: 0 })
    }
    if (url.startsWith("/api/v1/users/me/profile")) {
      return send(200, { preferences: {} })
    }
    if (url === "/openapi.json") {
      return send(200, {
        openapi: "3.0.0",
        info: { version: "recipe-e2e" },
        paths: {
          "/api/v1/health": {},
          "/api/v1/prompts/capabilities": {},
          "/api/v1/prompt-studio/prompts/create": {}
        }
      })
    }
    if (url === "/api/v1/prompts/capabilities") {
      if (mode === "unknown") return send(404, { detail: "not found" })
      return send(200, {
        prompt_improvement_v1: { supported: false, limits: LIMITS },
        single_text_recipe_v2: { supported: mode === "supported" },
        prompt_persistence: {
          create_authorized: mode === "supported",
          update_authorized: mode === "supported"
        }
      })
    }
    if (url === "/api/v1/prompt-studio/prompts/create" && method === "POST") {
      return send(200, {
        success: true,
        data: serverPrompt(entry.json || {}, ++nextId)
      })
    }
    if (
      url.startsWith("/api/v1/prompt-studio/prompts/update/") &&
      method === "PUT"
    ) {
      return send(200, {
        success: true,
        data: serverPrompt(entry.json || {}, Number(url.split("/").at(-1)))
      })
    }
    if (url.startsWith("/api/v1/characters")) return send(200, [])
    return send(404, { detail: "not found" })
  })
  await new Promise<void>((resolve) => server.listen(0, "127.0.0.1", resolve))
  const port = (server.address() as AddressInfo).port
  return {
    server,
    baseUrl: `http://127.0.0.1:${port}`,
    requests,
    creates: () =>
      requests.filter(
        (request) =>
          request.method === "POST" &&
          request.url === "/api/v1/prompt-studio/prompts/create"
      ),
    updates: () =>
      requests.filter(
        (request) =>
          request.method === "PUT" &&
          request.url.startsWith("/api/v1/prompt-studio/prompts/update/")
      )
  } satisfies RecipeMock
}

async function stopRecipeMock(mock: RecipeMock) {
  mock.server.closeAllConnections?.()
  await new Promise<void>((resolve) => mock.server.close(() => resolve()))
}

const seedConfig = (baseUrl: string) => ({
  __tldw_first_run_complete: true,
  __tldw_allow_offline: true,
  tldw_skip_landing_hub: true,
  promptStudioDefaults: {
    defaultProjectId: PROJECT_ID,
    autoSyncWorkspacePrompts: true
  },
  tldwConfig: {
    serverUrl: baseUrl,
    authMode: "single-user",
    authSource: "manual",
    apiKey: "extension-recipe-e2e-key",
    credentialSource: "manual",
    apiKeyPersistence: "device",
    apiKeyServerOrigin: new URL(baseUrl).origin
  }
})

async function launchSurface(
  mock: RecipeMock,
  surface: "sidepanel" | "options",
  viewport?: { width: number; height: number }
) {
  const launched = await launchWithExtension(EXT_PATH, {
    seedConfig: seedConfig(mock.baseUrl),
    seedLocalStorage: {
      "tldw:nextgenComposerEnabled": "0",
      "tldw:composerVariant": "v1",
      playgroundComposerOptionsExpanded: "true"
    }
  })
  expect(
    await grantHostPermission(
      launched.context,
      launched.extensionId,
      `${new URL(mock.baseUrl).origin}/*`
    )
  ).toBe(true)
  const page =
    surface === "sidepanel"
      ? await launched.openSidepanel("/chat")
      : launched.page
  if (surface === "options") {
    await page.goto(`${launched.optionsUrl}#/chat`, {
      waitUntil: "domcontentloaded"
    })
  }
  if (viewport) await page.setViewportSize(viewport)
  await waitForConnectionStore(page, `single-text-recipe:${surface}`)
  await forceConnected(
    page,
    { serverUrl: mock.baseUrl },
    `single-text-recipe:${surface}:connected`
  )
  const input = page.getByTestId("chat-input").filter({ visible: true })
  await expect(input).toHaveCount(1, { timeout: 20_000 })
  await expect(input).toBeEditable()
  return { ...launched, page, input }
}

async function openRecipe(page: Page) {
  await page.getByRole("button", { name: "Improve prompt" }).click()
  await page.getByRole("button", { name: /Build from recipe/ }).click()
  const builder = page.getByRole("region", {
    name: "Structured recipe builder"
  })
  await expect(builder).toBeVisible()
  return builder
}

async function expectNoHorizontalOverflow(page: Page) {
  const dimensions = await page.evaluate(() => ({
    viewportWidth: document.documentElement.clientWidth,
    scrollWidth: Math.max(
      document.documentElement.scrollWidth,
      document.body?.scrollWidth ?? 0
    )
  }))
  expect(dimensions.scrollWidth).toBeLessThanOrEqual(
    dimensions.viewportWidth + 1
  )
}

async function waitForRecipeDrawerContained(page: Page) {
  const dialog = page.getByRole("dialog", { name: "Build from recipe" })
  await expect
    .poll(() =>
      dialog.evaluate((element) => {
        const rect = element.getBoundingClientRect()
        return Math.max(0, -rect.left, rect.right - window.innerWidth)
      })
    )
    .toBeLessThanOrEqual(1)
}

async function fillTask(builder: Locator, value: string) {
  await builder
    .getByRole("textbox", { name: "Current value for Task (not saved)" })
    .fill(value)
}

function expectPersistedClearTaskRecipe(
  request: RecordedRequest,
  expectedObjectiveContent: string
) {
  expect(request.json).not.toBeNull()
  const body = request.json as JsonObject
  const serialized = JSON.stringify(body)
  expect(body.prompt_format).toBe("structured")
  expect(body.prompt_schema_version).toBe(2)
  expect(body.prompt_definition).toEqual({
    schema_version: 2,
    format: "structured",
    definition_kind: "single_text_recipe",
    assembly_config: {
      assembly_mode: "single_text",
      target_role: "user",
      render_format: "xml",
      block_separator: "\n\n"
    },
    variables: [
      {
        name: "task",
        label: "Task",
        description: "The task to complete.",
        required: true,
        default_value: "Extension saved default",
        input_type: "textarea",
        options: null,
        max_length: null
      }
    ],
    blocks: [
      {
        id: "objective",
        name: "Objective",
        section_key: "objective",
        role: "user",
        kind: "objective",
        content: expectedObjectiveContent,
        enabled: true,
        order: 10,
        is_template: true
      },
      {
        id: "context_inputs",
        name: "Context / inputs",
        section_key: "context_inputs",
        role: "user",
        kind: "context_inputs",
        content:
          "Use the context and inputs provided by the user. If essential information is missing, state what is needed before proceeding.",
        enabled: true,
        order: 20,
        is_template: false
      },
      {
        id: "constraints",
        name: "Constraints",
        section_key: "constraints",
        role: "user",
        kind: "constraints",
        content:
          "Follow every explicit constraint. Preserve supplied names, facts, code, and required formatting; do not invent requirements.",
        enabled: true,
        order: 30,
        is_template: false
      },
      {
        id: "output",
        name: "Output",
        section_key: "output",
        role: "user",
        kind: "output",
        content:
          "Return the requested result directly. Make it clear, complete, and concise.",
        enabled: true,
        order: 40,
        is_template: false
      }
    ]
  })
  expect(serialized).not.toContain(RUNTIME_SENTINEL)
  expect(serialized).not.toMatch(
    /runtimeValues|runtime_values|variable_values|resolved_values/
  )
}

const clearTaskDefinition = (schemaVersion = 2) => ({
  schema_version: schemaVersion,
  format: "structured",
  definition_kind: "single_text_recipe",
  assembly_config: {
    assembly_mode: "single_text",
    target_role: "user",
    render_format: "xml",
    block_separator: "\n\n"
  },
  variables: [
    {
      name: "task",
      label: "Task",
      required: true,
      default_value: null,
      input_type: "textarea"
    }
  ],
  blocks: [
    {
      id: "objective",
      name: "Objective",
      section_key: "objective",
      role: "user",
      content: "Complete this task: {{task}}",
      enabled: true,
      order: 10,
      is_template: true
    }
  ]
})

async function seedMixedVersions(page: Page) {
  await page.evaluate(
    ({ known, future }) =>
      new Promise<void>((resolve, reject) => {
        const open = indexedDB.open("PageAssistDatabase")
        open.onerror = () => reject(open.error)
        open.onsuccess = () => {
          const database = open.result
          const transaction = database.transaction("prompts", "readwrite")
          transaction.objectStore("prompts").put(known)
          transaction.objectStore("prompts").put(future)
          transaction.oncomplete = () => {
            database.close()
            resolve()
          }
          transaction.onerror = () => reject(transaction.error)
        }
      }),
    {
      known: {
        id: "extension-known-v2",
        title: "Extension known v2",
        name: "Extension known v2",
        content: "known",
        is_system: false,
        promptFormat: "structured",
        promptSchemaVersion: 2,
        structuredPromptDefinition: clearTaskDefinition(),
        syncStatus: "local",
        sourceSystem: "workspace",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        deletedAt: null
      },
      future: {
        id: "extension-future-v3",
        title: "EXTENSION_FUTURE_V3",
        name: "EXTENSION_FUTURE_V3",
        content: "EXTENSION_V3_SENTINEL",
        is_system: false,
        promptFormat: "structured",
        promptSchemaVersion: 3,
        structuredPromptDefinition: clearTaskDefinition(3),
        syncStatus: "local",
        sourceSystem: "workspace",
        createdAt: Date.now(),
        updatedAt: Date.now(),
        deletedAt: null
      }
    }
  )
}

async function readPrompt(page: Page, id: string) {
  return page.evaluate(
    (promptId) =>
      new Promise<unknown>((resolve, reject) => {
        const open = indexedDB.open("PageAssistDatabase")
        open.onerror = () => reject(open.error)
        open.onsuccess = () => {
          const database = open.result
          const transaction = database.transaction("prompts", "readonly")
          const request = transaction.objectStore("prompts").get(promptId)
          request.onsuccess = () => {
            database.close()
            resolve(request.result)
          }
          request.onerror = () => reject(request.error)
        }
      }),
    id
  )
}

test.describe("Packaged extension single-text structured recipes", () => {
  test.describe.configure({ mode: "serial" })

  test("sidepanel builds, saves, reopens, clones, updates, applies, and undoes without persisting runtime values", async () => {
    test.setTimeout(120_000)
    const mock = await startRecipeMock()
    let context: BrowserContext | null = null
    try {
      const launched = await launchSurface(mock, "sidepanel")
      context = launched.context
      await launched.input.fill(ORIGINAL_DRAFT)
      const builder = await openRecipe(launched.page)
      await fillTask(builder, RUNTIME_SENTINEL)
      await builder
        .getByRole("checkbox", {
          name: "Use a saved starter default for Task"
        })
        .check()
      await builder
        .getByRole("textbox", { name: "Starter default for Task (saved)" })
        .fill("Extension saved default")
      const compiled = await builder
        .getByRole("textbox", { name: "Compiled prompt preview" })
        .inputValue()
      const save = builder.getByRole("button", { name: "Save as new recipe" })
      await expect(save).toBeEnabled()
      await save.click()
      await expect.poll(() => mock.creates().length).toBe(1)
      expectPersistedClearTaskRecipe(
        mock.creates()[0],
        "Complete this task:\n\n{{task}}"
      )

      const source = builder.getByRole("combobox", { name: "Recipe source" })
      await source.selectOption({ label: "Untitled recipe" })
      await builder
        .getByRole("button", { name: "Edit Objective block" })
        .click()
      await builder
        .getByRole("textbox", { name: "Block content" })
        .fill("Extension update: {{task}}")
      await builder.getByRole("button", { name: "Update recipe" }).click()
      await expect.poll(() => mock.updates().length).toBe(1)
      expectPersistedClearTaskRecipe(
        mock.updates()[0],
        "Extension update: {{task}}"
      )
      await builder.getByRole("button", { name: "Save as new recipe" }).click()
      await expect.poll(() => mock.creates().length).toBe(2)
      expectPersistedClearTaskRecipe(
        mock.creates()[1],
        "Extension update: {{task}}"
      )

      await source.selectOption({ label: "Clear task" })
      await fillTask(builder, RUNTIME_SENTINEL)
      await builder
        .getByRole("button", { name: "Apply to user message" })
        .click()
      await expect(launched.input).toHaveValue(compiled)
      await launched.page.getByRole("button", { name: "Undo recipe" }).click()
      await expect(launched.input).toHaveValue(ORIGINAL_DRAFT)
      const trigger = launched.page.getByRole("button", {
        name: "Improve prompt"
      })
      await trigger.focus()
      await trigger.press("Enter")
      await launched.page
        .getByRole("button", { name: /Build from recipe/ })
        .press("Enter")
      const reopenedBuilder = launched.page.getByRole("region", {
        name: "Structured recipe builder"
      })
      await expect(reopenedBuilder).toBeVisible()
      await launched.page.keyboard.press("Escape")
      await expect(reopenedBuilder).not.toBeVisible()
      await expect(trigger).toBeFocused()
    } finally {
      await context?.close()
      await stopRecipeMock(mock)
    }
  })

  test("options chat exposes all starters through a keyboard-reachable, mobile recipe sheet that is axe-clean in both themes", async () => {
    test.setTimeout(120_000)
    const mock = await startRecipeMock()
    let context: BrowserContext | null = null
    try {
      const launched = await launchSurface(mock, "options", {
        width: 390,
        height: 780
      })
      context = launched.context
      expect(launched.page.url()).toContain("/options.html#/chat")
      const visitedThemes = new Set<string>()
      let compiled = ""
      for (let index = 0; index < 2; index += 1) {
        const trigger = launched.page.getByRole("button", {
          name: "Improve prompt"
        })
        await trigger.focus()
        await trigger.press("Enter")
        const action = launched.page.getByRole("button", {
          name: /Build from recipe/
        })
        await action.focus()
        await action.press("Enter")
        const builder = launched.page.getByRole("region", {
          name: "Structured recipe builder"
        })
        const source = builder.getByRole("combobox", { name: "Recipe source" })
        await expect(
          source.locator('optgroup[label="Starters"] option')
        ).toHaveCount(4)
        await source.selectOption({ label: "Research and analysis" })
        await builder
          .getByRole("textbox", {
            name: "Current value for Research question (not saved)"
          })
          .fill("Extension research question")
        await expect(
          builder.getByRole("textbox", { name: "Compiled prompt preview" })
        ).toContainText("Extension research question")
        await waitForRecipeDrawerContained(launched.page)
        await expectNoHorizontalOverflow(launched.page)
        const theme = await launched.page.evaluate(() =>
          document.documentElement.classList.contains("dark") ? "dark" : "light"
        )
        expect(
          visitedThemes.has(theme),
          `theme ${theme} was already checked`
        ).toBe(false)
        visitedThemes.add(theme)
        await launched.page.evaluate(AXE_SOURCE)
        const axe = await builder.evaluate(async (root) => {
          const axeApi = (
            window as unknown as {
              axe: {
                run: (
                  target: Element,
                  options: unknown
                ) => Promise<{ violations: unknown[] }>
              }
            }
          ).axe
          return axeApi.run(root, {
            resultTypes: ["violations"],
            runOnly: {
              type: "tag",
              values: ["wcag2a", "wcag2aa", "wcag21a", "wcag21aa"]
            }
          })
        })
        expect(
          axe.violations,
          `${theme} recipe accessibility violations`
        ).toEqual([])
        compiled = await builder
          .getByRole("textbox", { name: "Compiled prompt preview" })
          .inputValue()
        if (index === 0) {
          await launched.page.keyboard.press("Escape")
          await expect(builder).not.toBeVisible()
          await expect(trigger).toBeFocused()
          const nextTheme = theme === "dark" ? "light" : "dark"
          await launched.page.evaluate(
            (value) =>
              new Promise<void>((resolve) => {
                localStorage.setItem("theme", value)
                chrome.storage.local.set({ theme: value }, () => resolve())
              }),
            nextTheme
          )
          await launched.page.reload({ waitUntil: "domcontentloaded" })
          await expect(launched.input).toBeVisible()
          await expect
            .poll(() =>
              launched.page.evaluate(() =>
                document.documentElement.classList.contains("dark")
                  ? "dark"
                  : "light"
              )
            )
            .toBe(nextTheme)
        } else {
          await builder
            .getByRole("button", { name: "Apply to user message" })
            .click()
        }
      }
      expect(visitedThemes).toEqual(new Set(["dark", "light"]))
      await expect(launched.input).toHaveValue(compiled)
    } finally {
      await context?.close()
      await stopRecipeMock(mock)
    }
  })

  test("sidepanel mixed-version offline mode keeps v3 quarantined and local v2 apply available", async () => {
    test.setTimeout(120_000)
    const mock = await startRecipeMock()
    let context: BrowserContext | null = null
    try {
      const launched = await launchSurface(mock, "sidepanel")
      context = launched.context
      await seedMixedVersions(launched.page)
      const futureBefore = await readPrompt(
        launched.page,
        "extension-future-v3"
      )
      await launched.input.fill("Offline extension draft")
      const builder = await openRecipe(launched.page)
      await launched.page.evaluate(
        () =>
          new Promise<void>((resolve) =>
            chrome.storage.local.set({ __tldw_allow_offline: false }, () =>
              resolve()
            )
          )
      )
      await forceConnectionState(
        launched.page,
        {
          phase: "error",
          isConnected: false,
          isChecking: false,
          offlineBypass: false,
          errorKind: "unreachable",
          knowledgeStatus: "offline"
        },
        "single-text-recipe:offline"
      )
      const source = builder.getByRole("combobox", { name: "Recipe source" })
      await expect(
        source.locator('optgroup[label="Saved recipes"] option')
      ).toHaveCount(1)
      await expect(source).toContainText("Extension known v2")
      await expect(source).not.toContainText("EXTENSION_FUTURE_V3")
      await expect(builder).toContainText(
        "Recipe saving is unavailable offline"
      )
      await source.selectOption({ label: "Extension known v2" })
      await fillTask(builder, "Offline mixed-version task")
      await expect(
        builder.getByRole("button", { name: "Save as new recipe" })
      ).toBeDisabled()
      const compiled = await builder
        .getByRole("textbox", { name: "Compiled prompt preview" })
        .inputValue()
      await builder
        .getByRole("button", { name: "Apply to user message" })
        .click()
      await expect(launched.input).toHaveValue(compiled)
      expect(await readPrompt(launched.page, "extension-future-v3")).toEqual(
        futureBefore
      )
      expect(mock.creates()).toHaveLength(0)
      expect(mock.updates()).toHaveLength(0)
    } finally {
      await context?.close()
      await stopRecipeMock(mock)
    }
  })

  for (const mode of ["old", "unknown"] as const) {
    test(`options chat ${mode} server disables persistence but preserves local built-in apply`, async () => {
      test.setTimeout(120_000)
      const mock = await startRecipeMock(mode)
      let context: BrowserContext | null = null
      try {
        const launched = await launchSurface(mock, "options")
        context = launched.context
        await launched.input.fill(`${mode} extension draft`)
        const builder = await openRecipe(launched.page)
        await fillTask(builder, `${mode} extension local task`)
        await expect(
          builder.getByRole("button", { name: "Save as new recipe" })
        ).toBeDisabled()
        await expect(builder).toContainText(
          mode === "old"
            ? "This server does not support recipe saving yet"
            : "server capabilities could not be confirmed"
        )
        const compiled = await builder
          .getByRole("textbox", { name: "Compiled prompt preview" })
          .inputValue()
        await builder
          .getByRole("button", { name: "Apply to user message" })
          .click()
        await expect(launched.input).toHaveValue(compiled)
        expect(mock.creates()).toHaveLength(0)
      } finally {
        await context?.close()
        await stopRecipeMock(mock)
      }
    })
  }
})
