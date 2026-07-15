import {
  expect,
  test,
  type BrowserContext,
  type Page,
} from "@playwright/test"
import JSZip from "jszip"

import {
  mockPowerUserSkillsLibrary,
  mockSkillsBeginnerApi,
} from "../../../tldw-frontend/e2e/utils/skills-fixtures"
import { launchWithBuiltExtension } from "./utils/extension-build"

const SKILLS_PARITY_SERVER_URL = "http://skills-parity.invalid"
const SKILLS_PARITY_API_KEY = "skills-parity-test-key"
const DEFAULT_VIEWPORT = { width: 1280, height: 900 }
const MAX_DIAGNOSTIC_LENGTH = 300

type Diagnostics = {
  pageErrors: string[]
  consoleErrors: string[]
  requestFailures: string[]
  unexpectedApiRequests: string[]
}

const boundDiagnostic = (value: unknown): string =>
  String(value ?? "")
    .replaceAll(SKILLS_PARITY_API_KEY, "[redacted]")
    .replace(
      /(?:file:\/\/)?\/(?:Users|home|private|tmp|var\/folders)\/[^\s"'<>]+/gi,
      "[local-path]",
    )
    .replace(/\s+/g, " ")
    .slice(0, MAX_DIAGNOSTIC_LENGTH)

const requestIdentifier = (method: string, rawUrl: string): string => {
  try {
    const url = new URL(rawUrl)
    return boundDiagnostic(
      `${method.toUpperCase()} ${url.protocol}//${url.host}${url.pathname}`,
    )
  } catch {
    return boundDiagnostic(`${method.toUpperCase()} [invalid-url]`)
  }
}

function captureDiagnostics(page: Page): Diagnostics {
  const diagnostics: Diagnostics = {
    pageErrors: [],
    consoleErrors: [],
    requestFailures: [],
    unexpectedApiRequests: [],
  }

  page.on("pageerror", (error) => {
    diagnostics.pageErrors.push(boundDiagnostic(error.message))
  })
  page.on("console", (message) => {
    if (message.type() !== "error") return
    diagnostics.consoleErrors.push(boundDiagnostic(message.text()))
  })
  page.on("requestfailed", (request) => {
    diagnostics.requestFailures.push(
      boundDiagnostic(
        `${request.failure()?.errorText || "request failed"} :: ${requestIdentifier(
          request.method(),
          request.url(),
        )}`,
      ),
    )
  })

  return diagnostics
}

async function installUnexpectedApiGuard(
  page: Page,
  diagnostics: Diagnostics,
): Promise<void> {
  await page.route(`${SKILLS_PARITY_SERVER_URL}/api/**`, async (route) => {
    diagnostics.unexpectedApiRequests.push(
      requestIdentifier(route.request().method(), route.request().url()),
    )
    await route.fulfill({
      status: 501,
      contentType: "application/json",
      body: JSON.stringify({ detail: "Unhandled Skills parity API request" }),
    })
  })
}

async function installDirectRequestFallback(
  context: BrowserContext,
): Promise<void> {
  await context.addInitScript(() => {
    const patchedRuntimes = new Set<unknown>()
    const patchRuntime = (runtime: any) => {
      if (
        !runtime
        || patchedRuntimes.has(runtime)
        || typeof runtime.sendMessage !== "function"
      ) {
        return
      }
      patchedRuntimes.add(runtime)

      const originalSendMessage = runtime.sendMessage.bind(runtime)
      const sendMessage = (message: any, ...args: any[]) => {
        if (message?.type === "tldw:request") {
          throw new Error(
            "Could not establish connection. Receiving end does not exist.",
          )
        }
        return originalSendMessage(message, ...args)
      }

      let assigned = false
      try {
        runtime.sendMessage = sendMessage
        assigned = runtime.sendMessage === sendMessage
      } catch {
        assigned = false
      }

      if (!assigned) {
        try {
          Object.defineProperty(runtime, "sendMessage", {
            configurable: true,
            writable: true,
            value: sendMessage,
          })
        } catch {
          // The production transport remains authoritative if the runtime is immutable.
        }
      }
    }

    const extensionGlobals = globalThis as any
    patchRuntime(extensionGlobals.browser?.runtime)
    patchRuntime(extensionGlobals.chrome?.runtime)
  })
}

async function mockChatHandoffBootstrap(page: Page): Promise<void> {
  await page.route(
    `${SKILLS_PARITY_SERVER_URL}/api/v1/config/docs-info`,
    async (route) => {
      if (route.request().method() !== "GET") {
        await route.fallback()
        return
      }
      await route.fulfill({
        status: 200,
        contentType: "application/json",
        body: JSON.stringify({ capabilities: {} }),
      })
    },
  )
}

async function launchSkillsParity<TApi>(
  mockApi: (page: Page) => Promise<TApi>,
): Promise<{
  api: TApi
  context: BrowserContext
  diagnostics: Diagnostics
  page: Page
}> {
  let preparedContext: BrowserContext | undefined
  let diagnostics: Diagnostics | undefined
  let api: TApi | undefined

  try {
    const launch = await launchWithBuiltExtension({
      seedConfig: {
        __tldw_first_run_complete: true,
        tldwConfig: {
          serverUrl: SKILLS_PARITY_SERVER_URL,
          authMode: "single-user",
          apiKey: SKILLS_PARITY_API_KEY,
        },
      },
      optionsTarget: "/skills",
      prepareOptionsPage: async ({ context, page }) => {
        preparedContext = context
        await page.setViewportSize(DEFAULT_VIEWPORT)
        diagnostics = captureDiagnostics(page)
        await installUnexpectedApiGuard(page, diagnostics)
        await installDirectRequestFallback(context)
        api = await mockApi(page)
        await mockChatHandoffBootstrap(page)
      },
    })

    if (!diagnostics || api === undefined) {
      throw new Error("Skills parity harness did not finish setup")
    }

    return {
      api,
      context: launch.context,
      diagnostics,
      page: launch.page,
    }
  } catch (error) {
    await preparedContext?.close()
    throw error
  }
}

test.describe("Skills parity (extension)", () => {
  test("completes bootstrap and the beginner journey", async () => {
    test.setTimeout(120_000)

    let extensionContext: BrowserContext | undefined

    try {
      const launch = await launchSkillsParity(mockSkillsBeginnerApi)
      extensionContext = launch.context

      const { api, diagnostics, page } = launch
      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe("#/skills")
      await expect(page.getByRole("heading", { level: 1, name: "Skills" })).toBeVisible()
      await expect(
        page.getByRole("heading", { name: "Start with a reusable skill" }),
      ).toBeVisible()

      await page.getByRole("button", {
        name: "Seed built-ins",
        exact: true,
      }).click()

      await expect.poll(() => api.seedRequests.length).toBe(1)
      expect(api.seedRequests[0]?.searchParams.get("overwrite")).toBe("false")

      const successConfirmation = page.getByRole("status").filter({
        hasText: "Try a built-in skill now, or copy the chat invocation for later.",
      })
      await expect(successConfirmation).toContainText("Built-in skills seeded")
      await expect(
        page.getByRole("row", {
          name: /summarize.*Summarize source material/i,
        }),
      ).toBeVisible()

      await successConfirmation.getByRole("button", { name: "View skill" }).click()
      const details = page.getByRole("dialog", {
        name: "Skill details: summarize",
      })
      await expect(details).toBeVisible()
      await expect(
        details.getByText("Summarize source material", { exact: true }),
      ).toBeVisible()
      await expect(details.getByText("Mode", { exact: true })).toBeVisible()
      await expect(details.getByText("inline", { exact: true })).toBeVisible()

      await details.getByRole("button", { name: "Test run" }).click()
      const testRun = page.getByRole("dialog", { name: "Test run" })
      const argumentsInput = testRun.getByPlaceholder("Enter test arguments...")
      await argumentsInput.fill("A long article about Skills UX")
      await argumentsInput.press("Enter")

      await expect.poll(() => [...api.executeRequests]).toEqual([
        {
          args: "A long article about Skills UX",
          dry_run: true,
        },
      ])
      await expect(testRun.getByText("Dry render", { exact: true })).toBeVisible()

      await testRun.getByRole("button", { name: "Run test" }).click()
      await expect.poll(() => [...api.executeRequests]).toEqual([
        {
          args: "A long article about Skills UX",
          dry_run: true,
        },
        {
          args: "A long article about Skills UX",
          dry_run: false,
        },
      ])
      await expect(
        testRun.getByText("Rendered Prompt", { exact: true }),
      ).toBeVisible()

      await testRun.press("Escape")
      await expect(testRun).toBeHidden()
      await page.getByRole("button", { name: "Use summarize in chat" }).click()

      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe("#/chat")
      await expect(page.locator("#textarea-message")).toHaveValue("/skill summarize")

      expect(diagnostics.pageErrors).toEqual([])
      expect(diagnostics.consoleErrors).toEqual([])
      expect(diagnostics.requestFailures).toEqual([])
      expect(diagnostics.unexpectedApiRequests).toEqual([])
    } finally {
      await extensionContext?.close()
    }
  })

  test("covers the power-user hash and export contract", async () => {
    test.setTimeout(120_000)

    let extensionContext: BrowserContext | undefined

    try {
      const launch = await launchSkillsParity(mockPowerUserSkillsLibrary)
      extensionContext = launch.context

      const { api, diagnostics, page } = launch
      const withToolsHash =
        "#/skills?q=target&mode=fork&tools=with-tools&model=gpt-4.1-mini&sort=name&order=desc&pageSize=20"
      const withoutToolsHash =
        "#/skills?q=target&mode=fork&tools=without-tools&model=gpt-4.1-mini&sort=name&order=desc&pageSize=20"
      const targetSkill = page.getByText("target-research-formatter", {
        exact: true,
      })
      const lastListQuery = () => {
        const params = api.lastListUrl()?.searchParams
        return params
          ? {
              q: params.get("q"),
              context: params.get("context"),
              hasTools: params.get("has_tools"),
              model: params.get("model"),
              sort: params.get("sort"),
              order: params.get("order"),
              limit: params.get("limit"),
              offset: params.get("offset"),
            }
          : null
      }
      const expectedListQuery = {
        q: "target",
        context: "fork",
        hasTools: "true",
        model: "gpt-4.1-mini",
        sort: "name",
        order: "desc",
        limit: "20",
        offset: "0",
      }

      await expect(page.getByText("30 skills", { exact: true })).toBeVisible()
      await expect(page.locator(".ant-pagination").getByTitle("2")).toBeVisible()

      await page.getByLabel("Select archive-helper-01").check()
      await page.getByLabel("Select archive-helper-02").check()
      await expect(page.getByText("2 selected", { exact: true })).toHaveCount(1)

      const downloadedFilenames: string[] = []
      page.on("download", (download) => {
        downloadedFilenames.push(download.suggestedFilename())
      })
      const downloadPromise = page.waitForEvent("download")
      const exportButton = page.getByRole("button", { name: "Export selected" })
      await exportButton.click()
      const download = await downloadPromise

      await expect
        .poll(() => [...api.exportRequests].sort((a, b) => a.name.localeCompare(b.name)))
        .toEqual([
          { method: "GET", name: "archive-helper-01" },
          { method: "GET", name: "archive-helper-02" },
        ])
      await expect(exportButton).toBeEnabled()
      expect(downloadedFilenames).toEqual([download.suggestedFilename()])
      expect(download.suggestedFilename()).toMatch(
        /^skills-export-\d{4}-\d{2}-\d{2}\.zip$/,
      )
      expect(await download.failure()).toBeNull()
      const downloadStream = await download.createReadStream()
      const downloadChunks: Buffer[] = []
      for await (const chunk of downloadStream) {
        downloadChunks.push(Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk))
      }
      const aggregateArchive = await JSZip.loadAsync(Buffer.concat(downloadChunks))
      expect(Object.keys(aggregateArchive.files).sort()).toEqual([
        "archive-helper-01.zip",
        "archive-helper-02.zip",
      ])
      for (const nestedArchiveName of [
        "archive-helper-01.zip",
        "archive-helper-02.zip",
      ]) {
        const nestedArchiveEntry = aggregateArchive.file(nestedArchiveName)
        if (!nestedArchiveEntry) {
          throw new Error(`Missing nested archive ${nestedArchiveName}`)
        }
        const nestedArchive = await JSZip.loadAsync(
          await nestedArchiveEntry.async("uint8array"),
        )
        expect(Object.keys(nestedArchive.files)).toEqual([])
      }
      await expect(page.getByText("2 selected", { exact: true })).toHaveCount(1)

      await expect(page.getByText("10 / page", { exact: true })).toBeVisible()
      await page.getByText("10 / page", { exact: true }).click()
      await page.getByRole("option", { name: "20 / page", exact: true }).click()

      await expect
        .poll(() => api.lastListUrl()?.searchParams.get("limit"))
        .toBe("20")
      await expect.poll(() => page.evaluate(() => window.location.hash)).toContain(
        "pageSize=20",
      )

      await page.getByRole("button", { name: "Filters", exact: true }).click()
      const modelFilter = page.getByLabel("Filter by model")
      await modelFilter.fill("  GPT-4.1-MINI  ")
      await expect
        .poll(() => api.lastListUrl()?.searchParams.get("model"))
        .toBe("GPT-4.1-MINI")
      await expect(targetSkill).toBeVisible()
      await expect(
        page.getByText("batch-cleanup-helper", { exact: true }),
      ).toHaveCount(0)

      await modelFilter.fill("gpt-4.1-mini")
      await expect
        .poll(() => api.lastListUrl()?.searchParams.get("model"))
        .toBe("gpt-4.1-mini")
      await expect(targetSkill).toBeVisible()
      await expect(
        page.getByText("batch-cleanup-helper", { exact: true }),
      ).toHaveCount(0)

      await page.getByRole("button", { name: "Filters (1)", exact: true }).click()
      await page.getByPlaceholder("Search skills...").fill("target")
      await page.getByRole("button", { name: "Filters (1)", exact: true }).click()
      await page.getByLabel("Skill mode filter").click()
      await page.getByTitle("Fork", { exact: true }).click()
      await page.getByLabel("Skill tools filter").click()
      await page.getByTitle("Has tools", { exact: true }).click()
      await page.getByRole("button", { name: "Filters (3)", exact: true }).click()
      await page.getByRole("button", { name: "View options", exact: true }).click()
      await page.getByLabel("Sort by").click()
      await page.getByTitle("Name (Z-A)", { exact: true }).click()
      await page.getByRole("button", { name: "View options", exact: true }).click()

      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe(withToolsHash)
      await expect.poll(lastListQuery).toEqual(expectedListQuery)

      const listRequestsBeforeReload = api.listRequestCount()
      await page.reload({ waitUntil: "domcontentloaded" })
      await expect
        .poll(() => api.listRequestCount())
        .toBeGreaterThan(listRequestsBeforeReload)
      await expect.poll(lastListQuery).toEqual(expectedListQuery)
      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe(withToolsHash)
      await expect(page.getByPlaceholder("Search skills...")).toHaveValue("target")
      await expect(targetSkill).toBeVisible()
      await expect(page.getByText("Mode: Fork", { exact: true })).toBeVisible()
      await expect(page.getByText("Tools: Has tools", { exact: true })).toBeVisible()
      await expect(
        page.getByText("Model: gpt-4.1-mini", { exact: true }),
      ).toBeVisible()
      await page.getByRole("button", { name: "Filters (3)", exact: true }).click()
      await expect(page.getByTitle("Fork", { exact: true })).toBeVisible()
      await expect(page.getByTitle("Has tools", { exact: true })).toBeVisible()
      await expect(page.getByLabel("Filter by model")).toHaveValue("gpt-4.1-mini")
      await page.getByRole("button", { name: "Filters (3)", exact: true }).click()
      await page.getByRole("button", { name: "View options", exact: true }).click()
      await expect(page.getByTitle("Name (Z-A)", { exact: true })).toBeVisible()
      await page.getByRole("button", { name: "View options", exact: true }).click()

      await page.getByRole("button", { name: "Filters (3)", exact: true }).click()
      await page.getByLabel("Skill tools filter").click()
      await page.getByTitle("No tools", { exact: true }).click()
      await page.getByRole("button", { name: "Filters (3)", exact: true }).click()
      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe(withoutToolsHash)
      await expect(targetSkill).toHaveCount(0)

      await page.goBack()
      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe(withToolsHash)
      await expect(targetSkill).toBeVisible()

      await page.goForward()
      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe(withoutToolsHash)
      await expect(targetSkill).toHaveCount(0)

      expect(diagnostics.pageErrors).toEqual([])
      expect(diagnostics.consoleErrors).toEqual([])
      expect(diagnostics.requestFailures).toEqual([])
      expect(diagnostics.unexpectedApiRequests).toEqual([])
    } finally {
      await extensionContext?.close()
    }
  })
})
