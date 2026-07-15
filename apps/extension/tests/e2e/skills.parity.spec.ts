import {
  expect,
  test,
  type BrowserContext,
  type Page,
} from "@playwright/test"

import { mockSkillsBeginnerApi } from "../../../tldw-frontend/e2e/utils/skills-fixtures"
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

type BeginnerApiFixture = Awaited<ReturnType<typeof mockSkillsBeginnerApi>>

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

test.describe("Skills parity (extension)", () => {
  test("completes bootstrap and the beginner journey", async () => {
    test.setTimeout(120_000)

    let extensionContext: BrowserContext | undefined
    let diagnostics: Diagnostics | undefined
    let api: BeginnerApiFixture | undefined

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
          extensionContext = context
          await page.setViewportSize(DEFAULT_VIEWPORT)
          diagnostics = captureDiagnostics(page)
          await installUnexpectedApiGuard(page, diagnostics)
          await installDirectRequestFallback(context)
          api = await mockSkillsBeginnerApi(page)
          await mockChatHandoffBootstrap(page)
        },
      })
      extensionContext = launch.context

      const { page } = launch
      await expect.poll(() => page.evaluate(() => window.location.hash)).toBe("#/skills")
      await expect(page.getByRole("heading", { level: 1, name: "Skills" })).toBeVisible()
      await expect(
        page.getByRole("heading", { name: "Start with a reusable skill" }),
      ).toBeVisible()

      await page.getByRole("button", {
        name: "Seed built-ins",
        exact: true,
      }).click()

      expect(api?.seedRequests).toHaveLength(1)
      expect(api?.seedRequests[0]?.searchParams.get("overwrite")).toBe("false")

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

      await expect.poll(() => [...(api?.executeRequests ?? [])]).toEqual([
        {
          args: "A long article about Skills UX",
          dry_run: true,
        },
      ])
      await expect(testRun.getByText("Dry render", { exact: true })).toBeVisible()

      await testRun.getByRole("button", { name: "Run test" }).click()
      await expect.poll(() => [...(api?.executeRequests ?? [])]).toEqual([
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

      expect(diagnostics?.pageErrors).toEqual([])
      expect(diagnostics?.consoleErrors).toEqual([])
      expect(diagnostics?.requestFailures).toEqual([])
      expect(diagnostics?.unexpectedApiRequests).toEqual([])
    } finally {
      await extensionContext?.close()
    }
  })
})
