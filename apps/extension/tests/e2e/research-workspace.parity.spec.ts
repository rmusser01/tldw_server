import { expect, test, type Page } from "@playwright/test"
import { launchWithBuiltExtensionOrSkip } from "./utils/real-server"
import { forceConnected, waitForConnectionStore } from "./utils/connection"
import { runResearchWorkspaceParityContract } from "../../../test-utils/research-workspace"

const DEFAULT_SERVER_CONFIG = {
  serverUrl: "http://dummy-tldw",
  apiKey: "test-key"
}

const BENIGN_PAGE_ERROR_PATTERNS = [
  /AbortError/i
]

const BENIGN_CONSOLE_ERROR_PATTERNS = [
  /Failed to fetch models from tldw:\s+AbortError/i,
  /Failed to load resource: net::ERR_NAME_NOT_RESOLVED/i,
  /Failed to load resource: net::ERR_FILE_NOT_FOUND/i
]

const BENIGN_REQUEST_FAILURE_PATTERNS = [
  /ERR_NAME_NOT_RESOLVED/i,
  /ERR_FILE_NOT_FOUND/i
]

const isBenignByPattern = (value: string, patterns: RegExp[]): boolean =>
  patterns.some((pattern) => pattern.test(value))

const isBenignRequestFailure = (url: string, errorText: string): boolean => {
  if (
    url.startsWith(DEFAULT_SERVER_CONFIG.serverUrl) &&
    isBenignByPattern(errorText, BENIGN_REQUEST_FAILURE_PATTERNS)
  ) {
    return true
  }

  if (
    url.startsWith("chrome-extension://") &&
    /\/fonts\/.+\.(ttf|woff|woff2)$/i.test(url) &&
    /ERR_FILE_NOT_FOUND/i.test(errorText)
  ) {
    return true
  }

  return false
}

const seedLocalStorage = {
  "playground-tour-completed": "true",
  "tldw-tutorials": JSON.stringify({
    state: {
      completedTutorials: ["playground", "chat", "notes", "media", "settings"],
      seenPromptPages: ["/", "/chat", "/notes", "/media", "/settings", "/playground", "/research-workspace"]
    },
    version: 0
  })
}

const recoverOptionsErrorStateIfNeeded = async (page: Page): Promise<void> => {
  const reloadButton = page.getByRole("button", { name: /Reload Options/i }).first()
  if (await reloadButton.isVisible().catch(() => false)) {
    await reloadButton.click()
    await page.waitForLoadState("networkidle")
  }
}

test.describe("Research Workspace parity (extension)", () => {
  test("passes baseline + deterministic studio parity contract", async () => {
    const pageErrors: string[] = []
    const consoleErrors: string[] = []
    const requestFailures: string[] = []

    const { context, page, optionsUrl } = await launchWithBuiltExtensionOrSkip(test, {
      seedConfig: {
        __tldw_first_run_complete: true,
        tldwConfig: {
          serverUrl: DEFAULT_SERVER_CONFIG.serverUrl,
          authMode: "single-user",
          apiKey: DEFAULT_SERVER_CONFIG.apiKey
        }
      },
      seedLocalStorage
    })

    page.on("pageerror", (error) => {
      if (isBenignByPattern(error.message, BENIGN_PAGE_ERROR_PATTERNS)) return
      pageErrors.push(error.message)
    })
    page.on("console", (message) => {
      if (message.type() !== "error") return
      const text = message.text()
      if (isBenignByPattern(text, BENIGN_CONSOLE_ERROR_PATTERNS)) return
      consoleErrors.push(text)
    })
    page.on("requestfailed", (request) => {
      const url = request.url()
      const errorText = request.failure()?.errorText || "request failed"
      if (isBenignRequestFailure(url, errorText)) {
        return
      }
      requestFailures.push(`${errorText} :: ${url}`)
    })

    try {
      await waitForConnectionStore(page, "workspace-parity-extension")
      await forceConnected(
        page,
        { serverUrl: DEFAULT_SERVER_CONFIG.serverUrl },
        "workspace-parity-extension"
      )
      await recoverOptionsErrorStateIfNeeded(page)

      await runResearchWorkspaceParityContract({
        platform: "extension",
        page,
        optionsUrl
      })

      expect(pageErrors).toEqual([])
      expect(consoleErrors).toEqual([])
      expect(requestFailures).toEqual([])
    } finally {
      await context.close()
    }
  })
})
