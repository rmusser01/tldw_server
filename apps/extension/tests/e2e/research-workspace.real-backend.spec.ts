import { expect, test, type Page } from "@playwright/test"
import { launchWithBuiltExtensionOrSkip, requireRealServerConfig } from "./utils/real-server"
import { logConnectionSnapshot, waitForConnectionStore } from "./utils/connection"
import { grantHostPermission } from "./utils/permissions"
import { runResearchWorkspaceParityContract } from "../../../test-utils/research-workspace"

const shouldSkipHostPermission =
  process.env.TLDW_E2E_SKIP_HOST_PERMISSION !== "0" &&
  process.env.TLDW_E2E_SKIP_HOST_PERMISSION !== "false"

const BENIGN_PAGE_ERROR_PATTERNS = [
  /AbortError/i
]

const BENIGN_CONSOLE_ERROR_PATTERNS = [
  /Failed to fetch models from tldw:\s+AbortError/i,
  /Failed to load resource: net::ERR_FILE_NOT_FOUND/i
]

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

const isBenignByPattern = (value: string, patterns: RegExp[]): boolean =>
  patterns.some((pattern) => pattern.test(value))

const isBenignRequestFailure = (url: string, errorText: string): boolean => {
  if (
    url.startsWith("chrome-extension://") &&
    /\/fonts\/.+\.(ttf|woff|woff2)$/i.test(url) &&
    /ERR_FILE_NOT_FOUND/i.test(errorText)
  ) {
    return true
  }

  return false
}

const normalizeServerUrl = (value: string) =>
  value.match(/^https?:\/\//) ? value.replace(/\/$/, "") : `http://${value}`

const recoverOptionsErrorStateIfNeeded = async (page: Page): Promise<void> => {
  const reloadButton = page.getByRole("button", { name: /Reload Options/i }).first()
  if (await reloadButton.isVisible().catch(() => false)) {
    await reloadButton.click()
    await page.waitForLoadState("networkidle")
  }
}

const waitForConnected = async (page: Page, label: string): Promise<void> => {
  try {
    await page.waitForFunction(
      () => {
        const store = (window as any).__tldw_useConnectionStore
        const state = store?.getState?.().state
        return state?.isConnected === true && state?.phase === "connected"
      },
      undefined,
      { timeout: 30_000 }
    )
  } catch (error) {
    await logConnectionSnapshot(page, `${label}-connection-timeout`)
    throw error
  }
}

test.describe("Research Workspace parity (extension real backend)", () => {
  test("passes baseline + deterministic studio parity contract against a real server", async () => {
    const { serverUrl, apiKey } = requireRealServerConfig(test)
    const normalizedServerUrl = normalizeServerUrl(serverUrl)

    const pageErrors: string[] = []
    const consoleErrors: string[] = []
    const requestFailures: string[] = []

    const { context, page, optionsUrl, extensionId } = await launchWithBuiltExtensionOrSkip(
      test,
      {
        seedConfig: {
          __tldw_first_run_complete: true,
          tldwConfig: {
            serverUrl: normalizedServerUrl,
            authMode: "single-user",
            apiKey
          }
        },
        seedLocalStorage
      }
    )

    if (!shouldSkipHostPermission) {
      const origin = new URL(normalizedServerUrl).origin + "/*"
      const granted = await grantHostPermission(context, extensionId, origin)
      if (!granted) {
        test.skip(
          true,
          "Host permission not granted for real-server origin; allow it in chrome://extensions and re-run."
        )
      }
    }

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
      await waitForConnectionStore(page, "workspace-parity-extension-real")
      await recoverOptionsErrorStateIfNeeded(page)
      await waitForConnected(page, "workspace-parity-extension-real")

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
