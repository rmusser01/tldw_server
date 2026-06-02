import { expect, type Locator, type Page } from "@playwright/test"
import { ChatPage } from "../utils/page-objects"
import {
  isBenign,
  waitForAppShell,
  waitForConnection,
} from "../utils/helpers"
import { waitForStreamComplete } from "../utils/journey-helpers"
import { safeSegment, type DiagnosticsData, type OnboardingArtifact } from "./fixtures"

export async function openFirstRunSetup(page: Page): Promise<void> {
  await page.goto("/", { waitUntil: "domcontentloaded" })
  await waitForAppShell(page)

  const gateOverlay = page.getByTestId("first-run-gate-overlay").first()
  if (await gateOverlay.isVisible({ timeout: 3_000 }).catch(() => false)) {
    await page.getByTestId("first-run-get-started").click()
    await waitForAppShell(page)
    return
  }

  if (!page.url().includes("/setup")) {
    await page.goto("/setup", { waitUntil: "domcontentloaded" })
    await waitForAppShell(page)
  }
}

export async function attemptSingleUserConnection(
  page: Page,
  options: { serverUrl: string; apiKey: string }
): Promise<void> {
  const serverUrlInput = page.getByTestId("onboarding-server-url")
  const apiKeyInput = page.getByTestId("onboarding-api-key")

  await expect(serverUrlInput).toBeVisible({ timeout: 30_000 })
  await serverUrlInput.fill(options.serverUrl)
  await expect(apiKeyInput).toBeVisible({ timeout: 30_000 })
  await apiKeyInput.fill(options.apiKey)

  const connectButton = page.getByTestId("onboarding-connect")
  await expect(connectButton).toBeVisible({ timeout: 30_000 })
  await connectButton.click()
}

export async function connectSingleUser(
  page: Page,
  options: { serverUrl: string; apiKey: string }
): Promise<void> {
  await attemptSingleUserConnection(page, options)

  await waitForSetupConnectionReady(page)
}

async function waitForSetupConnectionReady(page: Page): Promise<void> {
  const successScreen = page.getByTestId("onboarding-success-screen")
  const chatInput = page
    .locator("#textarea-message")
    .or(page.getByTestId("chat-input"))
    .or(page.getByPlaceholder(/type a message/i))
    .first()
  const startChatButton = page.getByRole("button", { name: /start chatting/i }).first()

  await expect
    .poll(
      async () => {
        if (await successScreen.isVisible().catch(() => false)) {
          return "onboarding-success"
        }
        if (
          (await chatInput.isVisible().catch(() => false)) ||
          (await startChatButton.isVisible().catch(() => false))
        ) {
          return "chat-ready"
        }
        return "waiting"
      },
      { timeout: 60_000 }
    )
    .not.toBe("waiting")
}

export async function sendFirstChat(page: Page, prompt: string): Promise<string> {
  const chatPage = new ChatPage(page)
  const currentPath = new URL(page.url()).pathname
  if (currentPath !== "/chat") {
    await chatPage.goto()
  } else {
    await waitForBackendConnection(page)
  }
  await chatPage.waitForReady()
  await chatPage.sendMessage(prompt)
  await waitForStreamComplete(page)
  await chatPage.waitForResponse()

  const messages = await chatPage.getMessages()
  const assistantMessage = messages
    .filter((message) => message.role === "assistant")
    .at(-1)
  const content = assistantMessage?.content?.trim() ?? ""
  expect(content.length).toBeGreaterThan(0)
  return content
}

export async function captureStep(
  page: Page,
  artifact: OnboardingArtifact,
  scenarioId: string,
  stepName: string,
  extra: Record<string, unknown> = {}
): Promise<{ screenshotPath: string; jsonPath: string }> {
  const stem = `${safeSegment(scenarioId)}-${safeSegment(stepName)}`
  const screenshotPath = `${artifact.screenshotsDir}/${stem}.png`
  await page.screenshot({ path: screenshotPath, fullPage: true })
  const jsonPath = artifact.writeJson(`steps/${stem}.json`, {
    scenario_id: scenarioId,
    step_name: stepName,
    url: page.url(),
    captured_at: new Date().toISOString(),
    screenshot_path: screenshotPath,
    ...extra,
  })
  return { screenshotPath, jsonPath }
}

const UNSAFE_PRIMARY_DETAIL_PATTERN =
  /traceback|stack trace|authorization|x-api-key|request headers|\/Users\/|[A-Za-z]:\\|sk-[A-Za-z0-9_-]+/i

export async function expectNoUnsafePrimaryDetails(
  locator: Locator
): Promise<void> {
  await expect(locator).not.toContainText(UNSAFE_PRIMARY_DETAIL_PATTERN)
}

type ConsoleDiagnostic = DiagnosticsData["console"][number]
type DiagnosticsAllowance = {
  expectedEndpointOrigins?: string[]
  expectedConsoleText?: RegExp[]
}

const MODEL_METADATA_ENDPOINT = "/api/v1/llm/models/metadata"
const CHAT_SETTINGS_ENDPOINT_PREFIX = "/api/v1/chats/"
const CHAT_SETTINGS_ENDPOINT_SUFFIX = "/settings?scope_type=global"

function isBenignOnboardingConsoleEntry(entry: ConsoleDiagnostic): boolean {
  if (isBenign(entry.text)) {
    return true
  }

  if (
    /Failed to fetch (?:models from tldw|chat models): Error: Failed to fetch/.test(entry.text) &&
    entry.text.includes(`(GET ${MODEL_METADATA_ENDPOINT})`)
  ) {
    return true
  }

  const locationUrl = entry.location?.url ?? ""
  return (
    /404 \(Not Found\)/.test(entry.text) &&
    locationUrl.includes(CHAT_SETTINGS_ENDPOINT_PREFIX) &&
    locationUrl.includes(CHAT_SETTINGS_ENDPOINT_SUFFIX)
  )
}

const matchesExpectedEndpoint = (
  url: string | undefined,
  expectedEndpointOrigins: string[] = []
): boolean => {
  if (!url) return false
  return expectedEndpointOrigins.some((origin) =>
    url.startsWith(origin.replace(/\/$/, ""))
  )
}

export function assertNoCriticalDiagnostics(
  diagnostics: DiagnosticsData,
  allowance: DiagnosticsAllowance = {}
): void {
  const expectedEndpointOrigins = allowance.expectedEndpointOrigins ?? []
  const expectedConsoleText = allowance.expectedConsoleText ?? []
  const pageErrors = diagnostics.pageErrors.filter((error) => !isBenign(error.message))
  const consoleErrors = diagnostics.console.filter(
    (entry) =>
      entry.type === "error" &&
      !isBenignOnboardingConsoleEntry(entry) &&
      !matchesExpectedEndpoint(entry.location?.url, expectedEndpointOrigins) &&
      !expectedConsoleText.some((pattern) => pattern.test(entry.text))
  )
  const requestFailures = diagnostics.requestFailures.filter(
    (request) =>
      !isBenign(request.url) &&
      !isBenign(request.errorText) &&
      !matchesExpectedEndpoint(request.url, expectedEndpointOrigins)
  )

  if (pageErrors.length || consoleErrors.length || requestFailures.length) {
    throw new Error(
      [
        "Critical onboarding UAT diagnostics detected.",
        `pageErrors=${pageErrors.length}`,
        `consoleErrors=${consoleErrors.length}`,
        `requestFailures=${requestFailures.length}`,
      ].join(" ")
    )
  }
}

export async function waitForBackendConnection(page: Page): Promise<void> {
  await waitForConnection(page, 30_000)
}
