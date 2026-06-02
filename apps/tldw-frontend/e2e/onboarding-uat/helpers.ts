import { expect, type Page } from "@playwright/test"
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

export async function connectSingleUser(
  page: Page,
  options: { serverUrl: string; apiKey: string }
): Promise<void> {
  const serverUrlInput = page
    .getByTestId("onboarding-server-url")
    .or(page.getByLabel(/server url|api url|backend url/i))
    .first()
  const apiKeyInput = page
    .getByTestId("onboarding-api-key")
    .or(page.getByLabel(/api key/i))
    .first()

  await expect(serverUrlInput).toBeVisible({ timeout: 30_000 })
  await serverUrlInput.fill(options.serverUrl)
  await expect(apiKeyInput).toBeVisible({ timeout: 30_000 })
  await apiKeyInput.fill(options.apiKey)

  const connectButton = page
    .getByTestId("onboarding-connect")
    .or(page.getByRole("button", { name: /connect|continue|save/i }))
    .first()
  await expect(connectButton).toBeVisible({ timeout: 30_000 })
  await connectButton.click()

  const successScreen = page
    .getByTestId("onboarding-success-screen")
    .or(page.getByText(/connected|setup complete|ready to chat/i))
    .first()
  await expect(successScreen).toBeVisible({ timeout: 60_000 })
}

export async function sendFirstChat(page: Page, prompt: string): Promise<string> {
  const chatPage = new ChatPage(page)
  await chatPage.goto()
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

export function assertNoCriticalDiagnostics(diagnostics: DiagnosticsData): void {
  const pageErrors = diagnostics.pageErrors.filter((error) => !isBenign(error.message))
  const consoleErrors = diagnostics.console.filter(
    (entry) => entry.type === "error" && !isBenign(entry.text)
  )
  const requestFailures = diagnostics.requestFailures.filter(
    (request) => !isBenign(request.url) && !isBenign(request.errorText)
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
