import type { Locator, Page } from "@playwright/test"
import { test, expect } from "./fixtures"
import { ChatPage } from "../utils/page-objects"
import { waitForStreamComplete } from "../utils/journey-helpers"
import {
  assertNoCriticalDiagnostics,
  attemptSingleUserConnection,
  captureStep,
  connectSingleUser,
  expectNoUnsafePrimaryDetails,
  openFirstRunSetup,
} from "./helpers"

const serverUrl = process.env.TLDW_SERVER_URL || "http://127.0.0.1:8000"
const apiKey = process.env.TLDW_API_KEY || "THIS-IS-A-SECURE-KEY-123-UAT"
const unreachableSetupEndpoint = "http://127.0.0.1:65535"

async function lastAssistantText(page: Page): Promise<string> {
  const chatPage = new ChatPage(page)
  const messages = await chatPage.getMessages()
  return (
    messages
      .filter((message) => message.role === "assistant")
      .at(-1)
      ?.content?.trim() ?? ""
  )
}

async function sendChatAndWaitForRecoveryBanner(
  page: Page,
  prompt: string
): Promise<Locator> {
  const chatPage = new ChatPage(page)
  await chatPage.goto()
  await chatPage.waitForReady()
  await chatPage.sendMessage(prompt)

  const banner = page.getByTestId("playground-chat-error-banner")
  await expect(banner).toBeVisible({ timeout: 60_000 })
  await expectNoUnsafePrimaryDetails(banner)
  return banner
}

test.describe("Onboarding UAT recovery", () => {
  test("setup-endpoint-recovery keeps the user in setup and recovers after editing the server URL", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "setup-endpoint-recovery"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-setup-open")

    await attemptSingleUserConnection(page, {
      serverUrl: unreachableSetupEndpoint,
      apiKey,
    })

    const panel = page.getByTestId("onboarding-diagnostic-panel")
    await expect(panel).toBeVisible({ timeout: 30_000 })
    await expect(panel).toContainText(/server|connection|url/i)
    await expectNoUnsafePrimaryDetails(panel)
    await expect(page.getByTestId("onboarding-diagnostic-primary-action")).toBeVisible()
    await expect(page.getByTestId("onboarding-diagnostic-secondary-action-retry")).toBeVisible()
    await captureStep(page, artifact, scenarioId, "02-endpoint-diagnostic-visible", {
      failure_category: "refused",
      visible_actions: await panel.getByRole("button").allTextContents(),
    })

    await page.getByTestId("onboarding-diagnostic-primary-action").click()
    await connectSingleUser(page, { serverUrl, apiKey })
    await captureStep(page, artifact, scenarioId, "03-endpoint-recovered")

    assertNoCriticalDiagnostics(diagnostics, {
      expectedEndpointOrigins: [unreachableSetupEndpoint],
      expectedConsoleText: [/API key test failed: Failed to fetch/i],
    })
  })

  test("provider-retry-recovery shows inline first-chat recovery and succeeds after retry", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "provider-retry-recovery"

    await openFirstRunSetup(page)
    await connectSingleUser(page, { serverUrl, apiKey })
    await captureStep(page, artifact, scenarioId, "01-setup-connected")

    const banner = await sendChatAndWaitForRecoveryBanner(
      page,
      'Say "onboarding UAT ready" after a transient provider failure.'
    )
    await expect(page.getByTestId("playground-chat-error-retry")).toBeVisible()
    await expect(page.getByTestId("playground-chat-error-edit-provider")).toBeVisible()
    await expect(page.getByTestId("playground-chat-error-switch-provider")).toBeVisible()
    await expect(page.getByTestId("playground-chat-error-skip")).toHaveCount(0)
    await captureStep(page, artifact, scenarioId, "02-provider-diagnostic-visible", {
      failure_category: "provider_unavailable",
      visible_actions: await banner.getByRole("button").allTextContents(),
    })

    await page.getByTestId("playground-chat-error-retry").click()
    await waitForStreamComplete(page).catch(() => undefined)
    await expect
      .poll(() => lastAssistantText(page), {
        timeout: 60_000,
        message: "Expected retry to recover the first chat",
      })
      .toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "03-provider-retry-succeeded", {
      response_text: await lastAssistantText(page),
    })

    assertNoCriticalDiagnostics(diagnostics)
  })

  test("model-unavailable-recovery shows model/provider actions without raw provider detail", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "model-unavailable-recovery"

    await openFirstRunSetup(page)
    await connectSingleUser(page, { serverUrl, apiKey })
    await captureStep(page, artifact, scenarioId, "01-setup-connected")

    const banner = await sendChatAndWaitForRecoveryBanner(
      page,
      "Trigger the selected model unavailable recovery state."
    )
    await expect(banner).toContainText(/model|provider/i)
    await expect(page.getByTestId("playground-chat-error-retry")).toBeVisible()
    await expect(page.getByTestId("playground-chat-error-edit-provider")).toBeVisible()
    await expect(page.getByTestId("playground-chat-error-switch-provider")).toBeVisible()
    await expect(page.getByTestId("playground-chat-error-skip")).toHaveCount(0)
    await captureStep(page, artifact, scenarioId, "02-model-diagnostic-visible", {
      failure_category: "model_unavailable",
      visible_actions: await banner.getByRole("button").allTextContents(),
    })

    assertNoCriticalDiagnostics(diagnostics)
  })
})
