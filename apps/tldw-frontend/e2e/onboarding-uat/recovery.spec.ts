import { test, expect } from "./fixtures"
import {
  advanceWizardDefaultsToFirstChat,
  assertNoCriticalDiagnostics,
  captureStep,
  configureWizardProvider,
  DEFAULT_HOSTED_PROVIDER_MODEL,
  DEFAULT_LOCAL_PROVIDER_MODEL,
  openWizardProviderStep,
  expectNoUnsafePrimaryDetails,
  openFirstRunSetup,
  prepareHostedOpenAiFirstChat,
  saveWizardProviderAndContinue,
  sendWizardFirstChat,
  sendWizardFirstChatAndWaitForMilestone,
  UNREACHABLE_LOCAL_PROVIDER_ENDPOINT,
  validateWizardProvider,
  waitForWizardFirstChatRecovery,
} from "./helpers"

const mockOpenAiUrl = process.env.TLDW_MOCK_OPENAI_URL || "http://127.0.0.1:18112/v1"

test.describe("Onboarding UAT recovery", () => {
  test("setup-endpoint-recovery keeps the user in setup and recovers after editing a local provider endpoint", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "setup-endpoint-recovery"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-setup-open")

    await openWizardProviderStep(page, "local")
    await configureWizardProvider(page, {
      label: "Ollama",
      baseUrl: UNREACHABLE_LOCAL_PROVIDER_ENDPOINT,
      model: DEFAULT_LOCAL_PROVIDER_MODEL,
    })

    const validationFailure = page
      .getByText(/local_provider_unreachable|local provider endpoint is unreachable/i)
      .last()
    await expect(validationFailure).toBeVisible({ timeout: 30_000 })
    await expectNoUnsafePrimaryDetails(validationFailure)
    await expect(page.getByRole("button", { name: /^continue$/i })).toBeDisabled()
    await captureStep(page, artifact, scenarioId, "02-local-endpoint-diagnostic-visible", {
      failure_category: "local_provider_unreachable",
      endpoint_origin: new URL(UNREACHABLE_LOCAL_PROVIDER_ENDPOINT).origin,
    })

    await page.getByLabel(/^ollama base url$/i).fill(mockOpenAiUrl)
    await validateWizardProvider(page, "Ollama")
    await expect(
      page.getByRole("button", { name: DEFAULT_LOCAL_PROVIDER_MODEL })
    ).toBeVisible({ timeout: 30_000 })
    await saveWizardProviderAndContinue(page)
    await advanceWizardDefaultsToFirstChat(page)
    await captureStep(page, artifact, scenarioId, "03-endpoint-recovered-first-chat-ready", {
      provider: "ollama",
      model: DEFAULT_LOCAL_PROVIDER_MODEL,
      endpoint_origin: new URL(mockOpenAiUrl).origin,
    })

    const firstChat = await sendWizardFirstChatAndWaitForMilestone(
      page,
      'Say "onboarding UAT ready" after local endpoint recovery.'
    )
    expect(firstChat.response_text ?? "").toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "04-endpoint-recovered-first-chat-succeeded", {
      response_text: firstChat.response_text,
    })

    assertNoCriticalDiagnostics(diagnostics, {
      expectedEndpointOrigins: [UNREACHABLE_LOCAL_PROVIDER_ENDPOINT],
    })
  })

  test("provider-retry-recovery shows inline first-chat recovery and succeeds after retry", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "provider-retry-recovery"

    await openFirstRunSetup(page)
    await prepareHostedOpenAiFirstChat(page)
    await captureStep(page, artifact, scenarioId, "01-first-chat-ready", {
      provider: "openai",
      model: DEFAULT_HOSTED_PROVIDER_MODEL,
    })

    const failedFirstChat = await sendWizardFirstChat(
      page,
      'Say "onboarding UAT ready" after a transient provider failure.'
    )
    expect(failedFirstChat.status).toBe("failed")
    const banner = await waitForWizardFirstChatRecovery(page)
    await expect(banner.getByRole("button", { name: /^retry$/i })).toBeVisible()
    await expect(banner.getByRole("button", { name: /edit provider/i })).toBeVisible()
    await expect(banner.getByRole("button", { name: /switch provider/i })).toBeVisible()
    await expect(banner.getByRole("button", { name: /skip setup/i })).toBeVisible()
    await captureStep(page, artifact, scenarioId, "02-provider-diagnostic-visible", {
      failure_category: failedFirstChat.failure_category,
      visible_actions: await banner.getByRole("button").allTextContents(),
    })

    const retryResponse = page.waitForResponse(
      (response) =>
        response.url().includes("/api/v1/setup/first-run/first-chat") &&
        response.request().method().toUpperCase() === "POST",
      { timeout: 60_000 }
    )
    await banner.getByRole("button", { name: /^retry$/i }).click()
    const retryPayload = await retryResponse.then((response) =>
      response.json().catch(() => ({}))
    )
    expect(retryPayload.status).toBe("ready")
    await expect(
      page.getByRole("heading", { name: /add your first source/i })
    ).toBeVisible({ timeout: 60_000 })
    await captureStep(page, artifact, scenarioId, "03-provider-retry-succeeded", {
      response_text: retryPayload.response_text,
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
    await prepareHostedOpenAiFirstChat(page)
    await captureStep(page, artifact, scenarioId, "01-first-chat-ready", {
      provider: "openai",
      model: DEFAULT_HOSTED_PROVIDER_MODEL,
    })

    const failedFirstChat = await sendWizardFirstChat(
      page,
      "Trigger the selected model unavailable recovery state."
    )
    expect(failedFirstChat.status).toBe("failed")
    expect(failedFirstChat.failure_category).toBe("model_unavailable")
    const banner = await waitForWizardFirstChatRecovery(page)
    await expect(banner).toContainText(/model/i)
    await expect(banner.getByRole("button", { name: /^retry$/i })).toBeVisible()
    await expect(banner.getByRole("button", { name: /edit provider/i })).toBeVisible()
    await expect(banner.getByRole("button", { name: /switch provider/i })).toBeVisible()
    await expect(banner.getByRole("button", { name: /skip setup/i })).toBeVisible()
    await captureStep(page, artifact, scenarioId, "02-model-diagnostic-visible", {
      failure_category: failedFirstChat.failure_category,
      visible_actions: await banner.getByRole("button").allTextContents(),
    })

    assertNoCriticalDiagnostics(diagnostics)
  })
})
