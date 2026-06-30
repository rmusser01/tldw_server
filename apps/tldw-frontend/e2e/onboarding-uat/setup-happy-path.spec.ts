import { test, expect } from "./fixtures"
import {
  assertNoCriticalDiagnostics,
  captureStep,
  DEFAULT_HOSTED_PROVIDER_MODEL,
  DEFAULT_LOCAL_PROVIDER_MODEL,
  openFirstRunSetup,
  prepareHostedOpenAiFirstChat,
  prepareLocalOllamaFirstChat,
  sendWizardFirstChatAndWaitForMilestone,
} from "./helpers"

const mockOpenAiUrl = process.env.TLDW_MOCK_OPENAI_URL
if (!mockOpenAiUrl) {
  throw new Error("TLDW_MOCK_OPENAI_URL environment variable is required for UAT tests")
}

test.describe("Onboarding UAT setup to first chat", () => {
  test("hosted-openai-first-chat completes setup and receives a mock-backed assistant response", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "hosted-openai-first-chat"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-setup-open")

    await prepareHostedOpenAiFirstChat(page)
    await captureStep(page, artifact, scenarioId, "02-first-chat-ready", {
      provider: "openai",
      model: DEFAULT_HOSTED_PROVIDER_MODEL,
    })

    const firstChat = await sendWizardFirstChatAndWaitForMilestone(
      page,
      'Say "onboarding UAT ready" and one short sentence.'
    )
    const responseText = firstChat.response_text ?? ""
    expect(responseText).toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "03-first-chat-success-and-source-milestone", {
      response_text: responseText,
    })

    assertNoCriticalDiagnostics(diagnostics)
  })

  test("local-openai-first-chat records current local-provider setup support", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "local-openai-first-chat"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-local-setup-open")

    await prepareLocalOllamaFirstChat(page, {
      baseUrl: mockOpenAiUrl,
      model: DEFAULT_LOCAL_PROVIDER_MODEL,
    })
    await captureStep(page, artifact, scenarioId, "02-local-first-chat-ready", {
      provider: "ollama",
      model: DEFAULT_LOCAL_PROVIDER_MODEL,
      base_url_origin: new URL(mockOpenAiUrl).origin,
    })

    const firstChat = await sendWizardFirstChatAndWaitForMilestone(
      page,
      'Say "onboarding UAT ready" from the local provider path.'
    )
    const responseText = firstChat.response_text ?? ""
    expect(responseText).toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "03-local-first-chat-success", {
      response_text: responseText,
    })

    assertNoCriticalDiagnostics(diagnostics)
  })

  test("local-openai-discovered-model-first-chat completes setup with a discovered model", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "local-openai-discovered-model-first-chat"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-local-setup-open")

    await prepareLocalOllamaFirstChat(page, {
      baseUrl: mockOpenAiUrl,
      model: DEFAULT_LOCAL_PROVIDER_MODEL,
      expectedDiscoveredModel: DEFAULT_LOCAL_PROVIDER_MODEL,
    })
    await captureStep(page, artifact, scenarioId, "02-discovered-model-first-chat-ready", {
      provider: "ollama",
      model: DEFAULT_LOCAL_PROVIDER_MODEL,
      base_url_origin: new URL(mockOpenAiUrl).origin,
    })

    const firstChat = await sendWizardFirstChatAndWaitForMilestone(
      page,
      'Say "onboarding UAT ready" from the discovered local model path.'
    )
    const responseText = firstChat.response_text ?? ""
    expect(responseText).toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "03-discovered-model-first-chat-success", {
      response_text: responseText,
    })

    assertNoCriticalDiagnostics(diagnostics)
  })

  test("local-openai-manual-model-first-chat completes setup when model discovery is unavailable", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "local-openai-manual-model-first-chat"
    const manualModel = "manual-local-uat-chat"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-local-setup-open")

    await prepareLocalOllamaFirstChat(page, {
      baseUrl: mockOpenAiUrl,
      model: manualModel,
      expectedDiscoveredModel: null,
      expectManualModelFallback: true,
    })
    await captureStep(page, artifact, scenarioId, "02-manual-model-first-chat-ready", {
      provider: "ollama",
      model: manualModel,
      base_url_origin: new URL(mockOpenAiUrl).origin,
    })

    const firstChat = await sendWizardFirstChatAndWaitForMilestone(
      page,
      'Say "onboarding UAT ready" from the manual local model path.'
    )
    const responseText = firstChat.response_text ?? ""
    expect(responseText).toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "03-manual-model-first-chat-success", {
      response_text: responseText,
    })

    assertNoCriticalDiagnostics(diagnostics)
  })
})
