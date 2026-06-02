import { test, expect } from "./fixtures"
import {
  assertNoCriticalDiagnostics,
  captureStep,
  connectSingleUser,
  openFirstRunSetup,
  sendFirstChat,
} from "./helpers"

const serverUrl = process.env.TLDW_SERVER_URL || "http://127.0.0.1:8000"
const apiKey = process.env.TLDW_API_KEY || "THIS-IS-A-SECURE-KEY-123-UAT"

test.describe("Onboarding UAT setup to first chat", () => {
  test("hosted-openai-first-chat completes setup and receives a mock-backed assistant response", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "hosted-openai-first-chat"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-setup-open")

    await connectSingleUser(page, { serverUrl, apiKey })
    await captureStep(page, artifact, scenarioId, "02-setup-connected")

    const responseText = await sendFirstChat(
      page,
      'Say "onboarding UAT ready" and one short sentence.'
    )
    expect(responseText).toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "03-first-chat-success", {
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
    test.skip(
      process.env.TLDW_ONBOARDING_UAT_LOCAL_SUPPORTED !== "1",
      "Current UI lacks peer local provider setup; PR4 will expand this path."
    )

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-local-setup-open")

    await expect(page.getByText(/local|ollama|openai-compatible/i).first()).toBeVisible()
    await captureStep(page, artifact, scenarioId, "02-local-provider-visible")

    assertNoCriticalDiagnostics(diagnostics)
  })
})
