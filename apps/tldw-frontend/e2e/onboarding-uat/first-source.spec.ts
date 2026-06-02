import { test, expect } from "./fixtures"
import {
  assertNoCriticalDiagnostics,
  captureStep,
  openFirstRunSetup,
  prepareHostedOpenAiFirstChat,
  sendWizardFirstChatAndWaitForMilestone,
} from "./helpers"

test.describe("Onboarding UAT first source milestone", () => {
  test("first-source-after-chat opens Quick Ingest from the guided milestone", async ({
    firstRunPage: page,
    artifact,
    diagnostics,
  }) => {
    const scenarioId = "first-source-after-chat"

    await openFirstRunSetup(page)
    await captureStep(page, artifact, scenarioId, "01-setup-open")

    await prepareHostedOpenAiFirstChat(page)
    await captureStep(page, artifact, scenarioId, "02-first-chat-ready")

    const firstChat = await sendWizardFirstChatAndWaitForMilestone(
      page,
      'Say "onboarding UAT ready" before I continue.'
    )
    expect(firstChat.response_text ?? "").toContain("onboarding UAT ready")
    await captureStep(page, artifact, scenarioId, "03-first-source-milestone", {
      response_text: firstChat.response_text,
    })

    await expect(page.getByRole("radio", { name: /web url/i })).toBeChecked()
    await page.getByText("Paste", { exact: true }).click()
    await captureStep(page, artifact, scenarioId, "04-paste-source-selected")

    await page.getByRole("button", { name: /add source/i }).click()
    await expect(
      page.getByRole("textbox", { name: /pasted text input/i })
    ).toBeVisible({ timeout: 20_000 })

    const quickIngestDetail = await page.evaluate(() => {
      return (
        window as Window & {
          __tldwPendingQuickIngestOpen?: { detail?: Record<string, unknown> }
        }
      ).__tldwPendingQuickIngestOpen?.detail
    })
    expect(quickIngestDetail).toMatchObject({
      source: "first_source_milestone",
      firstSource: true,
      firstSourceKind: "paste_text",
    })
    await captureStep(page, artifact, scenarioId, "05-quick-ingest-open", {
      quick_ingest_detail: quickIngestDetail,
    })

    assertNoCriticalDiagnostics(diagnostics)
  })
})
