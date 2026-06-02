import { test, expect } from "./fixtures"
import {
  assertNoCriticalDiagnostics,
  captureStep,
  clickFirstSourceStarterQuestion,
  completeFirstSourcePasteIngest,
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

    const firstSourceSession = await completeFirstSourcePasteIngest(page)
    const firstMediaId = firstSourceSession.resultSummary?.firstMediaId
    expect(firstSourceSession.lifecycle).toBe("completed")
    expect(firstSourceSession.resultSummary?.status).toBe("success")
    expect(firstMediaId).toBeTruthy()
    await expect(page.getByText(/starter questions/i)).toBeVisible()
    await expect(
      page.getByRole("button", { name: "Summarize this source." })
    ).toBeVisible()
    await expect(
      page.getByRole("button", { name: "List the key claims." })
    ).toBeVisible()
    await expect(
      page.getByRole("button", { name: "What should I remember?" })
    ).toBeVisible()
    await captureStep(page, artifact, scenarioId, "06-first-source-ready", {
      first_source_session: firstSourceSession,
    })

    const starterHandoff = await clickFirstSourceStarterQuestion(
      page,
      "Summarize this source."
    )
    expect(starterHandoff).toMatchObject({
      mediaId: firstMediaId,
      mode: "rag_media",
      content: "Summarize this source.",
    })
    expect(starterHandoff.title).toBeTruthy()
    await captureStep(page, artifact, scenarioId, "07-starter-question-handoff", {
      starter_handoff: starterHandoff,
    })

    assertNoCriticalDiagnostics(diagnostics)
  })
})
