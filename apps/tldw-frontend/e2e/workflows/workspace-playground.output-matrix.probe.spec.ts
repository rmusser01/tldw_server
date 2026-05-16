import { readFileSync, statSync } from "node:fs"
import path from "node:path"

import {
  test,
  expect,
  skipIfServerUnavailable,
  assertNoCriticalErrors,
} from "../utils/fixtures"
import { expectApiCall } from "../utils/api-assertions"
import {
  seedAuth,
  fetchWithApiKey,
  TEST_CONFIG,
  generateTestId,
} from "../utils/helpers"
import { WorkspacePlaygroundPage } from "../utils/page-objects"

const DESKTOP_VIEWPORT = { width: 1440, height: 900 }

type LiveWorkspaceSource = {
  mediaId: number
  title: string
  type: "document"
  url: string
}

const normalizeWhitespace = (value: string): string =>
  value.replace(/\s+/g, " ").trim()

const seedLiveWorkspaceDocument = async (
  title: string,
  content: string,
): Promise<LiveWorkspaceSource> => {
  const fileName = `${title.toLowerCase().replace(/[^a-z0-9]+/g, "-")}.txt`
  const body = new FormData()
  body.append("media_type", "document")
  body.append("title", title)
  body.append("perform_analysis", "false")
  body.append("perform_chunking", "false")
  body.append("files", new Blob([content], { type: "text/plain" }), fileName)

  const response = await fetchWithApiKey(
    `${TEST_CONFIG.serverUrl}/api/v1/media/add`,
    TEST_CONFIG.apiKey,
    {
      method: "POST",
      body,
    },
  )
  if (!response.ok) {
    throw new Error(
      `Failed to seed output-matrix media "${title}": ${response.status} ${await response.text()}`,
    )
  }

  const payload = await response.json().catch(() => ({}))
  const result = Array.isArray(payload?.results)
    ? payload.results[0]
    : payload?.result || payload
  const mediaId = Number(result?.db_id ?? result?.media_id ?? result?.id)
  if (!Number.isFinite(mediaId) || mediaId <= 0) {
    throw new Error(
      `Output-matrix media seed for "${title}" returned no usable media id: ${JSON.stringify(
        payload,
      )}`,
    )
  }

  const expectedSnippet = normalizeWhitespace(content).slice(0, 48)
  await expect
    .poll(
      async () => {
        const details = await fetchWithApiKey(
          `${TEST_CONFIG.serverUrl}/api/v1/media/${mediaId}?include_content=true&include_versions=false&include_version_content=false`,
          TEST_CONFIG.apiKey,
        )
        if (!details.ok) return ""
        const body = await details.json().catch(() => ({}))
        return normalizeWhitespace(
          String(
            body?.content?.text ??
              body?.content?.content ??
              body?.transcript ??
              "",
          ),
        )
      },
      {
        timeout: 30_000,
        message: `Media ${mediaId} never exposed usable content for the workspace output matrix`,
      },
    )
    .toContain(expectedSnippet)

  return {
    mediaId,
    title,
    type: "document",
    url: `file://${fileName}`,
  }
}

const buildLiveWorkspaceSources = async (): Promise<LiveWorkspaceSource[]> => {
  const fixtureId = generateTestId("workspace-output-matrix")
  return Promise.all([
    seedLiveWorkspaceDocument(
      `WS ${fixtureId} Alpha`,
      `Workspace output matrix alpha briefing for Project Falcon.
January 10, 2026: Pilot rollout launched to two teams.
February 14, 2026: Training completed for 40 operators.
March 20, 2026: Full rollout completed.
Key metrics:
- Rollout completion improved from 68 percent to 80 percent, a gain of 12 percent.
- Customer retention improved from 70 percent to 81 percent, a gain of 11 percent.
Key findings:
- Training cadence and weekly office hours reduced rollout blockers.
- Leaders recommend keeping the training playbook and phased launches.
Token ${fixtureId}-alpha.`,
    ),
    seedLiveWorkspaceDocument(
      `WS ${fixtureId} Beta`,
      `Workspace output matrix beta review for Project Falcon.
January 12, 2026: Baseline survey completed.
February 18, 2026: Onboarding changes shipped.
March 22, 2026: Retention review published.
Key metrics:
- Retention improved from 64 percent to 82 percent, a gain of 18 percent.
- Rollout completion improved from 71 percent to 79 percent, a gain of 8 percent.
Key findings:
- Both teams agree training helped adoption.
- Beta reviewers note retention gains were stronger than rollout gains.
- Reviewers recommend expanding onboarding experiments and measuring long-term retention.
Token ${fixtureId}-beta.`,
    ),
  ])
}

const OUTPUTS = [
  { label: "Audio Summary", validation: "audio_summary" },
  { label: "Summary", validation: "summary" },
  { label: "Report", validation: "report" },
  { label: "Compare Sources", validation: "compare_sources" },
  { label: "Timeline", validation: "timeline" },
  { label: "Data Table", validation: "data_table" },
  { label: "Mind Map", validation: "mindmap" },
  { label: "Slides", validation: "slides" },
  { label: "Quiz", validation: "quiz" },
  { label: "Flashcards", validation: "flashcards" },
] as const

const assertNever = (value: never): never => {
  throw new Error(`Unhandled artifact validation kind: ${String(value)}`)
}

const FAILURE_TEXT_PATTERN =
  /failed to generate|generation failed|error encountered|encountered an error|no usable|download failed|could not/i
const SOURCE_SIGNAL_PATTERN =
  /(rollout|retention|training|onboarding|falcon|12\s*percent|18\s*percent|12%|18%|january|february|march)/i

const hasErrorText = (content: string) => FAILURE_TEXT_PATTERN.test(content)

const getMeaningfulLines = (content: string): string[] =>
  content
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter((line) => line.length > 0)

const readDownloadedText = (artifactPath: string): string =>
  readFileSync(artifactPath, "utf8")

const assertNoFailureText = (content: string, label: string) => {
  expect(content.trim().length, `${label} download should not be empty`).toBeGreaterThan(
    20,
  )
  expect(
    hasErrorText(content),
    `${label} download should not include failure text`,
  ).toBe(false)
}

const validateNarrativeArtifact = (
  label: string,
  content: string,
  options?: { minLength?: number; requireSourceSignal?: boolean },
) => {
  assertNoFailureText(content, label)
  expect(
    content.trim().length,
    `${label} should contain substantive text`,
  ).toBeGreaterThan(options?.minLength ?? 80)
  const sentenceCount = (content.match(/[.!?](?:\s|$)/g) ?? []).length
  expect(
    Math.max(getMeaningfulLines(content).length, sentenceCount),
    `${label} should contain more than a fragment`,
  ).toBeGreaterThan(1)
  if (options?.requireSourceSignal ?? true) {
    expect(
      SOURCE_SIGNAL_PATTERN.test(content),
      `${label} should reference source claims`,
    ).toBe(true)
  }
}

const validateReportArtifact = (content: string) => {
  validateNarrativeArtifact("Report", content, { minLength: 120 })
  const matchedSections = [
    "Executive Summary",
    "Key Findings",
    "Detailed Analysis",
    "Conclusions",
    "Recommendations",
  ].filter((section) => new RegExp(section, "i").test(content))
  expect(
    matchedSections.length,
    "Report should contain multiple expected report sections",
  ).toBeGreaterThanOrEqual(2)
}

const validateCompareSourcesArtifact = (content: string) => {
  validateNarrativeArtifact("Compare Sources", content, { minLength: 120 })
  expect(
    /(agree|disagree|conflict|difference|compare|comparison)/i.test(content),
    "Compare Sources should read like a comparison",
  ).toBe(true)
}

const validateTimelineArtifact = (content: string) => {
  validateNarrativeArtifact("Timeline", content, { minLength: 60 })
  expect(
    /(\bJanuary\b|\bFebruary\b|\bMarch\b|\b2026\b|(^[-*]\s)|(\btimeline\b)|(\bchronolog))/im.test(
      content,
    ),
    "Timeline should contain dated or timeline-like structure",
  ).toBe(true)
}

const validateDataTableArtifact = (content: string) => {
  assertNoFailureText(content, "Data Table")
  const lines = getMeaningfulLines(content)
  expect(
    lines.length,
    "Data Table should include a header and at least one row",
  ).toBeGreaterThanOrEqual(3)
  expect(/^\|.+\|$/.test(lines[0]), "Data Table header should be pipe-delimited").toBe(
    true,
  )
  const headerColumnCount = lines[0].split("|").filter(Boolean).length
  expect(
    headerColumnCount,
    "Data Table should contain multiple columns",
  ).toBeGreaterThanOrEqual(2)
  for (const line of lines.slice(2, 5)) {
    expect(
      line.split("|").filter(Boolean).length,
      "Data Table rows should have the same column count as the header",
    ).toBe(headerColumnCount)
  }
  expect(
    SOURCE_SIGNAL_PATTERN.test(content),
    "Data Table should reference the source claims",
  ).toBe(true)
}

const extractMermaid = (content: string): string => {
  const fenced = content.match(/```(?:mermaid)?\s*([\s\S]*?)```/i)
  return (fenced?.[1] ?? content).trim()
}

const validateMindMapArtifact = (content: string) => {
  assertNoFailureText(content, "Mind Map")
  const mermaid = extractMermaid(content)
  expect(/\bmindmap\b/i.test(mermaid), "Mind Map should be Mermaid mindmap syntax").toBe(
    true,
  )
  expect(
    getMeaningfulLines(mermaid).length,
    "Mind Map should contain multiple Mermaid nodes",
  ).toBeGreaterThanOrEqual(3)
}

const validateSlidesArtifact = (content: string) => {
  assertNoFailureText(content, "Slides")
  expect(/^#\s+/m.test(content), "Slides markdown should contain a title heading").toBe(
    true,
  )
  expect(
    /(^##\s+)|(\bslide\b)/im.test(content),
    "Slides markdown should contain slide sections",
  ).toBe(true)
  expect(
    SOURCE_SIGNAL_PATTERN.test(content),
    "Slides markdown should reference the source claims",
  ).toBe(true)
}

const validateQuizArtifact = (content: string) => {
  assertNoFailureText(content, "Quiz")
  const parsed = JSON.parse(content) as {
    title?: string
    questions?: Array<{
      question?: string
      options?: unknown[]
      answer?: string
    }>
  }
  expect(typeof parsed.title).toBe("string")
  expect(Array.isArray(parsed.questions), "Quiz should export a questions array").toBe(
    true,
  )
  expect(parsed.questions?.length ?? 0, "Quiz should contain at least one question").toBeGreaterThan(
    0,
  )
  const firstQuestion = parsed.questions?.[0]
  expect(typeof firstQuestion?.question).toBe("string")
  expect(Array.isArray(firstQuestion?.options)).toBe(true)
  expect((firstQuestion?.options?.length ?? 0) >= 2).toBe(true)
}

const validateFlashcardsArtifact = (content: string) => {
  assertNoFailureText(content, "Flashcards")
  if (content.trim().startsWith("{") || content.trim().startsWith("[")) {
    const parsed = JSON.parse(content) as Array<Record<string, unknown>> | Record<string, unknown>
    expect(parsed).toBeTruthy()
    return
  }
  expect(/Front:/i.test(content), "Flashcards export should contain card fronts").toBe(true)
  expect(/Back:/i.test(content), "Flashcards export should contain card backs").toBe(true)
}

const validateDownloadedArtifact = (
  output: (typeof OUTPUTS)[number],
  artifactPath: string,
  suggestedFilename: string,
) => {
  const lowerName = suggestedFilename.toLowerCase()
  if (output.validation === "slides" && !lowerName.endsWith(".md")) {
    expect(statSync(artifactPath).size).toBeGreaterThan(32)
    return
  }

  const content = readDownloadedText(artifactPath)
  const validation = output.validation
  switch (validation) {
    case "audio_summary":
      validateNarrativeArtifact("Audio Summary", content, { minLength: 80 })
      return
    case "summary":
      validateNarrativeArtifact("Summary", content, { minLength: 80 })
      return
    case "report":
      validateReportArtifact(content)
      return
    case "compare_sources":
      validateCompareSourcesArtifact(content)
      return
    case "timeline":
      validateTimelineArtifact(content)
      return
    case "data_table":
      validateDataTableArtifact(content)
      return
    case "mindmap":
      validateMindMapArtifact(content)
      return
    case "slides":
      validateSlidesArtifact(content)
      return
    case "quiz":
      validateQuizArtifact(content)
      return
    case "flashcards":
      validateFlashcardsArtifact(content)
      return
    default:
      return assertNever(validation)
  }
}

const disableNextJsPortalPointerInterception = async (
  page: import("@playwright/test").Page,
) => {
  await page.evaluate(() => {
    document.querySelectorAll("nextjs-portal").forEach((portal) => {
      ;(portal as HTMLElement).style.pointerEvents = "none"
    })
  })
}

const activateButton = async (
  page: import("@playwright/test").Page,
  button: import("@playwright/test").Locator,
) => {
  try {
    await button.click({ timeout: 5_000 })
  } catch (error) {
    if (!String(error).includes("nextjs-portal")) {
      throw error
    }
    await button.focus()
    await expect(button).toBeFocused({ timeout: 5_000 })
    await button.press("Enter")
  }
}

const prepareWorkspaceForOutput = async (
  page: import("@playwright/test").Page,
  outputLabel: string,
) => {
  const workspacePage = new WorkspacePlaygroundPage(page)
  const liveSources = await buildLiveWorkspaceSources()

  await workspacePage.goto()
  await workspacePage.waitForReady()
  await workspacePage.resetWorkspace(`Workspace ${generateTestId(outputLabel)}`)
  await workspacePage.seedSources(liveSources)
  await expect
    .poll(async () => (await workspacePage.getSourceIds()).length, {
      timeout: 10_000,
    })
    .toBeGreaterThanOrEqual(2)

  const sourceIds = await workspacePage.getSourceIds()
  await workspacePage.selectSourceById(sourceIds[0])
  await workspacePage.selectSourceById(sourceIds[1])
  await workspacePage.expectSourceSelected(sourceIds[0])
  await workspacePage.expectSourceSelected(sourceIds[1])
  await expect(workspacePage.getStudioArtifactCards()).toHaveCount(0)

  return workspacePage
}

const verifyDownloadedArtifact = async (
  download: import("@playwright/test").Download,
  output: (typeof OUTPUTS)[number],
) => {
  const artifactPath = path.join(
    process.cwd(),
    "test-results",
    download.suggestedFilename(),
  )
  await download.saveAs(artifactPath)
  validateDownloadedArtifact(output, artifactPath, download.suggestedFilename())
}

const generateAndDownloadOutput = async (
  page: import("@playwright/test").Page,
  workspacePage: WorkspacePlaygroundPage,
  output: (typeof OUTPUTS)[number],
) => {
  const cards = workspacePage.getStudioArtifactCards()
  const chatCompletionCall =
    output.label === "Data Table"
      ? expectApiCall(
          page,
          { method: "POST", url: "/api/v1/chat/completions" },
          20_000,
        )
      : null

  const outputButton = workspacePage.getStudioOutputButton(output.label)
  await disableNextJsPortalPointerInterception(page)
  await expect(outputButton).toBeVisible({ timeout: 15_000 })
  await expect(outputButton).toBeEnabled({ timeout: 15_000 })
  await outputButton.scrollIntoViewIfNeeded()
  await activateButton(page, outputButton)

  await expect(cards).toHaveCount(1, { timeout: 90_000 })
  const artifactCard = cards.first()
  await expect(artifactCard).toBeVisible({ timeout: 30_000 })
  await expect(artifactCard).toContainText(output.label, { timeout: 30_000 })

  if (chatCompletionCall) {
    const { request, response } = await chatCompletionCall
    const responseBody = await response.json().catch(() => null)
    if (response.status() !== 200) {
      throw new Error(
        `Data Table chat completion failed with ${response.status()}: ${JSON.stringify({
          requestBody: request.postDataJSON(),
          responseBody,
        })}`,
      )
    }
  }

  await expect
    .poll(
      async () => {
        const downloadVisible = await artifactCard
          .getByRole("button", { name: "Download" })
          .isVisible()
          .catch(() => false)
        if (downloadVisible) {
          return "completed"
        }

        const cardText = (await artifactCard.textContent()) || ""
        if (/failed|encountered an error|no usable/i.test(cardText)) {
          return `failed:${cardText}`
        }

        return "pending"
      },
      {
        timeout: output.label === "Slides" ? 180_000 : 120_000,
        message: `${output.label} did not reach a completed state`,
      },
    )
    .toBe("completed")

  await expect(artifactCard).not.toContainText(/failed|encountered an error/i, {
    timeout: 5_000,
  })

  const downloadButton = artifactCard.getByRole("button", { name: "Download" })
  await expect(downloadButton).toBeVisible({ timeout: 30_000 })
  await disableNextJsPortalPointerInterception(page)
  await downloadButton.scrollIntoViewIfNeeded()

  if (output.label === "Slides") {
    const directSlidesDownload = page
      .waitForEvent("download", { timeout: 3_000 })
      .then((event) => event)
      .catch(() => null)
    await disableNextJsPortalPointerInterception(page)
    await activateButton(page, downloadButton)
    const markdownOption = page.getByRole("button", { name: /Markdown/i }).last()
    const download =
      (await directSlidesDownload) ||
      (await markdownOption.isVisible({ timeout: 3_000 }).catch(() => false)
        ? await (async () => {
            await disableNextJsPortalPointerInterception(page)
            await markdownOption.scrollIntoViewIfNeeded()
            return Promise.all([
              page.waitForEvent("download", { timeout: 30_000 }),
              activateButton(page, markdownOption),
            ]).then(([event]) => event)
          })()
        : null)

    if (!download) {
      throw new Error(
        "Slides export did not trigger a direct download or show a format picker",
      )
    }

    await verifyDownloadedArtifact(download, output)
    return
  }

  const download = await Promise.all([
    page.waitForEvent("download", { timeout: 30_000 }),
    activateButton(page, downloadButton),
  ]).then(([event]) => event)

  await verifyDownloadedArtifact(download, output)
}

test.describe("Research Studio output matrix probe", () => {
  test.beforeEach(async ({ page }) => {
    await seedAuth(page)
    await page.setViewportSize(DESKTOP_VIEWPORT)
  })

  for (const output of OUTPUTS) {
    test(`generates and downloads ${output.label}`, async ({
      authedPage,
      serverInfo,
      diagnostics,
    }) => {
      test.setTimeout(output.label === "Slides" ? 240_000 : 180_000)
      skipIfServerUnavailable(serverInfo)

      const workspacePage = await prepareWorkspaceForOutput(
        authedPage,
        output.label.toLowerCase().replace(/\s+/g, "-"),
      )
      await generateAndDownloadOutput(authedPage, workspacePage, output)
      await assertNoCriticalErrors(diagnostics)
    })
  }
})
