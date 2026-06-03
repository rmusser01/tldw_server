import fs from "node:fs"
import path from "node:path"
import type { Page, Route } from "@playwright/test"
import {
  test,
  expect,
  assertNoCriticalErrors,
} from "../utils/fixtures"
import { TEST_CONFIG, dismissConnectionModals } from "../utils/helpers"

type ViewportTarget = {
  label: "desktop" | "mobile"
  width: number
  height: number
}

type OnboardingEvidenceStep = {
  viewport: ViewportTarget["label"]
  step: string
  screenshot: string
  note: string
}

const EVIDENCE_TAG = sanitizeEvidenceTag(
  process.env.TLDW_ONBOARDING_EVIDENCE_TAG || defaultEvidenceTag()
)
const EVIDENCE_DIR = path.resolve(
  process.cwd(),
  `../../Docs/Product/WebUI/evidence/m4_3_onboarding_${EVIDENCE_TAG}`
)

const VIEWPORTS: ViewportTarget[] = [
  { label: "desktop", width: 1440, height: 900 },
  { label: "mobile", width: 375, height: 812 },
]

function sanitizeEvidenceTag(rawTag: string): string {
  const normalized = rawTag.trim().replace(/[^a-zA-Z0-9._-]/g, "_")
  return normalized.length > 0 ? normalized : "local"
}

function defaultEvidenceTag(): string {
  return new Date().toISOString().slice(0, 10).replace(/-/g, "_")
}

function formatEvidenceDate(tag: string): string | null {
  return /^\d{4}_\d{2}_\d{2}$/.test(tag) ? tag.split("_").join("-") : null
}

function ensureEvidenceDirectory(): void {
  fs.mkdirSync(EVIDENCE_DIR, { recursive: true })
}

function writeViewportEvidence(
  viewport: ViewportTarget["label"],
  rows: OnboardingEvidenceStep[]
): void {
  fs.writeFileSync(
    path.join(EVIDENCE_DIR, `${viewport}-onboarding-results.json`),
    `${JSON.stringify(rows, null, 2)}\n`,
    "utf8"
  )
}

function readViewportEvidence(
  viewport: ViewportTarget["label"]
): OnboardingEvidenceStep[] {
  const filePath = path.join(EVIDENCE_DIR, `${viewport}-onboarding-results.json`)
  if (!fs.existsSync(filePath)) return []
  try {
    return JSON.parse(fs.readFileSync(filePath, "utf8")) as OnboardingEvidenceStep[]
  } catch {
    return []
  }
}

function toMarkdownRows(rows: OnboardingEvidenceStep[]): string {
  return rows
    .map(
      (row) =>
        `| ${row.viewport} | ${row.step} | \`${row.screenshot}\` | ${row.note} |`
    )
    .join("\n")
}

function writeEvidenceReadme(): void {
  const desktopRows = readViewportEvidence("desktop")
  const mobileRows = readViewportEvidence("mobile")
  const evidenceDate = formatEvidenceDate(EVIDENCE_TAG)
  const markdown = [
    "# M4.3 Onboarding First-Source Evidence",
    "",
    `Evidence Tag: ${EVIDENCE_TAG}`,
    evidenceDate ? `Date: ${evidenceDate}` : null,
    "",
    "## Desktop (1440x900)",
    "",
    "| Viewport | Step | Screenshot | Notes |",
    "|---|---|---|---|",
    toMarkdownRows(desktopRows),
    "",
    "## Mobile (375x812)",
    "",
    "| Viewport | Step | Screenshot | Notes |",
    "|---|---|---|---|",
    toMarkdownRows(mobileRows),
    "",
  ]
    .filter((line): line is string => line !== null)
    .join("\n")
  fs.writeFileSync(path.join(EVIDENCE_DIR, "README.md"), markdown, "utf8")
}

async function captureStep(
  page: Page,
  rows: OnboardingEvidenceStep[],
  viewport: ViewportTarget["label"],
  step: string,
  note: string
) {
  const screenshot = `${viewport}-${step}.png`
  await page.screenshot({
    path: path.join(EVIDENCE_DIR, screenshot),
    fullPage: true,
  })
  rows.push({
    viewport,
    step,
    screenshot,
    note,
  })
}

async function json(route: Route, body: unknown, status = 200): Promise<void> {
  await route.fulfill({
    status,
    contentType: "application/json",
    headers: {
      "access-control-allow-origin": "*",
      "access-control-allow-headers": "*",
      "access-control-allow-methods": "GET,POST,OPTIONS",
    },
    body: JSON.stringify(body),
  })
}

async function installCompletedFirstRunApi(page: Page): Promise<void> {
  await page.route(/\/api\/v1\/setup\/first-run\/(?:state|metadata)(?:\?.*)?$/, async (route) => {
    const request = route.request()
    const method = request.method().toUpperCase()
    const pathName = new URL(request.url()).pathname

    if (method === "OPTIONS") {
      await route.fulfill({
        status: 204,
        headers: {
          "access-control-allow-origin": "*",
          "access-control-allow-headers": "*",
          "access-control-allow-methods": "GET,POST,OPTIONS",
        },
      })
      return
    }

    if (method !== "GET") {
      await route.fallback()
      return
    }

    if (pathName === "/api/v1/setup/first-run/state") {
      await json(route, {
        status: "completed",
        current_step: null,
        completed_steps: ["first_chat"],
        skipped_steps: [],
        step_data: {},
        acknowledged_steps: ["first_chat"],
        first_chat: {
          completed: true,
          provider: "openai",
          model: "gpt-4.1-mini",
          response_id: "chatcmpl-first-source-e2e",
          completed_at: "2026-06-01T12:00:00Z",
        },
        skip_reason: null,
        created_at: "2026-06-01T12:00:00Z",
        updated_at: "2026-06-01T12:00:00Z",
        completed_at: "2026-06-01T12:00:00Z",
      })
      return
    }

    await json(route, {
      auth_mode: "single_user",
      bundled_single_user_auth_available: true,
      manual_auth_required: false,
      setup_required: false,
      setup_completed: true,
      remote_setup_enabled: false,
      connection: {
        frontend_origin: "http://localhost:8080",
        api_origin: TEST_CONFIG.serverUrl,
        browser_access: "local",
      },
      setup_paths: [],
      multi_user_exit: { guide_path: "/Docs/AuthNZ/Multi_User_Setup.md" },
    })
  })

  await page.route(/\/api\/v1\/media(?:\?.*)?$/, async (route) => {
    if (route.request().method().toUpperCase() !== "GET") {
      await route.fallback()
      return
    }
    await json(route, { items: [], total: 0, page: 1, results_per_page: 1 })
  })
}

async function closeQuickIngestIfOpen(page: Page): Promise<void> {
  const dialog = page.getByRole("dialog").filter({ hasText: /quick ingest/i }).first()
  if (!(await dialog.isVisible().catch(() => false))) return
  const closeButton = dialog.locator(".ant-modal-close").first()
  if (await closeButton.isVisible().catch(() => false)) {
    await closeButton.click()
  } else {
    await page.keyboard.press("Escape")
  }
  await expect(dialog).toBeHidden({ timeout: 10_000 })
}

async function setFirstSourceSessionState(
  page: Page,
  state:
    | { lifecycle: "processing" }
    | {
        lifecycle: "completed"
        resultStatus: "error" | "success"
        firstMediaId?: string | null
        label: string
        errorMessage?: string | null
      }
): Promise<void> {
  await page.evaluate((nextState) => {
    type QuickIngestSessionApi = {
      session?: {
        id: string
        resultSummary: Record<string, unknown>
      } | null
      createDraftSession: (seed: Record<string, unknown>) => {
        id: string
        resultSummary: Record<string, unknown>
      }
      upsertSession: (patch: Record<string, unknown>) => void
    }
    const store = (
      window as Window & {
        __tldw_useQuickIngestSessionStore?: {
          getState?: () => QuickIngestSessionApi
        }
      }
    ).__tldw_useQuickIngestSessionStore
    const api = store?.getState?.()
    if (!api) return
    const openDetail = {
      source: "first_source_milestone",
      preferredPreset: "quick",
      firstSource: true,
      firstSourceKind: "paste_text",
    }
    const session = api.session || api.createDraftSession({
      openDetail,
      firstSourceAddMode: "paste_text",
    })
    api.upsertSession({
      id: session.id,
      openDetail,
      firstSourceAddMode: "paste_text",
      lifecycle: nextState.lifecycle,
      resultSummary:
        nextState.lifecycle === "completed"
          ? {
              status: nextState.resultStatus,
              attemptedAt: Date.now(),
              completedAt: Date.now(),
              totalCount: 1,
              successCount: nextState.resultStatus === "success" ? 1 : 0,
              failedCount: nextState.resultStatus === "error" ? 1 : 0,
              cancelledCount: 0,
              firstMediaId: nextState.firstMediaId || null,
              primarySourceLabel: nextState.label,
              errorMessage: nextState.errorMessage || null,
            }
          : session.resultSummary,
    })
  }, state)
}

test.describe("Onboarding First-Source Journey", () => {
  test.describe.configure({ timeout: 120_000 })

  test.beforeEach(async ({ authedPage }) => {
    ensureEvidenceDirectory()
    await installCompletedFirstRunApi(authedPage)
    await authedPage.addInitScript((cfg) => {
      try {
        localStorage.setItem(
          "tldwConfig",
          JSON.stringify({
            serverUrl: cfg.serverUrl,
            authMode: "single-user",
            apiKey: cfg.apiKey,
          })
        )
      } catch {}
      try {
        localStorage.removeItem("tldw:first-source-milestone-dismissed")
      } catch {}
    }, TEST_CONFIG)
  })

  for (const viewport of VIEWPORTS) {
    test(`guides first source after first-chat completion (${viewport.label})`, async ({
      authedPage,
      diagnostics,
    }) => {
      const evidenceRows: OnboardingEvidenceStep[] = []

      await authedPage.setViewportSize({
        width: viewport.width,
        height: viewport.height,
      })

      await authedPage.goto("/", { waitUntil: "domcontentloaded" })
      await dismissConnectionModals(authedPage)
      await expect(
        authedPage.getByRole("heading", { name: /add your first source/i })
      ).toBeVisible({ timeout: 30_000 })
      await expect(
        authedPage.getByRole("radio", { name: /web url/i })
      ).toBeChecked()
      await expect(
        authedPage.getByRole("button", { name: /ask a question about this source/i })
      ).toHaveCount(0)
      await captureStep(
        authedPage,
        evidenceRows,
        viewport.label,
        "01-first-source-idle",
        "Completed first chat now guides the next milestone: add a first source."
      )

      await authedPage.getByText("Paste", { exact: true }).click()
      await authedPage.getByRole("button", { name: /add source/i }).click()
      await expect(
        authedPage.getByRole("textbox", { name: /pasted text input/i })
      ).toBeVisible({ timeout: 20_000 })
      await captureStep(
        authedPage,
        evidenceRows,
        viewport.label,
        "02-paste-source-entry",
        "Paste text choice opens Quick Ingest directly in paste mode."
      )
      await closeQuickIngestIfOpen(authedPage)

      await setFirstSourceSessionState(authedPage, { lifecycle: "processing" })
      await expect(authedPage.getByText(/processing your source/i)).toBeVisible({
        timeout: 10_000,
      })
      await captureStep(
        authedPage,
        evidenceRows,
        viewport.label,
        "03-source-processing",
        "Prompt remains inline while the first source is processing."
      )

      await setFirstSourceSessionState(authedPage, {
        lifecycle: "completed",
        resultStatus: "error",
        label: "Pasted notes",
        errorMessage: "Upload failed",
      })
      await expect(authedPage.getByText(/upload failed/i)).toBeVisible({
        timeout: 10_000,
      })
      await expect(authedPage.getByRole("button", { name: /retry/i })).toBeVisible()
      await captureStep(
        authedPage,
        evidenceRows,
        viewport.label,
        "04-source-error",
        "Failed first-source ingest offers inline retry without claiming grounded chat is ready."
      )

      await setFirstSourceSessionState(authedPage, {
        lifecycle: "completed",
        resultStatus: "success",
        firstMediaId: "m4-e2e-media-id",
        label: "Pasted notes",
      })
      await expect(authedPage.getByText(/starter questions/i)).toBeVisible({
        timeout: 10_000,
      })
      await expect(
        authedPage.getByRole("button", { name: /summarize this source/i })
      ).toBeVisible({ timeout: 10_000 })
      await captureStep(
        authedPage,
        evidenceRows,
        viewport.label,
        "05-source-ready",
        "Grounded starter questions appear only after a successful first-source media id exists."
      )

      writeViewportEvidence(viewport.label, evidenceRows)
      writeEvidenceReadme()
      await assertNoCriticalErrors(diagnostics)
    })
  }
})
