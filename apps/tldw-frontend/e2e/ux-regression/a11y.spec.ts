/**
 * Accessibility ratchet for /notes, /chat and the extension side panel (UX
 * review 2026-10, #3125).
 *
 * axe scans six page states, each in the dark and the light theme, and
 * compares serious and critical violations, per rule, with
 * baselines/a11y.json. A new rule, or more nodes for a baselined rule, is a
 * regression. A baselined rule that disappeared, or has fewer nodes, fails
 * until the baseline is updated. See README.md.
 *
 * The pages run at 1280x720 and the side panel at 420x900. The theme comes
 * from the stored preference (THEME_SETTING, mirrored to localStorage
 * "theme"), set before each load; with none the app starts dark. The light
 * theme was unscanned until #3124's follow-up, which let light-only contrast
 * failures (status and primary text on their own /10 tints) ship unseen.
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import type { BrowserContext, Page, TestInfo } from "@playwright/test"
import { skipIfServerUnavailable } from "../utils/fixtures"
// Extends the shared fixtures with `extension`, which only the side-panel
// test asks for.
import { test, expect } from "../utils/extension-sidepanel"
import {
  UX_RATCHET_AXE_TAGS,
  blockingViolations,
  scanA11y,
  summariseA11yViolations,
} from "../utils/a11y"
import {
  assertNoRatchetProblems,
  compareRatchet,
  ratchetEntryProblems,
  type RatchetEntry,
} from "../utils/ratchet"
import { createSeedApi, notesTotal, seedNotes, warmBackendOnce, type SeedApi } from "../utils/seed-api"
import { openUxRoute, primeUxRoutes, UX_ROUTE_TIMEOUT_MS } from "./ux-routes"

const BASELINE_FILE = "e2e/ux-regression/baselines/a11y.json"
const POPULATED_NOTES = 25
const NOTE_EDITOR_PLACEHOLDER = "Write your note here... (Markdown supported)"

type A11yState =
  | "notes-empty"
  | "notes-populated"
  | "notes-editor"
  | "chat-fresh"
  | "chat-model-picker"
  | "sidepanel-chat"

type A11yTheme = "dark" | "light"

/** Dark first: it is the default, and a failing light scan still reports both. */
const THEMES: readonly A11yTheme[] = ["dark", "light"]

type A11yBaselineEntry = Omit<RatchetEntry, "key" | "count"> & {
  state: A11yState
  theme: A11yTheme
  rule: string
  impact: "serious" | "critical"
  nodes: number
}

type A11yBaseline = { $comment?: string; entries: A11yBaselineEntry[] }

const STATE_LABELS: Record<A11yState, string> = {
  "notes-empty": "/notes (empty library)",
  "notes-populated": "/notes (populated list)",
  "notes-editor": "/notes (note open in the editor)",
  "chat-fresh": "/chat (fresh load)",
  "chat-model-picker": "/chat (model picker open)",
  "sidepanel-chat": "extension side panel /chat",
}

function loadBaseline(): A11yBaseline {
  const file = path.join(__dirname, "baselines", "a11y.json")
  const baseline = JSON.parse(readFileSync(file, "utf8")) as A11yBaseline
  const problems = baseline.entries.flatMap((entry, index) => [
    ...ratchetEntryProblems({ ...entry, key: entry.rule, count: entry.nodes }, `${BASELINE_FILE} entries[${index}]`),
    ...(entry.state in STATE_LABELS ? [] : [`${BASELINE_FILE} entries[${index}]: unknown state "${entry.state}"`]),
    ...(THEMES.includes(entry.theme) ? [] : [`${BASELINE_FILE} entries[${index}]: theme must be "dark" or "light"`]),
  ])
  if (problems.length > 0) throw new Error(`Invalid baseline:\n- ${problems.join("\n- ")}`)
  return baseline
}

/** Top the library up to POPULATED_NOTES so the list renders the same rows on every run. */
async function ensurePopulatedLibrary(api: SeedApi): Promise<void> {
  const existing = await notesTotal(api)
  if (existing < POPULATED_NOTES) {
    await seedNotes(api, { count: POPULATED_NOTES - existing, prefix: "uxr-a11y" })
  }
}

/** Store `theme` as the preference every later load in this page or context reads. */
async function preferTheme(target: Page | BrowserContext, theme: A11yTheme): Promise<void> {
  await target.addInitScript((value) => {
    try {
      window.localStorage.setItem("theme", value)
    } catch {
      // A page without storage keeps the default theme; expectTheme reports it.
    }
  }, theme)
}

async function expectTheme(page: Page, theme: A11yTheme): Promise<void> {
  await expect(page.locator("html"), `the page did not load in the ${theme} theme`).toHaveClass(
    new RegExp(`(^|\\s)${theme}(\\s|$)`)
  )
}

/** Every difference from the baseline for one state in one theme. */
async function a11yProblems(page: Page, state: A11yState, theme: A11yTheme, testInfo: TestInfo): Promise<string[]> {
  await expectTheme(page, theme)
  const violations = blockingViolations(
    await scanA11y(page, { tags: UX_RATCHET_AXE_TAGS, exclude: ["nextjs-portal"] })
  )
  const observed = summariseA11yViolations(violations)
  await testInfo.attach(`a11y-${state}-${theme}.json`, {
    body: JSON.stringify(
      Object.fromEntries(summariseA11yViolations(violations, { sampleSize: Infinity, includeHtml: true })),
      null,
      2
    ),
    contentType: "application/json",
  })

  const baseline = loadBaseline().entries.filter((entry) => entry.state === state && entry.theme === theme)
  return compareRatchet({
    scope: `${STATE_LABELS[state]}, ${theme} theme`,
    subject: "axe violation",
    baselineFile: BASELINE_FILE,
    observed,
    baseline: baseline.map((entry) => ({ ...entry, key: entry.rule, count: entry.nodes })),
  })
}

/**
 * Load `state` with `open` once per theme and compare each scan with the
 * baseline. Every theme is scanned before anything fails, so one run reports
 * both.
 */
async function expectA11yMatchesBaseline(
  page: Page,
  state: A11yState,
  testInfo: TestInfo,
  open: () => Promise<void>
): Promise<void> {
  const problems: string[] = []
  for (const theme of THEMES) {
    await preferTheme(page, theme)
    await open()
    problems.push(...(await a11yProblems(page, state, theme, testInfo)))
  }
  assertNoRatchetProblems(problems)
}

test.describe("Accessibility ratchet", () => {
  // The empty-library scan must run before anything seeds notes. The runner
  // gives every run a fresh backend and runs files and tests in order on one
  // worker, and this file sorts first in the project.
  test("/notes with an empty library", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    expect(
      await notesTotal(api),
      "The empty-library scan needs a fresh backend: run it through bun run e2e:ux-regression"
    ).toBe(0)

    await primeUxRoutes(authedPage)
    await expectA11yMatchesBaseline(authedPage, "notes-empty", testInfo, async () => {
      await openUxRoute(authedPage, "/notes")
      await expect(authedPage.getByText("No notes yet", { exact: true })).toBeVisible()
    })
  })

  test("/notes with a populated list", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    await ensurePopulatedLibrary(api)
    expect(
      await notesTotal(api),
      `The populated scan expects exactly ${POPULATED_NOTES} notes: run it on a fresh backend`
    ).toBe(POPULATED_NOTES)

    await primeUxRoutes(authedPage)
    await expectA11yMatchesBaseline(authedPage, "notes-populated", testInfo, async () => {
      await openUxRoute(authedPage, "/notes")
      await expect(authedPage.getByText(`Showing 1-20 of ${POPULATED_NOTES}`, { exact: true })).toBeVisible()
    })
  })

  test("/notes with a note open in the editor", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    await ensurePopulatedLibrary(api)
    const [note] = await seedNotes(api, { count: 1, prefix: "uxr-a11y-editor" })

    await primeUxRoutes(authedPage)
    await expectA11yMatchesBaseline(authedPage, "notes-editor", testInfo, async () => {
      await openUxRoute(authedPage, "/notes")
      await authedPage.getByTestId(`notes-open-button-${note.id}`).click()
      const editor = authedPage.getByPlaceholder(NOTE_EDITOR_PLACEHOLDER)
      await expect(editor).toHaveValue(/Seeded by the UX regression harness/, { timeout: UX_ROUTE_TIMEOUT_MS })
      // The text shows before the note finishes loading; until then the editor's
      // toolbar is disabled, and axe skips disabled controls, so scanning early
      // under-counts by a varying amount.
      await expect(authedPage.getByTestId("notes-input-mode-markdown")).toBeEnabled({ timeout: UX_ROUTE_TIMEOUT_MS })
    })
  })

  test("/chat on a fresh load", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(createSeedApi(request))
    await primeUxRoutes(authedPage)
    await expectA11yMatchesBaseline(authedPage, "chat-fresh", testInfo, async () => {
      await openUxRoute(authedPage, "/chat")
    })
  })

  test("/chat with the model picker open", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(createSeedApi(request))
    await primeUxRoutes(authedPage)
    await expectA11yMatchesBaseline(authedPage, "chat-model-picker", testInfo, async () => {
      await openUxRoute(authedPage, "/chat")
      await authedPage.getByTestId("model-selector").first().click()
      await expect(authedPage.getByTestId("model-selector-option").first()).toBeVisible()
    })
  })

  // The acceptance criteria for #3124 cover the side panel too. It loads the
  // built extension (see e2e/utils/extension-sidepanel.ts) at 420px.
  test("extension side panel chat", async ({ extension, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(createSeedApi(request))
    const problems: string[] = []
    const worker = extension.context.serviceWorkers().find((candidate) => candidate.url().startsWith("chrome-extension://"))
    expect(worker, "the extension's service worker").toBeDefined()
    for (const theme of THEMES) {
      // The extension keeps the preference in chrome.storage.local, JSON
      // encoded; a stored value wins over localStorage, so set both.
      await worker!.evaluate((value) => chrome.storage.local.set({ theme: JSON.stringify(value) }), theme)
      await preferTheme(extension.context, theme)
      const panel = await extension.openSidePanel()
      try {
        await expect(panel.locator("#textarea-message")).toBeVisible({ timeout: UX_ROUTE_TIMEOUT_MS })
        problems.push(...(await a11yProblems(panel, "sidepanel-chat", theme, testInfo)))
      } finally {
        await panel.close()
      }
    }
    assertNoRatchetProblems(problems)
  })
})
