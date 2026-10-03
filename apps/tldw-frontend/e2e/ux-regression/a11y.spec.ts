/**
 * Accessibility ratchet for /notes and /chat (UX review 2026-10, #3125).
 *
 * axe scans five page states and compares serious and critical violations,
 * per rule, with baselines/a11y.json. A new rule, or more nodes for a
 * baselined rule, is a regression. A baselined rule that disappeared, or has
 * fewer nodes, fails until the baseline is updated. See README.md.
 *
 * Runs in the light colour scheme (Playwright's default) at 1280x720.
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import type { Page, TestInfo } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
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

type A11yState = "notes-empty" | "notes-populated" | "notes-editor" | "chat-fresh" | "chat-model-picker"

type A11yBaselineEntry = Omit<RatchetEntry, "key" | "count"> & {
  state: A11yState
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
}

function loadBaseline(): A11yBaseline {
  const file = path.join(__dirname, "baselines", "a11y.json")
  const baseline = JSON.parse(readFileSync(file, "utf8")) as A11yBaseline
  const problems = baseline.entries.flatMap((entry, index) => [
    ...ratchetEntryProblems({ ...entry, key: entry.rule, count: entry.nodes }, `${BASELINE_FILE} entries[${index}]`),
    ...(entry.state in STATE_LABELS ? [] : [`${BASELINE_FILE} entries[${index}]: unknown state "${entry.state}"`]),
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

async function expectA11yMatchesBaseline(page: Page, state: A11yState, testInfo: TestInfo): Promise<void> {
  const violations = blockingViolations(
    await scanA11y(page, { tags: UX_RATCHET_AXE_TAGS, exclude: ["nextjs-portal"] })
  )
  const observed = summariseA11yViolations(violations)
  await testInfo.attach(`a11y-${state}.json`, {
    body: JSON.stringify(
      Object.fromEntries(summariseA11yViolations(violations, { sampleSize: Infinity, includeHtml: true })),
      null,
      2
    ),
    contentType: "application/json",
  })

  const baseline = loadBaseline().entries.filter((entry) => entry.state === state)
  assertNoRatchetProblems(
    compareRatchet({
      scope: STATE_LABELS[state],
      subject: "axe violation",
      baselineFile: BASELINE_FILE,
      observed,
      baseline: baseline.map((entry) => ({ ...entry, key: entry.rule, count: entry.nodes })),
    })
  )
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
    await openUxRoute(authedPage, "/notes")
    await expect(authedPage.getByText("No notes yet", { exact: true })).toBeVisible()
    await expectA11yMatchesBaseline(authedPage, "notes-empty", testInfo)
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
    await openUxRoute(authedPage, "/notes")
    await expect(authedPage.getByText(`Showing 1-20 of ${POPULATED_NOTES}`, { exact: true })).toBeVisible()
    await expectA11yMatchesBaseline(authedPage, "notes-populated", testInfo)
  })

  test("/notes with a note open in the editor", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    await ensurePopulatedLibrary(api)
    const [note] = await seedNotes(api, { count: 1, prefix: "uxr-a11y-editor" })

    await primeUxRoutes(authedPage)
    await openUxRoute(authedPage, "/notes")
    await authedPage.getByTestId(`notes-open-button-${note.id}`).click()
    const editor = authedPage.getByPlaceholder(NOTE_EDITOR_PLACEHOLDER)
    await expect(editor).toHaveValue(/Seeded by the UX regression harness/, { timeout: UX_ROUTE_TIMEOUT_MS })
    await expectA11yMatchesBaseline(authedPage, "notes-editor", testInfo)
  })

  test("/chat on a fresh load", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(createSeedApi(request))
    await primeUxRoutes(authedPage)
    await openUxRoute(authedPage, "/chat")
    await expectA11yMatchesBaseline(authedPage, "chat-fresh", testInfo)
  })

  test("/chat with the model picker open", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(createSeedApi(request))
    await primeUxRoutes(authedPage)
    await openUxRoute(authedPage, "/chat")
    await authedPage.getByTestId("model-selector").first().click()
    await expect(authedPage.getByTestId("model-selector-option").first()).toBeVisible()
    await expectA11yMatchesBaseline(authedPage, "chat-model-picker", testInfo)
  })
})
