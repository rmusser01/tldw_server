/**
 * Request-budget ratchet for /notes and /chat (UX review 2026-10, #3125).
 *
 * Each test loads a warm route, records every API request started in a fixed
 * window after navigation, and compares the result with
 * baselines/request-budget.json:
 *  - the total number of API requests;
 *  - duplicate GETs: the same path and query requested again within
 *    `duplicateWindowMs`;
 *  - polling: an endpoint requested `pollingMinHits` or more times after
 *    `idleFromMs`, when the page has nothing left to load.
 * Anything new, or a count outside its baselined tolerance, fails. So does a
 * baselined offender that is gone, until the baseline is updated. See
 * README.md.
 *
 * Counts come from `next dev`, so React StrictMode double-invokes effects and
 * some duplicates are dev-only; the baseline notes which ones.
 */
import { readFileSync } from "node:fs"
import path from "node:path"
import type { Page, TestInfo } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import {
  analyseRequests,
  serialiseRequestReport,
  trackRequests,
  type RequestBudgetReport,
  type RequestBudgetSettings,
} from "../utils/request-budget"
import {
  assertNoRatchetProblems,
  compareRatchet,
  ratchetEntryProblems,
  type RatchetEntry,
  type RatchetObservation,
} from "../utils/ratchet"
import { createSeedApi, notesTotal, seedNotes, warmBackendOnce } from "../utils/seed-api"
import { openUxRoute, primeUxRoutes } from "./ux-routes"

const BASELINE_FILE = "e2e/ux-regression/baselines/request-budget.json"
const SEEDED_NOTES = 25

type BudgetPage = "notes" | "chat"
type BudgetKind = "total" | "duplicate-get" | "polling"
type BudgetEntry = RatchetEntry & { page: BudgetPage; kind: BudgetKind }
type BudgetBaseline = { $comment?: string; settings: RequestBudgetSettings; entries: BudgetEntry[] }

const KINDS: Record<BudgetKind, string> = {
  total: "API request total",
  "duplicate-get": "duplicate GET",
  polling: "polling endpoint",
}

/** The only key of a "total" entry. */
const TOTAL_KEY = "api-requests"

function loadBaseline(): BudgetBaseline {
  const file = path.join(__dirname, "baselines", "request-budget.json")
  const baseline = JSON.parse(readFileSync(file, "utf8")) as BudgetBaseline
  const problems = baseline.entries.flatMap((entry, index) => {
    const where = `${BASELINE_FILE} entries[${index}]`
    return [
      ...ratchetEntryProblems(entry, where),
      ...(entry.kind in KINDS ? [] : [`${where}: unknown kind "${entry.kind}"`]),
      ...(entry.page === "notes" || entry.page === "chat" ? [] : [`${where}: unknown page "${entry.page}"`]),
      ...(entry.kind === "total" && entry.key !== TOTAL_KEY ? [`${where}: a total's key must be "${TOTAL_KEY}"`] : []),
    ]
  })
  const { windowMs, duplicateWindowMs, idleFromMs, pollingMinHits } = baseline.settings ?? ({} as RequestBudgetSettings)
  if (![windowMs, duplicateWindowMs, idleFromMs, pollingMinHits].every((value) => Number.isInteger(value) && value > 0)) {
    problems.push(`${BASELINE_FILE}: settings need positive integer windowMs, duplicateWindowMs, idleFromMs, pollingMinHits`)
  } else if (idleFromMs >= windowMs) {
    problems.push(`${BASELINE_FILE}: idleFromMs must be inside windowMs`)
  }
  if (problems.length > 0) throw new Error(`Invalid baseline:\n- ${problems.join("\n- ")}`)
  return baseline
}

const topKeys = (report: RequestBudgetReport, limit = 12): string[] =>
  Object.entries(report.perKey)
    .sort(([keyA, a], [keyB, b]) => b - a || keyA.localeCompare(keyB))
    .slice(0, limit)
    .map(([key, count]) => `${count}x ${key}`)

function observedFor(kind: BudgetKind, report: RequestBudgetReport): Map<string, RatchetObservation> {
  if (kind === "duplicate-get") return report.duplicateGets
  if (kind === "polling") return report.polling
  return new Map([[TOTAL_KEY, { count: report.total, details: topKeys(report) }]])
}

/**
 * Prime both routes in this page, then load `route` on a blank document and
 * record the API requests it starts during the settings window.
 */
async function measureRoute(
  page: Page,
  route: "/notes" | "/chat",
  settings: RequestBudgetSettings,
  testInfo: TestInfo
): Promise<RequestBudgetReport> {
  await primeUxRoutes(page)
  const tracker = trackRequests(page)
  await openUxRoute(page, route)
  const readyAt = tracker.elapsed()
  // Idle traffic is only meaningful once the page has finished loading.
  expect(readyAt, `${route} took ${readyAt} ms to become ready; idle starts at ${settings.idleFromMs} ms`).toBeLessThan(
    settings.idleFromMs
  )
  await page.waitForTimeout(Math.max(0, settings.windowMs - tracker.elapsed() + 250))
  const recorded = tracker.stop()
  const report = analyseRequests(recorded, settings)
  await testInfo.attach(`request-budget${route.replace("/", "-")}.json`, {
    body: JSON.stringify({ readyAt, ...serialiseRequestReport(report, recorded) }, null, 2),
    contentType: "application/json",
  })
  return report
}

function expectWithinBudget(page: BudgetPage, report: RequestBudgetReport, baseline: BudgetBaseline): void {
  const scope = `/${page} (${baseline.settings.windowMs / 1000}s after navigation)`
  assertNoRatchetProblems(
    (Object.keys(KINDS) as BudgetKind[]).flatMap((kind) =>
      compareRatchet({
        scope,
        subject: KINDS[kind],
        baselineFile: BASELINE_FILE,
        observed: observedFor(kind, report),
        baseline: baseline.entries.filter((entry) => entry.page === page && entry.kind === kind),
      })
    )
  )
}

test.describe("Request budget ratchet", () => {
  test("/notes stays within its request budget", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const baseline = loadBaseline()
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const existing = await notesTotal(api)
    if (existing < SEEDED_NOTES) {
      await seedNotes(api, { count: SEEDED_NOTES - existing, prefix: "uxr-budget" })
    }

    const report = await measureRoute(authedPage, "/notes", baseline.settings, testInfo)
    expectWithinBudget("notes", report, baseline)
  })

  test("/chat stays within its request budget", async ({ authedPage, serverInfo, request }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const baseline = loadBaseline()
    await warmBackendOnce(createSeedApi(request))

    const report = await measureRoute(authedPage, "/chat", baseline.settings, testInfo)
    expectWithinBudget("chat", report, baseline)
  })
})
