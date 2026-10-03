/**
 * Route readiness and priming for the ux-regression ratchets.
 *
 * Two things besides the code under test change what a page renders and
 * requests on load:
 *  - `next dev` compiles a route (and its dynamic imports) on first load;
 *  - the app caches some responses in browser storage, so a browser
 *    context's first visit requests more than later visits.
 * primeUxRoutes() loads /notes and /chat in the test's own page before every
 * measurement. The first call compiles the routes; every call leaves the
 * context in the same "visited before" state, whatever ran earlier in the
 * worker.
 */
import { expect, type Page } from "@playwright/test"
import { waitForConnection } from "../utils/helpers"

export const UX_ROUTE_TIMEOUT_MS = 45_000

export type UxRoute = "/notes" | "/chat"

/** /notes has finished loading its list: the empty state or the paging footer shows. */
export async function waitForNotesReady(page: Page): Promise<void> {
  await expect(page.getByTestId("notes-list-region")).toBeVisible({ timeout: UX_ROUTE_TIMEOUT_MS })
  await expect(page.getByTestId("notes-list-loading")).toHaveCount(0, { timeout: UX_ROUTE_TIMEOUT_MS })
  await expect(
    page.getByText(/^Showing \d+-\d+ of \d+$/).or(page.getByText("No notes yet", { exact: true })).first()
  ).toBeVisible({ timeout: UX_ROUTE_TIMEOUT_MS })
}

/** /chat shows the composer and a model chip that has resolved a model. */
export async function waitForChatReady(page: Page): Promise<void> {
  await expect(page.locator("#textarea-message")).toBeVisible({ timeout: UX_ROUTE_TIMEOUT_MS })
  const chip = page.getByTestId("model-selector").first()
  await expect(chip).toBeVisible({ timeout: UX_ROUTE_TIMEOUT_MS })
  await expect(chip).not.toHaveText(/select a model|loading/i, { timeout: UX_ROUTE_TIMEOUT_MS })
}

const READY: Record<UxRoute, (page: Page) => Promise<void>> = {
  "/notes": waitForNotesReady,
  "/chat": waitForChatReady,
}

/** Navigate to a route and wait until it is ready. */
export async function openUxRoute(page: Page, route: UxRoute, timeoutMs = UX_ROUTE_TIMEOUT_MS): Promise<void> {
  await page.goto(route, { waitUntil: "domcontentloaded", timeout: timeoutMs })
  await waitForConnection(page)
  await READY[route](page)
}

/** Load /notes and /chat once in this page; see the module comment. */
export async function primeUxRoutes(page: Page): Promise<void> {
  for (const route of Object.keys(READY) as UxRoute[]) {
    // A cold `next dev` compile of a route can take well over a minute.
    await openUxRoute(page, route, UX_ROUTE_TIMEOUT_MS * 3)
  }
  await page.goto("about:blank")
}
