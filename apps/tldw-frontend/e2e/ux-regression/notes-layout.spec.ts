/**
 * Notes layout and onboarding (G13, #3114) from the 2026-10-02 UX review.
 *
 * NO-01: /notes is a viewport-framed app page; neither the window nor the
 * layout's scroll owner can scroll, so the app header stays on screen.
 * NO-N1 / NO-02: the Notes tour reaches its end on a first visit with no note
 * open (the Connections step needs a note, so it is left out) and is recorded
 * as completed.
 */
import type { Page } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import { NotesPage } from "../utils/page-objects"
import { waitForNotesReady } from "./ux-routes"

test.use({ trace: "retain-on-failure" })

const readCompletedTutorials = (page: Page) =>
  page.evaluate(() => {
    try {
      const raw = localStorage.getItem("tldw-tutorials")
      const parsed = raw ? JSON.parse(raw) : null
      return (parsed?.state?.completedTutorials ?? []) as string[]
    } catch {
      return [] as string[]
    }
  })

/**
 * On a viewport-framed route the window never scrolls: WebLayout's
 * `main > div` (overflow-y-auto, header included) is the scroll owner. So
 * NO-01 is about that element and the header's position, not window.scrollY.
 */
const readFrame = (page: Page) =>
  page.evaluate(() => {
    const owner = document.querySelector<HTMLElement>("#main-content > div")
    const header = document.querySelector<HTMLElement>("#main-content header")
    return {
      documentOverflow: document.documentElement.scrollHeight - window.innerHeight,
      windowScrollY: window.scrollY,
      ownerOverflow: owner ? owner.scrollHeight - owner.clientHeight : null,
      ownerScrollTop: owner ? owner.scrollTop : null,
      headerTop: header ? header.getBoundingClientRect().top : null
    }
  })

test.describe("Notes layout and onboarding", () => {
  for (const viewport of [
    { width: 1440, height: 900 },
    { width: 1024, height: 768 }
  ]) {
    test(`NO-01: /notes fits a ${viewport.width}x${viewport.height} window without page scrolling`, async ({
      authedPage,
      serverInfo
    }) => {
      skipIfServerUnavailable(serverInfo)
      await authedPage.setViewportSize(viewport)
      const notes = new NotesPage(authedPage)
      await notes.goto()
      await waitForNotesReady(authedPage)

      const before = await readFrame(authedPage)
      expect(before.documentOverflow).toBeLessThanOrEqual(1)
      expect(before.headerTop, "the app header is rendered").not.toBeNull()
      // The layout's scroll owner has nothing to scroll, so the header cannot
      // be scrolled away inside it.
      expect(before.ownerOverflow, "the layout scroll owner is rendered").not.toBeNull()
      expect(before.ownerOverflow).toBeLessThanOrEqual(1)

      // Wheel over the editor header (it does not scroll itself, so the wheel
      // chains outward). Mouse wheel does not wait for scrolling, so let it
      // settle before reading positions.
      await authedPage.getByTestId("notes-editor-region").hover({ position: { x: 24, y: 12 } })
      await authedPage.mouse.wheel(0, 600)
      await authedPage.waitForTimeout(400)
      const after = await readFrame(authedPage)
      expect(after.headerTop).toBe(before.headerTop)
      expect(after.ownerScrollTop).toBe(0)
      expect(after.windowScrollY).toBe(0)
    })
  }

  test("NO-N1/NO-02: the Notes tour runs to the end and is recorded as completed", async ({
    authedPage,
    serverInfo
  }) => {
    skipIfServerUnavailable(serverInfo)
    await authedPage.setViewportSize({ width: 1440, height: 900 })
    const notes = new NotesPage(authedPage)
    await notes.goto()
    await waitForNotesReady(authedPage)

    // Start the tour from keyboard shortcuts help ("?" > Restart tutorial).
    await authedPage.getByTestId("notes-list-region").click({ position: { x: 4, y: 4 } })
    await authedPage.keyboard.press("?")
    await authedPage.getByTestId("notes-restart-tutorial").click()

    const tooltip = authedPage.locator(".react-joyride__tooltip")
    const primary = tooltip.locator('button[data-action="primary"]')
    for (let step = 0; step < 12; step += 1) {
      await expect(tooltip).toBeVisible({ timeout: 10_000 })
      const label = (await primary.innerText()).trim()
      // The progress label never promises more steps than the tour shows.
      const progress = label.match(/Step (\d+) of (\d+)/)
      if (progress) expect(Number(progress[1])).toBeLessThanOrEqual(Number(progress[2]))
      await primary.click()
      if (/finish/i.test(label)) break
    }

    await expect(tooltip).toHaveCount(0)
    await expect.poll(() => readCompletedTutorials(authedPage)).toContain("notes-basics")
    expect(await authedPage.evaluate(() => window.scrollY)).toBe(0)
  })
})
