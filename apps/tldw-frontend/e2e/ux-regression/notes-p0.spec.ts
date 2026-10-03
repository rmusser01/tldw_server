/**
 * Notes P0 reproductions from the 2026-10-02 UX review (tracking #3101).
 * Each test fails loudly once its defect is fixed; see e2e/ux-regression/README.md.
 */
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import { NotesPage } from "../utils/page-objects"
import { generateTestId } from "../utils/helpers"
import { expectKnownDefect } from "../utils/known-defect"
import { createSeedApi, notesTotal, seedNotes, warmBackendOnce } from "../utils/seed-api"

const NOTE_EDITOR_PLACEHOLDER = "Write your note here... (Markdown supported)"

test.describe("Notes P0 reproductions", () => {
  test("NL-01: the notes list reports every note in a library larger than 100", async ({
    authedPage,
    serverInfo,
    request,
  }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const existing = await notesTotal(api)
    if (existing < 105) {
      await seedNotes(api, { count: 105 - existing, prefix: generateTestId("uxr-nl01") })
    }
    const apiTotal = await notesTotal(api)
    expect(apiTotal).toBeGreaterThan(100)

    const notes = new NotesPage(authedPage)
    await notes.goto()
    const footer = authedPage.getByText(/^Showing \d+-\d+ of \d+$/)
    await expect(footer).toBeVisible()
    const shownTotal = Number((await footer.innerText()).match(/of (\d+)$/)?.[1])

    await expectKnownDefect(
      testInfo,
      { id: "NL-01", issue: 3103, summary: "The notes list is capped at the 100 most recent notes" },
      () => {
        expect(shownTotal).toBe(apiTotal)
      }
    )
  })

  test("NS-01: an edit made just before in-app navigation is saved or the user is asked", async ({
    authedPage,
    serverInfo,
    request,
  }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const token = generateTestId("uxr-ns01")
    const [note] = await seedNotes(api, { count: 1, prefix: token })

    // Compile /chat first: under `next dev` an uncompiled route keeps the
    // notes page mounted for seconds, long enough for autosave to hide the
    // defect.
    await authedPage.goto("/chat", { waitUntil: "domcontentloaded" })
    const notes = new NotesPage(authedPage)
    await notes.goto()
    await authedPage.getByTestId(`notes-open-button-${note.id}`).click()
    const editor = authedPage.getByPlaceholder(NOTE_EDITOR_PLACEHOLDER)
    await expect(editor).toHaveValue(/Seeded by the UX regression harness/)

    const marker = `edited-${token}`
    await editor.click()
    await authedPage.keyboard.press("ControlOrMeta+End")
    await authedPage.keyboard.type(` ${marker}`, { delay: 20 })
    await expect(editor).toHaveValue(new RegExp(marker))

    // Leave within the 5 s autosave debounce through client-side navigation,
    // the same path as any in-app link.
    await authedPage.evaluate(() => {
      ;(window as unknown as { next?: { router?: { push: (url: string) => unknown } } }).next?.router?.push("/chat")
    })
    // The scenario is only valid if we left well inside the 5 s debounce.
    await authedPage.waitForURL("**/chat", { timeout: 3_000 })

    await expectKnownDefect(
      testInfo,
      { id: "NS-01", issue: 3102, summary: "Unsaved note edits are discarded on in-app navigation" },
      async () => {
        await expect
          .poll(
            async () => {
              const leaveGuard = authedPage.getByRole("dialog").filter({ hasText: /unsaved|leave|discard/i })
              if (await leaveGuard.isVisible().catch(() => false)) return "asked"
              const response = await api.get(`/api/v1/notes/${note.id}`)
              const saved = await response.json()
              return String(saved?.content ?? "").includes(marker) ? "saved" : "lost"
            },
            { timeout: 10_000 }
          )
          .not.toBe("lost")
      }
    )
  })
})
