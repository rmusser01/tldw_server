/**
 * Notes Studio derived-note workflow.
 *
 * Proves that a user can:
 * - select Markdown text in Notes
 * - open Notes Studio
 * - create a derived note using Cornell + Accented
 * - land on the Studio shell
 * - trigger print export without crashing the frontend
 */
import { test, expect, skipIfServerUnavailable, assertNoCriticalErrors } from "../utils/fixtures"
import { NotesPage } from "../utils/page-objects"
import { generateTestId } from "../utils/helpers"
import { createSeedApi, warmBackendOnce } from "../utils/seed-api"

test.describe("Notes Studio derived note", () => {
  const testTitle = `Studio Derived Note ${generateTestId()}`
  const excerpt = "Cornell style should capture this line."
  const noteContent = [
    "# Study note",
    "",
    "This note includes a Markdown excerpt for Notes Studio.",
    "",
    `- ${excerpt}`,
    "- Accented handwriting should still render cleanly.",
  ].join("\n")

  let notes: NotesPage

  test.beforeEach(async ({ authedPage, serverInfo, request }) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(createSeedApi(request))

    await authedPage.addInitScript(() => {
      const openCalls: unknown[][] = []
      const printWindow = {
        document: {
          open() {},
          write() {},
          close() {},
        },
        focus() {},
        print() {},
      }

      Object.defineProperty(window, "__notesStudioPrintOpenCalls", {
        configurable: true,
        writable: true,
        value: openCalls,
      })

      Object.defineProperty(window, "open", {
        configurable: true,
        writable: true,
        value: (...args: unknown[]) => {
          openCalls.push(args)
          return printWindow as Window | null
        },
      })
    })

    notes = new NotesPage(authedPage)
    await notes.goto()
  })

  test("selects markdown, derives a Cornell studio note, and prints it", async ({
    authedPage,
    diagnostics,
  }) => {
    await test.step("Create a Markdown note and select the source excerpt", async () => {
      await notes.createNote({
        title: testTitle,
        content: noteContent,
      })
      await notes.selectMarkdownExcerpt(excerpt)
    })

    await test.step("Open Notes Studio and create a derived Cornell + Accented note", async () => {
      await notes.openNotesStudio()
      await notes.completeNotesStudioSetup({
        template: "Cornell",
        handwriting: "Accented",
        excerptText: excerpt,
      })
    })

    await test.step("Trigger print export without a crash", async () => {
      await expect(notes.notesStudioView).toBeVisible()
      await notes.triggerPrintExport()

      await expect.poll(
        async () => authedPage.evaluate(() => (window as any).__notesStudioPrintOpenCalls?.length ?? 0),
        { timeout: 10_000 }
      ).toBe(1)
    })

    await assertNoCriticalErrors(diagnostics)
  })
})
