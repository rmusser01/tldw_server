/**
 * Harness self-check: must pass. If this fails, the reproductions in this
 * project are not trustworthy (seeding, auth or the backend wiring is broken).
 */
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import { NotesPage } from "../utils/page-objects"
import { generateTestId } from "../utils/helpers"
import { createSeedApi, notesTotal, seedNotes, warmBackendOnce } from "../utils/seed-api"

test.describe("UX regression harness", () => {
  test("seeds notes through the real API and the web UI finds them", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const token = generateTestId("uxr-harness")

    const before = await notesTotal(api)
    const seeded = await seedNotes(api, { count: 3, prefix: token })
    expect(await notesTotal(api)).toBe(before + 3)

    const notes = new NotesPage(authedPage)
    await notes.goto()
    await notes.searchNotes(token)
    await notes.assertNoteVisible(seeded[2].title)
  })
})
