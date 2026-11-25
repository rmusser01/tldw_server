/**
 * Notes Workflow E2E Tests (Tier-1 Critical)
 *
 * Tests the core notes workflow:
 * - Page loads and renders correctly
 * - Create a note (fires POST /api/v1/notes)
 * - Search notes filters the list
 * - Delete a note (fires DELETE /api/v1/notes/{id})
 */
import { test, expect, skipIfServerUnavailable, assertNoCriticalErrors } from "../../utils/fixtures"
import { expectApiCall } from "../../utils/api-assertions"
import { NotesPage } from "../../utils/page-objects"
import { generateTestId } from "../../utils/helpers"

test.describe("Notes", () => {
  let notes: NotesPage

  test.beforeEach(async ({ authedPage, serverInfo }) => {
    skipIfServerUnavailable(serverInfo)
    notes = new NotesPage(authedPage)
    // Notes verifies the signed-in user before requesting the list.
    const [listCall] = await Promise.all([
      expectApiCall(authedPage, { method: "GET", url: /\/api\/v1\/notes\/?(?:\?|$)/ }, 30_000),
      notes.goto(),
    ])
    expect(listCall.response.ok()).toBe(true)
  })

  test("page loads with expected elements", async ({ diagnostics }) => {
    await notes.assertPageReady()
    await assertNoCriticalErrors(diagnostics)
  })

  test("create note fires API and shows result", async ({ authedPage, diagnostics }) => {
    const testTitle = `Test Note ${generateTestId()}`

    const apiCall = expectApiCall(authedPage, {
      method: "POST",
      url: "/api/v1/notes",
    })

    await notes.createNote({ title: testTitle, content: "Test content body" })

    const { response } = await apiCall
    expect(response.status()).toBeLessThan(400)

    await notes.assertNoteVisible(testTitle)
    await assertNoCriticalErrors(diagnostics)
  })

  test("search notes filters results", async ({ diagnostics }) => {
    const testTitle = `Searchable ${generateTestId()}`
    await notes.createNote({ title: testTitle, content: "Unique content for search" })

    await notes.searchNotes(testTitle)

    await notes.assertNoteVisible(testTitle)
    await assertNoCriticalErrors(diagnostics)
  })

  test("delete note fires API and removes from list", async ({ authedPage, diagnostics }) => {
    const testTitle = `Delete Me ${generateTestId()}`
    await notes.createNote({ title: testTitle, content: "To be deleted" })
    await notes.assertNoteVisible(testTitle)

    const apiCall = expectApiCall(authedPage, {
      method: "DELETE",
      url: "/api/v1/notes",
    })

    await notes.deleteNote(testTitle)

    const { response } = await apiCall
    expect(response.status()).toBeLessThan(400)

    await notes.assertNoteNotVisible(testTitle)
    await assertNoCriticalErrors(diagnostics)
  })
})
