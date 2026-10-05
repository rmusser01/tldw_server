/**
 * Notes P0 tests from the 2026-10-02 UX review (tracking #3101). NL-01, NS-01
 * and NS-N1 are fixed and guard their fixes with plain assertions. NE-04 is a
 * reproduction that fails loudly once its defect is fixed; see
 * e2e/ux-regression/README.md.
 */
import type { Page, Response } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import { NotesPage } from "../utils/page-objects"
import { generateTestId, seedAuth } from "../utils/helpers"
import { expectKnownDefect } from "../utils/known-defect"
import { createSeedApi, notesTotal, readNote, seedNotes, warmBackendOnce } from "../utils/seed-api"

const NOTE_EDITOR_PLACEHOLDER = "Write your note here... (Markdown supported)"

type PrintFlowEvent =
  | { kind: "open"; features: string; returnedWindow: boolean }
  | { kind: "print"; url: string; text: string }

const isNotePut = (noteId: string) => (response: Response) =>
  response.request().method() === "PUT" && new URL(response.url()).pathname.replace(/\/$/, "") === `/api/v1/notes/${noteId}`

/** Open a seeded note from the list and wait until the editor shows it. */
async function openSeededNote(page: Page, noteId: string) {
  await page.getByTestId(`notes-open-button-${noteId}`).click()
  const editor = page.getByPlaceholder(NOTE_EDITOR_PLACEHOLDER)
  await expect(editor).toHaveValue(/Seeded by the UX regression harness/)
  return editor
}

/** Append text at the end of the Markdown editor, typed like a person. */
async function appendToEditor(page: Page, text: string) {
  const editor = page.getByPlaceholder(NOTE_EDITOR_PLACEHOLDER)
  await editor.click()
  await page.keyboard.press("ControlOrMeta+End")
  await page.keyboard.type(text, { delay: 20 })
  await expect(editor).toHaveValue(new RegExp(text.trim()))
}

test.describe("Notes P0 reproductions", () => {
  test("NL-01: the notes list reports every note in a library larger than 100", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
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

    // NL-01 (#3103, fixed): the list used to be capped at the 100 most recent notes.
    expect(shownTotal).toBe(apiTotal)
  })

  test("NS-01: an edit made just before in-app navigation is saved or the user is asked", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
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

    // NS-01 (#3102, fixed): unsaved note edits used to be discarded on in-app navigation.
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
  })

  test("NS-N1: 'Reload notes' after a save conflict keeps the other tab's change", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const token = generateTestId("uxr-nsn1")
    const [note] = await seedNotes(api, { count: 1, prefix: token })
    const markerB = `tab-b-${token}`
    const markerA = `tab-a-${token}`

    // Tab A loads /notes first (compiling the route); tab B opens the note.
    // Tab A opens the note last, so its freshness check sees version 1 and
    // the conflict surfaces on save rather than as a "Remote changes" prompt.
    const pageA = authedPage
    await new NotesPage(pageA).goto()
    const pageB = await pageA.context().newPage()
    await seedAuth(pageB)
    await new NotesPage(pageB).goto()
    await openSeededNote(pageB, note.id)
    const editorA = await openSeededNote(pageA, note.id)

    // Tab B saves its change: the server moves to version 2.
    await appendToEditor(pageB, ` ${markerB}`)
    await pageB.getByTestId("notes-save-button").click()
    await expect
      .poll(async () => {
        const saved = await readNote(api, note.id)
        return saved.content.includes(markerB) ? saved.version : null
      })
      .toBe(2)
    await pageB.close()

    // Tab A, still on version 1, edits and saves: the server rejects it. The
    // single conflict panel (NS-03) replaced the "Reload notes" toast; its
    // "Use their version" is the reload.
    await appendToEditor(pageA, ` ${markerA}`)
    const conflictSave = pageA.waitForResponse(isNotePut(note.id))
    await pageA.getByTestId("notes-save-button").click()
    expect((await conflictSave).status()).toBe(409)
    const conflictPanel = pageA.getByTestId("notes-save-issue")
    await expect(conflictPanel).toHaveAttribute("data-kind", "conflict")
    const reloadAction = conflictPanel.getByTestId("notes-conflict-take-theirs")
    await expect(reloadAction).toBeVisible()

    // Take the server's version, then give tab A's autosave (5 s debounce)
    // time to run.
    const autosaveAfterReload = pageA
      .waitForResponse(isNotePut(note.id), { timeout: 15_000 })
      .then((response) => response.status(), () => null)
    await reloadAction.click()
    // The unsaved text is copied to the clipboard first. Where the browser
    // refuses the clipboard, the app asks before discarding the text.
    const discardConfirm = pageA
      .getByRole("dialog")
      .filter({ hasText: "Use their version?" })
      .getByRole("button", { name: "Use their version", exact: true })
    await expect
      .poll(async () => (await discardConfirm.isVisible()) || !(await conflictPanel.isVisible()), {
        message: "the reload should finish or ask before discarding the unsaved text",
      })
      .toBe(true)
    if (await discardConfirm.isVisible()) await discardConfirm.click()
    const autosaveStatus = await autosaveAfterReload
    const editorText = await editorA.inputValue()
    const server = await readNote(api, note.id)

    // NS-N1 (#3102, fixed): the reload used to advance the base version without
    // reloading the text, so autosave overwrote the other tab.
    expect(
      {
        editorShowsOtherTabsChange: editorText.includes(markerB),
        serverKeepsOtherTabsChange: server.content.includes(markerB),
      },
      `After 'Reload notes', tab A autosaved (${autosaveStatus ?? "no request"}); ` +
        `server v${server.version} content: ${JSON.stringify(server.content)}`
    ).toEqual({ editorShowsOtherTabsChange: true, serverKeepsOtherTabsChange: true })
  })

  test("NE-04: 'Print / Save as PDF' prints the note without a pop-up error", async ({
    authedPage,
    serverInfo,
    request,
  }, testInfo) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const token = generateTestId("uxr-ne04")
    const [note] = await seedNotes(api, { count: 1, prefix: token })

    // Record window.open() results and print() calls (with the text being
    // printed) from every document in the context: the page, a pop-up or an
    // iframe. Stubbing print() also keeps a real print dialog from opening.
    const context = authedPage.context()
    const opened: Array<{ features: string; returnedWindow: boolean }> = []
    const printed: Array<{ url: string; text: string }> = []
    await context.exposeBinding("__uxrRecord", (_source, event: PrintFlowEvent) => {
      if (event.kind === "open") opened.push({ features: event.features, returnedWindow: event.returnedWindow })
      else printed.push({ url: event.url, text: event.text })
    })
    await context.addInitScript(() => {
      const record = (event: unknown) =>
        void (window as unknown as { __uxrRecord?: (event: unknown) => Promise<void> }).__uxrRecord?.(event)
      const open = window.open.bind(window)
      window.open = (url?: string | URL, target?: string, features?: string) => {
        const win = open(url, target, features)
        record({ kind: "open", features: String(features ?? ""), returnedWindow: win != null })
        return win
      }
      window.print = () => record({ kind: "print", url: location.href, text: document.body?.innerText ?? "" })
    })
    const openedPages: Page[] = []
    context.on("page", (page) => openedPages.push(page))

    const notes = new NotesPage(authedPage)
    await notes.goto()
    await openSeededNote(authedPage, note.id)
    // The recorder must work before its silence can mean anything.
    await authedPage.evaluate(() => window.print())
    await expect.poll(() => printed.length).toBe(1)
    printed.length = 0

    await notes.triggerPrintExport()
    const popupBlockedError = authedPage.getByText(/please allow pop-ups/i)
    let popupBlockedErrorShown = false
    await expect
      .poll(
        async () => {
          if (await popupBlockedError.isVisible().catch(() => false)) popupBlockedErrorShown = true
          return popupBlockedErrorShown || printed.length > 0
        },
        { timeout: 15_000, message: "Print / Save as PDF neither printed nor reported an error" }
      )
      .toBe(true)
    const openedViews = await Promise.all(
      openedPages.map(async (page) => {
        await page.waitForLoadState("domcontentloaded").catch(() => {})
        return { url: page.url(), text: await page.locator("body").innerText().catch(() => "") }
      })
    )
    const printViewShowsNote =
      printed.some((call) => call.text.includes(note.title)) ||
      openedViews.some((view) => view.text.includes(note.title))

    await expectKnownDefect(
      testInfo,
      {
        id: "NE-04",
        issue: 3117,
        summary: "Print / Save as PDF always fails: opens a blank tab and blames pop-up blocking",
      },
      () => {
        expect(
          { popupBlockedError: popupBlockedErrorShown, printViewShowsNote },
          `window.open: ${JSON.stringify(opened)}; print(): ${JSON.stringify(printed)}`
        ).toEqual({ popupBlockedError: false, printViewShowsNote: true })
      }
    )
  })
})
