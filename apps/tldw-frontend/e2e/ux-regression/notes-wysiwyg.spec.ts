/**
 * NE-01 (#3102): the Notes WYSIWYG editor must type forward at human speed.
 *
 * The editor used to re-render its contentEditable from state on every input,
 * which moved the caret to the start, so "Hello world" was typed and saved as
 * "dlrow olleH". These are regression tests for the fix.
 *
 * Runs in the ux-regression project against an isolated real backend
 * (bun run e2e:ux-regression -- --grep NE-01). That project and its runner
 * arrive with the UX regression harness (#3125, PR #3136); this spec only
 * needs the shared e2e fixtures, so it does not import the harness's
 * e2e/utils/seed-api.ts and carries its own copy of warmBackendOnce.
 */
import type { APIRequestContext, Page } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import { NotesPage } from "../utils/page-objects"
import { TEST_CONFIG, generateTestId } from "../utils/helpers"

/** Roughly a fast human typist. */
const KEY_DELAY_MS = 150

const apiUrl = (path: string) => `${TEST_CONFIG.serverUrl.replace(/\/$/, "")}${path}`
const apiHeaders = () => ({ "X-API-KEY": TEST_CONFIG.apiKey })

let backendWarmed: Promise<void> | null = null

/**
 * Pay the backend's cold-start cost once per worker before driving the UI.
 * A cold backend freezes for ~30 s on its first /openapi.json build (#3135),
 * long enough for the page's auth check to fail and every save with it.
 * Same as warmBackendOnce in the harness's e2e/utils/seed-api.ts.
 */
function warmBackendOnce(request: APIRequestContext): Promise<void> {
  const warmups = [
    { path: "/openapi.json", timeout: 120_000 },
    { path: "/api/v1/llm/models/metadata", timeout: 60_000 },
  ]
  backendWarmed ??= (async () => {
    for (const { path, timeout } of warmups) {
      const response = await request.get(apiUrl(path), { headers: apiHeaders(), timeout })
      if (!response.ok()) throw new Error(`Warming ${path} failed: HTTP ${response.status()}`)
    }
  })()
  return backendWarmed
}

const isNoteCreate = (url: string, method: string) =>
  method === "POST" && /\/api\/v1\/notes\/?$/.test(new URL(url).pathname)

async function savedContent(request: APIRequestContext, noteId: string): Promise<string> {
  const response = await request.get(apiUrl(`/api/v1/notes/${encodeURIComponent(noteId)}`), {
    headers: apiHeaders(),
  })
  if (!response.ok()) return `HTTP ${response.status()}`
  const note = await response.json()
  return String(note?.content ?? "")
}

/**
 * Open a new note with a title and switch it to WYSIWYG. Returns the editor
 * and a promise for the id the first autosave creates.
 */
async function startWysiwygNote(page: Page, token: string) {
  const notes = new NotesPage(page)
  await notes.goto()
  await notes.newNoteButton.first().click()
  await expect(notes.titleInput).toBeVisible()
  await notes.titleInput.fill(`${token} note`)

  await page.getByTestId("notes-input-mode-wysiwyg").click()
  const editor = page.getByTestId("notes-wysiwyg-editor")
  await expect(editor).toBeVisible()

  const createdId = page
    .waitForResponse(
      (response) => isNoteCreate(response.url(), response.request().method()) && response.ok(),
      { timeout: 60_000 }
    )
    .then(async (response) => String((await response.json())?.id ?? ""))

  await editor.click()
  return { editor, createdId }
}

test.describe("Notes WYSIWYG editor (NE-01)", () => {
  test("NE-01: text typed at human speed in WYSIWYG reads forward in the editor and the saved note", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(request)
    const { editor, createdId } = await startWysiwygNote(authedPage, generateTestId("uxr-ne01"))

    await authedPage.keyboard.type("Hello world", { delay: KEY_DELAY_MS })

    // Soft, so a regression also reports what autosave stored.
    await expect.soft(editor).toHaveText("Hello world")
    const noteId = await createdId
    expect(noteId).not.toBe("")
    await expect.poll(() => savedContent(request, noteId), { timeout: 30_000 }).toBe("Hello world")
    await expect(editor).toHaveText("Hello world")
  })

  test("NE-01: typing continues at the caret after the new note's first autosave", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(request)
    const detailReads: string[] = []
    authedPage.on("response", (response) => {
      if (response.request().method() === "GET") detailReads.push(new URL(response.url()).pathname)
    })
    const { editor, createdId } = await startWysiwygNote(authedPage, generateTestId("uxr-ne01-resume"))

    await authedPage.keyboard.type("Hello world", { delay: KEY_DELAY_MS })
    const noteId = await createdId
    expect(noteId).not.toBe("")
    // The first save reloads the new note into the still-focused editor.
    await expect
      .poll(() => detailReads.some((pathname) => pathname.endsWith(`/api/v1/notes/${noteId}`)), {
        timeout: 30_000,
      })
      .toBe(true)
    await expect(editor).toHaveText("Hello world")
    await expect(editor).toBeFocused()

    await authedPage.keyboard.type(" again", { delay: KEY_DELAY_MS })

    await expect(editor).toHaveText("Hello world again")
    await expect
      .poll(() => savedContent(request, noteId), { timeout: 30_000 })
      .toBe("Hello world again")
  })

  test("NE-01: Heading and List format the WYSIWYG line without nesting", async ({
    authedPage,
    serverInfo,
    request,
  }) => {
    skipIfServerUnavailable(serverInfo)
    await warmBackendOnce(request)
    const { editor, createdId } = await startWysiwygNote(authedPage, generateTestId("uxr-ne01-toolbar"))

    await authedPage.keyboard.type("Plan", { delay: KEY_DELAY_MS })
    await authedPage.getByTestId("notes-toolbar-heading").click()
    const heading = editor.locator(":scope > h2")
    await expect(heading).toHaveText("Plan")

    await editor.press("End")
    await editor.press("Enter")
    await authedPage.keyboard.type("Buy milk", { delay: KEY_DELAY_MS })
    await authedPage.getByTestId("notes-toolbar-list").click()
    const item = editor.locator(":scope > ul > li")
    await expect(item).toHaveText("Buy milk")
    await expect(editor.locator("h2 ul, p ul, div ul")).toHaveCount(0)

    // Visible, not just present: Tailwind's preflight would otherwise render
    // the heading as body text and the list without bullets.
    const styles = await editor.evaluate((root) => {
      const h2 = root.querySelector("h2") as HTMLElement
      const ul = root.querySelector("ul") as HTMLElement
      const li = root.querySelector("li") as HTMLElement
      return {
        headingSize: parseFloat(getComputedStyle(h2).fontSize),
        itemSize: parseFloat(getComputedStyle(li).fontSize),
        listStyle: getComputedStyle(ul).listStyleType,
      }
    })
    expect(styles.headingSize).toBeGreaterThan(styles.itemSize)
    expect(styles.listStyle).toBe("disc")

    const noteId = await createdId
    expect(noteId).not.toBe("")
    await expect
      .poll(() => savedContent(request, noteId), { timeout: 30_000 })
      .toBe("## Plan\n\n- Buy milk")
  })
})
