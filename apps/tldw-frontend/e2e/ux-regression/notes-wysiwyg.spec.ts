/**
 * NE-01 (#3102): the Notes WYSIWYG editor must type forward at human speed.
 *
 * The editor used to re-render its contentEditable from state on every input,
 * which moved the caret to the start, so "Hello world" was typed and saved as
 * "dlrow olleH". These are regression tests for the fix.
 *
 * Runs in the ux-regression project against an isolated real backend
 * (bun run e2e:ux-regression -- --grep NE-01).
 */
import type { APIRequestContext, Page } from "@playwright/test"
import { test, expect, skipIfServerUnavailable } from "../utils/fixtures"
import { NotesPage } from "../utils/page-objects"
import { TEST_CONFIG, generateTestId } from "../utils/helpers"
import { createSeedApi, warmBackendOnce } from "../utils/seed-api"
import { launchExtension } from '../utils/extension-sidepanel'

test.use({ trace: 'retain-on-failure' })

/** Roughly a fast human typist. */
const KEY_DELAY_MS = 150

const apiUrl = (path: string) => `${TEST_CONFIG.serverUrl.replace(/\/$/, "")}${path}`
const apiHeaders = () => ({ "X-API-KEY": TEST_CONFIG.apiKey })

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
    await warmBackendOnce(createSeedApi(request))
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
    await warmBackendOnce(createSeedApi(request))
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
    await warmBackendOnce(createSeedApi(request))
    const { editor, createdId } = await startWysiwygNote(authedPage, generateTestId("uxr-ne01-toolbar"))

    await authedPage.keyboard.type("Plan", { delay: KEY_DELAY_MS })
    await authedPage.getByTestId("notes-toolbar-heading").click()
    await authedPage.getByRole('menuitem', { name: 'Heading 2', exact: true }).click()
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

for (const surface of ['WebUI', 'extension options'] as const) {
  test(`NE-06: ${surface} toolbar lists continue and survive an edited save/reopen`, async ({ authedPage, serverInfo, request }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const title = generateTestId('uxr-ne06-editor')
    const response = await api.post('/api/v1/notes/', { title, content: 'First', keywords: [] })
    expect(response.ok()).toBe(true)
    const noteId = String((await response.json()).id)
    const extension = surface === 'extension options'
      ? await launchExtension({ serverUrl: TEST_CONFIG.serverUrl, apiKey: TEST_CONFIG.apiKey }) : null
    const page = extension ? await extension.context.newPage() : authedPage
    const route = extension ? `chrome-extension://${extension.extensionId}/options.html#/notes` : '/notes'
    try {
      await page.setViewportSize({ width: 1280, height: 900 })
      if (extension) {
        await extension.context.serviceWorkers()[0].evaluate(() =>
          chrome.storage.local.set({ "notes-tutorial-shown": JSON.stringify("1") })
        )
      }
      await page.goto(route)
      await page.getByTestId(`notes-open-button-${noteId}`).click()
      if (extension) {
        const prompt = page.getByText('New to this page?', { exact: true })
        await expect(prompt).toBeVisible()
        await page.getByRole('button', { name: 'Not now', exact: true }).click()
        await expect(prompt).toBeHidden()
      }
      await page.getByTestId('notes-input-mode-wysiwyg').click()
      const editor = page.getByTestId('notes-wysiwyg-editor')
      await editor.click()
      await editor.press('End')
      for (const level of [1, 3]) {
        await page.getByTestId('notes-toolbar-heading').click()
        await page.getByRole('menuitem', { name: `Heading ${level}`, exact: true }).click()
        await expect(editor.locator(`h${level}`)).toHaveText('First')
      }
      await page.getByTestId('notes-toolbar-numbered').click()
      await expect(editor.locator('ol > li')).toHaveText(['First'])
      await editor.press('End')
      await editor.press('Enter')
      await page.keyboard.type('Second', { delay: KEY_DELAY_MS })
      await editor.press('Enter')
      await editor.press('Enter')
      await page.keyboard.type('Task', { delay: KEY_DELAY_MS })
      await page.getByTestId('notes-toolbar-checklist').click()
      await expect(editor.locator('ul > li > input')).toHaveCount(1)
      await editor.getByRole('checkbox').check()
      await editor.locator('ul > li').click()
      await editor.press('End')
      await editor.press('Enter')
      await page.keyboard.type('Next', { delay: KEY_DELAY_MS })
      await expect(editor.getByRole('checkbox')).toHaveCount(2)
      await expect(editor.getByRole('checkbox').last()).not.toBeChecked()
      await editor.press('Enter')
      await editor.press('Enter')
      await page.keyboard.type('Body edited', { delay: KEY_DELAY_MS })
      await page.getByTestId('notes-save-button').click()
      const expected = '1. First\n2. Second\n\n- [x] Task\n- [ ] Next\n\nBody edited'
      await expect.poll(() => savedContent(request, noteId)).toBe(expected)
      await page.getByTestId('notes-input-mode-markdown').click()
      await expect(page.getByPlaceholder('Write your note here... (Markdown supported)')).toHaveValue(expected)
      await page.goto(route)
      await page.getByTestId(`notes-open-button-${noteId}`).click()
      await page.getByTestId('notes-input-mode-wysiwyg').click()
      await expect(editor.locator('ol > li')).toHaveText(['First', 'Second'])
      await expect(editor.getByRole('checkbox').first()).toBeChecked()
      await expect(editor.getByRole('checkbox').last()).not.toBeChecked()
    } finally { await extension?.close() }
  })

  test(`NE-03: ${surface} long outline jumps within bounded wrapped editors`, async ({ authedPage, serverInfo, request }) => {
    skipIfServerUnavailable(serverInfo)
    const api = createSeedApi(request)
    await warmBackendOnce(api)
    const title = generateTestId('uxr-ne03-outline')
    const content = `# ${title}\n\n# Table of contents\n\n${Array.from({ length: 24 }, (_, index) => `## Section ${index}\n\n${'Wrapped prose '.repeat(80)}`).join('\n\n')}`
    const response = await api.post('/api/v1/notes/', { title, content, keywords: [] })
    expect(response.ok()).toBe(true)
    const noteId = String((await response.json()).id)
    const extension = surface === 'extension options'
      ? await launchExtension({ serverUrl: TEST_CONFIG.serverUrl, apiKey: TEST_CONFIG.apiKey }) : null
    const page = extension ? await extension.context.newPage() : authedPage
    const notes = new NotesPage(page)
    const route = extension ? `chrome-extension://${extension.extensionId}/options.html#/notes` : '/notes'
    try {
      await page.setViewportSize({ width: 1280, height: 900 })
      if (extension) {
        await extension.context.serviceWorkers()[0].evaluate(() =>
          chrome.storage.local.set({ "notes-tutorial-shown": JSON.stringify("1") })
        )
      }
      await page.goto(route)
      await page.getByTestId(`notes-open-button-${noteId}`).click()
      if (extension) {
        const prompt = page.getByText('New to this page?', { exact: true })
        await expect(prompt).toBeVisible()
        await page.getByRole('button', { name: 'Not now', exact: true }).click()
        await expect(prompt).toBeHidden()
      }
      const toggle = page.getByTestId('notes-toc-panel-toggle')
      await expect(toggle).toHaveAttribute('aria-expanded', 'false')
      await expect(page.getByTestId('notes-toc-panel-badge')).toHaveText('24')
      await expect(page.getByTestId('notes-save-button')).toBeInViewport()
      await expect(page.getByTestId('notes-toolbar-heading')).toBeInViewport()
      const initialEditor = page.getByTestId('notes-wysiwyg-editor').or(
        page.getByPlaceholder('Write your note here... (Markdown supported)')
      )
      await expect(initialEditor).toBeInViewport()
      for (const width of [1280, 760]) {
        await page.setViewportSize({ width, height: 900 })
        for (const mode of ['markdown', 'wysiwyg'] as const) {
          await page.getByTestId(`notes-input-mode-${mode}`).click()
          const editor = mode === 'markdown' ? page.getByPlaceholder('Write your note here... (Markdown supported)') : page.getByTestId('notes-wysiwyg-editor')
          await expect(editor).toBeInViewport()
          if (width === 760) {
            await notes.overflowMenuButton.click()
            await page.getByRole('menuitem', { name: 'Preview', exact: true }).click()
          } else {
            await page.getByRole('button', { name: 'Preview', exact: true }).click()
          }
          if (await toggle.getAttribute('aria-expanded') === 'false') await toggle.click()
          const lateHeading = page.getByTestId('notes-toc-item-section-23')
          await lateHeading.scrollIntoViewIfNeeded()
          const outer = await page.evaluate(() => ({ windowY: window.scrollY, body: document.querySelector('[data-testid="notes-editor-body"]')?.scrollTop || 0 }))
          await lateHeading.click()
          await expect(editor).toBeFocused()
          await expect(editor).toBeInViewport()
          await expect.poll(() => editor.evaluate((element) => element.scrollTop)).toBeGreaterThan(0)
          expect(await page.evaluate(() => ({ windowY: window.scrollY, body: document.querySelector('[data-testid="notes-editor-body"]')?.scrollTop || 0 }))).toEqual(outer)
          if (mode === 'markdown') {
            expect(await editor.evaluate((element) => (element as HTMLTextAreaElement).selectionStart)).toBe(content.indexOf('## Section 23'))
            expect(await editor.evaluate((element) => element.scrollTop / (element.scrollHeight - element.clientHeight))).toBeGreaterThan(0.85)
          } else {
            const geometry = await editor.evaluate((element) => {
              const heading = Array.from(element.querySelectorAll('h2')).find((node) => node.textContent === 'Section 23')!
              const target = heading.getBoundingClientRect(), viewport = element.getBoundingClientRect()
              return { top: target.top, bottom: target.bottom, viewportTop: viewport.top, viewportBottom: viewport.bottom }
            })
            expect(geometry.top).toBeGreaterThanOrEqual(geometry.viewportTop)
            expect(geometry.bottom).toBeLessThanOrEqual(geometry.viewportBottom)
          }
          await toggle.click()
          const preview = page.getByTestId('notes-split-preview-viewport')
          await editor.hover()
          await page.mouse.wheel(0, -200)
          await expect.poll(async () => {
            const ratio = await editor.evaluate((node) => node.scrollTop / (node.scrollHeight - node.clientHeight))
            const peer = await preview.evaluate((node) => node.scrollTop / (node.scrollHeight - node.clientHeight))
            return Math.abs(ratio - peer)
          }).toBeLessThan(0.02)
          await preview.hover()
          await page.mouse.wheel(0, -200)
          await expect.poll(async () => {
            const ratio = await editor.evaluate((node) => node.scrollTop / (node.scrollHeight - node.clientHeight))
            const peer = await preview.evaluate((node) => node.scrollTop / (node.scrollHeight - node.clientHeight))
            return Math.abs(ratio - peer)
          }).toBeLessThan(0.02)
        }
      }
      expect(await savedContent(request, noteId)).toBe(content)
    } finally { await extension?.close() }
  })
}
