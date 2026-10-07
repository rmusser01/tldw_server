/** I1 qualification; requires the temporary fixed-format action until I3 ships. */
import { test, expect } from '@playwright/test'
import { seedAuth, TEST_CONFIG } from './utils/helpers'

test.skip(process.env.TLDW_I1_QUALIFICATION !== '1', 'Failure reproducer: apply the documented temporary Notes wiring first; this is not production coverage.')

test('format is one native undo step on the controlled Notes page', async ({ page, request, browser }, testInfo) => {
  const title = `Native history ${testInfo.project.name} ${Date.now()}`
  const headers = { 'X-API-KEY': TEST_CONFIG.apiKey }
  const created = await request.post(`${TEST_CONFIG.serverUrl}/api/v1/notes/`, { headers, data: {title, content: '{"a":1}'} })
  expect(created.ok(), await created.text()).toBeTruthy()
  const note = await created.json()
  const trace: Record<string, string> = {browser: browser.version()}
  try {
    await seedAuth(page)
    await page.goto('/notes')
    await page.getByRole('button', {name: `Open note ${title}`, exact: true}).click()
    const editor = page.getByRole('textbox', { name: 'Note content', exact: true })
    await expect(editor).toHaveValue('{"a":1}')
    trace.A = await editor.inputValue()
    await editor.focus()
    await editor.press('ControlOrMeta+End')
    await editor.pressSequentially(' ')
    trace.B = await editor.inputValue()
    await page.getByRole('button', {name: 'Format document', exact: true}).click()
    await expect(editor).toHaveValue('{\n  "a": 1\n}')
    trace.C = await editor.inputValue()
    await expect(editor).toBeFocused()
    await editor.press('ControlOrMeta+End')
    await editor.pressSequentially(' ')
    trace.D = await editor.inputValue()
    await editor.press('ControlOrMeta+z')
    trace.undoD = await editor.inputValue()
    await expect(editor).toHaveValue(trace.C)
    await editor.press('ControlOrMeta+z')
    trace.undoC = await editor.inputValue()
    await expect(editor).toHaveValue(trace.B)
    await editor.press('ControlOrMeta+Shift+z')
    trace.redoC = await editor.inputValue()
    await expect(editor).toHaveValue(trace.C)
    await editor.press('ControlOrMeta+z')
    await editor.press('ControlOrMeta+z')
    trace.undoB = await editor.inputValue()
    await expect(editor).toHaveValue(trace.A)
  } finally {
    await testInfo.attach('native-history-trace.json', {body: JSON.stringify(trace, null, 2), contentType: 'application/json'})
    console.log(JSON.stringify(trace))
    await page.close()
    const current = await request.get(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {headers})
    const currentNote = await current.json()
    const removed = await request.delete(`${TEST_CONFIG.serverUrl}/api/v1/notes/${note.id}`, {headers: {...headers, "expected-version": String(currentNote.version)}})
    expect(removed.ok(), await removed.text()).toBeTruthy()
  }
})
