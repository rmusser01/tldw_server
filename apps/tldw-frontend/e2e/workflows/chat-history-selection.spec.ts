import { qualifySelectionRecordConcurrency, qualifyHistoryStorage, qualifyStorageInterleavings } from '../../../extension/tests/e2e/utils/history-storage'
import { test, expect } from '@playwright/test'
import { sendIndependentSourceViews, rejectUnsupportedLocalFork, startHistoryServer, seedWebStorage, seedHistory, historyUrl, choose, readStore, localForkAndSend, legacyBoundaries, abortLocalFork, largeLegacyReview, seedNativeReference, nativeHistoryUrl, nativeForkAndSend, send, unknownNativeFork, editNativeChildSettings, heldNativeForkNavigation, rejectNativeMetadataSettings, heldNativeFirstCreate, firstNativeCharacterSend, uncertainNativeCharacterSend, qualifyChildIsolation, qualifyForeignWorkspaceAndAccountRead } from '../../../extension/tests/e2e/utils/history-selection'

test('two mounted views share real IndexedDB and keep independent selected variants', async ({ browser }) => {
  test.setTimeout(90_000)
  const server = await startHistoryServer()
  const context = await browser.newContext()
  try {
    const first = await context.newPage()
    await seedWebStorage(first, server.url)
    await first.goto('/chat')
    await expect(first.getByTestId('chat-input')).toBeVisible({ timeout: 60000 })
    await seedHistory(first)
    await first.goto(historyUrl('/chat'))
    await choose(first, 'h1-a')
    const second = await context.newPage()
    await second.goto(historyUrl('/chat'))
    await choose(second, 'h1-b')
    const bookmarks = await readStore(first, 'historySelections')
    expect(bookmarks.filter(row => row.conversation_id === 'h1-source').map(row => row.view.cursor)).toEqual(expect.arrayContaining([{ kind: 'after_message', message_id: 'h1-a' }, { kind: 'after_message', message_id: 'h1-b' }]))
    await expect(first.getByText('H1 variant A', { exact: true })).toBeVisible()
    await expect(first.getByText('H1 variant B', { exact: true })).toHaveCount(0)
    await expect(second.getByText('H1 variant B', { exact: true })).toBeVisible()
    await first.reload()
    await expect(first.getByText('H1 variant A', { exact: true })).toBeVisible()
    expect((await readStore(first, 'messages')).length).toBe(3)
    await sendIndependentSourceViews(first, second, server)
  } finally { await context.close(); await server.close() }
})

test('a supported local fork adopts child ownership for immediate send and reopen', async ({ browser }) => {
  test.setTimeout(90_000)
  const server = await startHistoryServer()
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await seedHistory(page)
    await page.goto(historyUrl('/chat'))
    try { await localForkAndSend(page, server) } catch (error) { console.log('H1_HTTP_REQUESTS', JSON.stringify(server.requests)); throw error }
  } finally { await context.close(); await server.close() }
})

test('legacy alternatives survive confirmed before-first and empty boundaries on reopen', async ({ browser }) => {
  const server = await startHistoryServer()
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await seedHistory(page, { legacy: true })
    await page.goto(historyUrl('/chat'))
    await legacyBoundaries(page)
  } finally { await context.close(); await server.close() }
})

for (const scenario of ['unsupported required context', 'transaction abort', '20001-row legacy review', '20001-row full-tip transcript', '20001-row bubble full-tip transcript'] as const) {
  test(`real IndexedDB: ${scenario}`, async ({ browser }) => {
    test.setTimeout(120_000)
    const server = await startHistoryServer()
    const context = await browser.newContext()
    try {
      const page = await context.newPage()
      await seedWebStorage(page, server.url)
      if (scenario.includes('bubble')) await page.addInitScript(() => localStorage.setItem('chatShowCharacterPortraits', 'false'))
      await page.goto('/chat')
      await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
      await seedHistory(page, ['transaction abort', 'unsupported required context'].includes(scenario) ? {} : { legacy: true, count: 20001 })
      await page.goto(historyUrl('/chat'))
      if (scenario === 'unsupported required context') { await rejectUnsupportedLocalFork(page); expect(server.requests.filter(row => row.method === 'POST')).toEqual([]) }
      else if (scenario === 'transaction abort') await abortLocalFork(page)
      else await largeLegacyReview(page, scenario.includes('full-tip'), scenario.includes('bubble'))
    } finally { await context.close(); await server.close() }
  })
}

test('populated IndexedDB v15 upgrades to v16 preserving projection, bookmarks and send recovery', async ({ browser }) => {
  test.setTimeout(90_000)
  const server = await startHistoryServer()
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await seedHistory(page, { legacy: true })
    await page.goto(historyUrl('/chat'))
    await legacyBoundaries(page)
    await page.route('**/h1-storage-fixture', route => route.fulfill({ contentType: 'text/html', body: '<title>H1 storage fixture</title>' }))
    await page.goto('/h1-storage-fixture')
    const saved = await page.evaluate(async () => {
      const open = indexedDB.open('PageAssistDatabase')
      const db = await new Promise<IDBDatabase>((resolve, reject) => { open.onsuccess = () => resolve(open.result); open.onerror = () => reject(open.error) })
      const names = [...db.objectStoreNames].filter(name => name !== 'forkOperations')
      const tx = db.transaction(names)
      const definitions = names.map(name => {
        const store = tx.objectStore(name)
        return { name, keyPath: store.keyPath, autoIncrement: store.autoIncrement, indexes: [...store.indexNames].map(indexName => { const index = store.index(indexName); return { name: indexName, keyPath: index.keyPath, unique: index.unique, multiEntry: index.multiEntry } }) }
      })
      const data = Object.fromEntries(await Promise.all(names.map(async name => {
        const get = tx.objectStore(name).getAll()
        return [name, await new Promise<any[]>((resolve, reject) => { get.onsuccess = () => resolve(get.result); get.onerror = () => reject(get.error) })]
      })))
      const bookmark = data.historySelections.find((row: any) => row.client_session_id === 'fixture-origin')
      bookmark.pending_turns = { 'h1-unknown': { operation_id: 'h1-unknown', input_text: 'Retained uncertain input', result_text: 'Retained unsaved answer', state: 'generated_unsaved', persistence: 'client', input_id: 'unknown-input', assistant_id: 'unknown-answer', origin_view: bookmark.view, selection_digest: 'selection-fixture', request_context_digest: 'context-fixture', owner_key: bookmark.owner_key, conversation_id: bookmark.conversation_id, created_at: 1, input_images: [] } }
      db.close()
      await new Promise<void>((resolve, reject) => { const remove = indexedDB.deleteDatabase('PageAssistDatabase'); remove.onsuccess = () => resolve(); remove.onerror = () => reject(remove.error); remove.onblocked = () => reject(new Error('Database still open')) })
      const older = indexedDB.open('PageAssistDatabase', 150)
      older.onupgradeneeded = () => {
        for (const definition of definitions) {
          const store = older.result.createObjectStore(definition.name, { keyPath: definition.keyPath, autoIncrement: definition.autoIncrement })
          for (const index of definition.indexes) store.createIndex(index.name, index.keyPath, { unique: index.unique, multiEntry: index.multiEntry })
          for (const row of data[definition.name]) store.put(row)
        }
      }
      const v15 = await new Promise<IDBDatabase>((resolve, reject) => { older.onsuccess = () => resolve(older.result); older.onerror = () => reject(older.error) })
      const version = v15.version
      v15.close()
      return { data, version }
    })
    expect(saved.version).toBe(150)
    await page.goto('/chat')
    await expect(page.getByRole('button', { name: 'Review conversation history', exact: true })).toBeVisible()
    expect(await page.evaluate(async () => (await indexedDB.databases()).find(db => db.name === 'PageAssistDatabase')?.version)).toBe(160)
    expect(await readStore(page, 'historyProjections')).toEqual(saved.data.historyProjections)
    const original = (await readStore(page, 'historySelections')).find(row => row.client_session_id === 'fixture-origin')
    expect(original).toEqual(saved.data.historySelections.find((row: any) => row.client_session_id === 'fixture-origin'))
    expect(await readStore(page, 'messages')).toEqual(saved.data.messages)
    expect(await readStore(page, 'forkOperations')).toEqual([])
  } finally { await context.close(); await server.close() }
})

test('native HTTP fork adopts scoped child for next send and reopen through actual controller', async ({ browser }) => {
  test.setTimeout(90_000)
  const server = await startHistoryServer({ native: true })
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await seedNativeReference(page)
    await page.goto(nativeHistoryUrl('/chat'))
    try { const child = await nativeForkAndSend(page, server); await editNativeChildSettings(page, server, child) } catch (error) { console.log('H1_NATIVE_REQUESTS', JSON.stringify(server.requests)); throw error }
  } finally { await context.close(); await server.close() }
})


test('a new unowned draft binds a local owner before first send without automatic server copying', async ({ browser }) => {
  const server = await startHistoryServer({ native: true })
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await send(page, 'First owned draft turn')
    await expect.poll(async () => (await readStore(page, 'messages')).length).toBe(2)
    const histories = await readStore(page, 'chatHistories')
    expect(histories).toHaveLength(1)
    expect(histories[0].local_owner_key).toMatch(/^local-history-v1:/)
    expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(0)
  } finally { await context.close(); await server.close() }
})


test('unknown native copy survives reload and a second actual view without redispatch', async ({ browser }) => {
  const server = await startHistoryServer({ native: true, loseForkResponse: true })
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedNativeReference(page)
    await page.goto(nativeHistoryUrl('/chat'))
    await unknownNativeFork(page, server, async () => { const second = await context.newPage(); await second.goto(nativeHistoryUrl('/chat')); return second })
  } finally { await context.close(); await server.close() }
})

for (const portraits of [true, false]) {
  test('timeline edit and cancel work in actual ' + (portraits ? 'portrait card' : 'user bubble') + ' layout', async ({ browser }) => {
    const server = await startHistoryServer()
    const context = await browser.newContext()
    try {
      const page = await context.newPage()
      await seedWebStorage(page, server.url)
      await page.addInitScript(value => localStorage.setItem('chatShowCharacterPortraits', JSON.stringify(value)), portraits)
      await page.goto('/chat')
      await expect(page.getByTestId('chat-input')).toBeVisible()
      await seedHistory(page)
      await page.goto(historyUrl('/chat'))
      await choose(page, 'h1-a')
      await page.evaluate(() => window.dispatchEvent(new CustomEvent('tldw:timeline-action', { detail: { action: 'edit', historyId: 'h1-source', messageId: 'h1-user' } })))
      const card = page.locator('[data-message-id="h1-user"]')
      await expect(card.locator('textarea')).toBeVisible()
      await card.getByRole('button', { name: 'Cancel', exact: true }).click()
      await expect(card.locator('textarea')).toHaveCount(0)
      expect((await readStore(page, 'messages')).find(row => row.id === 'h1-user').content).toBe('H1 original question')
    } finally { await context.close(); await server.close() }
  })
}


for (const live of [false, true]) for (const boundary of ['create', 'settings'] as const) {
  test((live ? 'live navigation: ' : 'unload: ') + 'held native ' + boundary + ' response cannot redirect a later local view or downgrade its saved result', async ({ browser }) => {
    const server = await startHistoryServer({ native: true })
    const context = await browser.newContext()
    try {
      const page = await context.newPage()
      await seedWebStorage(page, server.url)
      await page.goto('/chat')
      await expect(page.getByTestId('chat-input')).toBeVisible()
      await seedNativeReference(page)
      await page.goto(nativeHistoryUrl('/chat'))
      await heldNativeForkNavigation(page, server, '/chat', boundary, live)
    } finally { await context.close(); await server.close() }
  })
}

test('production storage APIs preserve migration, concurrent writes and unavailable guards across delete import undo and reload', async ({ browser }) => {
  const server = await startHistoryServer()
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await qualifyHistoryStorage(page)
  } finally { await context.close(); await server.close() }
})

test('production storage interleaves migration and writers while distinguishing failed reads from empty data', async ({ browser }) => {
  const server = await startHistoryServer()
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await qualifyStorageInterleavings(page)
    await qualifySelectionRecordConcurrency(page)
  } finally { await context.close(); await server.close() }
})

test('rejected native metadata cannot redirect explicit settings to browser mirror storage', async ({ browser }) => {
  test.setTimeout(90_000)
  const server = await startHistoryServer({ native: true })
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await seedNativeReference(page)
    await page.goto(nativeHistoryUrl('/chat'))
    try { const child = await nativeForkAndSend(page, server); await rejectNativeMetadataSettings(page, server, child) } catch (error) { console.log('H1_NATIVE_REQUESTS', JSON.stringify(server.requests)); throw error }
  } finally { await context.close(); await server.close() }
})



for (const live of [false, true]) test((live ? 'live navigation: ' : 'unload: ') + 'held first native character create cannot adopt or dispatch after local navigation', async ({ browser }) => {
  test.setTimeout(60000)
  const server = await startHistoryServer()
  const context = await browser.newContext()
  try {
    const page = await context.newPage()
    await seedWebStorage(page, server.url)
    await page.goto('/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await heldNativeFirstCreate(page, server, '/chat', live)
  } finally { await context.close(); await server.close() }
})

for (const comparison of [false, true]) test('child edits deletion and copied-file removal isolate source: ' + (comparison ? 'comparison' : 'normal'), async ({ browser }) => {
  test.setTimeout(90000)
  const server = await startHistoryServer()
  const context = await browser.newContext()
  const page = await context.newPage()
  const base = '/chat'
  try {
    await seedWebStorage(page, server.url)
    await page.goto(base)
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await qualifyChildIsolation(page, server, base, comparison)
  } finally { await context.close(); await server.close() }
})

test('global H1 filters foreign workspace operations and invalidates held reads after account config changes', async ({ browser }) => {
  test.setTimeout(60000)
  const server = await startHistoryServer()
  const context = await browser.newContext()
  const page = await context.newPage()
  const base = '/chat'
  try {
    await seedWebStorage(page, server.url)
    await page.goto(base)
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedNativeReference(page)
    await page.goto(nativeHistoryUrl(base))

    await qualifyForeignWorkspaceAndAccountRead(page, server, base)
  } finally { await context.close(); await server.close() }
})

 test('first native character send follows acknowledged result and reopens for a second native turn', async ({ browser }) => {
 const server = await startHistoryServer(); const context = await browser.newContext();
 try { const page = await context.newPage(); await seedWebStorage(page, server.url); await page.goto('/chat'); await expect(page.getByTestId('chat-input')).toBeVisible(); await firstNativeCharacterSend(page, server) }
 finally { await context.close(); await server.close() }
})

for (const ack of ['missing', 'wrong'] as const) test('native character ' + ack + ' ACK remains uncertain without replay', async ({ browser }) => {
 const server = await startHistoryServer({ characterAck: ack }); const context = await browser.newContext()
 try { const page = await context.newPage(); await seedWebStorage(page, server.url); await page.goto('/chat'); await expect(page.getByTestId('chat-input')).toBeVisible(); await uncertainNativeCharacterSend(page, server) }
 finally { await context.close(); await server.close() }
})
