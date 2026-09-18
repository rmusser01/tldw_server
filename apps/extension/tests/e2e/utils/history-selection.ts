import { injectHistoryStorage } from './history-storage'
import { expect, test, type Page, type Request } from '@playwright/test'
import http from 'node:http'
import type { AddressInfo } from 'node:net'

/** Deterministic HTTP/provider boundary. This is not native DB authority evidence. */
export async function startHistoryServer(options: { native?: boolean; loseForkResponse?: boolean; characterAck?: 'missing' | 'wrong' } = {}) {
  const requests: { method: string; path: string; body: any }[] = []
  const nativeChats = new Map<string, any[]>([['native-source', [
    { id: 'native-user', role: 'user', content: 'Native original question', parent_message_id: null },
    { id: 'native-a', role: 'assistant', content: 'Native variant A', parent_message_id: 'native-user' },
    { id: 'native-b', role: 'assistant', content: 'Native variant B', parent_message_id: 'native-user' }
  ]]])
  const nativeMetadataOverrides = new Map<string, Record<string, unknown>>()
  const nativeSettings = new Map<string, any>()
  const holds = new Map<string, { started: () => void; released: Promise<void>; settled: (event: string) => void }>()
  const holdNext = (method: string, path: string) => {
    let signalStarted!: () => void
    let release!: () => void
    const started = new Promise<void>(resolve => { signalStarted = resolve })
    const released = new Promise<void>(resolve => { release = resolve })
    let signalSettled!: (event: string) => void
    const responseSettled = new Promise<string>(resolve => { signalSettled = resolve })
    holds.set(method + ' ' + path, { started: signalStarted, released, settled: signalSettled })
    return { started, release, responseSettled }
  }
  const server = http.createServer(async (req, res) => {
    const chunks: Buffer[] = []
    for await (const chunk of req) chunks.push(Buffer.from(chunk))
    const raw = Buffer.concat(chunks).toString()
    const body = raw ? JSON.parse(raw) : null
    const path = new URL(req.url!, 'http://fixture').pathname
    requests.push({ method: req.method!, path, body })
    const headers = {
      'access-control-allow-origin': req.headers.origin || '*',
      'access-control-allow-credentials': 'true',
      'access-control-allow-headers': req.headers['access-control-request-headers'] || '*',
      'access-control-allow-methods': 'GET,POST,PATCH,PUT,DELETE,OPTIONS'
    }
    const json = (value: unknown, status = 200) => {
      res.writeHead(status, { ...headers, 'content-type': 'application/json' })
      res.end(JSON.stringify(value))
    }
    if (req.method === 'OPTIONS') return json({})
    const held = holds.get(req.method + ' ' + path)
    if (held) {
      holds.delete(req.method + ' ' + path)
      res.once('finish', () => held.settled('finished'))
      res.once('close', () => held.settled('closed'))
    }
    const waitHeld = async () => { if (held) { held.started(); await held.released } }
    if (!(path === '/api/v1/chats/' && req.method === 'POST')) await waitHeld()
    {
      const captureId = /^\/api\/v1\/chat\/conversations\/([^/]+)\/history\/selection$/.exec(path)?.[1]
      if (captureId) {
        const source = nativeChats.get(captureId) || []
        const nodes = source.map(row => ({ id: row.id, revision: '1', parent_id: row.parent_message_id, role: row.role, settled: true, conversation_id: captureId, preview: row.content }))
        const view = { ...body.view, owner_key: 'native-h1-owner' }
        const selected: any[] = []
        if (view.cursor.kind !== 'empty') {
          let next = nodes.find(row => row.id === view.cursor.message_id)
          if (view.cursor.kind === 'before_message') next = nodes.find(row => row.id === next?.parent_id)
          while (next) { selected.unshift(next); next = nodes.find(row => row.id === next!.parent_id) }
        }
        return json({ status: 'captured', snapshot: { version: 1, owner_key: 'native-h1-owner', conversation_id: captureId, fences: { conversation: '1', history: String(source.length), settings: '1' }, nodes, source_digest: 'http-fixture-source', interpretation_status: { kind: 'parent_graph_v1' }, storage_context_digest: 'http-fixture-storage', native_fork_context: { policy: 'plain_v1', supported: !nativeSettings.get(captureId)?.authorNote, storage_context_digest: 'http-fixture-storage' } }, rows: selected, selected_content: selected.map(row => ({ id: row.id, revision: '1', message: source.find(item => item.id === row.id).content, images: [] })), view, purpose: body.purpose, storage_context_digest: 'http-fixture-storage' })
      }
      if (path === '/api/v1/chats/' && req.method === 'GET') return json([])
      if (path === '/api/v1/chats/' && req.method === 'POST') {
        const id = `native-child-${nativeChats.size}`
        nativeChats.set(id, [])
        if (body.character_id) nativeMetadataOverrides.set(id, { character_id: body.character_id, assistant_kind: 'character', assistant_id: body.character_id })
        await waitHeld()
        if (options.loseForkResponse) {
          res.writeHead(200, { ...headers, 'content-type': 'application/json', 'content-length': '1000' })
          res.write('{"id":', () => res.destroy())
          return
        }
        return json({ id, title: 'Forked conversation', scope_type: 'global', workspace_id: null, character_id: null, assistant_kind: null, assistant_id: null, ...nativeMetadataOverrides.get(id) })
      }
      const match = /^\/api\/v1\/chats\/([^/]+)(?:\/(settings|messages))?$/.exec(path)
      if (match && nativeChats.has(match[1])) {
        const id = match[1]
        if (match[2] === 'settings') {
          if (req.method === 'PUT') nativeSettings.set(id, { ...(nativeSettings.get(id) || {}), ...body.settings })
          return json({ conversation_id: id, settings: { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', ...(nativeSettings.get(id) || {}) } })
        }
        if (match[2] === 'messages') {
          if (req.method === 'GET') return json(nativeChats.get(id))
          const selection = body.tldw_history_selection_v1
          const admission = body.tldw_history_admission_v1
          const parent = selection?.messages.at(-1)?.id ?? admission?.input_message_id ?? body.parent_message_id ?? null
          const messageId = body.id || `${id}-row-${nativeChats.get(id)!.length}`
          const row = { ...body, id: messageId, role: body.role, content: body.content, parent_message_id: parent }
          nativeChats.get(id)!.push(row)
          return json({ id: messageId, conversation_id: id, parent_message_id: parent, sender: body.role, content: body.content, ...(selection ? { tldw_history_admission_v1: { version: 1, owner_key: 'native-h1-owner', conversation_id: id, input_message_id: messageId, input_message_revision: '1', selection_digest: selection.selection_digest, messages: selection.messages, originating_selection_revision: selection.selection_revision } } : {}) })
        }
        return json({ id, title: id === 'native-source' ? 'Native source' : 'Forked conversation', scope_type: 'global', workspace_id: null, character_id: null, assistant_kind: null, assistant_id: null, persona_memory_mode: null, ...nativeMetadataOverrides.get(id) })
      }
    }
    if (/health|ready|ping/.test(path)) return json({ status: 'ok', healthy: true })
    if (path.endsWith('/config/quickstart')) return json({ auth_mode: 'single_user', setup_required: false })
    if (path.endsWith('/users/me/profile')) return json({ profile_version: 'h1', preferences: {} })
    if (path.endsWith('/llm/models/metadata')) return json([{ id: 'h1-model', name: 'H1 Model', provider: 'openai', capabilities: ['chat'], context_length: 32768 }])
    if (path.endsWith('/llm/models')) return json(['h1-model'])
    if (path.endsWith('/llm/providers')) return json([{ id: 'openai', name: 'OpenAI', configured: true, available: true }])
    if (path === '/openapi.json') return json({ openapi: '3.0.0', info: { version: 'fixture' }, paths: { '/api/v1/chat/completions': {}, '/api/v1/llm/models': {}, '/api/v1/health': {} } })
    if (path.endsWith('/chat/completions')) {
      const selection = body.tldw_history_selection_v1
      if (selection && nativeMetadataOverrides.get(body.conversation_id)?.character_id) {
        const rows = nativeChats.get(body.conversation_id)!
        const inputId = `${body.conversation_id}-row-${rows.length}`
        const resultId = `${body.conversation_id}-row-${rows.length + 1}`
        rows.push({ id: inputId, role: 'user', content: body.messages.at(-1).content, parent_message_id: selection.messages.at(-1)?.id ?? null }, { id: resultId, role: 'assistant', content: 'H1 deterministic reply', parent_message_id: inputId })
        const admission = { version: 1, owner_key: 'native-h1-owner', conversation_id: body.conversation_id, input_message_id: inputId, input_message_revision: '1', selection_digest: selection.selection_digest, messages: selection.messages, originating_selection_revision: selection.selection_revision }
        if (options.characterAck === 'wrong') admission.selection_digest = 'wrong-selection'
        res.writeHead(200, { ...headers, 'content-type': 'text/event-stream' })
        if (options.characterAck === 'missing') {
          res.end(`data: ${JSON.stringify({ choices: [{ delta: { content: 'H1 deterministic reply' } }] })}\n\ndata: [DONE]\n\n`)
          return
        }
        res.end(`data: ${JSON.stringify({ tldw_history_admission_v1: admission, choices: [] })}\n\ndata: ${JSON.stringify({ tldw_message_id: resultId, tldw_conversation_id: body.conversation_id, choices: [{ delta: { content: 'H1 deterministic reply' } }] })}\n\ndata: [DONE]\n\n`)
        return
      }
      if (!body.stream) return json({ choices: [{ message: { role: 'assistant', content: 'H1 deterministic reply' } }] })
      res.writeHead(200, { ...headers, 'content-type': 'text/event-stream' })
      res.end(`data: ${JSON.stringify({ choices: [{ delta: { content: 'H1 deterministic reply' } }] })}\n\ndata: [DONE]\n\n`)
      return
    }
    if (path === '/api/v1/characters/7') return json({ id: 7, name: 'H1 Character', system_prompt: '', description: '' })
    if (path === '/api/v1/characters/') return json([{ id: 7, name: 'H1 Character', system_prompt: '', description: '' }])
    if (/characters|chat-sessions|presets|personas|prompts|folders|dictionaries|keywords/.test(path)) return json([])
    return json({ detail: 'Unimplemented H1 fixture route' }, 404)
  })
  await new Promise<void>(resolve => server.listen(0, '127.0.0.1', resolve))
  return {
    url: `http://127.0.0.1:${(server.address() as AddressInfo).port}`,
    requests,
    nativeChats,
    nativeSettings,
    nativeMetadataOverrides,
    holdNext,
    close: async () => { server.closeAllConnections(); await new Promise<void>(resolve => server.close(() => resolve())) }
  }
}

export const historySeedConfig = (serverUrl: string) => ({
  __tldw_first_run_complete: true,
  __tldw_allow_offline: true,
  isMigrated: true,
  assistant_setup_dismissed: true,
  selectedModel: JSON.stringify('tldw:h1-model'),
  serverUrl,
  tldwServerUrl: serverUrl,
  'tldw-api-host': serverUrl,
  authMode: 'single-user',
  apiKey: 'h1-fixture-key',
  tldwConfig: { serverUrl, authMode: 'single-user', apiKey: 'h1-fixture-key', authSource: 'manual', credentialSource: 'manual', apiKeyPersistence: 'device', apiKeyServerOrigin: new URL(serverUrl).origin }
})

export const nativeCharacterSeedConfig = (serverUrl: string) => ({
 ...historySeedConfig(serverUrl),
 selectedModel: JSON.stringify('tldw:openai:h1-model'),
 selectedAssistant: { kind: 'character', id: '7', name: 'H1 Character', metadata: { selectionMode: 'tracked' } }
})

export async function seedWebStorage(page: Page, serverUrl: string) {
  await page.addInitScript((config) => {
    if (localStorage.getItem('__h1_fixture_seeded')) return
    for (const [key, value] of Object.entries(config)) localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value))
    localStorage.setItem('selectedModel', 'tldw:h1-model')
    localStorage.setItem('__h1_fixture_seeded', 'true')
  }, historySeedConfig(serverUrl))
}

/** Native IndexedDB transactions seed only fixture data; owner actions run through mounted UI. */
export async function seedHistory(page: Page, options: { legacy?: boolean; count?: number; id?: string } = {}) {
  await page.waitForFunction(async () => (await indexedDB.databases()).some(db => db.name === 'PageAssistDatabase' && (db.version || 0) >= 160))
  const id = options.id || 'h1-source'
  await page.evaluate(async ({ id, legacy, count }) => {
    const request = indexedDB.open('PageAssistDatabase')
    const db = await new Promise<IDBDatabase>((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error) })
    if (!db.objectStoreNames.contains('messages')) throw new Error('H1 schema stores: ' + [...db.objectStoreNames].join(',') + ' version ' + db.version)
    const tx = db.transaction(['chatHistories', 'messages', 'userSettings', 'historySelections'], 'readwrite')
    tx.objectStore('historySelections').put({ profile_id: 'h1-profile', client_session_id: 'fixture-origin', owner_key: 'local-history-v1:h1-profile', conversation_id: id, view: { view_session_id: 'fixture-origin-view', owner_key: 'local-history-v1:h1-profile', conversation_id: id, interpretation: { kind: 'parent_graph_v1' }, cursor: { kind: 'after_message', message_id: count ? 'legacy-' + (count - 1) : 'h1-b' }, selection_revision: 0 } })
    tx.objectStore('userSettings').put({ id: 'main', history_profile_id: 'h1-profile' })
    tx.objectStore('chatHistories').put({ id, title: 'H1 Source', createdAt: 1, is_rag: false, local_owner_key: 'local-history-v1:h1-profile' })
    const rows = count ? Array.from({ length: count }, (_, index) => ({ id: `legacy-${index}`, role: index % 2 ? 'assistant' : 'user', content: `Legacy row ${index}`, createdAt: index + 1 })) : [
      { id: 'h1-user', role: 'user', content: 'H1 original question', createdAt: 1, parent_message_id: null },
      { id: 'h1-a', role: 'assistant', content: 'H1 variant A', createdAt: 2, parent_message_id: 'h1-user' },
      { id: 'h1-b', role: 'assistant', content: 'H1 variant B', createdAt: 3, parent_message_id: 'h1-user' }
    ]
    for (const row of rows) {
      const value: any = { ...row, history_id: id, name: row.role === 'user' ? 'You' : 'Assistant', images: [] }
      if (legacy) delete value.parent_message_id
      else value.history_provenance = { owner_key: 'local-history-v1:h1-profile', projection_id: null }
      tx.objectStore('messages').put(value)
    }
    await new Promise<void>((resolve, reject) => { tx.oncomplete = () => resolve(); tx.onabort = () => reject(tx.error); tx.onerror = () => reject(tx.error) })
    db.close()
  }, { id, legacy: !!options.legacy, count: options.count })
  return id
}

export async function readStore(page: Page, store: string) {
  return page.evaluate(async name => {
    const request = indexedDB.open('PageAssistDatabase')
    const db = await new Promise<IDBDatabase>((resolve, reject) => { request.onsuccess = () => resolve(request.result); request.onerror = () => reject(request.error) })
    const tx = db.transaction(name)
    const get = tx.objectStore(name).getAll()
    const rows = await new Promise<any[]>((resolve, reject) => { get.onsuccess = () => resolve(get.result); get.onerror = () => reject(get.error) })
    db.close()
    return rows
  }, store)
}

export const historyUrl = (base: string, id = 'h1-source') => `${base}?historySelection=${encodeURIComponent(JSON.stringify({ profile_id: 'h1-profile', client_session_id: 'fixture-origin', owner_key: 'local-history-v1:h1-profile', conversation_id: id, owner_kind: 'local' }))}`

export async function review(page: Page) {
  await expect(page.getByRole('button', { name: 'Review conversation history', exact: true })).toBeVisible({ timeout: 20000 })
  await page.getByRole('button', { name: 'Review conversation history', exact: true }).click()
  await expect(page.getByRole('list', { name: 'Complete source messages' })).toBeVisible()
  await page.screenshot({ path: test.info().outputPath('selected-history-review.png') })
}

export async function choose(page: Page, id: string) {
  await review(page)
  await page.locator(`[data-history-message-id="${id}"]`).getByRole('button', { name: 'Continue after this message' }).click()
  await page.getByRole('button', { name: 'Cancel review', exact: true }).click()
  await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.client_session_id !== 'fixture-origin' && row.view.cursor.message_id === id)).toBe(true)
}

export async function send(page: Page, text: string) {
  const input = page.getByTestId('chat-input')
  await expect(input).toBeEditable()
  await input.fill(text)
  await page.getByRole('button', { name: 'Send message', exact: true }).click()
  await expect(page.getByText('H1 deterministic reply', { exact: true }).last()).toBeVisible({ timeout: 20000 })
}

export async function forkAt(page: Page, id: string) {
  const card = page.locator(`[data-testid="chat-message"][data-message-id="${id}"]`)
  await card.hover()
  await card.getByRole('button', { name: 'More actions', exact: true }).click()
  await page.getByRole('button', { name: /New Branch/i }).click()
}

export async function localForkAndSend(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>) {
  page.setDefaultTimeout(15000)
  await choose(page, 'h1-a')
  expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(0)
  const source = (await readStore(page, 'messages')).filter(row => row.history_id === 'h1-source')
  await forkAt(page, 'h1-a')
  await expect.poll(async () => (await readStore(page, 'chatHistories')).length).toBe(2)
  const child = (await readStore(page, 'chatHistories')).find(row => row.id !== 'h1-source')
  expect(child.local_owner_key).toBe('local-history-v1:h1-profile')
  expect(child.server_chat_id).toBeUndefined()
  const copied = (await readStore(page, 'messages')).filter(row => row.history_id === child.id)
  expect(copied.map(row => row.content).sort()).toEqual(['H1 original question', 'H1 variant A'])
  const copiedUser = copied.find(row => row.role === 'user')
  const copiedAssistant = copied.find(row => row.role === 'assistant')
  expect(copiedAssistant.parent_message_id).toBe(copiedUser.id)
  expect(copied.map(row => row.id)).not.toEqual(expect.arrayContaining(['h1-user', 'h1-a']))
  await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.conversation_id === child.id)).toBe(true)
  await page.screenshot({ path: test.info().outputPath('local-fork-outcome.png') })
  await send(page, 'H1 child continuation')
  await expect.poll(async () => (await readStore(page, 'messages')).filter(row => row.history_id === child.id).length).toBe(4)
  const childRows = (await readStore(page, 'messages')).filter(row => row.history_id === child.id)
  expect(childRows.find(row => row.content === 'H1 child continuation').parent_message_id).toBe(copiedAssistant.id)
  const savedAnswer = childRows.find(row => row.content === 'H1 deterministic reply')
  await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.conversation_id === child.id && row.view.cursor.message_id === savedAnswer.id)).toBe(true)
  expect((await readStore(page, 'messages')).filter(row => row.history_id === 'h1-source')).toEqual(source)
  const request = server.requests.filter(row => row.path.endsWith('/chat/completions')).at(-1)!
  expect(request.body.messages.map((row: any) => row.content)).toEqual(expect.arrayContaining(['H1 original question', 'H1 variant A', 'H1 child continuation']))
  expect(JSON.stringify(request.body.messages)).not.toContain('H1 variant B')
  await page.reload()
  await expect(page.getByText('H1 child continuation', { exact: true })).toBeVisible()
  await expect(page.getByText('H1 deterministic reply', { exact: true })).toBeVisible()
  expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(0)
  return child.id as string
}

export async function legacyBoundaries(page: Page) {
  await review(page)
  await page.locator('[data-history-message-id="h1-b"] input[type="checkbox"]').uncheck()
  await page.getByRole('button', { name: 'Before first included message', exact: true }).click()
  await page.getByRole('button', { name: 'Confirm selected history', exact: true }).click()
  await expect.poll(async () => (await readStore(page, 'historyProjections')).length).toBe(1)
  const projection = (await readStore(page, 'historyProjections'))[0]
  expect(projection.ordered_path_ids).toEqual(['h1-user', 'h1-a'])
  expect(projection.source_members.map((row: any) => row.id).sort()).toEqual(['h1-a', 'h1-b', 'h1-user'])
  await page.reload()
  expect((await readStore(page, 'historySelections')).some(row => row.client_session_id !== 'fixture-origin' && row.view.cursor.kind === 'before_message' && row.view.cursor.message_id === 'h1-user')).toBe(true)
  await expect(page.getByText('H1 original question', { exact: true })).toHaveCount(0)
  await page.getByRole('button', { name: 'Use no prior messages', exact: true }).click()
  await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.client_session_id !== 'fixture-origin' && row.view.cursor.kind === 'empty')).toBe(true)
  await page.reload()
  await expect(page.getByRole('button', { name: 'Use no prior messages', exact: true })).toBeVisible()
  expect((await readStore(page, 'historySelections')).some(row => row.client_session_id !== 'fixture-origin' && row.view.cursor.kind === 'empty')).toBe(true)
  expect((await readStore(page, 'messages')).length).toBe(3)
}

export async function seedSidepanelReference(page: Page, id = 'h1-source') {
  await page.evaluate(async historyId => {
    const native = historyId === 'native-source'
    const state = {
      tabs: [{ id: 'h1-tab', label: 'H1 Source', historyId: native ? null : historyId, serverChatId: native ? historyId : null, serverChatTopic: null, updatedAt: Date.now() }],
      activeTabId: 'h1-tab',
      snapshotsById: { 'h1-tab': {
        historySelectionReference: { profile_id: 'h1-profile', client_session_id: native ? 'fixture-native' : 'fixture-origin', owner_key: native ? 'native-h1-owner' : 'local-history-v1:h1-profile', conversation_id: historyId },
        history: [], messages: [], historyId: native ? null : historyId, chatMode: 'normal', webSearch: false, toolChoice: 'none', selectedModel: 'tldw:h1-model', selectedSystemPrompt: null, selectedQuickPrompt: null, temporaryChat: false, useOCR: false,
        serverChatId: native ? historyId : null, serverChatState: null, serverChatTopic: null, serverChatClusterId: null, serverChatSource: null, serverChatExternalRef: null, queuedMessages: [], modelSettings: {}
      } }
    }
    await chrome.storage.local.set({ sidepanelChatTabsState: JSON.stringify(state) })
  }, id)
}

/** Abort the actual production child-write transaction, after its first child row. */
export async function abortLocalFork(page: Page) {
  await choose(page, 'h1-a')
  const before = await readStore(page, 'messages')
  await page.evaluate(() => {
    const original = IDBObjectStore.prototype.add
    ;(window as any).__h1AbortCount = 0
    IDBObjectStore.prototype.add = function(value: any, key?: IDBValidKey) {
      const request = key === undefined ? original.call(this, value) : original.call(this, value, key)
      if (this.name === 'messages' && value.history_id !== 'h1-source') {
        const tx = this.transaction
        request.addEventListener('success', () => {
          ;(window as any).__h1AbortCount++
          tx.abort()
          IDBObjectStore.prototype.add = original
        }, { once: true })
      }
      return request
    }
  })
  await forkAt(page, 'h1-a')
  await expect.poll(() => page.evaluate(() => (window as any).__h1AbortCount)).toBeGreaterThan(0)
  await expect.poll(async () => (await readStore(page, 'forkOperations')).some(row => row.state !== 'dispatching')).toBe(true)
  expect((await readStore(page, 'chatHistories')).length).toBe(1)
  expect(await readStore(page, 'messages')).toEqual(before)
  await page.reload()
  await expect(page.getByText('H1 variant A', { exact: true })).toBeVisible()
  expect((await readStore(page, 'chatHistories')).length).toBe(1)
  expect(await readStore(page, 'messages')).toEqual(before)
}

export async function largeLegacyReview(page: Page, fullTip = false, bubble = false) {
  await review(page)
  await expect(page.getByText('Complete source / included path: 20001 / 20001', { exact: true })).toBeVisible()
  const list = page.getByRole('list', { name: 'Complete source messages' })
  expect(await list.getByRole('listitem').count()).toBeLessThan(30)
  await list.evaluate(element => { element.scrollTop = element.scrollHeight })
  await expect(page.locator('[data-history-message-id="legacy-20000"]')).toBeVisible()
  await page.locator('[data-history-message-id="legacy-20000"] input[type="checkbox"]').uncheck()
  await page.getByRole('button', { name: fullTip ? 'Through last included message' : 'Before first included message', exact: true }).click()
  await page.screenshot({ path: test.info().outputPath('legacy-20001-review.png') })
  await page.getByRole('button', { name: 'Confirm selected history', exact: true }).click()
  await expect.poll(async () => (await readStore(page, 'historyProjections')).length, { timeout: 30000 }).toBe(1)
  const projection = (await readStore(page, 'historyProjections'))[0]
  expect(projection.source_members).toHaveLength(20001)
  expect(projection.ordered_path_ids).toHaveLength(20000)
  expect(projection.ordered_path_ids).not.toContain('legacy-20000')
  expect((await readStore(page, 'messages'))).toHaveLength(20001)
  if (fullTip) {
    const transcript = page.getByRole('log', { name: 'Chat messages' })
    await transcript.evaluate(element => { element.scrollTop = element.scrollHeight })
    await expect(page.locator('[data-message-id="legacy-19999"][data-testid="chat-message"]')).toBeVisible()
    expect(await page.getByTestId('chat-message').count()).toBeLessThan(100)
    await page.reload()
    await expect(page.getByTestId('chat-message').first()).toBeVisible()
    await transcript.evaluate(element => { element.scrollTop = element.scrollHeight })
    await expect(page.locator('[data-message-id="legacy-19999"][data-testid="chat-message"]')).toBeVisible()
    await page.keyboard.press('Control+f')
    await page.getByPlaceholder('Search messages in this conversation').fill('Legacy row 12345')
    await expect(page.locator('[data-message-id="legacy-12345"][data-testid="chat-message"]')).toBeVisible()
    await page.evaluate(() => window.dispatchEvent(new CustomEvent('tldw:timeline-action', { detail: { action: 'edit', historyId: 'h1-source', messageId: 'legacy-42' } })))
    await expect(page.locator('[data-message-id="legacy-42"] textarea')).toBeVisible()
    const editor = page.locator('[data-message-id="legacy-42"] textarea')
    const assertEditorFits = async () => {
      const bounds = await editor.evaluate(element => {
        const form = element.closest('form')!
        const parent = document.querySelector('[role="log"][aria-label="Chat messages"]')!
        const box = (node: Element) => { const r = node.getBoundingClientRect(); return { left: r.left, right: r.right, width: r.width } }
        return { scrollport: box(parent), elements: [element, form, ...form.querySelectorAll('button')].map(node => ({ name: node.tagName === 'BUTTON' ? node.textContent : node.tagName, ...box(node) })) }
      })
      await test.info().attach('bubble-editor-horizontal-bounds', { body: JSON.stringify(bounds), contentType: 'application/json' })
      for (const item of bounds.elements) {
        expect(item.left, item.name ?? 'editor').toBeGreaterThanOrEqual(bounds.scrollport.left)
        expect(item.right, item.name ?? 'editor').toBeLessThanOrEqual(bounds.scrollport.right)
      }
    }
    const sourceBefore = await readStore(page, 'messages')
    await editor.fill('UNSAVED REVIEW DRAFT')
    await transcript.evaluate(element => { element.scrollTop = element.scrollHeight })
    await expect(page.locator('[data-message-id="legacy-19999"][data-testid="chat-message"]')).toBeVisible()
    expect(await page.getByTestId('chat-message').count()).toBeLessThan(100)
    await page.getByPlaceholder('Search messages in this conversation').fill('Legacy row 42')
    await expect(editor).toBeVisible()
    await expect(editor).toBeInViewport()
    await expect(editor).toHaveValue('UNSAVED REVIEW DRAFT')
    await page.screenshot({ path: test.info().outputPath('legacy-full-tip-navigation.png') })
    await page.locator('[data-message-id="legacy-42"]').getByRole('button', { name: 'Save', exact: true }).click()
    await expect.poll(async () => (await readStore(page, 'messages')).find(row => row.id === 'legacy-42').content).toBe('UNSAVED REVIEW DRAFT')
    expect((await readStore(page, 'messages')).filter(row => row.id !== 'legacy-42')).toEqual(sourceBefore.filter(row => row.id !== 'legacy-42'))
    await page.getByPlaceholder('Search messages in this conversation').fill('UNSAVED REVIEW DRAFT')
    if (bubble) await expect(page.locator('[data-message-id="legacy-42"] .message-bubble')).toBeVisible()
    await page.evaluate(() => window.dispatchEvent(new CustomEvent('tldw:timeline-action', { detail: { action: 'edit', historyId: 'h1-source', messageId: 'legacy-42' } })))
    if (bubble) await expect(page.locator('[data-message-id="legacy-42"] .message-bubble textarea')).toBeVisible()
    await editor.fill('CANCEL ONLY THIS DRAFT')
    await transcript.evaluate(element => { element.scrollTop = element.scrollHeight })
    await expect(page.locator('[data-message-id="legacy-19999"][data-testid="chat-message"]')).toBeVisible()
    await page.getByPlaceholder('Search messages in this conversation').fill('UNSAVED REVIEW')
    await expect(editor).toBeVisible()
    await expect(editor).toBeInViewport()
    await expect(editor).toHaveValue('CANCEL ONLY THIS DRAFT')
    if (bubble) {
      await assertEditorFits()
      await page.screenshot({ path: test.info().outputPath('legacy-bubble-retained-editor.png') })
    }
    await page.locator('[data-message-id="legacy-42"]').getByRole('button', { name: 'Cancel', exact: true }).click()
    await expect(editor).toHaveCount(0)
    expect((await readStore(page, 'messages')).find(row => row.id === 'legacy-42').content).toBe('UNSAVED REVIEW DRAFT')
    await transcript.evaluate(element => { element.scrollTop = element.scrollHeight })
    await expect(page.locator('[data-message-id="legacy-19999"][data-testid="chat-message"]')).toBeVisible()
    await expect(page.locator('[data-message-id="legacy-42"]')).toHaveCount(0)
    if (bubble) {
      await page.getByPlaceholder('Search messages in this conversation').fill('UNSAVED REVIEW DRAFT')
      await expect(page.locator('[data-message-id="legacy-42"] .message-bubble')).toBeVisible()
      await page.evaluate(() => window.dispatchEvent(new CustomEvent('tldw:timeline-action', { detail: { action: 'edit', historyId: 'h1-source', messageId: 'legacy-42' } })))
      await expect(page.locator('[data-message-id="legacy-42"] .message-bubble textarea')).toBeVisible()
      await editor.fill('SAVED BUBBLE DRAFT')
      await assertEditorFits()
      await page.locator('[data-message-id="legacy-42"]').getByRole('button', { name: 'Save', exact: true }).click()
      await expect.poll(async () => (await readStore(page, 'messages')).find(row => row.id === 'legacy-42').content).toBe('SAVED BUBBLE DRAFT')
      expect((await readStore(page, 'messages')).filter(row => row.id !== 'legacy-42')).toEqual(sourceBefore.filter(row => row.id !== 'legacy-42'))
    }
  }
}

export async function seedNativeReference(page: Page) {
  await seedHistory(page)
  await page.evaluate(async () => {
    const open = indexedDB.open('PageAssistDatabase')
    const db = await new Promise<IDBDatabase>((resolve, reject) => { open.onsuccess = () => resolve(open.result); open.onerror = () => reject(open.error) })
    const tx = db.transaction('historySelections', 'readwrite')
    tx.objectStore('historySelections').put({ profile_id: 'h1-profile', client_session_id: 'fixture-native', owner_key: 'native-h1-owner', conversation_id: 'native-source', view: { view_session_id: 'native-fixture-view', owner_key: 'native-h1-owner', conversation_id: 'native-source', interpretation: { kind: 'parent_graph_v1' }, cursor: { kind: 'after_message', message_id: 'native-a' }, selection_revision: 0 } })
    await new Promise<void>((resolve, reject) => { tx.oncomplete = () => resolve(); tx.onabort = () => reject(tx.error) })
    db.close()
  })
}

export const nativeHistoryUrl = (base: string) => `${base}?historySelection=${encodeURIComponent(JSON.stringify({ profile_id: 'h1-profile', client_session_id: 'fixture-native', owner_key: 'native-h1-owner', conversation_id: 'native-source', owner_kind: 'native' }))}`

export async function nativeForkAndSend(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>) {
  page.setDefaultTimeout(15000)
  await expect(page.getByText('Native variant A', { exact: true })).toBeVisible()
  await page.evaluate(async () => {
    const poison = { schemaVersion: 2, updatedAt: '2099-01-01T00:00:00Z', authorNote: 'POISON AMBIENT SETTINGS' }
    const values = { 'chatSettings:server:native-child-1': poison, 'chatSettings:scratch': poison }
    if (location.protocol === 'chrome-extension:') { await chrome.storage.local.set(values); await chrome.storage.sync.set(values) }
    else for (const [key, value] of Object.entries(values)) localStorage.setItem(key, JSON.stringify(value))
  })
  expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(0)
  await forkAt(page, 'native-a')
  await expect.poll(async () => (await readStore(page, 'forkOperations')).find(row => row.candidate_child_id)?.state).toBe('completed')
  const operation = (await readStore(page, 'forkOperations')).find(row => row.candidate_child_id)
  const child = operation.candidate_child_id
  await expect.poll(() => server.requests.some(row => row.path === `/api/v1/chats/${child}/settings`)).toBe(true)
  expect(server.requests.filter(row => row.method === 'PUT' && row.path === '/api/v1/chats/' + child + '/settings')).toHaveLength(0)
  await send(page, 'Native child continuation')
  await expect.poll(() => server.nativeChats.get(child)?.length).toBe(4)
  expect(server.nativeChats.get(child)!.map(row => row.content)).toEqual(['Native original question', 'Native variant A', 'Native child continuation', 'H1 deterministic reply'])
  expect(server.nativeChats.get('native-source')).toHaveLength(3)
  const last = server.nativeChats.get(child)!.at(-1)
  await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.conversation_id === child && row.view.cursor.message_id === last.id)).toBe(true)
  await page.reload()
  await expect(page.getByText('Native child continuation', { exact: true })).toBeVisible()
  expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(1)
  return child as string
}


export async function unknownNativeFork(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>, secondPage: () => Promise<Page>) {
  await expect(page.getByText('Native variant A', { exact: true })).toBeVisible()
  const browserCreates: string[] = []
  page.on('request', request => { if (request.method() === 'POST' && new URL(request.url()).pathname === '/api/v1/chats/') browserCreates.push(request.url()) })
  await forkAt(page, 'native-a')
  await expect.poll(async () => (await readStore(page, 'forkOperations')).map(row => row.state)).toEqual(['unknown'])
  console.log('H1_UNKNOWN_DISPATCH', { browserCreates: browserCreates.length, serverCreates: server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/').length })
  const before = (await readStore(page, 'forkOperations'))[0]
  expect(before.candidate_child_id).toBeUndefined()
  await page.reload()
  await expect(page.getByText('Fork status unknown', { exact: true })).toBeVisible()
  await expect(page.getByText(/The app will not start another attempt automatically.*multiple server copies/)).toBeVisible()
  const second = await secondPage()
  await expect(second.getByText('Fork status unknown', { exact: true })).toBeVisible()
  await forkAt(second, 'native-a')
  await expect(second.getByText('Fork status unknown', { exact: true })).toBeVisible()
  expect(await readStore(second, 'forkOperations')).toEqual([before])
  expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(1)
  await page.screenshot({ path: test.info().outputPath('unknown-fork-reopened.png') })
}


export async function editNativeChildSettings(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>, child: string) {
  const open = async () => {
    const inspector = page.getByTestId('playground-cockpit-right-rail').getByTestId('playground-runtime-inspector')
    if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Restore runtime sidechannel', exact: true }).click()
    await expect(inspector).toBeVisible()
    await inspector.getByRole('button', { name: 'Open model settings', exact: true }).click()
    const dialog = page.getByRole('dialog', { name: 'Current Chat Model Settings', exact: true })
    await expect(dialog).toBeVisible()
    await dialog.getByRole('tab', { name: 'Conversation', exact: true }).click()
    return dialog
  }
  let dialog = await open()
  const placeholder = 'E.g., Keep responses grounded, avoid repetition, and progress the scene.'
  await expect(dialog.getByPlaceholder(placeholder)).toHaveValue('')
  await dialog.getByPlaceholder(placeholder).scrollIntoViewIfNeeded()
  await dialog.getByPlaceholder(placeholder).click()
  await dialog.getByPlaceholder(placeholder).fill('Legitimate child note')
  await expect(dialog.getByPlaceholder(placeholder)).toHaveValue('Legitimate child note')
  await dialog.getByPlaceholder(placeholder).press('Tab')
  await page.screenshot({ path: test.info().outputPath('native-settings-edit.png'), fullPage: true })
  await expect.poll(() => server.nativeSettings.get(child)?.authorNote).toBe('Legitimate child note')
  const writes = server.requests.filter(row => row.method === 'PUT' && row.path === '/api/v1/chats/' + child + '/settings')
  expect(writes.map(row => row.body.settings)).toEqual([{ authorNote: 'Legitimate child note' }])
  await dialog.getByRole('button', { name: 'Close', exact: true }).click()
  await page.reload()
  await expect(page.getByText('Native child continuation', { exact: true })).toBeVisible()
  dialog = await open()
  await expect(dialog.getByPlaceholder(placeholder)).toHaveValue('Legitimate child note')
  await dialog.getByPlaceholder(placeholder).fill('Edited again after reopen')
  await dialog.getByPlaceholder(placeholder).blur()
  await expect.poll(() => server.nativeSettings.get(child)?.authorNote).toBe('Edited again after reopen')
  expect(server.nativeSettings.has('native-source')).toBe(false)
}


export async function heldNativeForkNavigation(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>, base: string, boundary: 'create' | 'settings', live = false) {
  const held = server.holdNext(boundary === 'create' ? 'POST' : 'GET', boundary === 'create' ? '/api/v1/chats/' : '/api/v1/chats/native-child-1/settings')
  try {
    await expect(page.getByText('Native variant A', { exact: true })).toBeVisible()
    await forkAt(page, 'native-a')
    await held.started
    expect(server.nativeChats.size).toBe(2)
    if (boundary === 'settings') {
      expect(server.nativeChats.get('native-child-1')).toHaveLength(2)
      expect((await readStore(page, 'forkOperations'))[0].state).toBe('completed')
    }
    if (live) await page.evaluate(() => { (window as any).__h1LivePage = true; window.dispatchEvent(new CustomEvent('tldw:open-history', { detail: { historyId: 'h1-source', messageId: 'h1-b' } })) })
    else { await page.goto(historyUrl(base)); if (base.includes('#')) await page.reload() }
    await expect(page.getByText('H1 variant B', { exact: true })).toBeVisible()
    held.release()
    if (live) {
      await expect.poll(async () => (await readStore(page, 'forkOperations'))[0]?.state).toBe(boundary === 'create' ? 'partial' : 'completed')
      expect(await page.evaluate(() => (window as any).__h1LivePage)).toBe(true)
      expect((await readStore(page, 'forkOperations'))[0].candidate_child_id).toBe('native-child-1')
      if (boundary === 'create') {
        expect(server.nativeChats.get('native-child-1')).toEqual([])
        expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/native-child-1/messages')).toEqual([])
      }
    }
    await expect(page.getByText('H1 variant B', { exact: true })).toBeVisible()
    expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(1)
    expect(server.nativeChats.get('native-source')).toHaveLength(3)
    expect(server.requests.filter(row => row.method === 'PUT')).toHaveLength(0)
    await page.goto(nativeHistoryUrl(base))
    if (base.includes('#')) await page.reload()
    await expect(page.getByText(boundary === 'create' ? live ? 'Fork incomplete' : 'Fork status unknown' : 'Fork saved', { exact: true })).toBeVisible()
    if (boundary === 'create') await expect(page.getByText(/The app will not start another attempt automatically.*multiple server copies/)).toBeVisible()
    const operation = (await readStore(page, 'forkOperations'))[0]
    expect(operation.owner_key).toBe('native-h1-owner')
    expect(operation.conversation_id).toBe('native-source')
    expect(operation.state).toBe(boundary === 'create' ? live ? 'partial' : 'unknown' : 'completed')
  } finally { held.release() }
}

export async function rejectNativeMetadataSettings(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>, child: string) {
  const operation = (await readStore(page, 'forkOperations')).find(row => row.candidate_child_id === child)
  const settingsWrites = () => server.requests.filter(row => row.method === 'PUT' && row.path.endsWith('/settings'))
  const before = settingsWrites().length
  server.nativeMetadataOverrides.set(child, { scope_type: 'workspace', workspace_id: 'foreign' })
  await page.reload()
  await expect(page.getByTestId('playground-chat-shell').getByText('Selected history unavailable', { exact: true })).toBeVisible()
  await expect(page.getByTestId('playground-chat-shell').getByText('server_chat_scope_mismatch', { exact: true })).toHaveCount(1)
  const inspector = page.getByTestId('playground-cockpit-right-rail').getByTestId('playground-runtime-inspector')
  if (!await inspector.isVisible()) await page.getByRole('button', { name: 'Restore runtime sidechannel', exact: true }).click()
  await inspector.getByRole('button', { name: 'Open model settings', exact: true }).click()
  const dialog = page.getByRole('dialog', { name: 'Current Chat Model Settings', exact: true })
  await dialog.getByRole('tab', { name: 'Conversation', exact: true }).click()
  const note = dialog.getByPlaceholder('E.g., Keep responses grounded, avoid repetition, and progress the scene.')
  await expect(note).toHaveValue('')
  await note.scrollIntoViewIfNeeded()
  await note.click()
  await note.fill('REJECTED NATIVE NOTE')
  await note.press('Tab')
  await expect(page.getByText('fork_settings_owner_unavailable', { exact: true })).toBeVisible()
  const feedback = page.getByText('fork_settings_owner_unavailable', { exact: true })
  await feedback.evaluate(async element => {
    const notice = element.closest('.ant-notification-notice') ?? element
    await Promise.all(notice.getAnimations({ subtree: true }).map(animation => animation.finished.catch(() => undefined)))
  })
  await expect.poll(() => feedback.evaluate(element => {
    const rect = element.getBoundingClientRect()
    const hit = document.elementFromPoint(rect.x + rect.width / 2, rect.y + rect.height / 2)
    return { opacity: getComputedStyle(element).opacity, onTop: hit === element || element.contains(hit) }
  })).toEqual({ opacity: '1', onTop: true })
  console.log('H1_FEEDBACK_CAPTURE', await feedback.boundingBox())
  await feedback.screenshot({ path: test.info().outputPath('rejected-native-feedback-detail.png'), animations: 'disabled' })
  await page.screenshot({ path: test.info().outputPath('rejected-native-feedback.png'), fullPage: true, animations: 'disabled' })
  expect(settingsWrites()).toHaveLength(before)
  const notes = await page.evaluate(async () => {
    const values = location.protocol === 'chrome-extension:' ? { ...await chrome.storage.sync.get(null), ...await chrome.storage.local.get(null) } : Object.fromEntries(Object.keys(localStorage).map(key => [key, localStorage.getItem(key)]))
    return Object.entries(values).filter(([key]) => key.startsWith('chatSettings:')).map(([key, value]) => { let parsed = value; if (typeof value === 'string') { try { parsed = JSON.parse(value) } catch {} } return { key, note: (parsed as any)?.authorNote } })
  })
  expect(notes.some(row => row.note === 'REJECTED NATIVE NOTE')).toBe(false)
  expect((await readStore(page, 'forkOperations')).find(row => row.candidate_child_id === child)).toEqual(operation)
  await page.screenshot({ path: test.info().outputPath('rejected-native-settings.png'), fullPage: true })
}

async function selectNativeCharacter(page: Page) {
  await page.evaluate(async () => {
    const values = { selectedAssistant: { kind: 'character', id: '7', name: 'H1 Character', metadata: { selectionMode: 'tracked' } }, selectedModel: 'tldw:openai:h1-model' }
    if (location.protocol === 'chrome-extension:') { await chrome.storage.local.set(values); await chrome.storage.sync.set({ selectedModel: JSON.stringify(values.selectedModel) }) }
    else for (const [key, value] of Object.entries(values)) localStorage.setItem(key, typeof value === 'string' ? value : JSON.stringify(value))
  })
  await page.reload()
  await expect(page.getByTestId('chat-input')).toBeVisible()
  await observeQualificationErrors(page)
}

async function observeQualificationErrors(page: Page) {
  await page.evaluate(() => {
    const values: string[] = []
    ;(window as any).__h1QualificationErrors = values
    new MutationObserver(() => {
      for (const node of document.querySelectorAll('.ant-notification-notice-description, .ant-message-custom-content')) {
        const text = node.textContent || ''
        if (!values.includes(text)) values.push(text)
      }
    }).observe(document.body, { childList: true, subtree: true })
  })
}

export async function heldNativeFirstCreate(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>, base: string, live = false) {
  await selectNativeCharacter(page)
  await expect(page.getByTestId('chat-input')).toBeVisible()
  const held = server.holdNext('POST', '/api/v1/chats/')
  const context = page.context()
  let settle!: (value: { event: 'finished' | 'failed'; error?: string }) => void
  const requestSettled = new Promise<{ event: 'finished' | 'failed'; error?: string }>(resolve => { settle = resolve })
  const matches = (request: Request) => request.method() === 'POST' && request.url() === server.url + '/api/v1/chats/'
  const finished = (request: Request) => { if (matches(request)) settle({ event: 'finished' }) }
  const failed = (request: Request) => { if (matches(request)) settle({ event: 'failed', error: request.failure()?.errorText }) }
  context.on('requestfinished', finished)
  context.on('requestfailed', failed)
  try {
    await page.getByTestId('chat-input').fill('Held first character send')
    await page.getByRole('button', { name: 'Send message', exact: true }).click()
    await held.started
    expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/').map(row => row.body)).toEqual([{ character_id: '7', scope_type: 'global' }])
    if (live) await page.evaluate(() => { (window as any).__h1LivePage = true; window.dispatchEvent(new CustomEvent('tldw:open-history', { detail: { historyId: 'h1-source', messageId: 'h1-b' } })) })
    else { await page.goto(historyUrl(base)); if (base.includes('#')) await page.reload() }
    await expect(page.getByText('H1 variant B', { exact: true })).toBeVisible()
    if (live) {
      expect(await page.evaluate(() => (window as any).__h1LivePage)).toBe(true)
      expect(await page.evaluate(() => (window as Window & { __h1QualificationErrors?: string[] }).__h1QualificationErrors ?? [])).not.toContain('stale_selection')
    }
    held.release()
    const [delivery, request] = await Promise.all([held.responseSettled, requestSettled])
    if (request.event === 'failed') expect(request.error).toMatch(/abort|cancel|closed/i)
    if (live && request.event === 'finished') {
      // Both actual native handlers report this terminal catch after rejecting the late ACK.
      await expect.poll(() => page.evaluate(() => (window as Window & { __h1QualificationErrors?: string[] }).__h1QualificationErrors ?? [])).toContain('stale_selection')
    }
    console.log('H1_FIRST_CREATE_SETTLED', { live, delivery, request })
    await expect(page.getByRole('button', { name: /Stop (streaming|generation)/i })).toHaveCount(0)
    await expect(page.getByText('H1 variant B', { exact: true })).toBeVisible()
    expect(server.nativeChats.get('native-child-1')).toEqual([])
    expect(server.requests.filter(row => row.path.endsWith('/chat/completions'))).toEqual([])
    expect(server.requests.filter(row => row.method === 'POST' && row.path.includes('/messages'))).toEqual([])
  } finally { console.log('H1_FIRST_CREATE_DIAGNOSTICS', JSON.stringify({ errors: await page.evaluate(() => (window as any).__h1QualificationErrors).catch(() => []), requests: server.requests.filter(row => row.method === 'POST') })); held.release(); context.off('requestfinished', finished); context.off('requestfailed', failed) }
}

export async function firstNativeCharacterSend(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>) {
  await selectNativeCharacter(page)
  try { await send(page, 'First native character question') } catch (error) {
    console.log('H1_CHARACTER_FAILURE', JSON.stringify({ errors: await page.evaluate(() => (window as any).__h1QualificationErrors), requests: server.requests.filter(row => row.method === 'POST') }))
    throw error
  }
  await expect.poll(() => server.nativeChats.get('native-child-1')?.length).toBe(2)
  await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.conversation_id === 'native-child-1' && row.view.cursor.message_id === 'native-child-1-row-1')).toBe(true)
  const request = server.requests.find(row => row.method === 'POST' && row.path.endsWith('/chat/completions'))!
  expect(request.body.conversation_id).toBe('native-child-1')
  expect(request.body.tldw_history_selection_v1.messages).toEqual([])
  expect(request.body.save_to_db).toBe(true)
  await page.reload()
  await expect(page.locator('[data-message-id]').getByText('First native character question', { exact: true })).toBeVisible()
  await send(page, 'Second native character question')
  await expect.poll(() => server.nativeChats.get('native-child-1')?.length).toBe(4)
  expect(server.nativeChats.get('native-child-1')?.map(row => row.parent_message_id)).toEqual([null, 'native-child-1-row-0', 'native-child-1-row-1', 'native-child-1-row-2'])
  expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(1)
  expect(server.requests.filter(row => row.method === 'POST' && row.path.endsWith('/messages'))).toHaveLength(0)
  await page.screenshot({ path: test.info().outputPath('native-character-receipt.png'), fullPage: true })
}

export async function qualifyChildIsolation(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>, base: string, comparison: boolean, expand?: () => Promise<Page>) {
  await injectHistoryStorage(page)
  await page.evaluate(async compare => {
    const a = (window as any).__h1Storage
    await a.addFileToSession('h1-source', { id: 'source-file', filename: 'h1-source.txt', type: 'text/plain', content: 'H1 copied document', size: 18, uploadedAt: 1, processed: true, ingestJobId: 'source-job', ingestBatchId: 'source-batch', ingestIdempotencyKey: 'source-idempotency', documentDraftId: 'source-document-draft', processingResultRef: { kind: 'ingest_job', id: 'source-job' }, processingStatus: 'ready', processingMode: 'ingest_to_library' })
    if (compare) {
      if (location.protocol === 'chrome-extension:') await chrome.storage.sync.set({ ff_compareMode: true })
      else localStorage.setItem('ff_compareMode', 'true')
      for (const [id, type, model] of [['h1-user', 'compare:user', null], ['h1-a', 'compare:reply', 'tldw:h1-model'], ['h1-b', 'compare:reply', 'tldw:alternate-model']]) await a.db.messages.update(id, { messageType: type, clusterId: 'h1-cluster', ...(model ? { modelId: model } : {}) })
      const common = { history_id: 'h1-source', name: 'Assistant', images: [], clusterId: 'h1-round-2', history_provenance: { owner_key: 'local-history-v1:h1-profile', projection_id: null } }
      await a.db.messages.bulkAdd([
        { ...common, id: 'h1-user2', role: 'user', content: 'H1 second common prompt', createdAt: 4, parent_message_id: 'h1-b', messageType: 'compare:user' },
        { ...common, id: 'h1-a2', role: 'assistant', content: 'H1 second A answer', createdAt: 5, parent_message_id: 'h1-user2', messageType: 'compare:reply', modelId: 'tldw:h1-model' },
        { ...common, id: 'h1-b2', role: 'assistant', content: 'H1 second B answer', createdAt: 6, parent_message_id: 'h1-user2', messageType: 'compare:reply', modelId: 'tldw:alternate-model' }
      ])
      await a.db.compareStates.put({ history_id: 'h1-source', compareMode: true })
    }
  }, comparison)
  const source = (await readStore(page, 'messages')).filter(row => row.history_id === 'h1-source')
  const sourceFiles = (await readStore(page, 'sessionFiles')).find(row => row.sessionId === 'h1-source')
  if (expand) page = await expand()
  else { await page.goto(historyUrl(base)); if (base.includes('#')) await page.reload() }
  let child: string
  if (comparison) {
    await expect(page.getByTestId('compare-model-identity-h1-round-2-tldw:h1-model')).toBeVisible()
    await expect(page.getByText('Selected history unavailable', { exact: true })).toBeVisible()
    await page.reload()
    await expect(page.getByTestId('compare-model-identity-h1-round-2-tldw:h1-model')).toBeVisible()
    await expect(page.getByText('Selected history unavailable', { exact: true })).toBeVisible()
    await observeQualificationErrors(page)
    expect((await readStore(page, 'compareStates')).find(row => row.history_id === 'h1-source')?.compareMode).toBe(true)
    await forkAt(page, 'h1-a2')
    await expect.poll(async () => (await readStore(page, 'chatHistories')).length).toBe(2)
    child = (await readStore(page, 'chatHistories')).find(row => row.id !== 'h1-source').id
    const copied = (await readStore(page, 'messages')).filter(row => row.history_id === child)
    expect(copied.map(row => row.content).sort()).toEqual(['H1 original question', 'H1 variant A', 'H1 second common prompt', 'H1 second A answer'].sort())
    expect(copied.find(row => row.content === 'H1 second common prompt').parent_message_id).toBe(copied.find(row => row.content === 'H1 variant A').id)
    expect(copied.some(row => row.messageType || row.clusterId)).toBe(false)
    expect(copied.filter(row => row.role === 'assistant').map(row => row.modelId)).toEqual(['tldw:h1-model', 'tldw:h1-model'])
    await send(page, 'H1 child continuation')
    await expect.poll(async () => (await readStore(page, 'messages')).filter(row => row.history_id === child).length).toBe(6)
    const savedAnswer = (await readStore(page, 'messages')).find(row => row.history_id === child && row.content === 'H1 deterministic reply')
    await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.conversation_id === child && row.view.cursor.message_id === savedAnswer.id)).toBe(true)
    const request = server.requests.filter(row => row.path.endsWith('/chat/completions')).at(-1)!
    expect(JSON.stringify(request.body.messages)).not.toContain('H1 variant B')
    expect(JSON.stringify(request.body.messages)).not.toContain('H1 second B answer')
    expect(request.body.messages.map((row: any) => row.content)).toEqual(expect.arrayContaining(['H1 variant A', 'H1 second common prompt', 'H1 second A answer']))
    await page.reload()
    await expect(page.getByText('H1 child continuation', { exact: true })).toBeVisible()
  } else child = await localForkAndSend(page, server)
  const files = (await readStore(page, 'sessionFiles')).find(row => row.sessionId === child)
  expect(files.files).toHaveLength(1)
  expect(files.files[0].id).not.toBe('source-file')
  expect(files.files[0].content).toBe('H1 copied document')
  for (const field of ['ingestJobId', 'ingestBatchId', 'ingestIdempotencyKey', 'documentDraftId', 'processingResultRef', 'processingStatus', 'processingMode']) expect(files.files[0]).not.toHaveProperty(field)
  await injectHistoryStorage(page)
  await page.evaluate(async ({ child, id }) => { await (window as any).__h1Storage.removeFileFromSession(child, id) }, { child, id: files.files[0].id })
  expect((await readStore(page, 'sessionFiles')).find(row => row.sessionId === child).files).toEqual([])
  const user = (await readStore(page, 'messages')).find(row => row.history_id === child && row.content === 'H1 original question')
  const card = page.locator(`[data-message-id="${user.id}"]`)
  await card.hover()
  await card.getByRole('button', { name: 'Edit', exact: true }).click()
  await card.locator('textarea').fill('Edited only in child')
  await card.getByRole('button', { name: 'Save', exact: true }).click()
  await expect.poll(async () => (await readStore(page, 'messages')).find(row => row.id === user.id)?.content).toBe('Edited only in child')
  const answer = (await readStore(page, 'messages')).find(row => row.history_id === child && row.content === 'H1 deterministic reply')
  const answerCard = page.locator(`[data-message-id="${answer.id}"]`)
  await answerCard.hover()
  await answerCard.getByRole('button', { name: 'More actions', exact: true }).click()
  await page.getByRole('button', { name: 'Delete', exact: true }).click()
  await page.getByRole('dialog').getByRole('button', { name: 'Delete', exact: true }).click()
  await expect.poll(async () => (await readStore(page, 'messages')).some(row => row.id === answer.id)).toBe(false)
  await page.reload()
  await expect(page.getByText('Edited only in child', { exact: true })).toBeVisible()
  expect((await readStore(page, 'messages')).filter(row => row.history_id === 'h1-source')).toEqual(source)
  expect((await readStore(page, 'sessionFiles')).find(row => row.sessionId === 'h1-source')).toEqual(sourceFiles)
}

export async function qualifyForeignWorkspaceAndAccountRead(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>, base: string) {
  const child = await nativeForkAndSend(page, server)
  await injectHistoryStorage(page)
  await page.evaluate(async () => {
    const a = (window as any).__h1Storage
    const original = (await a.db.forkOperations.toArray())[0]
    const context = { kind: 'native', scope: { type: 'workspace', workspaceId: 'foreign-workspace' } }
    const source = (conversation_id: string) => a.historyDigest({ owner_key: original.owner_key, conversation_id, ...context })
    await a.db.forkOperations.put({ ...original, operation_id: 'foreign-workspace-operation', context, state: 'partial', candidate_child_id: 'foreign-workspace-child', source_key: source(original.conversation_id), candidate_key: source('foreign-workspace-child') })
  })
  await page.goto(nativeHistoryUrl(base))
  if (base.includes('#')) await page.reload()
  await expect(page.getByText('Native variant A', { exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Inspect copy: ' + child, exact: true })).toBeVisible()
  await expect(page.getByRole('button', { name: 'Inspect copy: foreign-workspace-child', exact: true })).toHaveCount(0)
  expect(server.requests.some(row => row.path.includes('foreign-workspace'))).toBe(false)
  const operations = await readStore(page, 'forkOperations')
  expect(operations).toHaveLength(2)
  const held = server.holdNext('POST', '/api/v1/chat/conversations/native-source/history/selection')
  const settingsPage = await page.context().newPage()
  try {
    await settingsPage.goto(base)
    await expect(settingsPage.getByTestId('chat-input')).toBeVisible()
    await page.reload()
    await held.started
    await settingsPage.evaluate(async () => {
      if (location.protocol === 'chrome-extension:') {
        const { tldwConfig } = await chrome.storage.local.get('tldwConfig')
        const value = typeof tldwConfig === 'string' ? JSON.parse(tldwConfig) : tldwConfig
        await chrome.storage.local.set({ tldwConfig: { ...value, apiKey: 'h1-other-account-key' } })
      } else {
        const value = JSON.parse(localStorage.getItem('tldwConfig')!)
        localStorage.setItem('tldwConfig', JSON.stringify({ ...value, apiKey: 'h1-other-account-key' }))
      }
    })
    await expect(page.getByTestId('playground-chat-shell').getByText('Selected history unavailable', { exact: true })).toBeVisible()
    await expect(page.getByTestId('playground-chat-shell').getByText('request_config_scope_changed', { exact: true })).toHaveCount(1)
    held.release()
    await expect(page.getByRole('button', { name: /^Inspect copy:/ })).toHaveCount(0)
    expect(await readStore(page, 'forkOperations')).toEqual(operations)
    expect(server.requests.some(row => row.path.includes('foreign-workspace'))).toBe(false)
    expect(server.requests.filter(row => row.method === 'PUT' && row.path.endsWith('/settings'))).toEqual([])
    await page.screenshot({ path: test.info().outputPath('account-invalidated-history.png'), fullPage: true })
  } finally { held.release(); await settingsPage.close() }
}

export async function uncertainNativeCharacterSend(page: Page, server: Awaited<ReturnType<typeof startHistoryServer>>) {
 await selectNativeCharacter(page)
 await page.getByTestId('chat-input').fill('Uncertain native character question')
 await page.getByRole('button', { name: 'Send message', exact: true }).click()
 await expect(page.getByText('Turn needs review', { exact: true })).toBeVisible()
 const pending = () => readStore(page, 'historySelections').then(rows => rows.flatMap(row => Object.values(row.pending_turns || {})))
 await expect.poll(async () => (await pending()).length).toBe(1)
 const original = await pending()
 expect(server.nativeChats.get('native-child-1')).toHaveLength(2)
 expect((await readStore(page, 'historySelections')).some(row => row.conversation_id === 'native-child-1' && row.view.cursor.kind === 'empty')).toBe(true)
 await page.reload()
 await expect(page.getByText('Turn needs review', { exact: true })).toBeVisible()
 expect(await pending()).toEqual(original)
 expect(server.requests.filter(row => row.method === 'POST' && row.path.endsWith('/chat/completions'))).toHaveLength(1)
 await page.screenshot({ path: test.info().outputPath('native-character-uncertain.png'), fullPage: true })
}

export async function sendIndependentSourceViews(first: Page, second: Page, server: Awaited<ReturnType<typeof startHistoryServer>>) {
  const reference = (page: Page) => page.evaluate(() => JSON.parse(sessionStorage.getItem('tldw-h1-playground-reference')!))
  const original = [await reference(first), await reference(second)]
  expect(original[0].client_session_id).not.toBe(original[1].client_session_id)
  const restored: string[] = []
  const held = server.holdNext('POST', '/api/v1/chat/completions')
  try {
    await first.getByTestId('chat-input').fill('Source A continuation')
    await first.getByRole('button', { name: 'Send message', exact: true }).click()
    await held.started
    await send(second, 'Source B continuation')
    await expect.poll(async () => (await readStore(second, 'messages')).filter(row => row.content === 'H1 deterministic reply').length).toBe(1)
    held.release()
    await expect.poll(async () => (await readStore(first, 'messages')).filter(row => row.content === 'H1 deterministic reply').length).toBe(2)
    const completions = server.requests.filter(row => row.method === 'POST' && row.path.endsWith('/chat/completions'))
    expect(completions).toHaveLength(2)
    const rows = await readStore(first, 'messages')
    expect(rows).toHaveLength(7)
    const results: string[] = []
    for (const [variant, page] of [['A', first], ['B', second]] as const) {
      const input = rows.find(row => row.content === `Source ${variant} continuation`)
      expect(input.parent_message_id).toBe(`h1-${variant.toLowerCase()}`)
      const result = rows.find(row => row.parent_message_id === input.id && row.role === 'assistant')
      expect(result.content).toBe('H1 deterministic reply')
      results.push(result.id)
      const request = completions.find(row => row.body.messages.some((message: { content: string }) => message.content === input.content))!
      expect(request.body.messages.map((message: { content: string }) => message.content)).toEqual(['H1 original question', `H1 variant ${variant}`, input.content])
      await expect.poll(async () => (await readStore(page, 'historySelections')).some(row => row.view.cursor.message_id === result.id)).toBe(true)
      const origin = original[variant === 'A' ? 0 : 1]
      const beforeReload = (await readStore(page, 'historySelections')).find(row => row.client_session_id === origin.client_session_id)
      expect(beforeReload.view.cursor).toEqual({ kind: 'after_message', message_id: result.id })
      await page.reload()
      await expect(page.locator('[data-message-id]').getByText(input.content, { exact: true })).toBeVisible()
      await expect(page.locator('[data-message-id]').getByText(`H1 variant ${variant === 'A' ? 'B' : 'A'}`, { exact: true })).toHaveCount(0)
      const fresh = await reference(page)
      expect(fresh.client_session_id).not.toBe(origin.client_session_id)
      restored.push(fresh.client_session_id)
      expect((await readStore(page, 'historySelections')).find(row => row.client_session_id === fresh.client_session_id).view.cursor).toEqual(beforeReload.view.cursor)
    }
    const views = (await readStore(first, 'historySelections')).filter(row => results.includes(row.view.cursor.message_id))
    const testedIds = [...original.map(row => row.client_session_id), ...restored]
    const testedViews = views.filter(row => testedIds.includes(row.client_session_id))
    expect(testedViews).toHaveLength(4)
    expect(new Set(testedViews.map(row => row.view.view_session_id)).size).toBe(4)
  } finally { held.release() }
}

export async function rejectUnsupportedLocalFork(page: Page) {
  await choose(page, 'h1-a')
  await injectHistoryStorage(page)
  await page.evaluate(async () => {
    const api = (window as unknown as Window & { __h1Storage: typeof import('./history-storage-entry') }).__h1Storage
    await api.saveChatSettingsForKey('local:h1-source', { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', authorNote: 'Required source context' })
  })
  await page.reload()
  await expect(page.getByText('H1 variant A', { exact: true })).toBeVisible()
  const before = { rows: await readStore(page, 'messages'), histories: await readStore(page, 'chatHistories'), files: await readStore(page, 'sessionFiles') }
  await forkAt(page, 'h1-a')
  await expect(page.getByText('unsupported_fork_chat_settings', { exact: true })).toBeVisible()
  expect(await readStore(page, 'messages')).toEqual(before.rows)
  expect(await readStore(page, 'chatHistories')).toEqual(before.histories)
  expect(await readStore(page, 'sessionFiles')).toEqual(before.files)
  expect(await readStore(page, 'forkOperations')).toEqual([])
}
