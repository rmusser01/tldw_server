import { createRequire } from 'node:module'
const { build } = createRequire(require.resolve('wxt'))('esbuild')
import path from 'node:path'
import { writeFile } from 'node:fs/promises'
import type { Page } from '@playwright/test'

export async function injectHistoryStorage(page: Page) {
  const extension = page.url().startsWith('chrome-extension:')
  const apps = path.resolve(__dirname, '../../../..')
  const ui = path.join(apps, 'packages/ui/src')
  const frontend = path.join(apps, 'tldw-frontend')
  const alias: Record<string, string> = { '@': ui, '~': ui }
  if (!extension) {
    alias['@plasmohq/storage/hook'] = path.join(frontend, 'extension/shims/plasmo-storage-hook.tsx')
    alias['@plasmohq/storage'] = path.join(frontend, 'extension/shims/plasmo-storage.ts')
    alias['wxt/browser'] = path.join(frontend, 'extension/shims/wxt-browser.ts')
  }
  const result = await build({
    entryPoints: [path.join(__dirname, 'history-storage-entry.ts')],
    bundle: true, write: false, format: 'iife', globalName: '__h1Storage', platform: 'browser',
    target: 'chrome120', alias, metafile: true,
    define: { 'process.env.NODE_ENV': '"production"', 'import.meta.env': JSON.stringify({ BROWSER: 'chrome', MANIFEST_VERSION: 3, MODE: 'production', PROD: true }) },
  })
  const label = extension ? 'extension' : 'webui'
  await writeFile(`/tmp/chatbook_h1_storage_${label}_metafile.json`, JSON.stringify({ alias, inputs: result.metafile!.inputs }, null, 2))
  // Evaluation through the automation protocol does not alter the product CSP.
  await page.evaluate(result.outputFiles[0].text)
  return page.evaluate(async () => {
    const api = (window as any).__h1Storage
    await api.db.open()
    return { name: api.db.name, version: api.db.verno, rows: await api.db.messages.count(), settingsBackend: api.chatSettingsStorageForKey('local:h1-source').constructor.name }
  })
}

export async function qualifyHistoryStorage(page: Page) {
  const { expect } = await import('@playwright/test')
  expect(await injectHistoryStorage(page)).toMatchObject({ name: 'PageAssistDatabase', version: 16, rows: 3 })
  const result = await page.evaluate(async () => {
    const a = (window as any).__h1Storage
    const key = 'local:h1-source'
    const payload = { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', authorNote: 'Migrated legacy note' }
    if (location.protocol === 'chrome-extension:') await chrome.storage.sync.set({ ['chatSettings:' + key]: JSON.stringify(payload) })
    else localStorage.setItem('chatSettings:' + key, JSON.stringify(payload))
    const migrated = await a.getChatSettingsForKey(key)
    const storage = a.chatSettingsStorageForKey(key)
    const persisted = await storage.get('chatSettings:' + key)
    const initialGuard = (await a.db.chatHistories.get('h1-source')).local_settings_guard
    const writes = await Promise.all([
      a.saveChatSettingsForKey(key, { ...payload, authorNote: 'First serialized writer' }),
      a.saveChatSettingsForKey(key, { ...payload, authorNote: 'Second serialized writer' })
    ])
    const final = await a.getChatSettingsForKey(key)
    const settledGuard = (await a.db.chatHistories.get('h1-source')).local_settings_guard
    const snapshot = await a.getFullChatData('h1-source')
    await a.deleteByHistoryId('h1-source')
    const deleted = !(await a.db.chatHistories.get('h1-source'))
    await a.restoreChat(snapshot)
    const restoredGuard = (await a.db.chatHistories.get('h1-source')).local_settings_guard
    const originalSet = storage.set
    storage.set = async () => { throw new DOMException('H1 injected persistent write unavailable', 'QuotaExceededError') }
    let failed
    try { failed = await a.saveChatSettingsForKey(key, { ...payload, authorNote: 'Must not persist' }) } finally { storage.set = originalSet }
    const pendingGuard = (await a.db.chatHistories.get('h1-source')).local_settings_guard
    const error = async (operation: () => Promise<unknown>) => { try { await operation(); return null } catch (e) { return String((e as Error).message) } }
    const deleteError = await error(() => a.deleteByHistoryId('h1-source'))
    const forkError = await error(() => a.withPlainLocalForkSettings('h1-source', () => { throw new Error('ERROR: crossed unavailable guard') }))
    await a.importChatHistoryV2([{ history: { ...snapshot.historyInfo, title: 'Imported readable rows', local_settings_guard: { initialized: true, revision: 'forged', pending: [] } }, messages: snapshot.messages }], { replaceExisting: true })
    const importedGuard = (await a.db.chatHistories.get('h1-source')).local_settings_guard
    return { migrated, persisted, initialGuard, writes, final, settledGuard, deleted, restoredGuard, failed, pendingGuard, deleteError, forkError, importedGuard, rows: await a.db.messages.count() }
  })
  expect(result.migrated.authorNote).toBe('Migrated legacy note')
  expect(result.persisted.authorNote).toBe('Migrated legacy note')
  expect(result.initialGuard).toMatchObject({ initialized: true, pending: [] })
  expect(result.writes).toEqual([true, true])
  expect(result.final.authorNote).toBe('Second serialized writer')
  expect(result.settledGuard.pending).toEqual([])
  expect(result.deleted).toBe(true)
  expect(result.restoredGuard).toEqual(result.settledGuard)
  expect(result.failed).toBe(false)
  expect(result.pendingGuard.pending).toHaveLength(1)
  expect(result.deleteError).toBe('history_settings_write_pending')
  expect(result.forkError).toBe('fork_chat_settings_unavailable')
  expect(result.importedGuard).toEqual(result.pendingGuard)
  expect(result.rows).toBe(3)
  await page.reload()
  await injectHistoryStorage(page)
  expect(await page.evaluate(async () => (await (window as any).__h1Storage.db.chatHistories.get('h1-source')).local_settings_guard)).toEqual(result.pendingGuard)
}

export async function qualifyStorageInterleavings(page: Page) {
  const { expect } = await import('@playwright/test')
  await injectHistoryStorage(page)
  const second = await page.context().newPage()
  try {
    await second.goto(page.url())
    await expect(second.getByTestId('chat-input')).toBeVisible()
    await injectHistoryStorage(second)
    await page.evaluate(async () => {
      const w = window as any, a = w.__h1Storage, key = 'chatSettings:local:h1-source'
      const legacy = { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', authorNote: 'Legacy during migration' }
      if (location.protocol === 'chrome-extension:') await chrome.storage.sync.set({ [key]: JSON.stringify(legacy) })
      else localStorage.setItem(key, JSON.stringify(legacy))
      const storage = a.chatSettingsStorageForKey('local:h1-source'), original = storage.get.bind(storage)
      let entered!: () => void
      const started = new Promise<void>(resolve => { entered = resolve })
      const gate = new Promise<void>(resolve => { w.__h1Release = resolve })
      storage.get = async (...args: any[]) => { storage.get = original; entered(); await gate; return original(...args) }
      w.__h1Migration = a.getChatSettingsForKey('local:h1-source')
      await started
    })
    await second.evaluate(() => { (window as any).__h1Write = (window as any).__h1Storage.saveChatSettingsForKey('local:h1-source', { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', authorNote: 'Later write wins' }) })
    await page.evaluate(async () => { (window as any).__h1Release(); await (window as any).__h1Migration })
    expect(await second.evaluate(async () => await (window as any).__h1Write)).toBe(true)
    expect(await page.evaluate(async () => (await (window as any).__h1Storage.getChatSettingsForKey('local:h1-source')).authorNote)).toBe('Later write wins')
    const read = await page.evaluate(async () => {
      const a = (window as any).__h1Storage, key = 'local:h1-source', storage = a.chatSettingsStorageForKey(key), original = storage.get.bind(storage)
      storage.get = async () => { throw new DOMException('H1 persistent read denied', 'SecurityError') }
      let error: string | null = null
      try { await a.withPlainLocalForkSettings('h1-source', async () => { throw new Error('must not reach mutation') }) } catch (e) { error = (e as Error).message } finally { storage.get = original }
      const saved = await a.saveChatSettingsForKey(key, { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', authorNote: '' })
      const empty = await a.withPlainLocalForkSettings('h1-source', async (validate: any) => { validate(await a.db.chatHistories.get('h1-source')); return true })
      return { error, saved, empty }
    })
    expect(read).toEqual({ error: 'fork_chat_settings_unavailable', saved: true, empty: true })
    await page.evaluate(async () => {
      const w = window as any, a = w.__h1Storage, storage = a.chatSettingsStorageForKey('local:h1-source'), original = storage.set.bind(storage)
      let entered!: () => void
      const started = new Promise<void>(resolve => { entered = resolve })
      const gate = new Promise<void>(resolve => { w.__h1Release = resolve })
      storage.set = async () => {
        w.__h1FailedToken = (await a.db.chatHistories.get('h1-source')).local_settings_guard.pending[0]
        entered()
        await gate
        throw new DOMException('H1 uncertain first writer', 'QuotaExceededError')
      }
      w.__h1Failed = a.saveChatSettingsForKey('local:h1-source', { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', authorNote: 'Failed writer' }).finally(() => { storage.set = original })
      await started
    })
    await second.evaluate(() => { (window as any).__h1Write = (window as any).__h1Storage.saveChatSettingsForKey('local:h1-source', { schemaVersion: 2, updatedAt: '2026-09-17T00:00:00Z', authorNote: 'Surviving writer' }) })
    expect(await page.evaluate(async () => { (window as any).__h1Release(); return await (window as any).__h1Failed })).toBe(false)
    expect(await second.evaluate(async () => await (window as any).__h1Write)).toBe(true)
    const final = await page.evaluate(async () => {
      const w = window as any, a = w.__h1Storage
      let error: string | null = null
      try { await a.withPlainLocalForkSettings('h1-source', async () => true) } catch (e) { error = (e as Error).message }
      return { token: w.__h1FailedToken, guard: (await a.db.chatHistories.get('h1-source')).local_settings_guard, settings: await a.getChatSettingsForKey('local:h1-source'), error }
    })
    expect(final.guard.pending).toEqual([final.token])
    expect(final.settings.authorNote).toBe('Surviving writer')
    expect(final.error).toBe('fork_chat_settings_unavailable')
  } finally { await second.close() }
}
