import { qualifyHistoryStorage, qualifyStorageInterleavings } from './utils/history-storage'
import { test, expect } from '@playwright/test'
import path from 'node:path'
import { launchWithExtension } from './utils/extension'
import { grantHostPermission } from './utils/permissions'
import { startHistoryServer, historySeedConfig, nativeCharacterSeedConfig, seedHistory, historyUrl, choose, readStore, localForkAndSend, legacyBoundaries, seedSidepanelReference, seedNativeReference, nativeHistoryUrl, nativeForkAndSend, unknownNativeFork, abortLocalFork, largeLegacyReview, editNativeChildSettings, rejectNativeMetadataSettings, heldNativeFirstCreate, firstNativeCharacterSend, uncertainNativeCharacterSend, heldNativeForkNavigation, qualifyChildIsolation, qualifyForeignWorkspaceAndAccountRead } from './utils/history-selection'

test('full-page selected variants use independent views over one real owner DB', async () => {
  test.setTimeout(120_000)
  const server = await startHistoryServer()
  const { context, page, extensionId, optionsUrl } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
  try {
    expect(await grantHostPermission(context, extensionId, `${server.url}/*`)).toBe(true)
    await page.goto(`${optionsUrl}#/chat`)
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await seedHistory(page)
    await page.goto(historyUrl(`${optionsUrl}#/chat`))
    await page.reload()
    await choose(page, 'h1-a')
    const second = await context.newPage()
    await second.goto(historyUrl(`${optionsUrl}#/chat`))
    await choose(second, 'h1-b')
    const bookmarks = await readStore(page, 'historySelections')
    expect(bookmarks.filter(row => row.conversation_id === 'h1-source').map(row => row.view.cursor)).toEqual(expect.arrayContaining([{ kind: 'after_message', message_id: 'h1-a' }, { kind: 'after_message', message_id: 'h1-b' }]))
    await expect(page.getByText('H1 variant A', { exact: true })).toBeVisible()
    await expect(page.getByText('H1 variant B', { exact: true })).toHaveCount(0)
    await page.reload()
    await expect(page.getByText('H1 variant A', { exact: true })).toBeVisible()
  } finally { await context.close(); await server.close() }
})

for (const surface of ['full page', 'compact sidepanel'] as const) {
  test(`${surface}: local fork adopts child for immediate send and reopen`, async () => {
    test.setTimeout(90_000)
    const server = await startHistoryServer()
    const launched = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
    const { context, page, extensionId, optionsUrl, openSidepanel } = launched
    try {
      expect(await grantHostPermission(context, extensionId, `${server.url}/*`)).toBe(true)
      await page.goto(`${optionsUrl}#/chat`)
      await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
      await seedHistory(page)
      let target = page
      if (surface === 'compact sidepanel') {
        await seedSidepanelReference(page)
        target = await openSidepanel('/chat')
        await target.setViewportSize({ width: 390, height: 780 })
      } else {
        await page.goto(historyUrl(`${optionsUrl}#/chat`))
        await page.reload()
      }
      await localForkAndSend(target, server)
    } finally { await context.close(); await server.close() }
  })
}

test('full page: legacy before-first and empty survive reload without losing alternatives', async () => {
  const server = await startHistoryServer()
  const { context, page, extensionId, optionsUrl } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
  try {
    expect(await grantHostPermission(context, extensionId, `${server.url}/*`)).toBe(true)
    await page.goto(`${optionsUrl}#/chat`)
    await expect(page.getByTestId('chat-input')).toBeVisible({ timeout: 30000 })
    await seedHistory(page, { legacy: true })
    await page.goto(historyUrl(`${optionsUrl}#/chat`))
    await page.reload()
    await legacyBoundaries(page)
  } finally { await context.close(); await server.close() }
})


for (const surface of ['full page', 'compact sidepanel'] as const) {
  test(surface + ': native fork adopts scoped child for immediate send and reopen', async () => {
    test.setTimeout(90_000)
    const server = await startHistoryServer({ native: true })
    const { context, page, extensionId, optionsUrl, openSidepanel } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
    try {
      expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true)
      await page.goto(optionsUrl + '#/chat')
      await expect(page.getByTestId('chat-input')).toBeVisible()
      await seedNativeReference(page)
      let target = page
      if (surface === 'compact sidepanel') {
        await seedSidepanelReference(page, 'native-source')
        target = await openSidepanel('/chat')
        await target.setViewportSize({ width: 390, height: 780 })
      } else {
        await page.goto(nativeHistoryUrl(optionsUrl + '#/chat'))
        await page.reload()
      }
      const child = await nativeForkAndSend(target, server)
      if (surface === 'full page') { await editNativeChildSettings(target, server, child); await rejectNativeMetadataSettings(target, server, child) }
    } finally { await context.close(); await server.close() }
  })
}

for (const scenario of ['transaction abort', '20001-row full-tip transcript', 'unknown native copy'] as const) {
  test('full page: ' + scenario, async () => {
    test.setTimeout(120_000)
    const server = await startHistoryServer({ native: true, loseForkResponse: scenario === 'unknown native copy' })
    const { context, page, extensionId, optionsUrl } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
    try {
      expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true)
      await page.goto(optionsUrl + '#/chat')
      await expect(page.getByTestId('chat-input')).toBeVisible()
      if (scenario === 'unknown native copy') {
        await seedNativeReference(page)
        await page.goto(nativeHistoryUrl(optionsUrl + '#/chat'))
        await page.reload()
        await unknownNativeFork(page, server, async () => { const second = await context.newPage(); await second.goto(nativeHistoryUrl(optionsUrl + '#/chat')); return second })
      } else {
        await seedHistory(page, scenario === 'transaction abort' ? {} : { legacy: true, count: 20001 })
        await page.goto(historyUrl(optionsUrl + '#/chat'))
        await page.reload()
        if (scenario === 'transaction abort') await abortLocalFork(page)
        else await largeLegacyReview(page, true)
      }
    } finally { await context.close(); await server.close() }
  })
}

test('production storage APIs preserve migration, concurrent writes and unavailable guards across delete import undo and reload', async () => {
  const server = await startHistoryServer()
  const { context, page, optionsUrl } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
  try {
    await page.goto(optionsUrl + '#/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await qualifyHistoryStorage(page)
  } finally { await context.close(); await server.close() }
})

test('production storage interleaves migration and writers while distinguishing failed reads from empty data', async () => {
  const server = await startHistoryServer()
  const { context, page, optionsUrl } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
  try {
    await page.goto(optionsUrl + '#/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await qualifyStorageInterleavings(page)
  } finally { await context.close(); await server.close() }
})

for (const live of [false, true]) test((live ? 'live navigation: ' : 'unload: ') + 'held first native character create cannot adopt or dispatch after local navigation', async () => {
  test.setTimeout(60000)
  const server = await startHistoryServer()
  const { context, page, optionsUrl, extensionId } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: nativeCharacterSeedConfig(server.url) })
  try {
    expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true)
    await page.goto(optionsUrl + '#/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await heldNativeFirstCreate(page, server, optionsUrl + '#/chat', live)
  } finally { await context.close(); await server.close() }
})

for (const comparison of [false, true]) test('child edits deletion and copied-file removal isolate source: ' + (comparison ? 'comparison' : 'normal'), async () => {
  test.setTimeout(90000)
  const server = await startHistoryServer()
  const { context, page, optionsUrl, extensionId } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
  const base = optionsUrl + '#/chat'
  try {
    expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true)
    await page.goto(base)
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await qualifyChildIsolation(page, server, base, comparison)
  } finally { await context.close(); await server.close() }
})

test('global H1 filters foreign workspace operations and invalidates held reads after account config changes', async () => {
  test.setTimeout(60000)
  const server = await startHistoryServer()
  const { context, page, optionsUrl, extensionId } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
  const base = optionsUrl + '#/chat'
  try {
    expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true)
    await page.goto(base)
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedNativeReference(page)
    await page.goto(nativeHistoryUrl(base))
    await page.reload()
    await qualifyForeignWorkspaceAndAccountRead(page, server, base)
  } finally { await context.close(); await server.close() }
})

for (const surface of ['full page', 'sidepanel'] as const) test(surface + ': first native character send follows acknowledged result and reopens', async () => {
 test.setTimeout(90000)
 const server = await startHistoryServer(); const { context, page, optionsUrl, extensionId, openSidepanel } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: nativeCharacterSeedConfig(server.url) })
 try { expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true); await page.goto(optionsUrl + '#/chat'); await expect(page.getByTestId('chat-input')).toBeVisible(); const target = surface === 'sidepanel' ? await openSidepanel('/chat') : page; if (surface === 'sidepanel') await target.setViewportSize({ width: 390, height: 720 }); await firstNativeCharacterSend(target, server) }
 finally { await context.close(); await server.close() }
})
for (const boundary of ['create', 'settings'] as const) test('live navigation: held native ' + boundary + ' cannot redirect local view', async () => {
 test.setTimeout(60000)
 const server = await startHistoryServer(); const { context, page, optionsUrl, extensionId } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
 const base = optionsUrl + '#/chat'
 try { expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true); await page.goto(base); await expect(page.getByTestId('chat-input')).toBeVisible(); await seedNativeReference(page); await page.goto(nativeHistoryUrl(base)); await page.reload(); await heldNativeForkNavigation(page, server, base, boundary, true) }
 finally { await context.close(); await server.close() }
})

for (const ack of ['missing', 'wrong'] as const) test('native character ' + ack + ' ACK remains uncertain without replay', async () => {
 const server = await startHistoryServer({ characterAck: ack }); const { context, page, optionsUrl, extensionId } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: nativeCharacterSeedConfig(server.url) })
 try { expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true); await page.goto(optionsUrl + '#/chat'); await expect(page.getByTestId('chat-input')).toBeVisible(); await uncertainNativeCharacterSend(page, server) }
 finally { await context.close(); await server.close() }
})

test('sidepanel: held first native create cannot adopt after live local navigation', async () => {
  test.setTimeout(60000)
  const server = await startHistoryServer()
  const { context, page, optionsUrl, extensionId, openSidepanel } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: nativeCharacterSeedConfig(server.url) })
  try {
    expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true)
    await page.goto(optionsUrl + '#/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    const sidepanel = await openSidepanel('/chat')
    await sidepanel.setViewportSize({ width: 390, height: 720 })
    await heldNativeFirstCreate(sidepanel, server, optionsUrl + '#/chat', true)
  } finally { await context.close(); await server.close() }
})

test('comparison sidepanel expansion preserves readable unsupported owner and forks its model chain', async () => {
  test.setTimeout(90000)
  const server = await startHistoryServer()
  const { context, page, optionsUrl, extensionId, openSidepanel } = await launchWithExtension(path.resolve('build/chrome-mv3'), { seedConfig: historySeedConfig(server.url) })
  try {
    expect(await grantHostPermission(context, extensionId, server.url + '/*')).toBe(true)
    await page.goto(optionsUrl + '#/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await qualifyChildIsolation(page, server, optionsUrl + '#/chat', true, async () => {
      await seedSidepanelReference(page)
      const sidepanel = await openSidepanel('/chat')
      await sidepanel.setViewportSize({ width: 390, height: 780 })
      await expect(sidepanel.getByText('Selected history unavailable', { exact: true })).toBeVisible()
      await expect(sidepanel.getByText('H1 second A answer', { exact: true })).toBeVisible()
      const [expanded] = await Promise.all([context.waitForEvent('page'), sidepanel.getByTestId('chat-open-full-screen').click()])
      await expect.poll(() => expanded.url()).toContain(extensionId + '/options.html')
      return expanded
    })
  } finally { await context.close(); await server.close() }
})
