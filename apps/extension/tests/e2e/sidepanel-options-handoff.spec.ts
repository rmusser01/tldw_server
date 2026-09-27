import { test, expect } from '@playwright/test'
import path from 'node:path'
import { launchWithExtension } from './utils/extension'
import { grantHostPermission } from './utils/permissions'
import { startHistoryServer, historySeedConfig, seedHistory, seedSidepanelReference, choose, readStore } from './utils/history-selection'

const EXT_PATH = path.resolve('build/chrome-mv3')

test('selected sidepanel expansion opens extension full page with an independent view over the same owner', async () => {
  const server = await startHistoryServer()
  const { context, page, optionsUrl, extensionId, openSidepanel } = await launchWithExtension(EXT_PATH, { seedConfig: historySeedConfig(server.url) })
  try {
    expect(await grantHostPermission(context, extensionId, `${server.url}/*`)).toBe(true)
    await page.goto(`${optionsUrl}#/chat`)
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await seedSidepanelReference(page)
    const sidepanel = await openSidepanel('/chat')
    await sidepanel.setViewportSize({ width: 390, height: 780 })
    await choose(sidepanel, 'h1-a')
    const [expanded] = await Promise.all([context.waitForEvent('page'), sidepanel.getByTestId('chat-open-full-screen').click()])
    await expect.poll(() => expanded.url()).toContain(`${extensionId}/options.html`)
    await expect(expanded.getByText('H1 variant A', { exact: true })).toBeVisible()
    await choose(expanded, 'h1-b')
    await expect(sidepanel.getByText('H1 variant A', { exact: true })).toBeVisible()
    await expect(sidepanel.getByText('H1 variant B', { exact: true })).toHaveCount(0)
    expect(new Set((await readStore(sidepanel, 'historySelections')).filter(row => row.client_session_id !== 'fixture-origin').map(row => row.view.view_session_id)).size).toBe(2)
  } finally { await context.close(); await server.close() }
})

test('explicit Continue in WebUI composer action preserves its existing full-app draft handoff', async () => {
  const server = await startHistoryServer()
  const { context, page, optionsUrl, extensionId, openSidepanel } = await launchWithExtension(EXT_PATH, { seedConfig: historySeedConfig(server.url) })
  try {
    expect(await grantHostPermission(context, extensionId, `${server.url}/*`)).toBe(true)
    await page.goto(optionsUrl + '#/chat')
    await expect(page.getByTestId('chat-input')).toBeVisible()
    await seedHistory(page)
    await seedSidepanelReference(page)
    const sidepanel = await openSidepanel('/chat')
    await choose(sidepanel, 'h1-a')
    await sidepanel.evaluate(() => localStorage.setItem('tldw-ui-mode', JSON.stringify({ state: { mode: 'pro' }, version: 1 })))
    await sidepanel.reload()
    await sidepanel.getByTestId('chat-input').fill('H1 explicit composer draft')
    await sidepanel.getByTestId('control-more-menu').click()
    const [expanded] = await Promise.all([context.waitForEvent('page'), sidepanel.getByTestId('chat-continue-in-webui').click()])
    await expect.poll(() => expanded.url()).toContain(`${extensionId}/options.html`)
    await expect(expanded.getByTestId('chat-input')).toHaveValue('H1 explicit composer draft')
    expect(server.requests.filter(row => row.method === 'POST' && row.path === '/api/v1/chats/')).toHaveLength(0)
  } finally { await context.close(); await server.close() }
})

test('settings written in options remain shared with the sidepanel through real extension storage', async () => {
  const server = await startHistoryServer()
  const { context, page, openSidepanel } = await launchWithExtension(EXT_PATH, { seedConfig: historySeedConfig(server.url) })
  try {
    await page.evaluate(() => chrome.storage.local.set({ userChatBubble: false }))
    const sidepanel = await openSidepanel('/chat')
    expect(await sidepanel.evaluate(async () => (await chrome.storage.local.get('userChatBubble')).userChatBubble)).toBe(false)
  } finally { await context.close(); await server.close() }
})
