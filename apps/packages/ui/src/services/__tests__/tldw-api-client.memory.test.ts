import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { TldwApiClient } from '@/services/tldw/TldwApiClient'

const request = vi.hoisted(() => vi.fn())
vi.mock('@/services/background-proxy', () => ({
  bgRequest: request,
  bgUpload: vi.fn(),
  bgStream: vi.fn()
}))
vi.mock('@/utils/safe-storage', () => ({
  createSafeStorage: () => ({
    get: async (key: string) => key === 'tldwConfig' ? {
      serverUrl: 'https://memory.test',
      authMode: 'single-user',
      apiKey: 'synthetic-memory-key',
      authSource: 'manual',
      credentialSource: 'manual',
      apiKeyPersistence: 'device',
      apiKeyServerOrigin: 'https://memory.test'
    } : null,
    set: async () => {},
    remove: async () => {}
  }),
  safeStorageSerde: { serialize: (value: unknown) => value, deserialize: (value: unknown) => value }
}))
let client: TldwApiClient
beforeEach(() => {
  vi.useFakeTimers()
  client = new TldwApiClient()
  request.mockReset()
  request.mockResolvedValue([{ id: '1', role: 'user', content: 'cached text' }])
})
afterEach(() => {
  client.chatMessagesCache.clear()
  client.characterCache.clear()
  vi.useRealTimers()
})

it('releases cached chat payloads when their TTL expires while idle', async () => {
  await client.listChatMessages('first')
  await vi.advanceTimersByTimeAsync(60_001)
  expect(client.chatMessagesCache.size).toBe(0)
})

it('sweeps old chat payloads when another chat is accessed after a clock jump', async () => {
  await client.listChatMessages('first')
  vi.setSystemTime(Date.now() + 600_000)
  await client.listChatMessages('second')
  expect(client.chatMessagesCache.size).toBe(1)
})

it('bounds both shared cache families across many distinct IDs', async () => {
  for (let id = 0; id < 50; id++) {
    await client.listChatMessages(id)
    client.characterCache.set(String(id), {
      value: { name: 'character' },
      expiresAt: Date.now() + 300_000
    })
  }
  expect(client.chatMessagesCache.size).toBeLessThanOrEqual(32)
  expect(client.characterCache.size).toBeLessThanOrEqual(32)
})

it('returns a large chat response without retaining it in the shared cache', async () => {
  const content = 'x'.repeat(5 * 1024 * 1024)
  request.mockResolvedValue([{ id: '1', role: 'user', content }])
  const result = await client.listChatMessages('large')
  expect(result[0].content).toBe(content)
  expect(client.chatMessagesCache.size).toBe(0)
})

it('keeps shared in-flight deduplication and cached results', async () => {
  const [first, second] = await Promise.all([
    client.listChatMessages('same'),
    client.listChatMessages('same')
  ])
  expect(second).toBe(first)
  expect(await client.listChatMessages('same')).toBe(first)
  expect(request).toHaveBeenCalledOnce()
})
