import '/Users/macbook-dev/Documents/GitHub/tldw_server2/apps/tldw-frontend/node_modules/fake-indexeddb/auto/index.mjs'
import { afterEach, describe, expect, it, vi } from 'vitest'
vi.mock('@/models', () => ({ pageAssistModel: async () => ({ stream: async function* () { throw new Error('Local refusal'); yield '' } }) }))
vi.mock('@/services/tldw/TldwApiClient', () => ({}))
vi.mock('@/db', () => ({}))
vi.mock('@/utils/mcp-disclosure', () => ({ applyMcpModuleDisclosureFromToolCalls: vi.fn() }))
vi.mock('@/services/title', () => ({ generateTitle: async () => 'Variant probe' }))
vi.mock('@/utils/update-page-title', () => ({ updatePageTitle: vi.fn() }))
import { runChatPipeline } from '@/hooks/chat-modes/chatModePipeline'
import { saveMessageOnError } from '@/hooks/chat-helper'
import { db } from '@/db/dexie/schema'
import { saveHistory, formatToMessage } from '@/db/dexie/helpers'
import { PageAssistDatabase } from '@/db/dexie/chat'

const mode: any = {
  id: 'normal',
  buildUserMessage: (ctx: any) => ({ isBot: false, name: 'You', message: ctx.message, images: [ctx.image], id: ctx.resolvedUserMessageId }),
  buildAssistantMessage: (ctx: any) => ({ isBot: true, name: 'Assistant', message: '', images: [], id: ctx.resolvedAssistantMessageId, parentMessageId: ctx.resolvedAssistantParentMessageId }),
  preparePrompt: async () => ({ chatHistory: [], humanMessage: { content: 'Question' }, sources: [] })
}
afterEach(async () => { await db.delete() })
describe('UAT124 actual pipeline + save helper + Dexie + restore formatter', () => {
  it.each([false, true])('preserves three retries as variants when raw parent supplied=%s', async (explicitParent) => {
    await db.open()
    const history = await saveHistory('Variants', false, 'web-ui')
    let messages: any[] = []
    let transcript: any[] = []
    for (let attempt = 0; attempt < 3; attempt++) {
      const prior = messages.find(message => message.isBot)
      if (attempt > 0) messages = messages.filter(message => !message.isBot)
      await runChatPipeline(mode, 'Question', 'data:image/png;base64,private-synthetic', attempt > 0,
        messages, transcript, new AbortController().signal, {
          selectedModel: 'test-model', useOCR: false,
          setMessages: (next: any) => { messages = typeof next === 'function' ? next(messages) : next },
          setHistory: (next: any) => { transcript = typeof next === 'function' ? next(transcript) : next },
          setIsProcessing: vi.fn(), setStreaming: vi.fn(), setAbortController: vi.fn(),
          historyId: history.id, setHistoryId: vi.fn(), userMessageId: 'user-local', assistantMessageId: `assistant-${attempt}`,
          ...(explicitParent ? { assistantParentMessageId: 'user-local' } : {}),
          regenerateFromMessage: prior, saveMessageOnError, saveMessageOnSuccess: vi.fn(),
        } as any)
    }
    expect(messages.filter(message => message.isBot)).toHaveLength(1)
    expect(messages.find(message => message.isBot).variants).toHaveLength(3)
    const stored = await new PageAssistDatabase().getChatHistory(history.id)
    console.info(JSON.stringify({ explicitParent, rows: stored.map(({ id, role, parent_message_id }) => ({id, role, parent_message_id})) }))
    db.close()
    await db.open()
    const restored = formatToMessage(await new PageAssistDatabase().getChatHistory(history.id))
    expect(restored.filter(message => message.isBot)).toHaveLength(1)
    expect(restored.find(message => message.isBot)?.variants?.map(variant => variant.id)).toEqual(['assistant-0', 'assistant-1', 'assistant-2'])
    expect(restored.find(message => message.isBot)?.activeVariantIndex).toBe(2)
    expect(restored.find(message => !message.isBot)?.images).toEqual(['data:image/png;base64,private-synthetic'])
  })
})
