import { beforeEach, expect, it, vi } from 'vitest'
const mocks = vi.hoisted(() => ({ read: vi.fn(), owner: vi.fn() }))
vi.mock('@/db/dexie/helpers', () => ({ getFullChatData: mocks.read, formatToMessage: (rows: any[]) => rows, formatToChatHistory: (rows: any[]) => rows }))
vi.mock('@/db/dexie/history-selection', () => ({ getLocalHistoryOwner: mocks.owner }))
vi.mock('@/db/dexie/chat', () => ({ PageAssistDatabase: class {} }))
vi.mock('@/hooks/chat/useHistorySelection', () => ({ useHistorySelectionContext: () => null }))
vi.mock('@/services/model-settings', () => ({ lastUsedChatModelEnabled: vi.fn() }))
vi.mock('@/utils/update-page-title', () => ({ updatePageTitle: vi.fn() }))
import { restoreReadableLocalComparison } from '../useLoadLocalConversation'
const owner = { kind: 'local', owner_key: 'local-history-v1:profile', profile_id: 'profile', conversation_id: 'comparison' }
const data = { historyInfo: { id: 'comparison' }, messages: [{ id: 'a', content: 'Readable A' }] }
const fixture = () => {
  let valid = true
  const state = { owner, view: { conversation_id: 'comparison' }, error: 'unsupported_comparison_history', capture: { status: 'unsupported_history_capability', snapshot: { owner_key: owner.owner_key } } }
  const control = { getCurrent: () => state, fence: () => () => valid } as any
  return { control, state, invalidate: () => { valid = false } }
}
beforeEach(() => { mocks.read.mockReset().mockResolvedValue(data); mocks.owner.mockReset().mockResolvedValue(owner) })
it('restores readable rows while keeping the comparison send capture unsupported', async () => {
  const { control, state } = fixture(), display = vi.fn()
  expect(await restoreReadableLocalComparison(control, display)).toBe(true)
  expect(display).toHaveBeenCalledWith({ history: data.messages, messages: data.messages })
  expect(state.capture.status).toBe('unsupported_history_capability')
})
it.each(['data', 'profile'])('does not publish after navigation during held %s read', async phase => {
  const f = fixture(), display = vi.fn()
  let release!: (value: any) => void
  const held = new Promise(resolve => { release = resolve })
  ;(phase === 'data' ? mocks.read : mocks.owner).mockReturnValue(held)
  const pending = restoreReadableLocalComparison(f.control, display)
  await vi.waitFor(() => expect(phase === 'data' ? mocks.read : mocks.owner).toHaveBeenCalled())
  f.invalidate()
  release(phase === 'data' ? data : owner)
  expect(await pending).toBe(false)
  expect(display).not.toHaveBeenCalled()
})
it('rejects a foreign profile without substituting its readable rows', async () => {
  const f = fixture(), display = vi.fn()
  mocks.owner.mockResolvedValue({ ...owner, profile_id: 'foreign', owner_key: 'local-history-v1:foreign' })
  expect(await restoreReadableLocalComparison(f.control, display)).toBe(false)
  expect(display).not.toHaveBeenCalled()
})
it('reuses validated restore data and never reads for an unverified owner', async () => {
  const f = fixture(), display = vi.fn()
  expect(await restoreReadableLocalComparison(f.control, display, data as any)).toBe(true)
  expect(mocks.read).not.toHaveBeenCalled()
  f.state.capture.snapshot.owner_key = 'foreign'
  display.mockClear()
  expect(await restoreReadableLocalComparison(f.control, display)).toBe(false)
  expect(display).not.toHaveBeenCalled()
})
