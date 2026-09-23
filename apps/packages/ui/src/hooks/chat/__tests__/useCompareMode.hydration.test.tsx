import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, expect, it, vi } from 'vitest'
import { useCompareMode } from '../useCompareMode'
import { useModelComparison } from '@/components/Option/Playground/hooks/useModelComparison'
import { useStoreMessageOption } from '@/store/option'

const boundary = vi.hoisted(() => ({ loading: true, enabled: false, save: vi.fn(), setFlag: vi.fn() }))
vi.mock('@plasmohq/storage/hook', () => ({ useStorage: (key: string, fallback: unknown) => [key === 'ff_compareMode' ? boundary.enabled : fallback, boundary.setFlag, { isLoading: boundary.loading }] }))
vi.mock('@/db/dexie/helpers', () => ({ getCompareState: vi.fn(async () => ({ compareMode: true, compareSelectedModels: ['A', 'B'] })), saveCompareState: (...args: unknown[]) => boundary.save(...args) }))
vi.mock('@/utils/compare-metrics', () => ({ trackCompareMetric: vi.fn() }))

const useBoth = (forceEnabled = false) => {
  const mode = useCompareMode({ historyId: 'source', forceEnabled })
  useModelComparison({
    ...mode,
    composerModels: [], selectedModel: 'A', setSelectedModel: vi.fn(),
    selectedCharacterName: null, ragPinnedResultsLength: 0, webSearch: false,
    hasPromptContext: false, jsonMode: false, voiceChatEnabled: false, t: key => key
  })
  return mode
}

beforeEach(() => {
  vi.useFakeTimers()
  boundary.loading = true; boundary.enabled = false; boundary.save.mockReset()
  useStoreMessageOption.setState({ compareMode: false, compareSelectedModels: [] })
})
afterEach(() => { vi.useRealTimers() })

it('keeps restored comparison unchanged while the enabled preference is delayed beyond the save interval', async () => {
  const hook = renderHook(() => useBoth())
  await act(async () => {})
  await act(async () => { await vi.advanceTimersByTimeAsync(500) })
  expect(useStoreMessageOption.getState().compareMode).toBe(true)
  expect(boundary.save).not.toHaveBeenCalled()
  boundary.loading = false; boundary.enabled = true
  hook.rerender()
  await act(async () => { await vi.advanceTimersByTimeAsync(250) })
  expect(hook.result.current.compareMode).toBe(true)
  expect(boundary.save).toHaveBeenLastCalledWith(expect.objectContaining({ history_id: 'source', compareMode: true }))
})

it('honors an actually settled disabled preference', async () => {
  boundary.loading = false
  const hook = renderHook(() => useBoth())
  await act(async () => {})
  await act(async () => { await vi.advanceTimersByTimeAsync(500) })
  expect(hook.result.current.compareMode).toBe(false)
  expect(boundary.save).toHaveBeenLastCalledWith(expect.objectContaining({ compareMode: false }))
})

it('retains the existing force-enabled qualification while preference storage loads', async () => {
  const hook = renderHook(() => useBoth(true))
  await act(async () => {})
  await act(async () => { await vi.advanceTimersByTimeAsync(500) })
  expect(hook.result.current.compareMode).toBe(true)
  expect(hook.result.current.compareFeatureEnabled).toBe(true)
})
