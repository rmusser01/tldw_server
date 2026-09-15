import React from 'react'
import { act, renderHook } from '@testing-library/react'
import { QueryClient, QueryClientProvider } from '@tanstack/react-query'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, expect, it, vi } from 'vitest'
import { usePromptEditor } from '../hooks/usePromptEditor'

const { save, update } = vi.hoisted(() => ({ save: vi.fn(), update: vi.fn() }))
vi.mock('@/db/dexie/helpers', () => ({
  savePrompt: save, updatePrompt: update, deletePromptById: vi.fn(), restorePrompt: vi.fn(),
  permanentlyDeletePrompt: vi.fn(), emptyTrash: vi.fn(), incrementPromptUsage: vi.fn()
}))
vi.mock('antd', () => ({ notification: { success: vi.fn(), error: vi.fn() } }))

const renderEditor = () => {
  const queryClient = new QueryClient({ defaultOptions: { mutations: { retry: false } } })
  return renderHook(() => usePromptEditor({
    queryClient, isOnline: true, t: (key) => key, guardPrivateMode: () => false,
    getPromptTexts: (record) => ({ systemText: record.system_prompt, userText: record.user_prompt }),
    getPromptKeywords: (record) => record.keywords || [], getPromptRecordById: () => null,
    confirmDanger: async () => true,
    syncPromptAfterLocalSave: async () => ({ attempted: true, success: true }),
    recipePersistenceAvailable: false
  }), { wrapper: ({ children }) => <MemoryRouter initialEntries={['/prompts?new=1']}><QueryClientProvider client={queryClient}>{children}</QueryClientProvider></MemoryRouter> })
}

beforeEach(() => {
  save.mockReset().mockImplementation(async (payload) => ({ ...payload, id: 'saved-prompt' }))
  update.mockReset().mockResolvedValue('saved-prompt')
})

it('adopts the saved identity and baseline so a second save updates the same prompt', async () => {
  const { result } = renderEditor()
  act(() => result.current.openFullEditor())
  await act(async () => { await result.current.handleFullEditorSubmit({ name: 'Pirate helper', system_prompt: 'Talk like a pirate.' }) })
  expect(result.current.fullEditorMode).toBe('edit')
  expect(result.current.fullEditorInitialValues).toMatchObject({ id: 'saved-prompt', name: 'Pirate helper', system_prompt: 'Talk like a pirate.' })
  await act(async () => { await result.current.handleFullEditorSubmit({ name: 'Pirate helper', system_prompt: 'Use short pirate answers.' }) })
  expect(update).toHaveBeenCalledWith(expect.objectContaining({ id: 'saved-prompt', system_prompt: 'Use short pirate answers.' }))
  expect(save).toHaveBeenCalledTimes(1)
})

it('does not replace a different editor with a late save response', async () => {
  let finish!: (value: unknown) => void
  save.mockImplementation(() => new Promise(resolve => { finish = resolve }))
  const { result } = renderEditor()
  act(() => result.current.openFullEditor())
  let saving: unknown
  await act(async () => { saving = result.current.handleFullEditorSubmit({ name: 'First draft' }) })
  act(() => {
    result.current.closeFullEditor()
    result.current.openFullEditor({ id: 'other-prompt', name: 'Other prompt', system_prompt: 'Be concise.' })
  })
  await act(async () => { finish({ id: 'saved-prompt' }); await saving })
  expect(result.current.fullEditorInitialValues).toMatchObject({ id: 'other-prompt', name: 'Other prompt' })
})

it('keeps a failed save as an unsaved create instead of adopting a false saved baseline', async () => {
  save.mockRejectedValue(new Error('Local save failed'))
  const { result } = renderEditor()
  act(() => result.current.openFullEditor())
  await act(async () => {
    await expect(result.current.handleFullEditorSubmit({ name: 'Keep this draft' })).rejects.toThrow('Local save failed')
  })
  expect(result.current.fullEditorMode).toBe('create')
  expect(result.current.fullEditorOpen).toBe(true)
})

it('ignores a failed save after another editor has opened', async () => {
  let fail!: (error: Error) => void
  save.mockImplementation(() => new Promise((_, reject) => { fail = reject }))
  const { result } = renderEditor()
  act(() => result.current.openFullEditor())
  let saving!: Promise<boolean>
  await act(async () => { saving = result.current.handleFullEditorSubmit({ name: 'First draft' }) })
  act(() => {
    result.current.closeFullEditor()
    result.current.openFullEditor({ id: 'other-prompt', name: 'Other prompt' })
  })
  await act(async () => {
    const outcome = expect(saving).resolves.toBe(false)
    fail(new Error('Earlier save failed'))
    await outcome
  })
  expect(result.current.fullEditorInitialValues).toMatchObject({ id: 'other-prompt', name: 'Other prompt' })
})
