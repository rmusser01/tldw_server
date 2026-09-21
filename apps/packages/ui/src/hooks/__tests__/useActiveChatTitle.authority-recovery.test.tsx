import { act, renderHook, waitFor } from '@testing-library/react'
import { useEffect, useRef, useState } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { useActiveChatTitle } from '../useActiveChatTitle'

const state = vi.hoisted(() => ({
  authorityLoading: false,
  config: {
    serverUrl: 'https://chat.example.test',
    authMode: 'single-user' as const,
    apiKey: 'alice-key'
  },
  option: {
    historyId: 'history-1',
    serverChatId: 'chat-1',
    serverChatTitle: 'Alice chat',
    serverChatMetaLoaded: true,
    temporaryChat: false,
  }
}))
const mocks = vi.hoisted(() => ({
  getTitleById: vi.fn(),
  getConfig: vi.fn(),
  updateChat: vi.fn(),
  loadServicePromptSnapshot: vi.fn(),
  updateHistory: vi.fn(),
  transaction: vi.fn(),
  setServerChatTitle: vi.fn(),
  setServerChatVersion: vi.fn()
}))

vi.mock('dexie-react-hooks', () => ({
  useLiveQuery: <T,>(query: () => Promise<T>, [owner, ready]: [unknown, unknown]) => {
    const [value, setValue] = useState<T | null>(null)
    const queryRef = useRef(query)
    queryRef.current = query
    useEffect(() => {
      let current = true
      void queryRef.current().then((next) => {
        if (current) setValue(next)
      })
      return () => { current = false }
    }, [owner, ready])
    return value
  }
}))
vi.mock('@/db/dexie/helpers', () => ({ getTitleById: mocks.getTitleById, updateHistory: mocks.updateHistory }))
vi.mock('@/db/dexie/chat-persistence-transaction', () => ({
  runChatPersistenceTransaction: mocks.transaction
}))
vi.mock('@/services/service-prompts', () => ({ loadServicePromptSnapshot: mocks.loadServicePromptSnapshot }))
vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: { getConfig: mocks.getConfig, updateChat: mocks.updateChat }
}))
vi.mock('@/hooks/useCanonicalConnectionConfig', () => ({
  useCanonicalConnectionConfig: () => ({ config: state.config, authorityLoading: state.authorityLoading })
}))
vi.mock('@/store/option', () => {
  const useStoreMessageOption = (selector?: (value: typeof state.option) => unknown) =>
    selector ? selector(state.option) : state.option
  useStoreMessageOption.getState = () => ({
    ...state.option,
    setServerChatTitle: mocks.setServerChatTitle,
    setServerChatVersion: mocks.setServerChatVersion
  })
  return { useStoreMessageOption }
})

describe('useActiveChatTitle authority recovery', () => {
  beforeEach(() => {
    vi.clearAllMocks()
    state.authorityLoading = false
    state.config = {
      serverUrl: 'https://chat.example.test',
      authMode: 'single-user',
      apiKey: 'alice-key'
    }
    state.option = {
      historyId: 'history-1',
      serverChatId: 'chat-1',
      serverChatTitle: 'Alice chat',
      serverChatMetaLoaded: true,
      temporaryChat: false
    }
    mocks.getTitleById.mockResolvedValue('')
    mocks.getConfig.mockImplementation(async () => state.config)
    mocks.updateChat.mockResolvedValue({ title: 'Renamed by Bob', version: 2 })
    mocks.loadServicePromptSnapshot.mockResolvedValue({
      requestScope: { userId: 'bob' },
      scopeSignal: new AbortController().signal,
      release: vi.fn()
    })
    mocks.transaction.mockImplementation(async (_signal: AbortSignal, operation: () => Promise<unknown>) => operation())
  })

  it('recovers only after the same selected chat is rehydrated for the new authority', async () => {
    const view = renderHook(() => useActiveChatTitle())
    expect(view.result.current).toMatchObject({ ready: true, title: 'Alice chat' })

    state.authorityLoading = true
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    view.rerender()
    expect(view.result.current).toMatchObject({ ready: false, title: '' })

    state.config = {
      serverUrl: 'https://chat.example.test',
      authMode: 'single-user',
      apiKey: 'bob-key'
    }
    state.authorityLoading = false
    view.rerender()

    expect(view.result.current).toMatchObject({ ready: false, title: '' })
    await act(async () => { await view.result.current.renameTitle('Blocked while loading') })
    expect(mocks.updateChat).not.toHaveBeenCalled()

    state.option = { ...state.option, serverChatMetaLoaded: false }
    view.rerender()
    expect(view.result.current).toMatchObject({ ready: false, title: '' })

    state.option = { ...state.option, serverChatTitle: 'Bob chat', serverChatMetaLoaded: true }
    view.rerender()
    expect(view.result.current).toMatchObject({ ready: true, title: 'Bob chat' })

    await act(async () => { await view.result.current.renameTitle('Renamed by Bob') })
    expect(mocks.updateChat).toHaveBeenCalledWith(
      'chat-1',
      { title: 'Renamed by Bob' },
      expect.objectContaining({ requestScope: { userId: 'bob' } })
    )
  })

  it('recovers a newly selected local chat after an authority boundary', async () => {
    const view = renderHook(() => useActiveChatTitle())
    expect(view.result.current.ready).toBe(true)

    state.authorityLoading = true
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    view.rerender()
    expect(view.result.current.ready).toBe(false)

    state.config = {
      serverUrl: 'https://chat.example.test',
      authMode: 'single-user',
      apiKey: 'bob-key'
    }
    state.authorityLoading = false
    state.option = {
      ...state.option,
      historyId: 'history-2',
      serverChatId: null,
      serverChatTitle: null,
      serverChatMetaLoaded: true
    }
    mocks.getTitleById.mockResolvedValue('Bob local title')
    view.rerender()

    expect(view.result.current.ready).toBe(true)
    await waitFor(() => expect(mocks.getTitleById).toHaveBeenCalledWith('history-2'))
    await waitFor(() => expect(view.result.current.title).toBe('Bob local title'))
    await act(async () => { await view.result.current.renameTitle('Bob local chat') })
    expect(mocks.updateHistory).toHaveBeenCalledWith('history-2', 'Bob local chat')
  })

  it('requires a new metadata cycle for each successive authority', async () => {
    const view = renderHook(() => useActiveChatTitle())
    expect(view.result.current).toMatchObject({ ready: true, title: 'Alice chat' })

    state.authorityLoading = true
    act(() => window.dispatchEvent(new CustomEvent('tldw:auth-principal-changed')))
    state.config = { ...state.config, apiKey: 'bob-key' }
    state.authorityLoading = false
    view.rerender()
    state.option = { ...state.option, serverChatMetaLoaded: false }
    view.rerender()
    state.option = { ...state.option, serverChatTitle: 'Bob chat', serverChatMetaLoaded: true }
    view.rerender()
    expect(view.result.current).toMatchObject({ ready: true, title: 'Bob chat' })

    state.config = { ...state.config, apiKey: 'carol-key' }
    view.rerender()

    expect(view.result.current).toMatchObject({ ready: false, title: '' })
    await act(async () => { await view.result.current.renameTitle('Blocked for Carol') })
    expect(mocks.updateChat).not.toHaveBeenCalled()

    state.option = { ...state.option, serverChatMetaLoaded: false }
    view.rerender()
    state.option = { ...state.option, serverChatTitle: 'Carol chat', serverChatMetaLoaded: true }
    view.rerender()
    expect(view.result.current).toMatchObject({ ready: true, title: 'Carol chat' })
  })
})
