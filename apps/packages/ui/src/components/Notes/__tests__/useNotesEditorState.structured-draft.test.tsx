import React from 'react';
import { act, renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { beforeEach, describe, expect, it, vi } from 'vitest';
import { normalizeOfflineDraftQueue } from '../notes-manager-utils';
import { useNotesEditorState, type UseNotesEditorStateDeps } from '../hooks/useNotesEditorState';

const mocks = vi.hoisted(() => ({
  request: vi.fn(),
  tasks: vi.fn(),
  activity: vi.fn(),
  error: vi.fn(),
}));
vi.mock('@/services/background-proxy', () => ({ bgRequest: mocks.request }));
vi.mock('@/services/notes-tasks', () => ({
  listNoteTasks: mocks.tasks,
  listTaskActivity: mocks.activity,
  markTaskActivityRead: vi.fn(),
  setNoteTaskStatus: vi.fn(),
}));
vi.mock('@/services/tldw/TldwAuth', () => ({
  tldwAuth: { getCurrentUser: vi.fn(async () => ({ id: 'alice', is_active: true })) },
}));
vi.mock('../hooks/useNotesGraphAuthorityScope', () => ({
  createNotesGraphAuthorityScope: (_origin: string, user: string) => user,
}));
vi.mock('@/services/settings/registry', async (importOriginal) => ({
  ...(await importOriginal<typeof import('@/services/settings/registry')>()),
  getSetting: vi.fn(async () => null),
  setSetting: vi.fn(async () => undefined),
  clearSetting: vi.fn(),
}));

const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  let reject!: (error: Error) => void;
  const promise = new Promise<T>((yes, no) => {
    resolve = yes;
    reject = no;
  });
  return { promise, resolve, reject };
};

const renderEditor = (isOnline = false) => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } });
  const deps: UseNotesEditorStateDeps = {
    authorityScope: 'alice',
    isOnline,
    connectionConfig: {
      serverUrl: 'https://notes.example.test',
      authMode: 'single-user',
      apiKey: 'test-key',
    },
    isMobileViewport: false,
    message: { error: mocks.error } as unknown as UseNotesEditorStateDeps['message'],
    confirmDanger: vi.fn(async () => true),
    queryClient,
    t: (key) => key,
    listMode: 'active',
    setListMode: vi.fn(),
    data: [],
    refetch: vi.fn(async () => undefined),
    setPage: vi.fn(),
    setQuery: vi.fn(),
    setQueryInput: vi.fn(),
    setKeywordTokens: vi.fn(),
    setSelectedNotebookId: vi.fn(),
    setMobileSidebarOpen: vi.fn(),
    editorKeywords: [],
    setEditorKeywords: vi.fn(),
    keywordSuggestionReturnFocusRef: { current: null },
    setKeywordSuggestionOptions: vi.fn(),
    setKeywordSuggestionSelection: vi.fn(),
    editorDisabled: false,
  };
  return renderHook(({ scope }) => useNotesEditorState({ ...deps, authorityScope: scope }), {
    initialProps: { scope: 'alice' },
    wrapper: ({ children }) => (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    ),
  });
};

describe('structured drafts', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
    mocks.tasks.mockResolvedValue({ tasks: [], reconciliation: null });
    mocks.activity.mockResolvedValue({ events: [] });
  });

  it('keeps legacy absent language unknown and preserves future language exactly', () => {
    const queue = normalizeOfflineDraftQueue({
      legacy: { content: '<>&' },
      future: { content: '雪', content_language: ' future ', revision: 7 },
    });
    expect(queue.legacy.content_language).toBeUndefined();
    expect(queue.future).toMatchObject({
      content: '雪',
      content_language: ' future ',
      revision: 7,
      draftVersion: 2,
    });
  });

  it('retains an empty language-bearing draft through a hook restart', async () => {
    mocks.request.mockResolvedValue({
      id: 'n',
      title: 'Saved',
      content: 'Original',
      content_language: 'json',
      version: 1,
    });
    const first = renderEditor();
    await act(async () => {
      await first.result.current.loadDetail('n');
    });
    act(() => {
      first.result.current.setContent('');
      first.result.current.setContentLanguage('yaml');
      first.result.current.setIsDirty(true);
    });
    await act(async () => {
      await first.result.current.saveNote({ showSuccessMessage: false });
    });
    first.unmount();
    const second = renderEditor();
    await act(async () => {
      await second.result.current.loadDetail('n');
    });
    expect(second.result.current.content).toBe('');
    expect(second.result.current.contentLanguage).toBe('yaml');
    expect(second.result.current.markdownAllowed).toBe(false);
  });

  it('does not derive WYSIWYG before language is known or for future modes', async () => {
    const view = renderEditor();
    act(() => {
      view.result.current.setEditorInputMode('wysiwyg');
      view.result.current.setContent('<>&');
    });
    expect(view.result.current.editorInputMode).toBe('markdown');
    expect(view.result.current.markdownAllowed).toBe(false);
    mocks.request.mockResolvedValue({
      id: 'n',
      title: 'Note',
      content: '<>&',
      content_language: 'future-mode',
      version: 1,
    });
    await act(async () => {
      await view.result.current.loadDetail('n');
    });
    expect(view.result.current.content).toBe('<>&');
    expect(view.result.current.editorInputMode).toBe('markdown');
  });

  it('retains a newer language revision after an older complete acknowledgement', async () => {
    const response = deferred<unknown>();
    mocks.request.mockImplementation(({ method, path }) => {
      if (path.endsWith('/capabilities')) return Promise.resolve({ content_language: true });
      if (method === 'PUT') return response.promise;
      return Promise.resolve({
        id: 'n',
        title: 'Saved',
        content: 'old',
        content_language: 'json',
        version: 1,
      });
    });
    const view = renderEditor(true);
    await act(async () => {
      await view.result.current.loadDetail('n');
    });
    act(() => {
      view.result.current.setContentLanguage('yaml');
    });
    let saving!: Promise<boolean>;
    act(() => {
      saving = view.result.current.saveNote({ showSuccessMessage: false });
    });
    await waitFor(() =>
      expect(mocks.request.mock.calls.some(([arg]) => arg.method === 'PUT')).toBe(true)
    );
    act(() => {
      view.result.current.setContentLanguage('jsonl');
      view.result.current.setContent('new');
    });
    await act(async () => {
      response.resolve({ id: 'n', content: 'old', content_language: 'yaml', version: 2 });
      await saving;
    });
    expect(view.result.current.contentLanguage).toBe('jsonl');
    expect(view.result.current.isDirty).toBe(true);
    expect(view.result.current.offlineDraftQueue['note:n'].content_language).toBe('jsonl');
    expect(view.result.current.offlineDraftQueue['note:n'].content).toBe('new');
  });

  it('keeps a language draft after an incomplete acknowledgement', async () => {
    mocks.request.mockImplementation(({ method, path }) => {
      if (path.endsWith('/capabilities')) return Promise.resolve({ content_language: true });
      if (method === 'PUT') return Promise.resolve({ id: 'n', content: 'old', version: 2 });
      return Promise.resolve({
        id: 'n',
        title: 'Saved',
        content: 'old',
        content_language: 'json',
        version: 1,
      });
    });
    const view = renderEditor(true);
    await act(async () => {
      await view.result.current.loadDetail('n');
    });
    act(() => {
      view.result.current.setContentLanguage('yaml');
    });
    await act(async () => {
      expect(await view.result.current.saveNote({ showSuccessMessage: false })).toBe(false);
    });
    expect(view.result.current.offlineDraftQueue['note:n'].content_language).toBe('yaml');
    expect(view.result.current.saveIndicator).toBe('error');
  });

  it('blocks navigation on quota failure and keeps an empty draft', async () => {
    mocks.request.mockResolvedValue({
      id: 'n',
      title: 'Saved',
      content: 'old',
      content_language: 'json',
      version: 1,
    });
    const view = renderEditor();
    await act(async () => {
      await view.result.current.loadDetail('n');
    });
    act(() => {
      view.result.current.setContent('');
      view.result.current.setIsDirty(true);
    });
    const storage = vi.spyOn(Storage.prototype, 'setItem').mockImplementation(() => {
      throw new Error('quota');
    });
    try {
      await act(async () => {
        expect(await view.result.current.handleSelectNote('other')).toBe(false);
      });
      expect(view.result.current.content).toBe('');
      expect(view.result.current.selectedId).toBe('n');
    } finally {
      storage.mockRestore();
    }
  });
});
