import React from 'react';
import type { TldwConfig } from '@/services/tldw/TldwApiClient';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import Head from 'next/head';
import initHeadManager from 'next/dist/client/head-manager';
import { HeadManagerContext } from 'next/dist/shared/lib/head-manager-context.shared-runtime';

const fixture = vi.hoisted(() => ({
  t: (key: string, fallback?: string | { defaultValue?: string }) =>
    typeof fallback === 'string' ? fallback : fallback?.defaultValue || key,
  config: {
    serverUrl: 'https://title.test',
    authMode: 'single-user',
    apiKey: 'synthetic-a',
  } as TldwConfig,
  authorityLoading: false,
  rows: new Map<string, string>(),
  listeners: new Set<() => void>(),
  read: vi.fn(),
  update: vi.fn(),
  updateChat: vi.fn(),
  snapshot: vi.fn(),
}));
vi.mock('next/dynamic', () => ({ default: () => () => <main>Chat workspace</main> }));
vi.mock('react-i18next', () => ({ useTranslation: () => ({ t: fixture.t }) }));
vi.mock('react-router-dom', () => ({
  useLocation: () => ({ pathname: '/chat' }),
  useNavigate: () => vi.fn(),
}));
vi.mock('@/hooks/useCanonicalConnectionConfig', () => ({
  useCanonicalConnectionConfig: () => ({
    config: fixture.config,
    authorityLoading: fixture.authorityLoading,
  }),
}));
vi.mock('@/store/option', async () => {
  const { create } = await import('zustand');
  return {
    useStoreMessageOption: create((set: (value: object) => void) => ({
      historyId: 'local-a',
      serverChatId: null,
      serverChatTitle: null,
      serverChatVersion: 1,
      temporaryChat: false,
      setServerChatTitle: (serverChatTitle: string) => set({ serverChatTitle }),
      setServerChatVersion: (serverChatVersion: number) => set({ serverChatVersion }),
    })),
  };
});
vi.mock('~/hooks/useMessageOption', async () => {
  const { useStoreMessageOption } = await import('@/store/option');
  return {
    useMessageOption: () => ({
      ...useStoreMessageOption(),
      clearChat: vi.fn(),
      setTemporaryChat: vi.fn(),
    }),
  };
});
vi.mock('@/store/playground-session', async () => {
  const { create } = await import('zustand');
  return {
    usePlaygroundSessionStore: create(() => ({
      scopeKey: null,
      lastUpdated: 0,
      historyId: 'local-a',
      serverChatId: null,
    })),
  };
});
// The production hook uses Dexie's reactive adapter. This fixture controls its
// read/write notifications; native UAT verifies the browser's actual database.
// Resolve the shared UI dependency; the web workspace installs a different version.
vi.mock('../../../packages/ui/node_modules/dexie-react-hooks', () => ({
  useLiveQuery: <T,>(query: () => Promise<T>, dependencies: React.DependencyList) => {
    const [value, setValue] = React.useState<T>();
    React.useEffect(() => {
      let active = true;
      const read = () => {
        void Promise.resolve()
          .then(query)
          .then((result) => {
            if (active) setValue(result);
          });
      };
      fixture.listeners.add(read);
      read();
      return () => {
        active = false;
        fixture.listeners.delete(read);
      };
      // eslint-disable-next-line react-hooks/exhaustive-deps -- Mirror the reactive adapter dependency contract.
    }, dependencies);
    return value;
  },
}));
vi.mock('@/db', () => ({ getTitleById: async () => '', updateHistory: async () => undefined }));
vi.mock('@/db/dexie/helpers', () => ({
  getTitleById: (...args: unknown[]) => fixture.read(...args),
  updateHistory: (...args: unknown[]) => fixture.update(...args),
}));
vi.mock('@/db/dexie/chat-persistence-transaction', () => ({
  runChatPersistenceTransaction: async (signal: AbortSignal, work: () => Promise<unknown>) => {
    if (signal.aborted) throw new Error('aborted');
    return work();
  },
}));
vi.mock('@/services/service-prompts', () => ({
  loadServicePromptSnapshot: (...args: unknown[]) => fixture.snapshot(...args),
}));
vi.mock('@/services/tldw/TldwApiClient', () => ({
  tldwClient: {
    getConfig: async () => fixture.config,
    updateChat: (...args: unknown[]) => fixture.updateChat(...args),
    listConversationShareLinks: async () => [],
  },
}));
vi.mock('@/hooks/useSetting', () => ({ useSetting: () => [false, vi.fn()] }));
vi.mock('@/hooks/useDarkmode', () => ({
  useDarkMode: () => ({ mode: 'dark', toggleDarkMode: vi.fn() }),
}));
vi.mock('@/hooks/useSelectedCharacter', () => ({ useSelectedCharacter: () => [null, vi.fn()] }));
vi.mock('@/components/Sidepanel/Chat/TtsClipsDrawer', () => ({ TtsClipsDrawer: () => null }));
vi.mock('@/components/Layouts/HeaderShortcuts', () => ({ HeaderShortcuts: () => null }));
vi.mock('antd', () => ({
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  Modal: () => null,
  Button: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button {...props}>{children}</button>
  ),
  InputNumber: () => null,
  Input: ({
    onPressEnter,
    size: _size,
    ...props
  }: React.InputHTMLAttributes<HTMLInputElement> & {
    onPressEnter?: () => void;
    size?: unknown;
  }) => (
    <input
      {...props}
      onKeyDown={(event) => {
        if (event.key === 'Enter') onPressEnter?.();
      }}
    />
  ),
}));

import ChatPage from '@web/pages/chat';
import { Header } from '@/components/Layouts/Header';
import { useStoreMessageOption } from '@/store/option';
import { usePlaygroundSessionStore } from '@/store/playground-session';

const change = (patch: object) =>
  act(() => {
    useStoreMessageOption.setState(patch);
  });
const deferred = <T,>() => {
  let resolve!: (value: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
};

beforeEach(async () => {
  vi.clearAllMocks();
  fixture.authorityLoading = false;
  fixture.config = {
    serverUrl: 'https://title.test',
    authMode: 'single-user',
    apiKey: 'synthetic-a',
  };
  fixture.rows = new Map([
    ['local-a', 'Local Cedar'],
    ['local-b', 'Local Birch'],
  ]);
  fixture.read.mockImplementation(async (id: string) => fixture.rows.get(id) || '');
  fixture.update.mockImplementation(async (id: string, title: string) => {
    fixture.rows.set(id, title);
    fixture.listeners.forEach((read) => read());
  });
  fixture.snapshot.mockImplementation(async () => {
    const signal = new AbortController().signal;
    return {
      scopeSignal: signal,
      scopeInvalidatedSignal: signal,
      requestScope: { config: fixture.config },
      release: vi.fn(),
    };
  });
  fixture.updateChat.mockImplementation(async (_id: string, patch: { title: string }) => ({
    id: 'server-a',
    title: patch.title,
    version: 2,
  }));
  useStoreMessageOption.setState({
    historyId: 'local-a',
    serverChatId: null,
    serverChatTitle: null,
    serverChatVersion: 1,
    temporaryChat: false,
  });
  usePlaygroundSessionStore.setState({
    scopeKey: null,
    lastUpdated: 0,
    historyId: 'local-a',
    serverChatId: null,
  });
});
afterEach(() => {
  cleanup();
  document.head.innerHTML = '';
});

describe('active Chat title ownership', () => {
  it('owns the real Next head through metadata hydration, rerender and remount', async () => {
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar server conversation' });
    const manager = initHeadManager();
    const view = render(
      <HeadManagerContext.Provider value={manager}>
        <ChatPage />
      </HeadManagerContext.Provider>
    );
    await waitFor(() => expect(document.title).toBe('Cedar server conversation | tldw'));
    view.rerender(
      <HeadManagerContext.Provider value={manager}>
        <ChatPage />
        <Head>
          <meta name="description" content="Chat" />
        </Head>
      </HeadManagerContext.Provider>
    );
    await waitFor(() => expect(document.title).toBe('Cedar server conversation | tldw'));
    change({ serverChatTitle: 'Hydrated Cedar' });
    await waitFor(() => expect(document.title).toBe('Hydrated Cedar | tldw'));
    view.unmount();
    render(
      <HeadManagerContext.Provider value={manager}>
        <ChatPage />
      </HeadManagerContext.Provider>
    );
    await waitFor(() => expect(document.title).toBe('Hydrated Cedar | tldw'));
    change({ historyId: null, serverChatId: null, serverChatTitle: null });
    await waitFor(() => expect(document.title).toBe('Chat | tldw'));
  });

  it('renders the canonical server title in the actual Header despite an empty legacy catalog', async () => {
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar server conversation' });
    render(<Header />);
    await screen.findByRole('button', { name: 'Cedar server conversation', exact: true });
    change({ serverChatTitle: 'Hydrated Cedar' });
    await screen.findByRole('button', { name: 'Hydrated Cedar', exact: true });
  });

  it('reads and renames a local-only title through Dexie without a server request', async () => {
    const view = render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Local Cedar', exact: true }));
    const input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Renamed local' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.rows.get('local-a')).toBe('Renamed local'));
    view.unmount();
    render(<Header />);
    await screen.findByRole('button', { name: 'Renamed local', exact: true });
    expect(fixture.snapshot).not.toHaveBeenCalled();
    expect(fixture.updateChat).not.toHaveBeenCalled();
  });

  it('does not publish a delayed local title after switching histories', async () => {
    const old = deferred<string>();
    fixture.read.mockImplementation((id: string) =>
      id === 'local-a' ? old.promise : Promise.resolve('Local Birch')
    );
    render(<Header />);
    change({ historyId: 'local-b' });
    await screen.findByRole('button', { name: 'Local Birch', exact: true });
    await act(async () => {
      old.resolve('Private old Cedar');
      await old.promise;
    });
    expect(screen.queryByText('Private old Cedar')).toBeNull();
  });
  it('renames the saved conversation and its local mirror with captured scope and version', async () => {
    change({
      serverChatId: 'server-a',
      serverChatTitle: 'Cedar server conversation',
      serverChatVersion: 7,
    });
    render(<Header />);
    fireEvent.click(
      await screen.findByRole('button', { name: 'Cedar server conversation', exact: true })
    );
    const input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Saved Cedar renamed' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() =>
      expect(fixture.updateChat).toHaveBeenCalledWith(
        'server-a',
        { title: 'Saved Cedar renamed' },
        expect.objectContaining({
          expectedVersion: 7,
          requestScope: expect.any(Object),
          signal: expect.any(AbortSignal),
        })
      )
    );
    await waitFor(() => expect(fixture.rows.get('local-a')).toBe('Saved Cedar renamed'));
    expect(useStoreMessageOption.getState().serverChatTitle).toBe('Saved Cedar renamed');
    expect(useStoreMessageOption.getState().serverChatVersion).toBe(2);
  });

  it('discards a delayed saved rename after A to B to A even with the same conversation IDs', async () => {
    const update = deferred<{ id: string; title: string; version: number }>();
    fixture.updateChat.mockReturnValue(update.promise);
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    const input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Stale rename' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.updateChat).toHaveBeenCalledTimes(1));
    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:auth-principal-changed', { detail: { kind: 'logout' } })
      );
    });
    change({ historyId: 'local-b', serverChatId: 'server-b', serverChatTitle: 'Birch' });
    change({ historyId: 'local-a', serverChatId: 'server-a', serverChatTitle: 'Fresh Cedar' });
    await act(async () => {
      update.resolve({ id: 'server-a', title: 'Stale rename', version: 8 });
      await update.promise;
    });
    expect(useStoreMessageOption.getState().serverChatTitle).toBe('Fresh Cedar');
    expect(fixture.rows.get('local-a')).toBe('Local Cedar');
  });

  it('does not restore an old local title on a canonical account change before Chat clears its IDs', async () => {
    const old = deferred<string>();
    fixture.read.mockReturnValue(old.promise);
    const view = render(<Header />);
    await waitFor(() => expect(fixture.read).toHaveBeenCalledTimes(1));
    fixture.config = { ...fixture.config, apiKey: 'synthetic-b' };
    view.rerender(<Header />);
    await act(async () => {
      old.resolve('Private Cedar');
      await old.promise;
    });
    expect(screen.queryByText('Private Cedar')).toBeNull();
    expect(fixture.read).toHaveBeenCalledTimes(1);
  });

  it('keeps an in-progress title edit across a benign canonical config event', async () => {
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    const input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Draft title' } });
    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:config-updated', { detail: { authorityChanged: false } })
      );
    });
    expect(screen.getByRole('textbox', { name: 'Rename conversation' })).toHaveValue('Draft title');
  });

  it('updates both mounted title surfaces through the reactive local adapter and survives Strict Mode', async () => {
    const manager = initHeadManager();
    render(
      <React.StrictMode>
        <HeadManagerContext.Provider value={manager}>
          <ChatPage />
          <Header />
        </HeadManagerContext.Provider>
      </React.StrictMode>
    );
    fireEvent.click(await screen.findByRole('button', { name: 'Local Cedar', exact: true }));
    const input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Local shared title' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(document.title).toBe('Local shared title | tldw'));
    await screen.findByRole('button', { name: 'Local shared title', exact: true });
    expect(fixture.rows.get('local-a')).toBe('Local shared title');
  });

  it('does not dispatch a rename if scope resolution finishes under another account', async () => {
    const scope = deferred<unknown>();
    fixture.snapshot.mockReturnValue(scope.promise);
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    const input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Alice title' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.snapshot).toHaveBeenCalledTimes(1));
    fixture.config = { ...fixture.config, apiKey: 'synthetic-b' };
    const release = vi.fn();
    const signal = new AbortController().signal;
    await act(async () => {
      scope.resolve({ requestScope: { config: fixture.config }, scopeSignal: signal, release });
      await scope.promise;
    });
    expect(fixture.updateChat).not.toHaveBeenCalled();
    expect(fixture.update).not.toHaveBeenCalled();
    expect(release).toHaveBeenCalledTimes(1);
  });

  it('retains a pending saved rename through same-owner canonical token rotation', async () => {
    const token = (jti: string) =>
      'test.' + btoa(JSON.stringify({ sub: 'Alice', jti })) + '.signature';
    fixture.config = {
      serverUrl: 'https://title.test',
      authMode: 'multi-user',
      accessToken: token('before'),
    };
    const update = deferred<{ id: string; title: string; version: number }>();
    fixture.updateChat.mockReturnValue(update.promise);
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    const view = render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    const input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Rotated own title' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.updateChat).toHaveBeenCalledTimes(1));
    fixture.config = { ...fixture.config, accessToken: token('after') };
    view.rerender(<Header />);
    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:config-updated', { detail: { authorityChanged: false } })
      );
    });
    await act(async () => {
      update.resolve({ id: 'server-a', title: 'Rotated own title', version: 3 });
      await update.promise;
    });
    await waitFor(() => expect(fixture.rows.get('local-a')).toBe('Rotated own title'));
    expect(useStoreMessageOption.getState().serverChatTitle).toBe('Rotated own title');
  });

  it('masks the first render while canonical authority is unresolved', async () => {
    fixture.authorityLoading = true;
    change({ serverChatId: 'server-a', serverChatTitle: 'Private Cedar' });
    const manager = initHeadManager();
    render(
      <HeadManagerContext.Provider value={manager}>
        <ChatPage />
        <Header />
      </HeadManagerContext.Provider>
    );
    expect(screen.queryByText('Private Cedar')).toBeNull();
    await waitFor(() => expect(document.title).toBe('Chat | tldw'));
    expect(fixture.read).not.toHaveBeenCalled();
  });

  it('keeps the last saved title when the server rejects rename, and permits retry', async () => {
    const error = vi.spyOn(console, 'error').mockImplementation(() => undefined);
    fixture.updateChat.mockRejectedValueOnce(new Error('Save failed'));
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    render(<Header />);
    const rename = async () => {
      fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
      const input = screen.getByRole('textbox', { name: 'Rename conversation' });
      fireEvent.change(input, { target: { value: 'Retry title' } });
      fireEvent.keyDown(input, { key: 'Enter' });
    };
    await rename();
    await waitFor(() => expect(error).toHaveBeenCalledTimes(1));
    expect(fixture.rows.get('local-a')).toBe('Local Cedar');
    fireEvent.click(await screen.findByRole('button', { name: 'Retry rename', exact: true }));
    const retryInput = screen.getByRole('textbox', { name: 'Rename conversation' });
    expect(retryInput).toHaveValue('Retry title');
    fireEvent.keyDown(retryInput, { key: 'Enter' });
    await screen.findByRole('button', { name: 'Retry title', exact: true });
    error.mockRestore();
  });

  it("never paints the previous conversation's edit draft during a selection switch", async () => {
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    const paints: string[] = [];
    render(
      <React.Profiler
        id="header"
        onRender={() => {
          paints.push(
            (
              document.querySelector(
                'input[aria-label="Rename conversation"]'
              ) as HTMLInputElement | null
            )?.value || ''
          );
        }}
      >
        <Header />
      </React.Profiler>
    );
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    fireEvent.change(screen.getByRole('textbox', { name: 'Rename conversation' }), {
      target: { value: 'Private draft title' },
    });
    paints.length = 0;
    change({ historyId: 'local-b', serverChatId: 'server-b', serverChatTitle: 'Birch' });
    expect(paints).not.toContain('Private draft title');
  });
  it('retains a newer title edit until the pending save finishes, then commits it explicitly', async () => {
    const first = deferred<{ id: string; title: string; version: number }>();
    fixture.updateChat.mockReturnValueOnce(first.promise);
    change({ serverChatId: 'server-a', serverChatTitle: 'Cedar' });
    render(<Header />);
    fireEvent.click(await screen.findByRole('button', { name: 'Cedar', exact: true }));
    let input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'First saved title' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.updateChat).toHaveBeenCalledTimes(1));
    fireEvent.click(screen.getByRole('button', { name: 'Cedar', exact: true }));
    input = screen.getByRole('textbox', { name: 'Rename conversation' });
    fireEvent.change(input, { target: { value: 'Second title draft' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(screen.getByRole('textbox', { name: 'Rename conversation' })).toHaveValue('Second title draft');
    expect(screen.getByRole('status')).toHaveTextContent('Saving conversation title');
    expect(fixture.updateChat).toHaveBeenCalledTimes(1);
    await act(async () => { first.resolve({ id: 'server-a', title: 'First saved title', version: 7 }); await first.promise; });
    await waitFor(() => expect(screen.queryByRole('status')).toBeNull());
    expect(screen.getByRole('textbox', { name: 'Rename conversation' })).toHaveValue('Second title draft');
    fireEvent.keyDown(input, { key: 'Enter' });
    await waitFor(() => expect(fixture.rows.get('local-a')).toBe('Second title draft'));
    expect(fixture.updateChat).toHaveBeenLastCalledWith('server-a', { title: 'Second title draft' }, expect.objectContaining({ expectedVersion: 7 }));
  });

});
