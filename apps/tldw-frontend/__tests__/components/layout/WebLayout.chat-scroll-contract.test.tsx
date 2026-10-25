import { HistorySelectionProvider, useHistorySelectionContext } from "@/hooks/chat/useHistorySelection";
// @vitest-environment jsdom
import React from 'react';
import { createPortal } from 'react-dom';
import { useChatShortcuts } from '@/hooks/keyboard/useKeyboardShortcuts';
import { useQuickChatStore } from '@/store/quick-chat';
import { QuickChatHelperButton } from '@/components/Common/QuickChatHelper';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderToStaticMarkup } from 'react-dom/server';
import { act, cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useOptionLayoutShellOverrides } from '@/components/Layouts/Layout';
import OptionLayout from '../../../components/layout/WebLayout';
import { PlaygroundEmpty } from '@/components/Option/Playground/PlaygroundEmpty';
import { useTutorialStore } from '@/store/tutorials';

const testModulePath = import.meta.url.startsWith('file:')
  ? fileURLToPath(import.meta.url)
  : import.meta.url;
const webLayoutSourcePath = resolve(
  dirname(testModulePath),
  '../../../components/layout/WebLayout.tsx'
);
const routerState = vi.hoisted(() => ({
  location: {
    pathname: '/chat',
    key: 'chat-start',
    search: '',
    hash: '',
  },
  navigate: vi.fn(),
}));

const layoutUiState = vi.hoisted(() => ({
  value: {
    chatSidebarCollapsed: true,
    setChatSidebarCollapsed: vi.fn(),
  },
}));

const routeTransitionState = vi.hoisted(() => ({
  value: {
    active: false,
    pendingPath: null as string | null,
    startedAt: null as number | null,
    stop: vi.fn(),
  },
}));

const messageOptionState = vi.hoisted(() => ({
  value: {
    clearChat: vi.fn(),
    useOCR: false,
    chatMode: 'chat',
    setChatMode: vi.fn(),
    webSearch: false,
    setWebSearch: vi.fn(),
  },
}));

const optionStoreState = vi.hoisted(() => ({
  value: {
    historyId: null as string | null,
    serverChatId: null as string | null,
  },
}));

const capabilityProbe = vi.hoisted(() => ({
  actualDrawer: false,
  paletteProps: null as { onToggleSidebar?: () => void; onSwitchModel?: () => void } | null,
}));

const connectionState = vi.hoisted(() => ({
  value: {
    checkOnce: vi.fn(async () => undefined),
    phase: 'connected',
    isConnected: true,
    isChecking: false,
    lastCheckedAt: 100 as number | null,
    checksSinceConfigChange: 1,
    consecutiveFailures: 0,
  },
}));

function deferred() {
  let resolve!: () => void;
  const promise = new Promise<void>((resolvePromise) => {
    resolve = resolvePromise;
  });
  return { promise, resolve };
}

const confirmDangerMock = vi.hoisted(() => vi.fn(async () => false));
const getUnreadCountMock = vi.hoisted(() => vi.fn(async () => ({ unread_count: 0 })));
const storageState = vi.hoisted(() => ({
  stickyChatInput: true,
}));
const featureFlagState = vi.hoisted(() => ({
  showChatSidebar: false,
}));
const mediaQueryState = vi.hoisted(() => ({
  isDesktop: false,
  isMobile: false,
}));
const chatSidebarMockState = vi.hoisted(() => ({
  props: [] as Array<Record<string, unknown>>,
}));
const backendUnavailableGateState = vi.hoisted(() => ({
  props: [] as Array<Record<string, unknown>>,
}));

vi.mock('antd', async (original) => {
  const actual = await original<typeof import('antd')>();
  return {
    Drawer: (props: React.ComponentProps<typeof actual.Drawer>) =>
      capabilityProbe.actualDrawer ? (
        <actual.Drawer {...props} />
      ) : props.open ? (
        <div data-testid="drawer">{props.children}</div>
      ) : null,
    Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  };
});

vi.mock('lucide-react', async (original) => ({
  ...(await original<typeof import('lucide-react')>()),
  EraserIcon: () => null,
  PanelLeftOpen: () => null,
  XIcon: () => null,
}));

vi.mock('@/components/Common/IconButton', () => ({
  IconButton: ({ children, ...props }: React.ButtonHTMLAttributes<HTMLButtonElement>) => (
    <button type="button" {...props}>
      {children}
    </button>
  ),
}));

vi.mock('react-router-dom', () => ({
  useLocation: () => routerState.location,
  useNavigate: () => routerState.navigate,
}));

vi.mock('react-i18next', () => ({
  useTranslation: () => ({
    t: (
      key: string,
      fallback?:
        | string
        | {
            defaultValue?: string;
          }
    ) => {
      if (typeof fallback === 'string') return fallback;
      if (fallback && typeof fallback === 'object' && fallback.defaultValue) {
        return fallback.defaultValue;
      }
      return key;
    },
  }),
}));

vi.mock('@tanstack/react-query', () => ({
  useQueryClient: () => ({
    invalidateQueries: vi.fn(),
    setQueryData: vi.fn(),
    getQueryData: vi.fn(),
  }),
}));

vi.mock('zustand/react/shallow', () => ({
  useShallow: <T,>(selector: T) => selector,
}));

vi.mock('@/libs/class-name', () => ({
  classNames: (...parts: Array<string | false | null | undefined>) =>
    parts.filter(Boolean).join(' '),
}));

vi.mock('@/db/dexie/chat', () => ({
  PageAssistDatabase: class {
    async deleteAllChatHistory(): Promise<void> {}
  },
}));

vi.mock('@/hooks/useMessageOption', () => ({
  useMessageOption: () => messageOptionState.value,
}));

vi.mock('@/hooks/keyboard/useKeyboardShortcuts', async (original) => {
  const actual = await original<typeof import('@/hooks/keyboard/useKeyboardShortcuts')>();
  return {
    isMac: false,
    useChatShortcuts: actual.useChatShortcuts,
    useSidebarShortcuts: actual.useSidebarShortcuts,
    useQuickChatShortcuts: actual.useQuickChatShortcuts,
    useModeNavigationShortcuts: () => undefined,
  };
});

vi.mock('@/store/option', async (original) => {
  const actual = await original<typeof import('@/store/option')>();
  return {
    ...actual,
    useStoreMessageOption: Object.assign(
      (selector: (state: typeof optionStoreState.value) => unknown) =>
        selector(optionStoreState.value),
      {
        getState: actual.useStoreMessageOption.getState,
        setState: actual.useStoreMessageOption.setState,
        subscribe: actual.useStoreMessageOption.subscribe,
      }
    ),
  };
});

vi.mock('@/store/layout-ui', () => ({
  useLayoutUiStore: (selector: (state: typeof layoutUiState.value) => unknown) =>
    selector(layoutUiState.value),
}));

vi.mock('@/store/route-transition', () => ({
  useRouteTransitionStore: (selector: (state: typeof routeTransitionState.value) => unknown) =>
    selector(routeTransitionState.value),
}));

vi.mock('@/components/Common/QuickChatHelper', () => ({
  QuickChatHelperButton: () => {
    const { isOpen } = useQuickChatStore();
    return (
      <>
        <div data-testid="quick-chat-helper" />
        {isOpen &&
          createPortal(
            <div role="dialog" aria-label="Quick Chat Helper probe">
              Quick Chat mounted
            </div>,
            document.body
          )}
      </>
    );
  },
}));

vi.mock('@/components/Common/NotesDock', () => ({
  NotesDockHost: () => <div data-testid="notes-dock-host" />,
}));

vi.mock('@/components/Common/PersonaBuddy', () => ({
  BuddyShellHost: () => <div data-testid="buddy-shell-host" />,
  BuddyShellRenderContextProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/Common/Settings/CurrentChatModelSettings', () => ({
  CurrentChatModelSettings: () => null,
}));

vi.mock('@/components/Option/Sidebar', () => ({
  Sidebar: () => null,
}));

vi.mock('@/components/Layouts/Header', () => ({
  Header: ({
    onToggleSidebar,
    sidebarCollapsed,
  }: {
    onToggleSidebar?: () => void;
    sidebarCollapsed?: boolean;
  }) => (
    <button
      data-history-controller={useHistorySelectionContext() ? "present" : "absent"}
      type="button"
      data-testid="header"
      aria-label={sidebarCollapsed ? 'Expand sidebar' : 'Collapse sidebar'}
      onClick={onToggleSidebar}
    />
  ),
}));

vi.mock('@/components/Layouts/QuickIngestButton', () => ({
  QuickIngestModalHost: () => null,
}));

vi.mock('@/hooks/useMigration', () => ({
  useMigration: () => ({ isLoading: false }),
}));

vi.mock('@/hooks/useStorageMigrations', () => ({
  useStorageMigrations: () => undefined,
}));

vi.mock('@/hooks/useLayoutEffectsOwner', () => ({
  useLayoutEffectsOwner: () => false,
}));

vi.mock('@/hooks/useFeatureFlags', () => ({
  useChatSidebar: () => [featureFlagState.showChatSidebar],
}));

vi.mock('@/hooks/useMediaQuery', async () => {
  const actual =
    await vi.importActual<typeof import('@/hooks/useMediaQuery')>('@/hooks/useMediaQuery');

  return {
    ...actual,
    useDesktop: () => mediaQueryState.isDesktop,
    useMobile: () => mediaQueryState.isMobile,
  };
});

vi.mock('@/hooks/useSetting', () => ({
  useSetting: () => [''],
}));

vi.mock('@plasmohq/storage/hook', () => ({
  useStorage: (key: string, defaultValue: unknown) => {
    if (key === 'stickyChatInput') {
      return [storageState.stickyChatInput];
    }
    return [defaultValue];
  },
}));

vi.mock('@/hooks/useServerOnline', () => ({
  useServerOnline: () => undefined,
}));

vi.mock('@/components/Common/ChatSidebar', () => ({
  ChatSidebar: (props: Record<string, unknown>) => {
    chatSidebarMockState.props.push(props);
    return (
      <aside
        data-testid="chat-sidebar"
        className={props.className as string | undefined}
        data-history-controller={useHistorySelectionContext() ? 'present' : 'absent'}
      >
        <button onClick={() => (props.onConversationSelected as (() => void) | undefined)?.()}>
          Select saved conversation
        </button>
        <button onClick={() => (props.onToggleCollapse as (() => void) | undefined)?.()}>Collapse sidebar</button>
        <QuickChatHelperButton />
      </aside>
    );
  },
}));

vi.mock('@/components/Common/EventHosts', () => ({
  EventOnlyHosts: () => null,
}));

vi.mock('@/components/Common/PageAssistLoader', () => ({
  PageAssistLoader: () => <div data-testid="page-assist-loader" />,
}));

vi.mock('@/utils/settings-return', () => ({
  setSettingsReturnTo: vi.fn(),
}));

vi.mock('@/components/Common/Workflow', () => ({
  WorkflowIntegrationHost: () => null,
}));

vi.mock('@/routes/route-paths', async (original) => ({
  ...(await original<typeof import('@/routes/route-paths')>()),
  VIEWPORT_CONSTRAINED_PATHS: ['/chat'],
}));

vi.mock('@/services/settings/ui-settings', () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: 'chatBackgroundImage',
  HEADER_SHORTCUTS_EXPANDED_SETTING: 'headerShortcutsExpanded',
}));

vi.mock('@/services/settings/registry', async (original) => ({
  ...(await original<typeof import('@/services/settings/registry')>()),
  setSetting: vi.fn(async () => undefined),
}));

vi.mock('@/services/request-events', () => ({
  BACKEND_UNREACHABLE_EVENT: 'tldw:backend-unreachable',
}));

vi.mock('@/components/Common/BackendRecoveryUiContext', () => ({
  useBackendRecoveryUi: () => ({ fatalBackendRecoveryActive: false }),
}));

vi.mock('@web/components/layout/BackendUnavailableModalGate', () => ({
  BackendUnavailableModalGate: (props: Record<string, unknown>) => {
    backendUnavailableGateState.props.push(props);
    return props.backendUnavailableDetail ? (
      <div
        data-presentation={String(props.presentation)}
        data-testid="backend-unavailable-modal"
      >
        <span data-testid="backend-unavailable-detail">
          {JSON.stringify(props.backendUnavailableDetail)}
        </span>
        <button type="button" onClick={() => (props.onRetry as () => void)()}>
          Retry
        </button>
      </div>
    ) : null;
  },
}));

vi.mock('@web/lib/api/notifications', () => ({
  getUnreadCount: (...args: unknown[]) => getUnreadCountMock(...args),
  listNotifications: vi.fn(async () => ({ items: [], total: 0 })),
  subscribeNotificationsStream: vi.fn(() => () => undefined),
}));

vi.mock('@web/lib/api', () => ({
  getApiBaseUrl: () => 'https://api.example.test/api/v1',
}));

vi.mock('@web/lib/authStorage', () => ({
  getApiBearer: () => null,
  getApiKey: () => 'test-api-key',
  getSessionAccessToken: () => null,
  getEffectiveStoredTldwConfig: () => null,
}));

vi.mock('@web/components/ui/ToastProvider', () => ({
  useToast: () => ({ show: vi.fn() }),
}));

vi.mock('@/components/Common/CommandPaletteHost', () => ({
  CommandPaletteHost: ({
    commandPaletteProps,
  }: {
    commandPaletteProps: NonNullable<typeof capabilityProbe.paletteProps>;
  }) => {
    capabilityProbe.paletteProps = commandPaletteProps;
    return null;
  },
}));

vi.mock('@/components/Common/TutorialRunner', () => ({
  TutorialRunner: () => <div data-testid="tutorial-runner" />,
}));

vi.mock('@/components/Timeline', () => ({
  TimelineModal: () => null,
}));

vi.mock('@/hooks/useConnectionState', () => ({
  useIsConnected: () => connectionState.value.isConnected,
  useConnectionActions: () => ({ checkOnce: connectionState.value.checkOnce }),
  useConnectionState: () => ({
    phase: connectionState.value.phase,
    isConnected: connectionState.value.isConnected,
    isChecking: connectionState.value.isChecking,
    lastCheckedAt: connectionState.value.lastCheckedAt,
    checksSinceConfigChange: connectionState.value.checksSinceConfigChange,
    consecutiveFailures: connectionState.value.consecutiveFailures,
  }),
  useConnectionUxState: () => ({
    isChecking: connectionState.value.isChecking,
  }),
}));

vi.mock('@/types/connection', () => ({
  ConnectionPhase: {
    CONNECTED: 'connected',
    ERROR: 'error',
  },
}));

vi.mock('@/components/Common/confirm-danger', () => ({
  useConfirmDanger: () => confirmDangerMock,
}));

vi.mock('@/context/demo-mode', () => ({
  DemoModeProvider: ({ children }: { children: React.ReactNode }) => <>{children}</>,
  useDemoMode: () => ({ demoEnabled: false }),
}));

describe('WebLayout /chat scroll contract', () => {
  beforeEach(() => {
    delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell;
    vi.clearAllMocks();
    capabilityProbe.actualDrawer = false;
    capabilityProbe.paletteProps = null;
    useQuickChatStore.getState().setIsOpen(false);
    routerState.location.pathname = '/chat';
    routerState.location.key = 'chat-start';
    routerState.location.search = '';
    routerState.location.hash = '';
    storageState.stickyChatInput = true;
    featureFlagState.showChatSidebar = false;
    layoutUiState.value.chatSidebarCollapsed = true;
    mediaQueryState.isDesktop = false;
    mediaQueryState.isMobile = false;
    chatSidebarMockState.props = [];
    backendUnavailableGateState.props = [];
    connectionState.value.checkOnce = vi.fn(async () => undefined);
    connectionState.value.phase = 'connected';
    connectionState.value.isConnected = true;
    connectionState.value.isChecking = false;
    connectionState.value.lastCheckedAt = 100;
    connectionState.value.checksSinceConfigChange = 1;
    connectionState.value.consecutiveFailures = 0;
  });

  afterEach(() => {
    cleanup();
    delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell;
  });

  it('keeps genuinely nested WebUI layouts inside one React-owned shell', () => {
    const view = render(<OptionLayout><OptionLayout><input aria-label="Nested source" defaultValue="ORBIT-742" /></OptionLayout></OptionLayout>);
    expect(view.container.querySelectorAll('main#main-content')).toHaveLength(1);
    expect(view.container.querySelector('main#main-content')).toContainElement(screen.getByRole('textbox', { name: 'Nested source' }));
  });

  it('keeps the replacement root and composer stable while a departing shell retires', () => {
    const departing = { mounted: true, ownerId: 'departing-media', setOverrides: vi.fn() };
    (globalThis as typeof globalThis & { __tldwOptionShell?: typeof departing }).__tldwOptionShell = departing;
    const content = <input aria-label="Imported source" defaultValue="ORBIT-742 complete source" />;
    const view = render(<OptionLayout>{content}</OptionLayout>);
    const input = screen.getByRole('textbox', { name: 'Imported source' });
    expect(view.container.querySelector('main#main-content')).toContainElement(input);
    act(() => {
      // A retiring owner can only release its own registration.
      if (departing.ownerId === 'departing-media') {
        departing.mounted = false;
      }
    });
    routerState.location.search = '';
    routerState.location.key = 'clean-handoff-url';
    view.rerender(<OptionLayout>{content}</OptionLayout>);
    expect(screen.getByRole('textbox', { name: 'Imported source' })).toBe(input);
    expect(input).toHaveValue('ORBIT-742 complete source');
    expect(view.container.querySelectorAll('main#main-content')).toHaveLength(1);
  });

  it.each(['/__debug__/sidepanel-chat', '/chat'])(
    'keeps one current New shortcut owner on %s',
    (pathname) => {
      routerState.location.pathname = pathname;
      const newTab = vi.fn();
      function CompactNewControl() {
        useChatShortcuts(newTab);
        return null;
      }
      render(
        <OptionLayout hideHeader hideSidebar>
          {pathname === '/__debug__/sidepanel-chat' && <CompactNewControl />}
          <textarea aria-label="Exact draft" defaultValue="Keep this source.  " />
        </OptionLayout>
      );
      fireEvent.keyDown(document, { key: 'u', ctrlKey: true, shiftKey: true });
      expect(messageOptionState.value.clearChat).toHaveBeenCalledTimes(pathname === '/chat' ? 1 : 0);
      expect(newTab).toHaveBeenCalledTimes(pathname === '/chat' ? 0 : 1);
      expect(screen.getByRole('textbox', { name: 'Exact draft' })).toHaveValue('Keep this source.  ');
    }
  );

  it.each([
    { hideHeader: true, hideSidebar: false },
    { hideHeader: false, hideSidebar: true },
  ])('does not dispatch or advertise hidden shell hosts with %j', async (overrides) => {
    useTutorialStore.getState().closeHelpModal();
    render(
      <OptionLayout {...overrides}>
        <textarea aria-label="Exact draft" defaultValue="Keep this draft.  " />
      </OptionLayout>
    );
    fireEvent.keyDown(document, { key: 'b', ctrlKey: true });
    fireEvent.keyDown(document, { key: 'h', ctrlKey: true, shiftKey: true });
    expect(screen.queryByTestId('drawer')).not.toBeInTheDocument();
    expect(useQuickChatStore.getState().isOpen).toBe(false);
    expect(capabilityProbe.paletteProps?.onToggleSidebar).toBeUndefined();
    if (overrides.hideHeader) expect(capabilityProbe.paletteProps?.onSwitchModel).toBeUndefined();
    else expect(capabilityProbe.paletteProps?.onSwitchModel).toBeTypeOf('function');
    act(() => window.dispatchEvent(new CustomEvent('tldw:open-chat-sidebar')));
    expect(screen.queryByTestId('drawer')).not.toBeInTheDocument();
    act(() => window.dispatchEvent(new CustomEvent('tldw:open-shortcuts-modal')));
    const help = await screen.findByRole('dialog', { name: 'Page Help' });
    expect(within(help).queryByText('Toggle sidebar')).not.toBeInTheDocument();
    expect(within(help).queryByText('Toggle Quick Chat Helper')).not.toBeInTheDocument();
    expect(screen.getByRole('textbox', { name: 'Exact draft' })).toHaveValue('Keep this draft.  ');
    fireEvent.keyDown(help, { key: 'Escape' });
  });

  it('keeps available legacy sidebar and Quick Chat keyboard actions working', () => {
    render(
      <OptionLayout>
        <textarea aria-label="Exact draft" defaultValue="Visible draft" />
      </OptionLayout>
    );
    fireEvent.keyDown(document, { key: 'b', ctrlKey: true });
    expect(screen.getByTestId('drawer')).toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'b', ctrlKey: true });
    expect(screen.queryByTestId('drawer')).not.toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'h', ctrlKey: true, shiftKey: true });
    expect(screen.getByRole('dialog', { name: 'Quick Chat Helper probe' })).toBeVisible();
    expect(capabilityProbe.paletteProps?.onToggleSidebar).toBeTypeOf('function');
    fireEvent.keyDown(document, { key: 'h', ctrlKey: true, shiftKey: true });
    expect(screen.queryByRole('dialog', { name: 'Quick Chat Helper probe' })).not.toBeInTheDocument();
  });

  it('uses a single named mobile sidebar frame and keeps its collapse action available', async () => {
    capabilityProbe.actualDrawer = true;
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isMobile = true;
    render(<OptionLayout><div>Chat content</div></OptionLayout>);
    act(() => window.dispatchEvent(new CustomEvent('tldw:open-chat-sidebar')));
    const dialog = await screen.findByRole('dialog', { name: 'Chats' });
    expect(within(dialog).getByRole('button', { name: 'Collapse sidebar' })).toBeVisible();
    expect(within(dialog).queryByRole('button', { name: 'Close', exact: true })).toBeNull();
    fireEvent.click(within(dialog).getByRole('button', { name: 'Collapse sidebar' }));
    await waitFor(() => expect(screen.queryByRole('dialog', { name: 'Chats' })).toBeNull());
  });

  it('demand-mounts the helper in a cold real mobile Drawer without opening the sidebar', async () => {
    capabilityProbe.actualDrawer = true;
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isMobile = true;
    render(
      <OptionLayout>
        <textarea aria-label="Exact draft" defaultValue="Mobile draft" />
      </OptionLayout>
    );
    expect(screen.queryByTestId('chat-sidebar')).not.toBeInTheDocument();
    fireEvent.keyDown(document, { key: 'h', ctrlKey: true, shiftKey: true });
    expect(await screen.findByRole('dialog', { name: 'Quick Chat Helper probe' })).toBeVisible();
    expect(screen.getByTestId('chat-sidebar')).not.toBeVisible();
    expect(screen.getByRole('textbox', { name: 'Exact draft' })).toHaveValue('Mobile draft');
  });

  it('keeps an ambiguous backend-unreachable candidate hidden when forced recovery finishes connected', async () => {
    const check = deferred();
    connectionState.value.checkOnce = vi.fn(() => check.promise);
    const { rerender } = render(
      <OptionLayout>
        <div data-testid="route-content">Research workspace route</div>
      </OptionLayout>
    );

    await act(async () => undefined);

    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:backend-unreachable', {
          detail: {
            method: 'GET',
            path: '/api/v1/llm/models/metadata',
            message: 'Failed to fetch',
            source: 'direct',
            timestamp: Date.now(),
          },
        })
      );
    });

    expect(connectionState.value.checkOnce).toHaveBeenCalledWith({ force: true });
    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();

    connectionState.value.isChecking = true;
    rerender(
      <OptionLayout>
        <div data-testid="route-content">Research workspace route</div>
      </OptionLayout>
    );
    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();

    connectionState.value.isChecking = false;
    connectionState.value.lastCheckedAt = 101;
    connectionState.value.checksSinceConfigChange = 2;
    check.resolve();
    await act(async () => check.promise);
    rerender(
      <OptionLayout>
        <div data-testid="route-content">Research workspace route</div>
      </OptionLayout>
    );

    await waitFor(() => {
      expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();
    });
  });

  it('shows diagnostics only after a fresh forced check confirms an outage and rechecks on Retry', async () => {
    const firstCheck = deferred();
    const retryCheck = deferred();
    connectionState.value.checkOnce = vi
      .fn()
      .mockImplementationOnce(() => firstCheck.promise)
      .mockImplementationOnce(() => retryCheck.promise);
    const route = <div data-testid="route-content">Research workspace route</div>;
    const { rerender } = render(<OptionLayout>{route}</OptionLayout>);

    await act(async () => undefined);
    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:backend-unreachable', {
          detail: {
            method: 'GET',
            path: '/api/v1/llm/models/metadata',
            message: 'Failed to fetch',
            source: 'direct',
            timestamp: Date.now(),
          },
        })
      );
    });

    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();

    connectionState.value.phase = 'error';
    connectionState.value.isConnected = false;
    connectionState.value.lastCheckedAt = 101;
    connectionState.value.checksSinceConfigChange = 2;
    firstCheck.resolve();
    await act(async () => firstCheck.promise);
    rerender(<OptionLayout>{route}</OptionLayout>);

    expect(screen.getByTestId('backend-unavailable-modal')).toHaveAttribute(
      'data-presentation',
      'modal'
    );
    expect(screen.getByTestId('backend-unavailable-detail')).toHaveTextContent(
      '"method":"GET"'
    );
    expect(screen.getByTestId('backend-unavailable-detail')).toHaveTextContent(
      '"path":"/api/v1/llm/models/metadata"'
    );
    expect(screen.getByTestId('backend-unavailable-detail')).toHaveTextContent(
      '"message":"Failed to fetch"'
    );

    fireEvent.click(screen.getByRole('button', { name: 'Retry' }));
    expect(connectionState.value.checkOnce).toHaveBeenCalledTimes(2);
    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();

    connectionState.value.phase = 'connected';
    connectionState.value.isConnected = true;
    connectionState.value.lastCheckedAt = 102;
    connectionState.value.checksSinceConfigChange = 3;
    retryCheck.resolve();
    await act(async () => retryCheck.promise);
    rerender(<OptionLayout>{route}</OptionLayout>);

    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();
  });

  it('shows diagnostics when a forced check fails inside the connection grace window', async () => {
    const check = deferred();
    connectionState.value.checkOnce = vi.fn(() => check.promise);
    const route = <div data-testid="route-content">Research workspace route</div>;
    const { rerender } = render(<OptionLayout>{route}</OptionLayout>);

    await act(async () => undefined);
    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:backend-unreachable', {
          detail: {
            method: 'GET',
            path: '/api/v1/llm/models/metadata',
            message: 'Failed to fetch',
            source: 'direct',
            timestamp: Date.now(),
          },
        })
      );
    });

    connectionState.value.lastCheckedAt = 101;
    connectionState.value.checksSinceConfigChange = 2;
    connectionState.value.consecutiveFailures = 1;
    check.resolve();
    await act(async () => check.promise);
    rerender(<OptionLayout>{route}</OptionLayout>);

    expect(screen.getByTestId('backend-unavailable-modal')).toBeInTheDocument();
  });

  it('ignores an older forced check that settles after a newer recovery', async () => {
    const olderCheck = deferred();
    const newerCheck = deferred();
    connectionState.value.checkOnce = vi
      .fn()
      .mockImplementationOnce(() => olderCheck.promise)
      .mockImplementationOnce(() => newerCheck.promise);
    const route = <div data-testid="route-content">Research workspace route</div>;
    const { rerender } = render(<OptionLayout>{route}</OptionLayout>);

    await act(async () => undefined);
    const dispatchOutageCandidate = (path: string) => {
      window.dispatchEvent(
        new CustomEvent('tldw:backend-unreachable', {
          detail: {
            method: 'GET',
            path,
            message: 'Failed to fetch',
            source: 'direct',
            timestamp: Date.now(),
          },
        })
      );
    };

    act(() => {
      dispatchOutageCandidate('/api/v1/older-request');
      dispatchOutageCandidate('/api/v1/newer-request');
    });
    expect(connectionState.value.checkOnce).toHaveBeenCalledTimes(2);

    connectionState.value.lastCheckedAt = 101;
    connectionState.value.checksSinceConfigChange = 2;
    newerCheck.resolve();
    await act(async () => newerCheck.promise);
    rerender(<OptionLayout>{route}</OptionLayout>);
    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();

    connectionState.value.phase = 'error';
    connectionState.value.isConnected = false;
    connectionState.value.lastCheckedAt = 102;
    connectionState.value.checksSinceConfigChange = 3;
    olderCheck.resolve();
    await act(async () => olderCheck.promise);
    rerender(<OptionLayout>{route}</OptionLayout>);

    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();
  });

  it('uses the non-blocking backend-unreachable presentation on settings routes', async () => {
    routerState.location.pathname = '/settings/ui';
    const check = deferred();
    connectionState.value.checkOnce = vi.fn(() => check.promise);

    const { rerender } = render(
      <OptionLayout>
        <div data-testid="route-content">Settings UI route</div>
      </OptionLayout>
    );

    await act(async () => undefined);

    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:backend-unreachable', {
          detail: {
            method: 'GET',
            path: '/api/v1/llm/models/metadata',
            message: 'Failed to fetch',
            source: 'direct',
            timestamp: Date.now(),
          },
        })
      );
    });

    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();
    connectionState.value.phase = 'error';
    connectionState.value.isConnected = false;
    connectionState.value.lastCheckedAt = 101;
    connectionState.value.checksSinceConfigChange = 2;
    check.resolve();
    await act(async () => check.promise);
    rerender(
      <OptionLayout>
        <div data-testid="route-content">Settings UI route</div>
      </OptionLayout>
    );

    expect(screen.getByTestId('backend-unavailable-modal')).toHaveAttribute(
      'data-presentation',
      'inline'
    );
    const latestGateProps =
      backendUnavailableGateState.props[backendUnavailableGateState.props.length - 1];
    expect(latestGateProps).toEqual(
      expect.objectContaining({ presentation: 'inline' })
    );
  });

  it('does not treat settings-prefixed non-settings routes as settings routes', async () => {
    routerState.location.pathname = '/settings-wizard';
    const check = deferred();
    connectionState.value.checkOnce = vi.fn(() => check.promise);

    const { rerender } = render(
      <OptionLayout>
        <div data-testid="route-content">Settings wizard route</div>
      </OptionLayout>
    );

    await act(async () => undefined);

    act(() => {
      window.dispatchEvent(
        new CustomEvent('tldw:backend-unreachable', {
          detail: {
            method: 'GET',
            path: '/api/v1/llm/models/metadata',
            message: 'Failed to fetch',
            source: 'direct',
            timestamp: Date.now(),
          },
        })
      );
    });

    expect(screen.queryByTestId('backend-unavailable-modal')).toBeNull();
    connectionState.value.phase = 'error';
    connectionState.value.isConnected = false;
    connectionState.value.lastCheckedAt = 101;
    connectionState.value.checksSinceConfigChange = 2;
    check.resolve();
    await act(async () => check.promise);
    rerender(
      <OptionLayout>
        <div data-testid="route-content">Settings wizard route</div>
      </OptionLayout>
    );

    expect(screen.getByTestId('backend-unavailable-modal')).toHaveAttribute(
      'data-presentation',
      'modal'
    );
  });

  it('marks the /chat route shell as transcript-owned when sticky chat input is active', () => {
    const html = renderToStaticMarkup(
      <OptionLayout hideSidebar>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    expect(html).toContain('data-chat-scroll-owner="transcript"');
  });

  it('removes hidden-header padding from viewport-constrained chat routes', () => {
    const html = renderToStaticMarkup(
      <OptionLayout hideHeader hideSidebar>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    expect(html).toContain('data-chat-scroll-owner="transcript"');
    for (const className of [
      'items-stretch',
      'justify-start',
      'overflow-hidden',
      'px-0',
      'py-0',
    ]) {
      expect(html).toContain(className);
    }
    expect(html).not.toContain('px-4 py-10');
  });

  it('does not poll the notification count while the header is hidden', async () => {
    render(
      <OptionLayout hideHeader hideSidebar>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    await act(async () => undefined);

    expect(getUnreadCountMock).not.toHaveBeenCalled();
  });

  it('lets the notification provider own live production scope resolution', () => {
    const source = readFileSync(webLayoutSourcePath, 'utf8');

    expect(source).toContain(
      '<NotificationLifecycleProvider enabled={enabled && !demoEnabled}>'
    );
    expect(source).not.toContain('<NotificationLifecycleProvider scopeKey={scopeKey}');
  });

  it('passes openResetKey when the shared ChatSidebar feature is enabled', () => {
    featureFlagState.showChatSidebar = true;

    renderToStaticMarkup(
      <OptionLayout>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    expect(chatSidebarMockState.props).toHaveLength(1);
    expect(chatSidebarMockState.props[0]).toEqual(
      expect.objectContaining({
        collapsed: true,
        openResetKey: expect.any(Number),
      })
    );
  });

  it('closes the mobile drawer after an accepted conversation selection on the same route', () => {
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isMobile = true;
    render(<OptionLayout><div>Chat content</div></OptionLayout>);
    act(() => window.dispatchEvent(new CustomEvent('tldw:open-chat-sidebar')));
    expect(screen.getByTestId('drawer')).toBeInTheDocument();
    fireEvent.click(screen.getByRole('button', { name: 'Select saved conversation' }));
    expect(screen.queryByTestId('drawer')).not.toBeInTheDocument();
    expect(routerState.location.pathname).toBe('/chat');
  });

  it('keeps the desktop sidebar after selecting a conversation', () => {
    featureFlagState.showChatSidebar = true;
    render(<OptionLayout><div>Chat content</div></OptionLayout>);
    layoutUiState.value.setChatSidebarCollapsed.mockClear();
    fireEvent.click(screen.getByRole('button', { name: 'Select saved conversation' }));
    expect(screen.getByTestId('chat-sidebar')).toBeInTheDocument();
    expect(layoutUiState.value.setChatSidebarCollapsed).not.toHaveBeenCalledWith(false);
  });

  it('still closes an open mobile drawer on an ordinary route change', () => {
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isMobile = true;
    const view = render(<OptionLayout><div>Content</div></OptionLayout>);
    act(() => window.dispatchEvent(new CustomEvent('tldw:open-chat-sidebar')));
    expect(screen.getByTestId('drawer')).toBeInTheDocument();
    routerState.location.pathname = '/notes';
    view.rerender(<OptionLayout><div>Content</div></OptionLayout>);
    expect(screen.queryByTestId('drawer')).not.toBeInTheDocument();
  });

  it('labels and opens the legacy sidebar from the Drawer state', () => {
    featureFlagState.showChatSidebar = false;
    layoutUiState.value.chatSidebarCollapsed = false;

    render(
      <OptionLayout>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    const sidebarToggle = screen.getByTestId('header');
    expect(sidebarToggle).toHaveAttribute('aria-label', 'Expand sidebar');
    expect(screen.queryByTestId('drawer')).toBeNull();

    fireEvent.click(sidebarToggle);

    expect(screen.getByTestId('drawer')).toBeInTheDocument();
    expect(sidebarToggle).toHaveAttribute('aria-label', 'Collapse sidebar');
  });

  it('mirrors shared layout reset-key wiring for desktop and mobile mounts', () => {
    const source = readFileSync(webLayoutSourcePath, 'utf8');

    expect(source).toContain('chatSidebarOpenResetKey');
    expect(source.match(/openResetKey=\{chatSidebarOpenResetKey\}/g)).toHaveLength(2);
    expect(source).toContain('signalChatSidebarOpen');
    expect(source).toContain('setChatSidebarOpenResetKey((value) => value + 1)');
    expect(source).toContain('if (!sidebarOpen) signalChatSidebarOpen()');
    expect(source).toContain('if (chatSidebarCollapsed) signalChatSidebarOpen()');
    expect(source).toContain("window.addEventListener('tldw:open-chat-sidebar', handler)");
    expect(source).not.toContain("if (typeof window === 'undefined' || !showChatSidebar");
  });

  it('does not render a chat-specific collapsed rail edge button', () => {
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isDesktop = true;

    render(
      <OptionLayout>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    expect(screen.queryByTestId('chat-sidebar-edge-expand')).toBeNull();
  });

  it.each([false, true])(
    'fulfills an explicit Find request once the Web shell is revealed (shared=%s)',
    (shared) => {
      featureFlagState.showChatSidebar = shared;
      mediaQueryState.isMobile = true;
      const view = render(
        <OptionLayout hideHeader hideSidebar>
          <div>Draft</div>
        </OptionLayout>
      );
      act(() => {
        window.dispatchEvent(new Event('tldw:open-chat-sidebar'));
        window.dispatchEvent(new Event('tldw:open-chat-sidebar'));
      });
      expect(screen.queryByTestId('drawer')).toBeNull();
      view.rerender(
        <OptionLayout>
          <div>Draft</div>
        </OptionLayout>
      );
      expect(screen.getByTestId('drawer')).toBeInTheDocument();
      expect(screen.getByTestId('header')).toHaveAttribute('aria-label', 'Collapse sidebar');
      if (shared) expect(chatSidebarMockState.props.at(-1)?.openResetKey).toBe(1);
    }
  );

  it.each([false, true])(
    'abandons a hidden Web Find request on route departure (shared=%s)',
    (shared) => {
      featureFlagState.showChatSidebar = shared;
      mediaQueryState.isMobile = true;
      const view = render(
        <OptionLayout hideHeader hideSidebar>
          <div>Draft</div>
        </OptionLayout>
      );
      act(() => window.dispatchEvent(new Event('tldw:open-chat-sidebar')));
      routerState.location.pathname = '/notes';
      view.rerender(
        <OptionLayout hideHeader hideSidebar>
          <div>Notes</div>
        </OptionLayout>
      );
      routerState.location.pathname = '/chat';
      view.rerender(
        <OptionLayout>
          <div>Draft</div>
        </OptionLayout>
      );
      expect(screen.queryByTestId('drawer')).toBeNull();
      expect(screen.getByTestId('header')).toHaveAttribute('aria-label', 'Expand sidebar');
      expect(layoutUiState.value.setChatSidebarCollapsed).not.toHaveBeenCalledWith(false);
    }
  );

  it('abandons a hidden Web Find request when the route incarnation changes', () => {
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isMobile = true;
    const view = render(
      <OptionLayout hideHeader hideSidebar>
        <div>Draft</div>
      </OptionLayout>
    );
    act(() => window.dispatchEvent(new Event('tldw:open-chat-sidebar')));
    routerState.location.key = 'chat-replaced';
    view.rerender(
      <OptionLayout>
        <div>Draft</div>
      </OptionLayout>
    );
    expect(screen.queryByTestId('drawer')).toBeNull();
  });

  it('fulfills a pending Find through the current mobile host after viewport change', () => {
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isMobile = false;
    const view = render(
      <OptionLayout hideHeader hideSidebar>
        <div>Draft</div>
      </OptionLayout>
    );
    act(() => window.dispatchEvent(new Event('tldw:open-chat-sidebar')));
    mediaQueryState.isMobile = true;
    view.rerender(
      <OptionLayout>
        <div>Draft</div>
      </OptionLayout>
    );
    expect(screen.getByTestId('drawer')).toBeInTheDocument();
    expect(layoutUiState.value.setChatSidebarCollapsed).not.toHaveBeenCalledWith(false);
  });

  it('opens the available legacy sidebar for Find conversations events', async () => {
    featureFlagState.showChatSidebar = false;

    render(
      <OptionLayout>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    expect(screen.queryByTestId('drawer')).toBeNull();

    await act(async () => undefined);
    act(() => {
      window.dispatchEvent(new CustomEvent('tldw:open-chat-sidebar'));
    });

    expect(screen.getByTestId('drawer')).toBeInTheDocument();
    expect(screen.getByTestId('header')).toHaveAttribute('aria-label', 'Collapse sidebar');
    expect(chatSidebarMockState.props).toHaveLength(0);
  });

  it.each([false, true])(
    'keeps the root shell when the first-run gate wrapper changes (initially gated: %s)',
    (initiallyGated) => {
      featureFlagState.showChatSidebar = true;
      mediaQueryState.isMobile = false;
      function FirstRunGate({ children }: { children: React.ReactNode }) {
        return <>{children}</>;
      }
      const tree = (gated: boolean) => {
        routerState.location.pathname = gated ? '/' : '/setup';
        const layout = (
          <OptionLayout hideHeader={!gated} hideSidebar={!gated}>
            <div>{gated ? 'Companion Home' : 'Setup complete'}</div>
          </OptionLayout>
        );
        return gated ? <FirstRunGate>{layout}</FirstRunGate> : layout;
      };
      const view = render(tree(initiallyGated));
      expect(screen.getByRole('main')).toBeInTheDocument();

      view.rerender(tree(!initiallyGated));

      expect(screen.getByRole('main')).toContainElement(
        screen.getByText(initiallyGated ? 'Setup complete' : 'Companion Home'),
      );
      if (initiallyGated) {
        expect(screen.queryByTestId('header')).toBeNull();
        expect(screen.queryByTestId('chat-sidebar')).toBeNull();
      } else {
        expect(screen.getByTestId('header')).toBeInTheDocument();
        expect(screen.getByTestId('chat-sidebar')).toBeInTheDocument();
      }
    },
  );

  it('keeps one shell for a genuinely nested WebLayout and releases its hiding request', () => {
    routerState.location.pathname = '/';
    featureFlagState.showChatSidebar = true;
    mediaQueryState.isMobile = false;
    const tree = (hideShell: boolean) => (
      <OptionLayout>
        <OptionLayout hideHeader={hideShell} hideSidebar={hideShell}>
          <div>Nested route content</div>
        </OptionLayout>
      </OptionLayout>
    );
    const view = render(tree(false));
    expect(screen.getAllByRole('main')).toHaveLength(1);
    expect(screen.getAllByTestId('header')).toHaveLength(1);
    expect(screen.getAllByTestId('chat-sidebar')).toHaveLength(1);

    view.rerender(tree(true));
    expect(screen.getByRole('main')).toContainElement(
      screen.getByText('Nested route content'),
    );
    expect(screen.queryByTestId('header')).toBeNull();
    expect(screen.queryByTestId('chat-sidebar')).toBeNull();

    view.rerender(tree(false));
    expect(screen.getAllByRole('main')).toHaveLength(1);
    expect(screen.getAllByTestId('header')).toHaveLength(1);
    expect(screen.getAllByTestId('chat-sidebar')).toHaveLength(1);
  });

  it('mounts the shared tutorial runner in the web shell so page tour controls can render overlays', async () => {
    render(
      <OptionLayout>
        <div data-testid="route-content">Research workspace route</div>
      </OptionLayout>
    );

    expect(await screen.findByTestId('tutorial-runner')).toBeInTheDocument();
  });

  it('keeps the shared tutorial runner mounted when global chrome is hidden', async () => {
    render(
      <OptionLayout hideHeader hideSidebar>
        <div data-testid="route-content">No-key research workspace route</div>
      </OptionLayout>
    );

    expect(await screen.findByTestId('tutorial-runner')).toBeInTheDocument();
  });

  it.each([
    { pathname: '/chat', hiddenLayoutClass: 'items-stretch' },
    { pathname: '/settings', hiddenLayoutClass: 'items-center' },
  ])(
    'keeps route content mounted while it requests and releases shell hiding on $pathname',
    async ({ pathname, hiddenLayoutClass }) => {
      routerState.location.pathname = pathname;
      const mounted = vi.fn();
      const unmounted = vi.fn();

      function RouteContent({ hideShell }: { hideShell: boolean }) {
        useOptionLayoutShellOverrides(
          hideShell ? { hideHeader: true, hideSidebar: true } : null
        );

        React.useEffect(() => {
          mounted();
          return unmounted;
        }, []);

        return <div data-testid="route-content">Route content</div>;
      }

      const tree = (hideShell: boolean) => (
        <OptionLayout>
          <RouteContent hideShell={hideShell} />
        </OptionLayout>
      );
      const view = render(tree(false));

      await waitFor(() => expect(mounted).toHaveBeenCalledTimes(1));
      const routeContent = screen.getByTestId('route-content');
      const routeContainer = routeContent.parentElement;

      view.rerender(tree(true));

      await waitFor(() => {
        expect(screen.queryByTestId('header')).toBeNull();
        expect(routeContainer).toHaveClass(hiddenLayoutClass);
      });
      expect(screen.getByTestId('route-content')).toBe(routeContent);
      expect(screen.getByTestId('route-content').parentElement).toBe(routeContainer);
      expect(mounted).toHaveBeenCalledTimes(1);
      expect(unmounted).not.toHaveBeenCalled();

      view.rerender(tree(false));

      await waitFor(() => expect(screen.getByTestId('header')).toBeInTheDocument());
      expect(screen.getByTestId('route-content')).toBe(routeContent);
      expect(screen.getByTestId('route-content').parentElement).toBe(routeContainer);
      expect(mounted).toHaveBeenCalledTimes(1);
      expect(unmounted).not.toHaveBeenCalled();
    }
  );

  it('preserves non-overridden media query exports in the test mock', async () => {
    const mediaQueryModule = await import('@/hooks/useMediaQuery');

    expect(mediaQueryModule.useTablet).toEqual(expect.any(Function));
    expect(mediaQueryModule.useMediaQuery).toEqual(expect.any(Function));
  });
});

describe('WebLayout bypass block (#2889)', () => {
  beforeEach(() => {
    delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell;
    vi.clearAllMocks();
    routerState.location.pathname = '/admin/server';
    connectionState.value.phase = 'connected';
    connectionState.value.isConnected = true;
  });

  afterEach(() => {
    cleanup();
    delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell;
  });

  it('renders a skip link as the first focusable element, targeting the main region', () => {
    const view = render(<OptionLayout><div>Content</div></OptionLayout>);

    const firstFocusable = view.container.querySelector(
      "a[href], button, [tabindex]:not([tabindex='-1'])"
    ) as HTMLElement;
    expect(firstFocusable).toBeTruthy();
    expect(firstFocusable.tagName).toBe('A');
    expect(firstFocusable).toHaveTextContent('Skip to main content');
    expect(firstFocusable).toHaveAttribute('href', '#main-content');

    const main = view.container.querySelector('main');
    expect(main).toHaveAttribute('id', 'main-content');
    expect(main).toHaveAttribute('tabindex', '-1');
  });
});

it('shares one H1 controller across the WebUI shell and page only for the chat lifetime', () => {
  delete (globalThis as any).__tldwOptionShell;
  routerState.location.pathname = '/chat';
  featureFlagState.showChatSidebar = true;
  mediaQueryState.isMobile = false;
  const seen: any[] = [];
  let outer: any;
  function Probe() {
    const current = useHistorySelectionContext();
    seen.push(current?.getReference ?? null);
    return <output data-testid="web-controller">{current ? 'present' : 'absent'}</output>;
  }
  function Page() {
    const current = useHistorySelectionContext()?.getReference ?? null;
    React.useEffect(() => { outer = current; }, [current]);
    return current ? <HistorySelectionProvider><Probe /></HistorySelectionProvider> : <Probe />;
  }
  const view = render(<OptionLayout><Page /></OptionLayout>);
  expect(view.getByTestId('header')).toHaveAttribute('data-history-controller', 'present');
  expect(view.getByTestId('chat-sidebar')).toHaveAttribute('data-history-controller', 'present');
  expect(seen.at(-1)).toBe(outer);
  const first = outer;
  routerState.location.pathname = '/workspace';
  view.rerender(<OptionLayout><Page /></OptionLayout>);
  expect(view.getByTestId('header')).toHaveAttribute('data-history-controller', 'absent');
  expect(seen.at(-1)).toBeNull();
  routerState.location.pathname = '/chat';
  view.rerender(<OptionLayout><Page /></OptionLayout>);
  expect(seen.at(-1)).toBe(outer);
  expect(outer).not.toBe(first);
  cleanup();
  delete (globalThis as any).__tldwOptionShell;
});

vi.mock('@/services/chat-history-selection', () => ({
  captureHistorySnapshot: vi.fn(),
  confirmLegacyHistoryProjection: vi.fn(),
}));


const capturePublication = vi.hoisted(() => ({
  callback: null as
    | null
    | ((capture: import('@/types/history-selection').HistorySelectionCaptureV1) => void),
}));
vi.mock('@/hooks/chat/useHistorySelection', async (original) => {
  const actual = await original<typeof import('@/hooks/chat/useHistorySelection')>();
  return {
    ...actual,
    HistorySelectionProvider: (props: Parameters<typeof actual.HistorySelectionProvider>[0]) => {
      if (props.onCapture) capturePublication.callback = props.onCapture;
      return React.createElement(actual.HistorySelectionProvider, props);
    },
  };
});

it('the actual outer shell capture cannot republish old messages after synchronous New', async () => {
  const { useStoreMessageOption } =
    await vi.importActual<typeof import('@/store/option')>('@/store/option');
  useStoreMessageOption.setState({ history: [], messages: [], serverChatId: 'source-chat' });
  capturePublication.callback = null;
  routerState.location = { pathname: '/chat', search: '', hash: '' };
  render(
    <OptionLayout>
      <div>Active chat</div>
    </OptionLayout>
  );
  expect(capturePublication.callback).not.toBeNull();
  const row = {
    id: 'source-answer',
    revision: 'v1',
    parent_id: null,
    role: 'assistant',
    settled: true,
  };
  const capture: import('@/types/history-selection').HistorySelectionCaptureV1 = {
    status: 'captured',
    purpose: 'send',
    storage_context_digest: 'context',
    snapshot: {
      version: 1,
      owner_key: 'native-owner',
      conversation_id: 'source-chat',
      fences: { conversation: '1', history: '1', settings: '0' },
      nodes: [row],
      source_digest: 'source',
      interpretation_status: { kind: 'parent_graph_v1' },
      storage_context_digest: 'context',
    },
    view: {
      view_session_id: 'source-view',
      owner_key: 'native-owner',
      conversation_id: 'source-chat',
      interpretation: { kind: 'parent_graph_v1' },
      cursor: { kind: 'after_message', message_id: row.id },
      selection_revision: 1,
    },
    rows: [row],
    selected_content: [
      { id: row.id, revision: row.revision, message: 'Captured earlier reply', images: [] },
    ],
  };
  let replaced = false;
  const unsubscribe = useStoreMessageOption.subscribe((state) => {
    if (
      !replaced &&
      state.history.some((message) => message.content === 'Captured earlier reply')
    ) {
      replaced = true;
      state.startNewChatDraft();
      useStoreMessageOption.setState({ history: [], messages: [], serverChatId: null });
    }
  });
  try {
    act(() => capturePublication.callback!(capture));
    expect(replaced).toBe(true);
    expect(useStoreMessageOption.getState().chatDraftAddress?.kind).toBe('fresh');
    expect(useStoreMessageOption.getState().history).toEqual([]);
    expect(useStoreMessageOption.getState().messages).toEqual([]);
  } finally {
    unsubscribe();
  }
});


describe('WebLayout chat tour rendered outcome', () => {
  beforeEach(() => {
    connectionState.value.phase = 'connected';
    connectionState.value.isConnected = true;
    delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell;
    routerState.location = { pathname: '/chat', search: '', hash: '' };
    featureFlagState.showChatSidebar = false;
    mediaQueryState.isDesktop = false;
    mediaQueryState.isMobile = false;
    optionStoreState.value = { historyId: null, serverChatId: null };
    useTutorialStore.getState().resetProgress();
  });
  afterEach(() => {
    cleanup();
    useTutorialStore.getState().resetProgress();
    delete (globalThis as typeof globalThis & { __tldwOptionShell?: unknown }).__tldwOptionShell;
  });
  it.each([false, true])(
    'visible Tour opens one real help modal with hideHeader=%s and closes back to its trigger',
    async (hideHeader) => {
      render(
        <OptionLayout hideHeader={hideHeader}>
          <PlaygroundEmpty />
          <textarea aria-label="Retained draft" defaultValue={'Exact draft.  \nSecond line.'} />
        </OptionLayout>
      );
      const tour = screen.getByRole('button', { name: 'Take a quick tour' });
      expect(screen.queryByRole('dialog', { name: 'Page Help' })).not.toBeInTheDocument();
      tour.focus();
      fireEvent.click(tour);
      const dialog = await screen.findByRole('dialog', { name: 'Page Help' });
      expect(dialog).toBeVisible();
      expect(screen.getAllByRole('dialog', { name: 'Page Help' })).toHaveLength(1);
      for (const name of ['Chat Basics', 'Tools & Attachments', 'Voice Chat'])
        expect(within(dialog).getByRole('heading', { name })).toBeVisible();
      fireEvent.keyDown(dialog, { key: 'Escape' });
      expect(screen.queryByRole('dialog', { name: 'Page Help' })).not.toBeInTheDocument();
      expect(tour).toHaveFocus();
      expect(screen.getByRole('textbox', { name: 'Retained draft' })).toHaveValue(
        'Exact draft.  \nSecond line.'
      );
    }
  );
  it.each(['tldw:open-shortcuts-modal', 'tldw:open-playground-shortcuts'])(
    'opens one shortcut reference through %s before and after the help chunk is loaded',
    async (eventName) => {
      render(
        <OptionLayout>
          <PlaygroundEmpty />
          <button type="button">Help initiator</button>
          <textarea aria-label="Retained draft" defaultValue={'Exact draft.  \nSecond line.'} />
        </OptionLayout>
      );
      const initiator = screen.getByRole('button', { name: 'Help initiator' });
      for (const _attempt of [0, 1]) {
        initiator.focus();
        act(() => window.dispatchEvent(new CustomEvent(eventName)));
        const dialog = await screen.findByRole('dialog', { name: 'Page Help' });
        await waitFor(() => expect(screen.getAllByRole('dialog')).toHaveLength(1));
        expect(within(dialog).getByText('Show keyboard shortcuts')).toBeVisible();
        expect(within(dialog).getByText('Chat', { exact: true })).toBeVisible();
        expect(
          within(dialog).queryByRole('heading', { name: 'Chat Basics' })
        ).not.toBeInTheDocument();
        fireEvent.keyDown(dialog, { key: 'Escape' });
        expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
        expect(initiator).toHaveFocus();
      }
      fireEvent.click(screen.getByRole('button', { name: 'Take a quick tour' }));
      const tourDialog = await screen.findByRole('dialog', { name: 'Page Help' });
      expect(within(tourDialog).getByRole('heading', { name: 'Chat Basics' })).toBeVisible();
      expect(screen.getByRole('textbox', { name: 'Retained draft' })).toHaveValue(
        'Exact draft.  \nSecond line.'
      );
    }
  );
  it('opens the shortcut reference on ? while preserving question marks and modifier keys in editable fields', async () => {
    render(
      <OptionLayout>
        <PlaygroundEmpty />
        <button type="button">Help initiator</button>
        <textarea aria-label="Retained draft" defaultValue="Exact question?" />
      </OptionLayout>
    );
    const field = screen.getByRole('textbox', { name: 'Retained draft' });
    field.focus();
    fireEvent.keyDown(field, { key: '?', shiftKey: true });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    const initiator = screen.getByRole('button', { name: 'Help initiator' });
    initiator.focus();
    fireEvent.keyDown(initiator, { key: '?', ctrlKey: true, shiftKey: true });
    expect(screen.queryByRole('dialog')).not.toBeInTheDocument();
    fireEvent.keyDown(initiator, { key: '?', shiftKey: true });
    const dialog = await screen.findByRole('dialog', { name: 'Page Help' });
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
    expect(within(dialog).getByText('Show keyboard shortcuts')).toBeVisible();
    fireEvent.keyDown(dialog, { key: 'Escape' });
    expect(initiator).toHaveFocus();
    expect(field).toHaveValue('Exact question?');
  });
  it('leaves a page-local help owner alone after the persistent shell mounts its route', async () => {
    // Reproduce Flashcards' window-bubble listener in a declared route fixture.
    function LocalHelpPage() {
      const [open, setOpen] = React.useState(false);
      React.useEffect(() => {
        const onQuestion = (event: KeyboardEvent) => {
          if (event.key === '?' || (event.shiftKey && event.key === '/')) {
            event.preventDefault();
            setOpen(true);
          }
        };
        window.addEventListener('keydown', onQuestion);
        return () => window.removeEventListener('keydown', onQuestion);
      }, []);
      return (
        <>
          <button>Local help initiator</button>
          {open && (
            <div role="dialog" aria-modal="true" aria-label="Flashcards help">
              Local reference
            </div>
          )}
        </>
      );
    }
    routerState.location.pathname = '/media';
    const view = render(
      <OptionLayout>
        <button>Chat help initiator</button>
      </OptionLayout>
    );
    await act(async () => undefined);
    routerState.location.pathname = '/flashcards';
    view.rerender(
      <OptionLayout>
        <LocalHelpPage />
      </OptionLayout>
    );
    const localTrigger = screen.getByRole('button', { name: 'Local help initiator' });
    localTrigger.focus();
    fireEvent.keyDown(localTrigger, { key: '?', shiftKey: true });
    expect(useTutorialStore.getState().isHelpModalOpen).toBe(false);
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
    expect(screen.getByRole('dialog', { name: 'Flashcards help' })).toBeVisible();

    view.rerender(
      <OptionLayout>
        <button>Local help initiator</button>
      </OptionLayout>
    );
    act(() => window.dispatchEvent(new CustomEvent('tldw:open-help-modal')));
    const explicitHelp = await screen.findByRole('dialog', { name: 'Page Help' });
    fireEvent.keyDown(explicitHelp, { key: 'Escape' });
    routerState.location.pathname = '/chat';
    view.rerender(
      <OptionLayout>
        <button>Chat help initiator</button>
      </OptionLayout>
    );
    const chatTrigger = screen.getByRole('button', { name: 'Chat help initiator' });
    chatTrigger.focus();
    fireEvent.keyDown(chatTrigger, { key: '?', shiftKey: true });
    expect(await screen.findByRole('dialog', { name: 'Page Help' })).toBeVisible();
    expect(screen.getAllByRole('dialog')).toHaveLength(1);
  });
  it('starts the registered chat tutorial from its actual help control', async () => {
    render(
      <OptionLayout>
        <PlaygroundEmpty />
      </OptionLayout>
    );
    fireEvent.click(screen.getByRole('button', { name: 'Take a quick tour' }));
    const dialog = await screen.findByRole('dialog', { name: 'Page Help' });
    fireEvent.click(within(dialog).getByRole('button', { name: 'Start', exact: true }));
    expect(useTutorialStore.getState().activeTutorialId).toBe('playground-basics');
    expect(screen.queryByRole('dialog', { name: 'Page Help' })).not.toBeInTheDocument();
  });
});
