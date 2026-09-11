// @vitest-environment jsdom
import React from 'react';
import { readFileSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { renderToStaticMarkup } from 'react-dom/server';
import { act, cleanup, fireEvent, render, screen, waitFor } from '@testing-library/react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { useOptionLayoutShellOverrides } from '@/components/Layouts/Layout';
import OptionLayout from '../../../components/layout/WebLayout';

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

const quickChatState = vi.hoisted(() => ({
  value: {
    isOpen: false,
    setIsOpen: vi.fn(),
  },
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

vi.mock('antd', () => ({
  Drawer: ({ children, open }: { children: React.ReactNode; open?: boolean }) =>
    open ? <div data-testid="drawer">{children}</div> : null,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('lucide-react', () => ({
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

vi.mock('@/hooks/keyboard/useKeyboardShortcuts', () => ({
  isMac: false,
  useChatShortcuts: () => undefined,
  useSidebarShortcuts: () => undefined,
  useQuickChatShortcuts: () => undefined,
  useModeNavigationShortcuts: () => undefined,
}));

vi.mock('@/store/quick-chat', () => ({
  useQuickChatStore: () => quickChatState.value,
}));

vi.mock('@/store/option', () => ({
  useStoreMessageOption: (selector: (state: typeof optionStoreState.value) => unknown) =>
    selector(optionStoreState.value),
}));

vi.mock('@/store/layout-ui', () => ({
  useLayoutUiStore: (selector: (state: typeof layoutUiState.value) => unknown) =>
    selector(layoutUiState.value),
}));

vi.mock('@/store/route-transition', () => ({
  useRouteTransitionStore: (selector: (state: typeof routeTransitionState.value) => unknown) =>
    selector(routeTransitionState.value),
}));

vi.mock('@/components/Common/QuickChatHelper', () => ({
  QuickChatHelperButton: () => <div data-testid="quick-chat-helper" />,
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
    return <aside data-testid="chat-sidebar" />;
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

vi.mock('@/routes/route-paths', () => ({
  VIEWPORT_CONSTRAINED_PATHS: ['/chat'],
}));

vi.mock('@/services/settings/ui-settings', () => ({
  CHAT_BACKGROUND_IMAGE_SETTING: 'chatBackgroundImage',
  HEADER_SHORTCUTS_EXPANDED_SETTING: 'headerShortcutsExpanded',
}));

vi.mock('@/services/settings/registry', () => ({
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
}));

vi.mock('@web/components/ui/ToastProvider', () => ({
  useToast: () => ({ show: vi.fn() }),
}));

vi.mock('@/components/Common/CommandPalette', () => ({
  CommandPalette: () => null,
}));

vi.mock('@/components/Common/TutorialRunner', () => ({
  TutorialRunner: () => <div data-testid="tutorial-runner" />,
}));

vi.mock('@/components/Common/KeyboardShortcutsModal', () => ({
  KeyboardShortcutsModal: () => null,
}));

vi.mock('@/hooks/useConnectionState', () => ({
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
    routerState.location.pathname = '/chat';
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
    expect(source).toContain("if (typeof window === 'undefined' || !showChatSidebar) return;");
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

  it('ignores chat sidebar open events when the shared ChatSidebar feature is disabled', async () => {
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

    expect(screen.queryByTestId('drawer')).toBeNull();
    expect(chatSidebarMockState.props).toHaveLength(0);
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
