import React from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { beforeEach, describe, expect, it, vi } from 'vitest';

import OptionLayout from '../../../components/layout/WebLayout';

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
  },
}));

const confirmDangerMock = vi.hoisted(() => vi.fn(async () => false));
const storageState = vi.hoisted(() => ({
  stickyChatInput: true,
}));

vi.mock('antd', () => ({
  Drawer: ({ children, open }: { children: React.ReactNode; open?: boolean }) =>
    open ? <div data-testid="drawer">{children}</div> : null,
  Tooltip: ({ children }: { children: React.ReactNode }) => <>{children}</>,
}));

vi.mock('lucide-react', () => ({
  EraserIcon: () => null,
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
  useChatShortcuts: () => undefined,
  useSidebarShortcuts: () => undefined,
  useQuickChatShortcuts: () => undefined,
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
  Header: () => <div data-testid="header" />,
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
  useChatSidebar: () => [false],
}));

vi.mock('@/hooks/useMediaQuery', () => ({
  useMobile: () => false,
}));

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
  ChatSidebar: () => null,
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
  BackendUnavailableModalGate: () => null,
}));

vi.mock('@web/lib/api/notifications', () => ({
  getUnreadCount: vi.fn(async () => ({ unread_count: 0 })),
}));

vi.mock('@/components/Common/CommandPalette', () => ({
  CommandPalette: () => null,
}));

vi.mock('@/hooks/useConnectionState', () => ({
  useConnectionActions: () => ({ checkOnce: connectionState.value.checkOnce }),
  useConnectionState: () => ({
    phase: connectionState.value.phase,
    isConnected: connectionState.value.isConnected,
  }),
  useConnectionUxState: () => ({
    isChecking: connectionState.value.isChecking,
  }),
}));

vi.mock('@/types/connection', () => ({
  ConnectionPhase: {
    CONNECTED: 'connected',
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
    vi.clearAllMocks();
    routerState.location.pathname = '/chat';
    routerState.location.search = '';
    routerState.location.hash = '';
    storageState.stickyChatInput = true;
  });

  it('marks the /chat route shell as transcript-owned when sticky chat input is active', () => {
    const html = renderToStaticMarkup(
      <OptionLayout hideSidebar>
        <div data-testid="chat-route-content">Chat route</div>
      </OptionLayout>
    );

    expect(html).toContain('data-chat-scroll-owner="transcript"');
  });
});
