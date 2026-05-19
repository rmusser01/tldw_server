/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import AIOpsPage from '../page';
import { api } from '@/lib/api-client';

const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
  usePermissions: () => ({
    user: { id: 1, uuid: 'u-1', username: 'Admin', role: 'admin' },
    permissions: [],
    permissionHints: [],
    roles: ['admin'],
    loading: false,
    hasPermission: () => true,
    hasRole: () => true,
    hasAnyPermission: () => true,
    hasAllPermissions: () => true,
    isAdmin: () => true,
    isSuperAdmin: () => false,
    refresh: async () => {},
  }),
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('next/navigation', () => ({
  useRouter: () => ({ push: vi.fn(), replace: vi.fn(), prefetch: vi.fn() }),
  usePathname: () => '/ai-ops',
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getACPAgentMetrics: vi.fn(),
    getRealtimeStats: vi.fn(),
    getACPSessions: vi.fn(),
  },
}));

const apiMock = vi.mocked(api);

const sampleAgentMetrics = {
  items: [
    {
      agent_type: 'research-assistant',
      session_count: 15,
      active_sessions: 3,
      total_prompt_tokens: 500000,
      total_completion_tokens: 250000,
      total_tokens: 750000,
      total_messages: 120,
      last_used_at: '2025-06-01T12:00:00Z',
      total_estimated_cost_usd: 12.5,
    },
    {
      agent_type: 'code-reviewer',
      session_count: 8,
      active_sessions: 0,
      total_prompt_tokens: 200000,
      total_completion_tokens: 100000,
      total_tokens: 300000,
      total_messages: 45,
      last_used_at: '2025-05-28T08:00:00Z',
      total_estimated_cost_usd: 4.25,
    },
  ],
};

const sampleRealtimeStats = {
  active_sessions: 5,
  tokens_today: {
    prompt: 10000,
    completion: 5000,
    total: 15000,
  },
};

const sampleSessions = {
  sessions: [
    {
      session_id: 'sess-abc123def456',
      user_id: 1,
      agent_type: 'research-assistant',
      name: 'Research Session',
      status: 'active',
      created_at: '2025-06-01T10:00:00Z',
      last_activity_at: '2025-06-01T11:30:00Z',
      message_count: 12,
      usage: {
        prompt_tokens: 5000,
        completion_tokens: 3000,
        total_tokens: 8000,
      },
      estimated_cost_usd: 0.45,
      model: 'gpt-4',
    },
    {
      session_id: 'sess-xyz789ghi012',
      user_id: 2,
      agent_type: 'code-reviewer',
      name: 'Code Review',
      status: 'closed',
      created_at: '2025-05-30T14:00:00Z',
      last_activity_at: '2025-05-30T15:00:00Z',
      message_count: 8,
      usage: {
        prompt_tokens: 2000,
        completion_tokens: 1000,
        total_tokens: 3000,
      },
      estimated_cost_usd: 0.12,
      model: 'gpt-3.5-turbo',
    },
  ],
  total: 2,
};

beforeEach(() => {
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getACPAgentMetrics.mockResolvedValue(sampleAgentMetrics);
  apiMock.getRealtimeStats.mockResolvedValue(sampleRealtimeStats);
  apiMock.getACPSessions.mockResolvedValue(sampleSessions);
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('AIOpsPage', () => {
  it('renders KPI cards with computed values', async () => {
    render(<AIOpsPage />);

    // Total AI Spend = 12.5 + 4.25 = 16.75
    expect(await screen.findByText('$16.75')).toBeInTheDocument();

    // Active Sessions from realtime stats = 5
    expect(screen.getByText('5')).toBeInTheDocument();

    // Total Tokens = 750000 + 300000 = 1050000 => "1.1M"
    expect(screen.getByText('1.1M')).toBeInTheDocument();

    // Active Agents: research-assistant has 3 active sessions, code-reviewer has 0 => 1
    expect(screen.getByText('Total AI Spend')).toBeInTheDocument();
    expect(screen.getByText('Active Sessions')).toBeInTheDocument();
    expect(screen.getByText('Total Tokens')).toBeInTheDocument();
    expect(screen.getByText('Active Agents')).toBeInTheDocument();
  });

  it('renders top agents table', async () => {
    render(<AIOpsPage />);

    expect(await screen.findByText('Top Agents by Cost')).toBeInTheDocument();
    // research-assistant appears in both agents and sessions tables
    const agentBadges = screen.getAllByText('research-assistant');
    expect(agentBadges.length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('code-reviewer').length).toBeGreaterThanOrEqual(1);
    // research-assistant cost: $12.50
    expect(screen.getByText('$12.50')).toBeInTheDocument();
    // code-reviewer cost: $4.25
    expect(screen.getByText('$4.25')).toBeInTheDocument();
  });

  it('renders recent sessions table', async () => {
    render(<AIOpsPage />);

    expect(await screen.findByText('Recent Sessions')).toBeInTheDocument();
    expect(screen.getByText('Research Session')).toBeInTheDocument();
    expect(screen.getByText('Code Review')).toBeInTheDocument();
    // Session IDs truncated
    expect(screen.getByText('sess-abc123d...')).toBeInTheDocument();
    expect(screen.getByText('sess-xyz789g...')).toBeInTheDocument();
  });

  it('handles API errors gracefully by showing empty/zero values', async () => {
    apiMock.getACPAgentMetrics.mockRejectedValue(new Error('Server error'));
    apiMock.getRealtimeStats.mockRejectedValue(new Error('Server error'));
    apiMock.getACPSessions.mockRejectedValue(new Error('Server error'));

    render(<AIOpsPage />);

    // The page uses Promise.allSettled, so individual failures are handled
    // without throwing. It should still render with empty/zero data.
    await waitFor(() => {
      expect(screen.queryByText('Loading agent metrics...')).not.toBeInTheDocument();
    });

    // Empty states show up for the tables
    expect(screen.getByText('No agent metrics available')).toBeInTheDocument();
    expect(screen.getByText('No recent sessions')).toBeInTheDocument();
  });

  it('shows loading state initially', async () => {
    // Make API calls hang
    apiMock.getACPAgentMetrics.mockReturnValue(new Promise(() => {}));
    apiMock.getRealtimeStats.mockReturnValue(new Promise(() => {}));
    apiMock.getACPSessions.mockReturnValue(new Promise(() => {}));

    render(<AIOpsPage />);

    expect(screen.getByText('Loading agent metrics...')).toBeInTheDocument();
    expect(screen.getByText('Loading sessions...')).toBeInTheDocument();
  });

  it('shows empty state when no agents or sessions exist', async () => {
    apiMock.getACPAgentMetrics.mockResolvedValue({ items: [] });
    apiMock.getRealtimeStats.mockResolvedValue(sampleRealtimeStats);
    apiMock.getACPSessions.mockResolvedValue({ sessions: [], total: 0 });

    render(<AIOpsPage />);

    expect(await screen.findByText('No agent metrics available')).toBeInTheDocument();
    expect(screen.getByText('No recent sessions')).toBeInTheDocument();
  });

  it('displays tokens today in the Active Sessions KPI subtitle', async () => {
    render(<AIOpsPage />);

    // 15000 tokens => "15.0K tokens today"
    expect(await screen.findByText('15.0K tokens today')).toBeInTheDocument();
  });

  it('displays session status badges', async () => {
    render(<AIOpsPage />);

    await screen.findByText('Recent Sessions');

    // The "Active" badge exists (from status badge and/or KPI labels)
    const activeBadges = screen.getAllByText('Active');
    expect(activeBadges.length).toBeGreaterThanOrEqual(1);
    expect(screen.getByText('Closed')).toBeInTheDocument();
  });
});
