/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen, waitFor, within } from '@testing-library/react';
import ByokDashboardPage from '../page';
import { api } from '@/lib/api-client';

const promptPrivilegedActionMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/OrgContextSwitcher', () => ({
  useOrgContext: () => ({ selectedOrg: null }),
  OrgContextSwitcher: () => <div data-testid="org-switcher" />,
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => promptPrivilegedActionMock,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/api-client', () => ({
  ApiError: class MockApiError extends Error {
    status: number;

    constructor(status: number, message: string) {
      super(message);
      this.status = status;
    }
  },
  api: {
    getMetricsText: vi.fn(),
    getAuditLogs: vi.fn(),
    getSharedProviderKeys: vi.fn(),
    getUsersPage: vi.fn(),
    getAdminUserByokKeys: vi.fn(),
    getLlmUsage: vi.fn(),
    createSharedProviderKey: vi.fn(),
    deleteSharedProviderKey: vi.fn(),
    testSharedProviderKey: vi.fn(),
    getOpenAIOAuthStatus: vi.fn(),
    startOpenAIOAuth: vi.fn(),
    refreshOpenAIOAuth: vi.fn(),
    disconnectOpenAIOAuth: vi.fn(),
    switchOpenAICredentialSource: vi.fn(),
    createByokValidationRun: vi.fn(),
    getByokValidationRuns: vi.fn(),
    getByokValidationRun: vi.fn(),
  },
}));

type ApiMock = {
  getMetricsText: ReturnType<typeof vi.fn>;
  getAuditLogs: ReturnType<typeof vi.fn>;
  getSharedProviderKeys: ReturnType<typeof vi.fn>;
  getUsersPage: ReturnType<typeof vi.fn>;
  getAdminUserByokKeys: ReturnType<typeof vi.fn>;
  getLlmUsage: ReturnType<typeof vi.fn>;
  createSharedProviderKey: ReturnType<typeof vi.fn>;
  deleteSharedProviderKey: ReturnType<typeof vi.fn>;
  testSharedProviderKey: ReturnType<typeof vi.fn>;
  getOpenAIOAuthStatus: ReturnType<typeof vi.fn>;
  startOpenAIOAuth: ReturnType<typeof vi.fn>;
  refreshOpenAIOAuth: ReturnType<typeof vi.fn>;
  disconnectOpenAIOAuth: ReturnType<typeof vi.fn>;
  switchOpenAICredentialSource: ReturnType<typeof vi.fn>;
  createByokValidationRun: ReturnType<typeof vi.fn>;
  getByokValidationRuns: ReturnType<typeof vi.fn>;
  getByokValidationRun: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const completedValidationRun = {
  id: 'run-1',
  status: 'complete',
  org_id: null,
  provider: 'openai',
  keys_checked: 12,
  valid_count: 10,
  invalid_count: 1,
  error_count: 1,
  requested_by_user_id: 1,
  requested_by_label: 'alice',
  job_id: 'job-1',
  scope_summary: 'All orgs • provider=openai',
  error_message: null,
  created_at: '2026-03-12T12:00:00Z',
  started_at: '2026-03-12T12:00:05Z',
  completed_at: '2026-03-12T12:00:15Z',
} as const;

beforeEach(() => {
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getMetricsText.mockResolvedValue('');
  apiMock.getAuditLogs.mockResolvedValue({ entries: [], total: 0, limit: 200, offset: 0 });
  apiMock.getSharedProviderKeys.mockResolvedValue({ items: [] });
  apiMock.getOpenAIOAuthStatus.mockResolvedValue({
    provider: 'openai',
    connected: false,
    auth_source: 'none',
  });
  apiMock.startOpenAIOAuth.mockResolvedValue({
    provider: 'openai',
    auth_url: 'https://oauth.example.com/authorize',
    auth_session_id: 'session-123',
    expires_at: new Date().toISOString(),
  });
  apiMock.refreshOpenAIOAuth.mockResolvedValue({
    provider: 'openai',
    status: 'refreshed',
    updated_at: new Date().toISOString(),
  });
  apiMock.disconnectOpenAIOAuth.mockResolvedValue({});
  apiMock.switchOpenAICredentialSource.mockResolvedValue({
    provider: 'openai',
    auth_source: 'oauth',
    updated_at: new Date().toISOString(),
  });
  apiMock.createByokValidationRun.mockResolvedValue(completedValidationRun);
  apiMock.getByokValidationRuns.mockResolvedValue({
    items: [completedValidationRun],
    total: 1,
    limit: 20,
    offset: 0,
  });
  apiMock.getByokValidationRun.mockResolvedValue(completedValidationRun);

  apiMock.getUsersPage.mockResolvedValue({
    items: [
      { id: 1, username: 'alice' },
      { id: 2, username: 'bob' },
    ],
    total: 2,
    page: 1,
    pages: 1,
    limit: 100,
  });

  apiMock.getAdminUserByokKeys.mockImplementation(async (userId: string) => {
    if (userId === '1') {
      return {
        user_id: 1,
        items: [{ provider: 'openai', key_hint: 'sk-a1', allowed: true }],
      };
    }
    return {
      user_id: 2,
      items: [{ provider: 'anthropic', key_hint: 'sk-b2', allowed: true }],
    };
  });

  apiMock.getLlmUsage.mockResolvedValue({
    items: [
      { user_id: 1, provider: 'openai', total_tokens: 120, total_cost_usd: 0.24 },
      { user_id: 1, provider: 'openai', total_tokens: 80, total_cost_usd: 0.16 },
      { user_id: 2, provider: 'anthropic', total_tokens: 50, total_cost_usd: 0.5 },
      { user_id: 2, provider: 'openai', total_tokens: 999, total_cost_usd: 9.99 },
    ],
    total: 4,
    page: 1,
    limit: 500,
  });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('ByokDashboardPage', () => {
  it('renders per-user BYOK usage with backend-backed validation history', async () => {
    render(<ByokDashboardPage />);

    expect(await screen.findByText('Per-User BYOK Usage')).toBeInTheDocument();

    const aliceRowLabel = await screen.findByText('alice');
    const aliceRow = aliceRowLabel.closest('tr');
    expect(aliceRow).not.toBeNull();
    expect(within(aliceRow as HTMLElement).getByText('openai')).toBeInTheDocument();
    expect(within(aliceRow as HTMLElement).getByText('sk-a1')).toBeInTheDocument();
    expect(within(aliceRow as HTMLElement).getByText('2')).toBeInTheDocument();
    expect(within(aliceRow as HTMLElement).getByText('200')).toBeInTheDocument();
    expect(within(aliceRow as HTMLElement).getByText(/\$0\.4/)).toBeInTheDocument();

    const bobRowLabel = screen.getByText('bob');
    const bobRow = bobRowLabel.closest('tr');
    expect(bobRow).not.toBeNull();
    expect(within(bobRow as HTMLElement).getByText('anthropic')).toBeInTheDocument();
    expect(within(bobRow as HTMLElement).getByText('sk-b2')).toBeInTheDocument();
    expect(within(bobRow as HTMLElement).getByText('1')).toBeInTheDocument();
    expect(within(bobRow as HTMLElement).getByText('50')).toBeInTheDocument();
    expect(within(bobRow as HTMLElement).getByText(/\$0\.5/)).toBeInTheDocument();

    expect(await screen.findByRole('button', { name: /run validation sweep/i })).toBeInTheDocument();
    expect(screen.getByText('All orgs • provider=openai')).toBeInTheDocument();
    expect(screen.getByText('12 checked')).toBeInTheDocument();
    expect(screen.getByText('10 valid')).toBeInTheDocument();
    expect(screen.getByText('1 invalid')).toBeInTheDocument();
    expect(screen.getByText('1 errors')).toBeInTheDocument();
    expect(
      screen.queryByText('Validation sweep control is hidden until backend batch validation support is available.')
    ).not.toBeInTheDocument();
    expect(screen.queryByText(/Placeholder telemetry views/)).not.toBeInTheDocument();
    expect(screen.queryByText('Key Activity (Placeholder)')).not.toBeInTheDocument();
  });

  it('starts OpenAI OAuth connect flow from the BYOK card', async () => {
    const openSpy = vi.spyOn(window, 'open').mockReturnValue({} as Window);

    render(<ByokDashboardPage />);

    const connectButton = await screen.findByRole('button', { name: 'Connect OpenAI' });
    fireEvent.click(connectButton);

    await waitFor(() => {
      expect(apiMock.startOpenAIOAuth).toHaveBeenCalledTimes(1);
      expect(openSpy).toHaveBeenCalledWith(
        'https://oauth.example.com/authorize',
        '_blank',
        'noopener,noreferrer'
      );
    });
    openSpy.mockRestore();
  });

  it('creates a validation sweep and polls until terminal state', async () => {
    apiMock.getByokValidationRuns.mockResolvedValue({
      items: [],
      total: 0,
      limit: 20,
      offset: 0,
    });
    apiMock.createByokValidationRun.mockResolvedValue({
      ...completedValidationRun,
      id: 'run-2',
      status: 'queued',
      started_at: null,
      completed_at: null,
      keys_checked: null,
      valid_count: null,
      invalid_count: null,
      error_count: null,
      job_id: null,
    });
    apiMock.getByokValidationRun
      .mockResolvedValueOnce({
        ...completedValidationRun,
        id: 'run-2',
        status: 'running',
        started_at: '2026-03-12T12:00:10Z',
        completed_at: null,
        keys_checked: null,
        valid_count: null,
        invalid_count: null,
        error_count: null,
        job_id: 'job-2',
      })
      .mockResolvedValueOnce({
        ...completedValidationRun,
        id: 'run-2',
        job_id: 'job-2',
      });

    render(<ByokDashboardPage />);

    expect(screen.getByText('BYOK Dashboards')).toBeInTheDocument();
    const runSweepButton = await screen.findByRole('button', { name: /run validation sweep/i });
    await waitFor(() => expect(runSweepButton).not.toBeDisabled());
    fireEvent.click(runSweepButton);

    await waitFor(() => {
      expect(apiMock.createByokValidationRun).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(apiMock.getByokValidationRun).toHaveBeenCalledWith('run-2');
      expect(screen.getByText('Complete')).toBeInTheDocument();
      expect(screen.getByText('12 checked')).toBeInTheDocument();
      expect(screen.getByText('10 valid')).toBeInTheDocument();
      expect(screen.getByText('1 invalid')).toBeInTheDocument();
      expect(screen.getByText('1 errors')).toBeInTheDocument();
      expect(toastSuccessMock).toHaveBeenCalled();
    }, { timeout: 5000 });
  }, 10000);

  it('deletes a shared key through privileged action dialog', async () => {
    promptPrivilegedActionMock.mockResolvedValue({ reason: 'test' });
    apiMock.deleteSharedProviderKey.mockResolvedValue({});
    apiMock.getSharedProviderKeys.mockResolvedValue({
      items: [
        {
          scope_type: 'global',
          scope_id: '*',
          provider: 'openai',
          key_hint: 'sk-abc...xyz',
          created_at: '2026-01-01T00:00:00Z',
        },
      ],
    });

    render(<ByokDashboardPage />);

    const deleteButton = await screen.findByRole('button', { name: /delete/i });
    fireEvent.click(deleteButton);

    await waitFor(() => {
      expect(promptPrivilegedActionMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Delete Shared Key' }),
      );
    });

    await waitFor(() => {
      expect(apiMock.deleteSharedProviderKey).toHaveBeenCalledWith('global', '*', 'openai');
    });
  });

  it('does not create fake validation history when create fails', async () => {
    apiMock.getByokValidationRuns.mockResolvedValue({
      items: [],
      total: 0,
      limit: 20,
      offset: 0,
    });
    apiMock.createByokValidationRun.mockRejectedValue(new Error('sweep failed'));

    render(<ByokDashboardPage />);

    expect(screen.getByText('BYOK Dashboards')).toBeInTheDocument();
    const runSweepButton = await screen.findByRole('button', { name: /run validation sweep/i });
    await waitFor(() => expect(runSweepButton).not.toBeDisabled());
    fireEvent.click(runSweepButton);

    await waitFor(() => {
      expect(toastErrorMock).toHaveBeenCalledWith('Validation sweep failed', 'sweep failed');
    });
    expect(screen.getByText('No validation runs yet.')).toBeInTheDocument();
    expect(toastSuccessMock).not.toHaveBeenCalled();
  });
});
