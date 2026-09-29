/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import SecurityPage from '../page';
import { api } from '@/lib/api-client';

const pushMock = vi.hoisted(() => vi.fn());

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: pushMock,
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getSecurityHealth: vi.fn(),
    getSecurityAlertStatus: vi.fn(),
    getUsersPage: vi.fn(),
    getUserApiKeys: vi.fn(),
  },
}));

type ApiMock = {
  getSecurityHealth: ReturnType<typeof vi.fn>;
  getSecurityAlertStatus: ReturnType<typeof vi.fn>;
  getUsersPage: ReturnType<typeof vi.fn>;
  getUserApiKeys: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  vi.useFakeTimers({ toFake: ['Date'] });
  vi.setSystemTime(new Date('2026-02-17T12:00:00Z'));
  pushMock.mockReset();

  apiMock.getSecurityHealth.mockResolvedValue({
    risk_score: 58,
    recent_security_events: 6,
    failed_logins_24h: 12,
    suspicious_activity: 2,
    mfa_adoption_rate: 50,
    active_sessions: 11,
    api_keys_active: 8,
    last_security_scan: '2026-02-17T00:00:00Z',
  });
  apiMock.getSecurityAlertStatus.mockResolvedValue({
    total_alerts: 3,
    critical_alerts: 1,
    warning_alerts: 2,
    unacknowledged: 2,
    recent_alerts: [],
  });
  apiMock.getUsersPage.mockResolvedValue({
    items: [
      { id: 1, username: 'alice' },
      { id: 2, username: 'bob' },
    ],
    total: 2,
    page: 1,
    limit: 100,
    pages: 1,
  });
  apiMock.getUserApiKeys.mockImplementation(async (userId: string) => {
    if (userId === '1') {
      return [
        {
          id: 'k-old',
          status: 'active',
          created_at: '2025-01-01T00:00:00Z',
          expires_at: null,
        },
      ];
    }
    return [
      {
        id: 'k-new',
        status: 'active',
        created_at: '2026-02-01T00:00:00Z',
        expires_at: null,
      },
    ];
  });
});

afterEach(() => {
  cleanup();
  vi.useRealTimers();
  vi.resetAllMocks();
});

describe('SecurityPage', () => {
  it('renders risk factor breakdown with weighted calculation', async () => {
    render(<SecurityPage />);

    expect(await screen.findByText('Risk factor breakdown')).toBeInTheDocument();
    expect(await screen.findByText('Users without MFA')).toBeInTheDocument();
    expect(await screen.findByText('API keys older than 180 days')).toBeInTheDocument();
    expect(await screen.findByText('Failed logins (24h)')).toBeInTheDocument();
    expect(await screen.findByText('Suspicious activity (24h)')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('Calculation: 3 + 2 + 12 + 8 = 25')).toBeInTheDocument();
    });
    expect(screen.getByTestId('risk-breakdown-estimated-score').textContent).toContain('Estimated 25/100');
  });

  it('shows remediation links for risk factors', async () => {
    render(<SecurityPage />);

    const mfaLink = await screen.findByRole('link', { name: 'Review MFA-disabled users' });
    const keysLink = await screen.findByRole('link', { name: 'Review active API keys' });
    const failedLoginLink = await screen.findByRole('link', { name: 'Inspect failed login events' });

    expect(mfaLink.getAttribute('href')).toBe('/users?mfa=disabled');
    expect(keysLink.getAttribute('href')).toBe('/api-keys?status=active');
    expect(failedLoginLink.getAttribute('href')).toBe('/audit?action=login.failed');
  });

  it('fails closed when security health is unavailable instead of rendering a zero-risk posture', async () => {
    apiMock.getSecurityHealth.mockRejectedValue(new Error('health unavailable'));

    render(<SecurityPage />);

    expect(await screen.findByText('Security health data is unavailable. Risk score and summary metrics cannot be calculated.')).toBeInTheDocument();
    expect(screen.getByTestId('security-risk-unavailable').textContent).toContain('Risk score unavailable.');
    expect(screen.queryByLabelText(/Security risk score:/)).not.toBeInTheDocument();
    expect(screen.queryByText('Low Risk')).not.toBeInTheDocument();
  });
});
