/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ResourceGovernorPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => confirmMock,
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => vi.fn().mockResolvedValue(true),
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getResourceGovernorPolicy: vi.fn(),
    simulateResourceGovernorPolicy: vi.fn(),
    updateResourceGovernorPolicy: vi.fn(),
    deleteResourceGovernorPolicy: vi.fn(),
    getUsers: vi.fn(),
    getUsersPage: vi.fn(),
    getUserOrgMemberships: vi.fn(),
    getLlmUsage: vi.fn(),
    getRateLimitEvents: vi.fn(),
    getRateLimitSummary: vi.fn(),
    getMetricsText: vi.fn(),
  },
}));

type ApiMock = {
  getResourceGovernorPolicy: ReturnType<typeof vi.fn>;
  simulateResourceGovernorPolicy: ReturnType<typeof vi.fn>;
  updateResourceGovernorPolicy: ReturnType<typeof vi.fn>;
  deleteResourceGovernorPolicy: ReturnType<typeof vi.fn>;
  getUsers: ReturnType<typeof vi.fn>;
  getUsersPage: ReturnType<typeof vi.fn>;
  getUserOrgMemberships: ReturnType<typeof vi.fn>;
  getLlmUsage: ReturnType<typeof vi.fn>;
  getRateLimitEvents: ReturnType<typeof vi.fn>;
  getRateLimitSummary: ReturnType<typeof vi.fn>;
  getMetricsText: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  confirmMock.mockResolvedValue({ reason: 'test audit reason', adminPassword: '' });
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();
  apiMock.getResourceGovernorPolicy.mockResolvedValue({ policies: [] });
  apiMock.simulateResourceGovernorPolicy.mockRejectedValue(new Error('simulate unavailable'));
  apiMock.updateResourceGovernorPolicy.mockResolvedValue({});
  apiMock.getUsers.mockResolvedValue([
    { id: 42, username: 'alice' },
    { id: 7, username: 'bob' },
  ]);
  apiMock.getUsersPage.mockResolvedValue({
    items: [
      { id: 42, username: 'alice', role: 'member' },
      { id: 7, username: 'bob', role: 'admin' },
    ],
    total: 2,
    page: 1,
    pages: 1,
    limit: 100,
  });
  apiMock.getUserOrgMemberships.mockImplementation(async (userId: string) => {
    if (userId === '42') {
      return [{ org_id: 99 }];
    }
    return [{ org_id: 1 }];
  });
  apiMock.getLlmUsage.mockResolvedValue({
    items: [{ user_id: 42 }, { user_id: 42 }, { user_id: 7 }],
    total: 3,
    page: 1,
    limit: 500,
  });
  apiMock.getRateLimitEvents.mockResolvedValue({
    items: [
      {
        user_id: 42,
        policy_id: 'chat.default',
        rejections_24h: 5,
        last_rejection_at: '2026-02-17T10:00:00Z',
      },
    ],
  });
  apiMock.getRateLimitSummary.mockResolvedValue({
    total_throttle_events: 5,
    period: '24h',
    top_throttled_entities: [],
    policy_headroom: [],
  });
  apiMock.getMetricsText.mockResolvedValue('');
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('ResourceGovernor policy form', () => {
  it('validates scope ID with accessible field error messaging', async () => {
    const user = userEvent.setup();
    render(<ResourceGovernorPage />);

    await user.click(screen.getByRole('button', { name: 'New Policy' }));
    await user.type(screen.getByRole('textbox', { name: /policy name/i }), 'Org guardrail');
    await user.selectOptions(screen.getAllByRole('combobox', { name: /scope/i })[0], 'org');
    await user.click(screen.getByRole('button', { name: 'Create Policy' }));

    const scopeIdInput = screen.getByRole('textbox', { name: /organization id/i });
    const errorText = await screen.findByText('Organization ID is required for org scope');

    expect(errorText).toBeInTheDocument();
    expect(scopeIdInput.getAttribute('aria-invalid')).toBe('true');
    expect(scopeIdInput.getAttribute('aria-describedby')).toBe('scope_id-error');
    expect(errorText.closest('[role=\"alert\"]')?.getAttribute('id')).toBe('scope_id-error');
    expect(apiMock.updateResourceGovernorPolicy).not.toHaveBeenCalled();
  });

  it('submits valid payload for a new policy', async () => {
    const user = userEvent.setup();
    render(<ResourceGovernorPage />);

    await user.click(screen.getByRole('button', { name: 'New Policy' }));
    await user.type(screen.getByRole('textbox', { name: /policy name/i }), 'Org guardrail');
    await user.selectOptions(screen.getAllByRole('combobox', { name: /scope/i })[0], 'org');
    await user.type(screen.getByRole('textbox', { name: /organization id/i }), '42');
    await user.type(screen.getByRole('spinbutton', { name: /max requests\/min/i }), '60');
    await user.click(screen.getByRole('button', { name: 'Create Policy' }));

    await waitFor(() => {
      expect(apiMock.updateResourceGovernorPolicy).toHaveBeenCalledWith(
        expect.objectContaining({
          name: 'Org guardrail',
          scope: 'org',
          scope_id: '42',
          resource_type: 'llm',
          max_requests_per_minute: 60,
          enabled: true,
        })
      );
    });
  });

  it('shows simulation result message with affected users and requests', async () => {
    const user = userEvent.setup();
    apiMock.simulateResourceGovernorPolicy.mockResolvedValueOnce({
      affected_users: 3,
      affected_requests_24h: 21,
    });
    render(<ResourceGovernorPage />);

    await user.click(screen.getByRole('button', { name: 'New Policy' }));
    await user.type(screen.getByRole('textbox', { name: /policy name/i }), 'Global guardrail');
    await user.click(screen.getByRole('button', { name: 'Simulate Impact' }));

    expect(
      await screen.findByText(/Would affect 3 users \/ 21 requests in last 24h\./)
    ).toBeInTheDocument();
  });

  it('renders policy resolution chain and winner explanation', async () => {
    const user = userEvent.setup();
    apiMock.getResourceGovernorPolicy.mockResolvedValueOnce({
      policies: [
        {
          id: 'p-global',
          name: 'Default LLM',
          scope: 'global',
          resource_type: 'llm',
          priority: 1,
          enabled: true,
        },
        {
          id: 'p-user',
          name: 'Power User',
          scope: 'user',
          scope_id: '42',
          resource_type: 'llm',
          priority: 10,
          enabled: true,
        },
      ],
    });
    render(<ResourceGovernorPage />);

    await waitFor(() => {
      expect(screen.queryByText('Loading scope context...')).not.toBeInTheDocument();
    });
    await user.selectOptions(screen.getByLabelText('User'), '42');
    await user.click(screen.getByRole('button', { name: 'Resolve Policy' }));

    expect(
      await screen.findByText(/Global policy "Default LLM".*Winner: Power User/i)
    ).toBeInTheDocument();
    expect(await screen.findByText('Reason')).toBeInTheDocument();
    expect(await screen.findByText('Winner')).toBeInTheDocument();
  });

  it('shows affected users count for each policy row', async () => {
    apiMock.getResourceGovernorPolicy.mockResolvedValueOnce({
      policies: [
        {
          id: 'p-global',
          name: 'Global Policy',
          scope: 'global',
          resource_type: 'llm',
          enabled: true,
        },
        {
          id: 'p-org',
          name: 'Org 99 Policy',
          scope: 'org',
          scope_id: '99',
          resource_type: 'llm',
          enabled: true,
        },
        {
          id: 'p-user',
          name: 'User 42 Policy',
          scope: 'user',
          scope_id: '42',
          resource_type: 'llm',
          enabled: true,
        },
      ],
    });
    render(<ResourceGovernorPage />);

    const globalRow = (await screen.findByText('Global Policy')).closest('tr');
    const orgRow = (await screen.findByText('Org 99 Policy')).closest('tr');
    const userRow = (await screen.findByText('User 42 Policy')).closest('tr');

    expect(globalRow).not.toBeNull();
    expect(orgRow).not.toBeNull();
    expect(userRow).not.toBeNull();

    expect(within(globalRow as HTMLElement).getByText(/^2$/)).toBeInTheDocument();
    expect(within(orgRow as HTMLElement).getByText(/^1$/)).toBeInTheDocument();
    expect(within(userRow as HTMLElement).getByText(/^1$/)).toBeInTheDocument();
  });

  it('renders rate limit events table rows', async () => {
    render(<ResourceGovernorPage />);

    expect(await screen.findByText('Rate Limit Events')).toBeInTheDocument();
    const row = (await screen.findByText('User 42')).closest('tr');
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getByText('chat.default')).toBeInTheDocument();
    expect(within(row as HTMLElement).getByText(/^5$/)).toBeInTheDocument();
  });

  it('renders rate limit analytics section with throttle count badge', async () => {
    apiMock.getRateLimitSummary.mockResolvedValueOnce({
      total_throttle_events: 12,
      period: '24h',
      top_throttled_entities: [
        { entity: 'user:42', rejections: 8, last_rejected_at: '2026-02-17T11:00:00Z' },
      ],
      policy_headroom: [
        { policy_id: 'chat.default', resource_type: 'llm', utilization_pct: 85, total_denials: 12, total_decisions: 100 },
      ],
    });
    apiMock.getRateLimitEvents.mockResolvedValueOnce({
      items: [
        {
          user_id: 42,
          policy_id: 'chat.default',
          rejections_24h: 8,
          last_rejection_at: '2026-02-17T11:00:00Z',
        },
      ],
    });

    render(<ResourceGovernorPage />);

    const analyticsCard = await screen.findByTestId('rate-limit-analytics-card');
    expect(analyticsCard).toBeInTheDocument();
    expect(screen.getByText('Rate Limit Analytics')).toBeInTheDocument();
    expect(screen.getByTestId('total-throttle-events').textContent).toBe('12');

    // Throttle event count badge
    const badge = await screen.findByTestId('throttle-event-count');
    expect(badge).toBeInTheDocument();
    expect(badge.textContent).toContain('rejections');
  });

  it('renders user autocomplete Select for policy resolution when users are loaded', async () => {
    const user = userEvent.setup();
    render(<ResourceGovernorPage />);

    // Wait for scope context to finish loading
    await waitFor(() => {
      expect(screen.queryByText('Loading scope context...')).not.toBeInTheDocument();
    });

    // The resolution section should render a <select> (design system Select) populated with users
    const userSelect = screen.getByLabelText('User');
    expect(userSelect.tagName).toBe('SELECT');

    // Verify the user options are populated from getUsersPage mock
    const options = within(userSelect as HTMLElement).getAllByRole('option');
    // First option is the placeholder "Select a user..."
    expect(options.length).toBeGreaterThanOrEqual(3);
    expect(options[0].textContent).toContain('Select a user');
    expect(options[1].textContent).toContain('alice');
    expect(options[2].textContent).toContain('bob');

    // Selecting a user should update the value
    await user.selectOptions(userSelect, '42');
    expect((userSelect as HTMLSelectElement).value).toBe('42');
  });

  it('resolves policy scope when admins enter a username instead of a numeric id', async () => {
    const user = userEvent.setup();
    apiMock.getResourceGovernorPolicy.mockResolvedValueOnce({
      policies: [
        {
          id: 'p-global',
          name: 'Default LLM',
          scope: 'global',
          resource_type: 'llm',
          priority: 1,
          enabled: true,
        },
        {
          id: 'p-user',
          name: 'Power User',
          scope: 'user',
          scope_id: '42',
          resource_type: 'llm',
          priority: 10,
          enabled: true,
        },
      ],
    });
    render(<ResourceGovernorPage />);

    await waitFor(() => {
      expect(screen.queryByText('Loading scope context...')).not.toBeInTheDocument();
    });
    await user.type(screen.getByLabelText(/User ID or name/i), 'alice');
    await user.click(screen.getByRole('button', { name: 'Resolve Policy' }));

    expect(
      await screen.findByText(/Global policy "Default LLM".*Winner: Power User/i)
    ).toBeInTheDocument();
  });
});
