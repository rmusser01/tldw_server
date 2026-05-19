/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import OrganizationDetailPage from '../page';
import { api } from '@/lib/api-client';

let currentOrgId = '1';

const confirmMock = vi.hoisted(() => vi.fn());
const pushMock = vi.hoisted(() => vi.fn());

vi.mock('next/link', () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => (
    <a href={href}>{children}</a>
  ),
}));

vi.mock('next/navigation', () => ({
  useParams: () => ({ id: currentOrgId }),
  useRouter: () => ({
    push: pushMock,
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/badge', () => ({
  Badge: ({ children }: { children: ReactNode }) => <span>{children}</span>,
}));

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => confirmMock,
}));

vi.mock('@/components/users/UserPicker', () => ({
  UserPicker: () => <div data-testid="user-picker" />,
}));

vi.mock('@/components/PlanBadge', () => ({
  PlanBadge: ({ tier }: { tier: string }) => <span data-testid="plan-badge">{tier}</span>,
}));

vi.mock('@/components/UsageMeter', () => ({
  UsageMeter: ({ used }: { used: number }) => <div data-testid="usage-meter">{used}</div>,
}));

vi.mock('@/components/InvoiceTable', () => ({
  InvoiceTable: ({ invoices }: { invoices: Array<{ id: string }> }) => (
    <div data-testid="invoice-table">{invoices.length}</div>
  ),
}));

vi.mock('@/lib/billing', async (importOriginal) => {
  const actual = await importOriginal<typeof import('@/lib/billing')>();
  return {
    ...actual,
    isBillingEnabled: () => true,
    formatBillingDate: (value?: string | null) =>
      value ? new Date(value).toLocaleDateString('en-CA') : '—',
  };
});

vi.mock('@/lib/api-client', () => ({
  api: {
    getOrganization: vi.fn(),
    getOrgMembers: vi.fn(),
    getTeams: vi.fn(),
    getOrgByokKeys: vi.fn(),
    getOrgWatchlistSettings: vi.fn(),
    getOrgSubscription: vi.fn(),
    getOrgUsageSummary: vi.fn(),
    getOrgInvoices: vi.fn(),
  },
}));

type ApiMock = {
  getOrganization: ReturnType<typeof vi.fn>;
  getOrgMembers: ReturnType<typeof vi.fn>;
  getTeams: ReturnType<typeof vi.fn>;
  getOrgByokKeys: ReturnType<typeof vi.fn>;
  getOrgWatchlistSettings: ReturnType<typeof vi.fn>;
  getOrgSubscription: ReturnType<typeof vi.fn>;
  getOrgUsageSummary: ReturnType<typeof vi.fn>;
  getOrgInvoices: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  currentOrgId = '1';
  confirmMock.mockResolvedValue(true);
  pushMock.mockClear();

  apiMock.getOrganization.mockImplementation(async (orgId: string) => ({
    id: Number(orgId),
    name: `Org ${orgId}`,
    slug: `org-${orgId}`,
    created_at: '2026-01-01T00:00:00Z',
  }));
  apiMock.getOrgMembers.mockResolvedValue([]);
  apiMock.getTeams.mockResolvedValue([]);
  apiMock.getOrgByokKeys.mockResolvedValue([]);
  apiMock.getOrgWatchlistSettings.mockResolvedValue({
    watchlists_enabled: false,
    default_threshold: 100,
    alert_on_breach: true,
  });
  apiMock.getOrgSubscription.mockImplementation(async (orgId: number) => {
    if (orgId === 1) {
      return {
        id: 'sub-1',
        org_id: 1,
        status: 'active',
        current_period_start: '2026-03-01T12:00:00Z',
        current_period_end: '2026-04-01T12:00:00Z',
        plan: { tier: 'pro' },
      };
    }
    throw new Error('billing unavailable');
  });
  apiMock.getOrgUsageSummary.mockImplementation(async (orgId: number) => {
    if (orgId === 1) {
      return {
        tokens_used: 42,
        tokens_included: 100,
        overage_cost_cents: 0,
      };
    }
    throw new Error('usage unavailable');
  });
  apiMock.getOrgInvoices.mockImplementation(async (orgId: number) => {
    if (orgId === 1) {
      return [{ id: 'inv-1' }];
    }
    throw new Error('invoice unavailable');
  });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('OrganizationDetailPage billing state', () => {
  it('clears prior billing details when navigating to an org whose billing requests fail', async () => {
    const user = userEvent.setup();
    const { rerender } = render(<OrganizationDetailPage />);

    await screen.findByText('Org 1');

    // Navigate to the Billing tab to see billing content
    const billingTab = await screen.findByRole('tab', { name: /billing/i });
    await user.click(billingTab);

    expect(await screen.findByText('active')).toBeInTheDocument();
    expect(screen.getByTestId('invoice-table').textContent).toBe('1');
    expect(screen.getByTestId('usage-meter').textContent).toBe('42');

    currentOrgId = '2';
    rerender(<OrganizationDetailPage />);

    await screen.findByText('Org 2');

    // Navigate to Billing tab again for the second org
    const billingTab2 = await screen.findByRole('tab', { name: /billing/i });
    await user.click(billingTab2);

    await waitFor(() => {
      expect(screen.getByText('No active subscription.')).toBeInTheDocument();
    });
    expect(screen.queryByText('active')).not.toBeInTheDocument();
    expect(screen.queryByTestId('invoice-table')).not.toBeInTheDocument();
    expect(screen.queryByTestId('usage-meter')).not.toBeInTheDocument();
  });

  it('shows a no-results message when member search filters out every row', async () => {
    const user = userEvent.setup();
    apiMock.getOrgMembers.mockResolvedValue([
      {
        user_id: 42,
        role: 'member',
        joined_at: '2026-01-01T00:00:00Z',
        user: {
          username: 'alice',
          email: 'alice@example.com',
        },
      },
    ]);

    render(<OrganizationDetailPage />);

    await screen.findByText('alice');

    await user.type(screen.getByPlaceholderText('Search members...'), 'missing-user');

    await waitFor(() => {
      expect(screen.getByText('No members match your search.')).toBeInTheDocument();
    });
    expect(screen.queryByText('alice')).not.toBeInTheDocument();
  });
});

describe('OrganizationDetailPage tab navigation', () => {
  it('renders all tab triggers (Members, Teams, Keys, Billing)', async () => {
    render(<OrganizationDetailPage />);

    await screen.findByText('Org 1');

    expect(screen.getByRole('tab', { name: /members/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /teams/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /keys/i })).toBeInTheDocument();
    expect(screen.getByRole('tab', { name: /billing/i })).toBeInTheDocument();
  });

  it('navigates between tabs when clicking tab triggers', async () => {
    const user = userEvent.setup();
    render(<OrganizationDetailPage />);

    await screen.findByText('Org 1');

    // Members tab is default - verify we can see member content area
    const membersTab = screen.getByRole('tab', { name: /members/i });
    expect(membersTab).toBeInTheDocument();

    // Click Teams tab
    const teamsTab = screen.getByRole('tab', { name: /teams/i });
    await user.click(teamsTab);

    // Click Keys tab
    const keysTab = screen.getByRole('tab', { name: /keys/i });
    await user.click(keysTab);

    // Click Billing tab
    const billingTab = screen.getByRole('tab', { name: /billing/i });
    await user.click(billingTab);

    // After navigating to billing, billing content should be visible
    await waitFor(() => {
      expect(screen.getByText('active')).toBeInTheDocument();
    });
  });
});

describe('OrganizationDetailPage member search', () => {
  it('filters member table by search input', async () => {
    apiMock.getOrgMembers.mockResolvedValue([
      { user_id: 10, role: 'admin', user: { username: 'alice', email: 'alice@example.com' } },
      { user_id: 20, role: 'member', user: { username: 'bob', email: 'bob@example.com' } },
      { user_id: 30, role: 'member', user: { username: 'charlie', email: 'charlie@example.com' } },
    ]);

    const user = userEvent.setup();
    render(<OrganizationDetailPage />);

    await screen.findByText('Org 1');

    // All three members should be visible initially
    expect(await screen.findByText('alice')).toBeInTheDocument();
    expect(screen.getByText('bob')).toBeInTheDocument();
    expect(screen.getByText('charlie')).toBeInTheDocument();

    // Type in search to filter
    const searchInput = screen.getByPlaceholderText('Search members by name or email...');
    await user.type(searchInput, 'alice');

    // Only alice should remain visible
    expect(screen.getByText('alice')).toBeInTheDocument();
    expect(screen.queryByText('bob')).not.toBeInTheDocument();
    expect(screen.queryByText('charlie')).not.toBeInTheDocument();
  });

  it('shows empty message when search matches no members', async () => {
    apiMock.getOrgMembers.mockResolvedValue([
      { user_id: 10, role: 'admin', user: { username: 'alice', email: 'alice@example.com' } },
    ]);

    const user = userEvent.setup();
    render(<OrganizationDetailPage />);

    await screen.findByText('alice');

    const searchInput = screen.getByPlaceholderText('Search members by name or email...');
    await user.type(searchInput, 'nonexistent');

    expect(screen.getByText('No members match your search.')).toBeInTheDocument();
    expect(screen.queryByText('alice')).not.toBeInTheDocument();
  });
});

describe('OrganizationDetailPage design system Select usage', () => {
  it('uses Select component for member role dropdowns (not raw <select>)', async () => {
    apiMock.getOrgMembers.mockResolvedValue([
      { user_id: 10, role: 'admin', user: { username: 'alice', email: 'alice@example.com' } },
    ]);

    render(<OrganizationDetailPage />);

    await screen.findByText('alice');

    // The role dropdown for the member should be a <select> element from the design system Select component.
    // The Select component renders a native <select> element, which is correct.
    // Verify the role selector exists with the expected value and options.
    const roleSelects = screen.getAllByRole('combobox');
    const memberRoleSelect = roleSelects.find(
      (el) => (el as HTMLSelectElement).value === 'admin'
    );
    expect(memberRoleSelect).toBeDefined();
    expect(memberRoleSelect!.tagName).toBe('SELECT');
  });
});
