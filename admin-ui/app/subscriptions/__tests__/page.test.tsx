/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import SubscriptionsPage from '../page';
import { api } from '@/lib/api-client';

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

vi.mock('@/components/OrgContextSwitcher', () => ({
  useOrgContext: () => ({ selectedOrg: null }),
  OrgContextSwitcher: () => <div data-testid="org-switcher" />,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('next/link', () => ({
  default: ({ children, href }: { children: ReactNode; href: string }) => (
    <a href={href}>{children}</a>
  ),
}));

vi.mock('@/lib/billing', () => ({
  isBillingEnabled: () => true,
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getSubscriptions: vi.fn(),
  },
}));

type ApiMock = {
  getSubscriptions: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const sampleSubscription = {
  id: 'sub_1',
  org_id: 42,
  org_name: 'Acme Corp',
  plan_id: 'plan_pro',
  plan: {
    id: 'plan_pro',
    name: 'Pro',
    tier: 'pro' as const,
    stripe_product_id: 'prod_1',
    stripe_price_id: 'price_1',
    monthly_price_cents: 2900,
    included_token_credits: 100000,
    overage_rate_per_1k_tokens_cents: 5,
    features: ['feature_a'],
    is_default: false,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  stripe_subscription_id: 'sub_stripe_1',
  status: 'active' as const,
  current_period_start: '2024-06-01T00:00:00Z',
  current_period_end: '2024-07-01T00:00:00Z',
  created_at: '2024-01-15T00:00:00Z',
  updated_at: '2024-06-01T00:00:00Z',
  at_risk: false,
  at_risk_reasons: [],
  days_past_due: 0,
  days_since_created: 168,
  cancel_at_period_end: false,
};

const atRiskSubscription = {
  id: 'sub_2',
  org_id: 99,
  org_name: 'Risky Inc',
  plan_id: 'plan_pro',
  plan: {
    id: 'plan_pro',
    name: 'Pro',
    tier: 'pro' as const,
    stripe_product_id: 'prod_1',
    stripe_price_id: 'price_1',
    monthly_price_cents: 2900,
    included_token_credits: 100000,
    overage_rate_per_1k_tokens_cents: 5,
    features: [],
    is_default: false,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  stripe_subscription_id: 'sub_stripe_2',
  status: 'past_due' as const,
  current_period_start: '2024-05-01T00:00:00Z',
  current_period_end: '2024-06-01T00:00:00Z',
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-06-01T00:00:00Z',
  at_risk: true,
  at_risk_reasons: ['past_due_extended'],
  days_past_due: 14,
  days_since_created: 200,
  cancel_at_period_end: false,
};

const cancellingSubscription = {
  id: 'sub_3',
  org_id: 77,
  org_name: null,
  plan_id: 'plan_pro',
  plan: {
    id: 'plan_pro',
    name: 'Pro',
    tier: 'pro' as const,
    stripe_product_id: 'prod_1',
    stripe_price_id: 'price_1',
    monthly_price_cents: 2900,
    included_token_credits: 100000,
    overage_rate_per_1k_tokens_cents: 5,
    features: [],
    is_default: false,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  },
  stripe_subscription_id: 'sub_stripe_3',
  status: 'active' as const,
  current_period_start: '2024-06-01T00:00:00Z',
  current_period_end: '2024-07-01T00:00:00Z',
  created_at: '2024-02-01T00:00:00Z',
  updated_at: '2024-06-01T00:00:00Z',
  at_risk: true,
  at_risk_reasons: ['cancelling'],
  days_past_due: 0,
  days_since_created: 150,
  cancel_at_period_end: true,
};

describe('SubscriptionsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  afterEach(cleanup);

  it('renders subscriptions in the table with org name', async () => {
    apiMock.getSubscriptions.mockResolvedValue([sampleSubscription]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Acme Corp')).toBeInTheDocument();
    });

    // Org ID shown in parentheses
    expect(screen.getByText('(42)')).toBeInTheDocument();
    // Plan badge
    expect(screen.getByText('Pro')).toBeInTheDocument();
    // Status badge
    expect(screen.getByText('active')).toBeInTheDocument();
    // Organization link
    const link = screen.getByText('Acme Corp').closest('a');
    expect(link?.getAttribute('href')).toBe('/organizations/42');
  });

  it('falls back to "Org {id}" when org_name is missing', async () => {
    const subWithoutName = { ...sampleSubscription, org_name: undefined };
    apiMock.getSubscriptions.mockResolvedValue([subWithoutName]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Org 42')).toBeInTheDocument();
    });
  });

  it('shows empty state when no subscriptions', async () => {
    apiMock.getSubscriptions.mockResolvedValue([]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getByText('No subscriptions found')).toBeInTheDocument();
    });
  });

  it('shows error when fetch fails', async () => {
    apiMock.getSubscriptions.mockRejectedValue(new Error('Network error'));

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Network error')).toBeInTheDocument();
    });

    expect(toastErrorMock).toHaveBeenCalledWith('Network error');
  });

  it('renders Needs Attention section for at-risk subscriptions', async () => {
    apiMock.getSubscriptions.mockResolvedValue([sampleSubscription, atRiskSubscription]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getByTestId('needs-attention-section')).toBeInTheDocument();
    });

    expect(screen.getByText('Needs Attention (1)')).toBeInTheDocument();
    // At-risk sub appears in both Needs Attention and table
    expect(screen.getAllByText('Risky Inc').length).toBeGreaterThanOrEqual(1);
  });

  it('does not render Needs Attention section when no at-risk subscriptions', async () => {
    apiMock.getSubscriptions.mockResolvedValue([sampleSubscription]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getByText('Acme Corp')).toBeInTheDocument();
    });

    expect(screen.queryByTestId('needs-attention-section')).not.toBeInTheDocument();
  });

  it('shows at-risk badges for past_due_extended subscriptions', async () => {
    apiMock.getSubscriptions.mockResolvedValue([atRiskSubscription]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getAllByTestId('badge-past-due').length).toBeGreaterThan(0);
    });
  });

  it('shows cancelling badge for subscriptions set to cancel', async () => {
    apiMock.getSubscriptions.mockResolvedValue([cancellingSubscription]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getAllByTestId('badge-cancelling').length).toBeGreaterThan(0);
    });

    // Falls back to Org {id} when org_name is null
    // Appears in both Needs Attention section and table row
    expect(screen.getAllByText('Org 77').length).toBeGreaterThanOrEqual(1);
  });

  it('highlights at-risk rows with background color', async () => {
    apiMock.getSubscriptions.mockResolvedValue([sampleSubscription, atRiskSubscription]);

    render(<SubscriptionsPage />);

    await waitFor(() => {
      expect(screen.getAllByText('Risky Inc').length).toBeGreaterThanOrEqual(1);
    });

    // The at-risk row (table row) should have bg-destructive/5 class
    // Find the table row containing Risky Inc
    const riskyLinks = screen.getAllByText('Risky Inc');
    const atRiskRow = riskyLinks.map((el) => el.closest('tr')).find((tr) => tr !== null);
    expect(atRiskRow?.className).toContain('bg-destructive/5');

    // The normal row should not
    const normalRow = screen.getByText('Acme Corp').closest('tr');
    expect(normalRow?.className).not.toContain('bg-destructive/5');
  });
});
