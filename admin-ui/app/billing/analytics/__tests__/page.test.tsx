/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import BillingAnalyticsPage from '../page';
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
  usePathname: () => '/billing/analytics',
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getBillingAnalytics: vi.fn(),
  },
}));

const apiMock = vi.mocked(api);

const sampleAnalytics = {
  mrr_cents: 249900,
  subscriber_count: 42,
  active_count: 35,
  trialing_count: 5,
  past_due_count: 2,
  canceled_count: 3,
  plan_distribution: [
    { plan_name: 'Pro', count: 25 },
    { plan_name: 'Business', count: 10 },
    { plan_name: 'Starter', count: 7 },
  ],
  trial_conversion_rate_pct: 68.5,
};

// Helper to control the billing flag per-test
let billingEnabled = true;

vi.mock('@/lib/billing', () => ({
  isBillingEnabled: () => billingEnabled,
}));

beforeEach(() => {
  billingEnabled = true;
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();
  apiMock.getBillingAnalytics.mockResolvedValue(sampleAnalytics);
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('BillingAnalyticsPage', () => {
  it('renders MRR, subscribers, and trial conversion cards', async () => {
    render(<BillingAnalyticsPage />);

    // MRR: 249900 cents = $2499.00
    expect(await screen.findByText('$2499.00/mo')).toBeInTheDocument();

    // Subscribers
    expect(screen.getByText('42')).toBeInTheDocument();
    expect(screen.getByText('35 active, 5 trialing')).toBeInTheDocument();

    // Trial conversion
    expect(screen.getByText('68.5%')).toBeInTheDocument();

    // Past due
    expect(screen.getByText('2')).toBeInTheDocument();
    expect(screen.getByText('3 canceled')).toBeInTheDocument();
  });

  it('shows billing-disabled alert when billing is not enabled', async () => {
    billingEnabled = false;
    render(<BillingAnalyticsPage />);

    expect(
      await screen.findByText(/billing is not enabled/i)
    ).toBeInTheDocument();
    expect(screen.getByText(/NEXT_PUBLIC_BILLING_ENABLED=true/i)).toBeInTheDocument();
    expect(apiMock.getBillingAnalytics).not.toHaveBeenCalled();
  });

  it('handles API error gracefully', async () => {
    apiMock.getBillingAnalytics.mockRejectedValue(new Error('Payment service unavailable'));
    render(<BillingAnalyticsPage />);

    expect(await screen.findByText('Payment service unavailable')).toBeInTheDocument();
  });

  it('renders plan distribution bars', async () => {
    render(<BillingAnalyticsPage />);

    expect(await screen.findByText('Plan Distribution')).toBeInTheDocument();
    expect(screen.getByText('Pro')).toBeInTheDocument();
    expect(screen.getByText('25 subscribers')).toBeInTheDocument();
    expect(screen.getByText('Business')).toBeInTheDocument();
    expect(screen.getByText('10 subscribers')).toBeInTheDocument();
    expect(screen.getByText('Starter')).toBeInTheDocument();
    expect(screen.getByText('7 subscribers')).toBeInTheDocument();
  });

  it('shows loading skeletons while data is being fetched', async () => {
    // Make the API call hang indefinitely
    apiMock.getBillingAnalytics.mockReturnValue(new Promise(() => {}));
    const { container } = render(<BillingAnalyticsPage />);

    // CardSkeleton renders placeholder divs with animate-pulse
    const skeletons = container.querySelectorAll('[class*="animate-pulse"]');
    expect(skeletons.length).toBeGreaterThan(0);
  });

  it('shows page heading and subtext', async () => {
    render(<BillingAnalyticsPage />);

    expect(await screen.findByText('Revenue Analytics')).toBeInTheDocument();
    expect(screen.getByText('Billing metrics and subscription overview')).toBeInTheDocument();
  });

  it('renders KPI card titles correctly', async () => {
    render(<BillingAnalyticsPage />);

    expect(await screen.findByText('Monthly Recurring Revenue')).toBeInTheDocument();
    expect(screen.getByText('Subscribers')).toBeInTheDocument();
    expect(screen.getByText('Trial Conversion')).toBeInTheDocument();
    expect(screen.getByText('Past Due')).toBeInTheDocument();
  });

  it('renders singular subscriber text for count of 1', async () => {
    apiMock.getBillingAnalytics.mockResolvedValue({
      ...sampleAnalytics,
      plan_distribution: [{ plan_name: 'Solo', count: 1 }],
    });
    render(<BillingAnalyticsPage />);

    expect(await screen.findByText('1 subscriber')).toBeInTheDocument();
  });
});
