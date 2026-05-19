/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import PlansPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const privilegedActionMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('next/link', () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a>,
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

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => confirmMock,
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => privilegedActionMock,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getPlans: vi.fn(),
    getSubscriptions: vi.fn(),
    createPlan: vi.fn(),
    updatePlan: vi.fn(),
    deletePlan: vi.fn(),
    getSubscriptions: vi.fn(),
  },
}));

let billingEnabled = true;

vi.mock('@/lib/billing', () => ({
  isBillingEnabled: () => billingEnabled,
}));

type ApiMock = {
  getPlans: ReturnType<typeof vi.fn>;
  getSubscriptions: ReturnType<typeof vi.fn>;
  createPlan: ReturnType<typeof vi.fn>;
  updatePlan: ReturnType<typeof vi.fn>;
  deletePlan: ReturnType<typeof vi.fn>;
  getSubscriptions: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const samplePlans = [
  {
    id: 'plan_free',
    name: 'Free',
    tier: 'free' as const,
    stripe_product_id: '',
    stripe_price_id: '',
    monthly_price_cents: 0,
    included_token_credits: 10000,
    overage_rate_per_1k_tokens_cents: 0,
    features: ['basic_search'],
    is_default: true,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  },
  {
    id: 'plan_pro',
    name: 'Pro',
    tier: 'pro' as const,
    stripe_product_id: 'prod_abc',
    stripe_price_id: 'price_abc',
    monthly_price_cents: 2999,
    included_token_credits: 500000,
    overage_rate_per_1k_tokens_cents: 5,
    features: ['basic_search', 'advanced_rag', 'priority_support'],
    is_default: false,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  },
  {
    id: 'plan_ent',
    name: 'Enterprise',
    tier: 'enterprise' as const,
    stripe_product_id: 'prod_xyz',
    stripe_price_id: 'price_xyz',
    monthly_price_cents: 9999,
    included_token_credits: 2000000,
    overage_rate_per_1k_tokens_cents: 3,
    features: ['basic_search', 'advanced_rag', 'priority_support', 'sso', 'custom_models'],
    is_default: false,
    created_at: '2026-01-01T00:00:00Z',
    updated_at: '2026-01-01T00:00:00Z',
  },
];

beforeEach(() => {
  billingEnabled = true;
  confirmMock.mockResolvedValue({ reason: 'test audit reason', adminPassword: '' });
  privilegedActionMock.mockResolvedValue({ reason: 'test audit reason', adminPassword: '' });
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();
  apiMock.getPlans.mockResolvedValue(samplePlans);
  apiMock.getSubscriptions.mockResolvedValue([]);
  apiMock.createPlan.mockResolvedValue(samplePlans[0]);
  apiMock.updatePlan.mockResolvedValue(samplePlans[0]);
  apiMock.deletePlan.mockResolvedValue({});
  apiMock.getSubscriptions.mockResolvedValue([]);
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('PlansPage', () => {
  it('renders plan cards with names and prices', async () => {
    render(<PlansPage />);

    // "Free" appears both as card title and PlanBadge, so use getAllByText
    expect((await screen.findAllByText('Free')).length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Pro').length).toBeGreaterThanOrEqual(1);
    expect(screen.getAllByText('Enterprise').length).toBeGreaterThanOrEqual(1);

    expect(screen.getByText('$0.00/mo')).toBeInTheDocument();
    expect(screen.getByText('$29.99/mo')).toBeInTheDocument();
    expect(screen.getByText('$99.99/mo')).toBeInTheDocument();
  });

  it('shows included token credits for each plan', async () => {
    render(<PlansPage />);

    expect(await screen.findByText('10,000')).toBeInTheDocument();
    expect(screen.getByText('500,000')).toBeInTheDocument();
    expect(screen.getByText('2,000,000')).toBeInTheDocument();
  });

  it('shows feature counts', async () => {
    render(<PlansPage />);

    await screen.findByText('$0.00/mo');
    const featureCells = screen.getAllByText('Features');
    expect(featureCells).toHaveLength(3);
  });

  it('shows billing not enabled message when billing is off', async () => {
    billingEnabled = false;
    render(<PlansPage />);

    expect(await screen.findByText(/Billing is not enabled/)).toBeInTheDocument();
    expect(screen.queryByText('Create Plan')).not.toBeInTheDocument();
  });

  it('opens create plan dialog with form fields', async () => {
    const user = userEvent.setup();
    render(<PlansPage />);

    await screen.findByText('$0.00/mo');

    await user.click(screen.getByRole('button', { name: /Create Plan/ }));

    expect(await screen.findByText('Add a new subscription plan.')).toBeInTheDocument();
    expect(screen.getByLabelText(/Name/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Tier/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Monthly Price/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Included Token Credits/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Overage Rate/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Stripe Product ID/)).toBeInTheDocument();
    expect(screen.getByLabelText(/Stripe Price ID/)).toBeInTheDocument();
  });

  it('calls deletePlan when delete is confirmed (no subscribers)', async () => {
    apiMock.getSubscriptions.mockResolvedValue([]);
    const user = userEvent.setup();
    render(<PlansPage />);

    await screen.findByText('$0.00/mo');

    const deleteButtons = screen.getAllByRole('button', { name: /Delete/ });
    await user.click(deleteButtons[0]);

    await waitFor(() => {
      expect(privilegedActionMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Delete Plan',
          message: expect.stringContaining('Free'),
        })
      );
    });

    await waitFor(() => {
      expect(apiMock.deletePlan).toHaveBeenCalledWith('plan_free');
    });
  });

  it('warns about active subscribers before deletion', async () => {
    apiMock.getSubscriptions.mockResolvedValue([
      { id: 'sub_1', plan_id: 'plan_free', org_id: 1, status: 'active' },
      { id: 'sub_2', plan_id: 'plan_free', org_id: 2, status: 'active' },
    ]);
    const user = userEvent.setup();
    render(<PlansPage />);

    await screen.findByText('$0.00/mo');

    const deleteButtons = screen.getAllByRole('button', { name: /Delete/ });
    await user.click(deleteButtons[0]);

    await waitFor(() => {
      expect(confirmMock).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('2 active subscription'),
        })
      );
    });
  });

  it('shows warning when subscriber check fails', async () => {
    apiMock.getSubscriptions.mockRejectedValue(new Error('Network error'));
    const user = userEvent.setup();
    render(<PlansPage />);

    await screen.findByText('$0.00/mo');

    const deleteButtons = screen.getAllByRole('button', { name: /Delete/ });
    await user.click(deleteButtons[0]);

    await waitFor(() => {
      expect(confirmMock).toHaveBeenCalledWith(
        expect.objectContaining({
          message: expect.stringContaining('Could not verify'),
        })
      );
    });
  });
});
