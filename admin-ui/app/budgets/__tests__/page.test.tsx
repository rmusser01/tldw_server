/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, cleanup, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import BudgetsPage from '../page';
import { api } from '@/lib/api-client';

const setPaginationValuesMock = vi.hoisted(() => vi.fn());
const clearPaginationMock = vi.hoisted(() => vi.fn());
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

vi.mock('@/lib/use-url-state', () => ({
  useUrlMultiState: () => ([
    { page: 1, pageSize: 20 },
    setPaginationValuesMock,
    clearPaginationMock,
  ]),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getBudgets: vi.fn(),
    getNotificationSettings: vi.fn(),
    updateBudget: vi.fn(),
  },
}));

type ApiMock = {
  getBudgets: ReturnType<typeof vi.fn>;
  getNotificationSettings: ReturnType<typeof vi.fn>;
  updateBudget: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const budgetRow = {
  org_id: 11,
  org_name: 'Acme Co',
  org_slug: 'acme',
  plan_name: 'pro',
  plan_display_name: 'Pro Plan',
  budgets: {
    budget_day_usd: 100,
    budget_month_usd: 200,
    budget_day_tokens: 300,
    budget_month_tokens: 400,
    alert_thresholds: { global: [80, 95] },
    enforcement_mode: { global: 'soft' },
  },
  updated_at: '2024-01-01T00:00:00Z',
};

beforeEach(() => {
  promptPrivilegedActionMock.mockResolvedValue({ reason: 'test', adminPassword: '' });
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getBudgets.mockResolvedValue({
    items: [budgetRow],
    total: 1,
    page: 1,
    limit: 20,
  });
  apiMock.getNotificationSettings.mockResolvedValue({
    channels: [{ type: 'email', enabled: true, config: {} }],
    alert_threshold: 'warning',
  });
  apiMock.updateBudget.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('BudgetsPage', () => {
  it('shows loading skeleton while fetching', async () => {
    apiMock.getBudgets.mockImplementation(() => new Promise(() => {}));

    render(<BudgetsPage />);

    expect(await screen.findByTestId('table-skeleton')).toBeTruthy();
  });

  it('renders budget rows with plan, caps, and edit actions', async () => {
    render(<BudgetsPage />);

    expect(await screen.findByText('Acme Co')).toBeTruthy();
    expect(screen.getByText('Pro Plan')).toBeTruthy();
    expect(screen.getByText('$100.00')).toBeTruthy();
    expect(screen.getByText('$200.00')).toBeTruthy();
    expect(screen.getByText('300')).toBeTruthy();
    expect(screen.getByText('400')).toBeTruthy();
    expect(screen.queryByText('Read-only')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /edit/i })).toBeInTheDocument();
  });

  it('shows monitoring notification channel wiring status', async () => {
    render(<BudgetsPage />);
    expect(await screen.findByText(/Budget threshold alerts are wired to monitoring notification channels/i)).toBeInTheDocument();
  });

  it('validates budget edit dialog caps and blocks invalid save', async () => {
    const user = userEvent.setup();
    render(<BudgetsPage />);

    await screen.findByText('Acme Co');
    await user.click(screen.getByRole('button', { name: /edit/i }));

    const dayUsdInput = screen.getByLabelText(/daily usd cap/i);
    await user.clear(dayUsdInput);
    await user.type(dayUsdInput, '-1');
    await user.click(screen.getByRole('button', { name: /save budget/i }));

    expect(await screen.findByText('Daily USD cap must be a positive number.')).toBeInTheDocument();
    expect(apiMock.updateBudget).not.toHaveBeenCalled();
  });

  it('allows enforcement mode selection changes in the edit dialog', async () => {
    const user = userEvent.setup();
    render(<BudgetsPage />);

    await screen.findByText('Acme Co');
    await user.click(screen.getByRole('button', { name: /edit/i }));

    const select = screen.getByTestId('budget-enforcement-budget_day_usd');
    await user.selectOptions(select, 'hard');
    expect((select as HTMLSelectElement).value).toBe('hard');
  });

  it('requires confirmation before enabling hard enforcement and saves payload on confirm', async () => {
    const user = userEvent.setup();
    render(<BudgetsPage />);

    await screen.findByText('Acme Co');
    await user.click(screen.getByRole('button', { name: /edit/i }));

    const select = screen.getByTestId('budget-enforcement-budget_day_usd');
    await user.selectOptions(select, 'hard');

    await user.click(screen.getByRole('button', { name: /save budget/i }));

    await waitFor(() => {
      expect(promptPrivilegedActionMock).toHaveBeenCalled();
    });
    await waitFor(() => {
      expect(apiMock.updateBudget).toHaveBeenCalledWith(
        '11',
        expect.objectContaining({
          budgets: expect.objectContaining({
            enforcement_mode: expect.objectContaining({
              per_metric: expect.objectContaining({
                budget_day_usd: 'hard',
              }),
            }),
          }),
        })
      );
    });
  });

  it('displays empty state when no budgets exist', async () => {
    apiMock.getBudgets.mockResolvedValue({ items: [], total: 0, page: 1, limit: 20 });

    render(<BudgetsPage />);

    expect(await screen.findByText('No budgets found for the selected scope.')).toBeTruthy();
  });

  it('displays error message on API failure', async () => {
    apiMock.getBudgets.mockRejectedValue(new Error('Network error'));

    render(<BudgetsPage />);

    expect(await screen.findByText('Network error')).toBeTruthy();
  });

  it('updates pagination controls', async () => {
    apiMock.getBudgets.mockResolvedValue({
      items: [budgetRow],
      total: 60,
      page: 1,
      limit: 20,
    });

    render(<BudgetsPage />);

    await screen.findByText('Acme Co');
    setPaginationValuesMock.mockClear();
    clearPaginationMock.mockClear();

    const user = userEvent.setup();
    await user.click(screen.getByRole('button', { name: 'Go to page 2' }));
    expect(setPaginationValuesMock).toHaveBeenCalledWith({ page: 2 });

    await user.selectOptions(screen.getByRole('combobox'), '50');
    expect(setPaginationValuesMock).toHaveBeenCalledWith({ pageSize: 50, page: 1 });
  });
});
