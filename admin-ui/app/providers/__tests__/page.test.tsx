/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ProvidersPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

vi.mock('next/link', () => ({
  default: ({ href, children, className }: { href: string; children: ReactNode; className?: string }) => (
    <a href={href} className={className}>{children}</a>
  ),
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => confirmMock,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getLLMProviders: vi.fn(),
    getUsers: vi.fn(),
    getOrganizations: vi.fn(),
    getLLMProviderOverrides: vi.fn(),
    getLlmUsageSummary: vi.fn(),
    getLlmUsage: vi.fn(),
    updateLLMProviderOverride: vi.fn(),
    deleteLLMProviderOverride: vi.fn(),
    testLLMProvider: vi.fn(),
    getUserByokKeys: vi.fn(),
    getOrgByokKeys: vi.fn(),
    createUserByokKey: vi.fn(),
    createOrgByokKey: vi.fn(),
    deleteUserByokKey: vi.fn(),
    deleteOrgByokKey: vi.fn(),
  },
}));

const apiMock = vi.mocked(api);

beforeEach(() => {
  confirmMock.mockResolvedValue(true);
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getLLMProviders.mockResolvedValue([
    {
      name: 'openai',
      enabled: true,
      models: ['gpt-4o', 'gpt-3.5-turbo', 'gpt-4.1'],
      default_model: 'gpt-4o',
    },
  ]);
  apiMock.getUsers.mockResolvedValue([]);
  apiMock.getOrganizations.mockResolvedValue([]);
  apiMock.getLLMProviderOverrides.mockResolvedValue({ items: [] });
  apiMock.getLlmUsageSummary.mockImplementation((params?: Record<string, string | string[]>) => {
    const groupBy = params?.group_by;
    const normalizedGroupBy = Array.isArray(groupBy) ? groupBy : [groupBy];
    if (normalizedGroupBy.includes('day')) {
      return Promise.resolve({
        items: [
          {
            group_value: 'openai',
            group_value_secondary: '2026-02-11',
            requests: 400,
            errors: 4,
            input_tokens: 100000,
            output_tokens: 50000,
            total_tokens: 150000,
            total_cost_usd: 145.2,
            latency_avg_ms: 310,
          },
          {
            group_value: 'openai',
            group_value_secondary: '2026-02-12',
            requests: 350,
            errors: 2,
            input_tokens: 90000,
            output_tokens: 40000,
            total_tokens: 130000,
            total_cost_usd: 128.75,
            latency_avg_ms: 298,
          },
        ],
      });
    }
    return Promise.resolve({
      items: [
        {
          group_value: 'openai',
          requests: 2400,
          errors: 24,
          input_tokens: 800000,
          output_tokens: 400000,
          total_tokens: 1200000,
          total_cost_usd: 1234.56,
          latency_avg_ms: 321,
        },
      ],
    });
  });
  apiMock.getLlmUsage.mockResolvedValue({
    items: [
      {
        model: 'gpt-4o',
        status: 200,
        prompt_tokens: 100,
        completion_tokens: 50,
        total_tokens: 150,
        total_cost_usd: 0.30,
        latency_ms: 100,
      },
      {
        model: 'gpt-4o',
        status: 500,
        prompt_tokens: 20,
        completion_tokens: 10,
        total_tokens: 30,
        total_cost_usd: 0.05,
        latency_ms: 120,
      },
      {
        model: 'gpt-3.5-turbo',
        status: 200,
        prompt_tokens: 30,
        completion_tokens: 20,
        total_tokens: 50,
        total_cost_usd: 0.10,
        latency_ms: 95,
      },
      {
        model: 'gpt-4.1',
        status: 200,
        prompt_tokens: 200,
        completion_tokens: 100,
        total_tokens: 300,
        total_cost_usd: 0.80,
        latency_ms: 200,
      },
    ],
  });
  apiMock.updateLLMProviderOverride.mockResolvedValue({});
  apiMock.deleteLLMProviderOverride.mockResolvedValue({});
  apiMock.testLLMProvider.mockResolvedValue({});
  apiMock.getUserByokKeys.mockResolvedValue([]);
  apiMock.getOrgByokKeys.mockResolvedValue([]);
  apiMock.createUserByokKey.mockResolvedValue({});
  apiMock.createOrgByokKey.mockResolvedValue({});
  apiMock.deleteUserByokKey.mockResolvedValue({});
  apiMock.deleteOrgByokKey.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('ProvidersPage usage enhancements', () => {
  it('expands provider rows and renders per-model usage breakdown', async () => {
    const user = userEvent.setup();
    render(<ProvidersPage />);

    const rowLabel = (await screen.findAllByText('OpenAI')).find((node) => Boolean(node.closest('tr')));
    expect(rowLabel).toBeTruthy();
    if (!rowLabel) throw new Error('OpenAI row label not found');
    await user.click(screen.getByRole('button', { name: /expand model usage for openai/i }));

    expect(await screen.findByText('Per-model usage (7d)')).toBeInTheDocument();
    expect(await screen.findByText('$0.35')).toBeInTheDocument();
    expect(await screen.findByText('$0.80')).toBeInTheDocument();

    await waitFor(() => {
      expect(apiMock.getLlmUsage).toHaveBeenCalledWith(expect.objectContaining({
        provider: 'openai',
        limit: '500',
        page: '1',
      }));
    });
  });

  it('formats provider token and request values with compact notation', async () => {
    render(<ProvidersPage />);

    const rowLabel = (await screen.findAllByText('OpenAI')).find((node) => Boolean(node.closest('tr')));
    if (!rowLabel) throw new Error('OpenAI row label not found');
    const row = rowLabel.closest('tr');
    expect(row).not.toBeNull();

    expect(within(row as HTMLElement).getByText('2.4K')).toBeInTheDocument();
    expect(within(row as HTMLElement).getByText('1.2M')).toBeInTheDocument();
    expect(within(row as HTMLElement).getByText('$1,234.56')).toBeInTheDocument();
    expect(within(row as HTMLElement).getByText('1.0%')).toBeInTheDocument();
  });

  it('shows metric fallback placeholders when usage summary endpoint fails', async () => {
    apiMock.getLlmUsageSummary.mockRejectedValueOnce(new Error('summary unavailable'));
    render(<ProvidersPage />);

    const rowLabel = (await screen.findAllByText('OpenAI')).find((node) => Boolean(node.closest('tr')));
    if (!rowLabel) throw new Error('OpenAI row label not found');
    const row = rowLabel.closest('tr');
    expect(row).not.toBeNull();
    expect(within(row as HTMLElement).getAllByText('—').length).toBeGreaterThanOrEqual(4);
  });

  it('renders provider token sparkline trend data', async () => {
    render(<ProvidersPage />);

    const sparkline = await screen.findByTestId('provider-token-sparkline-openai');
    const polyline = sparkline.querySelector('polyline');
    expect(polyline).toBeTruthy();
    expect(polyline?.getAttribute('points')).toContain(',');
    expect(apiMock.getLlmUsageSummary).toHaveBeenCalledWith(expect.objectContaining({
      group_by: ['provider', 'day'],
    }));
  });

  it('renders deprecated model badge and shows migration guidance dialog', async () => {
    const user = userEvent.setup();
    render(<ProvidersPage />);

    const deprecationButton = await screen.findByRole('button', {
      name: /view deprecation details for gpt-3\.5-turbo/i,
    });
    await user.click(deprecationButton);

    expect(await screen.findByText('Deprecated Model Warning')).toBeInTheDocument();
    expect(
      await screen.findByText(/This model is deprecated\. 1 requests used it in the last 7 days\./i)
    ).toBeInTheDocument();
    expect(await screen.findByText(/Consider migrating to gpt-4\.1-mini\./i)).toBeInTheDocument();
  });
});

describe('ProvidersPage provider override denylist', () => {
  it('saves denied models from the override dialog', async () => {
    const user = userEvent.setup();
    render(<ProvidersPage />);

    await user.click(await screen.findByTitle('Manage override'));
    await user.type(screen.getByLabelText('Denied models'), 'gpt-4o, gpt-4.1');
    await user.click(screen.getByRole('button', { name: 'Save' }));

    await waitFor(() => {
      expect(apiMock.updateLLMProviderOverride).toHaveBeenCalledWith(
        'openai',
        expect.objectContaining({ allowed_models: [], denied_models: ['gpt-4o', 'gpt-4.1'] }),
      );
    });
  });

  it('shows the stored denylist in the row and the dialog', async () => {
    apiMock.getLLMProviderOverrides.mockResolvedValue({ items: [{ provider: 'openai', denied_models: ['gpt-4o'] }] });
    const user = userEvent.setup();
    render(<ProvidersPage />);

    expect(await screen.findByText('1 models denied')).toBeInTheDocument();
    await user.click(screen.getByTitle('Manage override'));
    expect((screen.getByLabelText('Denied models') as HTMLInputElement).value).toBe('gpt-4o');
  });
});
