/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import DependenciesPage from '../page';
import { api } from '@/lib/api-client';

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
    getLLMProviders: vi.fn(),
    getLlmUsageSummary: vi.fn(),
    getMetricsText: vi.fn(),
    getDependenciesUptimeHistory: vi.fn(),
    testLLMProvider: vi.fn(),
    getSystemDependencies: vi.fn(),
    getDependencyUptime: vi.fn(),
  },
}));

type ApiMock = {
  getLLMProviders: ReturnType<typeof vi.fn>;
  getLlmUsageSummary: ReturnType<typeof vi.fn>;
  getMetricsText: ReturnType<typeof vi.fn>;
  getDependenciesUptimeHistory: ReturnType<typeof vi.fn>;
  testLLMProvider: ReturnType<typeof vi.fn>;
  getSystemDependencies: ReturnType<typeof vi.fn>;
  getDependencyUptime: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

const SYSTEM_DEPS_FIXTURE = {
  items: [
    { name: 'AuthNZ Database', status: 'healthy', latency_ms: 2.3, error: null, metadata: { type: 'sqlite' } },
    { name: 'ChaChaNotes', status: 'healthy', latency_ms: 1.1, error: null, metadata: {} },
    { name: 'Workflows Engine', status: 'degraded', latency_ms: 15.0, error: 'Queue depth unknown', metadata: {} },
    { name: 'Embeddings Service', status: 'down', latency_ms: 5000.0, error: 'Timeout', metadata: {} },
    { name: 'Metrics Registry', status: 'healthy', latency_ms: 0.5, error: null, metadata: {} },
  ],
  checked_at: '2026-03-27T10:00:00Z',
};

beforeEach(() => {
  apiMock.getLLMProviders.mockResolvedValue({
    providers: [
      {
        name: 'openai',
        enabled: true,
        default_model: 'gpt-4o-mini',
        models: ['gpt-4o-mini'],
      },
      {
        name: 'anthropic',
        enabled: true,
        default_model: 'claude-3-5-sonnet',
        models: ['claude-3-5-sonnet'],
      },
    ],
  });

  apiMock.getLlmUsageSummary.mockImplementation(async (params?: { group_by?: string | string[] }) => {
    if (Array.isArray(params?.group_by)) {
      return [
        { group_value: 'openai', group_value_secondary: '2026-02-12', requests: 20, errors: 1 },
        { group_value: 'openai', group_value_secondary: '2026-02-13', requests: 18, errors: 0 },
        { group_value: 'openai', group_value_secondary: '2026-02-14', requests: 15, errors: 2 },
        { group_value: 'anthropic', group_value_secondary: '2026-02-14', requests: 11, errors: 0 },
      ];
    }
    return [
      { group_value: 'openai', requests: 80, errors: 4, latency_avg_ms: 142 },
      { group_value: 'anthropic', requests: 30, errors: 1, latency_avg_ms: 201 },
    ];
  });

  apiMock.getMetricsText.mockResolvedValue(
    'llm_provider_requests_total{provider="openai",status="ok"} 80'
  );
  apiMock.getDependenciesUptimeHistory.mockResolvedValue({ services: {} });

  apiMock.testLLMProvider.mockImplementation(async ({ provider }: { provider: string }) => {
    if (provider === 'anthropic') {
      throw new Error('timeout');
    }
    return { provider, status: 'ok', model: 'gpt-4o-mini' };
  });

  apiMock.getSystemDependencies.mockResolvedValue(SYSTEM_DEPS_FIXTURE);

  apiMock.getDependencyUptime.mockImplementation(async (name: string) => ({
    dependency_name: name,
    days: 7,
    total_checks: 168,
    healthy_checks: name === 'Embeddings Service' ? 150 : 168,
    uptime_pct: name === 'Embeddings Service' ? 89.3 : 100.0,
    avg_latency_ms: 2.5,
    downtime_minutes: name === 'Embeddings Service' ? 1080 : 0,
    sparkline: Array.from({ length: 168 }, (_, i) =>
      name === 'Embeddings Service' && i > 150 ? 0 : 1,
    ),
  }));
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

const getProviderRow = async (providerName: string) => {
  const label = await screen.findByText(providerName);
  const row = label.closest('tr');
  expect(row).not.toBeNull();
  return row as HTMLElement;
};

describe('DependenciesPage', () => {
  it('renders dependency health grid with passive telemetry and unknown initial status', async () => {
    render(<DependenciesPage />);

    expect(await screen.findByRole('heading', { name: 'External Dependencies' })).toBeInTheDocument();
    const openaiRow = await getProviderRow('Openai');
    const anthropicRow = await getProviderRow('Anthropic');

    expect(within(openaiRow).getByText('Unknown')).toBeInTheDocument();
    expect(within(openaiRow).getByText('5.0%')).toBeInTheDocument();
    expect(within(openaiRow).getAllByText('Never')).toHaveLength(2);
    expect(within(anthropicRow).getByText('Unknown')).toBeInTheDocument();
    expect(within(anthropicRow).getByText('3.3%')).toBeInTheDocument();
    expect(screen.getByRole('img', { name: 'openai availability trend' })).toBeInTheDocument();
    expect(apiMock.testLLMProvider).not.toHaveBeenCalled();
  });

  it('refreshes passive telemetry without running provider connectivity checks', async () => {
    const user = userEvent.setup();

    render(<DependenciesPage />);

    await getProviderRow('Openai');

    expect(apiMock.getLLMProviders).toHaveBeenCalledTimes(1);
    expect(apiMock.getLlmUsageSummary).toHaveBeenCalledTimes(2);
    expect(apiMock.getMetricsText).toHaveBeenCalledTimes(1);
    expect(apiMock.testLLMProvider).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /refresh data/i }));

    await waitFor(() => {
      expect(apiMock.getLLMProviders).toHaveBeenCalledTimes(2);
    });
    expect(apiMock.getLlmUsageSummary).toHaveBeenCalledTimes(4);
    expect(apiMock.getMetricsText).toHaveBeenCalledTimes(2);
    expect(apiMock.testLLMProvider).not.toHaveBeenCalled();
  });

  it('runs all enabled provider checks only after clicking run all checks', async () => {
    const user = userEvent.setup();

    apiMock.getLLMProviders.mockResolvedValueOnce({
      providers: [
        {
          name: 'openai',
          enabled: true,
          default_model: 'gpt-4o-mini',
          models: ['gpt-4o-mini'],
        },
        {
          name: 'anthropic',
          enabled: true,
          default_model: 'claude-3-5-sonnet',
          models: ['claude-3-5-sonnet'],
        },
        {
          name: 'groq',
          enabled: false,
          default_model: 'llama-3.1-8b-instant',
          models: ['llama-3.1-8b-instant'],
        },
      ],
    });

    render(<DependenciesPage />);

    const openaiRow = await getProviderRow('Openai');
    const anthropicRow = await getProviderRow('Anthropic');
    const groqRow = await getProviderRow('Groq');

    expect(apiMock.testLLMProvider).not.toHaveBeenCalled();

    await user.click(screen.getByRole('button', { name: /run all checks/i }));

    await waitFor(() => {
      expect(within(openaiRow).getByText('Reachable')).toBeInTheDocument();
    });
    expect(within(anthropicRow).getByText('Unreachable')).toBeInTheDocument();
    expect(within(groqRow).getByText('Unknown')).toBeInTheDocument();
    expect(apiMock.testLLMProvider).toHaveBeenCalledTimes(2);
    expect(apiMock.testLLMProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'openai',
        use_override: true,
      })
    );
    expect(apiMock.testLLMProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'anthropic',
        use_override: true,
      })
    );
  });

  it('runs per-provider connectivity test and updates displayed status only after click', async () => {
    const user = userEvent.setup();

    apiMock.getLLMProviders.mockResolvedValueOnce({
      providers: [
        {
          name: 'openai',
          enabled: true,
          default_model: 'gpt-4o-mini',
          models: ['gpt-4o-mini'],
        },
      ],
    });

    apiMock.testLLMProvider.mockResolvedValue({
      provider: 'openai',
      status: 'ok',
      model: 'gpt-4o-mini',
    });

    render(<DependenciesPage />);

    const openaiRow = await getProviderRow('Openai');
    expect(within(openaiRow).getByText('Unknown')).toBeInTheDocument();
    expect(apiMock.testLLMProvider).not.toHaveBeenCalled();

    const testButton = within(openaiRow).getByRole('button', {
      name: /test openai connectivity/i,
    });
    await user.click(testButton);

    await waitFor(() => {
      expect(within(openaiRow).getByText('Reachable')).toBeInTheDocument();
    });
    expect(apiMock.testLLMProvider).toHaveBeenCalledWith(
      expect.objectContaining({
        provider: 'openai',
        use_override: true,
      })
    );
  });

  it('highlights unreachable providers with red row styling and last-success info', async () => {
    const user = userEvent.setup();

    render(<DependenciesPage />);

    const row = await getProviderRow('Anthropic');
    expect(row.className).not.toContain('bg-red-50');
    expect(within(row).getByText('Unknown')).toBeInTheDocument();

    await user.click(within(row).getByRole('button', { name: /test anthropic connectivity/i }));

    await waitFor(() => {
      expect(within(row).getByText('Unreachable')).toBeInTheDocument();
    });
    expect(row.className).toContain('bg-red-50');
    expect(within(row).getAllByText('Never')).toHaveLength(1);
  });

  // --- System Dependencies tests ---

  it('renders system dependencies table with backend health data', async () => {
    render(<DependenciesPage />);

    expect(await screen.findByText('System Dependencies')).toBeInTheDocument();

    const dbRow = await screen.findByText('AuthNZ Database');
    expect(dbRow.closest('tr')).not.toBeNull();

    expect(await screen.findByText('ChaChaNotes')).toBeInTheDocument();
    expect(await screen.findByText('Workflows Engine')).toBeInTheDocument();
    expect(await screen.findByText('Embeddings Service')).toBeInTheDocument();
    expect(await screen.findByText('Metrics Registry')).toBeInTheDocument();
  });

  it('shows correct status badges for system dependencies', async () => {
    render(<DependenciesPage />);

    await screen.findByText('AuthNZ Database');

    const healthyBadges = screen.getAllByText('Healthy');
    expect(healthyBadges.length).toBe(3);

    expect(screen.getByText('Degraded')).toBeInTheDocument();
    expect(screen.getByText('Down')).toBeInTheDocument();
  });

  it('displays system dependency error messages', async () => {
    render(<DependenciesPage />);

    expect(await screen.findByText('Queue depth unknown')).toBeInTheDocument();
    expect(screen.getByText('Timeout')).toBeInTheDocument();
  });

  it('shows system component summary counts', async () => {
    render(<DependenciesPage />);

    await screen.findByText('System Components');
    const summarySection = screen.getByText('System Components').closest('div')?.parentElement;
    expect(summarySection).not.toBeNull();

    await screen.findByText('Components Healthy');
  });

  it('shows fallback when system dependencies endpoint fails', async () => {
    apiMock.getSystemDependencies.mockRejectedValue(new Error('Not found'));

    render(<DependenciesPage />);

    // Verify the API was actually called (not just rendering initial empty state)
    await waitFor(() => {
      expect(apiMock.getSystemDependencies).toHaveBeenCalledTimes(1);
    });

    await waitFor(() => {
      expect(screen.getByText(/no system dependency data available/i)).toBeInTheDocument();
    });
  });

  it('calls getSystemDependencies on page load', async () => {
    render(<DependenciesPage />);

    await waitFor(() => {
      expect(apiMock.getSystemDependencies).toHaveBeenCalledTimes(1);
    });
  });

  it('reloads system dependencies when refresh is clicked', async () => {
    const user = userEvent.setup();

    render(<DependenciesPage />);

    await screen.findByText('AuthNZ Database');

    expect(apiMock.getSystemDependencies).toHaveBeenCalledTimes(1);

    await user.click(screen.getByRole('button', { name: /refresh data/i }));

    await waitFor(() => {
      expect(apiMock.getSystemDependencies).toHaveBeenCalledTimes(2);
    });
  });

  // --- Uptime History tests ---

  it('fetches uptime stats for each system dependency after load', async () => {
    render(<DependenciesPage />);

    await screen.findByText('AuthNZ Database');

    await waitFor(() => {
      expect(apiMock.getDependencyUptime).toHaveBeenCalledTimes(5);
    });

    expect(apiMock.getDependencyUptime).toHaveBeenCalledWith('AuthNZ Database', 7);
    expect(apiMock.getDependencyUptime).toHaveBeenCalledWith('ChaChaNotes', 7);
    expect(apiMock.getDependencyUptime).toHaveBeenCalledWith('Embeddings Service', 7);
  });

  it('shows 7-day uptime percentage badges for system dependencies', async () => {
    render(<DependenciesPage />);

    // Wait for uptime badges to appear -- 100.0% appears for multiple deps + LLM sparklines
    await waitFor(() => {
      const allUptimeBadges = screen.getAllByText('100.0%');
      // At least 4 system deps with 100% uptime (all except Embeddings Service)
      expect(allUptimeBadges.length).toBeGreaterThanOrEqual(4);
    });
    expect(screen.getByText('89.3%')).toBeInTheDocument();
  });

  it('renders uptime sparkline SVGs for system dependencies', async () => {
    render(<DependenciesPage />);

    await screen.findByText('AuthNZ Database');

    await waitFor(() => {
      const sparklines = screen.getAllByRole('img', { name: /uptime sparkline/i });
      expect(sparklines.length).toBe(5);
    });
  });

  it('shows table headers for uptime columns', async () => {
    render(<DependenciesPage />);

    await screen.findByText('AuthNZ Database');

    expect(screen.getByText('7d Uptime')).toBeInTheDocument();
    expect(screen.getByText('Trend')).toBeInTheDocument();
  });

  it('gracefully handles uptime endpoint failures', async () => {
    apiMock.getDependencyUptime.mockRejectedValue(new Error('Not available'));

    render(<DependenciesPage />);

    // System deps should still render
    expect(await screen.findByText('AuthNZ Database')).toBeInTheDocument();

    // No uptime badges should appear, but no crash
    await waitFor(() => {
      expect(apiMock.getDependencyUptime).toHaveBeenCalled();
    });
  });

  it('surfaces uptime history failures instead of silently falling back to no data', async () => {
    apiMock.getDependenciesUptimeHistory.mockRejectedValue(new Error('uptime endpoint down'));

    render(<DependenciesPage />);

    expect(await screen.findByText(/Some dependency telemetry is unavailable:/i)).toBeInTheDocument();
    expect(screen.getByText('Uptime history unavailable: uptime endpoint down')).toBeInTheDocument();
  });
});
