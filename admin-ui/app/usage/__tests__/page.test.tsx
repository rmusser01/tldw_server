/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UsagePage from '../page';
import {
  getRouterAnalyticsAccess,
  getRouterAnalyticsConversations,
  getRouterAnalyticsLog,
  getRouterAnalyticsMeta,
  getRouterAnalyticsModels,
  getRouterAnalyticsNetwork,
  getRouterAnalyticsProviders,
  getRouterAnalyticsQuota,
  getRouterAnalyticsStatus,
  getRouterAnalyticsStatusBreakdowns,
} from '@/lib/router-analytics-client';

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => <div data-testid="layout">{children}</div>,
}));

vi.mock('@/lib/router-analytics-client', () => ({
  getRouterAnalyticsStatus: vi.fn(),
  getRouterAnalyticsStatusBreakdowns: vi.fn(),
  getRouterAnalyticsQuota: vi.fn(),
  getRouterAnalyticsProviders: vi.fn(),
  getRouterAnalyticsAccess: vi.fn(),
  getRouterAnalyticsNetwork: vi.fn(),
  getRouterAnalyticsModels: vi.fn(),
  getRouterAnalyticsConversations: vi.fn(),
  getRouterAnalyticsLog: vi.fn(),
  getRouterAnalyticsMeta: vi.fn(),
}));

type RouterClientMock = {
  getRouterAnalyticsStatus: ReturnType<typeof vi.fn>;
  getRouterAnalyticsStatusBreakdowns: ReturnType<typeof vi.fn>;
  getRouterAnalyticsQuota: ReturnType<typeof vi.fn>;
  getRouterAnalyticsProviders: ReturnType<typeof vi.fn>;
  getRouterAnalyticsAccess: ReturnType<typeof vi.fn>;
  getRouterAnalyticsNetwork: ReturnType<typeof vi.fn>;
  getRouterAnalyticsModels: ReturnType<typeof vi.fn>;
  getRouterAnalyticsConversations: ReturnType<typeof vi.fn>;
  getRouterAnalyticsLog: ReturnType<typeof vi.fn>;
  getRouterAnalyticsMeta: ReturnType<typeof vi.fn>;
};

const routerClientMock = {
  getRouterAnalyticsStatus: getRouterAnalyticsStatus,
  getRouterAnalyticsStatusBreakdowns: getRouterAnalyticsStatusBreakdowns,
  getRouterAnalyticsQuota: getRouterAnalyticsQuota,
  getRouterAnalyticsProviders: getRouterAnalyticsProviders,
  getRouterAnalyticsAccess: getRouterAnalyticsAccess,
  getRouterAnalyticsNetwork: getRouterAnalyticsNetwork,
  getRouterAnalyticsModels: getRouterAnalyticsModels,
  getRouterAnalyticsConversations: getRouterAnalyticsConversations,
  getRouterAnalyticsLog: getRouterAnalyticsLog,
  getRouterAnalyticsMeta: getRouterAnalyticsMeta,
} as unknown as RouterClientMock;

beforeEach(() => {
  routerClientMock.getRouterAnalyticsStatus.mockResolvedValue({
    kpis: {
      requests: 4,
      prompt_tokens: 52,
      generated_tokens: 32,
      total_tokens: 84,
      avg_latency_ms: 500,
      avg_gen_toks_per_s: 16,
    },
    series: [
      {
        ts: '2026-03-01T10:20:00Z',
        provider: 'openai',
        model: 'gpt-4o-mini',
        requests: 1,
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
      },
      {
        ts: '2026-03-01T10:25:00Z',
        provider: 'groq',
        model: 'llama-3.3-70b',
        requests: 2,
        prompt_tokens: 35,
        completion_tokens: 10,
        total_tokens: 45,
      },
    ],
    providers_available: 3,
    providers_online: 2,
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsStatusBreakdowns.mockResolvedValue({
    providers: [
      { key: 'groq', label: 'groq', requests: 2, prompt_tokens: 35, completion_tokens: 10, total_tokens: 45, errors: 1 },
      { key: 'openai', label: 'openai', requests: 1, prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, errors: 0 },
    ],
    models: [
      { key: 'llama-3.3-70b', label: 'llama-3.3-70b', requests: 2, prompt_tokens: 35, completion_tokens: 10, total_tokens: 45, errors: 1 },
      { key: 'gpt-4o-mini', label: 'gpt-4o-mini', requests: 1, prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, errors: 0 },
    ],
    token_names: [
      { key: 'Admin', label: 'Admin', requests: 1, prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, errors: 0 },
      { key: 'Ops', label: 'Ops', requests: 2, prompt_tokens: 35, completion_tokens: 10, total_tokens: 45, errors: 1 },
    ],
    remote_ips: [
      { key: '127.0.0.1', label: '127.0.0.1', requests: 1, prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, errors: 0 },
      { key: 'unknown', label: 'unknown', requests: 1, prompt_tokens: 7, completion_tokens: 2, total_tokens: 9, errors: 1 },
    ],
    user_agents: [
      { key: 'curl/8.8.0', label: 'curl/8.8.0', requests: 1, prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, errors: 0 },
      { key: 'unknown', label: 'unknown', requests: 1, prompt_tokens: 7, completion_tokens: 2, total_tokens: 9, errors: 1 },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsMeta.mockResolvedValue({
    providers: [{ value: 'openai', label: 'openai' }, { value: 'groq', label: 'groq' }],
    models: [{ value: 'gpt-4o-mini', label: 'gpt-4o-mini' }, { value: 'llama-3.3-70b', label: 'llama-3.3-70b' }],
    tokens: [
      { value: '11', label: 'Admin', key_id: 11 },
      { value: '12', label: 'Ops', key_id: 12 },
    ],
    ranges: ['realtime', '1h', '8h', '24h', '7d', '30d'],
    granularities: ['1m', '5m', '15m', '1h'],
    generated_at: '2026-03-01T10:30:00Z',
  });

  routerClientMock.getRouterAnalyticsQuota.mockResolvedValue({
    summary: {
      keys_total: 3,
      keys_over_budget: 1,
      budgeted_keys: 2,
    },
    items: [
      {
        key_id: 12,
        token_name: 'Ops',
        requests: 2,
        total_tokens: 45,
        total_cost_usd: 0.04,
        day_tokens: { used: 45, limit: 30, utilization_pct: 150, exceeded: true },
        month_tokens: { used: 45, limit: 100, utilization_pct: 45, exceeded: false },
        day_usd: { used: 0.04, limit: 0.05, utilization_pct: 80, exceeded: false },
        month_usd: { used: 0.04, limit: 1.0, utilization_pct: 4, exceeded: false },
        over_budget: true,
        reasons: ['day_tokens_exceeded:45/30'],
        last_seen_at: '2026-03-01T10:25:00Z',
      },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsProviders.mockResolvedValue({
    summary: {
      providers_total: 3,
      providers_online: 2,
      failover_events: 2,
    },
    items: [
      {
        provider: 'groq',
        requests: 2,
        prompt_tokens: 35,
        completion_tokens: 10,
        total_tokens: 45,
        total_cost_usd: 0.04,
        avg_latency_ms: 350,
        errors: 1,
        success_rate_pct: 50,
        online: true,
      },
      {
        provider: 'openai',
        requests: 1,
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        total_cost_usd: 0.03,
        avg_latency_ms: 1000,
        errors: 0,
        success_rate_pct: 100,
        online: true,
      },
      {
        provider: 'anthropic',
        requests: 1,
        prompt_tokens: 7,
        completion_tokens: 2,
        total_tokens: 9,
        total_cost_usd: 0,
        avg_latency_ms: 300,
        errors: 1,
        success_rate_pct: 0,
        online: false,
      },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsAccess.mockResolvedValue({
    summary: {
      token_names_total: 3,
      remote_ips_total: 2,
      user_agents_total: 2,
      anonymous_requests: 1,
    },
    token_names: [
      { key: 'Ops', label: 'Ops', requests: 2, prompt_tokens: 35, completion_tokens: 10, total_tokens: 45, errors: 1 },
      { key: 'Admin', label: 'Admin', requests: 1, prompt_tokens: 10, completion_tokens: 20, total_tokens: 30, errors: 0 },
      { key: 'unknown', label: 'unknown', requests: 1, prompt_tokens: 7, completion_tokens: 2, total_tokens: 9, errors: 1 },
    ],
    remote_ips: [
      { key: '10.0.0.5', label: '10.0.0.5', requests: 2, prompt_tokens: 35, completion_tokens: 10, total_tokens: 45, errors: 1 },
      { key: 'unknown', label: 'unknown', requests: 1, prompt_tokens: 7, completion_tokens: 2, total_tokens: 9, errors: 1 },
    ],
    user_agents: [
      { key: 'python-httpx/1.0', label: 'python-httpx/1.0', requests: 2, prompt_tokens: 35, completion_tokens: 10, total_tokens: 45, errors: 1 },
      { key: 'unknown', label: 'unknown', requests: 1, prompt_tokens: 7, completion_tokens: 2, total_tokens: 9, errors: 1 },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsNetwork.mockResolvedValue({
    summary: {
      remote_ips_total: 2,
      endpoints_total: 1,
      operations_total: 1,
      error_requests: 2,
    },
    remote_ips: [
      { key: '10.0.0.5', label: '10.0.0.5', requests: 2, prompt_tokens: 35, completion_tokens: 10, total_tokens: 45, errors: 1 },
      { key: 'unknown', label: 'unknown', requests: 1, prompt_tokens: 7, completion_tokens: 2, total_tokens: 9, errors: 1 },
    ],
    endpoints: [
      { key: '/api/v1/chat/completions', label: '/api/v1/chat/completions', requests: 4, prompt_tokens: 52, completion_tokens: 32, total_tokens: 84, errors: 2 },
    ],
    operations: [
      { key: 'chat', label: 'chat', requests: 4, prompt_tokens: 52, completion_tokens: 32, total_tokens: 84, errors: 2 },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsModels.mockResolvedValue({
    summary: {
      models_total: 3,
      models_online: 2,
      providers_total: 3,
      error_requests: 2,
    },
    items: [
      {
        model: 'llama-3.3-70b',
        provider: 'groq',
        requests: 2,
        prompt_tokens: 35,
        completion_tokens: 10,
        total_tokens: 45,
        total_cost_usd: 0.04,
        avg_latency_ms: 350,
        errors: 1,
        success_rate_pct: 50,
        online: true,
      },
      {
        model: 'gpt-4o-mini',
        provider: 'openai',
        requests: 1,
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        total_cost_usd: 0.03,
        avg_latency_ms: 1000,
        errors: 0,
        success_rate_pct: 100,
        online: true,
      },
      {
        model: 'claude-3.5',
        provider: 'anthropic',
        requests: 1,
        prompt_tokens: 7,
        completion_tokens: 2,
        total_tokens: 9,
        total_cost_usd: 0,
        avg_latency_ms: 300,
        errors: 1,
        success_rate_pct: 0,
        online: false,
      },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsConversations.mockResolvedValue({
    summary: {
      conversations_total: 3,
      active_conversations: 2,
      avg_requests_per_conversation: 1.3,
      error_requests: 2,
    },
    items: [
      {
        conversation_id: 'conv-2',
        requests: 2,
        prompt_tokens: 35,
        completion_tokens: 10,
        total_tokens: 45,
        total_cost_usd: 0.04,
        avg_latency_ms: 350,
        errors: 1,
        success_rate_pct: 50,
        last_seen_at: '2026-03-01T10:26:00Z',
      },
      {
        conversation_id: 'conv-1',
        requests: 1,
        prompt_tokens: 10,
        completion_tokens: 20,
        total_tokens: 30,
        total_cost_usd: 0.03,
        avg_latency_ms: 1000,
        errors: 0,
        success_rate_pct: 100,
        last_seen_at: '2026-03-01T10:20:00Z',
      },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });

  routerClientMock.getRouterAnalyticsLog.mockResolvedValue({
    summary: {
      requests_total: 4,
      error_requests: 2,
      estimated_requests: 2,
      request_ids_total: 4,
    },
    items: [
      {
        ts: '2026-03-01T10:27:00Z',
        request_id: 'req-4',
        conversation_id: 'conv-3',
        provider: 'anthropic',
        model: 'claude-3.5',
        token_name: 'unknown',
        endpoint: '/api/v1/chat/completions',
        operation: 'chat',
        status: 503,
        latency_ms: 300,
        prompt_tokens: 7,
        completion_tokens: 2,
        total_tokens: 9,
        total_cost_usd: 0,
        remote_ip: 'unknown',
        user_agent: 'unknown',
        estimated: true,
        error: true,
      },
      {
        ts: '2026-03-01T10:26:00Z',
        request_id: 'req-3',
        conversation_id: 'conv-2',
        provider: 'groq',
        model: 'llama-3.3-70b',
        token_name: 'Ops',
        endpoint: '/api/v1/chat/completions',
        operation: 'chat',
        status: 500,
        latency_ms: 200,
        prompt_tokens: 5,
        completion_tokens: 0,
        total_tokens: 5,
        total_cost_usd: 0,
        remote_ip: '10.0.0.5',
        user_agent: 'python-httpx/1.0',
        estimated: true,
        error: true,
      },
    ],
    generated_at: '2026-03-01T10:30:00Z',
    data_window: {
      start: '2026-03-01T09:30:00Z',
      end: '2026-03-01T10:30:00Z',
      range: '1h',
    },
    partial: false,
  });
});

afterEach(() => {
  cleanup();
  vi.clearAllMocks();
});

describe('UsagePage router analytics status shell', () => {
  it('renders status cards, timeline, and breakdown tables', async () => {
    render(<UsagePage />);

    expect(await screen.findByRole('heading', { name: 'Usage Stats' })).toBeInTheDocument();
    expect(screen.getByText('Providers available: 3 • Online: 2')).toBeInTheDocument();

    expect(screen.getAllByText('Requests').length).toBeGreaterThan(0);
    expect(screen.getByText('4')).toBeInTheDocument();
    expect(screen.getByText('Prompt / Generated')).toBeInTheDocument();
    expect(screen.getByText('52 / 32')).toBeInTheDocument();

    expect(screen.getByText('Usage by model (tokens / bucket)')).toBeInTheDocument();
    expect(screen.getAllByText('llama-3.3-70b').length).toBeGreaterThan(0);

    expect(screen.getByRole('heading', { name: 'Providers' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Models' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Token Names' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Remote IPs' })).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'User Agents' })).toBeInTheDocument();
  });

  it('re-fetches with new range selection', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    const rangeSelect = screen.getByLabelText('Time Range');
    await user.selectOptions(rangeSelect, '24h');

    await waitFor(() => {
      expect(routerClientMock.getRouterAnalyticsStatus).toHaveBeenLastCalledWith(
        expect.objectContaining({ range: '24h' })
      );
    });
  });

  it('uses selected token key id for router analytics filters', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await screen.findByRole('option', { name: 'Ops' });
    const tokenSelect = screen.getByLabelText('Token');
    await user.selectOptions(tokenSelect, '12');

    await waitFor(() => {
      expect(routerClientMock.getRouterAnalyticsStatus).toHaveBeenLastCalledWith(
        expect.objectContaining({ tokenId: 12 })
      );
    });
  });

  it('renders log tab data from router analytics log endpoint', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Log/i }));

    expect(await screen.findByText('Request log')).toBeInTheDocument();
    expect(screen.getByText('Estimated requests')).toBeInTheDocument();
    expect(screen.getByText('req-4')).toBeInTheDocument();
  });

  it('renders quota tab data from router analytics quota endpoint', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Quota/i }));

    expect(await screen.findByText('Quota utilization')).toBeInTheDocument();
    expect(screen.getByText('Keys over budget')).toBeInTheDocument();
    expect(screen.getAllByText('Ops').length).toBeGreaterThan(0);
    expect(screen.getByText(/150\.0%/)).toBeInTheDocument();
  });

  it('renders providers tab data from router analytics providers endpoint', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Providers/i }));

    expect(await screen.findByText('Provider health and load')).toBeInTheDocument();
    expect(screen.getByText('Failover events')).toBeInTheDocument();
    expect(screen.getByText('50.0%')).toBeInTheDocument();
    expect(screen.getByText('Offline')).toBeInTheDocument();
  });

  it('exposes PP and TG headers with accessible names and hides the decorative tab separator', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Providers/i }));

    expect(await screen.findByRole('columnheader', { name: 'Prompt Tokens' })).toBeInTheDocument();
    expect(screen.getByRole('columnheader', { name: 'Total Generated Tokens' })).toBeInTheDocument();

    const tabList = screen.getByRole('tablist');
    const separator = tabList.querySelector('span[aria-hidden="true"]');
    expect(separator).toBeInTheDocument();
    expect(separator?.getAttribute('aria-hidden')).toBe('true');
  });

  it('renders access tab data from router analytics access endpoint', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Access/i }));

    expect(await screen.findByText('Anonymous requests')).toBeInTheDocument();
    expect(screen.getByText('Token Names (Access)')).toBeInTheDocument();
    expect(screen.getByText('Remote IPs (Access)')).toBeInTheDocument();
    expect(screen.getByText('User Agents (Access)')).toBeInTheDocument();
  });

  it('renders network tab data from router analytics network endpoint', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Network/i }));

    expect(await screen.findByText('Error requests')).toBeInTheDocument();
    expect(screen.getByText('Remote IPs (Network)')).toBeInTheDocument();
    expect(screen.getByText('Endpoints (Network)')).toBeInTheDocument();
    expect(screen.getByText('Operations (Network)')).toBeInTheDocument();
  });

  it('renders models tab data from router analytics models endpoint', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Models/i }));

    expect(await screen.findByText('Providers covered')).toBeInTheDocument();
    expect(screen.getByText('Model health and load')).toBeInTheDocument();
    expect(screen.getByText('claude-3.5')).toBeInTheDocument();
  });

  it('renders conversations tab data from router analytics conversations endpoint', async () => {
    const user = userEvent.setup();
    render(<UsagePage />);

    await screen.findByRole('heading', { name: 'Usage Stats' });
    await user.click(screen.getByRole('tab', { name: /Conversations/i }));

    expect(await screen.findByText('Active conversations')).toBeInTheDocument();
    expect(screen.getByText('Conversation activity')).toBeInTheDocument();
    expect(screen.getByText('conv-2')).toBeInTheDocument();
  });
});
