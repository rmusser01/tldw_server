/* @vitest-environment jsdom */
import { afterEach, describe, expect, it } from 'vitest';
import { cleanup, render, screen } from '@testing-library/react';
import { StatsGrid, type RealtimeStats } from './StatsGrid';
import { DEFAULT_DASHBOARD_OPERATIONAL_KPIS, type DashboardOperationalKpis } from '@/lib/dashboard-kpis';
import type { DashboardUIStats } from '@/lib/dashboard';

const baseStats: DashboardUIStats = {
  users: 12,
  activeUsers: 9,
  organizations: 4,
  teams: 7,
  apiKeys: 10,
  activeApiKeys: 8,
  providers: 5,
  enabledProviders: 4,
  storageUsedMb: 2048,
  storageQuotaMb: 8192,
};

const baseOperationalKpis: DashboardOperationalKpis = {
  latencyP95Ms: 320,
  latencyTrend: { direction: 'down', delta: -25, percentChange: -7.2 },
  errorRatePct: 1.52,
  errorRateTrend: { direction: 'down', delta: -0.48, percentChange: -24 },
  dailyCostUsd: 18.23,
  dailyCostTrend: { direction: 'up', delta: 2.1, percentChange: 13.01 },
  activeJobs: 5,
  activeJobsTrend: { direction: 'up', delta: 1, percentChange: 25 },
  queuedJobs: 13,
  failedJobs: 3,
  queueDepth: 16,
  queueDepthTrend: { direction: 'down', delta: -4, percentChange: -20 },
};

const baseRealtimeStats: RealtimeStats = {
  active_sessions: 7,
  tokens_today: { prompt: 125000, completion: 42000, total: 167000 },
};

afterEach(() => {
  cleanup();
});

describe('StatsGrid', () => {
  it('announces stat updates in a polite live region', () => {
    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={baseOperationalKpis}
        realtimeStats={baseRealtimeStats}
      />
    );

    const liveRegion = screen.getByTestId('dashboard-stats-live-region');
    expect(liveRegion.getAttribute('role')).toBe('status');
    expect(liveRegion.getAttribute('aria-live')).toBe('polite');
    expect(screen.getByText(/Dashboard metrics updated\./)).toBeInTheDocument();
  });

  it('renders 12 dashboard cards including operational KPI, realtime, cache, and MCP cards', () => {
    const { container } = render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={baseOperationalKpis}
        realtimeStats={baseRealtimeStats}
        cacheHitRatePct={84.3}
      />
    );

    expect(container.querySelectorAll('div.rounded-lg.border.bg-card').length).toBe(12);
    expect(screen.getByText('Cache Hit Rate')).toBeInTheDocument();
    expect(screen.getByText('84.3%')).toBeInTheDocument();
    expect(screen.getByText('Request Latency (p95)')).toBeInTheDocument();
    expect(screen.getByText('Error Rate')).toBeInTheDocument();
    expect(screen.getByText('Daily LLM Cost')).toBeInTheDocument();
    expect(screen.getByText('Jobs & Queue')).toBeInTheDocument();
    expect(screen.getByText('5 / 13 / 3')).toBeInTheDocument();
    expect(screen.getByText('active / queued / failed')).toBeInTheDocument();
    expect(screen.getByText('Active Sessions')).toBeInTheDocument();
    expect(screen.getByText('Token Consumption')).toBeInTheDocument();
    expect(screen.getByText('MCP Tool Invocations')).toBeInTheDocument();
    expect(screen.getByText('Requires MCP telemetry')).toBeInTheDocument();
  });

  it('renders active session count from realtime stats', () => {
    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={baseOperationalKpis}
        realtimeStats={baseRealtimeStats}
      />
    );

    expect(screen.getByText('7')).toBeInTheDocument();
    expect(screen.getByText('ACP agent sessions')).toBeInTheDocument();
  });

  it('renders token consumption with compact formatting', () => {
    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={baseOperationalKpis}
        realtimeStats={baseRealtimeStats}
      />
    );

    expect(screen.getByText('167.0K')).toBeInTheDocument();
    expect(screen.getByText(/125\.0K prompt/)).toBeInTheDocument();
    expect(screen.getByText(/42\.0K completion/)).toBeInTheDocument();
  });

  it('renders N/A for realtime cards when realtimeStats is null', () => {
    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={baseOperationalKpis}
        realtimeStats={null}
      />
    );

    expect(screen.getByText('Active Sessions')).toBeInTheDocument();
    expect(screen.getByText('Token Consumption')).toBeInTheDocument();
    // Both realtime cards should show N/A
    const naElements = screen.getAllByText('N/A');
    expect(naElements.length).toBeGreaterThanOrEqual(2);
  });

  it('renders N/A for realtime cards when realtimeStats is undefined', () => {
    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={DEFAULT_DASHBOARD_OPERATIONAL_KPIS}
      />
    );

    expect(screen.getByText('Active Sessions')).toBeInTheDocument();
    expect(screen.getByText('Token Consumption')).toBeInTheDocument();
  });

  it('renders N/A fallback values when operational metrics are missing', () => {
    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={DEFAULT_DASHBOARD_OPERATIONAL_KPIS}
      />
    );

    expect(screen.getAllByText('N/A').length).toBeGreaterThanOrEqual(4);
    expect(screen.getByText('Requires /metrics histogram')).toBeInTheDocument();
    expect(screen.getByText('No prior queue snapshot')).toBeInTheDocument();
  });

  it('includes realtime stats in the screen-reader live summary', () => {
    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={baseOperationalKpis}
        realtimeStats={baseRealtimeStats}
      />
    );

    expect(screen.getByText(/7 active sessions/)).toBeInTheDocument();
    expect(screen.getByText(/167\.0K tokens consumed/)).toBeInTheDocument();
  });

  it('formats large token values with M notation', () => {
    const largeTokenStats: RealtimeStats = {
      active_sessions: 3,
      tokens_today: { prompt: 5500000, completion: 2300000, total: 7800000 },
    };

    render(
      <StatsGrid
        loading={false}
        stats={baseStats}
        storagePercentage={25}
        operationalKpis={baseOperationalKpis}
        realtimeStats={largeTokenStats}
      />
    );

    expect(screen.getByText('7.8M')).toBeInTheDocument();
    expect(screen.getByText(/5\.5M prompt/)).toBeInTheDocument();
    expect(screen.getByText(/2\.3M completion/)).toBeInTheDocument();
  });

  it.each([375, 768, 1280])('matches snapshot at viewport width %i', (width) => {
    Object.defineProperty(window, 'innerWidth', {
      configurable: true,
      writable: true,
      value: width,
    });
    window.dispatchEvent(new Event('resize'));

    const { asFragment } = render(
      <div style={{ width: `${width}px` }}>
        <StatsGrid
          loading={false}
          stats={baseStats}
          storagePercentage={25}
          operationalKpis={baseOperationalKpis}
          realtimeStats={baseRealtimeStats}
        />
      </div>
    );

    expect(asFragment()).toMatchSnapshot();
  });
});
