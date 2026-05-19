/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { ActivitySection } from './ActivitySection';
import type { DashboardSystemHealth } from '@/lib/dashboard-health';

vi.mock('next/link', () => ({
  default: ({ href, children, ...props }: { href: string; children: ReactNode }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

vi.mock('recharts', () => {
  const MockContainer = ({ children }: { children?: ReactNode }) => <div>{children}</div>;
  return {
    ResponsiveContainer: MockContainer,
    AreaChart: MockContainer,
    Area: () => <div />,
    Line: () => <div />,
    CartesianGrid: () => <div />,
    XAxis: () => <div />,
    YAxis: () => <div />,
    Tooltip: () => <div />,
  };
});

afterEach(() => {
  cleanup();
});

const activityData = [
  { name: 'Mon', requests: 1200, users: 120 },
  { name: 'Tue', requests: 1400, users: 136 },
];

describe('ActivitySection', () => {
  it('renders the expanded 8-subsystem health grid with timestamps', () => {
    const systemHealth: DashboardSystemHealth = {
      api: { status: 'healthy', checkedAt: '2026-02-17T10:00:00.000Z' },
      database: { status: 'healthy', checkedAt: '2026-02-17T10:00:01.000Z' },
      llm: { status: 'degraded', checkedAt: '2026-02-17T10:00:02.000Z' },
      rag: { status: 'healthy', checkedAt: '2026-02-17T10:00:03.000Z' },
      tts: { status: 'down', checkedAt: '2026-02-17T10:00:04.000Z' },
      stt: { status: 'healthy', checkedAt: '2026-02-17T10:00:05.000Z' },
      embeddings: { status: 'degraded', checkedAt: '2026-02-17T10:00:06.000Z' },
      cache: {
        status: 'healthy',
        checkedAt: '2026-02-17T10:00:07.000Z',
        cacheHitRatePct: 84.3,
      },
      jobQueue: {
        status: 'healthy',
        checkedAt: '2026-02-17T10:00:08.000Z',
      },
    };

    render(
      <ActivitySection
        activityChartData={activityData}
        systemHealth={systemHealth}
        activityRange="7d"
        onActivityRangeChange={vi.fn()}
      />
    );

    expect(screen.getByText('API Server')).toBeInTheDocument();
    expect(screen.getByText('Database')).toBeInTheDocument();
    expect(screen.getByText('LLM Services')).toBeInTheDocument();
    expect(screen.getByText('RAG Service')).toBeInTheDocument();
    expect(screen.getByText('TTS Service')).toBeInTheDocument();
    expect(screen.getByText('STT Service')).toBeInTheDocument();
    expect(screen.getByText('Embeddings')).toBeInTheDocument();
    expect(screen.getByText('RAG Cache')).toBeInTheDocument();
    expect(screen.getByText('Job Queue')).toBeInTheDocument();
    expect(screen.getByText('Cache hit rate: 84.3%')).toBeInTheDocument();
    expect(screen.getAllByText(/Last checked:/).length).toBe(9);
  });

  it('keeps health rows visible when some subsystem checks fail', () => {
    const systemHealth: DashboardSystemHealth = {
      api: { status: 'healthy', checkedAt: '2026-02-17T10:00:00.000Z' },
      database: { status: 'healthy', checkedAt: '2026-02-17T10:00:01.000Z' },
      llm: { status: 'down', checkedAt: '2026-02-17T10:05:00.000Z' },
      rag: { status: 'down', checkedAt: '2026-02-17T10:05:00.000Z' },
      tts: { status: 'down', checkedAt: '2026-02-17T10:05:00.000Z' },
      stt: { status: 'down', checkedAt: '2026-02-17T10:05:00.000Z' },
      embeddings: { status: 'down', checkedAt: '2026-02-17T10:05:00.000Z' },
      cache: { status: 'down', checkedAt: '2026-02-17T10:05:00.000Z' },
      jobQueue: { status: 'down', checkedAt: '2026-02-17T10:05:00.000Z', errorMessage: '5 quarantined job(s)' },
    };

    render(
      <ActivitySection
        activityChartData={activityData}
        systemHealth={systemHealth}
        activityRange="7d"
        onActivityRangeChange={vi.fn()}
      />
    );

    expect(screen.getAllByText('Down').length).toBeGreaterThanOrEqual(7);
    expect(screen.getByText('5 quarantined job(s)')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'View Details' })).toBeInTheDocument();
  });

  it('calls range change handler when selecting a different activity range', () => {
    const onActivityRangeChange = vi.fn();
    const systemHealth: DashboardSystemHealth = {
      api: { status: 'healthy', checkedAt: '2026-02-17T10:00:00.000Z' },
      database: { status: 'healthy', checkedAt: '2026-02-17T10:00:01.000Z' },
      llm: { status: 'healthy', checkedAt: '2026-02-17T10:00:02.000Z' },
      rag: { status: 'healthy', checkedAt: '2026-02-17T10:00:03.000Z' },
      tts: { status: 'healthy', checkedAt: '2026-02-17T10:00:04.000Z' },
      stt: { status: 'healthy', checkedAt: '2026-02-17T10:00:05.000Z' },
      embeddings: { status: 'healthy', checkedAt: '2026-02-17T10:00:06.000Z' },
      cache: { status: 'healthy', checkedAt: '2026-02-17T10:00:07.000Z' },
      jobQueue: { status: 'healthy', checkedAt: '2026-02-17T10:00:08.000Z' },
    };

    render(
      <ActivitySection
        activityChartData={activityData}
        systemHealth={systemHealth}
        activityRange="7d"
        onActivityRangeChange={onActivityRangeChange}
      />
    );

    fireEvent.click(screen.getByRole('button', { name: '24h' }));
    expect(onActivityRangeChange).toHaveBeenCalledWith('24h');
  });
});
