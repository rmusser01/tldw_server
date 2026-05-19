/* @vitest-environment jsdom */
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import MonitoringPageHeader from './MonitoringPageHeader';

describe('MonitoringPageHeader', () => {
  afterEach(() => {
    cleanup();
  });

  it('renders heading content and refreshes on click', () => {
    const onRefresh = vi.fn();
    const now = new Date();

    render(
      <MonitoringPageHeader
        lastUpdated={new Date('2026-03-01T10:15:00.000Z')}
        loading={false}
        onRefresh={onRefresh}
        lastRefreshed={now}
        autoRefreshEnabled
        onAutoRefreshToggle={vi.fn()}
      />
    );

    expect(screen.getByRole('heading', { name: 'Monitoring' })).toBeInTheDocument();
    expect(screen.getByText('System health, metrics, and alerts')).toBeInTheDocument();
    expect(screen.getByTestId('last-updated-label')).toBeInTheDocument();
    expect(screen.getByTestId('auto-refresh-toggle')).toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Refresh' }));
    expect(onRefresh).toHaveBeenCalledTimes(1);
  });

  it('disables refresh while loading', () => {
    render(
      <MonitoringPageHeader
        lastUpdated={null}
        loading
        onRefresh={vi.fn()}
      />
    );

    expect(screen.getByRole('button', { name: 'Refresh' })).toBeDisabled();
    expect(screen.queryByText(/Last updated:/)).not.toBeInTheDocument();
  });
});
