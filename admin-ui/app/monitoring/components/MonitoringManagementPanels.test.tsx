/* @vitest-environment jsdom */
import { render, screen } from '@testing-library/react';
import { describe, expect, it, vi } from 'vitest';
import MonitoringManagementPanels from './MonitoringManagementPanels';

vi.mock('./AlertRulesPanel', () => ({
  default: ({
    rules,
  }: {
    rules: Array<{ id: string }>;
  }) => <div data-testid="alert-rules-panel">{`rules:${rules.length}`}</div>,
}));

vi.mock('./AlertsPanel', () => ({
  default: ({
    alerts,
  }: {
    alerts: Array<{ id: string }>;
  }) => <div data-testid="alerts-panel">{`alerts:${alerts.length}`}</div>,
}));

vi.mock('./WatchlistsPanel', () => ({
  default: ({
    watchlists,
  }: {
    watchlists: Array<{ id: string }>;
  }) => <div data-testid="watchlists-panel">{`watchlists:${watchlists.length}`}</div>,
}));

vi.mock('./NotificationsPanel', () => ({
  default: ({
    recentNotifications,
  }: {
    recentNotifications: Array<{ id: string }>;
  }) => (
    <div data-testid="notifications-panel">
      {`notifications:${recentNotifications.length}`}
    </div>
  ),
}));

vi.mock('./ErrorBreakdownPanel', () => ({
  default: () => <div data-testid="error-breakdown-panel">error-breakdown</div>,
}));

vi.mock('./SystemStatusPanel', () => ({
  default: ({
    systemStatus,
  }: {
    systemStatus: Array<{ key: string }>;
  }) => <div data-testid="system-status-panel">{`status:${systemStatus.length}`}</div>,
}));

describe('MonitoringManagementPanels', () => {
  it('renders all management panels without the local-only rules disclaimer', () => {
    render(
      <MonitoringManagementPanels
        alertRulesPanelProps={{
          rules: [{ id: 'rule-1' } as never],
          draft: {
            metric: 'cpu',
            operator: '>',
            threshold: '90',
            durationMinutes: '5',
            severity: 'critical',
          },
          errors: {},
          saving: false,
          mutationsEnabled: true,
          onDraftChange: vi.fn(),
          onCreateRule: vi.fn(),
          onDeleteRule: vi.fn(),
        }}
        alertsPanelProps={{
          alerts: [{ id: 'alert-1' } as never],
          history: [],
          showSnoozed: false,
          assignableUsers: [],
          loading: false,
          onToggleShowSnoozed: vi.fn(),
          onAcknowledge: vi.fn(),
          onDismiss: vi.fn(),
          onAssign: vi.fn(),
          onSnooze: vi.fn(),
          onEscalate: vi.fn(),
          localActionsEnabled: true,
        }}
        watchlistsPanelProps={{
          watchlists: [{ id: 'watchlist-1' } as never],
          loading: false,
          showCreateWatchlist: false,
          setShowCreateWatchlist: vi.fn(),
          newWatchlist: {
            name: '',
            description: '',
            target: '',
            type: 'resource',
            threshold: 80,
          },
          setNewWatchlist: vi.fn(),
          onCreate: vi.fn(),
          onDelete: vi.fn(),
          deletingWatchlistId: null,
        }}
        notificationsPanelProps={{
          settings: null,
          recentNotifications: [{ id: 'notification-1' } as never],
          loading: false,
          saving: false,
          canSave: false,
          onSave: vi.fn(),
          onTest: vi.fn(),
        }}
        systemStatusPanelProps={{
          systemStatus: [{ key: 'api' } as never],
        }}
      />
    );

    expect(screen.getByTestId('alert-rules-panel').textContent).toBe('rules:1');
    expect(screen.getByTestId('alerts-panel').textContent).toBe('alerts:1');
    expect(screen.getByTestId('watchlists-panel').textContent).toBe('watchlists:1');
    expect(screen.getByTestId('error-breakdown-panel')).toBeInTheDocument();
    expect(screen.getByTestId('notifications-panel').textContent).toBe('notifications:1');
    expect(screen.getByTestId('system-status-panel').textContent).toBe('status:1');
    expect(
      screen.queryByText('Alert rules are stored locally until a backend alert-rules endpoint is available.')
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText('Alert rule editing is unavailable until a backend alert-rules endpoint is available.')
    ).not.toBeInTheDocument();
    expect(
      screen.queryByText('Alert assignment, snoozing, and escalation are unavailable until backend alert mutation endpoints are available.')
    ).not.toBeInTheDocument();
  });
});
