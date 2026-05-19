/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, fireEvent } from '@testing-library/react';
import { AlertsBanner, summarizeAlertSeverities, findMostRelevantAlert } from './AlertsBanner';

vi.mock('next/link', () => ({
  default: ({ href, children, ...props }: { href: string; children: ReactNode }) => (
    <a href={href} {...props}>
      {children}
    </a>
  ),
}));

afterEach(() => {
  cleanup();
});

describe('summarizeAlertSeverities', () => {
  it('groups alerts into critical, warning, and info buckets', () => {
    const summary = summarizeAlertSeverities([
      { severity: 'critical' },
      { severity: 'warning' },
      { severity: 'error' },
      { severity: 'info' },
      {},
    ]);

    expect(summary).toEqual({
      critical: 2,
      warning: 1,
      info: 2,
    });
  });
});

describe('findMostRelevantAlert', () => {
  it('returns null when no alerts have messages', () => {
    expect(findMostRelevantAlert([{ severity: 'critical' }])).toBeNull();
  });

  it('returns the highest-severity alert with a message', () => {
    const result = findMostRelevantAlert([
      { severity: 'info', message: 'Info msg' },
      { severity: 'critical', message: 'Critical msg' },
      { severity: 'warning', message: 'Warning msg' },
    ]);
    expect(result?.message).toBe('Critical msg');
  });

  it('prefers more recent alert when severity is equal', () => {
    const result = findMostRelevantAlert([
      { severity: 'critical', message: 'Older', created_at: '2026-01-01T00:00:00Z' },
      { severity: 'critical', message: 'Newer', created_at: '2026-03-01T00:00:00Z' },
    ]);
    expect(result?.message).toBe('Newer');
  });
});

describe('AlertsBanner', () => {
  it('renders nothing when there are no active alerts', () => {
    const { container } = render(<AlertsBanner alerts={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows severity breakdown and critical red styling for mixed alerts', () => {
    render(
      <AlertsBanner
        alerts={[
          { severity: 'critical' },
          { severity: 'warning' },
          { severity: 'info' },
          { severity: 'error' },
        ]}
      />
    );

    expect(screen.getByText('2 critical')).toBeInTheDocument();
    expect(screen.getByText('1 warning')).toBeInTheDocument();
    expect(screen.getByText('1 info')).toBeInTheDocument();
    expect(screen.getByText('4 active alerts require attention.')).toBeInTheDocument();
    expect(screen.getByRole('alert').className).toContain('bg-red-50');
  });

  it('displays the most relevant alert message inline', () => {
    render(
      <AlertsBanner
        alerts={[
          { severity: 'info', message: 'Low priority' },
          { severity: 'critical', message: 'Database connection pool exhausted' },
        ]}
      />
    );

    expect(screen.getByTestId('top-alert-message').textContent).toBe(
      'Database connection pool exhausted'
    );
  });

  it('renders Acknowledge All button when onAcknowledge is provided', () => {
    const handleAck = vi.fn();
    render(
      <AlertsBanner
        alerts={[{ severity: 'warning', id: '1' }]}
        onAcknowledge={handleAck}
      />
    );

    const btn = screen.getByTestId('acknowledge-alerts-btn');
    expect(btn).toBeInTheDocument();
    fireEvent.click(btn);
    expect(handleAck).toHaveBeenCalledTimes(1);
  });

  it('does not render Acknowledge All button when onAcknowledge is not provided', () => {
    render(
      <AlertsBanner alerts={[{ severity: 'warning' }]} />
    );

    expect(screen.queryByTestId('acknowledge-alerts-btn')).toBeNull();
  });
});
