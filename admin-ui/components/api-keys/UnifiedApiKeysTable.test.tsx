/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, describe, expect, it, vi } from 'vitest';
import { cleanup, fireEvent, render, screen } from '@testing-library/react';
import { UnifiedApiKeysTable } from './UnifiedApiKeysTable';

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

describe('UnifiedApiKeysTable', () => {
  it('renders mixed key statuses in unified key table', () => {
    render(
      <UnifiedApiKeysTable
        rows={[
          {
            keyId: '101',
            keyPrefix: 'sk-active',
            ownerUserId: 1,
            ownerUsername: 'alice',
            ownerEmail: 'alice@example.com',
            createdAt: '2026-02-17T00:00:00Z',
            lastUsedAt: '2026-02-17T01:00:00Z',
            expiresAt: null,
            status: 'active',
            requestCount24h: 10,
            errorRate24h: 0.1,
          },
          {
            keyId: '102',
            keyPrefix: 'sk-revoked',
            ownerUserId: 2,
            ownerUsername: 'bob',
            ownerEmail: 'bob@example.com',
            createdAt: '2026-02-16T00:00:00Z',
            lastUsedAt: null,
            expiresAt: null,
            status: 'revoked',
            requestCount24h: null,
            errorRate24h: null,
          },
          {
            keyId: '103',
            keyPrefix: 'sk-expired',
            ownerUserId: 3,
            ownerUsername: 'carol',
            ownerEmail: 'carol@example.com',
            createdAt: '2026-02-15T00:00:00Z',
            lastUsedAt: null,
            expiresAt: '2026-02-16T00:00:00Z',
            status: 'expired',
            requestCount24h: null,
            errorRate24h: null,
          },
        ]}
      />
    );

    expect(screen.getByText('Key ID')).toBeInTheDocument();
    expect(screen.getByText('Owner')).toBeInTheDocument();
    expect(screen.getByText('Age')).toBeInTheDocument();
    expect(screen.getByText('Expiry')).toBeInTheDocument();
    expect(screen.getByText('Activity')).toBeInTheDocument();
    // Telemetry columns hidden when all values are null
    expect(screen.queryByText('Requests (24h)')).not.toBeInTheDocument();
    expect(screen.queryByText('Error Rate (24h)')).not.toBeInTheDocument();
    expect(screen.getByText('Active')).toBeInTheDocument();
    expect(screen.getByText('Revoked')).toBeInTheDocument();
    expect(screen.getAllByText('Expired').length).toBeGreaterThanOrEqual(1);
  });

  it('shows telemetry columns when data is available', () => {
    render(
      <UnifiedApiKeysTable
        rows={[
          {
            keyId: '101',
            keyPrefix: 'sk-active',
            ownerUserId: 1,
            ownerUsername: 'alice',
            ownerEmail: 'alice@example.com',
            createdAt: '2026-02-17T00:00:00Z',
            lastUsedAt: '2026-02-17T01:00:00Z',
            expiresAt: null,
            status: 'active',
            requestCount24h: 42,
            errorRate24h: 0.05,
          },
        ]}
      />
    );

    expect(screen.getByText('Requests (24h)')).toBeInTheDocument();
    expect(screen.getByText('Error Rate (24h)')).toBeInTheDocument();
  });

  it('supports row selection callbacks for bulk actions', () => {
    const onToggleRowSelection = vi.fn();
    const onToggleAllSelection = vi.fn();
    render(
      <UnifiedApiKeysTable
        rows={[
          {
            keyId: '101',
            keyPrefix: 'sk-active',
            ownerUserId: 1,
            ownerUsername: 'alice',
            ownerEmail: 'alice@example.com',
            createdAt: '2026-02-17T00:00:00Z',
            lastUsedAt: '2026-02-17T01:00:00Z',
            expiresAt: null,
            status: 'active',
            requestCount24h: null,
            errorRate24h: null,
          },
        ]}
        selectedRowIds={new Set<string>()}
        onToggleRowSelection={onToggleRowSelection}
        onToggleAllSelection={onToggleAllSelection}
      />
    );

    fireEvent.click(screen.getByLabelText('Select key sk-active'));
    expect(onToggleRowSelection).toHaveBeenCalledWith('1:101', true);

    fireEvent.click(screen.getByLabelText('Select all keys'));
    expect(onToggleAllSelection).toHaveBeenCalledWith(['1:101'], true);
  });

  it('exposes the age legend to assistive technology', () => {
    render(
      <UnifiedApiKeysTable
        rows={[
          {
            keyId: '101',
            keyPrefix: 'sk-active',
            ownerUserId: 1,
            ownerUsername: 'alice',
            ownerEmail: 'alice@example.com',
            createdAt: '2026-02-17T00:00:00Z',
            lastUsedAt: '2026-02-17T01:00:00Z',
            expiresAt: null,
            status: 'active',
            requestCount24h: 10,
            errorRate24h: 0.1,
          },
        ]}
      />
    );

    expect(
      screen.getAllByText(/Green under 90 days, yellow 90 to 180 days, red over 180 days/i).length
    ).toBeGreaterThan(0);
  });
});
