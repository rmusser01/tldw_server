import { describe, expect, it } from 'vitest';
import {
  buildKeyHygieneSummary,
  buildUnifiedApiKeyRows,
  filterByHygiene,
  filterUnifiedApiKeyRows,
  formatCostUsd,
  formatErrorRate24h,
  formatRequestCount24h,
  formatTokenCount,
  getKeyAgeIndicator,
  getKeyExpiryIndicator,
  resolveUnifiedApiKeyStatus,
  type ApiKeyMetadataLike,
  type UnifiedApiKeyRow,
} from './api-keys-hub';
import type { UserWithKeyCount } from '@/types';

describe('resolveUnifiedApiKeyStatus', () => {
  it('classifies active, revoked, and expired keys', () => {
    expect(resolveUnifiedApiKeyStatus('active', null)).toBe('active');
    expect(resolveUnifiedApiKeyStatus('revoked', null)).toBe('revoked');
    expect(resolveUnifiedApiKeyStatus('active', '2000-01-01T00:00:00Z')).toBe('expired');
  });
});

describe('buildUnifiedApiKeyRows', () => {
  it('builds flat rows with mixed statuses across users', () => {
    const users: UserWithKeyCount[] = [
      {
        id: 1,
        uuid: 'u1',
        username: 'alice',
        email: 'alice@example.com',
        role: 'admin',
        is_active: true,
        is_verified: true,
        storage_quota_mb: 1024,
        storage_used_mb: 20,
        created_at: '2026-02-01T00:00:00Z',
        updated_at: '2026-02-01T00:00:00Z',
      },
      {
        id: 2,
        uuid: 'u2',
        username: 'bob',
        email: 'bob@example.com',
        role: 'user',
        is_active: true,
        is_verified: true,
        storage_quota_mb: 1024,
        storage_used_mb: 20,
        created_at: '2026-02-01T00:00:00Z',
        updated_at: '2026-02-01T00:00:00Z',
      },
    ];

    const keysByUserId: Record<number, ApiKeyMetadataLike[]> = {
      1: [
        {
          id: 101,
          key_prefix: 'sk-alive',
          status: 'active',
          created_at: '2026-02-16T00:00:00Z',
        },
        {
          id: 102,
          key_prefix: 'sk-revoked',
          status: 'revoked',
          created_at: '2026-02-15T00:00:00Z',
        },
      ],
      2: [
        {
          id: 103,
          key_prefix: 'sk-expired',
          status: 'active',
          created_at: '2026-02-14T00:00:00Z',
          expires_at: '2000-01-01T00:00:00Z',
        },
      ],
    };

    const rows = buildUnifiedApiKeyRows(users, keysByUserId);

    expect(rows).toHaveLength(3);
    expect(rows.find((row) => row.keyId === '101')?.status).toBe('active');
    expect(rows.find((row) => row.keyId === '102')?.status).toBe('revoked');
    expect(rows.find((row) => row.keyId === '103')?.status).toBe('expired');
  });
});

describe('filterUnifiedApiKeyRows', () => {
  const rows = [
    {
      keyId: '101',
      keyPrefix: 'sk-alpha',
      ownerUserId: 1,
      ownerUsername: 'alice',
      ownerEmail: 'alice@example.com',
      createdAt: '2026-02-10T00:00:00Z',
      lastUsedAt: null,
      expiresAt: null,
      status: 'active' as const,
      requestCount24h: null,
      errorRate24h: null,
      totalTokens: null,
      estimatedCostUsd: null,
    },
    {
      keyId: '102',
      keyPrefix: 'sk-beta',
      ownerUserId: 2,
      ownerUsername: 'bob',
      ownerEmail: 'bob@example.com',
      createdAt: '2026-01-10T00:00:00Z',
      lastUsedAt: null,
      expiresAt: null,
      status: 'revoked' as const,
      requestCount24h: null,
      errorRate24h: null,
      totalTokens: null,
      estimatedCostUsd: null,
    },
  ];

  it('supports combined owner/status/date/search filters', () => {
    const filtered = filterUnifiedApiKeyRows(rows, {
      search: 'beta',
      ownerUserId: 2,
      status: 'revoked',
      createdBefore: '2026-01-31',
    });

    expect(filtered).toHaveLength(1);
    expect(filtered[0].keyId).toBe('102');
  });
});

describe('metric formatters', () => {
  it('returns N/A when 24h metrics are unavailable', () => {
    expect(formatRequestCount24h(null)).toBe('N/A');
    expect(formatErrorRate24h(null)).toBe('N/A');
  });
});

describe('stage 2 hygiene helpers', () => {
  it('computes age badge thresholds', () => {
    const now = new Date('2026-02-17T00:00:00Z');
    expect(getKeyAgeIndicator('2026-02-10T00:00:00Z', now)?.color).toBe('green');
    expect(getKeyAgeIndicator('2025-10-31T00:00:00Z', now)?.color).toBe('yellow');
    expect(getKeyAgeIndicator('2025-08-01T00:00:00Z', now)?.color).toBe('red');
  });

  it('formats expiration countdown for soon-to-expire keys', () => {
    const now = new Date('2026-02-17T00:00:00Z');
    expect(getKeyExpiryIndicator('2026-03-10T00:00:00Z', now)?.color).toBe('yellow');
    expect(getKeyExpiryIndicator('2026-02-20T00:00:00Z', now)?.color).toBe('red');
    expect(getKeyExpiryIndicator('2026-04-20T00:00:00Z', now)).toBeNull();
  });

  it('computes hygiene summary counts and score', () => {
    const now = new Date('2026-02-17T00:00:00Z');
    const summary = buildKeyHygieneSummary([
      {
        keyId: '1',
        keyPrefix: 'sk-1',
        ownerUserId: 1,
        ownerUsername: 'alice',
        ownerEmail: 'alice@example.com',
        createdAt: '2025-08-01T00:00:00Z',
        lastUsedAt: '2025-12-01T00:00:00Z',
        expiresAt: '2026-02-20T00:00:00Z',
        status: 'active',
        requestCount24h: null,
        errorRate24h: null,
        totalTokens: null,
        estimatedCostUsd: null,
      },
      {
        keyId: '2',
        keyPrefix: 'sk-2',
        ownerUserId: 2,
        ownerUsername: 'bob',
        ownerEmail: 'bob@example.com',
        createdAt: '2026-02-01T00:00:00Z',
        lastUsedAt: '2026-02-16T00:00:00Z',
        expiresAt: null,
        status: 'active',
        requestCount24h: null,
        errorRate24h: null,
        totalTokens: null,
        estimatedCostUsd: null,
      },
    ], now);

    expect(summary.keysNeedingRotation).toBe(1);
    expect(summary.keysExpiringSoon).toBe(1);
    expect(summary.keysInactive).toBe(1);
    expect(summary.hygieneScore).toBe(0);
  });
});

describe('filterByHygiene', () => {
  const now = new Date('2026-02-17T00:00:00Z');

  const makeRow = (overrides: Partial<UnifiedApiKeyRow>): UnifiedApiKeyRow => ({
    keyId: '1',
    keyPrefix: 'sk-1',
    ownerUserId: 1,
    ownerUsername: 'alice',
    ownerEmail: 'alice@example.com',
    createdAt: '2026-02-01T00:00:00Z',
    lastUsedAt: '2026-02-16T00:00:00Z',
    expiresAt: null,
    status: 'active',
    requestCount24h: null,
    errorRate24h: null,
    totalTokens: null,
    estimatedCostUsd: null,
    ...overrides,
  });

  const rows: UnifiedApiKeyRow[] = [
    makeRow({ keyId: 'old', createdAt: '2025-06-01T00:00:00Z' }),          // >180 days old
    makeRow({ keyId: 'expiring', expiresAt: '2026-03-01T00:00:00Z' }),     // expires within 30 days
    makeRow({ keyId: 'inactive', lastUsedAt: '2026-01-01T00:00:00Z' }),    // no use in 47 days
    makeRow({ keyId: 'healthy' }),                                          // recent, active, used recently
    makeRow({ keyId: 'revoked', status: 'revoked' }),                       // non-active, excluded by all hygiene filters
  ];

  it('returns all rows when filter is none', () => {
    expect(filterByHygiene(rows, 'none', now)).toHaveLength(5);
  });

  it('filters to only keys needing rotation (>180 days old)', () => {
    const result = filterByHygiene(rows, 'needs-rotation', now);
    expect(result).toHaveLength(1);
    expect(result[0].keyId).toBe('old');
  });

  it('filters to only expiring-soon keys', () => {
    const result = filterByHygiene(rows, 'expiring-soon', now);
    expect(result).toHaveLength(1);
    expect(result[0].keyId).toBe('expiring');
  });

  it('filters to only inactive keys', () => {
    const result = filterByHygiene(rows, 'inactive', now);
    expect(result).toHaveLength(1);
    expect(result[0].keyId).toBe('inactive');
  });
});

describe('usage attribution', () => {
  it('enriches rows with usage data when usageByKeyId is provided', () => {
    const users: UserWithKeyCount[] = [
      {
        id: 1,
        uuid: 'u1',
        username: 'alice',
        email: 'alice@example.com',
        role: 'admin',
        is_active: true,
        is_verified: true,
        storage_quota_mb: 1024,
        storage_used_mb: 20,
        created_at: '2026-02-01T00:00:00Z',
        updated_at: '2026-02-01T00:00:00Z',
      },
    ];

    const keysByUserId: Record<number, ApiKeyMetadataLike[]> = {
      1: [
        {
          id: 101,
          key_prefix: 'sk-alice',
          status: 'active',
          created_at: '2026-02-16T00:00:00Z',
        },
      ],
    };

    const usageByKeyId = {
      '101': { key_id: '101', total_tokens: 50000, estimated_cost_usd: 1.23 },
    };

    const rows = buildUnifiedApiKeyRows(users, keysByUserId, usageByKeyId);
    expect(rows).toHaveLength(1);
    expect(rows[0].totalTokens).toBe(50000);
    expect(rows[0].estimatedCostUsd).toBe(1.23);
  });

  it('sets usage fields to null when no usage data is provided', () => {
    const users: UserWithKeyCount[] = [
      {
        id: 1,
        uuid: 'u1',
        username: 'alice',
        email: 'alice@example.com',
        role: 'admin',
        is_active: true,
        is_verified: true,
        storage_quota_mb: 1024,
        storage_used_mb: 20,
        created_at: '2026-02-01T00:00:00Z',
        updated_at: '2026-02-01T00:00:00Z',
      },
    ];

    const keysByUserId: Record<number, ApiKeyMetadataLike[]> = {
      1: [
        { id: 101, key_prefix: 'sk-alice', status: 'active', created_at: '2026-02-16T00:00:00Z' },
      ],
    };

    const rows = buildUnifiedApiKeyRows(users, keysByUserId);
    expect(rows[0].totalTokens).toBeNull();
    expect(rows[0].estimatedCostUsd).toBeNull();
  });
});

describe('usage formatters', () => {
  it('formats token counts with abbreviated suffixes', () => {
    expect(formatTokenCount(null)).toBe('--');
    expect(formatTokenCount(0)).toBe('0');
    expect(formatTokenCount(500)).toBe('500');
    expect(formatTokenCount(1500)).toBe('1.5K');
    expect(formatTokenCount(1_500_000)).toBe('1.5M');
  });

  it('formats cost values as USD strings', () => {
    expect(formatCostUsd(null)).toBe('--');
    expect(formatCostUsd(0)).toBe('$0.00');
    expect(formatCostUsd(0.005)).toBe('<$0.01');
    expect(formatCostUsd(1.5)).toBe('$1.50');
    expect(formatCostUsd(123.456)).toBe('$123.46');
  });
});
