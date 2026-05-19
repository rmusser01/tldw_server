import type { UserWithKeyCount } from '@/types';

export type UnifiedApiKeyStatus = 'active' | 'revoked' | 'expired';

export interface ApiKeyMetadataLike {
  id: number | string;
  key_prefix?: string | null;
  status?: string | null;
  created_at?: string | null;
  expires_at?: string | null;
  last_used_at?: string | null;
}

export interface UnifiedApiKeyRow {
  keyId: string;
  keyPrefix: string;
  ownerUserId: number;
  ownerUsername: string;
  ownerEmail: string;
  createdAt: string | null;
  lastUsedAt: string | null;
  expiresAt: string | null;
  status: UnifiedApiKeyStatus;
  requestCount24h: number | null;
  errorRate24h: number | null;
  /** Cumulative token count from usage attribution (null if not loaded). */
  totalTokens: number | null;
  /** Estimated cumulative cost in USD from usage attribution (null if not loaded). */
  estimatedCostUsd: number | null;
}

export interface KeyAgeIndicator {
  ageDays: number;
  color: 'green' | 'yellow' | 'red';
  label: string;
}

export interface KeyExpiryIndicator {
  daysRemaining: number;
  color: 'yellow' | 'red';
  label: string;
}

export interface KeyHygieneSummary {
  keysNeedingRotation: number;
  keysExpiringSoon: number;
  keysInactive: number;
  hygieneScore: number;
}

export interface UnifiedApiKeyFilters {
  search: string;
  ownerUserId: number | null;
  status: 'all' | UnifiedApiKeyStatus;
  createdBefore: string;
}

const toTimestamp = (value?: string | null): number | null => {
  if (!value) return null;
  const parsed = Date.parse(value);
  return Number.isFinite(parsed) ? parsed : null;
};

const ONE_DAY_MS = 24 * 60 * 60 * 1000;

export const resolveUnifiedApiKeyStatus = (
  status: string | null | undefined,
  expiresAt: string | null | undefined
): UnifiedApiKeyStatus => {
  const normalized = typeof status === 'string' ? status.trim().toLowerCase() : '';
  if (normalized === 'revoked' || normalized === 'inactive' || normalized === 'disabled') {
    return 'revoked';
  }

  const expiresAtMs = toTimestamp(expiresAt);
  if (expiresAtMs !== null && expiresAtMs < Date.now()) {
    return 'expired';
  }

  return 'active';
};

export interface ApiKeyUsageLike {
  key_id: string;
  total_tokens?: number;
  estimated_cost_usd?: number;
}

export const buildUnifiedApiKeyRows = (
  users: UserWithKeyCount[],
  keysByUserId: Record<number, ApiKeyMetadataLike[]>,
  usageByKeyId?: Record<string, ApiKeyUsageLike>,
): UnifiedApiKeyRow[] => {
  const rows: UnifiedApiKeyRow[] = [];

  users.forEach((user) => {
    const userKeys = keysByUserId[user.id] ?? [];
    userKeys.forEach((key) => {
      const keyId = String(key.id);
      const keyPrefix = (key.key_prefix || '').trim();
      const usage = usageByKeyId?.[keyId];
      rows.push({
        keyId,
        keyPrefix,
        ownerUserId: user.id,
        ownerUsername: user.username,
        ownerEmail: user.email,
        createdAt: key.created_at ?? null,
        lastUsedAt: key.last_used_at ?? null,
        expiresAt: key.expires_at ?? null,
        status: resolveUnifiedApiKeyStatus(key.status, key.expires_at),
        requestCount24h: null,
        errorRate24h: null,
        totalTokens: usage?.total_tokens ?? null,
        estimatedCostUsd: usage?.estimated_cost_usd ?? null,
      });
    });
  });

  return rows.sort((a, b) => {
    const left = toTimestamp(a.createdAt);
    const right = toTimestamp(b.createdAt);
    if (left === null && right === null) return 0;
    if (left === null) return 1;
    if (right === null) return -1;
    return right - left;
  });
};

const parseCreatedBeforeCutoff = (createdBefore: string): number | null => {
  const normalized = createdBefore.trim();
  if (!normalized) return null;
  const cutoff = new Date(`${normalized}T23:59:59.999`);
  const timestamp = cutoff.getTime();
  return Number.isFinite(timestamp) ? timestamp : null;
};

export const filterUnifiedApiKeyRows = (
  rows: UnifiedApiKeyRow[],
  filters: UnifiedApiKeyFilters
): UnifiedApiKeyRow[] => {
  const search = filters.search.trim().toLowerCase();
  const createdBeforeCutoff = parseCreatedBeforeCutoff(filters.createdBefore);

  return rows.filter((row) => {
    if (filters.ownerUserId !== null && row.ownerUserId !== filters.ownerUserId) {
      return false;
    }

    if (filters.status !== 'all' && row.status !== filters.status) {
      return false;
    }

    if (createdBeforeCutoff !== null) {
      const createdAtMs = toTimestamp(row.createdAt);
      if (createdAtMs === null || createdAtMs > createdBeforeCutoff) {
        return false;
      }
    }

    if (!search) {
      return true;
    }

    return (
      row.keyPrefix.toLowerCase().includes(search)
      || row.keyId.toLowerCase().includes(search)
      || row.ownerUsername.toLowerCase().includes(search)
    );
  });
};

export const getKeyAgeIndicator = (
  createdAt: string | null,
  now: Date = new Date()
): KeyAgeIndicator | null => {
  const createdAtMs = toTimestamp(createdAt);
  if (createdAtMs === null) return null;

  const ageDays = Math.max(0, Math.floor((now.getTime() - createdAtMs) / ONE_DAY_MS));
  if (ageDays > 180) {
    return {
      ageDays,
      color: 'red',
      label: `${ageDays}d old`,
    };
  }
  if (ageDays >= 90) {
    return {
      ageDays,
      color: 'yellow',
      label: `${ageDays}d old`,
    };
  }
  return {
    ageDays,
    color: 'green',
    label: `${ageDays}d old`,
  };
};

export const getKeyExpiryIndicator = (
  expiresAt: string | null,
  now: Date = new Date()
): KeyExpiryIndicator | null => {
  const expiresAtMs = toTimestamp(expiresAt);
  if (expiresAtMs === null) return null;

  const daysRemaining = Math.ceil((expiresAtMs - now.getTime()) / ONE_DAY_MS);
  if (daysRemaining < 0) return null;
  if (daysRemaining >= 30) return null;

  if (daysRemaining < 7) {
    return {
      daysRemaining,
      color: 'red',
      label: `Expires in ${daysRemaining}d`,
    };
  }
  return {
    daysRemaining,
    color: 'yellow',
    label: `Expires in ${daysRemaining}d`,
  };
};

export const isInactiveKey = (
  lastUsedAt: string | null,
  thresholdDays: number = 30,
  now: Date = new Date()
): boolean => {
  const lastUsedAtMs = toTimestamp(lastUsedAt);
  if (lastUsedAtMs === null) return false;
  const inactiveDays = Math.floor((now.getTime() - lastUsedAtMs) / ONE_DAY_MS);
  return inactiveDays > thresholdDays;
};

export const buildKeyHygieneSummary = (
  rows: UnifiedApiKeyRow[],
  now: Date = new Date()
): KeyHygieneSummary => {
  let keysNeedingRotation = 0;
  let keysExpiringSoon = 0;
  let keysInactive = 0;

  rows.forEach((row) => {
    if (row.status !== 'active') {
      return;
    }

    const age = getKeyAgeIndicator(row.createdAt, now);
    if (age && age.ageDays > 180) {
      keysNeedingRotation += 1;
    }

    const expiresSoon = getKeyExpiryIndicator(row.expiresAt, now);
    if (expiresSoon) {
      keysExpiringSoon += 1;
    }

    if (isInactiveKey(row.lastUsedAt, 30, now)) {
      keysInactive += 1;
    }
  });

  const totalActive = rows.filter((row) => row.status === 'active').length;
  if (totalActive === 0) {
    return {
      keysNeedingRotation,
      keysExpiringSoon,
      keysInactive,
      hygieneScore: 100,
    };
  }

  const riskCount = keysNeedingRotation + keysExpiringSoon + keysInactive;
  const hygieneScore = Math.max(0, Math.round(((totalActive - riskCount) / totalActive) * 100));

  return {
    keysNeedingRotation,
    keysExpiringSoon,
    keysInactive,
    hygieneScore,
  };
};

export type HygieneFilter = 'none' | 'needs-rotation' | 'expiring-soon' | 'inactive';

export const filterByHygiene = (
  rows: UnifiedApiKeyRow[],
  hygieneFilter: HygieneFilter,
  now: Date = new Date()
): UnifiedApiKeyRow[] => {
  if (hygieneFilter === 'none') return rows;

  return rows.filter((row) => {
    if (row.status !== 'active') return false;

    switch (hygieneFilter) {
      case 'needs-rotation': {
        const age = getKeyAgeIndicator(row.createdAt, now);
        return age !== null && age.ageDays > 180;
      }
      case 'expiring-soon': {
        return getKeyExpiryIndicator(row.expiresAt, now) !== null;
      }
      case 'inactive': {
        return isInactiveKey(row.lastUsedAt, 30, now);
      }
      default:
        return true;
    }
  });
};

export const formatRequestCount24h = (value: number | null): string => {
  if (value === null || !Number.isFinite(value)) {
    return 'N/A';
  }
  return value.toLocaleString();
};

export const formatErrorRate24h = (value: number | null): string => {
  if (value === null || !Number.isFinite(value)) {
    return 'N/A';
  }
  return `${value.toFixed(2)}%`;
};

export const formatTokenCount = (value: number | null): string => {
  if (value === null || !Number.isFinite(value)) {
    return '--';
  }
  if (value >= 1_000_000) {
    return `${(value / 1_000_000).toFixed(1)}M`;
  }
  if (value >= 1_000) {
    return `${(value / 1_000).toFixed(1)}K`;
  }
  return value.toLocaleString();
};

export const formatCostUsd = (value: number | null): string => {
  if (value === null || !Number.isFinite(value)) {
    return '--';
  }
  if (value === 0) {
    return '$0.00';
  }
  if (value < 0.01) {
    return '<$0.01';
  }
  return `$${value.toFixed(2)}`;
};
