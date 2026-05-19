export type DashboardHealthStatus = 'healthy' | 'degraded' | 'down' | 'unknown';

export const DASHBOARD_SUBSYSTEMS = [
  { key: 'api', label: 'API Server' },
  { key: 'database', label: 'Database' },
  { key: 'llm', label: 'LLM Services' },
  { key: 'rag', label: 'RAG Service' },
  { key: 'tts', label: 'TTS Service' },
  { key: 'stt', label: 'STT Service' },
  { key: 'embeddings', label: 'Embeddings' },
  { key: 'cache', label: 'RAG Cache' },
  { key: 'jobQueue', label: 'Job Queue' },
] as const;

export type DashboardSubsystemKey = (typeof DASHBOARD_SUBSYSTEMS)[number]['key'];

export interface DashboardSubsystemHealth {
  status: DashboardHealthStatus;
  checkedAt?: string;
  cacheHitRatePct?: number | null;
  errorMessage?: string | null;
}

export type DashboardSystemHealth = Record<DashboardSubsystemKey, DashboardSubsystemHealth>;

export const DEFAULT_DASHBOARD_SYSTEM_HEALTH: DashboardSystemHealth = {
  api: { status: 'unknown' },
  database: { status: 'unknown' },
  llm: { status: 'unknown' },
  rag: { status: 'unknown' },
  tts: { status: 'unknown' },
  stt: { status: 'unknown' },
  embeddings: { status: 'unknown' },
  cache: { status: 'unknown' },
  jobQueue: { status: 'unknown' },
};

interface BuildDashboardSystemHealthArgs {
  healthResult: PromiseSettledResult<unknown>;
  llmHealthResult: PromiseSettledResult<unknown>;
  ragHealthResult: PromiseSettledResult<unknown>;
  ttsHealthResult: PromiseSettledResult<unknown>;
  sttHealthResult: PromiseSettledResult<unknown>;
  embeddingsHealthResult: PromiseSettledResult<unknown>;
  metricsTextResult?: PromiseSettledResult<unknown>;
  jobsStatsResult?: PromiseSettledResult<unknown>;
  referenceTime?: string;
}

const toRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : null;

const toLowerString = (value: unknown): string =>
  typeof value === 'string'
    ? value.trim().toLowerCase()
    : '';

const normalizeTimestamp = (timestamp: unknown, fallback: string): string => {
  if (typeof timestamp !== 'string' || !timestamp.trim()) {
    return fallback;
  }
  const parsed = new Date(timestamp);
  if (Number.isNaN(parsed.getTime())) {
    return fallback;
  }
  return parsed.toISOString();
};

const toFiniteNumber = (value: unknown): number | null => {
  if (typeof value === 'number' && Number.isFinite(value)) {
    return value;
  }
  if (typeof value === 'string' && value.trim()) {
    const parsed = Number.parseFloat(value.replace('%', '').trim());
    return Number.isFinite(parsed) ? parsed : null;
  }
  return null;
};

const normalizeRateToPercent = (value: unknown): number | null => {
  const parsed = toFiniteNumber(value);
  if (parsed === null) return null;
  const percent = parsed <= 1 ? parsed * 100 : parsed;
  if (!Number.isFinite(percent) || percent < 0) return null;
  return Math.min(percent, 100);
};

export const extractRagCacheHitRatePctFromMetricsText = (metricsText: string): number | null => {
  const metricPattern = /^rag_cache_(hits|misses)_total(?:\{[^}]*\})?\s+([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)$/;

  let hits = 0;
  let misses = 0;

  metricsText.split('\n').forEach((line) => {
    const match = line.match(metricPattern);
    if (!match) return;
    const value = Number.parseFloat(match[2]);
    if (!Number.isFinite(value) || value < 0) return;
    if (match[1] === 'hits') {
      hits += value;
    } else {
      misses += value;
    }
  });

  const total = hits + misses;
  if (total <= 0) return null;
  return (hits / total) * 100;
};

const extractRagCacheHitRatePctFromHealthPayload = (payload: unknown): number | null => {
  const root = toRecord(payload);
  const components = toRecord(root?.components);
  const cache = toRecord(components?.cache);
  return normalizeRateToPercent(cache?.hit_rate);
};

export const normalizeDashboardHealthStatus = (status: unknown): DashboardHealthStatus => {
  const normalized = toLowerString(status);
  if (!normalized) {
    return 'unknown';
  }
  if (['ok', 'healthy', 'ready', 'alive', 'available', 'enabled'].includes(normalized)) {
    return 'healthy';
  }
  if (['degraded', 'warning', 'limited', 'partial'].includes(normalized)) {
    return 'degraded';
  }
  if ([
    'unhealthy',
    'down',
    'error',
    'critical',
    'not_ready',
    'failed',
    'unavailable',
    'circuit_open',
  ].includes(normalized)) {
    return 'down';
  }
  return 'unknown';
};

const extractErrorMessage = (payload: Record<string, unknown>): string | null => {
  if (typeof payload.error === 'string' && payload.error.trim()) {
    return payload.error.trim();
  }
  if (typeof payload.message === 'string' && payload.message.trim()) {
    return payload.message.trim();
  }
  if (typeof payload.detail === 'string' && payload.detail.trim()) {
    return payload.detail.trim();
  }
  if (typeof payload.reason === 'string' && payload.reason.trim()) {
    return payload.reason.trim();
  }
  return null;
};

const resolveSettledStatus = (
  result: PromiseSettledResult<unknown>,
  getStatusFromPayload: (payload: Record<string, unknown>) => DashboardHealthStatus,
  fallbackCheckedAt: string
): DashboardSubsystemHealth => {
  if (result.status !== 'fulfilled') {
    const reason = result.reason;
    const errorMsg = reason instanceof Error
      ? reason.message
      : typeof reason === 'string' ? reason : 'Unreachable';
    return {
      status: 'down',
      checkedAt: fallbackCheckedAt,
      errorMessage: errorMsg,
    };
  }

  const payload = toRecord(result.value);
  if (!payload) {
    return {
      status: 'unknown',
      checkedAt: fallbackCheckedAt,
    };
  }

  const status = getStatusFromPayload(payload);
  const errorMessage = status !== 'healthy' ? extractErrorMessage(payload) : null;

  return {
    status,
    checkedAt: normalizeTimestamp(payload.timestamp, fallbackCheckedAt),
    errorMessage,
  };
};

const resolveSttStatus = (payload: Record<string, unknown>): DashboardHealthStatus => {
  if (typeof payload.available === 'boolean') {
    return payload.available ? 'healthy' : 'down';
  }
  return normalizeDashboardHealthStatus(payload.status);
};

export const buildDashboardSystemHealth = ({
  healthResult,
  llmHealthResult,
  ragHealthResult,
  ttsHealthResult,
  sttHealthResult,
  embeddingsHealthResult,
  metricsTextResult,
  jobsStatsResult,
  referenceTime = new Date().toISOString(),
}: BuildDashboardSystemHealthArgs): DashboardSystemHealth => {
  const apiHealth = resolveSettledStatus(
    healthResult,
    (payload) => normalizeDashboardHealthStatus(payload.status),
    referenceTime
  );

  const databaseHealth = (() => {
    if (healthResult.status !== 'fulfilled') {
      return {
        status: 'down',
        checkedAt: referenceTime,
      } satisfies DashboardSubsystemHealth;
    }
    const payload = toRecord(healthResult.value);
    const checks = toRecord(payload?.checks);
    const databaseCheck = toRecord(checks?.database);
    return {
      status: normalizeDashboardHealthStatus(databaseCheck?.status),
      checkedAt: normalizeTimestamp(payload?.timestamp, referenceTime),
    } satisfies DashboardSubsystemHealth;
  })();

  const llmHealth = resolveSettledStatus(
    llmHealthResult,
    (payload) => normalizeDashboardHealthStatus(payload.status),
    referenceTime
  );

  const ragHealth = resolveSettledStatus(
    ragHealthResult,
    (payload) => normalizeDashboardHealthStatus(payload.status),
    referenceTime
  );

  const cacheHitRateFromMetrics = (() => {
    if (metricsTextResult?.status !== 'fulfilled') return null;
    if (typeof metricsTextResult.value !== 'string') return null;
    return extractRagCacheHitRatePctFromMetricsText(metricsTextResult.value);
  })();

  const cacheHealth = (() => {
    if (ragHealthResult.status !== 'fulfilled') {
      return {
        status: 'down',
        checkedAt: referenceTime,
        cacheHitRatePct: cacheHitRateFromMetrics,
      } satisfies DashboardSubsystemHealth;
    }
    const payload = toRecord(ragHealthResult.value);
    const components = toRecord(payload?.components);
    const cache = toRecord(components?.cache);
    const cacheHitRateFromHealth = extractRagCacheHitRatePctFromHealthPayload(ragHealthResult.value);
    return {
      status: normalizeDashboardHealthStatus(cache?.status),
      checkedAt: normalizeTimestamp(payload?.timestamp, referenceTime),
      cacheHitRatePct: cacheHitRateFromHealth ?? cacheHitRateFromMetrics,
    } satisfies DashboardSubsystemHealth;
  })();

  const ttsHealth = resolveSettledStatus(
    ttsHealthResult,
    (payload) => normalizeDashboardHealthStatus(payload.status),
    referenceTime
  );

  const sttHealth = resolveSettledStatus(sttHealthResult, resolveSttStatus, referenceTime);

  const embeddingsHealth = resolveSettledStatus(
    embeddingsHealthResult,
    (payload) => normalizeDashboardHealthStatus(payload.status),
    referenceTime
  );

  const jobQueueHealth: DashboardSubsystemHealth = (() => {
    if (!jobsStatsResult || jobsStatsResult.status !== 'fulfilled') {
      return { status: 'unknown' as DashboardHealthStatus, checkedAt: referenceTime };
    }
    const rows = Array.isArray(jobsStatsResult.value)
      ? jobsStatsResult.value
      : (toRecord(jobsStatsResult.value)?.items as unknown[] ?? []);
    const payload = toRecord(jobsStatsResult.value);
    const failed = typeof payload?.failed === 'number' ? payload.failed : 0;
    const queueDepth = typeof payload?.queue_depth === 'number' ? payload.queue_depth : 0;
    if (!Array.isArray(rows) || rows.length === 0) {
      // No row-level stats; fall back to top-level failed / queue_depth counters
      const status: DashboardHealthStatus =
        failed > 0 ? 'degraded' :
        queueDepth > 100 ? 'degraded' :
        'healthy';
      return {
        status,
        checkedAt: referenceTime,
        errorMessage: failed > 0 ? `${failed} failed jobs` : queueDepth > 100 ? `Queue depth: ${queueDepth}` : null,
      };
    }
    let quarantined = 0;
    for (const row of rows) {
      const rec = toRecord(row);
      if (rec) {
        const q = toFiniteNumber(rec.quarantined);
        if (q !== null) quarantined += q;
      }
    }
    const status: DashboardHealthStatus =
      quarantined > 10 ? 'down' : quarantined > 0 ? 'degraded' : failed > 0 ? 'degraded' : queueDepth > 100 ? 'degraded' : 'healthy';
    const messages: string[] = [];
    if (quarantined > 0) messages.push(`${quarantined} quarantined job(s)`);
    if (failed > 0) messages.push(`${failed} failed jobs`);
    if (queueDepth > 100) messages.push(`Queue depth: ${queueDepth}`);
    return {
      status,
      checkedAt: referenceTime,
      errorMessage: messages.length > 0 ? messages.join('; ') : null,
    };
  })();

  return {
    api: apiHealth,
    database: databaseHealth,
    llm: llmHealth,
    rag: ragHealth,
    tts: ttsHealth,
    stt: sttHealth,
    embeddings: embeddingsHealth,
    cache: cacheHealth,
    jobQueue: jobQueueHealth,
  };
};
