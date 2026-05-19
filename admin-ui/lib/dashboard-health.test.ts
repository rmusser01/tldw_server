import { describe, expect, it } from 'vitest';
import {
  buildDashboardSystemHealth,
  extractRagCacheHitRatePctFromMetricsText,
  normalizeDashboardHealthStatus,
} from './dashboard-health';

describe('normalizeDashboardHealthStatus', () => {
  it('maps healthy/degraded/down statuses from health endpoints', () => {
    expect(normalizeDashboardHealthStatus('ok')).toBe('healthy');
    expect(normalizeDashboardHealthStatus('healthy')).toBe('healthy');
    expect(normalizeDashboardHealthStatus('degraded')).toBe('degraded');
    expect(normalizeDashboardHealthStatus('warning')).toBe('degraded');
    expect(normalizeDashboardHealthStatus('unhealthy')).toBe('down');
    expect(normalizeDashboardHealthStatus('error')).toBe('down');
  });
});

describe('buildDashboardSystemHealth', () => {
  it('normalizes subsystem statuses from all Stage 2 health endpoints', () => {
    const referenceTime = '2026-02-17T10:00:00.000Z';
    const systemHealth = buildDashboardSystemHealth({
      referenceTime,
      healthResult: {
        status: 'fulfilled',
        value: {
          status: 'ok',
          timestamp: '2026-02-17T09:59:00Z',
          checks: {
            database: { status: 'healthy' },
          },
        },
      },
      llmHealthResult: {
        status: 'fulfilled',
        value: {
          status: 'degraded',
          timestamp: '2026-02-17T09:59:10Z',
        },
      },
      ragHealthResult: {
        status: 'fulfilled',
        value: {
          status: 'unhealthy',
          timestamp: '2026-02-17T09:59:20Z',
          components: {
            cache: { status: 'degraded', hit_rate: 0.82 },
          },
        },
      },
      ttsHealthResult: {
        status: 'fulfilled',
        value: {
          status: 'healthy',
          timestamp: '2026-02-17T09:59:30Z',
        },
      },
      sttHealthResult: {
        status: 'fulfilled',
        value: {
          available: false,
          timestamp: '2026-02-17T09:59:40Z',
        },
      },
      embeddingsHealthResult: {
        status: 'fulfilled',
        value: {
          status: 'degraded',
          timestamp: '2026-02-17T09:59:50Z',
        },
      },
    });

    expect(systemHealth.api.status).toBe('healthy');
    expect(systemHealth.database.status).toBe('healthy');
    expect(systemHealth.llm.status).toBe('degraded');
    expect(systemHealth.rag.status).toBe('down');
    expect(systemHealth.cache.status).toBe('degraded');
    expect(systemHealth.cache.cacheHitRatePct).toBeCloseTo(82, 2);
    expect(systemHealth.tts.status).toBe('healthy');
    expect(systemHealth.stt.status).toBe('down');
    expect(systemHealth.embeddings.status).toBe('degraded');
    expect(systemHealth.jobQueue.status).toBe('unknown');
    expect(systemHealth.api.checkedAt).toBe('2026-02-17T09:59:00.000Z');
    expect(systemHealth.cache.checkedAt).toBe('2026-02-17T09:59:20.000Z');
  });

  it('marks subsystems down when individual health endpoint calls fail', () => {
    const referenceTime = '2026-02-17T10:05:00.000Z';
    const rejected = { status: 'rejected', reason: new Error('unavailable') } as const;
    const systemHealth = buildDashboardSystemHealth({
      referenceTime,
      healthResult: rejected,
      llmHealthResult: rejected,
      ragHealthResult: rejected,
      ttsHealthResult: rejected,
      sttHealthResult: rejected,
      embeddingsHealthResult: rejected,
    });

    expect(systemHealth.api.status).toBe('down');
    expect(systemHealth.database.status).toBe('down');
    expect(systemHealth.llm.status).toBe('down');
    expect(systemHealth.rag.status).toBe('down');
    expect(systemHealth.cache.status).toBe('down');
    expect(systemHealth.tts.status).toBe('down');
    expect(systemHealth.stt.status).toBe('down');
    expect(systemHealth.embeddings.status).toBe('down');
    expect(systemHealth.jobQueue.status).toBe('unknown');
    expect(systemHealth.api.checkedAt).toBe(referenceTime);
    expect(systemHealth.stt.checkedAt).toBe(referenceTime);
  });

  it('falls back to /metrics text parsing when rag health omits cache hit rate', () => {
    const referenceTime = '2026-02-17T10:10:00.000Z';
    const systemHealth = buildDashboardSystemHealth({
      referenceTime,
      healthResult: {
        status: 'fulfilled',
        value: { status: 'ok', timestamp: referenceTime, checks: { database: { status: 'healthy' } } },
      },
      llmHealthResult: {
        status: 'fulfilled',
        value: { status: 'healthy', timestamp: referenceTime },
      },
      ragHealthResult: {
        status: 'fulfilled',
        value: {
          status: 'healthy',
          timestamp: referenceTime,
          components: { cache: { status: 'healthy' } },
        },
      },
      ttsHealthResult: {
        status: 'fulfilled',
        value: { status: 'healthy', timestamp: referenceTime },
      },
      sttHealthResult: {
        status: 'fulfilled',
        value: { available: true, timestamp: referenceTime },
      },
      embeddingsHealthResult: {
        status: 'fulfilled',
        value: { status: 'healthy', timestamp: referenceTime },
      },
      metricsTextResult: {
        status: 'fulfilled',
        value: [
          'rag_cache_hits_total{cache_type="semantic"} 90',
          'rag_cache_misses_total{cache_type="semantic"} 10',
        ].join('\n'),
      },
    });

    expect(systemHealth.cache.cacheHitRatePct).toBeCloseTo(90, 2);
  });
});

describe('extractRagCacheHitRatePctFromMetricsText', () => {
  it('parses cache hit rate from prometheus counters', () => {
    const metricsText = [
      'rag_cache_hits_total{cache_type="semantic"} 60',
      'rag_cache_hits_total{cache_type="exact"} 40',
      'rag_cache_misses_total{cache_type="semantic"} 20',
      'rag_cache_misses_total{cache_type="exact"} 20',
    ].join('\n');

    const hitRate = extractRagCacheHitRatePctFromMetricsText(metricsText);
    expect(hitRate).toBeCloseTo(71.4285, 3);
  });

  it('returns null when cache counters are unavailable', () => {
    expect(extractRagCacheHitRatePctFromMetricsText('http_requests_total 12')).toBeNull();
  });
});
