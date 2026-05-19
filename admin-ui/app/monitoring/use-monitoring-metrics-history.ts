import { useCallback, useEffect, useMemo, useState } from 'react';
import { useUrlState } from '@/lib/use-url-state';
import {
  normalizeMonitoringMetricsPayload,
  resolveMonitoringRangeParams,
  type MonitoringTimeRangeOption,
} from '@/lib/monitoring-metrics';
import { logger } from '@/lib/logger';
import type { MetricsHistoryPoint } from './types';

export type MonitoringMetricsApiClient = {
  getMonitoringMetrics: (params: {
    start: string;
    end: string;
    granularity: string;
  }) => Promise<unknown>;
};

type UseMonitoringMetricsHistoryArgs = {
  apiClient: MonitoringMetricsApiClient;
  pollIntervalMs?: number;
  onManualRangeLoadSuccess?: () => void;
};

const DEFAULT_METRICS_HISTORY_POLL_MS = 5 * 60 * 1000;
const VALID_TIME_RANGES: MonitoringTimeRangeOption[] = ['1h', '24h', '7d', '30d', 'custom'];

const toDatetimeLocalInputValue = (value: Date): string => {
  const year = value.getFullYear();
  const month = String(value.getMonth() + 1).padStart(2, '0');
  const day = String(value.getDate()).padStart(2, '0');
  const hours = String(value.getHours()).padStart(2, '0');
  const minutes = String(value.getMinutes()).padStart(2, '0');
  return `${year}-${month}-${day}T${hours}:${minutes}`;
};

export const useMonitoringMetricsHistory = ({
  apiClient,
  pollIntervalMs = DEFAULT_METRICS_HISTORY_POLL_MS,
  onManualRangeLoadSuccess,
}: UseMonitoringMetricsHistoryArgs) => {
  const defaultCustomRange = useMemo(() => {
    const now = new Date();
    return {
      start: toDatetimeLocalInputValue(new Date(now.getTime() - (24 * 60 * 60 * 1000))),
      end: toDatetimeLocalInputValue(now),
    };
  }, []);

  const [metricsHistory, setMetricsHistory] = useState<MetricsHistoryPoint[]>([]);
  const [timeRangeRaw, setTimeRange] = useUrlState<MonitoringTimeRangeOption>('range', { defaultValue: '24h' });
  const timeRange = useMemo<MonitoringTimeRangeOption>(() => {
    const candidate = timeRangeRaw ?? '24h';
    return VALID_TIME_RANGES.includes(candidate) ? candidate : '24h';
  }, [timeRangeRaw]);
  const [customRangeStart, setCustomRangeStart] = useState<string>(defaultCustomRange.start);
  const [customRangeEnd, setCustomRangeEnd] = useState<string>(defaultCustomRange.end);
  const [rangeValidationError, setRangeValidationError] = useState('');
  const [activeRangeLabel, setActiveRangeLabel] = useState('24h');

  useEffect(() => {
    if ((timeRangeRaw ?? '24h') !== timeRange) {
      setTimeRange(timeRange);
    }
  }, [setTimeRange, timeRange, timeRangeRaw]);

  const loadMetricsHistoryForRange = useCallback(async (
    selectedRange: MonitoringTimeRangeOption,
    customStart = customRangeStart,
    customEnd = customRangeEnd
  ): Promise<boolean> => {
    const resolvedRange = resolveMonitoringRangeParams(selectedRange, customStart, customEnd);
    if (!resolvedRange.ok) {
      setRangeValidationError(resolvedRange.error);
      return false;
    }

    setRangeValidationError('');
    const rangeParams = resolvedRange.params;
    setActiveRangeLabel(rangeParams.rangeLabel);

    try {
      const historyPayload = await apiClient.getMonitoringMetrics({
        start: rangeParams.start,
        end: rangeParams.end,
        granularity: rangeParams.granularity,
      });
      const normalized = normalizeMonitoringMetricsPayload(historyPayload, rangeParams.end);
      if (normalized.length > 0) {
        setMetricsHistory(normalized);
        return true;
      }
      throw new Error('No monitoring metrics history returned');
    } catch (historyErr: unknown) {
      logger.warn('Failed to load monitoring metrics history endpoint', { component: 'useMonitoringMetricsHistory', error: historyErr instanceof Error ? historyErr.message : String(historyErr) });
      setMetricsHistory([]);
      return false;
    }
  }, [apiClient, customRangeEnd, customRangeStart]);

  useEffect(() => {
    void loadMetricsHistoryForRange(timeRange, customRangeStart, customRangeEnd);
    const intervalId = window.setInterval(() => {
      void loadMetricsHistoryForRange(timeRange, customRangeStart, customRangeEnd);
    }, pollIntervalMs);
    return () => window.clearInterval(intervalId);
  }, [customRangeEnd, customRangeStart, loadMetricsHistoryForRange, pollIntervalMs, timeRange]);

  const handleSelectTimeRange = useCallback(async (nextRange: MonitoringTimeRangeOption): Promise<boolean> => {
    setTimeRange(nextRange);
    if (nextRange === 'custom') {
      return false;
    }
    const loaded = await loadMetricsHistoryForRange(nextRange, customRangeStart, customRangeEnd);
    if (loaded) {
      onManualRangeLoadSuccess?.();
    }
    return loaded;
  }, [customRangeEnd, customRangeStart, loadMetricsHistoryForRange, onManualRangeLoadSuccess]);

  const handleApplyCustomTimeRange = useCallback(async (): Promise<boolean> => {
    setTimeRange('custom');
    const loaded = await loadMetricsHistoryForRange('custom', customRangeStart, customRangeEnd);
    if (loaded) {
      onManualRangeLoadSuccess?.();
    }
    return loaded;
  }, [customRangeEnd, customRangeStart, loadMetricsHistoryForRange, onManualRangeLoadSuccess]);

  return {
    metricsHistory,
    timeRange,
    customRangeStart,
    customRangeEnd,
    rangeValidationError,
    activeRangeLabel,
    setCustomRangeStart,
    setCustomRangeEnd,
    loadMetricsHistoryForRange,
    handleSelectTimeRange,
    handleApplyCustomTimeRange,
  };
};
