import { useEffect, useState } from 'react';
import {
  DEFAULT_ALERT_RULE_DRAFT,
  validateAlertRuleDraft,
} from '@/lib/monitoring-alerts';
import { logger } from '@/lib/logger';
import type {
  AlertRule,
  AlertRuleDraft,
  AlertRuleValidationErrors,
} from './types';

export type AlertRulesApiClient = {
  getAdminAlertRules: () => Promise<unknown>;
  createAdminAlertRule: (data: Record<string, unknown>) => Promise<unknown>;
  deleteAdminAlertRule: (ruleId: string) => Promise<unknown>;
};

type UseAlertRulesArgs = {
  apiClient: AlertRulesApiClient;
  setError?: (message: string) => void;
  setSuccess: (message: string) => void;
};

const normalizeAlertRule = (value: unknown): AlertRule | null => {
  if (typeof value !== 'object' || value === null) return null;
  const raw = value as Record<string, unknown>;
  const idValue = raw.id;
  if (idValue === undefined || idValue === null) return null;

  const metric = typeof raw.metric === 'string' ? raw.metric : null;
  const operator = typeof raw.operator === 'string' ? raw.operator : null;
  const threshold = typeof raw.threshold === 'number' ? raw.threshold : Number(raw.threshold);
  const durationMinutes = typeof raw.duration_minutes === 'number'
    ? raw.duration_minutes
    : Number(raw.duration_minutes ?? raw.durationMinutes);
  const severity = typeof raw.severity === 'string' ? raw.severity : null;
  const createdAtRaw = typeof raw.created_at === 'string'
    ? raw.created_at
    : typeof raw.createdAt === 'string'
      ? raw.createdAt
      : new Date().toISOString();

  if (!metric || !operator || !Number.isFinite(threshold) || !Number.isFinite(durationMinutes) || !severity) {
    return null;
  }

  return {
    id: String(idValue),
    metric: metric as AlertRule['metric'],
    operator: operator as AlertRule['operator'],
    threshold,
    durationMinutes: durationMinutes as AlertRule['durationMinutes'],
    severity: severity as AlertRule['severity'],
    createdAt: createdAtRaw,
  };
};

const normalizeAlertRulesPayload = (payload: unknown): AlertRule[] => {
  if (Array.isArray(payload)) {
    return payload
      .map((item) => normalizeAlertRule(item))
      .filter((item): item is AlertRule => item !== null);
  }
  if (typeof payload !== 'object' || payload === null) {
    return [];
  }
  const rawItems = (payload as { items?: unknown }).items;
  if (!Array.isArray(rawItems)) {
    return [];
  }
  return rawItems
    .map((item) => normalizeAlertRule(item))
    .filter((item): item is AlertRule => item !== null);
};

export const useAlertRules = ({
  apiClient,
  setError,
  setSuccess,
}: UseAlertRulesArgs) => {
  const [alertRules, setAlertRules] = useState<AlertRule[]>([]);
  const [alertRuleDraft, setAlertRuleDraft] = useState<AlertRuleDraft>(DEFAULT_ALERT_RULE_DRAFT);
  const [alertRuleValidationErrors, setAlertRuleValidationErrors] = useState<AlertRuleValidationErrors>({});
  const [alertRulesSaving, setAlertRulesSaving] = useState(false);

  useEffect(() => {
    let cancelled = false;

    const loadAlertRules = async () => {
      setAlertRulesSaving(true);
      try {
        const payload = await apiClient.getAdminAlertRules();
        if (!cancelled) {
          setAlertRules(normalizeAlertRulesPayload(payload));
        }
      } catch (error) {
        logger.error('Failed to load alert rules', { component: 'useAlertRules', error: error instanceof Error ? error.message : String(error) });
        if (!cancelled) {
          setError?.(
            error instanceof Error && error.message
              ? error.message
              : 'Failed to load alert rules'
          );
        }
      } finally {
        if (!cancelled) {
          setAlertRulesSaving(false);
        }
      }
    };

    void loadAlertRules();

    return () => {
      cancelled = true;
    };
  }, [apiClient, setError]);

  const handleAlertRuleDraftChange = (draft: AlertRuleDraft) => {
    setAlertRuleDraft(draft);
    setAlertRuleValidationErrors({});
  };

  const handleCreateAlertRule = async () => {
    const validation = validateAlertRuleDraft(alertRuleDraft);
    if (!validation.valid) {
      setAlertRuleValidationErrors(validation.errors);
      return;
    }

    setAlertRulesSaving(true);
    try {
      const payload = await apiClient.createAdminAlertRule({
        metric: alertRuleDraft.metric,
        operator: alertRuleDraft.operator,
        threshold: Number(alertRuleDraft.threshold),
        duration_minutes: Number(alertRuleDraft.durationMinutes),
        severity: alertRuleDraft.severity,
        enabled: true,
      });
      const rawRule = typeof payload === 'object' && payload !== null && 'item' in payload
        ? (payload as { item?: unknown }).item
        : payload;
      const newRule = normalizeAlertRule(rawRule);
      if (!newRule) {
        throw new Error('Failed to create alert rule');
      }
      setAlertRules((prev) => [newRule, ...prev]);
      setAlertRuleValidationErrors({});
      setAlertRuleDraft(DEFAULT_ALERT_RULE_DRAFT);
      setSuccess('Alert rule added');
    } catch (error) {
      logger.error('Failed to create alert rule', { component: 'useAlertRules', error: error instanceof Error ? error.message : String(error) });
      setError?.(
        error instanceof Error && error.message
          ? error.message
          : 'Failed to create alert rule'
      );
    } finally {
      setAlertRulesSaving(false);
    }
  };

  const handleDeleteAlertRule = async (rule: AlertRule) => {
    setAlertRulesSaving(true);
    try {
      await apiClient.deleteAdminAlertRule(rule.id);
      setAlertRules((prev) => prev.filter((item) => item.id !== rule.id));
      setSuccess('Alert rule deleted');
    } catch (error) {
      logger.error('Failed to delete alert rule', { component: 'useAlertRules', error: error instanceof Error ? error.message : String(error) });
      setError?.(
        error instanceof Error && error.message
          ? error.message
          : 'Failed to delete alert rule'
      );
    } finally {
      setAlertRulesSaving(false);
    }
  };

  return {
    alertRules,
    alertRuleDraft,
    alertRuleValidationErrors,
    alertRulesSaving,
    handleAlertRuleDraftChange,
    handleCreateAlertRule,
    handleDeleteAlertRule,
  };
};
