import { useCallback, useEffect, useState } from 'react';

import { canonicalWebhookApi, detectWebhookApi } from '@/lib/api-client';
import type { LegacyWebhookDeliveryView, LegacyWebhookView } from '@/lib/api-client';
import type { WebhookCatalog, WebhookListResponse, WebhookStatus } from '@/types';
import {
  canLoadCanonicalData,
  safeWebhookError,
  WEBHOOK_PAGE_SIZE,
  type ConflictState,
  type SafeError,
  type WebhookMode,
} from './webhook-controller-shared';
import type { WebhookApiError } from '@/lib/http';

type ShowError = (title: string, description?: string) => void;

type UseWebhookControlPlaneOptions = {
  showError: ShowError;
};

/** Own status-first mode detection, registration reads, pagination, and conflict refresh. */
export const useWebhookControlPlane = ({ showError }: UseWebhookControlPlaneOptions) => {
  const [mode, setMode] = useState<WebhookMode>(null);
  const [status, setStatus] = useState<WebhookStatus | null>(null);
  const [catalog, setCatalog] = useState<WebhookCatalog | null>(null);
  const [canonicalPage, setCanonicalPage] = useState<WebhookListResponse>({
    items: [],
    total: 0,
    limit: WEBHOOK_PAGE_SIZE,
    offset: 0,
  });
  const [legacyItems, setLegacyItems] = useState<LegacyWebhookView[]>([]);
  const [legacyTotal, setLegacyTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [statusError, setStatusError] = useState<SafeError | null>(null);
  const [conflict, setConflict] = useState<ConflictState | null>(null);
  const [legacyExpandedId, setLegacyExpandedId] = useState<string | null>(null);
  const [legacyDeliveries, setLegacyDeliveries] = useState<LegacyWebhookDeliveryView[]>([]);
  const [legacyDeliveryLoading, setLegacyDeliveryLoading] = useState(false);

  const loadControlPlane = useCallback(async (requestedOffset = 0) => {
    setLoading(true);
    setStatusError(null);
    try {
      const detected = await detectWebhookApi();
      setStatus(detected.status);
      setMode(detected.kind);
      setConflict(null);

      if (detected.kind === 'canonical') {
        setLegacyItems([]);
        setLegacyTotal(0);
        setLegacyExpandedId(null);
        setLegacyDeliveries([]);
        if (!canLoadCanonicalData(detected.status)) {
          setCatalog(null);
          setCanonicalPage({
            items: [],
            total: 0,
            limit: WEBHOOK_PAGE_SIZE,
            offset: 0,
          });
          setOffset(0);
          return;
        }
        const [nextCatalog, nextPage] = await Promise.all([
          detected.client.getWebhookCatalog(),
          detected.client.getWebhooks({ limit: WEBHOOK_PAGE_SIZE, offset: requestedOffset }),
        ]);
        setCatalog(nextCatalog);
        setCanonicalPage(nextPage);
        setOffset(nextPage.offset);
        return;
      }

      setCatalog(null);
      setCanonicalPage({ items: [], total: 0, limit: WEBHOOK_PAGE_SIZE, offset: 0 });
      const legacyPage = await detected.client.getWebhooks({
        limit: WEBHOOK_PAGE_SIZE,
        offset: requestedOffset,
      });
      setLegacyItems(legacyPage.items);
      setLegacyTotal(legacyPage.total);
      setOffset(requestedOffset);
    } catch (error) {
      setMode(null);
      setStatus(null);
      setCatalog(null);
      setCanonicalPage({ items: [], total: 0, limit: WEBHOOK_PAGE_SIZE, offset: 0 });
      setLegacyItems([]);
      setLegacyTotal(0);
      setStatusError(safeWebhookError(error, 'Webhook status could not be loaded.'));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadCanonicalPage = useCallback(async (requestedOffset: number) => {
    setLoading(true);
    try {
      const nextPage = await canonicalWebhookApi.getWebhooks({
        limit: WEBHOOK_PAGE_SIZE,
        offset: requestedOffset,
      });
      setCanonicalPage(nextPage);
      setOffset(nextPage.offset);
    } catch (error) {
      const bounded = safeWebhookError(error, 'Webhook registrations could not be loaded.');
      showError('Unable to load webhooks', bounded.message);
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    void loadControlPlane(0);
  }, [loadControlPlane]);

  const recoverConditionalConflict = useCallback(async (
    error: WebhookApiError & { status: 412 | 428 },
    webhookId: number,
    action: string,
  ) => {
    try {
      const current = await canonicalWebhookApi.getWebhook(webhookId);
      setCanonicalPage((page) => ({
        ...page,
        items: page.items.map((registration) => (
          registration.id === webhookId ? current.data : registration
        )),
      }));
      setConflict({ status: error.status, action, registration: current.data });
    } catch {
      setConflict(null);
      showError('Webhook changed', 'Reload the registrations before trying this action again.');
    }
  }, [showError]);

  const goToPage = useCallback(async (nextOffset: number) => {
    const boundedOffset = Math.max(0, nextOffset);
    if (mode === 'canonical') {
      await loadCanonicalPage(boundedOffset);
      return;
    }
    await loadControlPlane(boundedOffset);
  }, [loadCanonicalPage, loadControlPlane, mode]);

  const canonicalCreateBlocked = mode === 'canonical' && (
    !status
    || !canLoadCanonicalData(status)
    || status.key_state !== 'available'
    || status.limits.registrations_over_limit
  );
  const hasCanonicalNext = offset + canonicalPage.limit < canonicalPage.total;
  const hasLegacyNext = offset + WEBHOOK_PAGE_SIZE < legacyTotal;

  return {
    mode,
    status,
    catalog,
    canonicalPage,
    legacyItems,
    offset,
    loading,
    statusError,
    conflict,
    legacyExpandedId,
    legacyDeliveries,
    legacyDeliveryLoading,
    addDisabled: loading || mode === null || canonicalCreateBlocked,
    visibleTotal: mode === 'canonical' ? canonicalPage.total : legacyTotal,
    visibleCount: mode === 'canonical' ? canonicalPage.items.length : legacyItems.length,
    hasPrevious: offset > 0,
    hasNext: mode === 'canonical' ? hasCanonicalNext : hasLegacyNext,
    setConflict,
    setLegacyExpandedId,
    setLegacyDeliveries,
    setLegacyDeliveryLoading,
    loadControlPlane,
    recoverConditionalConflict,
    goToPage,
  };
};
