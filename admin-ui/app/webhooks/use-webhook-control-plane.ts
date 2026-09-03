import { useCallback, useEffect, useState } from 'react';

import { canonicalWebhookApi } from '@/lib/api-client';
import type { WebhookApiError } from '@/lib/http';
import type { WebhookCatalog, WebhookListResponse, WebhookStatus } from '@/types';
import {
  canLoadCanonicalData,
  safeWebhookError,
  WEBHOOK_PAGE_SIZE,
  type ConflictState,
  type SafeError,
} from './webhook-controller-shared';

type ShowError = (title: string, description?: string) => void;

type UseWebhookControlPlaneOptions = {
  showError: ShowError;
};

const emptyPage = (): WebhookListResponse => ({
  items: [],
  total: 0,
  limit: WEBHOOK_PAGE_SIZE,
  offset: 0,
});

/** Own status-first canonical reads, registration pagination, and conflict refresh. */
export const useWebhookControlPlane = ({ showError }: UseWebhookControlPlaneOptions) => {
  const [status, setStatus] = useState<WebhookStatus | null>(null);
  const [catalog, setCatalog] = useState<WebhookCatalog | null>(null);
  const [canonicalPage, setCanonicalPage] = useState<WebhookListResponse>(emptyPage);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [statusError, setStatusError] = useState<SafeError | null>(null);
  const [conflict, setConflict] = useState<ConflictState | null>(null);

  const loadControlPlane = useCallback(async (requestedOffset = 0) => {
    setLoading(true);
    setStatusError(null);
    let nextStatus: WebhookStatus;
    try {
      nextStatus = await canonicalWebhookApi.getWebhookStatus();
    } catch (error) {
      setStatus(null);
      setCatalog(null);
      setCanonicalPage(emptyPage());
      setOffset(0);
      setStatusError(safeWebhookError(error, 'Webhook status could not be loaded.'));
      setLoading(false);
      return;
    }

    setStatus(nextStatus);
    setConflict(null);
    if (!canLoadCanonicalData(nextStatus)) {
      setCatalog(null);
      setCanonicalPage(emptyPage());
      setOffset(0);
      setLoading(false);
      return;
    }

    try {
      const [nextCatalog, nextPage] = await Promise.all([
        canonicalWebhookApi.getWebhookCatalog(),
        canonicalWebhookApi.getWebhooks({ limit: WEBHOOK_PAGE_SIZE, offset: requestedOffset }),
      ]);
      setCatalog(nextCatalog);
      setCanonicalPage(nextPage);
      setOffset(nextPage.offset);
    } catch (error) {
      setCatalog(null);
      setCanonicalPage(emptyPage());
      setOffset(0);
      const bounded = safeWebhookError(error, 'Webhook registrations could not be loaded.');
      showError('Unable to load webhooks', bounded.message);
    } finally {
      setLoading(false);
    }
  }, [showError]);

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
    await loadCanonicalPage(Math.max(0, nextOffset));
  }, [loadCanonicalPage]);

  const ready = status !== null && canLoadCanonicalData(status);

  return {
    status,
    catalog,
    canonicalPage,
    offset,
    loading,
    statusError,
    conflict,
    ready,
    addDisabled: (
      loading
      || !ready
      || status.key_state !== 'available'
      || status.limits.registrations_over_limit
    ),
    visibleTotal: canonicalPage.total,
    visibleCount: canonicalPage.items.length,
    hasPrevious: offset > 0,
    hasNext: offset + canonicalPage.limit < canonicalPage.total,
    setConflict,
    loadControlPlane,
    recoverConditionalConflict,
    goToPage,
  };
};
