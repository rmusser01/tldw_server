import { useCallback, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';

import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import { canonicalWebhookApi } from '@/lib/api-client';
import { generateIdempotencyKey, createIdempotentCommand } from '@/lib/idempotent-command';
import { WebhookTransportError } from '@/lib/http';
import type {
  WebhookCreateRequest,
  WebhookDelivery,
  WebhookDeliveryListResponse,
  WebhookPatchRequest,
  WebhookRegistration,
} from '@/types';
import {
  activationBlockReason,
  isConditionalWebhookError,
  queuedDeliveryWarning,
  registrationReviewSummary,
  safeWebhookError,
  type PendingSecretCommand,
  type SecretCommandResult,
} from './webhook-controller-shared';
import { useWebhookControlPlane } from './use-webhook-control-plane';
import { useWebhookSecretCommands } from './use-webhook-secret-commands';
import { validateWebhookUrl } from './webhook-url';

export {
  activationBlockReason,
  canLoadCanonicalData,
  WEBHOOK_PAGE_SIZE,
} from './webhook-controller-shared';

type EditorState = {
  kind: 'metadata' | 'destination';
  registration: WebhookRegistration;
};

type PendingTest = {
  webhookId: number;
  deliveryConfigVersion: number;
  etag: string;
  idempotencyKey: string;
};

type PendingRedelivery = PendingTest & {
  deliveryId: string;
  confirmChangedConfiguration: boolean;
};

const emptyDeliveryPage = (): WebhookDeliveryListResponse => ({
  items: [],
  total: 0,
  limit: 50,
  offset: 0,
});

const isAmbiguousTransportFailure = (error: unknown): boolean => (
  error instanceof WebhookTransportError
  || error instanceof TypeError
  || (error instanceof Error && ['AbortError', 'NetworkError'].includes(error.name))
);

export const useWebhooksPageController = () => {
  const promptPrivileged = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();

  const {
    status,
    catalog,
    canonicalPage,
    offset,
    loading,
    statusError,
    conflict,
    ready,
    addDisabled,
    visibleTotal,
    visibleCount,
    hasPrevious,
    hasNext,
    setConflict,
    loadControlPlane,
    recoverConditionalConflict,
    goToPage,
  } = useWebhookControlPlane({ showError });

  const [createOpen, setCreateOpen] = useState(false);
  const [createUrl, setCreateUrl] = useState('');
  const [createUrlError, setCreateUrlError] = useState('');
  const [createDescription, setCreateDescription] = useState('');
  const [createTimeout, setCreateTimeout] = useState('10');
  const [createEvents, setCreateEvents] = useState<string[]>([]);
  const [editor, setEditor] = useState<EditorState | null>(null);
  const [editDescription, setEditDescription] = useState('');
  const [editTimeout, setEditTimeout] = useState('10');
  const [editEvents, setEditEvents] = useState<string[]>([]);
  const [replacementUrl, setReplacementUrl] = useState('');
  const [replacementUrlError, setReplacementUrlError] = useState('');
  const [mutatingId, setMutatingId] = useState<number | null>(null);

  const [expandedId, setExpandedId] = useState<number | null>(null);
  const [deliveryPage, setDeliveryPage] = useState<WebhookDeliveryListResponse>(emptyDeliveryPage);
  const [deliveryLoading, setDeliveryLoading] = useState(false);
  const [deliveryError, setDeliveryError] = useState('');
  const [testingId, setTestingId] = useState<number | null>(null);
  const [testStatus, setTestStatus] = useState('');
  const [testRetryAvailable, setTestRetryAvailable] = useState(false);
  const [redeliveringId, setRedeliveringId] = useState<string | null>(null);
  const [redeliveryStatus, setRedeliveryStatus] = useState('');
  const [redeliveryRetryAvailable, setRedeliveryRetryAvailable] = useState(false);

  const expandedIdRef = useRef<number | null>(null);
  const activeHistoryRequestRef = useRef<symbol | null>(null);
  const pendingTestRef = useRef<PendingTest | null>(null);
  const pendingRedeliveryRef = useRef<PendingRedelivery | null>(null);
  const activeTestRef = useRef<symbol | null>(null);
  const activeRedeliveryRef = useRef<symbol | null>(null);
  const commandGenerationRef = useRef(0);

  const clearDeliveryCommandState = useCallback((synchronous = false) => {
    commandGenerationRef.current += 1;
    pendingTestRef.current = null;
    pendingRedeliveryRef.current = null;
    activeTestRef.current = null;
    activeRedeliveryRef.current = null;
    activeHistoryRequestRef.current = null;
    const clear = () => {
      setTestingId(null);
      setTestStatus('');
      setTestRetryAvailable(false);
      setRedeliveringId(null);
      setRedeliveryStatus('');
      setRedeliveryRetryAvailable(false);
      setDeliveryLoading(false);
    };
    if (synchronous) {
      flushSync(clear);
    } else {
      clear();
    }
  }, []);

  useEffect(() => {
    const handlePageHide = () => clearDeliveryCommandState(true);
    const handlePageShow = (event: PageTransitionEvent) => {
      if (event.persisted) clearDeliveryCommandState(true);
    };
    window.addEventListener('pagehide', handlePageHide);
    window.addEventListener('pageshow', handlePageShow);
    return () => {
      window.removeEventListener('pagehide', handlePageHide);
      window.removeEventListener('pageshow', handlePageShow);
      commandGenerationRef.current += 1;
      pendingTestRef.current = null;
      pendingRedeliveryRef.current = null;
      activeTestRef.current = null;
      activeRedeliveryRef.current = null;
      activeHistoryRequestRef.current = null;
    };
  }, [clearDeliveryCommandState]);

  const clearCreateForm = useCallback(() => {
    setCreateUrl('');
    setCreateUrlError('');
    setCreateDescription('');
    setCreateTimeout('10');
    setCreateEvents([]);
  }, []);

  const {
    secretState,
    secretCopied,
    secretAcknowledged,
    secretWarning,
    commandError,
    commandBusy,
    pendingOperation,
    hasPendingCommand,
    sensitiveCommandLocked,
    setCommandError,
    setSecretAcknowledged,
    clearSensitiveCommandState,
    startSecretCommand,
    retrySecretCommand: retryPendingSecretCommand,
    handleCopySecret,
    requestSecretClose,
  } = useWebhookSecretCommands({
    clearCreateForm,
    loadControlPlane,
    offset,
    recoverConditionalConflict,
    setCreateOpen,
    showError,
    success,
  });
  const creating = commandBusy && pendingOperation === 'create';

  const beginCanonicalCreate = async () => {
    if (!catalog || createEvents.length === 0 || !createUrl.trim()) return;
    const destination = validateWebhookUrl(createUrl);
    if (!destination.valid) {
      setCreateUrlError(destination.message);
      return;
    }
    setCreateUrlError('');
    const timeout = Number(createTimeout);
    if (!Number.isInteger(timeout) || timeout < 1 || timeout > 30) {
      setCommandError('Timeout must be a whole number from 1 to 30 seconds.');
      return;
    }
    const body: WebhookCreateRequest = {
      url: destination.value,
      event_types: [...createEvents],
      description: createDescription.trim(),
      timeout_seconds: timeout,
    };
    const command = createIdempotentCommand<WebhookCreateRequest, SecretCommandResult>(
      'create',
      body,
      ({ body: requestBody, idempotencyKey }) => canonicalWebhookApi.createWebhook(
        requestBody,
        idempotencyKey,
      ),
    );
    await startSecretCommand({ command, operation: 'create', webhookId: null });
  };

  const retrySecretCommand = async () => {
    if (hasPendingCommand) await retryPendingSecretCommand();
  };

  const openCreate = () => {
    clearCreateForm();
    setCommandError('');
    setCreateOpen(true);
  };

  const handleCreateOpenChange = (open: boolean) => {
    if (!open && sensitiveCommandLocked) {
      setCommandError('Resolve or reload the submitted command before closing this dialog.');
      return;
    }
    setCreateOpen(open);
    if (!open) clearCreateForm();
  };

  const toggleCreateEvent = (eventType: string) => {
    setCreateEvents((current) => current.includes(eventType)
      ? current.filter((entry) => entry !== eventType)
      : [...current, eventType]);
  };

  const toggleEditEvent = (eventType: string) => {
    setEditEvents((current) => current.includes(eventType)
      ? current.filter((entry) => entry !== eventType)
      : [...current, eventType]);
  };

  const openMetadataEditor = (registration: WebhookRegistration) => {
    setEditor({ kind: 'metadata', registration });
    setEditDescription(registration.description);
    setEditTimeout(String(registration.timeout_seconds));
    setEditEvents([...registration.event_types]);
    setReplacementUrl('');
    setReplacementUrlError('');
    setConflict(null);
  };

  const openDestinationEditor = (registration: WebhookRegistration) => {
    setEditor({ kind: 'destination', registration });
    setReplacementUrl('');
    setReplacementUrlError('');
    setConflict(null);
  };

  const performConditionalUpdate = async (
    webhookId: number,
    body: WebhookPatchRequest,
    action: string,
    warnAboutQueuedDeliveries = true,
  ) => {
    setMutatingId(webhookId);
    try {
      const current = await canonicalWebhookApi.getWebhook(webhookId);
      const confirmed = await promptPrivileged({
        title: action,
        message: `${registrationReviewSummary(current.data)}${
          warnAboutQueuedDeliveries ? ` ${queuedDeliveryWarning}` : ''
        }`,
        confirmText: action,
        confirmationOnly: true,
      });
      if (!confirmed) return;
      await canonicalWebhookApi.updateWebhook(webhookId, body, current.etag);
      setEditor(null);
      setConflict(null);
      success('Webhook updated');
      await loadControlPlane(offset);
    } catch (error) {
      setEditor(null);
      if (isConditionalWebhookError(error)) {
        await recoverConditionalConflict(error, webhookId, action);
      } else {
        const bounded = safeWebhookError(error, 'The webhook could not be updated.');
        showError('Webhook update failed', bounded.message);
      }
    } finally {
      setMutatingId(null);
    }
  };

  const submitEditor = async () => {
    if (!editor) return;
    if (editor.kind === 'destination') {
      if (!replacementUrl.trim()) return;
      const destination = validateWebhookUrl(replacementUrl);
      if (!destination.valid) {
        setReplacementUrlError(destination.message);
        return;
      }
      setReplacementUrlError('');
      await performConditionalUpdate(
        editor.registration.id,
        { url: destination.value },
        'Replace destination',
      );
      return;
    }
    const timeout = Number(editTimeout);
    if (!Number.isInteger(timeout) || timeout < 1 || timeout > 30 || editEvents.length === 0) {
      showError(
        'Invalid webhook metadata',
        'Select at least one event and use a timeout from 1 to 30 seconds.',
      );
      return;
    }
    await performConditionalUpdate(editor.registration.id, {
      description: editDescription.trim(),
      event_types: [...editEvents],
      timeout_seconds: timeout,
    }, 'Save changes');
  };

  const toggleCanonicalRegistration = async (registration: WebhookRegistration) => {
    if (!status) return;
    const blockReason = activationBlockReason(registration, status);
    if (blockReason) return;
    await performConditionalUpdate(
      registration.id,
      { active: !registration.active },
      registration.active ? 'Disable webhook' : 'Enable webhook',
      registration.active,
    );
  };

  const deleteCanonicalRegistration = async (registration: WebhookRegistration) => {
    setMutatingId(registration.id);
    try {
      const current = await canonicalWebhookApi.getWebhook(registration.id);
      const confirmed = await promptPrivileged({
        title: 'Delete webhook',
        message: `${registrationReviewSummary(current.data)} Delete this registration? This cannot be undone. ${queuedDeliveryWarning}`,
        confirmText: 'Delete webhook',
        confirmationOnly: true,
      });
      if (!confirmed) return;
      await canonicalWebhookApi.deleteWebhook(registration.id, current.etag);
      success('Webhook deleted');
      await loadControlPlane(0);
    } catch (error) {
      if (isConditionalWebhookError(error)) {
        await recoverConditionalConflict(error, registration.id, 'delete');
      } else {
        const bounded = safeWebhookError(error, 'The webhook could not be deleted.');
        showError('Webhook deletion failed', bounded.message);
      }
    } finally {
      setMutatingId(null);
    }
  };

  const rotateCanonicalSecret = async (registration: WebhookRegistration) => {
    setMutatingId(registration.id);
    try {
      const current = await canonicalWebhookApi.getWebhook(registration.id);
      if (current.data.active) {
        showError('Rotation blocked', 'Disable the webhook before generating a new signing secret.');
        return;
      }
      const confirmed = await promptPrivileged({
        title: 'Generate a new signing secret',
        message: `${registrationReviewSummary(current.data)} Generate a new signing secret? The previous secret will stop working. ${queuedDeliveryWarning}`,
        confirmText: 'Generate secret',
        confirmationOnly: true,
      });
      if (!confirmed) return;
      const command = createIdempotentCommand<
        { webhookId: number; etag: string },
        SecretCommandResult
      >(
        'rotate',
        { webhookId: registration.id, etag: current.etag },
        ({ body, idempotencyKey }) => canonicalWebhookApi.rotateWebhookSecret(
          body.webhookId,
          body.etag,
          idempotencyKey,
        ),
      );
      const pending: PendingSecretCommand = {
        command,
        operation: 'rotate',
        webhookId: registration.id,
      };
      await startSecretCommand(pending);
    } catch (error) {
      if (isConditionalWebhookError(error)) {
        await recoverConditionalConflict(error, registration.id, 'secret rotation');
      } else {
        const bounded = safeWebhookError(error, 'A new signing secret could not be generated.');
        showError('Secret rotation failed', bounded.message);
      }
    } finally {
      setMutatingId(null);
    }
  };

  const loadDeliveryHistory = useCallback(async (webhookId: number) => {
    const requestToken = Symbol('webhook-delivery-history');
    activeHistoryRequestRef.current = requestToken;
    setDeliveryLoading(true);
    setDeliveryError('');
    try {
      const page = await canonicalWebhookApi.getWebhookDeliveries(webhookId, {
        limit: 50,
        offset: 0,
      });
      if (
        activeHistoryRequestRef.current !== requestToken
        || expandedIdRef.current !== webhookId
      ) return;
      setDeliveryPage(page);
    } catch (error) {
      if (
        activeHistoryRequestRef.current !== requestToken
        || expandedIdRef.current !== webhookId
      ) return;
      const bounded = safeWebhookError(error, 'Delivery history could not be loaded.');
      setDeliveryPage(emptyDeliveryPage());
      setDeliveryError(bounded.message);
    } finally {
      if (activeHistoryRequestRef.current === requestToken) {
        activeHistoryRequestRef.current = null;
        setDeliveryLoading(false);
      }
    }
  }, []);

  const toggleDeliveryHistory = async (registration: WebhookRegistration) => {
    if (expandedIdRef.current === registration.id) {
      expandedIdRef.current = null;
      activeHistoryRequestRef.current = null;
      setExpandedId(null);
      setDeliveryPage(emptyDeliveryPage());
      setDeliveryError('');
      setDeliveryLoading(false);
      return;
    }
    expandedIdRef.current = registration.id;
    activeHistoryRequestRef.current = null;
    setExpandedId(registration.id);
    setDeliveryPage(emptyDeliveryPage());
    await loadDeliveryHistory(registration.id);
  };

  const runTestCommand = useCallback(async (pending: PendingTest) => {
    const token = Symbol('webhook-test');
    const generation = commandGenerationRef.current;
    activeTestRef.current = token;
    pendingTestRef.current = pending;
    setTestingId(pending.webhookId);
    setTestRetryAvailable(false);
    setTestStatus('');
    const current = () => (
      commandGenerationRef.current === generation && activeTestRef.current === token
    );
    try {
      const response = await canonicalWebhookApi.testWebhook(
        pending.webhookId,
        { delivery_config_version: pending.deliveryConfigVersion },
        pending.etag,
        pending.idempotencyKey,
      );
      if (!current()) return;
      if (response.status === 202) {
        setTestRetryAvailable(true);
        setTestStatus(
          `Test delivery is still processing. Retry the same test in ${response.retryAfterSeconds ?? 1}s.`,
        );
        return;
      }
      pendingTestRef.current = null;
      setTestStatus(
        response.data.delivery.state === 'succeeded'
          ? 'Test delivery succeeded.'
          : `Test delivery finished with state ${response.data.delivery.state}.`,
      );
      success('Webhook test completed');
      if (expandedIdRef.current === pending.webhookId) {
        await loadDeliveryHistory(pending.webhookId);
      }
    } catch (error) {
      if (!current()) return;
      if (isAmbiguousTransportFailure(error)) {
        setTestRetryAvailable(true);
        setTestStatus('The test result is unknown. Retry the same test to recover its persisted result.');
      } else {
        pendingTestRef.current = null;
        const bounded = safeWebhookError(error, 'The persisted webhook test failed.');
        showError('Webhook test failed', bounded.message);
      }
    } finally {
      if (current()) {
        activeTestRef.current = null;
        setTestingId(null);
      }
    }
  }, [loadDeliveryHistory, showError, success]);

  const testCanonicalRegistration = async (registration: WebhookRegistration) => {
    pendingTestRef.current = null;
    setTestRetryAvailable(false);
    setTestStatus('');
    try {
      const current = await canonicalWebhookApi.getWebhook(registration.id);
      const confirmed = await promptPrivileged({
        title: 'Run webhook test',
        message: `${registrationReviewSummary(current.data)} Send one persisted test to ${current.data.target_hostname}?`,
        confirmText: 'Run test',
        confirmationOnly: true,
      });
      if (!confirmed) return;
      await runTestCommand({
        webhookId: registration.id,
        deliveryConfigVersion: current.data.delivery_config_version,
        etag: current.etag,
        idempotencyKey: generateIdempotencyKey(),
      });
    } catch (error) {
      const bounded = safeWebhookError(error, 'The current webhook could not be loaded.');
      showError('Webhook test failed', bounded.message);
    }
  };

  const retrySameTest = async () => {
    const pending = pendingTestRef.current;
    if (pending) await runTestCommand(pending);
  };

  const runRedeliveryCommand = useCallback(async (pending: PendingRedelivery) => {
    const token = Symbol('webhook-redelivery');
    const generation = commandGenerationRef.current;
    activeRedeliveryRef.current = token;
    pendingRedeliveryRef.current = pending;
    setRedeliveringId(pending.deliveryId);
    setRedeliveryRetryAvailable(false);
    setRedeliveryStatus('');
    const current = () => (
      commandGenerationRef.current === generation && activeRedeliveryRef.current === token
    );
    try {
      await canonicalWebhookApi.redeliverWebhook(
        pending.webhookId,
        pending.deliveryId,
        {
          delivery_config_version: pending.deliveryConfigVersion,
          confirm_changed_configuration: pending.confirmChangedConfiguration,
        },
        pending.etag,
        pending.idempotencyKey,
      );
      if (!current()) return;
      pendingRedeliveryRef.current = null;
      setRedeliveryStatus('Manual redelivery accepted.');
      success('Webhook redelivery accepted');
      if (expandedIdRef.current === pending.webhookId) {
        await loadDeliveryHistory(pending.webhookId);
      }
    } catch (error) {
      if (!current()) return;
      if (isAmbiguousTransportFailure(error)) {
        setRedeliveryRetryAvailable(true);
        setRedeliveryStatus(
          'The redelivery result is unknown. Retry the same command to recover its acceptance.',
        );
      } else if (isConditionalWebhookError(error)) {
        pendingRedeliveryRef.current = null;
        await recoverConditionalConflict(error, pending.webhookId, 'manual redelivery');
      } else {
        pendingRedeliveryRef.current = null;
        const bounded = safeWebhookError(error, 'The manual redelivery could not be accepted.');
        showError('Webhook redelivery failed', bounded.message);
      }
    } finally {
      if (current()) {
        activeRedeliveryRef.current = null;
        setRedeliveringId(null);
      }
    }
  }, [loadDeliveryHistory, recoverConditionalConflict, showError, success]);

  const redeliverWebhook = async (delivery: WebhookDelivery) => {
    try {
      const current = await canonicalWebhookApi.getWebhook(delivery.webhook_id);
      const changed = delivery.delivery_config_version !== current.data.delivery_config_version;
      const message = changed
        ? `The configuration changed from version ${delivery.delivery_config_version} to ${current.data.delivery_config_version}. Redelivery will use the current redacted destination ${current.data.target_hostname}. Confirm delivery to the changed configuration.`
        : `Redeliver ${delivery.event_type} to the current redacted destination ${current.data.target_hostname}?`;
      const confirmed = await promptPrivileged({
        title: 'Redeliver webhook event',
        message,
        confirmText: 'Redeliver event',
        confirmationOnly: true,
      });
      if (!confirmed) return;
      await runRedeliveryCommand({
        webhookId: current.data.id,
        deliveryId: delivery.id,
        deliveryConfigVersion: current.data.delivery_config_version,
        confirmChangedConfiguration: changed,
        etag: current.etag,
        idempotencyKey: generateIdempotencyKey(),
      });
    } catch (error) {
      const bounded = safeWebhookError(error, 'The current webhook could not be loaded.');
      showError('Webhook redelivery failed', bounded.message);
    }
  };

  const retrySameRedelivery = async () => {
    const pending = pendingRedeliveryRef.current;
    if (pending) await runRedeliveryCommand(pending);
  };

  return {
    status,
    catalog,
    canonicalPage,
    offset,
    loading,
    statusError,
    ready,
    createOpen,
    createUrl,
    createUrlError,
    createDescription,
    createTimeout,
    createEvents,
    creating,
    editor,
    editDescription,
    editTimeout,
    editEvents,
    replacementUrl,
    replacementUrlError,
    mutatingId,
    conflict,
    secretState,
    secretCopied,
    secretAcknowledged,
    secretWarning,
    commandError,
    commandBusy,
    pendingOperation,
    sensitiveCommandLocked,
    expandedId,
    deliveryPage,
    deliveryLoading,
    deliveryError,
    testingId,
    testStatus,
    testRetryAvailable,
    redeliveringId,
    redeliveryStatus,
    redeliveryRetryAvailable,
    addDisabled,
    visibleTotal,
    visibleCount,
    hasPrevious,
    hasNext,
    setCommandError,
    setCreateDescription,
    setCreateOpen,
    setCreateTimeout,
    setCreateUrl,
    setCreateUrlError,
    setEditDescription,
    setEditTimeout,
    setEditor,
    setReplacementUrl,
    setReplacementUrlError,
    setSecretAcknowledged,
    clearSensitiveCommandState,
    loadControlPlane,
    retrySecretCommand,
    beginCanonicalCreate,
    openCreate,
    handleCreateOpenChange,
    toggleCreateEvent,
    toggleEditEvent,
    openMetadataEditor,
    openDestinationEditor,
    submitEditor,
    toggleCanonicalRegistration,
    deleteCanonicalRegistration,
    rotateCanonicalSecret,
    handleCopySecret,
    requestSecretClose,
    toggleDeliveryHistory,
    testCanonicalRegistration,
    retrySameTest,
    redeliverWebhook,
    retrySameRedelivery,
    goToPage,
  };
};
