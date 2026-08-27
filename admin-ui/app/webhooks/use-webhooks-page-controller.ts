import { useCallback, useEffect, useRef, useState } from 'react';

import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import {
  canonicalWebhookApi,
  legacyWebhookApi,
} from '@/lib/api-client';
import type { LegacyWebhookView } from '@/lib/api-client';
import { createIdempotentCommand } from '@/lib/idempotent-command';
import type {
  WebhookCreateRequest,
  WebhookPatchRequest,
  WebhookRegistration,
} from '@/types';
import {
  activationBlockReason,
  isConditionalWebhookError,
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

export const useWebhooksPageController = () => {
  const promptPrivileged = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();

  const {
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
    addDisabled,
    visibleTotal,
    visibleCount,
    hasPrevious,
    hasNext,
    setConflict,
    setLegacyExpandedId,
    setLegacyDeliveries,
    setLegacyDeliveryLoading,
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
  const [legacyEvents, setLegacyEvents] = useState('');
  const [legacyEnabled, setLegacyEnabled] = useState(true);

  const [editor, setEditor] = useState<EditorState | null>(null);
  const [editDescription, setEditDescription] = useState('');
  const [editTimeout, setEditTimeout] = useState('10');
  const [editEvents, setEditEvents] = useState<string[]>([]);
  const [replacementUrl, setReplacementUrl] = useState('');
  const [replacementUrlError, setReplacementUrlError] = useState('');
  const [mutatingId, setMutatingId] = useState<number | null>(null);
  const legacyExpandedIdRef = useRef<string | null>(legacyExpandedId);
  const activeLegacyDeliveryRequestRef = useRef<symbol | null>(null);

  useEffect(() => {
    if (legacyExpandedIdRef.current === legacyExpandedId) return;
    legacyExpandedIdRef.current = legacyExpandedId;
    activeLegacyDeliveryRequestRef.current = null;
  }, [legacyExpandedId]);

  useEffect(() => () => {
    legacyExpandedIdRef.current = null;
    activeLegacyDeliveryRequestRef.current = null;
  }, []);

  const clearCreateForm = useCallback(() => {
    setCreateUrl('');
    setCreateUrlError('');
    setCreateDescription('');
    setCreateTimeout('10');
    setCreateEvents([]);
    setLegacyEvents('');
    setLegacyEnabled(true);
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
    startLegacySecretCommand,
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
    const pending: PendingSecretCommand = { command, operation: 'create', webhookId: null };
    await startSecretCommand(pending);
  };

  const retrySecretCommand = async () => {
    if (!hasPendingCommand) return;
    await retryPendingSecretCommand();
  };

  const beginLegacyCreate = async () => {
    const events = legacyEvents.split(',').map((event) => event.trim()).filter(Boolean);
    if (!createUrl.trim() || events.length === 0) return;
    const destination = validateWebhookUrl(createUrl);
    if (!destination.valid) {
      setCreateUrlError(destination.message);
      return;
    }
    setCreateUrlError('');
    setCommandError('');
    await startLegacySecretCommand(async () => {
      const response = await legacyWebhookApi.createWebhook({
        url: destination.value,
        events,
        enabled: legacyEnabled,
      });
      return { signing_secret: response.signingSecret, replayed: false };
    });
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
  ) => {
    setMutatingId(webhookId);
    try {
      const current = await canonicalWebhookApi.getWebhook(webhookId);
      const confirmed = await promptPrivileged({
        title: action,
        message: registrationReviewSummary(current.data),
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
      showError('Invalid webhook metadata', 'Select at least one event and use a timeout from 1 to 30 seconds.');
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
    );
  };

  const deleteCanonicalRegistration = async (registration: WebhookRegistration) => {
    setMutatingId(registration.id);
    try {
      const current = await canonicalWebhookApi.getWebhook(registration.id);
      const confirmed = await promptPrivileged({
        title: 'Delete webhook',
        message: `${registrationReviewSummary(current.data)} Delete this registration? This cannot be undone.`,
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
        message: `${registrationReviewSummary(current.data)} Generate a new signing secret? The previous secret will stop working.`,
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

  const toggleLegacyEnabled = async (registration: LegacyWebhookView) => {
    try {
      await legacyWebhookApi.updateWebhook(registration.id, { enabled: !registration.enabled });
      success(registration.enabled ? 'Legacy webhook disabled' : 'Legacy webhook enabled');
      await loadControlPlane(offset);
    } catch {
      showError('Legacy webhook update failed');
    }
  };

  const deleteLegacyRegistration = async (registration: LegacyWebhookView) => {
    const confirmed = await promptPrivileged({
      title: 'Delete legacy webhook',
      message: `Delete legacy webhook ${registration.id}? This cannot be undone.`,
      confirmText: 'Delete webhook',
      confirmationOnly: true,
    });
    if (!confirmed) return;
    try {
      await legacyWebhookApi.deleteWebhook(registration.id);
      success('Legacy webhook deleted');
      await loadControlPlane(0);
    } catch {
      showError('Legacy webhook deletion failed');
    }
  };

  const beginLegacyDeliveryRequest = (
    registrationId: string,
    expandRegistration: boolean,
  ) => {
    if (!expandRegistration && legacyExpandedIdRef.current !== registrationId) return null;
    const requestToken = Symbol('legacy-delivery-history');
    activeLegacyDeliveryRequestRef.current = requestToken;
    if (expandRegistration) {
      legacyExpandedIdRef.current = registrationId;
      setLegacyExpandedId(registrationId);
      setLegacyDeliveries([]);
    }
    setLegacyDeliveryLoading(true);
    return requestToken;
  };

  const isActiveLegacyDeliveryRequest = (
    registrationId: string,
    requestToken: symbol,
  ) => (
    legacyExpandedIdRef.current === registrationId
    && activeLegacyDeliveryRequestRef.current === requestToken
  );

  const loadLegacyDeliveryHistory = async (
    registrationId: string,
    requestToken: symbol,
  ) => {
    try {
      const history = await legacyWebhookApi.getWebhookDeliveries(registrationId, {
        limit: 50,
        offset: 0,
      });
      if (!isActiveLegacyDeliveryRequest(registrationId, requestToken)) return;
      setLegacyDeliveries(history.items);
    } catch {
      if (!isActiveLegacyDeliveryRequest(registrationId, requestToken)) return;
      setLegacyDeliveries([]);
      showError('Legacy delivery history could not be loaded');
    } finally {
      if (isActiveLegacyDeliveryRequest(registrationId, requestToken)) {
        activeLegacyDeliveryRequestRef.current = null;
        setLegacyDeliveryLoading(false);
      }
    }
  };

  const testLegacyRegistration = async (registration: LegacyWebhookView) => {
    const refreshIfStillExpanded = legacyExpandedIdRef.current === registration.id;
    try {
      const delivery = await legacyWebhookApi.testWebhook(registration.id);
      if (delivery.success) {
        success('Legacy test delivery succeeded');
      } else {
        showError('Legacy test delivery failed');
      }
      if (refreshIfStillExpanded) {
        const requestToken = beginLegacyDeliveryRequest(registration.id, false);
        if (requestToken) {
          await loadLegacyDeliveryHistory(registration.id, requestToken);
        }
      }
    } catch {
      showError('Legacy test delivery failed');
    }
  };

  const toggleLegacyDeliveries = async (registration: LegacyWebhookView) => {
    if (legacyExpandedIdRef.current === registration.id) {
      legacyExpandedIdRef.current = null;
      activeLegacyDeliveryRequestRef.current = null;
      setLegacyExpandedId(null);
      setLegacyDeliveries([]);
      setLegacyDeliveryLoading(false);
      return;
    }
    const requestToken = beginLegacyDeliveryRequest(registration.id, true);
    if (requestToken) {
      await loadLegacyDeliveryHistory(registration.id, requestToken);
    }
  };

  return {
    mode,
    status,
    catalog,
    canonicalPage,
    legacyItems,
    offset,
    loading,
    statusError,
    createOpen,
    createUrl,
    createUrlError,
    createDescription,
    createTimeout,
    createEvents,
    legacyEvents,
    legacyEnabled,
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
    hasPendingCommand,
    sensitiveCommandLocked,
    legacyExpandedId,
    legacyDeliveries,
    legacyDeliveryLoading,
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
    setLegacyEnabled,
    setLegacyEvents,
    setReplacementUrl,
    setReplacementUrlError,
    setSecretAcknowledged,
    clearSensitiveCommandState,
    loadControlPlane,
    retrySecretCommand,
    beginCanonicalCreate,
    beginLegacyCreate,
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
    toggleLegacyEnabled,
    deleteLegacyRegistration,
    testLegacyRegistration,
    toggleLegacyDeliveries,
    goToPage,
  };
};
