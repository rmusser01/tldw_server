'use client';

import { Fragment, useCallback, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';
import {
  AlertTriangle,
  ChevronLeft,
  ChevronRight,
  Copy,
  Edit3,
  KeyRound,
  Link2,
  Play,
  Plus,
  RefreshCw,
  RotateCw,
  Trash2,
  Webhook,
} from 'lucide-react';

import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import {
  Dialog,
  DialogContent,
  DialogDescription,
  DialogFooter,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog';
import { EmptyState } from '@/components/ui/empty-state';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import {
  canonicalWebhookApi,
  detectWebhookApi,
  legacyWebhookApi,
} from '@/lib/api-client';
import type { LegacyWebhookDeliveryView, LegacyWebhookView } from '@/lib/api-client';
import { formatDateTime } from '@/lib/format';
import {
  WebhookApiError,
  WebhookContractError,
  WebhookTransportError,
} from '@/lib/http';
import {
  createIdempotentCommand,
} from '@/lib/idempotent-command';
import type { IdempotentCommand } from '@/lib/idempotent-command';
import type {
  WebhookCatalog,
  WebhookCreateRequest,
  WebhookListResponse,
  WebhookPatchRequest,
  WebhookRegistration,
  WebhookSecretResponse,
  WebhookStatus,
} from '@/types';

const PAGE_SIZE = 20;

type WebhookMode = 'canonical' | 'legacy' | null;
type SecretOperation = 'create' | 'rotate';

type SecretCommandResult = {
  data: WebhookSecretResponse;
  etag: string;
  status: number;
  requestId: string | null;
};

type PendingSecretCommand = {
  command: IdempotentCommand<SecretCommandResult>;
  operation: SecretOperation;
  webhookId: number | null;
};

type SecretState = {
  value: string;
  replayed: boolean;
  operation: SecretOperation;
};

type EditorState = {
  kind: 'metadata' | 'destination';
  registration: WebhookRegistration;
};

type ConflictState = {
  status: 412 | 428;
  action: string;
  registration: WebhookRegistration;
};

type SafeError = {
  message: string;
  requestId: string | null;
};

const canLoadCanonicalData = (status: WebhookStatus): boolean => (
  status.mode !== 'off'
  && status.schema_ready
  && status.migration.phase === 'complete'
);

const safeError = (error: unknown, fallback: string): SafeError => {
  if (
    error instanceof WebhookApiError
    || error instanceof WebhookContractError
    || error instanceof WebhookTransportError
  ) {
    return { message: error.message, requestId: error.requestId };
  }
  return { message: fallback, requestId: null };
};

const isConditionalError = (error: unknown): error is WebhookApiError & { status: 412 | 428 } => (
  error instanceof WebhookApiError && (error.status === 412 || error.status === 428)
);

const activationBlockReason = (
  registration: WebhookRegistration,
  status: WebhookStatus,
): string | null => {
  if (registration.active) return null;
  if (!status.delivery_capability_ready) return 'Delivery capability is unavailable';
  if (registration.secret_rotation_required) return 'Generate a new signing secret before activation';
  if (status.key_state !== 'available') return 'Webhook signing key is unavailable';
  if (status.limits.active_registrations_over_limit) return 'Active registration limit is exceeded';
  return null;
};

const registrationReviewSummary = (registration: WebhookRegistration): string => (
  `Webhook ${registration.id}, revision ${registration.revision}: `
  + `${registration.target_display}; ${registration.active ? 'active' : 'inactive'}; `
  + `${registration.timeout_seconds}s timeout; `
  + `description "${registration.description || 'None'}"; `
  + `events ${registration.event_types.join(', ')}.`
);

function StatusAlerts({ status }: { status: WebhookStatus }) {
  const required = status.migration.secret_rotation_required_count;
  return (
    <div className="space-y-3" aria-label="Webhook operational status">
      {status.mode === 'off' && (
        <Alert variant="destructive">
          <AlertDescription>
            The webhook control plane is off. Enable it in deployment configuration before managing registrations.
          </AlertDescription>
        </Alert>
      )}
      {status.migration.phase !== 'complete' && (
        <Alert variant="destructive">
          <AlertDescription>
            Webhook migration is not complete ({status.migration.phase}). Complete or resume migration before using canonical registrations.
          </AlertDescription>
        </Alert>
      )}
      {status.key_state !== 'available' && (
        <Alert variant="destructive">
          <AlertDescription>
            The webhook signing key is unavailable. Creation, rotation, and activation are blocked.
          </AlertDescription>
        </Alert>
      )}
      {status.limits.registrations_over_limit && (
        <Alert variant="destructive">
          <AlertDescription>
            The webhook registration limit is exceeded. Delete registrations or raise the configured limit.
          </AlertDescription>
        </Alert>
      )}
      {status.limits.active_registrations_over_limit && (
        <Alert variant="destructive">
          <AlertDescription>
            The active webhook limit is exceeded. New activations are blocked.
          </AlertDescription>
        </Alert>
      )}
      {!status.delivery_capability_ready && (
        <Alert>
          <AlertDescription>
            Webhook delivery capability is unavailable. Registrations can be prepared, but activation is disabled.
          </AlertDescription>
        </Alert>
      )}
      {required > 0 && (
        <Alert>
          <AlertDescription>
            {required} registration{required === 1 ? '' : 's'} require{required === 1 ? 's' : ''} a new signing secret before activation.
          </AlertDescription>
        </Alert>
      )}
      {status.migration.legacy_file_restore_permitted ? (
        status.migration.rollback_window_expires_at && (
          <Alert>
            <AlertDescription>
              Legacy restore remains available until {formatDateTime(status.migration.rollback_window_expires_at)}.
            </AlertDescription>
          </Alert>
        )
      ) : (
        <Alert>
          <AlertDescription>
            Legacy restore is unavailable. Continue with the forward-fix runbook for any registration issue.
          </AlertDescription>
        </Alert>
      )}
    </div>
  );
}

function WebhooksPageContent() {
  const promptPrivileged = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();

  const [mode, setMode] = useState<WebhookMode>(null);
  const [status, setStatus] = useState<WebhookStatus | null>(null);
  const [catalog, setCatalog] = useState<WebhookCatalog | null>(null);
  const [canonicalPage, setCanonicalPage] = useState<WebhookListResponse>({
    items: [],
    total: 0,
    limit: PAGE_SIZE,
    offset: 0,
  });
  const [legacyItems, setLegacyItems] = useState<LegacyWebhookView[]>([]);
  const [legacyTotal, setLegacyTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(true);
  const [statusError, setStatusError] = useState<SafeError | null>(null);

  const [createOpen, setCreateOpen] = useState(false);
  const [createUrl, setCreateUrl] = useState('');
  const [createDescription, setCreateDescription] = useState('');
  const [createTimeout, setCreateTimeout] = useState('10');
  const [createEvents, setCreateEvents] = useState<string[]>([]);
  const [legacyEvents, setLegacyEvents] = useState('');
  const [legacyEnabled, setLegacyEnabled] = useState(true);
  const [creating, setCreating] = useState(false);

  const [editor, setEditor] = useState<EditorState | null>(null);
  const [editDescription, setEditDescription] = useState('');
  const [editTimeout, setEditTimeout] = useState('10');
  const [editEvents, setEditEvents] = useState<string[]>([]);
  const [replacementUrl, setReplacementUrl] = useState('');
  const [mutatingId, setMutatingId] = useState<number | null>(null);
  const [conflict, setConflict] = useState<ConflictState | null>(null);

  const [secretState, setSecretState] = useState<SecretState | null>(null);
  const [secretCopied, setSecretCopied] = useState(false);
  const [secretAcknowledged, setSecretAcknowledged] = useState(false);
  const [secretWarning, setSecretWarning] = useState('');
  const [commandError, setCommandError] = useState('');
  const [commandBusy, setCommandBusy] = useState(false);
  const [pendingOperation, setPendingOperation] = useState<SecretOperation | null>(null);
  const pendingCommandRef = useRef<PendingSecretCommand | null>(null);
  const secretRef = useRef<SecretState | null>(null);

  const [legacyExpandedId, setLegacyExpandedId] = useState<string | null>(null);
  const [legacyDeliveries, setLegacyDeliveries] = useState<LegacyWebhookDeliveryView[]>([]);
  const [legacyDeliveryLoading, setLegacyDeliveryLoading] = useState(false);

  const clearCreateForm = useCallback(() => {
    setCreateUrl('');
    setCreateDescription('');
    setCreateTimeout('10');
    setCreateEvents([]);
    setLegacyEvents('');
    setLegacyEnabled(true);
  }, []);

  const clearSensitiveCommandState = useCallback((synchronous = false) => {
    pendingCommandRef.current = null;
    secretRef.current = null;
    const clearState = () => {
      setSecretState(null);
      setSecretCopied(false);
      setSecretAcknowledged(false);
      setSecretWarning('');
      setCommandError('');
      setCommandBusy(false);
      setPendingOperation(null);
    };
    if (synchronous) {
      flushSync(clearState);
      return;
    }
    clearState();
  }, []);

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
          setCanonicalPage({ items: [], total: 0, limit: PAGE_SIZE, offset: 0 });
          setOffset(0);
          return;
        }
        const [nextCatalog, nextPage] = await Promise.all([
          detected.client.getWebhookCatalog(),
          detected.client.getWebhooks({ limit: PAGE_SIZE, offset: requestedOffset }),
        ]);
        setCatalog(nextCatalog);
        setCanonicalPage(nextPage);
        setOffset(nextPage.offset);
        return;
      }

      setCatalog(null);
      setCanonicalPage({ items: [], total: 0, limit: PAGE_SIZE, offset: 0 });
      const legacyPage = await detected.client.getWebhooks({
        limit: PAGE_SIZE,
        offset: requestedOffset,
      });
      setLegacyItems(legacyPage.items);
      setLegacyTotal(legacyPage.total);
      setOffset(requestedOffset);
    } catch (error) {
      setMode(null);
      setStatus(null);
      setCatalog(null);
      setCanonicalPage({ items: [], total: 0, limit: PAGE_SIZE, offset: 0 });
      setLegacyItems([]);
      setLegacyTotal(0);
      setStatusError(safeError(error, 'Webhook status could not be loaded.'));
    } finally {
      setLoading(false);
    }
  }, []);

  const loadCanonicalPage = useCallback(async (requestedOffset: number) => {
    setLoading(true);
    try {
      const nextPage = await canonicalWebhookApi.getWebhooks({
        limit: PAGE_SIZE,
        offset: requestedOffset,
      });
      setCanonicalPage(nextPage);
      setOffset(nextPage.offset);
    } catch (error) {
      const bounded = safeError(error, 'Webhook registrations could not be loaded.');
      showError('Unable to load webhooks', bounded.message);
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    void loadControlPlane(0);
  }, [loadControlPlane]);

  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!secretRef.current && !pendingCommandRef.current) return;
      event.preventDefault();
      event.returnValue = '';
    };
    const handlePageHide = () => {
      clearSensitiveCommandState(true);
    };
    const handlePageShow = (event: PageTransitionEvent) => {
      if (event.persisted) clearSensitiveCommandState(true);
    };
    window.addEventListener('beforeunload', handleBeforeUnload);
    window.addEventListener('pagehide', handlePageHide);
    window.addEventListener('pageshow', handlePageShow);
    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload);
      window.removeEventListener('pagehide', handlePageHide);
      window.removeEventListener('pageshow', handlePageShow);
      pendingCommandRef.current = null;
      secretRef.current = null;
    };
  }, [clearSensitiveCommandState]);

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

  const revealSecret = useCallback((
    response: Pick<WebhookSecretResponse, 'signing_secret' | 'replayed'>,
    operation: SecretOperation,
  ) => {
    const nextSecret = {
      value: response.signing_secret,
      replayed: response.replayed,
      operation,
    };
    secretRef.current = nextSecret;
    setSecretState(nextSecret);
    setSecretCopied(false);
    setSecretAcknowledged(false);
    setSecretWarning('');
  }, []);

  const runSecretCommand = useCallback(async (
    pending: PendingSecretCommand,
    retry: boolean,
  ) => {
    setCommandBusy(true);
    setCommandError('');
    try {
      const response = retry ? await pending.command.retry() : await pending.command.run();
      pendingCommandRef.current = null;
      setPendingOperation(null);
      setCreateOpen(false);
      clearCreateForm();
      revealSecret(response.data, pending.operation);
      success(
        pending.operation === 'create' ? 'Webhook created' : 'Signing secret generated',
        'The registration remains inactive until explicitly enabled.',
      );
      await loadControlPlane(pending.operation === 'create' ? 0 : offset);
    } catch (error) {
      if (pending.command.canRetry) {
        setCommandError(
          'The connection was lost after submission. Retry the same command, or reload and inspect inactive registrations before creating another one. A lost secret can only be replaced by generating a new secret.',
        );
      } else {
        pendingCommandRef.current = null;
        setPendingOperation(null);
        if (pending.webhookId !== null && isConditionalError(error)) {
          await recoverConditionalConflict(error, pending.webhookId, 'secret rotation');
        } else {
          const bounded = safeError(error, 'The webhook command failed.');
          showError('Webhook command failed', bounded.message);
        }
      }
    } finally {
      setCommandBusy(false);
    }
  }, [clearCreateForm, loadControlPlane, offset, recoverConditionalConflict, revealSecret, showError, success]);

  const beginCanonicalCreate = async () => {
    if (!catalog || createEvents.length === 0 || !createUrl.trim()) return;
    const timeout = Number(createTimeout);
    if (!Number.isInteger(timeout) || timeout < 1 || timeout > 30) {
      setCommandError('Timeout must be a whole number from 1 to 30 seconds.');
      return;
    }
    const body: WebhookCreateRequest = {
      url: createUrl.trim(),
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
    const pending = { command, operation: 'create' as const, webhookId: null };
    pendingCommandRef.current = pending;
    setPendingOperation('create');
    setCreating(true);
    try {
      await runSecretCommand(pending, false);
    } finally {
      setCreating(false);
    }
  };

  const retrySecretCommand = async () => {
    const pending = pendingCommandRef.current;
    if (!pending) return;
    setCreating(pending.operation === 'create');
    try {
      await runSecretCommand(pending, true);
    } finally {
      setCreating(false);
    }
  };

  const beginLegacyCreate = async () => {
    const events = legacyEvents.split(',').map((event) => event.trim()).filter(Boolean);
    if (!createUrl.trim() || events.length === 0) return;
    setCreating(true);
    setCommandError('');
    try {
      const response = await legacyWebhookApi.createWebhook({
        url: createUrl.trim(),
        events,
        enabled: legacyEnabled,
      });
      setCreateOpen(false);
      clearCreateForm();
      revealSecret({ signing_secret: response.signingSecret, replayed: false }, 'create');
      success('Legacy webhook created');
      await loadControlPlane(0);
    } catch {
      showError('Webhook creation failed', 'The legacy webhook could not be created.');
    } finally {
      setCreating(false);
    }
  };

  const openCreate = () => {
    clearCreateForm();
    setCommandError('');
    setCreateOpen(true);
  };

  const handleCreateOpenChange = (open: boolean) => {
    if (!open && pendingCommandRef.current) {
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
    setConflict(null);
  };

  const openDestinationEditor = (registration: WebhookRegistration) => {
    setEditor({ kind: 'destination', registration });
    setReplacementUrl('');
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
      if (isConditionalError(error)) {
        await recoverConditionalConflict(error, webhookId, action);
      } else {
        const bounded = safeError(error, 'The webhook could not be updated.');
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
      await performConditionalUpdate(
        editor.registration.id,
        { url: replacementUrl.trim() },
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
      if (isConditionalError(error)) {
        await recoverConditionalConflict(error, registration.id, 'delete');
      } else {
        const bounded = safeError(error, 'The webhook could not be deleted.');
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
      const pending = {
        command,
        operation: 'rotate' as const,
        webhookId: registration.id,
      };
      pendingCommandRef.current = pending;
      setPendingOperation('rotate');
      await runSecretCommand(pending, false);
    } catch (error) {
      if (isConditionalError(error)) {
        await recoverConditionalConflict(error, registration.id, 'secret rotation');
      } else {
        const bounded = safeError(error, 'A new signing secret could not be generated.');
        showError('Secret rotation failed', bounded.message);
      }
    } finally {
      setMutatingId(null);
    }
  };

  const handleCopySecret = async () => {
    const current = secretRef.current;
    if (!current) return;
    try {
      await navigator.clipboard.writeText(current.value);
      setSecretCopied(true);
      setSecretWarning('');
    } catch {
      setSecretWarning('Clipboard access failed. Select the secret and copy it manually.');
    }
  };

  const requestSecretClose = () => {
    if (!secretCopied || !secretAcknowledged) {
      setSecretWarning('Copy and acknowledge the secret before closing this dialog.');
      return;
    }
    clearSensitiveCommandState(false);
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

  const testLegacyRegistration = async (registration: LegacyWebhookView) => {
    try {
      const delivery = await legacyWebhookApi.testWebhook(registration.id);
      if (delivery.success) {
        success('Legacy test delivery succeeded');
      } else {
        showError('Legacy test delivery failed');
      }
      if (legacyExpandedId === registration.id) {
        const history = await legacyWebhookApi.getWebhookDeliveries(registration.id, {
          limit: 50,
          offset: 0,
        });
        setLegacyDeliveries(history.items);
      }
    } catch {
      showError('Legacy test delivery failed');
    }
  };

  const toggleLegacyDeliveries = async (registration: LegacyWebhookView) => {
    if (legacyExpandedId === registration.id) {
      setLegacyExpandedId(null);
      setLegacyDeliveries([]);
      return;
    }
    setLegacyExpandedId(registration.id);
    setLegacyDeliveryLoading(true);
    try {
      const history = await legacyWebhookApi.getWebhookDeliveries(registration.id, {
        limit: 50,
        offset: 0,
      });
      setLegacyDeliveries(history.items);
    } catch {
      setLegacyDeliveries([]);
      showError('Legacy delivery history could not be loaded');
    } finally {
      setLegacyDeliveryLoading(false);
    }
  };

  const canonicalCreateBlocked = mode === 'canonical' && (
    !status
    || !canLoadCanonicalData(status)
    || status.key_state !== 'available'
    || status.limits.registrations_over_limit
  );
  const addDisabled = loading || mode === null || canonicalCreateBlocked;
  const hasCanonicalNext = offset + canonicalPage.limit < canonicalPage.total;
  const hasLegacyNext = offset + PAGE_SIZE < legacyTotal;
  const visibleTotal = mode === 'canonical' ? canonicalPage.total : legacyTotal;
  const visibleCount = mode === 'canonical' ? canonicalPage.items.length : legacyItems.length;
  const hasPrevious = offset > 0;
  const hasNext = mode === 'canonical' ? hasCanonicalNext : hasLegacyNext;

  const goToPage = async (nextOffset: number) => {
    const boundedOffset = Math.max(0, nextOffset);
    if (mode === 'canonical') {
      await loadCanonicalPage(boundedOffset);
      return;
    }
    await loadControlPlane(boundedOffset);
  };

  return (
    <ResponsiveLayout>
      <div className="space-y-6 p-4 lg:p-8">
        <header className="flex flex-col gap-4 border-b pb-5 sm:flex-row sm:items-center sm:justify-between">
          <div>
            <h1 className="text-2xl font-semibold">Webhooks</h1>
            <p className="mt-1 text-sm text-muted-foreground">
              Manage outgoing event registrations and signing secrets.
            </p>
          </div>
          <div className="flex flex-wrap gap-2">
            <Button
              type="button"
              variant="outline"
              size="sm"
              onClick={() => void loadControlPlane(offset)}
              loading={loading}
              loadingText="Refreshing"
            >
              <RefreshCw className="h-4 w-4" aria-hidden="true" />
              Refresh
            </Button>
            <Button type="button" size="sm" onClick={openCreate} disabled={addDisabled}>
              <Plus className="h-4 w-4" aria-hidden="true" />
              Add webhook
            </Button>
          </div>
        </header>

        {statusError && (
          <Alert variant="destructive">
            <AlertTriangle className="h-4 w-4" aria-hidden="true" />
            <AlertDescription className="space-y-3">
              <p>{statusError.message}</p>
              {statusError.requestId && (
                <p className="font-mono text-xs">Request ID: {statusError.requestId}</p>
              )}
              <Button type="button" variant="outline" size="sm" onClick={() => void loadControlPlane(0)}>
                <RefreshCw className="h-4 w-4" aria-hidden="true" />
                Retry status
              </Button>
            </AlertDescription>
          </Alert>
        )}

        {status && mode === 'canonical' && <StatusAlerts status={status} />}

        {mode === 'legacy' && (
          <Alert>
            <AlertDescription>
              <strong>Legacy compatibility mode.</strong> ETags and secret rotation are unavailable. Complete migration before switching to canonical management.
            </AlertDescription>
          </Alert>
        )}

        {conflict && (
          <Alert variant="destructive">
            <AlertDescription>
              <p>
                Review the current webhook before retrying {conflict.action}. The previous command was not retried automatically.
              </p>
              <p className="mt-2">
                Current revision {conflict.registration.revision}: {conflict.registration.description || 'No description'} at {conflict.registration.target_display}.
              </p>
            </AlertDescription>
          </Alert>
        )}

        {pendingOperation === 'rotate' && commandError && (
          <Alert variant="destructive">
            <AlertDescription className="space-y-3">
              <p>{commandError}</p>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="button"
                  size="sm"
                  onClick={() => void retrySecretCommand()}
                  loading={commandBusy}
                  loadingText="Retrying"
                >
                  <RotateCw className="h-4 w-4" aria-hidden="true" />
                  Retry same command
                </Button>
                <Button
                  type="button"
                  size="sm"
                  variant="outline"
                  onClick={() => {
                    clearSensitiveCommandState(false);
                    void loadControlPlane(offset);
                  }}
                >
                  Reload registrations
                </Button>
              </div>
            </AlertDescription>
          </Alert>
        )}

        {loading && visibleCount === 0 && !statusError ? (
          <div className="py-12 text-center text-sm text-muted-foreground" role="status" aria-live="polite">
            Loading webhooks...
          </div>
        ) : mode === 'canonical' && status && canLoadCanonicalData(status) ? (
          canonicalPage.items.length === 0 ? (
            <EmptyState
              icon={Webhook}
              title="No webhooks configured"
              description="Create an inactive registration, store its signing secret, then enable it when delivery is ready."
            />
          ) : (
            <div className="overflow-x-auto rounded-md border">
              <Table>
                <TableHeader>
                  <TableRow>
                    <TableHead>Destination</TableHead>
                    <TableHead>Description</TableHead>
                    <TableHead>Events</TableHead>
                    <TableHead>Status</TableHead>
                    <TableHead>Updated</TableHead>
                    <TableHead className="text-right">Actions</TableHead>
                  </TableRow>
                </TableHeader>
                <TableBody>
                  {canonicalPage.items.map((registration) => {
                    const activationReason = activationBlockReason(registration, status);
                    const busy = mutatingId === registration.id;
                    const rotationBlocked = registration.active
                      ? 'Disable the webhook before generating a new secret'
                      : status.key_state !== 'available'
                        ? 'Webhook signing key is unavailable'
                        : null;
                    return (
                      <TableRow key={registration.id}>
                        <TableCell>
                          <div className="max-w-64">
                            <p className="truncate font-mono text-sm" title={registration.target_display}>
                              {registration.target_display}
                            </p>
                            <p className="text-xs text-muted-foreground">ID {registration.id}, revision {registration.revision}</p>
                          </div>
                        </TableCell>
                        <TableCell className="max-w-56">
                          <span className="line-clamp-2 text-sm">{registration.description || 'No description'}</span>
                        </TableCell>
                        <TableCell>
                          <div className="flex max-w-64 flex-wrap gap-1">
                            {registration.event_types.map((eventType) => (
                              <Badge key={eventType} variant="secondary" className="text-xs">
                                {eventType}
                              </Badge>
                            ))}
                          </div>
                        </TableCell>
                        <TableCell>
                          <div className="space-y-1">
                            <Badge variant={registration.active ? 'default' : 'outline'}>
                              {registration.active ? 'Active' : 'Inactive'}
                            </Badge>
                            {registration.secret_rotation_required && (
                              <Badge variant="destructive">Secret rotation required</Badge>
                            )}
                          </div>
                        </TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {formatDateTime(registration.updated_at, { fallback: 'Unknown' })}
                        </TableCell>
                        <TableCell>
                          <div className="flex min-w-max flex-wrap justify-end gap-1">
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => openMetadataEditor(registration)}
                              disabled={busy}
                            >
                              <Edit3 className="h-4 w-4" aria-hidden="true" />
                              Edit metadata
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => openDestinationEditor(registration)}
                              disabled={busy}
                            >
                              <Link2 className="h-4 w-4" aria-hidden="true" />
                              Replace destination
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => void rotateCanonicalSecret(registration)}
                              disabled={busy || Boolean(rotationBlocked)}
                              title={rotationBlocked ?? undefined}
                            >
                              <KeyRound className="h-4 w-4" aria-hidden="true" />
                              Generate a new secret
                            </Button>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              onClick={() => void toggleCanonicalRegistration(registration)}
                              disabled={busy || Boolean(activationReason)}
                              title={activationReason ?? undefined}
                            >
                              {registration.active ? 'Disable' : 'Enable'}
                            </Button>
                            <Button
                              type="button"
                              variant="ghost"
                              size="icon"
                              className="h-9 w-9"
                              onClick={() => void deleteCanonicalRegistration(registration)}
                              disabled={busy}
                              aria-label="Delete webhook"
                            >
                              <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
                            </Button>
                          </div>
                        </TableCell>
                      </TableRow>
                    );
                  })}
                </TableBody>
              </Table>
            </div>
          )
        ) : mode === 'legacy' ? (
          legacyItems.length === 0 ? (
            <EmptyState icon={Webhook} title="No legacy webhooks configured" />
          ) : (
            <div className="space-y-4">
              <div className="overflow-x-auto rounded-md border">
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Destination</TableHead>
                      <TableHead>Events</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead className="text-right">Legacy actions</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {legacyItems.map((registration) => (
                      <Fragment key={registration.id}>
                        <TableRow>
                          <TableCell className="max-w-72 truncate font-mono text-sm">
                            {registration.targetUrl}
                          </TableCell>
                          <TableCell>
                            <div className="flex flex-wrap gap-1">
                              {registration.eventTypes.map((eventType) => (
                                <Badge key={eventType} variant="secondary">{eventType}</Badge>
                              ))}
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant={registration.enabled ? 'default' : 'outline'}>
                              {registration.enabled ? 'Enabled' : 'Disabled'}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            <div className="flex min-w-max justify-end gap-1">
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void testLegacyRegistration(registration)}
                              >
                                <Play className="h-4 w-4" aria-hidden="true" />
                                Test
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void toggleLegacyDeliveries(registration)}
                                aria-label={legacyExpandedId === registration.id
                                  ? 'Hide delivery history'
                                  : 'Show delivery history'}
                              >
                                Delivery history
                              </Button>
                              <Button
                                type="button"
                                variant="outline"
                                size="sm"
                                onClick={() => void toggleLegacyEnabled(registration)}
                              >
                                {registration.enabled ? 'Disable' : 'Enable'}
                              </Button>
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="h-9 w-9"
                                onClick={() => void deleteLegacyRegistration(registration)}
                                aria-label="Delete legacy webhook"
                              >
                                <Trash2 className="h-4 w-4 text-destructive" aria-hidden="true" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                        {legacyExpandedId === registration.id && (
                          <TableRow>
                            <TableCell colSpan={4}>
                              <section aria-label="Delivery history" className="space-y-2 py-2">
                                <h2 className="text-sm font-semibold">Delivery history</h2>
                                {legacyDeliveryLoading ? (
                                  <p className="text-sm text-muted-foreground">Loading delivery history...</p>
                                ) : legacyDeliveries.length === 0 ? (
                                  <p className="text-sm text-muted-foreground">No legacy deliveries recorded.</p>
                                ) : (
                                  <ul className="space-y-2">
                                    {legacyDeliveries.map((delivery) => (
                                      <li key={delivery.id} className="flex flex-wrap gap-x-4 gap-y-1 border-t pt-2 text-sm">
                                        <span className="font-mono">{delivery.eventType}</span>
                                        <span>{delivery.success ? 'Succeeded' : 'Failed'}</span>
                                        <span>{delivery.statusCode ?? 'No HTTP status'}</span>
                                        <span>{formatDateTime(delivery.attemptedAt, { fallback: 'Unknown time' })}</span>
                                      </li>
                                    ))}
                                  </ul>
                                )}
                              </section>
                            </TableCell>
                          </TableRow>
                        )}
                      </Fragment>
                    ))}
                  </TableBody>
                </Table>
              </div>
            </div>
          )
        ) : null}

        {mode && visibleTotal > 0 && (
          <nav className="flex flex-wrap items-center justify-between gap-3 border-t pt-4" aria-label="Webhook pagination">
            <p className="text-sm text-muted-foreground">
              Showing {offset + 1}-{offset + visibleCount} of {visibleTotal}
            </p>
            <div className="flex gap-2">
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void goToPage(offset - PAGE_SIZE)}
                disabled={!hasPrevious || loading}
                aria-label="Previous page"
              >
                <ChevronLeft className="h-4 w-4" aria-hidden="true" />
                Previous
              </Button>
              <Button
                type="button"
                variant="outline"
                size="sm"
                onClick={() => void goToPage(offset + PAGE_SIZE)}
                disabled={!hasNext || loading}
                aria-label="Next page"
              >
                Next
                <ChevronRight className="h-4 w-4" aria-hidden="true" />
              </Button>
            </div>
          </nav>
        )}
      </div>

      <Dialog open={createOpen} onOpenChange={handleCreateOpenChange}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Add webhook</DialogTitle>
            <DialogDescription>
              {mode === 'canonical'
                ? 'Create an inactive registration and store its generated signing secret.'
                : 'Create a registration through the legacy compatibility API.'}
            </DialogDescription>
          </DialogHeader>
          <div className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="webhook-create-url">Destination URL</Label>
              <Input
                id="webhook-create-url"
                type="url"
                autoComplete="off"
                value={createUrl}
                onChange={(event) => setCreateUrl(event.target.value)}
                placeholder="https://receiver.example/hooks/events"
                disabled={Boolean(pendingCommandRef.current)}
              />
            </div>
            {mode === 'canonical' ? (
              <>
                <div className="space-y-2">
                  <Label htmlFor="webhook-create-description">Description</Label>
                  <Input
                    id="webhook-create-description"
                    value={createDescription}
                    onChange={(event) => setCreateDescription(event.target.value)}
                    maxLength={500}
                    disabled={Boolean(pendingCommandRef.current)}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="webhook-create-timeout">Timeout (seconds)</Label>
                  <Input
                    id="webhook-create-timeout"
                    type="number"
                    min={1}
                    max={30}
                    step={1}
                    value={createTimeout}
                    onChange={(event) => setCreateTimeout(event.target.value)}
                    disabled={Boolean(pendingCommandRef.current)}
                  />
                </div>
                <fieldset className="space-y-3">
                  <legend className="text-sm font-medium">Events</legend>
                  {catalog?.events.map((event) => (
                    <label key={event.event_type} className="flex items-start gap-3 rounded-md border p-3">
                      <Checkbox
                        checked={createEvents.includes(event.event_type)}
                        onCheckedChange={() => toggleCreateEvent(event.event_type)}
                        disabled={Boolean(pendingCommandRef.current)}
                      />
                      <span className="min-w-0">
                        <span className="block break-all font-mono text-sm">{event.event_type}</span>
                        <span className="block text-xs text-muted-foreground">{event.description}</span>
                      </span>
                    </label>
                  ))}
                </fieldset>
              </>
            ) : (
              <>
                <div className="space-y-2">
                  <Label htmlFor="legacy-webhook-events">Events</Label>
                  <Input
                    id="legacy-webhook-events"
                    value={legacyEvents}
                    onChange={(event) => setLegacyEvents(event.target.value)}
                    placeholder="incident.created, user.created"
                  />
                </div>
                <label className="flex items-center gap-2 text-sm">
                  <Checkbox
                    checked={legacyEnabled}
                    onCheckedChange={(checked) => setLegacyEnabled(checked === true)}
                  />
                  Enabled
                </label>
              </>
            )}
            {commandError && (
              <Alert variant="destructive">
                <AlertDescription className="space-y-3">
                  <p>{commandError}</p>
                  {pendingOperation === 'create' && (
                    <div className="flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="sm"
                        onClick={() => void retrySecretCommand()}
                        loading={commandBusy}
                        loadingText="Retrying"
                      >
                        <RotateCw className="h-4 w-4" aria-hidden="true" />
                        Retry same command
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        onClick={() => {
                          clearSensitiveCommandState(false);
                          setCreateOpen(false);
                          void loadControlPlane(0);
                        }}
                      >
                        Reload registrations
                      </Button>
                    </div>
                  )}
                </AlertDescription>
              </Alert>
            )}
          </div>
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => handleCreateOpenChange(false)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void (mode === 'canonical' ? beginCanonicalCreate() : beginLegacyCreate())}
              disabled={
                creating
                || Boolean(pendingCommandRef.current)
                || !createUrl.trim()
                || (mode === 'canonical' ? createEvents.length === 0 : !legacyEvents.trim())
              }
              loading={creating}
              loadingText="Creating"
            >
              Create
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(editor)} onOpenChange={(open) => !open && setEditor(null)}>
        <DialogContent className="max-h-[90vh] overflow-y-auto sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>
              {editor?.kind === 'destination' ? 'Replace webhook destination' : 'Edit webhook metadata'}
            </DialogTitle>
            <DialogDescription>
              {editor?.kind === 'destination'
                ? `Current redacted destination: ${editor.registration.target_display}`
                : 'A fresh registration revision will be shown for confirmation before saving.'}
            </DialogDescription>
          </DialogHeader>
          {editor?.kind === 'destination' ? (
            <div className="space-y-2">
              <Label htmlFor="webhook-replacement-url">New destination URL</Label>
              <Input
                id="webhook-replacement-url"
                type="url"
                autoComplete="off"
                value={replacementUrl}
                onChange={(event) => setReplacementUrl(event.target.value)}
                placeholder="https://receiver.example/hooks/new"
              />
            </div>
          ) : editor ? (
            <div className="space-y-4">
              <div className="space-y-2">
                <Label htmlFor="webhook-edit-description">Description</Label>
                <Input
                  id="webhook-edit-description"
                  value={editDescription}
                  onChange={(event) => setEditDescription(event.target.value)}
                  maxLength={500}
                />
              </div>
              <div className="space-y-2">
                <Label htmlFor="webhook-edit-timeout">Timeout (seconds)</Label>
                <Input
                  id="webhook-edit-timeout"
                  type="number"
                  min={1}
                  max={30}
                  value={editTimeout}
                  onChange={(event) => setEditTimeout(event.target.value)}
                />
              </div>
              <fieldset className="space-y-2">
                <legend className="text-sm font-medium">Events</legend>
                {catalog?.events.map((event) => (
                  <label key={event.event_type} className="flex items-start gap-3 rounded-md border p-3">
                    <Checkbox
                      checked={editEvents.includes(event.event_type)}
                      onCheckedChange={() => toggleEditEvent(event.event_type)}
                    />
                    <span>
                      <span className="block break-all font-mono text-sm">{event.event_type}</span>
                      <span className="block text-xs text-muted-foreground">{event.description}</span>
                    </span>
                  </label>
                ))}
              </fieldset>
            </div>
          ) : null}
          <DialogFooter>
            <Button type="button" variant="outline" onClick={() => setEditor(null)}>
              Cancel
            </Button>
            <Button
              type="button"
              onClick={() => void submitEditor()}
              loading={editor ? mutatingId === editor.registration.id : false}
              loadingText="Saving"
              disabled={editor?.kind === 'destination' ? !replacementUrl.trim() : editEvents.length === 0}
            >
              {editor?.kind === 'destination' ? 'Save destination' : 'Save changes'}
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>

      <Dialog open={Boolean(secretState)} onOpenChange={(open) => !open && requestSecretClose()}>
        <DialogContent className="sm:max-w-xl">
          <DialogHeader>
            <DialogTitle>Signing secret</DialogTitle>
            <DialogDescription>
              This value is shown once. Store it before leaving this page.
            </DialogDescription>
          </DialogHeader>
          {secretState && (
            <div className="space-y-4">
              {secretState.replayed && (
                <Alert>
                  <AlertDescription>
                    This response was recovered from the original command using its idempotency key.
                  </AlertDescription>
                </Alert>
              )}
              <div className="flex items-center gap-2">
                <Label htmlFor="webhook-signing-secret" className="sr-only">Signing secret</Label>
                <Input
                  id="webhook-signing-secret"
                  readOnly
                  value={secretState.value}
                  className="min-w-0 font-mono text-sm"
                />
                <Button
                  type="button"
                  variant="outline"
                  size="icon"
                  onClick={() => void handleCopySecret()}
                  aria-label="Copy signing secret"
                >
                  <Copy className="h-4 w-4" aria-hidden="true" />
                </Button>
              </div>
              {secretCopied && (
                <p className="text-sm text-emerald-700" role="status">Copied to clipboard.</p>
              )}
              <label className="flex items-start gap-3 rounded-md border p-3 text-sm">
                <Checkbox
                  checked={secretAcknowledged}
                  onCheckedChange={(checked) => setSecretAcknowledged(checked === true)}
                />
                <span>I have stored this signing secret in the destination service.</span>
              </label>
              {secretWarning && (
                <Alert variant="destructive">
                  <AlertDescription>{secretWarning}</AlertDescription>
                </Alert>
              )}
            </div>
          )}
          <DialogFooter>
            <Button
              type="button"
              onClick={requestSecretClose}
              disabled={!secretCopied || !secretAcknowledged}
            >
              Done
            </Button>
          </DialogFooter>
        </DialogContent>
      </Dialog>
    </ResponsiveLayout>
  );
}

export default function WebhooksPage() {
  return (
    <PermissionGuard
      role={['admin', 'super_admin', 'owner']}
      requireAuth
      variant="route"
    >
      <WebhooksPageContent />
    </PermissionGuard>
  );
}
