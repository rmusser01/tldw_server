import { useCallback, useEffect, useRef, useState } from 'react';
import { flushSync } from 'react-dom';

import type { WebhookApiError } from '@/lib/http';
import type { WebhookSecretResponse } from '@/types';
import {
  isConditionalWebhookError,
  safeWebhookError,
  type PendingSecretCommand,
  type SecretOperation,
  type SecretState,
} from './webhook-controller-shared';

type Toast = (title: string, description?: string) => void;
type SecretCommandLease = {
  lifecycleGeneration: number;
  token: symbol;
};

type UseWebhookSecretCommandsOptions = {
  clearCreateForm: () => void;
  loadControlPlane: (requestedOffset?: number) => Promise<void>;
  offset: number;
  recoverConditionalConflict: (
    error: WebhookApiError & { status: 412 | 428 },
    webhookId: number,
    action: string,
  ) => Promise<void>;
  setCreateOpen: (open: boolean) => void;
  showError: Toast;
  success: Toast;
};

/** Own retryable idempotent commands and the memory-only signing-secret lifecycle. */
export const useWebhookSecretCommands = ({
  clearCreateForm,
  loadControlPlane,
  offset,
  recoverConditionalConflict,
  setCreateOpen,
  showError,
  success,
}: UseWebhookSecretCommandsOptions) => {
  const [secretState, setSecretState] = useState<SecretState | null>(null);
  const [secretCopied, setSecretCopied] = useState(false);
  const [secretAcknowledged, setSecretAcknowledged] = useState(false);
  const [secretWarning, setSecretWarning] = useState('');
  const [commandError, setCommandError] = useState('');
  const [commandBusy, setCommandBusy] = useState(false);
  const [pendingOperation, setPendingOperation] = useState<SecretOperation | null>(null);
  const pendingCommandRef = useRef<PendingSecretCommand | null>(null);
  const secretRef = useRef<SecretState | null>(null);
  const activeCommandRef = useRef<symbol | null>(null);
  const lifecycleGenerationRef = useRef(0);
  const copyAttemptRef = useRef(0);

  const clearSensitiveCommandState = useCallback((synchronous = false) => {
    lifecycleGenerationRef.current += 1;
    copyAttemptRef.current += 1;
    pendingCommandRef.current = null;
    secretRef.current = null;
    activeCommandRef.current = null;
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

  useEffect(() => {
    const handleBeforeUnload = (event: BeforeUnloadEvent) => {
      if (!secretRef.current && !pendingCommandRef.current && !activeCommandRef.current) return;
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
      lifecycleGenerationRef.current += 1;
      copyAttemptRef.current += 1;
      pendingCommandRef.current = null;
      secretRef.current = null;
      activeCommandRef.current = null;
    };
  }, [clearSensitiveCommandState]);

  const revealSecret = useCallback((
    response: Pick<WebhookSecretResponse, 'signing_secret' | 'replayed'>,
    operation: SecretOperation,
    lifecycleGeneration: number,
  ) => {
    if (lifecycleGenerationRef.current !== lifecycleGeneration) return false;
    copyAttemptRef.current += 1;
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
    return true;
  }, []);

  const acquireCommandLease = useCallback((allowPendingCommand = false) => {
    if (
      activeCommandRef.current
      || secretRef.current
      || (!allowPendingCommand && pendingCommandRef.current)
    ) {
      showError(
        'Signing secret command blocked',
        'Finish the current signing-secret command and store its one-time secret before starting another.',
      );
      return null;
    }
    const lease = {
      lifecycleGeneration: lifecycleGenerationRef.current,
      token: Symbol('webhook-secret-command'),
    };
    activeCommandRef.current = lease.token;
    setCommandBusy(true);
    setCommandError('');
    return lease;
  }, [showError]);

  const isCommandLeaseCurrent = useCallback((lease: SecretCommandLease) => (
    lifecycleGenerationRef.current === lease.lifecycleGeneration
    && activeCommandRef.current === lease.token
  ), []);

  const releaseCommandLease = useCallback((lease: SecretCommandLease) => {
    if (!isCommandLeaseCurrent(lease)) return;
    activeCommandRef.current = null;
    setCommandBusy(false);
  }, [isCommandLeaseCurrent]);

  const runSecretCommand = useCallback(async (
    pending: PendingSecretCommand,
    retry: boolean,
    lease: SecretCommandLease,
  ) => {
    const isCurrent = () => (
      isCommandLeaseCurrent(lease)
      && pendingCommandRef.current === pending
    );
    try {
      const response = retry ? await pending.command.retry() : await pending.command.run();
      if (!isCurrent()) return;
      pendingCommandRef.current = null;
      setPendingOperation(null);
      setCreateOpen(false);
      clearCreateForm();
      revealSecret(response.data, pending.operation, lease.lifecycleGeneration);
      success(
        pending.operation === 'create' ? 'Webhook created' : 'Signing secret generated',
        'The registration remains inactive until explicitly enabled.',
      );
      await loadControlPlane(pending.operation === 'create' ? 0 : offset);
    } catch (error) {
      if (!isCurrent()) return;
      if (pending.command.canRetry) {
        setCommandError(
          'The connection was lost after submission. Retry the same command, or reload and inspect inactive registrations before creating another one. A lost secret can only be replaced by generating a new secret.',
        );
      } else {
        pendingCommandRef.current = null;
        setPendingOperation(null);
        if (pending.webhookId !== null && isConditionalWebhookError(error)) {
          await recoverConditionalConflict(error, pending.webhookId, 'secret rotation');
        } else {
          const bounded = safeWebhookError(error, 'The webhook command failed.');
          showError('Webhook command failed', bounded.message);
        }
      }
    } finally {
      releaseCommandLease(lease);
    }
  }, [
    clearCreateForm,
    loadControlPlane,
    offset,
    recoverConditionalConflict,
    isCommandLeaseCurrent,
    releaseCommandLease,
    revealSecret,
    setCreateOpen,
    showError,
    success,
  ]);

  const startSecretCommand = useCallback(async (pending: PendingSecretCommand) => {
    const lease = acquireCommandLease();
    if (!lease) return false;
    pendingCommandRef.current = pending;
    setPendingOperation(pending.operation);
    await runSecretCommand(pending, false, lease);
    return true;
  }, [acquireCommandLease, runSecretCommand]);

  const retrySecretCommand = useCallback(async () => {
    const pending = pendingCommandRef.current;
    if (!pending) return;
    const lease = acquireCommandLease(true);
    if (!lease) return;
    await runSecretCommand(pending, true, lease);
  }, [acquireCommandLease, runSecretCommand]);

  const handleCopySecret = useCallback(async () => {
    const current = secretRef.current;
    if (!current) return;
    const lifecycleGeneration = lifecycleGenerationRef.current;
    const copyAttempt = copyAttemptRef.current + 1;
    copyAttemptRef.current = copyAttempt;
    const isCurrent = () => (
      lifecycleGenerationRef.current === lifecycleGeneration
      && copyAttemptRef.current === copyAttempt
      && secretRef.current === current
    );
    try {
      await navigator.clipboard.writeText(current.value);
      if (!isCurrent()) return;
      setSecretCopied(true);
      setSecretWarning('');
    } catch {
      if (!isCurrent()) return;
      setSecretWarning('Clipboard access failed. Select the secret and copy it manually.');
    }
  }, []);

  const requestSecretClose = useCallback(() => {
    if (!secretCopied || !secretAcknowledged) {
      setSecretWarning('Copy and acknowledge the secret before closing this dialog.');
      return;
    }
    clearSensitiveCommandState(false);
  }, [clearSensitiveCommandState, secretAcknowledged, secretCopied]);

  return {
    secretState,
    secretCopied,
    secretAcknowledged,
    secretWarning,
    commandError,
    commandBusy,
    pendingOperation,
    hasPendingCommand: pendingCommandRef.current !== null,
    sensitiveCommandLocked: (
      commandBusy
      || activeCommandRef.current !== null
      || pendingCommandRef.current !== null
      || secretState !== null
    ),
    setCommandError,
    setSecretAcknowledged,
    clearSensitiveCommandState,
    startSecretCommand,
    retrySecretCommand,
    handleCopySecret,
    requestSecretClose,
  };
};
