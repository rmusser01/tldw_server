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
        if (pending.webhookId !== null && isConditionalWebhookError(error)) {
          await recoverConditionalConflict(error, pending.webhookId, 'secret rotation');
        } else {
          const bounded = safeWebhookError(error, 'The webhook command failed.');
          showError('Webhook command failed', bounded.message);
        }
      }
    } finally {
      setCommandBusy(false);
    }
  }, [
    clearCreateForm,
    loadControlPlane,
    offset,
    recoverConditionalConflict,
    revealSecret,
    setCreateOpen,
    showError,
    success,
  ]);

  const startSecretCommand = useCallback(async (pending: PendingSecretCommand) => {
    pendingCommandRef.current = pending;
    setPendingOperation(pending.operation);
    await runSecretCommand(pending, false);
  }, [runSecretCommand]);

  const retrySecretCommand = useCallback(async () => {
    const pending = pendingCommandRef.current;
    if (!pending) return;
    await runSecretCommand(pending, true);
  }, [runSecretCommand]);

  const handleCopySecret = useCallback(async () => {
    const current = secretRef.current;
    if (!current) return;
    try {
      await navigator.clipboard.writeText(current.value);
      setSecretCopied(true);
      setSecretWarning('');
    } catch {
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
    setCommandError,
    setSecretAcknowledged,
    clearSensitiveCommandState,
    revealSecret,
    startSecretCommand,
    retrySecretCommand,
    handleCopySecret,
    requestSecretClose,
  };
};
