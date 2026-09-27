import { useCallback, useEffect, useRef, useState } from 'react';
import { apiClient, getApiBaseUrl } from '@web/lib/api';
import { AUTH_CREDENTIALS_CHANGED_EVENT } from '@web/lib/auth-events';
import { fetchCurrentPrincipal } from '@/services/tldw/verified-principal';
import {
  clearVNCommands, createVNCommandScope, readVNCommands, sameVNCommandScope,
  VNRecoveryStorageError, writeVNCommands,
  type VNCommandScope, type VNPendingCommand,
} from '@web/lib/vnGenerationRecovery';

type Capture = { scope: VNCommandScope; epoch: number };

export function useVNGenerationRecovery(onBoundary: (resetAccount: boolean) => void) {
  const [commands, setCommands] = useState<VNPendingCommand[]>([]);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [unreadable, setUnreadable] = useState(false);
  const [verifiedRevision, setVerifiedRevision] = useState(0);
  const mounted = useRef(false);
  const epoch = useRef(0);
  const scope = useRef<VNCommandScope | null>(null);
  const lastAuthority = useRef<VNCommandScope | null>(null);
  const records = useRef<VNPendingCommand[]>([]);
  const boundary = useRef(onBoundary);
  boundary.current = onBoundary;

  const report = useCallback((failure: unknown) => {
    setError(failure instanceof Error ? failure.message : 'Current server and account could not be verified.');
    setUnreadable(failure instanceof VNRecoveryStorageError && failure.unreadable);
  }, []);

  const isCurrent = useCallback((capture: Capture) => {
    try {
      return mounted.current && epoch.current === capture.epoch && sameVNCommandScope(scope.current, capture.scope) &&
        createVNCommandScope(getApiBaseUrl(), Number(capture.scope.principal)).server === capture.scope.server;
    } catch { return false; }
  }, []);

  const verify = useCallback(async (reloadDetails = false): Promise<Capture | null> => {
    let revision = epoch.current;
    const server = getApiBaseUrl();
    try {
      const principal = await fetchCurrentPrincipal(apiClient.get);
      if (!mounted.current || revision !== epoch.current || server !== getApiBaseUrl()) return null;
      if (principal?.is_active !== true) throw new Error('Current server and account could not be verified.');
      const next = createVNCommandScope(server, principal.id);
      const changed = lastAuthority.current && !sameVNCommandScope(lastAuthority.current, next);
      if (changed) {
        ++epoch.current;
        revision = epoch.current;
        boundary.current(true);
      }
      scope.current = next;
      lastAuthority.current = next;
      const saved = readVNCommands(next);
      records.current = saved;
      setCommands(saved);
      setReady(true);
      setError(null);
      setUnreadable(false);
      if (reloadDetails) setVerifiedRevision((value) => value + 1);
      return changed ? null : { scope: next, epoch: epoch.current };
    } catch (failure) {
      if (!mounted.current || revision !== epoch.current) return null;
      setReady(false);
      report(failure);
      return null;
    }
  }, [report]);

  const remember = useCallback((capture: Capture, command: VNPendingCommand): boolean => {
    if (!isCurrent(capture)) return false;
    const existing = records.current.find((record) => record.packId === command.packId);
    if (existing && existing.request.idempotency_key !== command.request.idempotency_key) return false;
    const next = [...records.current.filter((record) => record.packId !== command.packId), command];
    try {
      writeVNCommands(capture.scope, next);
      records.current = next;
      setCommands(next);
      setError(null);
      return true;
    } catch (failure) {
      report(failure);
      return false;
    }
  }, [isCurrent, report]);

  const forget = useCallback((capture: Capture, command: VNPendingCommand) => {
    if (!isCurrent(capture)) return;
    const next = records.current.filter((record) => record.packId !== command.packId ||
      record.request.idempotency_key !== command.request.idempotency_key);
    try {
      writeVNCommands(capture.scope, next);
      records.current = next;
      setCommands(next);
      setError(null);
    } catch {
      setError('The request result was received, but its saved recovery entry could not be cleared. Recover it again after enabling session storage.');
    }
  }, [isCurrent]);

  const discardUnreadable = useCallback(() => {
    try {
      clearVNCommands();
      void verify();
    } catch (failure) {
      report(failure);
      setUnreadable(true);
    }
  }, [report, verify]);

  const deactivate = useCallback(() => {
    mounted.current = false;
    ++epoch.current;
  }, []);

  useEffect(() => {
    mounted.current = true;
    const invalidate = (clear = false, logout = false) => {
      ++epoch.current;
      scope.current = null;
      records.current = [];
      setCommands([]);
      setReady(false);
      boundary.current(clear);
      if (clear) {
        lastAuthority.current = null;
        try { clearVNCommands(); } catch (failure) { report(failure); }
      }
      if (logout) setError('Sign in again to send generation requests.');
      else void verify(true);
    };
    const principalChanged = (event: Event) => invalidate(true, (event as CustomEvent).detail?.kind === 'logout');
    const credentialsChanged = (event: Event) => {
      const logout = (event as CustomEvent).detail?.authenticated === false;
      invalidate(logout, logout);
    };
    const revalidate = () => invalidate();
    const storageChanged = (event: StorageEvent) => {
      if (event.key === null || ['access_token', 'accessToken', 'apiKey', 'tldwConfig', 'tldwCookieSessionConfig'].includes(event.key)) revalidate();
    };
    void verify();
    window.addEventListener('tldw:auth-principal-changed', principalChanged);
    window.addEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, credentialsChanged);
    window.addEventListener('tldw:config-updated', revalidate);
    window.addEventListener('focus', revalidate);
    window.addEventListener('pageshow', revalidate);
    window.addEventListener('storage', storageChanged);
    return () => {
      deactivate();
      window.removeEventListener('tldw:auth-principal-changed', principalChanged);
      window.removeEventListener(AUTH_CREDENTIALS_CHANGED_EVENT, credentialsChanged);
      window.removeEventListener('tldw:config-updated', revalidate);
      window.removeEventListener('focus', revalidate);
      window.removeEventListener('pageshow', revalidate);
      window.removeEventListener('storage', storageChanged);
    };
  }, [deactivate, report, verify]);

  return { commands, ready, error, unreadable, verifiedRevision, verify, remember, forget, isCurrent, discardUnreadable };
}
