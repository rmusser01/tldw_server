import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Archive, ClipboardList, Images, LayoutGrid, Settings, RefreshCw, RotateCcw, Trash2 } from 'lucide-react';
import { ApiError } from '@web/lib/api';
import {
  clearPendingVNAssetGeneration,
  createVNAssetIdempotencyKey,
  readPendingVNAssetGeneration,
  type PendingVNAssetGeneration,
} from '@web/lib/vnAssetIdempotency';
import { useVNGenerationRecovery } from '@web/hooks/useVNGenerationRecovery';
import { readVNCommands, sameVNCommandScope, type VNCommandScope } from '@web/lib/vnGenerationRecovery';
import { isVNGenerationRejected } from '@web/lib/api/vnGenerationErrors';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import GenerationMonitor from '@web/components/vn-assets/GenerationMonitor';
import MatrixEditor from '@web/components/vn-assets/MatrixEditor';
import PackList from '@web/components/vn-assets/PackList';
import PackSetup from '@web/components/vn-assets/PackSetup';
import PortabilityPanel from '@web/components/vn-assets/PortabilityPanel';
import ReadinessPanel from '@web/components/vn-assets/ReadinessPanel';
import ReviewBoard from '@web/components/vn-assets/ReviewBoard';
import {
  applyVNAssetMatrix,
  bulkReviewVNAssetItems,
  cancelVNAssetGeneration,
  createVNAssetPack,
  getStarterMatrices,
  getVNAssetGeneration,
  getVNAssetGenerationPreflight,
  getVNAssetReadiness,
  listVNAssetItems,
  listVNAssetPacks,
  listVNAssetSlots,
  setPreferredVNAssetItem,
  startVNAssetGeneration,
  retryVNAssetSlot,
} from '@web/lib/api/vnAssets';
import type {
  VNAssetBulkReviewRequest,
  VNAssetGenerationStatus,
  VNAssetGenerationPreflight,
  VNAssetItem,
  VNAssetPack,
  VNAssetReadiness,
  VNAssetSlot,
  VNAssetStarterMatrix,
} from '@web/types/vn-assets';

const workflowSteps = [
  { key: 'setup', label: 'Setup', icon: Settings },
  { key: 'matrix', label: 'Matrix', icon: LayoutGrid },
  { key: 'generation', label: 'Generation', icon: Images },
  { key: 'review', label: 'Review', icon: ClipboardList },
  { key: 'portability', label: 'Portability', icon: Archive },
] as const;

function plannedAssetLabel(count: number): string {
  return `${count} planned ${count === 1 ? 'asset' : 'assets'}`;
}

function isMissingGenerationResource(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
    && ['slot_not_found', 'pack_not_found'].includes(error.detail ?? error.message);
}

function legacyOperation(scope: VNCommandScope, packId: number): string {
  return JSON.stringify([scope.server, scope.principal, packId]);
}

function selectedPackStorageKey(scope: VNCommandScope): string {
  return `vn-assets:selected-pack:v2:${encodeURIComponent(scope.server)}:${encodeURIComponent(scope.principal)}`;
}

function readSelectedPackId(scope: VNCommandScope): number | null {
  const key = selectedPackStorageKey(scope);
  if (typeof window === 'undefined') return null;
  try {
    const id = Number(window.sessionStorage.getItem(key));
    return Number.isSafeInteger(id) && id > 0 ? id : null;
  } catch {
    return null;
  }
}

export default function VNAssetsWorkbench() {
  const [packs, setPacks] = useState<VNAssetPack[]>([]);
  const [selectedPack, setSelectedPack] = useState<VNAssetPack | null>(null);
  const [starterMatrices, setStarterMatrices] = useState<VNAssetStarterMatrix[]>([]);
  const [slots, setSlots] = useState<VNAssetSlot[]>([]);
  const [items, setItems] = useState<VNAssetItem[]>([]);
  const [generation, setGeneration] = useState<VNAssetGenerationStatus | null>(null);
  const [preflight, setPreflight] = useState<VNAssetGenerationPreflight | null>(null);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [preflightRevision, setPreflightRevision] = useState(0);
  const [loadedPackId, setLoadedPackId] = useState<number | null>(null);
  const generationCommandPending = useRef(new Map<number, symbol>());
  const [pendingCommands, setPendingCommands] = useState<Record<number, { kind: 'start' | 'retry' | 'cancel'; slotId?: number }>>({});
  const legacyReceipts = useRef(new Map<string, PendingVNAssetGeneration>());
  const autoReconciledKeys = useRef(new Set<string>());
  const selectedPackIdRef = useRef<number | null>(null);
  const refreshRevision = useRef(0);
  const refreshInFlight = useRef<{ packId: number; revision: number; promise: Promise<void> } | null>(null);
  selectedPackIdRef.current = selectedPack?.id ?? null;
  const [readiness, setReadiness] = useState<VNAssetReadiness | null>(null);
  const [activeWorkflowStep, setActiveWorkflowStep] = useState<(typeof workflowSteps)[number]['key']>('setup');
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isApplyingMatrix, setIsApplyingMatrix] = useState(false);
  const selectedCommand = selectedPack ? pendingCommands[selectedPack.id] : undefined;
  const isStartingGeneration = selectedCommand?.kind === 'start';
  const isCancellingGeneration = selectedCommand?.kind === 'cancel';
  const retryingSlotId = selectedCommand?.kind === 'retry' ? selectedCommand.slotId ?? null : null;
  const [error, setError] = useState<string | null>(null);
  const [title, setTitle] = useState('Untitled VN asset pack');
  const [primaryCharacterId, setPrimaryCharacterId] = useState('1');
  const [discardConfirmed, setDiscardConfirmed] = useState(false);
  const [accountRevision, setAccountRevision] = useState(0);
  const recovery = useVNGenerationRecovery(useCallback((resetAccount: boolean) => {
    if (resetAccount) {
      generationCommandPending.current.clear();
      setPendingCommands({});
      legacyReceipts.current.clear();
      autoReconciledKeys.current.clear();
    }
    setDiscardConfirmed(false);
    ++refreshRevision.current;
    if (resetAccount) {
      selectedPackIdRef.current = null;
      setSelectedPack(null);
      setPacks([]);
      setAccountRevision((value) => value + 1);
    }
  }, []));
  const { authority, captureCurrent, isAuthority } = recovery;
  const savedCommand = recovery.commands.find((command) => command.packId === selectedPack?.id);

  const starterMatrix = starterMatrices[0] ?? null;

  const finishGenerationCommand = useCallback((packId: number, token: symbol, notSent = false): void => {
    if (generationCommandPending.current.get(packId) !== token) return;
    if (notSent && selectedPackIdRef.current === packId) {
      setError('The request was not sent because verification was interrupted. Try again.');
    }
    generationCommandPending.current.delete(packId);
    setPendingCommands((previous) => {
      const next = { ...previous };
      delete next[packId];
      return next;
    });
  }, []);

  const readinessBadge = useMemo(() => {
    if (!readiness) return 'Setup';
    return readiness.ready ? 'Ready' : readiness.status;
  }, [readiness]);

  const refreshPackDetails = useCallback((pack: VNAssetPack, afterMutation = false): Promise<void> => {
    if (!authority || !isAuthority(authority) || selectedPackIdRef.current !== pack.id) return Promise.resolve();
    if (afterMutation) ++refreshRevision.current;
    const revision = refreshRevision.current;
    const pending = refreshInFlight.current;
    if (pending?.packId === pack.id && pending.revision === revision) return pending.promise;
    const isCurrent = () => isAuthority(authority) && selectedPackIdRef.current === pack.id && revision === refreshRevision.current;

    const load = async () => {
      try {
        if (afterMutation && pending?.packId === pack.id) await pending.promise;
        if (!isCurrent()) return;
        // Read status first so terminal batches cannot strand earlier item/slot snapshots.
        const nextGeneration = await getVNAssetGeneration(pack.id);
        if (!isCurrent()) return;
        const results = await Promise.allSettled([
          listVNAssetSlots(pack.id),
          listVNAssetItems(pack.id),
          getVNAssetReadiness(pack.id),
        ]);
        if (!isCurrent()) return;
        const [nextSlots, nextItems, nextReadiness] = results;
        // Let every request finish before another refresh, including on failure.
        if (nextSlots.status === 'rejected') throw nextSlots.reason;
        if (nextItems.status === 'rejected') throw nextItems.reason;
        if (nextReadiness.status === 'rejected') throw nextReadiness.reason;
        setSlots(nextSlots.value);
        setItems(nextItems.value);
        setGeneration(nextGeneration);
        setReadiness(nextReadiness.value);
        setLoadedPackId(pack.id);
        setError((previous) => previous === 'Could not refresh generation progress. Refresh to try again.' ||
          previous === 'Could not load generation status. Refresh to try again.' ? null : previous);
      } catch (loadError) {
        if (isCurrent()) throw loadError;
      }
    };
    const promise = load().finally(() => {
      if (refreshInFlight.current?.promise === promise) refreshInFlight.current = null;
    });
    refreshInFlight.current = { packId: pack.id, revision, promise };
    return promise;
  }, [authority, isAuthority]);

  const reconcilePendingGeneration = useCallback(async (pack: VNAssetPack): Promise<boolean> => {
    // Key-only parent receipts remain separate from dev's complete, scoped requests.
    const original = recovery.captureCurrent();
    if (!original || original.scope.principal !== String(pack.owner_user_id)) return false;
    const operation = legacyOperation(original.scope, pack.id);
    const pending = readPendingVNAssetGeneration(original.scope, pack.id) ?? legacyReceipts.current.get(operation);
    if (!pending || !recovery.ready || recovery.commands.some((command) => command.packId === pack.id) ||
        generationCommandPending.current.has(pack.id)) return false;
    legacyReceipts.current.set(operation, pending);
    const token = Symbol('legacy-generation-command');
    generationCommandPending.current.set(pack.id, token);
    ++refreshRevision.current;
    setPendingCommands((previous) => ({
      ...previous,
      [pack.id]: { kind: pending.kind, slotId: pending.kind === 'retry' ? pending.slotId : undefined },
    }));
    const capture = await recovery.verify();
    if (!capture || !sameVNCommandScope(original.scope, capture.scope) || !recovery.isCurrent(capture) ||
        capture.scope.principal !== String(pack.owner_user_id)) {
      finishGenerationCommand(pack.id, token, !capture);
      return false;
    }
    const forget = () => {
      clearPendingVNAssetGeneration(capture.scope, pack.id, pending.key);
      if (legacyReceipts.current.get(operation)?.key === pending.key) legacyReceipts.current.delete(operation);
    };
    try {
      if (readVNCommands(capture.scope).some((command) => command.packId === pack.id)) return false;
      const request = { idempotency_key: pending.key };
      const recovered = pending.kind === 'start'
        ? await startVNAssetGeneration(pack.id, request)
        : await retryVNAssetSlot(pack.id, pending.slotId!, request);
      if (!recovery.isCurrent(capture)) return false;
      forget();
      if (selectedPackIdRef.current === pack.id) {
        setGeneration(recovered);
        setError(null);
      }
    } catch (recoveryError) {
      if (!recovery.isCurrent(capture)) return false;
      if (isMissingGenerationResource(recoveryError) || isVNGenerationRejected(recoveryError)) forget();
      if (selectedPackIdRef.current === pack.id) {
        setError(recoveryError instanceof Error ? recoveryError.message
          : `Could not reconcile the pending generation request (pack ${pack.id}, kind ${pending.kind}${pending.kind === 'retry' && typeof pending.slotId === 'number' && Number.isSafeInteger(pending.slotId) && pending.slotId > 0 ? `, slot ${pending.slotId}` : ''}).`);
      }
    } finally {
      finishGenerationCommand(pack.id, token);
    }
    return true;
  }, [finishGenerationCommand, recovery]);

  useEffect((): void => {
    if (!authority && recovery.error) setIsLoading(false);
  }, [authority, recovery.error]);

  useEffect(() => {
    if (!authority) return;
    let cancelled = false;
    const isCurrent = () => !cancelled && isAuthority(authority);

    async function loadInitialState() {
      setIsLoading(true);
      setError(null);
      try {
        const [nextPacks, matrices] = await Promise.all([
          listVNAssetPacks(),
          getStarterMatrices(),
        ]);
        if (!isCurrent()) return;
        setPacks(nextPacks);
        setStarterMatrices(matrices.matrices ?? []);
      } catch (loadError) {
        if (isCurrent()) {
          setError(loadError instanceof Error ? loadError.message : 'Failed to load VN asset packs');
        }
      } finally {
        if (isCurrent()) {
          setIsLoading(false);
        }
      }
    }

    void loadInitialState();
    return () => {
      cancelled = true;
    };
  }, [accountRevision, authority, isAuthority]);

  useEffect(() => {
    setError(null);
  }, [selectedPack?.id, accountRevision]);

  useEffect(() => {
    if (!authority || !isAuthority(authority) || selectedPack || !packs.length) return;
    const capture = captureCurrent();
    // Defer persisted selection until revalidation finishes without restarting the list.
    if (!capture && !recovery.unreadable) return;
    const selectedId = capture && sameVNCommandScope(authority, capture.scope) ? readSelectedPackId(capture.scope) : null;
    setSelectedPack(packs.find((pack) => pack.id === selectedId && String(pack.owner_user_id) === authority.principal) ?? packs[0]);
  }, [packs, selectedPack, authority, isAuthority, captureCurrent, recovery.ready, recovery.unreadable]);

  useEffect(() => {
    const capture = captureCurrent();
    if (!recovery.ready || !capture || !selectedPack || capture.scope.principal !== String(selectedPack.owner_user_id)) return;
    const key = selectedPackStorageKey(capture.scope);
    try {
      window.sessionStorage.setItem(key, String(selectedPack.id));
    } catch {
      // Pack selection still works when tab storage is unavailable.
    }
  }, [selectedPack, recovery.ready, captureCurrent]);

  useEffect(() => {
    setLoadedPackId(null);
    setGeneration(null);
    setSlots([]);
    setItems([]);
    setReadiness(null);
    const revision = ++refreshRevision.current;
    if (!selectedPack || !recovery.ready) {
      return;
    }

    let cancelled = false;
    async function loadPackDetails() {
      try {
        await refreshPackDetails(selectedPack);
      } catch {
        if (!cancelled && revision === refreshRevision.current) {
          setError('Could not load generation status. Refresh to try again.');
        }
      }
    }

    void loadPackDetails();
    return () => {
      cancelled = true;
    };
  }, [selectedPack, refreshPackDetails, recovery.ready, recovery.verifiedRevision]);

  useEffect(() => {
    if (!selectedPack || loadedPackId !== selectedPack.id || !recovery.ready) return;
    const original = recovery.captureCurrent();
    if (!original || original.scope.principal !== String(selectedPack.owner_user_id)) return;
    const operation = legacyOperation(original.scope, selectedPack.id);
    const pending = readPendingVNAssetGeneration(original.scope, selectedPack.id) ?? legacyReceipts.current.get(operation);
    if (!pending) return;
    const attempt = JSON.stringify([operation, pending.key]);
    if (autoReconciledKeys.current.has(attempt)) return;
    autoReconciledKeys.current.add(attempt);
    if (recovery.commands.some((command) => command.packId === selectedPack.id)) return;
    void (async () => {
      const reconciled = await reconcilePendingGeneration(selectedPack);
      if (reconciled && recovery.isCurrent(original) && selectedPackIdRef.current === selectedPack.id) {
        try {
          await refreshPackDetails(selectedPack, true);
        } catch {
          if (recovery.isCurrent(original)) setError('Could not refresh generation progress. Refresh to try again.');
        }
      }
    })();
  }, [selectedPack, loadedPackId, reconcilePendingGeneration, refreshPackDetails, recovery]);

  useEffect(() => {
    let cancelled = false;
    setPreflight(null);
    setPreflightError(null);
    if (!selectedPack) return;
    getVNAssetGenerationPreflight(selectedPack.id).then((result) => {
      if (!cancelled) setPreflight(result);
    }).catch(() => {
      if (!cancelled) setPreflightError('Generation configuration could not be checked. Refresh to try again.');
    });
    return () => { cancelled = true; };
  }, [selectedPack, preflightRevision]);

  useEffect(() => {
    if (!selectedPack || loadedPackId !== selectedPack.id || !['queued', 'enqueued', 'processing'].includes(generation?.status ?? '')) return;
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout>;
    const poll = async () => {
      try {
        if (!generationCommandPending.current.has(selectedPack.id)) await refreshPackDetails(selectedPack);
      } catch {
        if (!cancelled) setError('Could not refresh generation progress. Refresh to try again.');
      } finally {
        if (!cancelled) timer = setTimeout(poll, 3000);
      }
    };
    timer = setTimeout(poll, 3000);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [selectedPack, loadedPackId, generation?.status, refreshPackDetails]);

  const handleRefreshGeneration = useCallback(async () => {
    if (!selectedPack || generationCommandPending.current.has(selectedPack.id)) return;
    const original = recovery.captureCurrent();
    if (!original) return;
    setPreflightRevision((revision) => revision + 1);
    setError(null);
    try {
      if (!savedCommand) {
        const pending = original.scope.principal === String(selectedPack.owner_user_id) && (
          readPendingVNAssetGeneration(original.scope, selectedPack.id) ??
          legacyReceipts.current.get(legacyOperation(original.scope, selectedPack.id))
        );
        if (pending && !await reconcilePendingGeneration(selectedPack)) return;
      }
      if (!recovery.isCurrent(original)) return;
      await refreshPackDetails(selectedPack);
    } catch {
      if (recovery.isCurrent(original) && selectedPackIdRef.current === selectedPack.id) {
        setError('Could not refresh generation progress. Refresh to try again.');
      }
    }
  }, [selectedPack, reconcilePendingGeneration, refreshPackDetails, savedCommand, recovery]);

  const handleCreatePack = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const parsedCharacterId = Number(primaryCharacterId);
    if (!title.trim() || !Number.isInteger(parsedCharacterId) || parsedCharacterId <= 0) {
      setError('Enter a pack title and a positive character ID.');
      return;
    }

    setIsCreating(true);
    setError(null);
    try {
      const created = await createVNAssetPack({
        title: title.trim(),
        primary_character_id: parsedCharacterId,
        apply_starter_matrix: false,
      });
      setPacks((previous) => [created, ...previous.filter((pack) => pack.id !== created.id)]);
      setSelectedPack(created);
      setActiveWorkflowStep('matrix');
    } catch (createError) {
      setError(createError instanceof Error ? createError.message : 'Failed to create VN asset pack');
    } finally {
      setIsCreating(false);
    }
  }, [primaryCharacterId, title]);

  const handleApplyStarterMatrix = useCallback(async (matrixKey: string, overrides: Record<string, unknown>) => {
    if (!selectedPack || !starterMatrix) return;
    setIsApplyingMatrix(true);
    setError(null);
    try {
      const nextSlots = await applyVNAssetMatrix(selectedPack.id, matrixKey, overrides);
      const plannedOutputCount = nextSlots.reduce(
        (total, slot) => total + Math.max(0, slot.variant_count),
        0
      );
      setSlots(nextSlots);
      setSelectedPack((previous) =>
        previous && previous.id === selectedPack.id
          ? { ...previous, planned_output_count: plannedOutputCount }
          : previous
      );
      setPacks((previous) =>
        previous.map((pack) =>
          pack.id === selectedPack.id ? { ...pack, planned_output_count: plannedOutputCount } : pack
        )
      );
      setActiveWorkflowStep('generation');
    } catch (applyError) {
      setError(applyError instanceof Error ? applyError.message : 'Failed to apply starter matrix');
    } finally {
      setIsApplyingMatrix(false);
    }
  }, [selectedPack, starterMatrix]);

  const runGeneration = useCallback(async (slotId?: number, recover = false) => {
    if (!selectedPack || loadedPackId !== selectedPack.id || !recovery.ready ||
        generationCommandPending.current.has(selectedPack.id) || (savedCommand && !recover)) return;
    const original = recovery.captureCurrent();
    if (!original) return;
    const operation = legacyOperation(original.scope, selectedPack.id);
    const stored = original.scope.principal === String(selectedPack.owner_user_id) && (
      readPendingVNAssetGeneration(original.scope, selectedPack.id) ?? legacyReceipts.current.get(operation)
    );
    if (stored && !recover) {
      if (stored.kind !== (slotId === undefined ? 'start' : 'retry') ||
          (stored.kind === 'retry' && stored.slotId !== slotId)) {
        setError('Finish the previous generation request with Refresh before starting another.');
      } else {
        await reconcilePendingGeneration(selectedPack);
      }
      return;
    }
    const token = Symbol('generation-command');
    generationCommandPending.current.set(selectedPack.id, token);
    ++refreshRevision.current;
    const packId = selectedPack.id;
    const sourceBatchId = slotId === undefined ? null : generation?.failed_slot_batch_ids?.[slotId];
    const command = recover ? savedCommand : {
      packId, slotId,
      request: {
        idempotency_key: createVNAssetIdempotencyKey('vn-generation'),
        ...(slotId !== undefined && sourceBatchId != null ? { source_batch_id: sourceBatchId } : {}),
      },
    };
    if (!command) { finishGenerationCommand(packId, token); return; }
    setPendingCommands((previous) => ({ ...previous, [packId]: { kind: command.slotId === undefined ? 'start' : 'retry', slotId: command.slotId } }));
    setError(null);
    const capture = await recovery.verify();
    if (!capture || !sameVNCommandScope(original.scope, capture.scope) || !recovery.isCurrent(capture)) {
      finishGenerationCommand(packId, token, !capture);
      return;
    }
    if (!recover && capture.scope.principal === String(selectedPack.owner_user_id) && (
      readPendingVNAssetGeneration(capture.scope, packId) ?? legacyReceipts.current.get(operation)
    )) {
      if (selectedPackIdRef.current === packId) {
        setError('Finish the previous generation request with Refresh before starting another.');
      }
      finishGenerationCommand(packId, token);
      return;
    }
    if (!recovery.remember(capture, command)) {
      finishGenerationCommand(packId, token);
      return;
    }
    try {
      const nextGeneration = command.slotId === undefined
        ? await startVNAssetGeneration(packId, command.request)
        : await retryVNAssetSlot(packId, command.slotId, command.request);
      if (!recovery.isCurrent(capture)) return;
      recovery.forget(capture, command);
      if (selectedPackIdRef.current === packId) {
        setGeneration(nextGeneration);
        setActiveWorkflowStep('generation');
      }
    } catch (startError) {
      if (!recovery.isCurrent(capture)) return;
      if (!recover && isVNGenerationRejected(startError)) recovery.forget(capture, command);
      if (selectedPackIdRef.current === packId) {
        setError(startError instanceof Error ? startError.message : recover
          ? `Could not reconcile the pending generation request (pack ${packId}, kind ${command.slotId === undefined ? 'start' : 'retry'}${command.slotId === undefined ? '' : `, slot ${command.slotId}`}).`
          : 'Generation could not be started. Retry to check the same request.');
      }
    } finally {
      finishGenerationCommand(packId, token);
    }
  }, [selectedPack, loadedPackId, generation, finishGenerationCommand, recovery, savedCommand, reconcilePendingGeneration]);

  const handleCancelGeneration = useCallback(async () => {
    if (!selectedPack || !recovery.ready || savedCommand || generationCommandPending.current.has(selectedPack.id)) return;
    const packId = selectedPack.id;
    const original = recovery.captureCurrent();
    if (!original) return;
    const operation = legacyOperation(original.scope, packId);
    const pending = original.scope.principal === String(selectedPack.owner_user_id) && (
      readPendingVNAssetGeneration(original.scope, packId) ?? legacyReceipts.current.get(operation)
    );
    const token = Symbol('cancel-command');
    generationCommandPending.current.set(selectedPack.id, token);
    ++refreshRevision.current;
    setPendingCommands((previous) => ({ ...previous, [packId]: { kind: 'cancel' } }));
    setError(null);
    const capture = await recovery.verify();
    if (!capture || !sameVNCommandScope(original.scope, capture.scope) || !recovery.isCurrent(capture)) {
      finishGenerationCommand(selectedPack.id, token, !capture);
      return;
    }
    try {
      const nextGeneration = await cancelVNAssetGeneration(packId);
      if (!recovery.isCurrent(capture)) return;
      if (pending) clearPendingVNAssetGeneration(capture.scope, packId, pending.key);
      if (pending && legacyReceipts.current.get(operation)?.key === pending.key) legacyReceipts.current.delete(operation);
      if (selectedPackIdRef.current === packId) setGeneration(nextGeneration);
    } catch (cancelError) {
      if (!recovery.isCurrent(capture)) return;
      if (selectedPackIdRef.current === packId) {
        setError(cancelError instanceof Error ? cancelError.message : 'Failed to cancel generation');
      }
    } finally {
      finishGenerationCommand(packId, token);
    }
  }, [selectedPack, finishGenerationCommand, recovery, savedCommand]);

  const handleBulkReview = useCallback(async (request: VNAssetBulkReviewRequest) => {
    if (!selectedPack) return;
    setError(null);
    try {
      const reviewedItems = await bulkReviewVNAssetItems(selectedPack.id, request);
      if (selectedPackIdRef.current !== selectedPack.id) return;
      setItems((previous) => {
        const reviewedById = new Map(reviewedItems.map((item) => [item.id, item]));
        return previous.map((item) => reviewedById.get(item.id) ?? item);
      });
      await refreshPackDetails(selectedPack, true);
    } catch (reviewError) {
      setError(reviewError instanceof Error ? reviewError.message : 'Failed to update review status');
    }
  }, [refreshPackDetails, selectedPack]);

  const handleSetPreferred = useCallback(async (itemId: number) => {
    if (!selectedPack) return;
    setError(null);
    try {
      const preferredItem = await setPreferredVNAssetItem(selectedPack.id, itemId);
      if (selectedPackIdRef.current !== selectedPack.id) return;
      setItems((previous) =>
        previous.map((item) =>
          item.slot_id === preferredItem.slot_id
            ? { ...item, preferred: item.id === preferredItem.id }
            : item
        )
      );
      await refreshPackDetails(selectedPack, true);
    } catch (preferredError) {
      setError(preferredError instanceof Error ? preferredError.message : 'Failed to set preferred item');
    }
  }, [selectedPack, refreshPackDetails]);

  return (
    <main className="min-h-screen bg-bg text-text">
      <div className="mx-auto flex w-full max-w-7xl flex-col gap-6 px-6 py-6">
        <header className="flex flex-col gap-3 border-b border-border pb-4">
          <div className="flex flex-wrap items-center gap-3">
            <h1 className="text-2xl font-semibold">VN asset packs</h1>
            <Badge variant={readiness?.ready ? 'success' : 'warning'}>{readinessBadge}</Badge>
          </div>
          <p className="max-w-3xl text-sm text-text-muted">
            Offline visual-novel asset setup for character sprites, backgrounds, CGs, and reviewable generated variants.
          </p>
          <div className="flex flex-wrap gap-1" role="tablist" aria-label="VN asset workflow">
            {workflowSteps.map((step) => {
              const Icon = step.icon;
              const active = activeWorkflowStep === step.key;

              return (
                <Button
                  key={step.key}
                  aria-selected={active}
                  className="gap-2"
                  onClick={() => setActiveWorkflowStep(step.key)}
                  role="tab"
                  size="sm"
                  type="button"
                  variant={active ? 'primary' : 'secondary'}
                >
                  <Icon aria-hidden className="h-4 w-4" />
                  {step.label}
                </Button>
              );
            })}
          </div>
        </header>

        {isLoading && <p className="text-sm text-text-muted">Loading VN asset packs...</p>}
        {error && (
          <div role="alert" className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            {error}
          </div>
        )}
        {recovery.error && (
          <div role="alert" className="flex flex-col gap-2 border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
            <p>{recovery.error}</p>
            <div className="flex flex-wrap items-center gap-2">
              <Button size="sm" variant="secondary" className="gap-2" onClick={() => void recovery.verify()}>
                <RefreshCw aria-hidden className="h-4 w-4" /> Retry recovery check
              </Button>
              {recovery.unreadable && <>
                <label className="flex items-start gap-2">
                  <input type="checkbox" checked={discardConfirmed} onChange={(event) => setDiscardConfirmed(event.target.checked)} />
                  I checked server status. Discarding does not cancel accepted work and may allow duplicate generation.
                </label>
                <Button size="sm" variant="secondary" className="gap-2" disabled={!discardConfirmed} onClick={() => {
                  setDiscardConfirmed(false);
                  recovery.discardUnreadable();
                }}><Trash2 aria-hidden className="h-4 w-4" /> Discard unreadable requests</Button>
              </>}
            </div>
          </div>
        )}

        <section className="grid grid-cols-1 gap-4 lg:grid-cols-[280px_minmax(0,1fr)]">
          <PackList
            packs={packs}
            selectedPackId={selectedPack?.id}
            onSelectPack={(pack) => {
              setSelectedPack(pack);
              setActiveWorkflowStep('matrix');
            }}
          />

          <div className="grid min-w-0 grid-cols-1 gap-4">
            <PackSetup
              isCreating={isCreating}
              primaryCharacterId={primaryCharacterId}
              title={title}
              onCreatePack={handleCreatePack}
              onPrimaryCharacterIdChange={setPrimaryCharacterId}
              onTitleChange={setTitle}
            />

            <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(320px,420px)]">
              <div className="rounded-md border border-border bg-surface p-4">
                <h2 className="mb-4 text-lg font-semibold">Selected pack</h2>
                {selectedPack ? (
                  <div className="grid gap-3 sm:grid-cols-3">
                    <div>
                      <p className="text-xs uppercase tracking-normal text-text-muted">Selected pack</p>
                      <p className="font-medium">Selected pack: {selectedPack.title}</p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-normal text-text-muted">Character</p>
                      <p className="font-medium">Character {selectedPack.primary_character_id}</p>
                    </div>
                    <div>
                      <p className="text-xs uppercase tracking-normal text-text-muted">Plan</p>
                      <p className="font-medium">
                        {plannedAssetLabel(selectedPack.planned_output_count ?? 0)}
                      </p>
                    </div>
                  </div>
                ) : (
                  <p className="text-sm text-text-muted">Create or select a pack to start planning assets.</p>
                )}
              </div>

              <ReadinessPanel readiness={readiness} />
            </section>

            <section className="grid grid-cols-1 gap-4 xl:grid-cols-[minmax(0,1fr)_minmax(320px,420px)]">
              <MatrixEditor
                isApplying={isApplyingMatrix}
                matrix={starterMatrix}
                selectedPackId={selectedPack?.id}
                onApplyMatrix={handleApplyStarterMatrix}
              />
              <div className="min-w-0">
                {savedCommand && <div role="status" className="mb-3 flex flex-wrap items-center justify-between gap-2 border-l-2 border-warn px-3 py-2 text-sm">
                  <p className="min-w-0 break-words">Unconfirmed {savedCommand.slotId === undefined ? 'Start' : `Retry for slot ${savedCommand.slotId}`} request.</p>
                  <Button size="sm" variant="secondary" className="gap-2" disabled={!recovery.ready || !!selectedCommand || loadedPackId !== selectedPack?.id}
                    onClick={() => void runGeneration(undefined, true)}>
                    <RotateCcw aria-hidden className="h-4 w-4" /> Recover pending request
                  </Button>
                </div>}
                <GenerationMonitor
                  generation={generation}
                  isCancelling={isCancellingGeneration}
                  isStarting={isStartingGeneration}
                  slots={slots}
                  onCancelGeneration={handleCancelGeneration}
                  onStartGeneration={() => void runGeneration()}
                  onRetrySlot={(slotId) => void runGeneration(slotId)}
                  retryingSlotId={retryingSlotId}
                  disabled={loadedPackId !== selectedPack?.id || !recovery.ready || !!savedCommand}
                  onRefresh={selectedPack ? handleRefreshGeneration : undefined}
                  preflight={preflight}
                  preflightError={preflightError}
                />
              </div>
            </section>

            <ReviewBoard
              key={selectedPack?.id ?? 'no-pack'}
              items={items}
              onBulkReview={handleBulkReview}
              onSetPreferred={handleSetPreferred}
            />

            <PortabilityPanel selectedPack={selectedPack} />
          </div>
        </section>
      </div>
    </main>
  );
}
