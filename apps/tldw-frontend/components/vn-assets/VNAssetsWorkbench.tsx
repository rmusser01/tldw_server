import React, { FormEvent, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Archive, ClipboardList, Images, LayoutGrid, Settings } from 'lucide-react';
import { createVNAssetIdempotencyKey } from '@web/lib/vnAssetIdempotency';
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
  const [retryingSlotId, setRetryingSlotId] = useState<number | null>(null);
  const generationCommandPending = useRef(false);
  const generationKeys = useRef(new Map<string, string>());
  const selectedPackIdRef = useRef<number | null>(null);
  const refreshRevision = useRef(0);
  const refreshInFlight = useRef<{ packId: number; revision: number; promise: Promise<void> } | null>(null);
  selectedPackIdRef.current = selectedPack?.id ?? null;
  const [readiness, setReadiness] = useState<VNAssetReadiness | null>(null);
  const [activeWorkflowStep, setActiveWorkflowStep] = useState<(typeof workflowSteps)[number]['key']>('setup');
  const [isLoading, setIsLoading] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isApplyingMatrix, setIsApplyingMatrix] = useState(false);
  const [isStartingGeneration, setIsStartingGeneration] = useState(false);
  const [isCancellingGeneration, setIsCancellingGeneration] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [title, setTitle] = useState('Untitled VN asset pack');
  const [primaryCharacterId, setPrimaryCharacterId] = useState('1');

  const starterMatrix = starterMatrices[0] ?? null;

  const readinessBadge = useMemo(() => {
    if (!readiness) return 'Setup';
    return readiness.ready ? 'Ready' : readiness.status;
  }, [readiness]);

  const refreshPackDetails = useCallback((pack: VNAssetPack, afterMutation = false): Promise<void> => {
    if (selectedPackIdRef.current !== pack.id) return Promise.resolve();
    if (afterMutation) ++refreshRevision.current;
    const revision = refreshRevision.current;
    const pending = refreshInFlight.current;
    if (pending?.packId === pack.id && pending.revision === revision) return pending.promise;
    const isCurrent = () => selectedPackIdRef.current === pack.id && revision === refreshRevision.current;

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
        setError((previous) => previous === 'Could not refresh generation progress. Refresh to try again.' ? null : previous);
      } catch (loadError) {
        if (isCurrent()) throw loadError;
      }
    };
    const promise = load().finally(() => {
      if (refreshInFlight.current?.promise === promise) refreshInFlight.current = null;
    });
    refreshInFlight.current = { packId: pack.id, revision, promise };
    return promise;
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadInitialState() {
      setIsLoading(true);
      setError(null);
      try {
        const [nextPacks, matrices] = await Promise.all([
          listVNAssetPacks(),
          getStarterMatrices(),
        ]);
        if (cancelled) return;
        setPacks(nextPacks);
        setStarterMatrices(matrices.matrices ?? []);
        setSelectedPack(nextPacks[0] ?? null);
      } catch (loadError) {
        if (!cancelled) {
          setError(loadError instanceof Error ? loadError.message : 'Failed to load VN asset packs');
        }
      } finally {
        if (!cancelled) {
          setIsLoading(false);
        }
      }
    }

    void loadInitialState();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    setLoadedPackId(null);
    setGeneration(null);
    setSlots([]);
    setItems([]);
    setReadiness(null);
    setError(null);
    const revision = ++refreshRevision.current;
    if (!selectedPack) {
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
  }, [selectedPack, refreshPackDetails]);

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
        if (!generationCommandPending.current) await refreshPackDetails(selectedPack);
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
    if (!selectedPack || generationCommandPending.current) return;
    setPreflightRevision((revision) => revision + 1);
    setError(null);
    try {
      await refreshPackDetails(selectedPack);
    } catch {
      if (selectedPackIdRef.current === selectedPack.id) {
        setError('Could not refresh generation progress. Refresh to try again.');
      }
    }
  }, [selectedPack, refreshPackDetails]);

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

  const runGeneration = useCallback(async (slotId?: number) => {
    if (!selectedPack || loadedPackId !== selectedPack.id || generationCommandPending.current) return;
    generationCommandPending.current = true;
    ++refreshRevision.current;
    const packId = selectedPack.id;
    const operation = `${packId}:${slotId ?? 'start'}`;
    const idempotencyKey = generationKeys.current.get(operation) ?? createVNAssetIdempotencyKey('vn-generation');
    generationKeys.current.set(operation, idempotencyKey);
    if (slotId === undefined) setIsStartingGeneration(true);
    else setRetryingSlotId(slotId);
    setError(null);
    try {
      const request = { idempotency_key: idempotencyKey };
      const nextGeneration = slotId === undefined
        ? await startVNAssetGeneration(packId, request)
        : await retryVNAssetSlot(packId, slotId, request);
      generationKeys.current.delete(operation);
      if (selectedPackIdRef.current === packId) {
        setGeneration(nextGeneration);
        setActiveWorkflowStep('generation');
      }
    } catch (startError) {
      if (selectedPackIdRef.current === packId) {
        setError(startError instanceof Error ? startError.message : 'Generation could not be started. Retry to check the same request.');
      }
    } finally {
      generationCommandPending.current = false;
      setIsStartingGeneration(false);
      setRetryingSlotId(null);
    }
  }, [selectedPack, loadedPackId]);

  const handleCancelGeneration = useCallback(async () => {
    if (!selectedPack || generationCommandPending.current) return;
    generationCommandPending.current = true;
    ++refreshRevision.current;
    setIsCancellingGeneration(true);
    setError(null);
    try {
      const nextGeneration = await cancelVNAssetGeneration(selectedPack.id);
      if (selectedPackIdRef.current === selectedPack.id) setGeneration(nextGeneration);
    } catch (cancelError) {
      if (selectedPackIdRef.current === selectedPack.id) {
        setError(cancelError instanceof Error ? cancelError.message : 'Failed to cancel generation');
      }
    } finally {
      generationCommandPending.current = false;
      setIsCancellingGeneration(false);
    }
  }, [selectedPack]);

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
              <GenerationMonitor
                generation={generation}
                isCancelling={isCancellingGeneration}
                isStarting={isStartingGeneration}
                slots={slots}
                onCancelGeneration={handleCancelGeneration}
                onStartGeneration={() => void runGeneration()}
                onRetrySlot={(slotId) => void runGeneration(slotId)}
                retryingSlotId={retryingSlotId}
                disabled={loadedPackId !== selectedPack?.id}
                onRefresh={selectedPack ? handleRefreshGeneration : undefined}
                preflight={preflight}
                preflightError={preflightError}
              />
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
