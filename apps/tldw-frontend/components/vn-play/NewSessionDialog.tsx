import React, { FormEvent, useEffect, useMemo, useState } from 'react';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import { listCharacters } from '@web/lib/api/characters';
import { getVNAssetReadiness, listVNAssetPacks } from '@web/lib/api/vnAssets';
import type { CharacterSummary } from '@web/types/characters';
import type { VNAssetPack, VNAssetReadiness } from '@web/types/vn-assets';
import type { VNPlayMode, VNPlaySessionCreate } from '@web/types/vn-play';

export interface NewSessionDialogProps {
  initialMode: VNPlayMode;
  isCreating: boolean;
  open: boolean;
  onClose: () => void;
  onCreateSession: (request: VNPlaySessionCreate) => Promise<void>;
}

type SelectorMode = 'selectors' | 'manual';
type ReadinessByPackId = Record<number, VNAssetReadiness>;

const SELECT_CLASS =
  'mt-1 block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary';
const APPROVED_PACK_STATUSES = new Set(['approved', 'ready', 'runtime_ready', 'active']);

function parsePositiveInteger(value: string): number | null {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function characterName(character: CharacterSummary | null): string {
  return character?.name?.trim() || (character ? `Character ${character.id}` : 'selected character');
}

function formatTags(tags: CharacterSummary['tags']): string | null {
  if (Array.isArray(tags)) {
    const normalized = tags.map((tag) => tag.trim()).filter(Boolean);
    return normalized.length ? normalized.join(', ') : null;
  }

  if (typeof tags === 'string') {
    return tags.trim() || null;
  }

  return null;
}

function isApprovedPackStatus(status?: string | null): boolean {
  if (!status) return true;
  return APPROVED_PACK_STATUSES.has(status.toLowerCase());
}

function chooseBestPack(
  packs: VNAssetPack[],
  characterId: number | null,
  readinessByPackId: ReadinessByPackId
): VNAssetPack | null {
  if (!characterId) return null;

  const compatiblePacks = packs.filter((pack) => pack.primary_character_id === characterId);
  return (
    compatiblePacks.find((pack) => readinessByPackId[pack.id]?.ready && isApprovedPackStatus(pack.status)) ??
    compatiblePacks[0] ??
    null
  );
}

async function loadReadinessForPacks(packs: VNAssetPack[]): Promise<ReadinessByPackId> {
  const entries = await Promise.all(
    packs.map(async (pack) => {
      try {
        const readiness = await getVNAssetReadiness(pack.id);
        return [pack.id, readiness] as const;
      } catch (error) {
        const message = error instanceof Error ? error.message : 'Readiness request failed';
        return [
          pack.id,
          {
            ready: false,
            status: 'readiness_unavailable',
            warnings: [],
            errors: [`Could not load readiness for ${pack.title}: ${message}`],
          },
        ] as const;
      }
    })
  );

  return entries.reduce<ReadinessByPackId>((accumulator, [packId, readiness]) => {
    accumulator[packId] = readiness;
    return accumulator;
  }, {});
}

function buildPackOptionLabel(
  pack: VNAssetPack,
  selectedCharacter: CharacterSummary | null,
  readiness?: VNAssetReadiness
): string {
  const parts = [pack.title];
  if (selectedCharacter && pack.primary_character_id !== selectedCharacter.id) {
    parts.push(`incompatible with ${characterName(selectedCharacter)}`);
  } else if (readiness && !readiness.ready) {
    parts.push('not ready');
  }
  return parts.join(' - ');
}

function buildSelectedPackWarnings({
  contentRating,
  pack,
  readiness,
  selectedCharacter,
}: {
  contentRating: string;
  pack: VNAssetPack | null;
  readiness?: VNAssetReadiness;
  selectedCharacter: CharacterSummary | null;
}): string[] {
  if (!pack) return [];

  const warnings: string[] = [];
  if (selectedCharacter && pack.primary_character_id !== selectedCharacter.id) {
    warnings.push(`${pack.title} is attached to character ${pack.primary_character_id}, not ${characterName(selectedCharacter)}.`);
  }

  if (pack.status && !isApprovedPackStatus(pack.status)) {
    warnings.push(`Pack status is ${pack.status}; review or approve it before starting VN Play.`);
  }

  if (readiness) {
    if (!readiness.ready) {
      warnings.push(`Readiness status: ${readiness.status}.`);
    }
    warnings.push(...readiness.warnings);
    warnings.push(...readiness.errors);
  }

  if (
    pack.content_rating &&
    contentRating.trim() &&
    pack.content_rating.toLowerCase() !== contentRating.trim().toLowerCase()
  ) {
    warnings.push(`Pack content rating ${pack.content_rating} differs from session rating ${contentRating.trim()}.`);
  }

  return warnings;
}

function hasBlockingPackIssue(
  pack: VNAssetPack | null,
  selectedCharacter: CharacterSummary | null,
  readiness?: VNAssetReadiness
): boolean {
  if (!pack || !selectedCharacter) return true;
  if (pack.primary_character_id !== selectedCharacter.id) return true;
  if (pack.status && !isApprovedPackStatus(pack.status)) return true;
  if (!readiness?.ready) return true;
  return false;
}

export default function NewSessionDialog({
  initialMode,
  isCreating,
  open,
  onClose,
  onCreateSession,
}: NewSessionDialogProps) {
  const [mode, setMode] = useState<VNPlayMode>(initialMode);
  const [title, setTitle] = useState('Untitled VN play session');
  const [primaryCharacterId, setPrimaryCharacterId] = useState('1');
  const [vnAssetPackId, setVnAssetPackId] = useState('1');
  const [selectedCharacterId, setSelectedCharacterId] = useState('');
  const [selectedPackId, setSelectedPackId] = useState('');
  const [linkedChatId, setLinkedChatId] = useState('');
  const [contentRating, setContentRating] = useState('general');
  const [formError, setFormError] = useState<string | null>(null);
  const [selectorMode, setSelectorMode] = useState<SelectorMode>('selectors');
  const [characters, setCharacters] = useState<CharacterSummary[]>([]);
  const [assetPacks, setAssetPacks] = useState<VNAssetPack[]>([]);
  const [readinessByPackId, setReadinessByPackId] = useState<ReadinessByPackId>({});
  const [isLoadingSelectors, setIsLoadingSelectors] = useState(false);
  const [selectorError, setSelectorError] = useState<string | null>(null);

  useEffect(() => {
    if (open) {
      setMode(initialMode);
      setFormError(null);
      setSelectorError(null);
      setSelectorMode('selectors');
    }
  }, [initialMode, open]);

  useEffect(() => {
    if (!open) return;

    let cancelled = false;

    async function loadSelectorData() {
      setIsLoadingSelectors(true);
      try {
        const [nextCharacters, nextPacks] = await Promise.all([
          listCharacters(),
          listVNAssetPacks(),
        ]);
        const nextReadiness = await loadReadinessForPacks(nextPacks);
        if (cancelled) return;

        const firstCharacter = nextCharacters[0] ?? null;
        const nextCharacterId = firstCharacter ? String(firstCharacter.id) : '';
        const bestPack = chooseBestPack(nextPacks, firstCharacter?.id ?? null, nextReadiness);

        setCharacters(nextCharacters);
        setAssetPacks(nextPacks);
        setReadinessByPackId(nextReadiness);
        setSelectedCharacterId(nextCharacterId);
        setSelectedPackId(bestPack ? String(bestPack.id) : '');
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : 'Failed to load setup selectors';
          setSelectorError(message);
          setSelectorMode('manual');
          setCharacters([]);
          setAssetPacks([]);
          setReadinessByPackId({});
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSelectors(false);
        }
      }
    }

    void loadSelectorData();
    return () => {
      cancelled = true;
    };
  }, [open]);

  const selectedCharacterIdNumber = parsePositiveInteger(selectedCharacterId);
  const selectedPackIdNumber = parsePositiveInteger(selectedPackId);

  const selectedCharacter = useMemo(
    () => characters.find((character) => character.id === selectedCharacterIdNumber) ?? null,
    [characters, selectedCharacterIdNumber]
  );
  const selectedPack = useMemo(
    () => assetPacks.find((pack) => pack.id === selectedPackIdNumber) ?? null,
    [assetPacks, selectedPackIdNumber]
  );
  const selectedReadiness = selectedPack ? readinessByPackId[selectedPack.id] : undefined;

  const orderedPacks = useMemo(() => {
    if (!selectedCharacterIdNumber) return assetPacks;
    return [...assetPacks].sort((left, right) => {
      const leftCompatible = left.primary_character_id === selectedCharacterIdNumber;
      const rightCompatible = right.primary_character_id === selectedCharacterIdNumber;
      if (leftCompatible !== rightCompatible) {
        return leftCompatible ? -1 : 1;
      }
      return left.title.localeCompare(right.title);
    });
  }, [assetPacks, selectedCharacterIdNumber]);

  const incompatiblePacks = useMemo(() => {
    if (!selectedCharacter) return [];
    return assetPacks.filter((pack) => pack.primary_character_id !== selectedCharacter.id);
  }, [assetPacks, selectedCharacter]);

  const selectedPackWarnings = useMemo(
    () =>
      buildSelectedPackWarnings({
        contentRating,
        pack: selectedPack,
        readiness: selectedReadiness,
        selectedCharacter,
      }),
    [contentRating, selectedCharacter, selectedPack, selectedReadiness]
  );

  const selectorSubmitDisabled =
    selectorMode === 'selectors' &&
    (isLoadingSelectors || hasBlockingPackIssue(selectedPack, selectedCharacter, selectedReadiness));

  useEffect(() => {
    if (!open || selectorMode !== 'selectors') return;
    const bestPack = chooseBestPack(assetPacks, selectedCharacterIdNumber, readinessByPackId);
    setSelectedPackId(bestPack ? String(bestPack.id) : '');
  }, [assetPacks, open, readinessByPackId, selectedCharacterIdNumber, selectorMode]);

  if (!open) return null;

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const usingManualIds = selectorMode === 'manual';
    const parsedPrimaryCharacterId = usingManualIds
      ? parsePositiveInteger(primaryCharacterId)
      : selectedCharacterIdNumber;
    const parsedPackId = usingManualIds ? parsePositiveInteger(vnAssetPackId) : selectedPackIdNumber;
    const trimmedTitle = title.trim();

    if (!trimmedTitle || !parsedPrimaryCharacterId || !parsedPackId) {
      setFormError('Enter a title, character ID, and asset pack ID.');
      return;
    }

    if (!usingManualIds && hasBlockingPackIssue(selectedPack, selectedCharacter, selectedReadiness)) {
      setFormError('Select a compatible runtime-ready character and asset pack.');
      return;
    }

    setFormError(null);
    await onCreateSession({
      mode,
      title: trimmedTitle,
      primary_character_id: parsedPrimaryCharacterId,
      vn_asset_pack_id: parsedPackId,
      linked_chat_id: linkedChatId.trim() || null,
      content_rating: contentRating.trim() || 'general',
    });
  };

  const selectedCharacterTags = formatTags(selectedCharacter?.tags);

  return (
    <div className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">New VN play session</h2>
          <p className="text-sm text-text-muted">{mode === 'story' ? 'Story/CYOA' : 'Freeform'}</p>
        </div>
        <Button onClick={onClose} size="sm" type="button" variant="ghost">
          Close
        </Button>
      </div>

      <form className="grid gap-3 sm:grid-cols-2" onSubmit={handleSubmit}>
        <label className="block text-sm font-medium text-text">
          Mode
          <select
            className={SELECT_CLASS}
            value={mode}
            onChange={(event) => setMode(event.target.value as VNPlayMode)}
          >
            <option value="freeform">Freeform</option>
            <option value="story">Story/CYOA</option>
          </select>
        </label>
        <Input label="Title" value={title} onChange={(event) => setTitle(event.target.value)} />

        {selectorMode === 'manual' ? (
          <>
            {selectorError && (
              <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                Could not load setup selectors. Manual ID entry is available for this session. {selectorError}
              </div>
            )}
            <Input
              inputMode="numeric"
              label="Primary character ID"
              value={primaryCharacterId}
              onChange={(event) => setPrimaryCharacterId(event.target.value)}
            />
            <Input
              inputMode="numeric"
              label="VN asset pack ID"
              value={vnAssetPackId}
              onChange={(event) => setVnAssetPackId(event.target.value)}
            />
          </>
        ) : (
          <>
            <div>
              <label htmlFor="new-session-character" className="mb-1 block text-sm font-medium text-text">
                Character
              </label>
              <select
                className={SELECT_CLASS}
                disabled={isLoadingSelectors || characters.length === 0}
                id="new-session-character"
                value={selectedCharacterId}
                onChange={(event) => setSelectedCharacterId(event.target.value)}
              >
                <option value="">Select a character</option>
                {characters.map((character) => (
                  <option key={character.id} value={character.id}>
                    {characterName(character)}
                  </option>
                ))}
              </select>
              {isLoadingSelectors && <p className="mt-1 text-sm text-text-muted">Loading setup options...</p>}
              {!isLoadingSelectors && characters.length === 0 && (
                <div className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
                  <p className="font-medium">No characters available.</p>
                  <p>Create or import a character before starting VN Play.</p>
                </div>
              )}
              {selectedCharacter && (
                <div className="mt-2 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-muted">
                  <p className="font-medium text-text">{characterName(selectedCharacter)}</p>
                  {selectedCharacter.description && <p>{selectedCharacter.description}</p>}
                  {selectedCharacterTags && <p>{selectedCharacterTags}</p>}
                  {selectedCharacter.image_present && <p>Image attached</p>}
                </div>
              )}
            </div>

            <div>
              <label htmlFor="new-session-vn-asset-pack" className="mb-1 block text-sm font-medium text-text">
                VN asset pack
              </label>
              <select
                className={SELECT_CLASS}
                disabled={isLoadingSelectors || assetPacks.length === 0 || !selectedCharacter}
                id="new-session-vn-asset-pack"
                value={selectedPackId}
                onChange={(event) => setSelectedPackId(event.target.value)}
              >
                <option value="">Select a runtime-ready pack</option>
                {orderedPacks.map((pack) => {
                  const readiness = readinessByPackId[pack.id];
                  const compatible = selectedCharacter ? pack.primary_character_id === selectedCharacter.id : false;
                  const disabled = !compatible || !readiness?.ready || !isApprovedPackStatus(pack.status);
                  return (
                    <option key={pack.id} disabled={disabled} value={pack.id}>
                      {buildPackOptionLabel(pack, selectedCharacter, readiness)}
                    </option>
                  );
                })}
              </select>
              {!isLoadingSelectors && assetPacks.length === 0 && (
                <div className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
                  <p className="font-medium">No VN asset packs available.</p>
                  <p>Prepare or review a VN asset pack before starting VN Play.</p>
                </div>
              )}
              {!isLoadingSelectors && assetPacks.length > 0 && selectedCharacter && !selectedPack && (
                <div className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
                  No compatible runtime-ready VN asset pack is available for {characterName(selectedCharacter)}.
                </div>
              )}
              {selectedPack && (
                <div className="mt-2 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-muted">
                  <p className="font-medium text-text">{selectedPack.title}</p>
                  {selectedPack.description && <p>{selectedPack.description}</p>}
                  <p>Pack content rating: {selectedPack.content_rating || 'general'}</p>
                  <p>Trust level: new sessions start as local; review imported packs before use.</p>
                </div>
              )}
            </div>

            {incompatiblePacks.length > 0 && selectedCharacter && (
              <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                <p className="font-medium">Some packs are attached to other characters.</p>
                <ul className="mt-1 list-disc space-y-1 pl-5">
                  {incompatiblePacks.map((pack) => (
                    <li key={pack.id}>
                      {pack.title} is attached to character {pack.primary_character_id}, not {characterName(selectedCharacter)}.
                    </li>
                  ))}
                </ul>
              </div>
            )}

            {selectedPackWarnings.length > 0 && (
              <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                <p className="font-medium">Review pack readiness before starting.</p>
                <ul className="mt-1 list-disc space-y-1 pl-5">
                  {selectedPackWarnings.map((warning) => (
                    <li key={warning}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
          </>
        )}

        <Input
          label="Linked chat ID"
          placeholder="Optional"
          value={linkedChatId}
          onChange={(event) => setLinkedChatId(event.target.value)}
        />
        <Input
          label="Content rating"
          value={contentRating}
          onChange={(event) => setContentRating(event.target.value)}
        />

        {formError && (
          <div className="rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger sm:col-span-2">
            {formError}
          </div>
        )}

        <div className="flex flex-wrap justify-end gap-2 sm:col-span-2">
          <Button onClick={onClose} type="button" variant="secondary">
            Cancel
          </Button>
          <Button disabled={selectorSubmitDisabled} loading={isCreating} type="submit">
            Create session
          </Button>
        </div>
      </form>
    </div>
  );
}
