import React, { FormEvent, useEffect, useMemo, useRef, useState } from 'react';
import Link from 'next/link';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import { listVNPlaySetupOptions } from '@web/lib/api/vnPlay';
import type {
  VNPlayMode,
  VNPlaySessionCreate,
  VNPlaySetupAssetPackOption,
  VNPlaySetupCharacterOption,
  VNPlaySetupEmptyState,
  VNPlaySetupOptionsResponse,
  VNPlaySetupScriptVersionOption,
  VNPlaySetupWarningSummary,
} from '@web/types/vn-play';

export interface NewSessionDialogProps {
  initialMode: VNPlayMode;
  isCreating: boolean;
  open: boolean;
  onClose: () => void;
  onCreateSession: (request: VNPlaySessionCreate) => Promise<void>;
}

type SelectorMode = 'selectors' | 'manual';

const SELECT_CLASS =
  'mt-1 block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary';
const EMPTY_ASSET_PACKS: VNPlaySetupAssetPackOption[] = [];
const EMPTY_SCRIPT_VERSIONS: VNPlaySetupScriptVersionOption[] = [];
const EMPTY_EMPTY_STATES: VNPlaySetupEmptyState[] = [];

function parsePositiveInteger(value: string): number | null {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
}

function characterName(character: VNPlaySetupCharacterOption | null): string {
  return character?.name?.trim() || (character ? `Character ${character.id}` : 'selected character');
}

function formatTags(tags: string[] | undefined): string | null {
  const normalized = (tags ?? []).map((tag) => tag.trim()).filter(Boolean);
  return normalized.length ? normalized.join(', ') : null;
}

function characterOptionLabel(character: VNPlaySetupCharacterOption): string {
  const parts = [characterName(character), `#${character.id}`];
  const tags = formatTags(character.tags);
  if (tags) {
    parts.push(tags);
  }
  if (character.favorite) {
    parts.push('favorite');
  }
  return parts.join(' - ');
}

function humanizeValue(value: string): string {
  return value.replace(/_/g, ' ');
}

function requiresAcknowledgement(summary: VNPlaySetupWarningSummary | null | undefined): boolean {
  return Boolean(summary?.requires_acknowledgement);
}

function acknowledgementWarningCodes(summary: VNPlaySetupWarningSummary): string[] {
  const requiredCodes = requiredAcknowledgementWarningCodes(summary);

  if (requiredCodes.length > 0) {
    return requiredCodes;
  }

  return summary.warnings.map((warning) => warning.code).filter(Boolean);
}

function requiredAcknowledgementWarningCodes(summary: VNPlaySetupWarningSummary): string[] {
  return summary.warnings
    .filter((warning) => warning.requires_acknowledgement)
    .map((warning) => warning.code)
    .filter(Boolean);
}

function requiresPackAcknowledgement(pack: VNPlaySetupAssetPackOption | null): boolean {
  return requiresAcknowledgement(pack?.warning_summary);
}

function acknowledgementKey(pack: VNPlaySetupAssetPackOption | null): string {
  if (!pack || !requiresPackAcknowledgement(pack)) return '';
  return [pack.id, ...acknowledgementWarningCodes(pack.warning_summary)].join(':');
}

function buildSetupAcknowledgement(pack: VNPlaySetupAssetPackOption) {
  return {
    asset_pack_id: pack.id,
    warning_codes: acknowledgementWarningCodes(pack.warning_summary),
    highest_severity: pack.warning_summary.highest_severity,
  };
}

function packOptionLabel(pack: VNPlaySetupAssetPackOption): string {
  const parts = [pack.title];
  if (pack.recommended) {
    parts.push('recommended');
  }
  if (pack.compatibility.status === 'different_character') {
    parts.push('different character');
  } else if (!pack.ready) {
    parts.push('not ready');
  } else if (requiresPackAcknowledgement(pack)) {
    parts.push('review required');
  }
  return parts.join(' - ');
}

function scriptOptionLabel(script: VNPlaySetupScriptVersionOption): string {
  const parts = [`${script.title} v${script.version_number}`];
  if (script.label) {
    parts.push(script.label);
  }
  if (script.recommended) {
    parts.push('recommended');
  }
  if (!script.ready) {
    parts.push('not ready');
  } else if (requiresAcknowledgement(script.warning_summary)) {
    parts.push('review required');
  }
  return parts.join(' - ');
}

function setupCharacters(options: VNPlaySetupOptionsResponse | null): VNPlaySetupCharacterOption[] {
  if (!options) return [];
  const selected = options.selected_character;
  if (!selected || options.characters.some((character) => character.id === selected.id)) {
    return options.characters;
  }
  return [selected, ...options.characters];
}

function selectedCharacterFromOptions(
  options: VNPlaySetupOptionsResponse | null,
  selectedCharacterId: number | null
): VNPlaySetupCharacterOption | null {
  if (!options || !selectedCharacterId) return null;
  if (options.selected_character?.id === selectedCharacterId) {
    return options.selected_character;
  }
  return options.characters.find((character) => character.id === selectedCharacterId) ?? null;
}

function defaultCharacterId(
  options: VNPlaySetupOptionsResponse,
  currentCharacterId: number | null
): number | null {
  if (currentCharacterId && setupCharacters(options).some((character) => character.id === currentCharacterId)) {
    return currentCharacterId;
  }
  return options.defaults.character_id ?? options.selected_character?.id ?? options.characters[0]?.id ?? null;
}

function hasBlockingPackIssue(
  pack: VNPlaySetupAssetPackOption | null,
  selectedCharacter: VNPlaySetupCharacterOption | null
): boolean {
  if (!pack || !selectedCharacter) return true;
  if (pack.compatibility.status === 'different_character') return true;
  if (!pack.ready) return true;
  return false;
}

function defaultPackId(
  options: VNPlaySetupOptionsResponse,
  selectedCharacter: VNPlaySetupCharacterOption | null,
  currentPackId: number | null
): number | null {
  const currentPack = currentPackId
    ? options.asset_packs.find((pack) => pack.id === currentPackId) ?? null
    : null;
  if (currentPack && !hasBlockingPackIssue(currentPack, selectedCharacter)) {
    return currentPack.id;
  }
  return (
    options.defaults.asset_pack_id ??
    options.asset_packs.find((pack) => pack.recommended)?.id ??
    options.asset_packs.find((pack) => !hasBlockingPackIssue(pack, selectedCharacter))?.id ??
    null
  );
}

function packWarningMessages(pack: VNPlaySetupAssetPackOption | null): string[] {
  if (!pack) return [];
  const messages: string[] = [];
  if (!pack.ready) {
    messages.push(`Readiness status: ${pack.readiness_status}.`);
  }
  messages.push(...pack.readiness_warnings);
  messages.push(...pack.readiness_errors);
  messages.push(...pack.warning_summary.warnings.map((warning) => warning.message));
  return messages.filter(Boolean);
}

function scriptWarningMessages(script: VNPlaySetupScriptVersionOption | null): string[] {
  if (!script) return [];
  return script.warning_summary.warnings.map((warning) => warning.message).filter(Boolean);
}

function emptyStatesFor(
  emptyStates: VNPlaySetupEmptyState[],
  codes: string[]
): VNPlaySetupEmptyState[] {
  const codeSet = new Set(codes);
  return emptyStates.filter((state) => codeSet.has(state.code));
}

function EmptyStateGuidance({
  states,
  workspaceHref,
  workspaceLabel,
}: {
  states: VNPlaySetupEmptyState[];
  workspaceHref: string;
  workspaceLabel: string;
}) {
  if (states.length === 0) return null;
  return (
    <div className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
      {states.map((state) => (
        <p key={`${state.code}-${state.scope}`}>{state.message}</p>
      ))}
      <a className="mt-1 inline-block underline" href={workspaceHref}>
        Open {workspaceLabel}
      </a>
    </div>
  );
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
  const [scriptId, setScriptId] = useState('');
  const [scriptVersionId, setScriptVersionId] = useState('');
  const [selectedCharacterId, setSelectedCharacterId] = useState('');
  const [selectedPackId, setSelectedPackId] = useState('');
  const [selectedScriptVersionId, setSelectedScriptVersionId] = useState('');
  const [linkedChatId, setLinkedChatId] = useState('');
  const [contentRating, setContentRating] = useState('general');
  const [formError, setFormError] = useState<string | null>(null);
  const [selectorMode, setSelectorMode] = useState<SelectorMode>('selectors');
  const [setupOptions, setSetupOptions] = useState<VNPlaySetupOptionsResponse | null>(null);
  const [isLoadingSelectors, setIsLoadingSelectors] = useState(false);
  const [selectorError, setSelectorError] = useState<string | null>(null);
  const [acknowledgedSetupWarnings, setAcknowledgedSetupWarnings] = useState(false);
  const selectedPackIdRef = useRef(selectedPackId);
  const selectedScriptVersionIdRef = useRef(selectedScriptVersionId);
  const applyingDefaultCharacterIdRef = useRef<string | null>(null);

  useEffect(() => {
    selectedPackIdRef.current = selectedPackId;
  }, [selectedPackId]);

  useEffect(() => {
    selectedScriptVersionIdRef.current = selectedScriptVersionId;
  }, [selectedScriptVersionId]);

  useEffect(() => {
    if (open) {
      applyingDefaultCharacterIdRef.current = null;
      setMode(initialMode);
      setTitle('Untitled VN play session');
      setPrimaryCharacterId('1');
      setVnAssetPackId('1');
      setScriptId('');
      setScriptVersionId('');
      setLinkedChatId('');
      setFormError(null);
      setSelectorError(null);
      setSelectorMode('selectors');
      setSetupOptions(null);
      setSelectedCharacterId('');
      setSelectedPackId('');
      setSelectedScriptVersionId('');
      setContentRating('general');
      setAcknowledgedSetupWarnings(false);
    }
  }, [initialMode, open]);

  useEffect(() => {
    if (!open || selectorMode !== 'selectors') return;
    if (applyingDefaultCharacterIdRef.current === selectedCharacterId) {
      applyingDefaultCharacterIdRef.current = null;
      return;
    }

    let cancelled = false;

    async function loadSetupOptions() {
      setIsLoadingSelectors(true);
      try {
        const selectedCharacterIdNumber = parsePositiveInteger(selectedCharacterId);
        const nextOptions = await listVNPlaySetupOptions({
          content_rating: contentRating.trim() || 'general',
          mode,
          ...(selectedCharacterIdNumber ? { selected_character_id: selectedCharacterIdNumber } : {}),
        });
        if (cancelled) return;

        const nextCharacterId = defaultCharacterId(nextOptions, selectedCharacterIdNumber);
        const nextSelectedCharacter = selectedCharacterFromOptions(nextOptions, nextCharacterId);
        const nextPackId = defaultPackId(
          nextOptions,
          nextSelectedCharacter,
          parsePositiveInteger(selectedPackIdRef.current)
        );
        const currentScriptVersionId = parsePositiveInteger(selectedScriptVersionIdRef.current);
        const nextScriptVersions = nextOptions.script_versions ?? EMPTY_SCRIPT_VERSIONS;
        const nextScriptVersion =
          nextScriptVersions.find((script) => script.id === currentScriptVersionId) ??
          (nextOptions.defaults.script_version_id
            ? nextScriptVersions.find((script) => script.id === nextOptions.defaults.script_version_id)
            : undefined) ??
          nextScriptVersions.find((script) => script.recommended) ??
          nextScriptVersions.find((script) => script.ready) ??
          nextScriptVersions[0];
        const nextCharacterIdValue = nextCharacterId ? String(nextCharacterId) : '';
        const nextPackIdValue = nextPackId ? String(nextPackId) : '';

        setSetupOptions(nextOptions);
        if (nextCharacterIdValue !== selectedCharacterId) {
          applyingDefaultCharacterIdRef.current = nextCharacterIdValue;
          setSelectedCharacterId(nextCharacterIdValue);
        }
        setSelectedPackId(nextPackIdValue);
        setSelectedScriptVersionId(nextScriptVersion ? String(nextScriptVersion.id) : '');
      } catch (error) {
        if (!cancelled) {
          const message = error instanceof Error ? error.message : 'Failed to load setup options';
          setSelectorError(message);
          setSelectorMode('manual');
          setSetupOptions(null);
        }
      } finally {
        if (!cancelled) {
          setIsLoadingSelectors(false);
        }
      }
    }

    void loadSetupOptions();
    return () => {
      cancelled = true;
    };
  }, [contentRating, mode, open, selectedCharacterId, selectorMode]);

  const selectedCharacterIdNumber = parsePositiveInteger(selectedCharacterId);
  const selectedPackIdNumber = parsePositiveInteger(selectedPackId);
  const selectedScriptVersionIdNumber = parsePositiveInteger(selectedScriptVersionId);
  const characters = useMemo(() => setupCharacters(setupOptions), [setupOptions]);
  const assetPacks = setupOptions?.asset_packs ?? EMPTY_ASSET_PACKS;
  const scriptVersions = setupOptions?.script_versions ?? EMPTY_SCRIPT_VERSIONS;
  const emptyStates = setupOptions?.empty_states ?? EMPTY_EMPTY_STATES;

  const selectedCharacter = useMemo(
    () => selectedCharacterFromOptions(setupOptions, selectedCharacterIdNumber),
    [setupOptions, selectedCharacterIdNumber]
  );
  const selectedPack = useMemo(
    () => assetPacks.find((pack) => pack.id === selectedPackIdNumber) ?? null,
    [assetPacks, selectedPackIdNumber]
  );
  const selectedScriptVersion = useMemo(
    () => scriptVersions.find((script) => script.id === selectedScriptVersionIdNumber) ?? null,
    [scriptVersions, selectedScriptVersionIdNumber]
  );
  const selectedScriptAssetPack = useMemo(
    () =>
      selectedScriptVersion
        ? assetPacks.find((pack) => pack.id === selectedScriptVersion.asset_pack_id) ?? null
        : null,
    [assetPacks, selectedScriptVersion]
  );
  const selectedPackAcknowledgementKey = useMemo(
    () =>
      mode === 'scripted_story' && selectedScriptVersion
        ? [
            selectedScriptVersion.id,
            ...acknowledgementWarningCodes(selectedScriptVersion.warning_summary),
          ].join(':')
        : acknowledgementKey(selectedPack),
    [mode, selectedPack, selectedScriptVersion]
  );

  useEffect(() => {
    setAcknowledgedSetupWarnings(false);
  }, [selectedPackAcknowledgementKey]);

  const selectedPackWarnings = useMemo(() => packWarningMessages(selectedPack), [selectedPack]);
  const selectedScriptWarnings = useMemo(
    () => scriptWarningMessages(selectedScriptVersion),
    [selectedScriptVersion]
  );
  const selectedPackRequiresAcknowledgement = requiresPackAcknowledgement(selectedPack);
  const selectedScriptRequiresAcknowledgement = requiresAcknowledgement(selectedScriptVersion?.warning_summary);
  const incompatiblePacks = useMemo(
    () => assetPacks.filter((pack) => pack.compatibility.status === 'different_character'),
    [assetPacks]
  );
  const selectorSubmitDisabled =
    selectorMode === 'selectors' &&
    (isLoadingSelectors ||
      (mode === 'scripted_story'
        ? !selectedScriptVersion ||
          !selectedScriptVersion.ready ||
          !selectedScriptAssetPack ||
          (selectedScriptRequiresAcknowledgement && !acknowledgedSetupWarnings)
        : hasBlockingPackIssue(selectedPack, selectedCharacter) ||
          (selectedPackRequiresAcknowledgement && !acknowledgedSetupWarnings)));

  if (!open) return null;

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const usingManualIds = selectorMode === 'manual';
    const parsedPrimaryCharacterId = usingManualIds
      ? parsePositiveInteger(primaryCharacterId)
      : mode === 'scripted_story'
        ? selectedScriptAssetPack?.primary_character_id ?? null
        : selectedCharacterIdNumber;
    const parsedPackId = usingManualIds
      ? parsePositiveInteger(vnAssetPackId)
      : mode === 'scripted_story'
        ? selectedScriptVersion?.asset_pack_id ?? null
        : selectedPackIdNumber;
    const parsedScriptId = usingManualIds && mode === 'scripted_story' ? parsePositiveInteger(scriptId) : null;
    const parsedScriptVersionId =
      usingManualIds && mode === 'scripted_story' ? parsePositiveInteger(scriptVersionId) : null;
    const trimmedTitle = title.trim();

    if (!trimmedTitle || !parsedPrimaryCharacterId || !parsedPackId) {
      setFormError('Enter a title, character ID, and asset pack ID.');
      return;
    }
    if (usingManualIds && mode === 'scripted_story' && (!parsedScriptId || !parsedScriptVersionId)) {
      setFormError('Enter script ID and script version ID for scripted story sessions.');
      return;
    }

    if (!usingManualIds && mode === 'scripted_story' && (!selectedScriptVersion || !selectedScriptVersion.ready || !selectedScriptAssetPack)) {
      setFormError('Select a runtime-ready published script version.');
      return;
    }
    if (!usingManualIds && mode !== 'scripted_story' && hasBlockingPackIssue(selectedPack, selectedCharacter)) {
      setFormError('Select a compatible runtime-ready character and asset pack.');
      return;
    }
    if (
      !usingManualIds &&
      ((mode === 'scripted_story' && selectedScriptRequiresAcknowledgement) ||
        (mode !== 'scripted_story' && selectedPackRequiresAcknowledgement)) &&
      !acknowledgedSetupWarnings
    ) {
      setFormError('Acknowledge setup warnings before creating this session.');
      return;
    }

    setFormError(null);
    const request: VNPlaySessionCreate = {
      mode,
      title: trimmedTitle,
      primary_character_id: parsedPrimaryCharacterId,
      vn_asset_pack_id: parsedPackId,
      linked_chat_id: linkedChatId.trim() || null,
      content_rating:
        !usingManualIds && mode === 'scripted_story' && selectedScriptVersion
          ? selectedScriptVersion.content_rating
          : contentRating.trim() || 'general',
    };

    if (!usingManualIds && mode === 'scripted_story' && selectedScriptVersion) {
      request.script_id = selectedScriptVersion.script_id;
      request.script_version_id = selectedScriptVersion.id;
      if (selectedScriptRequiresAcknowledgement && acknowledgedSetupWarnings) {
        request.acknowledgements = acknowledgementWarningCodes(selectedScriptVersion.warning_summary);
      }
    } else if (usingManualIds && mode === 'scripted_story' && parsedScriptId && parsedScriptVersionId) {
      request.script_id = parsedScriptId;
      request.script_version_id = parsedScriptVersionId;
    } else if (!usingManualIds && selectedPackRequiresAcknowledgement && acknowledgedSetupWarnings && selectedPack) {
      request.settings = {
        setup_acknowledgements: [buildSetupAcknowledgement(selectedPack)],
      };
    }

    await onCreateSession(request);
  };

  const selectedCharacterTags = formatTags(selectedCharacter?.tags);
  const characterEmptyStates = emptyStatesFor(emptyStates, [
    'no_characters',
    'selected_character_not_found',
  ]);
  const packEmptyStates = emptyStatesFor(emptyStates, [
    'no_asset_packs',
    'no_ready_packs',
    'no_compatible_packs',
  ]);
  const scriptEmptyStates = emptyStatesFor(emptyStates, [
    'no_script_versions',
    'no_ready_script_versions',
    'no_published_scripts',
  ]);

  return (
    <div className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex items-start justify-between gap-3">
        <div>
          <h2 className="text-lg font-semibold">New VN play session</h2>
          <p className="text-sm text-text-muted">
            {mode === 'scripted_story' ? 'Scripted Story' : mode === 'story' ? 'Story/CYOA' : 'Freeform'}
          </p>
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
            <option value="scripted_story">Scripted Story</option>
          </select>
        </label>
        <Input label="Title" value={title} onChange={(event) => setTitle(event.target.value)} />

        {selectorMode === 'manual' ? (
          <>
            {selectorError && (
              <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                Could not load setup options. Manual ID entry is available for this session. {selectorError}
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
            {mode === 'scripted_story' && (
              <>
                <Input
                  inputMode="numeric"
                  label="Script ID"
                  value={scriptId}
                  onChange={(event) => setScriptId(event.target.value)}
                />
                <Input
                  inputMode="numeric"
                  label="Script version ID"
                  value={scriptVersionId}
                  onChange={(event) => setScriptVersionId(event.target.value)}
                />
              </>
            )}
          </>
        ) : (
          <>
            {mode === 'scripted_story' ? (
              <div className="sm:col-span-2">
                <label htmlFor="new-session-script-version" className="mb-1 block text-sm font-medium text-text">
                  Published script version
                </label>
                <select
                  className={SELECT_CLASS}
                  disabled={isLoadingSelectors || scriptVersions.length === 0}
                  id="new-session-script-version"
                  value={selectedScriptVersionId}
                  onChange={(event) => setSelectedScriptVersionId(event.target.value)}
                >
                  <option value="">Select a published script version</option>
                  {scriptVersions.map((script) => (
                    <option key={script.id} disabled={!script.ready} value={script.id}>
                      {scriptOptionLabel(script)}
                    </option>
                  ))}
                </select>
                {!isLoadingSelectors && (
                  <EmptyStateGuidance
                    states={scriptEmptyStates}
                    workspaceHref="/vn-scripts"
                    workspaceLabel="VN scripts"
                  />
                )}
                {!isLoadingSelectors && scriptVersions.length === 0 && scriptEmptyStates.length === 0 && (
                  <div className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
                    No published VN script versions are available.{' '}
                    <Link className="underline" href="/vn-scripts">
                      Open VN scripts
                    </Link>
                  </div>
                )}
                {selectedScriptVersion && (
                  <div className="mt-2 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-muted">
                    <p className="font-medium text-text">
                      {selectedScriptVersion.title} v{selectedScriptVersion.version_number}
                    </p>
                    <p>Content rating: {selectedScriptVersion.content_rating || 'general'}</p>
                    <p>Asset pack: {selectedScriptVersion.asset_pack_id}</p>
                    <p>Policy profile: {selectedScriptVersion.policy_profile_id}</p>
                    <p>Generation profile: {selectedScriptVersion.generation_profile_key}</p>
                  </div>
                )}
              </div>
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
                    {characterOptionLabel(character)}
                  </option>
                ))}
              </select>
              {isLoadingSelectors && <p className="mt-1 text-sm text-text-muted">Loading setup options...</p>}
              {!isLoadingSelectors && (
                <EmptyStateGuidance
                  states={characterEmptyStates}
                  workspaceHref="/characters"
                  workspaceLabel="characters"
                />
              )}
              {selectedCharacter && (
                <div className="mt-2 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-muted">
                  <p className="font-medium text-text">{characterName(selectedCharacter)}</p>
                  {selectedCharacter.description_preview && <p>{selectedCharacter.description_preview}</p>}
                  {selectedCharacterTags && <p>{selectedCharacterTags}</p>}
                  {selectedCharacter.has_image && <p>Image attached</p>}
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
                {assetPacks.map((pack) => (
                  <option
                    key={pack.id}
                    disabled={hasBlockingPackIssue(pack, selectedCharacter)}
                    value={pack.id}
                  >
                    {packOptionLabel(pack)}
                  </option>
                ))}
              </select>
              {!isLoadingSelectors && (
                <EmptyStateGuidance
                  states={packEmptyStates}
                  workspaceHref="/vn-assets"
                  workspaceLabel="VN asset packs"
                />
              )}
              {!isLoadingSelectors && assetPacks.length > 0 && selectedCharacter && !selectedPack && (
                <div className="mt-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn">
                  No compatible runtime-ready VN asset pack is available for {characterName(selectedCharacter)}.
                </div>
              )}
              {selectedPack && (
                <div className="mt-2 rounded-md border border-border bg-bg px-3 py-2 text-sm text-text-muted">
                  <p className="font-medium text-text">{selectedPack.title}</p>
                  <p>Pack content rating: {selectedPack.content_rating || 'general'}</p>
                  <p>Trust level: {humanizeValue(selectedPack.trust_level)}</p>
                  <p>Readiness: {humanizeValue(selectedPack.readiness_status)}</p>
                </div>
              )}
            </div>
              </>
            )}

            {mode !== 'scripted_story' && incompatiblePacks.length > 0 && selectedCharacter && (
              <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                <p className="font-medium">Some packs are attached to other characters.</p>
                <ul className="mt-1 list-disc space-y-1 pl-5">
                  {incompatiblePacks.map((pack) => {
                    const message =
                      pack.warning_summary.warnings[0]?.message ??
                      `${pack.title} is attached to character ${pack.primary_character_id}, not ${characterName(selectedCharacter)}.`;
                    return <li key={pack.id}>{message}</li>;
                  })}
                </ul>
              </div>
            )}

            {mode !== 'scripted_story' && selectedPackWarnings.length > 0 && (
              <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                <p className="font-medium">Review pack readiness before starting.</p>
                <ul className="mt-1 list-disc space-y-1 pl-5">
                  {selectedPackWarnings.map((warning, index) => (
                    <li key={`${warning}-${index}`}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
            {mode === 'scripted_story' && selectedScriptWarnings.length > 0 && (
              <div className="rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                <p className="font-medium">Review script readiness before starting.</p>
                <ul className="mt-1 list-disc space-y-1 pl-5">
                  {selectedScriptWarnings.map((warning, index) => (
                    <li key={`${warning}-${index}`}>{warning}</li>
                  ))}
                </ul>
              </div>
            )}
            {((mode === 'scripted_story' && selectedScriptRequiresAcknowledgement) ||
              (mode !== 'scripted_story' && selectedPackRequiresAcknowledgement)) && (
              <label className="flex items-start gap-2 rounded-md border border-warn/30 bg-warn/10 px-3 py-2 text-sm text-warn sm:col-span-2">
                <input
                  checked={acknowledgedSetupWarnings}
                  className="mt-1"
                  onChange={(event) => setAcknowledgedSetupWarnings(event.target.checked)}
                  type="checkbox"
                />
                <span>I understand and want to proceed with these warnings.</span>
              </label>
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
