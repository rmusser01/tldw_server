import React, { FormEvent, useCallback, useEffect, useMemo, useState } from 'react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import { Input } from '@web/components/ui/Input';
import JsonEditor from '@web/components/ui/JsonEditor';
import JsonViewer from '@web/components/ui/JsonViewer';
import {
  createVNScript,
  evaluateVNScriptVersionPolicy,
  getVNScriptDiagnostics,
  getVNScriptDraft,
  getVNScriptManifestSnapshot,
  listVNScripts,
  listVNScriptVersions,
  publishVNScript,
  putVNScriptDraft,
  validateVNScriptDraft,
} from '@web/lib/api/vnScripts';
import type {
  VNScriptContentRating,
  VNScriptDiagnosticsResponse,
  VNScriptDraftResponse,
  VNScriptManifestSnapshotResponse,
  VNScriptResponse,
  VNScriptValidationResponse,
  VNScriptVersionPolicyEvaluateResponse,
  VNScriptVersionResponse,
} from '@web/types/vn-scripts';

const contentRatings: VNScriptContentRating[] = ['general', 'teen', 'suggestive', 'mature'];

function formatJson(value: unknown): string {
  return JSON.stringify(value ?? {}, null, 2);
}

function errorMessage(error: unknown, fallback: string): string {
  if (error instanceof Error && error.message) return error.message;
  return fallback;
}

function createPublishIdempotencyKey(scriptId: number): string {
  const suffix =
    typeof crypto !== 'undefined' && 'randomUUID' in crypto
      ? crypto.randomUUID()
      : Math.random().toString(36).slice(2);
  return `vn-script-publish-${scriptId}-${Date.now()}-${suffix}`;
}

function validationBadge(validation: Record<string, unknown>): string {
  if (typeof validation.valid === 'boolean') {
    return validation.valid ? 'Valid' : 'Invalid';
  }
  if (typeof validation.status === 'string') return validation.status;
  return 'Validation saved';
}

function statusVariant(status: string): 'neutral' | 'success' | 'warning' | 'danger' | 'info' {
  if (status === 'ready' || status === 'published') return 'success';
  if (status === 'archived') return 'neutral';
  if (status === 'draft') return 'warning';
  return 'info';
}

export default function VNScriptsWorkbench() {
  const [scripts, setScripts] = useState<VNScriptResponse[]>([]);
  const [selectedScript, setSelectedScript] = useState<VNScriptResponse | null>(null);
  const [draft, setDraft] = useState<VNScriptDraftResponse | null>(null);
  const [draftText, setDraftText] = useState('{}');
  const [versions, setVersions] = useState<VNScriptVersionResponse[]>([]);
  const [validation, setValidation] = useState<VNScriptValidationResponse | null>(null);
  const [diagnostics, setDiagnostics] = useState<VNScriptDiagnosticsResponse | null>(null);
  const [manifestSnapshots, setManifestSnapshots] = useState<Record<number, VNScriptManifestSnapshotResponse>>({});
  const [policySummaries, setPolicySummaries] = useState<Record<number, VNScriptVersionPolicyEvaluateResponse>>({});
  const [isLoadingScripts, setIsLoadingScripts] = useState(true);
  const [isCreating, setIsCreating] = useState(false);
  const [isSavingDraft, setIsSavingDraft] = useState(false);
  const [isValidating, setIsValidating] = useState(false);
  const [isLoadingDiagnostics, setIsLoadingDiagnostics] = useState(false);
  const [isPublishing, setIsPublishing] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [editorError, setEditorError] = useState<string | null>(null);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  const [title, setTitle] = useState('Untitled VN script');
  const [primaryAssetPackId, setPrimaryAssetPackId] = useState('1');
  const [policyProfileId, setPolicyProfileId] = useState('');
  const [generationProfileId, setGenerationProfileId] = useState('');
  const [contentRating, setContentRating] = useState<VNScriptContentRating>('teen');
  const [publishLabel, setPublishLabel] = useState('');

  const selectedMeta = useMemo(() => {
    if (!selectedScript) return null;
    return [
      ['Asset pack', selectedScript.primary_asset_pack_id],
      ['Policy', selectedScript.policy_profile_id],
      ['Generation', selectedScript.generation_profile_id],
      ['Rating', selectedScript.content_rating],
    ];
  }, [selectedScript]);

  const refreshVersions = useCallback(async (scriptId: number) => {
    const nextVersions = await listVNScriptVersions(scriptId);
    setVersions(nextVersions.items ?? []);
  }, []);

  useEffect(() => {
    let cancelled = false;

    async function loadScripts() {
      setIsLoadingScripts(true);
      setError(null);
      try {
        const response = await listVNScripts({ limit: 25, offset: 0 });
        if (cancelled) return;
        const nextScripts = response.items ?? [];
        setScripts(nextScripts);
        setSelectedScript(nextScripts[0] ?? null);
      } catch (loadError) {
        if (!cancelled) setError(errorMessage(loadError, 'Failed to load VN scripts'));
      } finally {
        if (!cancelled) setIsLoadingScripts(false);
      }
    }

    void loadScripts();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (!selectedScript) {
      setDraft(null);
      setDraftText('{}');
      setVersions([]);
      setValidation(null);
      setDiagnostics(null);
      setManifestSnapshots({});
      setPolicySummaries({});
      return;
    }

    let cancelled = false;
    async function loadScriptDetails() {
      setError(null);
      setEditorError(null);
      setStatusMessage(null);
      setValidation(null);
      setDiagnostics(null);
      setManifestSnapshots({});
      setPolicySummaries({});
      try {
        const [nextDraft, nextVersions] = await Promise.all([
          getVNScriptDraft(selectedScript.id),
          listVNScriptVersions(selectedScript.id),
        ]);
        if (cancelled) return;
        setDraft(nextDraft);
        setDraftText(formatJson(nextDraft.draft));
        setVersions(nextVersions.items ?? []);
      } catch (loadError) {
        if (!cancelled) {
          setDraft(null);
          setDraftText('{}');
          setVersions([]);
          setError(errorMessage(loadError, 'Failed to load script draft or versions'));
        }
      }
    }

    void loadScriptDetails();
    return () => {
      cancelled = true;
    };
  }, [selectedScript]);

  const parseDraftText = useCallback((): Record<string, unknown> | null => {
    try {
      const parsed = JSON.parse(draftText);
      if (!parsed || typeof parsed !== 'object' || Array.isArray(parsed)) {
        setEditorError('Draft JSON must be an object.');
        return null;
      }
      setEditorError(null);
      return parsed as Record<string, unknown>;
    } catch (parseError) {
      setEditorError(
        `Draft JSON is invalid: ${parseError instanceof Error ? parseError.message : 'Parse failed'}`
      );
      return null;
    }
  }, [draftText]);

  const handleCreateScript = useCallback(async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const parsedAssetPackId = Number(primaryAssetPackId);
    if (!title.trim() || !Number.isInteger(parsedAssetPackId) || parsedAssetPackId <= 0) {
      setError('Enter a title and a positive primary asset pack ID.');
      return;
    }

    setIsCreating(true);
    setError(null);
    setStatusMessage(null);
    try {
      const created = await createVNScript({
        title: title.trim(),
        primary_asset_pack_id: parsedAssetPackId,
        ...(policyProfileId.trim() ? { policy_profile_id: policyProfileId.trim() } : {}),
        ...(generationProfileId.trim() ? { generation_profile_id: generationProfileId.trim() } : {}),
        content_rating: contentRating,
      });
      setScripts((previous) => [created, ...previous.filter((script) => script.id !== created.id)]);
      setSelectedScript(created);
      setStatusMessage(`Created script #${created.id}.`);
    } catch (createError) {
      setError(errorMessage(createError, 'Failed to create VN script'));
    } finally {
      setIsCreating(false);
    }
  }, [contentRating, generationProfileId, policyProfileId, primaryAssetPackId, title]);

  const handleSaveDraft = useCallback(async () => {
    if (!selectedScript || !draft) return;
    const parsed = parseDraftText();
    if (!parsed) return;

    setIsSavingDraft(true);
    setError(null);
    setStatusMessage(null);
    try {
      const updated = await putVNScriptDraft(selectedScript.id, {
        if_revision: draft.revision,
        draft: parsed,
      });
      setDraft(updated);
      setDraftText(formatJson(updated.draft));
      setStatusMessage(`Draft saved at revision ${updated.revision}.`);
    } catch (saveError) {
      setError(errorMessage(saveError, 'Failed to save draft'));
    } finally {
      setIsSavingDraft(false);
    }
  }, [draft, parseDraftText, selectedScript]);

  const handleValidate = useCallback(async () => {
    if (!selectedScript) return;
    const requestDraft = parseDraftText();
    if (!requestDraft) return;

    setIsValidating(true);
    setError(null);
    setStatusMessage(null);
    try {
      const result = await validateVNScriptDraft(selectedScript.id, {
        draft: requestDraft,
      });
      setValidation(result);
    } catch (validateError) {
      setError(errorMessage(validateError, 'Failed to validate draft'));
    } finally {
      setIsValidating(false);
    }
  }, [parseDraftText, selectedScript]);

  const handleDiagnostics = useCallback(async () => {
    if (!selectedScript) return;
    setIsLoadingDiagnostics(true);
    setError(null);
    try {
      const result = await getVNScriptDiagnostics(selectedScript.id);
      setDiagnostics(result);
    } catch (diagnosticsError) {
      setError(errorMessage(diagnosticsError, 'Failed to load diagnostics'));
    } finally {
      setIsLoadingDiagnostics(false);
    }
  }, [selectedScript]);

  const handlePublish = useCallback(async () => {
    if (!selectedScript || !draft) return;
    setIsPublishing(true);
    setError(null);
    setStatusMessage(null);
    try {
      const response = await publishVNScript(selectedScript.id, {
        draft_revision: draft.revision,
        label: publishLabel.trim() || null,
        idempotency_key: createPublishIdempotencyKey(selectedScript.id),
        acknowledgements: [],
      });
      setStatusMessage(`Published version ${response.version_number}.`);
      await refreshVersions(selectedScript.id);
    } catch (publishError) {
      setError(errorMessage(publishError, 'Failed to publish script'));
    } finally {
      setIsPublishing(false);
    }
  }, [draft, publishLabel, refreshVersions, selectedScript]);

  const handleManifestSnapshot = useCallback(async (version: VNScriptVersionResponse) => {
    if (!selectedScript) return;
    setError(null);
    try {
      const snapshot = await getVNScriptManifestSnapshot(selectedScript.id, version.id);
      setManifestSnapshots((previous) => ({ ...previous, [version.id]: snapshot }));
    } catch (manifestError) {
      setError(errorMessage(manifestError, 'Failed to load manifest snapshot'));
    }
  }, [selectedScript]);

  const handlePolicyEvaluate = useCallback(async (version: VNScriptVersionResponse) => {
    if (!selectedScript) return;
    setError(null);
    try {
      const summary = await evaluateVNScriptVersionPolicy(selectedScript.id, version.id);
      setPolicySummaries((previous) => ({ ...previous, [version.id]: summary }));
    } catch (policyError) {
      setError(errorMessage(policyError, 'Failed to evaluate version policy'));
    }
  }, [selectedScript]);

  return (
    <main className="min-h-screen bg-bg text-text">
      <div className="flex min-h-screen flex-col gap-3 p-4">
        <header className="flex flex-wrap items-center justify-between gap-3 border-b border-border pb-3">
          <div>
            <h1 className="text-xl font-semibold">VN Scripts</h1>
            <p className="text-sm text-text-muted">Author JSON drafts, inspect backend validation, and publish immutable versions.</p>
          </div>
          <div className="flex items-center gap-2 text-xs text-text-muted">
            <Badge variant="info">Backend-owned validation</Badge>
            <span>{scripts.length} loaded</span>
          </div>
        </header>

        {(error || editorError || statusMessage) && (
          <section className="grid gap-2 md:grid-cols-3">
            {error && <div className="rounded-md border border-danger/30 bg-danger/10 p-2 text-sm text-danger">{error}</div>}
            {editorError && <div className="rounded-md border border-warn/30 bg-warn/10 p-2 text-sm text-warn">{editorError}</div>}
            {statusMessage && <div className="rounded-md border border-success/30 bg-success/10 p-2 text-sm text-success">{statusMessage}</div>}
          </section>
        )}

        <div className="grid min-h-0 flex-1 gap-3 xl:grid-cols-[320px_minmax(0,1fr)_420px]">
          <aside className="flex min-h-0 flex-col gap-3 border-r border-border pr-3">
            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-2 flex items-center justify-between">
                <h2 className="text-sm font-semibold">Scripts</h2>
                {isLoadingScripts && <span className="text-xs text-text-muted">Loading...</span>}
              </div>
              <div className="max-h-80 space-y-2 overflow-auto">
                {scripts.length === 0 && !isLoadingScripts && (
                  <p className="text-sm text-text-muted">No VN scripts yet.</p>
                )}
                {scripts.map((script) => (
                  <button
                    key={script.id}
                    type="button"
                    onClick={() => setSelectedScript(script)}
                    className={`w-full rounded-md border p-2 text-left text-sm transition ${
                      selectedScript?.id === script.id
                        ? 'border-primary bg-primary/10'
                        : 'border-border bg-bg hover:bg-surface2'
                    }`}
                  >
                    <span className="block font-medium">{script.title}</span>
                    <span className="mt-1 flex flex-wrap items-center gap-2 text-xs text-text-muted">
                      <Badge variant={statusVariant(script.status)}>{script.status}</Badge>
                      <span>#{script.id}</span>
                      <span>Pack {script.primary_asset_pack_id}</span>
                    </span>
                  </button>
                ))}
              </div>
            </section>

            <form onSubmit={handleCreateScript} className="rounded-md border border-border bg-surface p-3">
              <h2 className="mb-3 text-sm font-semibold">Create script</h2>
              <div className="space-y-3">
                <Input label="Title" value={title} onChange={(event) => setTitle(event.target.value)} />
                <Input
                  label="Primary asset pack ID"
                  inputMode="numeric"
                  value={primaryAssetPackId}
                  onChange={(event) => setPrimaryAssetPackId(event.target.value)}
                />
                <Input
                  label="Policy profile ID"
                  value={policyProfileId}
                  onChange={(event) => setPolicyProfileId(event.target.value)}
                />
                <Input
                  label="Generation profile ID"
                  value={generationProfileId}
                  onChange={(event) => setGenerationProfileId(event.target.value)}
                />
                <label className="block text-sm font-medium text-text" htmlFor="vn-script-content-rating">
                  Content rating
                </label>
                <select
                  id="vn-script-content-rating"
                  className="block w-full rounded-md border-border bg-bg shadow-sm focus:border-primary focus:ring-primary"
                  value={contentRating}
                  onChange={(event) => setContentRating(event.target.value as VNScriptContentRating)}
                >
                  {contentRatings.map((rating) => (
                    <option key={rating} value={rating}>{rating}</option>
                  ))}
                </select>
                <Button type="submit" size="sm" loading={isCreating} className="w-full">Create script</Button>
              </div>
            </form>
          </aside>

          <section className="flex min-h-0 flex-col gap-3">
            <div className="rounded-md border border-border bg-surface p-3">
              {selectedScript ? (
                <>
                  <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                      <h2 className="text-lg font-semibold">{selectedScript.title}</h2>
                      <p className="text-sm text-text-muted">Script #{selectedScript.id}</p>
                    </div>
                    <Badge variant={statusVariant(selectedScript.status)}>{selectedScript.status}</Badge>
                  </div>
                  <dl className="mt-3 grid gap-2 text-sm sm:grid-cols-2 lg:grid-cols-4">
                    {selectedMeta?.map(([label, value]) => (
                      <div key={label} className="rounded-md border border-border bg-bg p-2">
                        <dt className="text-xs uppercase text-text-muted">{label}</dt>
                        <dd className="mt-1 break-words font-medium">{value}</dd>
                      </div>
                    ))}
                  </dl>
                </>
              ) : (
                <p className="text-sm text-text-muted">Select or create a script to edit its draft.</p>
              )}
            </div>

            <div className="min-h-0 flex-1 rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div>
                  <h2 className="text-sm font-semibold">Draft JSON</h2>
                  <p className="text-xs text-text-muted">
                    Revision {draft?.revision ?? 'n/a'}; JSON parsing only, opcode semantics stay on the API server.
                  </p>
                </div>
                <div className="flex gap-2">
                  <Button type="button" size="sm" variant="secondary" loading={isValidating} disabled={!selectedScript} onClick={handleValidate}>
                    Validate
                  </Button>
                  <Button type="button" size="sm" loading={isSavingDraft} disabled={!selectedScript || !draft} onClick={handleSaveDraft}>
                    Save draft
                  </Button>
                </div>
              </div>
              <JsonEditor value={draftText} onChange={setDraftText} height="52vh" readOnly={!selectedScript} />
            </div>
          </section>

          <aside className="flex min-h-0 flex-col gap-3 overflow-auto">
            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">Validation</h2>
                {validation && (
                  <Badge variant={validation.valid ? 'success' : 'danger'}>
                    {validation.valid ? 'Valid' : 'Invalid'}
                  </Badge>
                )}
              </div>
              {validation ? (
                <div className="space-y-3 text-sm">
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Errors</h3>
                    {validation.errors.length === 0 ? (
                      <p className="text-text-muted">No errors.</p>
                    ) : (
                      <ul className="space-y-1">
                        {validation.errors.map((item, index) => (
                          <li key={index} className="rounded-md bg-danger/10 p-2 text-danger">{formatJson(item)}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                  <div>
                    <h3 className="mb-1 text-xs font-semibold uppercase text-text-muted">Warnings</h3>
                    {validation.warnings.length === 0 ? (
                      <p className="text-text-muted">No warnings.</p>
                    ) : (
                      <ul className="space-y-1">
                        {validation.warnings.map((item, index) => (
                          <li key={index} className="rounded-md bg-warn/10 p-2 text-warn">{formatJson(item)}</li>
                        ))}
                      </ul>
                    )}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-text-muted">Run backend validation to see script readiness.</p>
              )}
            </section>

            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">Diagnostics</h2>
                <Button type="button" size="xs" variant="secondary" loading={isLoadingDiagnostics} disabled={!selectedScript} onClick={handleDiagnostics}>
                  Diagnostics
                </Button>
              </div>
              {diagnostics ? (
                <JsonViewer data={diagnostics.diagnostics} className="max-h-56 rounded-md bg-bg p-2 text-xs" />
              ) : (
                <p className="text-sm text-text-muted">Diagnostics are loaded on demand from the draft endpoint.</p>
              )}
            </section>

            <section className="rounded-md border border-border bg-surface p-3">
              <h2 className="mb-3 text-sm font-semibold">Publish</h2>
              <div className="space-y-3">
                <Input label="Publish label" value={publishLabel} onChange={(event) => setPublishLabel(event.target.value)} />
                <p className="text-xs text-text-muted">
                  Publish acknowledgements are only sent when the backend exposes publish-safe codes.
                </p>
                <Button type="button" size="sm" loading={isPublishing} disabled={!selectedScript || !draft} onClick={handlePublish}>
                  Publish
                </Button>
              </div>
            </section>

            <section className="rounded-md border border-border bg-surface p-3">
              <div className="mb-3 flex items-center justify-between gap-2">
                <h2 className="text-sm font-semibold">Published versions</h2>
                <Badge variant="neutral">{versions.length}</Badge>
              </div>
              <div className="space-y-3">
                {versions.length === 0 && <p className="text-sm text-text-muted">No published versions yet.</p>}
                {versions.map((version) => (
                  <article
                    key={version.id}
                    data-testid={`version-${version.id}`}
                    className="rounded-md border border-border bg-bg p-3"
                  >
                    <div className="flex flex-wrap items-start justify-between gap-2">
                      <div>
                        <h3 className="font-medium">Version {version.version_number}</h3>
                        {version.label && <p className="text-sm text-text-muted">{version.label}</p>}
                      </div>
                      <Badge variant={validationBadge(version.validation) === 'Valid' ? 'success' : 'warning'}>
                        {validationBadge(version.validation)}
                      </Badge>
                    </div>
                    <dl className="mt-3 grid grid-cols-2 gap-2 text-xs">
                      <div><dt className="text-text-muted">Manifest</dt><dd>Manifest {version.manifest_snapshot_id}</dd></div>
                      <div><dt className="text-text-muted">Policy</dt><dd>Policy {version.policy_snapshot_id}</dd></div>
                      <div><dt className="text-text-muted">Generation</dt><dd>Generation {version.generation_profile_snapshot_id}</dd></div>
                      <div><dt className="text-text-muted">Created</dt><dd>{version.created_at}</dd></div>
                    </dl>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <Button
                        type="button"
                        size="xs"
                        variant="secondary"
                        aria-label={`Load manifest for version ${version.version_number}`}
                        onClick={() => void handleManifestSnapshot(version)}
                      >
                        Manifest
                      </Button>
                      <Button
                        type="button"
                        size="xs"
                        variant="secondary"
                        aria-label={`Evaluate policy for version ${version.version_number}`}
                        onClick={() => void handlePolicyEvaluate(version)}
                      >
                        Policy
                      </Button>
                    </div>
                    {manifestSnapshots[version.id] && (
                      <div className="mt-3">
                        <h4 className="mb-1 text-xs font-semibold uppercase text-text-muted">Manifest summary</h4>
                        <JsonViewer data={manifestSnapshots[version.id]} className="max-h-48 rounded-md bg-surface2 p-2 text-xs" />
                      </div>
                    )}
                    {policySummaries[version.id] && (
                      <div className="mt-3">
                        <h4 className="mb-1 text-xs font-semibold uppercase text-text-muted">Policy summary</h4>
                        <JsonViewer data={policySummaries[version.id]} className="max-h-48 rounded-md bg-surface2 p-2 text-xs" />
                      </div>
                    )}
                  </article>
                ))}
              </div>
            </section>
          </aside>
        </div>
      </div>
    </main>
  );
}
