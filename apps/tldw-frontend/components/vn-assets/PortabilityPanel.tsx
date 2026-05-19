import React, { useMemo, useState } from 'react';
import type { ChangeEvent } from 'react';
import { Archive, Upload } from 'lucide-react';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import {
  commitVNPackImport,
  createVNPackImportPreview,
  exportVNAssetPack,
  getVNPackImportPreview,
} from '@web/lib/api/vnAssets';
import type {
  VNAssetPack,
  VNPackExportResponse,
  VNPackImportCommitStartResponse,
  VNPackImportPreview,
} from '@web/types/vn-assets';

export interface PortabilityPanelProps {
  selectedPack?: VNAssetPack | null;
}

type TrustMode = 'trusted_restore' | 'untrusted_import';

interface UpdateCandidate {
  target_pack_id?: number;
  diffs?: Array<Record<string, unknown>>;
  blocked?: boolean;
  requires_confirmation?: boolean;
}

function asRecord(value: unknown): Record<string, unknown> {
  return value && typeof value === 'object' && !Array.isArray(value) ? (value as Record<string, unknown>) : {};
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : [];
}

function previewTitle(preview: VNPackImportPreview | null): string {
  const summary = preview?.bundle_summary ?? {};
  const title = summary.pack_title ?? summary.title;
  return typeof title === 'string' && title.trim() ? title : 'Imported pack';
}

function updateCandidateForPack(preview: VNPackImportPreview | null, packId?: number): UpdateCandidate | null {
  if (!preview || !packId) return null;
  const proposedPlan = asRecord(preview.proposed_plan);
  const updateExisting = asRecord(proposedPlan.update_existing);
  return (asArray(updateExisting.candidate_packs).find((candidate) => {
    const record = asRecord(candidate);
    return Number(record.target_pack_id) === packId;
  }) as UpdateCandidate | undefined) ?? null;
}

function riskyDiffs(candidate: UpdateCandidate | null): Array<Record<string, unknown>> {
  if (!candidate) return [];
  return asArray(candidate.diffs).map(asRecord).filter((diff) => Boolean(diff.requires_confirmation));
}

function statusVariant(status?: string): 'danger' | 'info' | 'neutral' | 'success' | 'warning' {
  if (status === 'completed') return 'success';
  if (status === 'failed') return 'danger';
  if (status === 'queued' || status === 'processing') return 'info';
  if (status === 'cancelled') return 'warning';
  return 'neutral';
}

const IMPORT_PREVIEW_TERMINAL_STATUSES = new Set(['completed', 'failed', 'cancelled', 'quarantined', 'deleted']);
const IMPORT_PREVIEW_POLL_INTERVAL_MS = 1000;
const IMPORT_PREVIEW_MAX_POLLS = 60;

function createVNAssetIdempotencyKey(prefix: string): string {
  const uuid = globalThis.crypto?.randomUUID?.();
  return `${prefix}-${uuid ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`}`;
}

function waitForImportPreviewPoll(): Promise<void> {
  return new Promise((resolve) => {
    globalThis.setTimeout(resolve, IMPORT_PREVIEW_POLL_INTERVAL_MS);
  });
}

async function pollVNPackImportPreview(
  previewId: number,
  onPreview: (preview: VNPackImportPreview) => void
): Promise<VNPackImportPreview> {
  let preview = await getVNPackImportPreview(previewId);
  onPreview(preview);

  for (
    let attempts = 1;
    !IMPORT_PREVIEW_TERMINAL_STATUSES.has(preview.status) && attempts < IMPORT_PREVIEW_MAX_POLLS;
    attempts += 1
  ) {
    await waitForImportPreviewPoll();
    preview = await getVNPackImportPreview(previewId);
    onPreview(preview);
  }

  return preview;
}

export default function PortabilityPanel({ selectedPack }: PortabilityPanelProps) {
  const [includeCharacterPayload, setIncludeCharacterPayload] = useState(false);
  const [includeWorldBookPayloads, setIncludeWorldBookPayloads] = useState(false);
  const [includeFullProvenance, setIncludeFullProvenance] = useState(false);
  const [strictExport, setStrictExport] = useState(false);
  const [exportJob, setExportJob] = useState<VNPackExportResponse | null>(null);
  const [importPreview, setImportPreview] = useState<VNPackImportPreview | null>(null);
  const [importJob, setImportJob] = useState<VNPackImportCommitStartResponse | null>(null);
  const [trustMode, setTrustMode] = useState<TrustMode>('trusted_restore');
  const [confirmRiskyDiffs, setConfirmRiskyDiffs] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isCommitting, setIsCommitting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const updateCandidate = useMemo(
    () => updateCandidateForPack(importPreview, selectedPack?.id),
    [importPreview, selectedPack?.id]
  );
  const riskyUpdateDiffs = useMemo(() => riskyDiffs(updateCandidate), [updateCandidate]);
  const requiredChoices = useMemo(() => asArray(importPreview?.required_choices), [importPreview]);
  const targetMode = updateCandidate && !updateCandidate.blocked ? 'update_existing' : 'create_new';

  const handleExport = async () => {
    if (!selectedPack) return;
    setIsExporting(true);
    setError(null);
    try {
      const response = await exportVNAssetPack(selectedPack.id, {
        include_character_payload: includeCharacterPayload,
        include_world_book_payloads: includeWorldBookPayloads,
        include_full_provenance: includeFullProvenance,
        strict: strictExport,
        warn_for_sharing: true,
        idempotency_key: createVNAssetIdempotencyKey('vn-export'),
      });
      setExportJob(response);
    } catch (exportError) {
      setError(exportError instanceof Error ? exportError.message : 'Failed to start export');
    } finally {
      setIsExporting(false);
    }
  };

  const handleArchiveUpload = async (event: ChangeEvent<HTMLInputElement>) => {
    const archive = event.target.files?.[0];
    if (!archive) return;
    setIsUploading(true);
    setError(null);
    setImportPreview(null);
    setImportJob(null);
    setConfirmRiskyDiffs(false);
    try {
      const previewStart = await createVNPackImportPreview(
        archive,
        createVNAssetIdempotencyKey('vn-import-preview')
      );
      await pollVNPackImportPreview(previewStart.preview_id, setImportPreview);
    } catch (uploadError) {
      setError(uploadError instanceof Error ? uploadError.message : 'Failed to create import preview');
    } finally {
      setIsUploading(false);
      event.target.value = '';
    }
  };

  const handleCommitImport = async () => {
    if (!selectedPack || !importPreview) return;
    if (riskyUpdateDiffs.length > 0 && !confirmRiskyDiffs) {
      setError('Confirm risky update diffs before committing this update.');
      return;
    }
    setIsCommitting(true);
    setError(null);
    try {
      const response = await commitVNPackImport({
        preview_id: importPreview.preview_id,
        trust_mode: trustMode,
        target_mode: targetMode,
        character_action: 'link_existing_character',
        target_character_id: selectedPack.primary_character_id,
        target_pack_id: targetMode === 'update_existing' ? selectedPack.id : null,
        conflict_decisions: {
          confirm_all_risky_diffs: riskyUpdateDiffs.length > 0 && confirmRiskyDiffs,
        },
        idempotency_key: createVNAssetIdempotencyKey('vn-import-commit'),
      });
      setImportJob(response);
    } catch (commitError) {
      setError(commitError instanceof Error ? commitError.message : 'Failed to commit import');
    } finally {
      setIsCommitting(false);
    }
  };

  return (
    <section className="rounded-md border border-border bg-surface p-4">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-2">
        <h2 className="text-lg font-semibold">Portability</h2>
        <Badge variant="neutral">Backup bundle</Badge>
      </div>

      <div className="grid gap-4 xl:grid-cols-2">
        <div className="rounded-md border border-border bg-bg p-3">
          <div className="mb-3 flex items-center gap-2">
            <Archive aria-hidden className="h-4 w-4 text-primary" />
            <h3 className="font-medium">Export</h3>
          </div>
          <p className="mb-3 text-sm text-warn">VN pack exports are not encrypted.</p>
          <div className="grid gap-2 text-sm">
            <label className="flex items-center gap-2">
              <input
                checked={includeCharacterPayload}
                className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                type="checkbox"
                onChange={(event) => setIncludeCharacterPayload(event.target.checked)}
              />
              Include character payload
            </label>
            <label className="flex items-center gap-2">
              <input
                checked={includeWorldBookPayloads}
                className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                type="checkbox"
                onChange={(event) => setIncludeWorldBookPayloads(event.target.checked)}
              />
              Include world book payloads
            </label>
            <label className="flex items-center gap-2">
              <input
                checked={includeFullProvenance}
                className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                type="checkbox"
                onChange={(event) => setIncludeFullProvenance(event.target.checked)}
              />
              Include full provenance
            </label>
            <label className="flex items-center gap-2">
              <input
                checked={strictExport}
                className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                type="checkbox"
                onChange={(event) => setStrictExport(event.target.checked)}
              />
              Strict missing-file check
            </label>
          </div>
          <Button
            className="mt-4 gap-2"
            disabled={!selectedPack}
            loading={isExporting}
            onClick={handleExport}
            size="sm"
            type="button"
          >
            <Archive aria-hidden className="h-4 w-4" />
            Export backup bundle
          </Button>
          {exportJob && (
            <div className="mt-3 grid gap-1 text-sm">
              <p>Export job: {exportJob.job_id}</p>
              <p>Export status: {exportJob.status}</p>
              {exportJob.download_url && (
                <a className="text-primary underline" href={exportJob.download_url}>
                  Download export
                </a>
              )}
            </div>
          )}
        </div>

        <div className="rounded-md border border-border bg-bg p-3">
          <div className="mb-3 flex items-center gap-2">
            <Upload aria-hidden className="h-4 w-4 text-primary" />
            <h3 className="font-medium">Import</h3>
          </div>
          <label className="grid gap-1 text-sm font-medium" htmlFor="vn-pack-import-archive">
            Import VN pack archive
            <input
              id="vn-pack-import-archive"
              accept=".tldw-vnpack,.zip,application/zip"
              className="rounded-md border border-border bg-surface px-3 py-2 text-sm"
              disabled={isUploading}
              type="file"
              onChange={handleArchiveUpload}
            />
          </label>
          {isUploading && <p className="mt-3 text-sm text-text-muted">Creating import preview...</p>}
          {importPreview && (
            <div className="mt-3 grid gap-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span>Preview status: {importPreview.status}</span>
                <Badge variant={statusVariant(importPreview.status)}>{importPreview.stage}</Badge>
              </div>
              <p className="font-medium">{previewTitle(importPreview)}</p>
              {requiredChoices.length > 0 && (
                <div className="rounded-md border border-border bg-surface p-3">
                  <h4 className="font-medium">Character resolution</h4>
                  <div className="mt-2 grid gap-1">
                    {requiredChoices.map((choice, index) => {
                      const record = asRecord(choice);
                      const message = typeof record.message === 'string' ? record.message : 'Resolve imported character.';
                      return <p key={`${message}-${index}`}>{message}</p>;
                    })}
                  </div>
                </div>
              )}
              {riskyUpdateDiffs.length > 0 && (
                <div className="rounded-md border border-warn/40 bg-warn/10 p-3">
                  <h4 className="font-medium text-warn">Risky update diffs</h4>
                  <ul className="mt-2 grid gap-1">
                    {riskyUpdateDiffs.map((diff) => (
                      <li key={String(diff.diff_id)}>{String(diff.kind ?? 'update_diff')}</li>
                    ))}
                  </ul>
                  <label className="mt-3 flex items-center gap-2">
                    <input
                      checked={confirmRiskyDiffs}
                      className="h-4 w-4 rounded border-border text-primary focus:ring-primary"
                      type="checkbox"
                      onChange={(event) => setConfirmRiskyDiffs(event.target.checked)}
                    />
                    Confirm risky update diffs
                  </label>
                </div>
              )}
              <label className="grid gap-1 font-medium" htmlFor="vn-pack-trust-mode">
                Trust mode
                <select
                  id="vn-pack-trust-mode"
                  className="rounded-md border border-border bg-surface px-3 py-2 text-sm"
                  value={trustMode}
                  onChange={(event) => setTrustMode(event.target.value as TrustMode)}
                >
                  <option value="trusted_restore">Trusted restore</option>
                  <option value="untrusted_import">Untrusted import</option>
                </select>
              </label>
              <Button
                disabled={importPreview.status !== 'completed' || !selectedPack}
                loading={isCommitting}
                onClick={handleCommitImport}
                size="sm"
                type="button"
              >
                Commit import
              </Button>
            </div>
          )}
          {importJob && (
            <p className="mt-3 text-sm">Import job: {importJob.job_id}</p>
          )}
        </div>
      </div>

      {error && (
        <div className="mt-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          {error}
        </div>
      )}
    </section>
  );
}
