import React, { useMemo, useRef, useState } from 'react';
import type { ChangeEvent } from 'react';
import { Archive, Upload } from 'lucide-react';
import { ApiError } from '@/services/tldw/api-error';
import { downloadBlob } from '@/utils/download-blob';
import { createVNAssetIdempotencyKey } from '@web/lib/vnAssetIdempotency';
import { Badge } from '@web/components/ui/Badge';
import { Button } from '@web/components/ui/Button';
import {
  commitVNPackImport,
  createVNPackImportPreview,
  deleteVNPackImportPreview,
  downloadVNPackExport,
  exportVNAssetPack,
  getVNPackExportJob,
  getVNPackImportJob,
  getVNPackImportPreview,
} from '@web/lib/api/vnAssets';
import {
  isPortabilityTerminal, useVNPortabilityReceipts, useVNPortabilityStatus, vnPortabilityRequestConfig,
  type VNPortabilityAuthority,
} from '@web/hooks/useVNPortability';
import type { ApiRequestConfig } from '@web/types/common';
import type { VNAssetPack, VNPackImportJob, VNPackImportPreview } from '@web/types/vn-assets';

export interface PortabilityPanelProps {
  selectedPack?: VNAssetPack | null;
  recovery: VNPortabilityAuthority;
  onOpenImportedPack?: (packId: number, config: ApiRequestConfig) => Promise<void>;
}

type TrustMode = 'trusted_restore' | 'untrusted_import';
type ReceiptUpdate = Parameters<ReturnType<typeof useVNPortabilityReceipts>['save']>[1];
type Action = 'export' | 'preview' | 'import' | 'download' | 'remove' | 'open';

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
  if (status === 'failed' || status === 'quarantined') return 'danger';
  if (status === 'queued' || status === 'processing' || status === 'running') return 'info';
  if (status === 'cancelled') return 'warning';
  return 'neutral';
}

function isExpired(value?: string | null): boolean {
  return !!value && Date.parse(value) <= Date.now();
}

function rejectedBeforeAdmission(error: unknown): boolean {
  return error instanceof ApiError && ([400, 401, 403, 404, 412, 413, 422].includes(error.status ?? 0) ||
    (error.status === 409 && error.detail === 'import_preview_not_completed'));
}

function quotaFailure(job?: VNPackImportJob): boolean {
  return !!job && (job.status === 'failed' || job.status === 'quarantined') &&
    (job.error_message?.startsWith('Storage quota exceeded:') === true ||
      job.error_code === 'storage_quota_exceeded');
}

function StatusCheck({ label, status }: {
  label: string;
  status: { paused?: boolean; unavailable?: boolean; check: () => void };
}) {
  if (!status.paused) return null;
  return (
    <div className="flex flex-wrap items-center gap-2">
      <span className="text-text-muted">
        {status.unavailable ? 'This job is unavailable.' : 'Status updates are paused.'}
      </span>
      <Button size="sm" variant="secondary" type="button" onClick={status.check}>Check {label} status</Button>
    </div>
  );
}

export default function PortabilityPanel({ selectedPack, recovery, onOpenImportedPack }: PortabilityPanelProps) {
  const [includeCharacterPayload, setIncludeCharacterPayload] = useState(false);
  const [includeWorldBookPayloads, setIncludeWorldBookPayloads] = useState(false);
  const [includeFullProvenance, setIncludeFullProvenance] = useState(false);
  const [strictExport, setStrictExport] = useState(false);
  const { receipts, storageUnavailable, save } = useVNPortabilityReceipts(recovery.authority);
  const visibleReceipts = recovery.ready ? receipts : null;
  const exportReceipt = visibleReceipts?.export;
  const previewReceipt = visibleReceipts?.preview;
  const importReceipt = visibleReceipts?.import;
  const exportStatus = useVNPortabilityStatus(recovery, exportReceipt?.jobId, getVNPackExportJob);
  const previewStatus = useVNPortabilityStatus(recovery, previewReceipt?.previewId, getVNPackImportPreview);
  const importStatus = useVNPortabilityStatus(recovery, importReceipt?.jobId, getVNPackImportJob);
  const exportJob = exportStatus.job;
  const importPreview = previewStatus.job;
  const importJob = importStatus.job;
  const [trustMode, setTrustMode] = useState<TrustMode>('trusted_restore');
  const [confirmRiskyDiffs, setConfirmRiskyDiffs] = useState(false);
  const [busy, setBusy] = useState<Action | null>(null);
  const inFlight = useRef(false);
  const [error, setError] = useState<string | null>(null);

  const updateCandidate = useMemo(
    () => updateCandidateForPack(importPreview, selectedPack?.id),
    [importPreview, selectedPack?.id]
  );
  const riskyUpdateDiffs = useMemo(() => riskyDiffs(updateCandidate), [updateCandidate]);
  const requiredChoices = useMemo(() => asArray(importPreview?.required_choices), [importPreview]);
  const hasCharacterNameConflict = asArray(importPreview?.conflicts).some(
    (conflict) => asRecord(conflict).kind === 'character_candidate'
  );
  const canImportIncludedCharacter = !hasCharacterNameConflict && requiredChoices.some((choice) =>
    asArray(asRecord(choice).allowed_actions).includes('import_included_character')
  );
  const canAct = recovery.ready && !!receipts && !busy;
  const archiveState = importPreview?.vn_status;
  const previewState = archiveState === 'deleting' || archiveState === 'deleted' || archiveState === 'expired'
    ? archiveState : importPreview?.status;
  const previewReady = previewState === 'completed' && importPreview?.vn_status === 'completed' &&
    !isExpired(importPreview.expires_at);
  const restoredPackId = importJob?.vn_status === 'completed' ? importJob.pack_id : null;
  const exportUnconfirmed = !!exportReceipt && !exportReceipt.jobId;
  const previewUnconfirmed = !!previewReceipt && !previewReceipt.previewId;
  const importUnconfirmed = !!importReceipt && !importReceipt.jobId;
  const importActive = !!importReceipt && !importStatus.unavailable &&
    (!importJob || !isPortabilityTerminal(importJob.status));
  const stagedArchive = !!previewReceipt?.previewId && previewState !== 'deleted' && !previewStatus.unavailable;
  const canChooseArchive = !importActive && !stagedArchive;
  const canRemoveArchive = !importActive && stagedArchive && !!previewState && isPortabilityTerminal(previewState);
  const canExport = canAct && (exportUnconfirmed || (!!selectedPack &&
    (!exportReceipt || (exportJob && isPortabilityTerminal(exportJob.status)) || exportStatus.unavailable)));
  const canCommit = canAct && previewReady && !importReceipt && (!!selectedPack || canImportIncludedCharacter);
  const canRetryImport = canAct && previewReady && quotaFailure(importJob) && !importJob?.pack_id &&
    importReceipt?.request.target_mode === 'create_new' && importJob?.created_records?.writes_started === false;
  const targetMode = updateCandidate && !updateCandidate.blocked ? 'update_existing' : 'create_new';

  async function runAction(
    action: Action,
    failureMessage: string,
    run: (config: ApiRequestConfig, persist: (patch: ReceiptUpdate) => void, isCurrent: () => boolean) => Promise<void>
  ) {
    if (!canAct || inFlight.current) return;
    const capture = recovery.captureCurrent();
    if (!capture) return;
    const isCurrent = () => recovery.isCurrent(capture);
    inFlight.current = true;
    setBusy(action);
    setError(null);
    try {
      await run(vnPortabilityRequestConfig(capture.scope), (patch) => {
        if (!isCurrent()) throw new Error('Portability account changed.');
        save(capture.scope, patch);
      }, isCurrent);
    } catch {
      if (isCurrent()) setError(failureMessage);
    } finally {
      inFlight.current = false;
      setBusy(null);
    }
  }

  async function startTrackedRequest(
    action: 'export' | 'preview' | 'import',
    pending: ReceiptUpdate,
    replay: boolean,
    request: (config: ApiRequestConfig) => Promise<ReceiptUpdate>
  ) {
    await runAction(action, `Unable to confirm the ${action} request. Check the saved request before trying again.`, async (config, persist) => {
      persist(pending);
      try {
        persist(await request(config));
      } catch (error) {
        if (!replay && rejectedBeforeAdmission(error)) persist({ [action]: visibleReceipts?.[action] });
        if (action === 'import' && error instanceof ApiError && error.detail === 'import_preview_not_completed') {
          previewStatus.check();
        }
        throw error;
      }
    });
  }

  const handleExport = () => {
    if (!canExport) return;
    const entry = exportUnconfirmed ? exportReceipt : selectedPack && {
      packId: selectedPack.id,
      request: {
        include_character_payload: includeCharacterPayload,
        include_world_book_payloads: includeWorldBookPayloads,
        include_full_provenance: includeFullProvenance,
        strict: strictExport,
        warn_for_sharing: true as const,
        idempotency_key: createVNAssetIdempotencyKey('vn-export'),
      },
    };
    if (!entry) return;
    void startTrackedRequest('export', { export: entry }, exportUnconfirmed, async (config) => {
      const response = await exportVNAssetPack(entry.packId, entry.request, config);
      return { export: { ...entry, jobId: response.job_id } };
    });
  };

  const handleArchiveUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const archive = event.target.files?.[0];
    event.target.value = '';
    if (!archive || !canChooseArchive) return;
    const idempotencyKey = previewUnconfirmed
      ? previewReceipt.idempotencyKey
      : createVNAssetIdempotencyKey('vn-import-preview');
    setConfirmRiskyDiffs(false);
    void startTrackedRequest('preview', { preview: { idempotencyKey } }, previewUnconfirmed, async (config) => {
      const response = await createVNPackImportPreview(archive, idempotencyKey, config);
      return { preview: { idempotencyKey, previewId: response.preview_id }, import: undefined };
    });
  };

  const submitImport = (request: NonNullable<ReceiptUpdate['import']>['request'], replay = false) => {
    void startTrackedRequest('import', { import: { request } }, replay, async (config) => {
      const response = await commitVNPackImport(request, config);
      return { import: { request, jobId: response.job_id } };
    });
  };

  const handleCommitImport = () => {
    if (!importPreview || !canCommit) return;
    if (riskyUpdateDiffs.length > 0 && !confirmRiskyDiffs) {
      setError('Confirm risky update diffs before committing this update.');
      return;
    }
    submitImport({
      preview_id: importPreview.preview_id,
      trust_mode: trustMode,
      target_mode: targetMode,
      character_action: selectedPack ? 'link_existing_character' : 'import_included_character',
      target_character_id: selectedPack?.primary_character_id ?? null,
      target_pack_id: targetMode === 'update_existing' ? selectedPack?.id ?? null : null,
      conflict_decisions: {
        confirm_all_risky_diffs: riskyUpdateDiffs.length > 0 && confirmRiskyDiffs,
      },
      idempotency_key: createVNAssetIdempotencyKey('vn-import-commit'),
    });
  };

  const canDownload = canAct && exportJob?.status === 'completed' && !isExpired(exportJob.expires_at);
  const handleDownload = () => {
    if (!canDownload || !exportReceipt?.jobId) return;
    const { jobId, packId } = exportReceipt;
    void runAction('download', 'Unable to download the backup. Check its status and try again.', async (config, _, isCurrent) => {
      try {
        const blob = await downloadVNPackExport(jobId, config);
        if (isCurrent()) downloadBlob(blob, `vn-pack-${packId}.tldw-vnpack`);
      } catch (error) {
        exportStatus.check();
        throw error;
      }
    });
  };

  const handleRemoveArchive = () => {
    if (!canRemoveArchive || !previewReceipt?.previewId) return;
    const { previewId } = previewReceipt;
    void runAction('remove', 'Unable to confirm staged archive removal. Check the preview status and try again.', async (config, persist) => {
      try {
        await deleteVNPackImportPreview(previewId, config);
        persist({ preview: undefined });
      } catch (error) {
        previewStatus.check();
        throw error;
      }
    });
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
            disabled={!canExport}
            loading={busy === 'export'}
            onClick={handleExport}
            size="sm"
            type="button"
          >
            <Archive aria-hidden className="h-4 w-4" />
            {exportUnconfirmed ? 'Recover export request' : 'Export backup bundle'}
          </Button>
          {exportReceipt?.jobId && (
            <div className="mt-3 grid gap-1 text-sm">
              <p>Export job: {exportReceipt.jobId}</p>
              <p role="status">Export status: {exportJob?.status ?? 'checking'}</p>
              <StatusCheck label="export" status={exportStatus} />
              {exportJob?.status === 'completed' && (
                isExpired(exportJob.expires_at) ? (
                  <p className="text-text-muted">This backup expired. Start a new export.</p>
                ) : (
                  <Button disabled={!canDownload} loading={busy === 'download'} size="sm" type="button" onClick={handleDownload}>
                    Download export
                  </Button>
                )
              )}
              {(exportJob?.status === 'failed' || exportJob?.status === 'quarantined') && (
                <p role="alert" className="text-danger">
                  {exportJob.error_message?.startsWith('missing_asset_bytes:')
                    ? 'An image is missing. Restore the image or turn off the strict missing-file check and export again.'
                    : 'The backup could not be created. Check the export job before trying again.'}
                </p>
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
              className="min-w-0 w-full rounded-md border border-border bg-surface px-3 py-2 text-sm"
              disabled={!canAct || !canChooseArchive}
              type="file"
              onChange={handleArchiveUpload}
            />
          </label>
          {busy === 'preview' && <p role="status" className="mt-3 text-sm text-text-muted">Uploading archive…</p>}
          {previewUnconfirmed && (
            <p className="mt-3 text-sm text-text-muted">Choose the same archive to recover the upload.</p>
          )}
          {previewReceipt?.previewId && !importPreview && !previewStatus.paused && (
            <p role="status" className="mt-3 text-sm text-text-muted">Checking the staged archive…</p>
          )}
          <StatusCheck label="preview" status={previewStatus} />
          {previewState === 'deleted' && <p role="status" className="mt-3 text-sm">Preview status: deleted</p>}
          {importPreview && previewState !== 'deleted' && (
            <div className="mt-3 grid gap-3 text-sm">
              <div className="flex flex-wrap items-center gap-2">
                <span role="status">Preview status: {previewState}</span>
                {previewState !== 'deleting' && importPreview.stage !== previewState && (
                  <Badge variant={statusVariant(previewState)}>{importPreview.stage}</Badge>
                )}
              </div>
              <p className="font-medium">{previewTitle(importPreview)}</p>
              {isExpired(importPreview.expires_at) && (
                <p className="text-text-muted">This preview expired. Remove the staged archive before uploading it again.</p>
              )}
              {(importPreview.status === 'failed' || importPreview.status === 'quarantined') && importPreview.error_message && (
                <p className="text-danger" role="alert">
                  {importPreview.error_message.startsWith('invalid_archive:')
                    ? 'The archive could not be read. Download it again or choose another VN pack backup.'
                    : 'Archive validation failed. Check the backup and try again.'}
                </p>
              )}
              {asArray(importPreview.validation_warnings).some((warning) =>
                typeof warning === 'string' && warning.startsWith('missing_asset_bytes:')
              ) && (
                <p className="text-warn">Assets with missing image files will import as hidden items.</p>
              )}
              {previewReady && !selectedPack && (
                <p className="text-text-muted">
                  {hasCharacterNameConflict && 'A character with this name already exists. '}
                  {canImportIncludedCharacter
                    ? 'Import creates a pack with the included character.'
                    : 'Select a pack to link its primary character before importing.'}
                </p>
              )}
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
                  disabled={!!importReceipt}
                  onChange={(event) => setTrustMode(event.target.value as TrustMode)}
                >
                  <option value="trusted_restore">Trusted restore</option>
                  <option value="untrusted_import">Untrusted import</option>
                </select>
              </label>
              <Button
                disabled={!canCommit}
                loading={busy === 'import'}
                onClick={handleCommitImport}
                size="sm"
                type="button"
              >
                Commit import
              </Button>
            </div>
          )}
          {importUnconfirmed && importReceipt && (
            <Button className="mt-3" size="sm" variant="secondary" type="button" disabled={!canAct}
              onClick={() => submitImport(importReceipt.request, true)}>
              Recover import request
            </Button>
          )}
          {importReceipt?.jobId && (
            <div className="mt-3 grid gap-2 text-sm">
              <p>Import job: {importReceipt.jobId}</p>
              <p role="status">Import status: {restoredPackId ? 'completed' : importJob?.status ?? 'checking'}</p>
              <StatusCheck label="import" status={importStatus} />
              {!restoredPackId && quotaFailure(importJob) && (
                <p role="alert" className="text-danger">Storage quota exceeded. Free up space or ask an administrator to raise your quota.</p>
              )}
              {!restoredPackId && importJob && (importJob.status === 'failed' || importJob.status === 'quarantined') && !quotaFailure(importJob) && (
                <p role="alert" className="text-danger">Import failed. Review the pack and job details before trying another restore.</p>
              )}
              {canRetryImport && (
                <Button size="sm" variant="secondary" type="button" onClick={() => submitImport({
                  ...importReceipt.request,
                  idempotency_key: `vn-import-retry-${importReceipt.jobId}`,
                })}>
                  Retry import
                </Button>
              )}
              {restoredPackId && onOpenImportedPack && (
                <Button size="sm" type="button" disabled={!canAct} onClick={() => {
                  void runAction('open', 'Unable to open the imported pack. Refresh the pack list and try again.', async (config) => {
                    await onOpenImportedPack(restoredPackId, config);
                  });
                }}>
                  Open imported pack
                </Button>
              )}
            </div>
          )}
          {canRemoveArchive && (
            <Button className="mt-3" size="sm" variant="secondary" type="button"
              disabled={!canAct} loading={busy === 'remove'} onClick={handleRemoveArchive}>
              Remove staged archive
            </Button>
          )}
        </div>
      </div>

      {(error || storageUnavailable) && (
        <div role="alert" className="mt-4 rounded-md border border-danger/30 bg-danger/10 px-3 py-2 text-sm text-danger">
          {storageUnavailable ? 'Saved portability requests could not be read. Check session storage and reload the workbench.' : error}
        </div>
      )}
    </section>
  );
}
