import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VNAssetPack } from '@web/types/vn-assets';

const mocks = vi.hoisted(() => ({
  commitVNPackImport: vi.fn(),
  createVNPackImportPreview: vi.fn(),
  exportVNAssetPack: vi.fn(),
  getVNPackExportJob: vi.fn(),
  getVNPackImportJob: vi.fn(),
  getVNPackImportPreview: vi.fn(),
  downloadVNPackExport: vi.fn(),
  deleteVNPackImportPreview: vi.fn(),
  downloadBlob: vi.fn(),
}));

vi.mock('@web/lib/api/vnAssets', () => ({
  commitVNPackImport: (...args: unknown[]) => mocks.commitVNPackImport(...args),
  createVNPackImportPreview: (...args: unknown[]) => mocks.createVNPackImportPreview(...args),
  exportVNAssetPack: (...args: unknown[]) => mocks.exportVNAssetPack(...args),
  getVNPackExportJob: (...args: unknown[]) => mocks.getVNPackExportJob(...args),
  getVNPackImportJob: (...args: unknown[]) => mocks.getVNPackImportJob(...args),
  getVNPackImportPreview: (...args: unknown[]) => mocks.getVNPackImportPreview(...args),
  downloadVNPackExport: (...args: unknown[]) => mocks.downloadVNPackExport(...args),
  deleteVNPackImportPreview: (...args: unknown[]) => mocks.deleteVNPackImportPreview(...args),
}));

vi.mock('@/utils/download-blob', () => ({ downloadBlob: mocks.downloadBlob }));

import PortabilityPanel from '@web/components/vn-assets/PortabilityPanel';

const authority = { server: 'http://localhost:8000/api/v1', principal: '1' };
const expectedAccount = { baseURL: authority.server, headers: { 'X-TLDW-Expected-User-ID': '1' } };
const recovery = {
  authority,
  ready: true,
  captureCurrent: () => ({ scope: authority, epoch: 1 }),
  isCurrent: () => true,
};

const pack: VNAssetPack = {
  id: 9,
  title: 'Orbital Library',
  owner_user_id: 1,
  primary_character_id: 42,
  planned_output_count: 24,
};

describe('PortabilityPanel', () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.resetAllMocks();
    sessionStorage.clear();
    mocks.getVNPackExportJob.mockResolvedValue({
      job_id: 'export-job', status: 'queued', vn_status: 'queued', stage: 'queued',
    });
    mocks.getVNPackImportJob.mockResolvedValue({
      job_id: 'commit-job', preview_id: 33, status: 'queued', vn_status: 'queued', stage: 'queued',
    });
    mocks.exportVNAssetPack.mockResolvedValue({
      job_id: 'export-job',
      portability_job_id: 1,
      operation: 'export',
      pack_id: 9,
      status: 'queued',
      stage: 'queued',
      download_url: null,
    });
    mocks.createVNPackImportPreview.mockResolvedValue({
      job_id: 'preview-job',
      portability_job_id: 2,
      operation: 'import_preview',
      preview_id: 33,
      status: 'queued',
      stage: 'queued',
    });
    mocks.getVNPackImportPreview.mockResolvedValue({
      preview_id: 33,
      job_id: 'preview-job',
      portability_job_id: 2,
      operation: 'import_preview',
      status: 'completed',
      vn_status: 'completed',
      stage: 'completed',
      bundle_summary: { pack_title: 'Imported Backup', item_count: 2 },
      required_choices: [
        {
          kind: 'primary_character',
          message: 'Resolve imported primary character.',
          allowed_actions: ['link_existing_character'],
        },
      ],
      proposed_plan: {
        update_existing: {
          allowed: true,
          candidate_packs: [
            {
              target_pack_id: 9,
              matched_slots: [{ source_slot_id: 300, local_slot_id: 4, identity: 'sprite:sprite.primary.neutral' }],
              added_slots: [],
              matched_items: [],
              added_items: [{ source_item_id: 401, source_slot_id: 300 }],
              diffs: [
                {
                  diff_id: 'diff-risky',
                  kind: 'slot_metadata_diff',
                  severity: 'review',
                  requires_confirmation: true,
                  identity: 'sprite:sprite.primary.neutral',
                  fields: ['labels'],
                },
              ],
              requires_confirmation: true,
              blocked: false,
            },
          ],
        },
      },
      conflicts: [],
      validation_warnings: [],
      quota_estimate: { asset_bytes: 1024 },
    });
    mocks.commitVNPackImport.mockResolvedValue({
      job_id: 'commit-job',
      portability_job_id: 3,
      operation: 'import_commit',
      preview_id: 33,
      import_id: 44,
      status: 'queued',
      stage: 'queued',
    });
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('recovers a completed export after reload and downloads through the authenticated client', async () => {
    const blob = new Blob(['backup'], { type: 'application/zip' });
    mocks.getVNPackExportJob.mockResolvedValue({
      job_id: 'export-job', status: 'completed', vn_status: 'completed', stage: 'completed', pack_id: 9,
    });
    mocks.downloadVNPackExport.mockResolvedValue(blob);
    const user = userEvent.setup();
    const view = render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    await user.click(screen.getByRole('button', { name: 'Export backup bundle' }));
    await screen.findByText('Export status: completed');
    view.unmount();
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    await user.click(await screen.findByRole('button', { name: 'Download export' }));
    expect(mocks.getVNPackExportJob).toHaveBeenCalledTimes(2);
    expect(mocks.exportVNAssetPack).toHaveBeenCalledTimes(1);
    expect(mocks.getVNPackExportJob).toHaveBeenLastCalledWith('export-job', { ...expectedAccount, signal: expect.any(AbortSignal) });
    expect(mocks.downloadVNPackExport).toHaveBeenCalledWith('export-job', expectedAccount);
    expect(mocks.downloadBlob).toHaveBeenCalledWith(blob, 'vn-pack-9.tldw-vnpack');
  });

  it('replays an unconfirmed import after reload with the identical payload and key', async () => {
    mocks.getVNPackImportPreview.mockResolvedValue({
      preview_id: 33,
      status: 'completed',
      vn_status: 'completed',
      stage: 'completed',
      required_choices: [{ allowed_actions: ['import_included_character'] }],
    });
    mocks.commitVNPackImport.mockRejectedValueOnce(new Error('Lost response'));
    mocks.getVNPackImportJob.mockResolvedValue({
      job_id: 'commit-job', preview_id: 33, status: 'completed', vn_status: 'completed', stage: 'completed', pack_id: 19,
    });
    const user = userEvent.setup();
    const view = render(<PortabilityPanel recovery={recovery} />);
    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['backup'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );
    await user.selectOptions(await screen.findByLabelText('Trust mode'), 'untrusted_import');
    await user.click(screen.getByRole('button', { name: 'Commit import' }));
    await screen.findByRole('button', { name: 'Recover import request' });

    view.unmount();
    render(<PortabilityPanel recovery={recovery} />);
    await user.click(await screen.findByRole('button', { name: 'Recover import request' }));
    expect(mocks.commitVNPackImport).toHaveBeenCalledTimes(2);
    expect(mocks.commitVNPackImport.mock.calls[1]).toEqual(mocks.commitVNPackImport.mock.calls[0]);
    expect(await screen.findByText('Import status: completed')).toBeInTheDocument();
  });

  it.each([false, true, undefined])('reports a storage limit and checks whether any attempt wrote data (%s)', async (writesStarted) => {
    mocks.getVNPackImportPreview.mockResolvedValue({
      preview_id: 33, status: 'completed', vn_status: 'completed', stage: 'completed',
      required_choices: [{ allowed_actions: ['import_included_character'] }],
    });
    mocks.getVNPackImportJob.mockResolvedValue({
      job_id: 'commit-job', preview_id: 33, status: 'failed', vn_status: 'failed', stage: 'failed',
      error_code: 'import_failed', error_message: 'Storage quota exceeded: 0.00MB / 0MB',
      id_maps: { packs: {}, slots: {}, items: {}, characters: {} },
      created_records: { pack_id: null, slot_ids: [], item_ids: [], generated_file_ids: [], writes_started: writesStarted },
      cleanup_status: { status: 'completed' },
    });
    const user = userEvent.setup();
    render(<PortabilityPanel recovery={recovery} />);
    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['backup'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );
    await user.click(await screen.findByRole('button', { name: 'Commit import' }));
    expect(await screen.findByRole('alert')).toHaveTextContent(/storage quota/i);
    if (writesStarted !== false) {
      expect(screen.queryByRole('button', { name: 'Retry import' })).toBeNull();
      return;
    }
    await user.click(screen.getByRole('button', { name: 'Retry import' }));
    const [first, second] = mocks.commitVNPackImport.mock.calls.map(([request]) => request);
    expect(second).toEqual({ ...first, idempotency_key: 'vn-import-retry-commit-job' });
  });

  it('opens the restored pack if the job fails after the journal records completion', async () => {
    const openPack = vi.fn().mockResolvedValue(undefined);
    mocks.getVNPackImportJob.mockResolvedValue({
      job_id: 'commit-job', preview_id: 33, status: 'failed', vn_status: 'completed', stage: 'completed', pack_id: 19,
    });
    const user = userEvent.setup();
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} onOpenImportedPack={openPack} />);
    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['backup'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );
    await user.click(await screen.findByLabelText('Confirm risky update diffs'));
    await user.click(screen.getByRole('button', { name: 'Commit import' }));

    expect(await screen.findByText('Import status: completed')).toBeInTheDocument();
    await user.click(screen.getByRole('button', { name: 'Open imported pack' }));
    expect(openPack).toHaveBeenCalledWith(19, expectedAccount);
    expect(screen.queryByText(/import failed/i)).toBeNull();
  });

  it('does not commit an archive deleted even when its validation job completed', async () => {
    mocks.getVNPackImportPreview.mockResolvedValue({
      preview_id: 33, status: 'completed', vn_status: 'deleted', stage: 'deleted',
    });
    const user = userEvent.setup();
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);
    const input = screen.getByLabelText('Import VN pack archive');
    await user.upload(input, new File(['removed backup'], 'removed.tldw-vnpack'));

    expect(await screen.findByText('Preview status: deleted')).toBeInTheDocument();
    expect(screen.queryByRole('button', { name: 'Commit import' })).toBeNull();
    expect(input).toBeEnabled();
  });

  it('removes a staged archive after its preview is quarantined', async () => {
    mocks.getVNPackImportPreview.mockResolvedValue({
      preview_id: 33, status: 'quarantined', vn_status: 'failed', stage: 'failed',
      error_message: 'invalid_archive: bad_zip_file',
    });
    const user = userEvent.setup();
    render(<PortabilityPanel recovery={recovery} />);
    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['bad backup'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );
    const remove = await screen.findByRole('button', { name: 'Remove staged archive' });
    expect(screen.getByLabelText('Import VN pack archive')).toBeDisabled();
    await user.click(remove);
    expect(mocks.deleteVNPackImportPreview).toHaveBeenCalledWith(33, expectedAccount);
    await waitFor(() => expect(screen.queryByText('Preview status: quarantined')).toBeNull());
    expect(screen.getByLabelText('Import VN pack archive')).toBeEnabled();
  });

  it('renders backup warning and explicit export toggles', () => {
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    expect(screen.getByText('Backup bundle')).toBeInTheDocument();
    expect(screen.getByText(/not encrypted/i)).toBeInTheDocument();
    expect(screen.getByLabelText('Include character payload')).toBeInTheDocument();
    expect(screen.getByLabelText('Include world book payloads')).toBeInTheDocument();
    expect(screen.getByLabelText('Include full provenance')).toBeInTheDocument();
  });

  it('starts export with selected options', async () => {
    const user = userEvent.setup();
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    await user.click(screen.getByLabelText('Include character payload'));
    await user.click(screen.getByLabelText('Include world book payloads'));
    await user.click(screen.getByLabelText('Include full provenance'));
    await user.click(screen.getByRole('button', { name: 'Export backup bundle' }));

    await waitFor(() => {
      expect(mocks.exportVNAssetPack).toHaveBeenCalledWith(9, {
        include_character_payload: true,
        include_world_book_payloads: true,
        include_full_provenance: true,
        strict: false,
        warn_for_sharing: true,
        idempotency_key: expect.stringMatching(/^vn-export-/),
      }, expectedAccount);
    });
    expect(await screen.findByText('Export job: export-job')).toBeInTheDocument();
  });

  it('uploads an archive, renders required character resolution, and commits with trust mode', async () => {
    const user = userEvent.setup();
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    const archive = new File(['vnpack'], 'orbital.tldw-vnpack', { type: 'application/zip' });
    await user.upload(screen.getByLabelText('Import VN pack archive'), archive);

    await waitFor(() => {
      expect(mocks.createVNPackImportPreview).toHaveBeenCalledWith(
        archive,
        expect.stringMatching(/^vn-import-preview-/),
        expectedAccount,
      );
    });
    expect(await screen.findByText('Preview status: completed')).toBeInTheDocument();
    expect(screen.getByText('Character resolution')).toBeInTheDocument();
    expect(screen.getByText('Resolve imported primary character.')).toBeInTheDocument();

    await user.selectOptions(screen.getByLabelText('Trust mode'), 'untrusted_import');
    await user.click(screen.getByLabelText('Confirm risky update diffs'));
    await user.click(screen.getByRole('button', { name: 'Commit import' }));

    await waitFor(() => {
      expect(mocks.commitVNPackImport).toHaveBeenCalledWith({
        preview_id: 33,
        trust_mode: 'untrusted_import',
        target_mode: 'update_existing',
        character_action: 'link_existing_character',
        target_character_id: 42,
        target_pack_id: 9,
        conflict_decisions: { confirm_all_risky_diffs: true },
        idempotency_key: expect.stringMatching(/^vn-import-commit-/),
      }, expectedAccount);
    });
  });

  it('restores the included character and pack on an empty installation', async () => {
    const user = userEvent.setup();
    mocks.getVNPackImportPreview.mockResolvedValueOnce({
      preview_id: 33,
      status: 'completed',
      vn_status: 'completed',
      stage: 'completed',
      required_choices: [{ allowed_actions: ['import_included_character', 'link_existing_character'] }],
    });
    render(<PortabilityPanel recovery={recovery} />);

    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['backup'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );

    const commit = await screen.findByRole('button', { name: 'Commit import' });
    expect(commit).toBeEnabled();
    await user.click(commit);
    expect(mocks.commitVNPackImport).toHaveBeenCalledWith(expect.objectContaining({
      preview_id: 33,
      target_mode: 'create_new',
      character_action: 'import_included_character',
      target_character_id: null,
      target_pack_id: null,
    }), expectedAccount);
  });

  it('guides the user to link a local character with the same name', async () => {
    const user = userEvent.setup();
    mocks.getVNPackImportPreview.mockResolvedValueOnce({
      preview_id: 33,
      status: 'completed',
      vn_status: 'completed',
      stage: 'completed',
      required_choices: [{ allowed_actions: ['import_included_character', 'link_existing_character'] }],
      conflicts: [{ kind: 'character_candidate', signals: ['normalized_name'], local_character_id: 42 }],
    });
    render(<PortabilityPanel recovery={recovery} />);

    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['backup'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );

    expect(await screen.findByRole('button', { name: 'Commit import' })).toBeDisabled();
    expect(screen.getByText(/a character with this name already exists/i)).toBeInTheDocument();
  });

  it('explains why the archive was quarantined and prevents commit', async () => {
    const user = userEvent.setup();
    mocks.getVNPackImportPreview.mockResolvedValueOnce({
      preview_id: 33,
      status: 'quarantined',
      stage: 'failed',
      error_message: 'invalid_archive: bad_zip_file',
    });
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['incomplete'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );

    expect(await screen.findByRole('alert')).toHaveTextContent(
      'The archive could not be read. Download it again or choose another VN pack backup.'
    );
    expect(screen.getByRole('button', { name: 'Commit import' })).toBeDisabled();
  });

  it('ignores a previous validation error after preview succeeds on retry', async () => {
    const user = userEvent.setup();
    mocks.getVNPackImportPreview.mockResolvedValueOnce({
      preview_id: 33,
      status: 'completed',
      vn_status: 'completed',
      stage: 'completed',
      error_message: 'temporary preview failure',
    });
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    await user.upload(
      screen.getByLabelText('Import VN pack archive'),
      new File(['backup'], 'orbital.tldw-vnpack', { type: 'application/zip' })
    );

    await screen.findByText('Preview status: completed');
    expect(screen.queryByRole('alert')).toBeNull();
    expect(screen.getByRole('button', { name: 'Commit import' })).toBeEnabled();
  });

  it('polls an import preview until it reaches a terminal status', async () => {
    const setTimeoutSpy = vi.spyOn(globalThis, 'setTimeout').mockImplementation((handler, _timeout, ...args) => {
      if (typeof handler === 'function') {
        queueMicrotask(() => handler(...args));
      }
      return 0 as ReturnType<typeof globalThis.setTimeout>;
    });
    mocks.getVNPackImportPreview
      .mockResolvedValueOnce({
        preview_id: 33,
        job_id: 'preview-job',
        portability_job_id: 2,
        operation: 'import_preview',
        status: 'queued',
        vn_status: 'queued',
        stage: 'queued',
        bundle_summary: {},
        required_choices: [],
        proposed_plan: {},
        conflicts: [],
        validation_warnings: [],
        quota_estimate: {},
      })
      .mockResolvedValueOnce({
        preview_id: 33,
        job_id: 'preview-job',
        portability_job_id: 2,
        operation: 'import_preview',
        status: 'processing',
        vn_status: 'processing',
        stage: 'validating_archive',
        bundle_summary: {},
        required_choices: [],
        proposed_plan: {},
        conflicts: [],
        validation_warnings: [],
        quota_estimate: {},
      })
      .mockResolvedValueOnce({
        preview_id: 33,
        job_id: 'preview-job',
        portability_job_id: 2,
        operation: 'import_preview',
        status: 'completed',
        vn_status: 'completed',
        stage: 'completed',
        bundle_summary: { pack_title: 'Imported Backup', item_count: 2 },
        required_choices: [],
        proposed_plan: {},
        conflicts: [],
        validation_warnings: [],
        quota_estimate: { asset_bytes: 1024 },
      });

    try {
      render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

      const archive = new File(['vnpack'], 'orbital.tldw-vnpack', { type: 'application/zip' });
      await act(async () => {
        fireEvent.change(screen.getByLabelText('Import VN pack archive'), {
          target: { files: [archive] },
        });
        for (let index = 0; index < 8; index += 1) {
          await Promise.resolve();
        }
      });

      expect(setTimeoutSpy).toHaveBeenCalledTimes(2);
      expect(mocks.getVNPackImportPreview).toHaveBeenCalledTimes(3);
      expect(screen.getByText('Preview status: completed')).toBeInTheDocument();
    } finally {
      setTimeoutSpy.mockRestore();
    }
  });

  it('requires confirmation before committing risky update-existing diffs', async () => {
    const user = userEvent.setup();
    render(<PortabilityPanel recovery={recovery} selectedPack={pack} />);

    const archive = new File(['vnpack'], 'orbital.tldw-vnpack', { type: 'application/zip' });
    await user.upload(screen.getByLabelText('Import VN pack archive'), archive);
    await screen.findByText('Risky update diffs');

    await user.click(screen.getByRole('button', { name: 'Commit import' }));

    expect(mocks.commitVNPackImport).not.toHaveBeenCalled();
    expect(screen.getByText('Confirm risky update diffs before committing this update.')).toBeInTheDocument();
  });
});
