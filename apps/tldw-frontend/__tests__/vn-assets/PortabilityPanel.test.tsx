import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, fireEvent, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import type { VNAssetPack } from '@web/types/vn-assets';

const mocks = vi.hoisted(() => ({
  commitVNPackImport: vi.fn(),
  createVNPackImportPreview: vi.fn(),
  exportVNAssetPack: vi.fn(),
  getVNPackImportPreview: vi.fn(),
}));

vi.mock('@web/lib/api/vnAssets', () => ({
  commitVNPackImport: (...args: unknown[]) => mocks.commitVNPackImport(...args),
  createVNPackImportPreview: (...args: unknown[]) => mocks.createVNPackImportPreview(...args),
  exportVNAssetPack: (...args: unknown[]) => mocks.exportVNAssetPack(...args),
  getVNPackImportPreview: (...args: unknown[]) => mocks.getVNPackImportPreview(...args),
}));

import PortabilityPanel from '@web/components/vn-assets/PortabilityPanel';

const pack: VNAssetPack = {
  id: 9,
  title: 'Orbital Library',
  primary_character_id: 42,
  planned_output_count: 24,
};

describe('PortabilityPanel', () => {
  beforeEach(() => {
    vi.useRealTimers();
    vi.clearAllMocks();
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

  it('renders backup warning and explicit export toggles', () => {
    render(<PortabilityPanel selectedPack={pack} />);

    expect(screen.getByText('Backup bundle')).toBeInTheDocument();
    expect(screen.getByText(/not encrypted/i)).toBeInTheDocument();
    expect(screen.getByLabelText('Include character payload')).toBeInTheDocument();
    expect(screen.getByLabelText('Include world book payloads')).toBeInTheDocument();
    expect(screen.getByLabelText('Include full provenance')).toBeInTheDocument();
  });

  it('starts export with selected options', async () => {
    const user = userEvent.setup();
    render(<PortabilityPanel selectedPack={pack} />);

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
      });
    });
    expect(await screen.findByText('Export job: export-job')).toBeInTheDocument();
  });

  it('uploads an archive, renders required character resolution, and commits with trust mode', async () => {
    const user = userEvent.setup();
    render(<PortabilityPanel selectedPack={pack} />);

    const archive = new File(['vnpack'], 'orbital.tldw-vnpack', { type: 'application/zip' });
    await user.upload(screen.getByLabelText('Import VN pack archive'), archive);

    await waitFor(() => {
      expect(mocks.createVNPackImportPreview).toHaveBeenCalledWith(
        archive,
        expect.stringMatching(/^vn-import-preview-/)
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
      });
    });
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
      render(<PortabilityPanel selectedPack={pack} />);

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
    render(<PortabilityPanel selectedPack={pack} />);

    const archive = new File(['vnpack'], 'orbital.tldw-vnpack', { type: 'application/zip' });
    await user.upload(screen.getByLabelText('Import VN pack archive'), archive);
    await screen.findByText('Risky update diffs');

    await user.click(screen.getByRole('button', { name: 'Commit import' }));

    expect(mocks.commitVNPackImport).not.toHaveBeenCalled();
    expect(screen.getByText('Confirm risky update diffs before committing this update.')).toBeInTheDocument();
  });
});
