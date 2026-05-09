import { beforeEach, describe, expect, it, vi } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';

const mocks = vi.hoisted(() => ({
  applyVNAssetMatrix: vi.fn(),
  commitVNPackImport: vi.fn(),
  createVNAssetPack: vi.fn(),
  createVNPackImportPreview: vi.fn(),
  exportVNAssetPack: vi.fn(),
  getStarterMatrices: vi.fn(),
  getVNAssetGeneration: vi.fn(),
  getVNAssetReadiness: vi.fn(),
  getVNPackImportPreview: vi.fn(),
  listVNAssetItems: vi.fn(),
  listVNAssetPacks: vi.fn(),
  listVNAssetSlots: vi.fn(),
}));

vi.mock('@web/lib/api/vnAssets', () => ({
  applyVNAssetMatrix: (...args: unknown[]) => mocks.applyVNAssetMatrix(...args),
  commitVNPackImport: (...args: unknown[]) => mocks.commitVNPackImport(...args),
  createVNAssetPack: (...args: unknown[]) => mocks.createVNAssetPack(...args),
  createVNPackImportPreview: (...args: unknown[]) => mocks.createVNPackImportPreview(...args),
  exportVNAssetPack: (...args: unknown[]) => mocks.exportVNAssetPack(...args),
  getStarterMatrices: (...args: unknown[]) => mocks.getStarterMatrices(...args),
  getVNAssetGeneration: (...args: unknown[]) => mocks.getVNAssetGeneration(...args),
  getVNAssetReadiness: (...args: unknown[]) => mocks.getVNAssetReadiness(...args),
  getVNPackImportPreview: (...args: unknown[]) => mocks.getVNPackImportPreview(...args),
  listVNAssetItems: (...args: unknown[]) => mocks.listVNAssetItems(...args),
  listVNAssetPacks: (...args: unknown[]) => mocks.listVNAssetPacks(...args),
  listVNAssetSlots: (...args: unknown[]) => mocks.listVNAssetSlots(...args),
}));

import VNAssetsWorkbench from '@web/components/vn-assets/VNAssetsWorkbench';

describe('VNAssetsWorkbench', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mocks.applyVNAssetMatrix.mockResolvedValue([]);
    mocks.commitVNPackImport.mockResolvedValue({});
    mocks.createVNAssetPack.mockResolvedValue({
      id: 7,
      title: 'Orbital Library',
      primary_character_id: 42,
      planned_output_count: 0,
      status: 'draft',
    });
    mocks.createVNPackImportPreview.mockResolvedValue({});
    mocks.exportVNAssetPack.mockResolvedValue({});
    mocks.getStarterMatrices.mockResolvedValue({
      matrices: [
        {
          key: 'starter',
          title: 'Starter',
          slot_count: 8,
          planned_output_count: 24,
          asset_types: ['background', 'sprite', 'cg'],
        },
      ],
    });
    mocks.getVNAssetGeneration.mockResolvedValue({ status: 'idle' });
    mocks.getVNAssetReadiness.mockResolvedValue({ ready: false, status: 'not_ready', warnings: [], errors: [] });
    mocks.getVNPackImportPreview.mockResolvedValue({});
    mocks.listVNAssetItems.mockResolvedValue([]);
    mocks.listVNAssetPacks.mockResolvedValue([]);
    mocks.listVNAssetSlots.mockResolvedValue([]);
  });

  it('renders loading, empty, setup, matrix preview, and placeholders', async () => {
    render(<VNAssetsWorkbench />);

    expect(screen.getByText('Loading VN asset packs...')).toBeInTheDocument();
    expect(await screen.findByText('No asset packs yet.')).toBeInTheDocument();
    expect(screen.getByLabelText('Pack title')).toBeInTheDocument();
    expect(screen.getByLabelText('Primary character ID')).toBeInTheDocument();
    expect(screen.getByText('Starter matrix')).toBeInTheDocument();
    expect(screen.getByText('24 planned assets')).toBeInTheDocument();
    expect(screen.getByText('Generation monitor')).toBeInTheDocument();
    expect(screen.getByText('Review board')).toBeInTheDocument();
    expect(screen.getByRole('heading', { name: 'Portability' })).toBeInTheDocument();
  });

  it('creates a pack and selects it for the workbench summary', async () => {
    const user = userEvent.setup();
    render(<VNAssetsWorkbench />);

    await screen.findByText('No asset packs yet.');
    await user.clear(screen.getByLabelText('Pack title'));
    await user.type(screen.getByLabelText('Pack title'), 'Orbital Library');
    await user.clear(screen.getByLabelText('Primary character ID'));
    await user.type(screen.getByLabelText('Primary character ID'), '42');
    await user.click(screen.getByRole('button', { name: 'Create pack' }));

    await waitFor(() => {
      expect(mocks.createVNAssetPack).toHaveBeenCalledWith({
        title: 'Orbital Library',
        primary_character_id: 42,
        apply_starter_matrix: false,
      });
    });
    expect(await screen.findByText('Orbital Library')).toBeInTheDocument();
    expect(screen.getAllByText('Character 42').length).toBeGreaterThan(0);
    expect(screen.getByText('0 planned assets')).toBeInTheDocument();
  });

  it('uses returned slot variants for the planned asset count after matrix apply', async () => {
    const user = userEvent.setup();
    mocks.applyVNAssetMatrix.mockResolvedValue([
      {
        id: 1,
        pack_id: 7,
        asset_type: 'sprite',
        slot_key: 'sprite.primary',
        variant_count: 2,
        status: 'planned',
      },
      {
        id: 2,
        pack_id: 7,
        asset_type: 'background',
        slot_key: 'background.interior',
        variant_count: 3,
        status: 'planned',
      },
    ]);
    render(<VNAssetsWorkbench />);

    await screen.findByText('No asset packs yet.');
    await user.click(screen.getByRole('button', { name: 'Create pack' }));
    await user.click(await screen.findByRole('button', { name: 'Apply starter matrix' }));

    await waitFor(() => {
      expect(mocks.applyVNAssetMatrix).toHaveBeenCalledWith(7, 'starter', { variant_count: 1 });
    });
    expect(await screen.findByText('5 planned assets')).toBeInTheDocument();
  });
});
