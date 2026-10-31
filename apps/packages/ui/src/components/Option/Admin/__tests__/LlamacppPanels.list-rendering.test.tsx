import React from 'react';
import { describe, expect, it, vi } from 'vitest';
import { render, screen, within } from '@testing-library/react';
import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';
import { LlamacppAssetsPanel } from '../LlamacppAssetsPanel';
import { LlamacppInventoryPanel } from '../LlamacppInventoryPanel';
import { LlamacppProfilesPanel } from '../LlamacppProfilesPanel';
import { LlamacppRuntimePanel } from '../LlamacppRuntimePanel';
import type { LlamacppAsset, LlamacppProfile } from '@/types/llamacpp-admin';

void i18next.use(initReactI18next).init({ lng: 'en', resources: {} });

const asset: LlamacppAsset = {
  asset_id: 'gguf:model',
  kind: 'gguf',
  identity_basis: 'resolved_path',
  path: '/models/model.gguf',
  resolved_path: '/models/model.gguf',
  display_name: 'Local model',
  source: 'models_dir',
  metadata: {},
  capabilities: ['unknown'],
  mmproj_asset_ids: [],
  base_model_asset_ids: [],
  warnings: [],
};
const projector: LlamacppAsset = {
  ...asset,
  asset_id: 'mmproj:model',
  kind: 'mmproj',
  display_name: 'Matching projector',
  path: '/models/mmproj.gguf',
  resolved_path: '/models/mmproj.gguf',
};
const profile: LlamacppProfile = {
  profile_id: 'local',
  name: 'Local profile',
  enabled: true,
  mode: 'vision',
  model_id: asset.asset_id,
  model_path: asset.path,
  mmproj_model_id: projector.asset_id,
  host: '127.0.0.1',
  port: 8181,
  port_policy: 'explicit',
  server_args: {},
  autostart: false,
  restart_policy: {},
  tags: [],
};
const assetsProps = {
  assets: { assets: [asset, projector], warnings: [], scan_limited: false },
  onRegisterPath: vi.fn(),
  onImportFolder: vi.fn(),
  onReload: vi.fn(),
};
const inventoryProps = {
  inventory: {
    models: [
      {
        model_id: 'gguf:model',
        display_name: 'Inventory model',
        basename: 'model.gguf',
        path: asset.path,
        source: 'models_dir',
        metadata: {},
        warnings: [],
      },
    ],
    warnings: [],
    scan_limited: false,
  },
  onSelectModel: vi.fn(),
  onRegisterPath: vi.fn(),
  onReload: vi.fn(),
};
const profilesProps = {
  profiles: [profile],
  assets: assetsProps.assets,
  onRefresh: vi.fn(),
  onCreate: vi.fn(),
  onUpdate: vi.fn(),
  onDelete: vi.fn(),
};
const runtimeProps = {
  profiles: [profile],
  runtimes: [],
  onRefresh: vi.fn(),
  onStart: vi.fn(),
  onStop: vi.fn(),
  onPause: vi.fn(),
  onResume: vi.fn(),
  onUseInChat: vi.fn(),
};

const populatedPanels = [
  {
    name: 'asset groups',
    content: 'Matching projector',
    render: () => <LlamacppAssetsPanel {...assetsProps} />,
  },
  {
    name: 'downloads',
    content: 'Queued model',
    render: () => (
      <LlamacppAssetsPanel
        {...assetsProps}
        assets={{ assets: [], warnings: [], scan_limited: false }}
        onCancelDownload={vi.fn()}
        downloads={{
          jobs: [
            {
              job_id: 'download-1',
              status: 'running',
              operation: 'download',
              queue: 'acquisition',
              source_label: 'Queued model',
              progress: { progress_percent: 25 },
              warnings: [],
            },
          ],
        }}
      />
    ),
  },
  {
    name: 'profiles',
    content: 'Local profile',
    render: () => <LlamacppProfilesPanel {...profilesProps} />,
  },
  {
    name: 'runtimes',
    content: 'Local profile',
    render: () => <LlamacppRuntimePanel {...runtimeProps} />,
  },
  {
    name: 'inventory',
    content: 'Inventory model',
    render: () => <LlamacppInventoryPanel {...inventoryProps} />,
  },
];

describe('llama.cpp admin list rendering with real AntD', () => {
  it.each(populatedPanels)(
    'renders $name without deprecated component errors',
    ({ render: panel, content }) => {
      // Observe real AntD warnings before the global test log filter.
      const errors = vi.spyOn(console, 'error').mockImplementation(() => undefined);
      render(panel());
      expect(screen.getByText(content)).toBeVisible();
      expect(errors.mock.calls).toEqual([]);
    }
  );

  it('keeps model and projector groups in separate accessible lists', () => {
    render(<LlamacppAssetsPanel {...assetsProps} />);
    expect(
      within(screen.getByRole('region', { name: 'GGUF models' })).getByRole('list')
    ).toHaveTextContent('Local model');
    expect(
      within(screen.getByRole('region', { name: 'mmproj projectors' })).getByRole('list')
    ).toHaveTextContent('Matching projector');
  });

  it('keeps download status visible while refreshing its list', () => {
    render(
      <LlamacppAssetsPanel
        {...assetsProps}
        loadingDownloads
        downloads={{
          jobs: [
            {
              job_id: 'download-1',
              status: 'running',
              operation: 'download',
              queue: 'acquisition',
              source_label: 'Refreshing download',
              progress: { progress_percent: 25 },
              warnings: [],
            },
          ],
        }}
      />
    );
    const downloads = screen.getByRole('list', { name: 'Downloads' });
    expect(downloads).toHaveAttribute('aria-busy', 'true');
    expect(within(downloads).getByText('Refreshing download')).toBeVisible();
    expect(within(downloads).getByText('25%')).toBeVisible();
  });

  it('retains empty guidance and hides stale rows while the cards load', () => {
    const { rerender } = render(<LlamacppAssetsPanel {...assetsProps} assets={null} />);
    expect(screen.getByText(/No llama.cpp assets detected/)).toBeVisible();
    rerender(<LlamacppAssetsPanel {...assetsProps} loading />);
    expect(screen.queryByText('Local model')).toBeNull();
    rerender(<LlamacppInventoryPanel {...inventoryProps} inventory={null} />);
    expect(screen.getByText(/No local GGUF models detected/)).toBeVisible();
    rerender(<LlamacppInventoryPanel {...inventoryProps} loading />);
    expect(screen.queryByText('Inventory model')).toBeNull();
    rerender(<LlamacppProfilesPanel {...profilesProps} profiles={[]} />);
    expect(screen.getByText('No saved llama.cpp profiles are available.')).toBeVisible();
    rerender(<LlamacppProfilesPanel {...profilesProps} loading />);
    expect(screen.queryByText('Local profile')).toBeNull();
    rerender(<LlamacppRuntimePanel {...runtimeProps} profiles={[]} />);
    expect(screen.getByText('No llama.cpp runtime profiles are available.')).toBeVisible();
    rerender(<LlamacppRuntimePanel {...runtimeProps} loading />);
    expect(screen.queryByText('Local profile')).toBeNull();
  });
});
