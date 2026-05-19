/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import FlagsPage from '../page';
import { api } from '@/lib/api-client';

const promptPrivilegedActionMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());
const toastWarningMock = vi.hoisted(() => vi.fn());

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => promptPrivilegedActionMock,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
    warning: toastWarningMock,
  }),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getMaintenanceMode: vi.fn(),
    updateMaintenanceMode: vi.fn(),
    getFeatureFlags: vi.fn(),
    upsertFeatureFlag: vi.fn(),
    deleteFeatureFlag: vi.fn(),
  },
}));

type ApiMock = {
  getMaintenanceMode: ReturnType<typeof vi.fn>;
  updateMaintenanceMode: ReturnType<typeof vi.fn>;
  getFeatureFlags: ReturnType<typeof vi.fn>;
  upsertFeatureFlag: ReturnType<typeof vi.fn>;
  deleteFeatureFlag: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  promptPrivilegedActionMock.mockResolvedValue({ reason: 'test', adminPassword: '' });
  toastSuccessMock.mockReset();
  toastErrorMock.mockReset();
  toastWarningMock.mockReset();

  apiMock.getMaintenanceMode.mockResolvedValue({
    enabled: false,
    message: '',
    allowlist_user_ids: [],
    allowlist_emails: [],
  });
  apiMock.getFeatureFlags.mockResolvedValue({
    items: [
      {
        key: 'checkout.redesign',
        scope: 'global',
        enabled: true,
        description: 'Checkout redesign rollout',
        rollout_percent: 37,
        target_user_ids: [11, 42],
        variant_value: 'variant_a',
        updated_at: '2026-02-17T10:00:00Z',
        history: [
          {
            timestamp: '2026-02-17T10:00:00Z',
            enabled: true,
            actor: 'ops@example.com',
            before: {
              scope: 'global',
              enabled: true,
              target_user_ids: [11],
              rollout_percent: 10,
              variant_value: 'control',
            },
            after: {
              scope: 'global',
              enabled: true,
              target_user_ids: [11, 42],
              rollout_percent: 37,
              variant_value: 'variant_a',
            },
          },
        ],
      },
    ],
  });
  apiMock.upsertFeatureFlag.mockResolvedValue({});
  apiMock.deleteFeatureFlag.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('FlagsPage Stage 4', () => {
  it('renders rollout percentage progress and history diff', async () => {
    const user = userEvent.setup();
    render(<FlagsPage />);

    expect(await screen.findByText('Feature Flags')).toBeInTheDocument();
    expect(await screen.findByText('37% rollout')).toBeInTheDocument();
    expect(screen.getByRole('progressbar', { name: 'Rollout 37%' }).getAttribute('aria-valuenow')).toBe('37');

    await user.click(screen.getByText('1 changes'));
    expect(await screen.findByText('Rollout %:')).toBeInTheDocument();
    expect(screen.getByText('Target users:')).toBeInTheDocument();
    expect(screen.getByText('Variant:')).toBeInTheDocument();
  });

  it('validates target user list parsing before save', async () => {
    const user = userEvent.setup();
    render(<FlagsPage />);

    await screen.findByText('Feature Flags');
    await user.type(screen.getByLabelText('Flag Key'), 'new.experiment');
    await user.clear(screen.getByLabelText('Rollout %'));
    await user.type(screen.getByLabelText('Rollout %'), '50');
    await user.type(screen.getByLabelText('Target User IDs (comma-separated)'), '1, abc, 3');
    await user.click(screen.getByRole('button', { name: 'Save Flag' }));

    expect(apiMock.upsertFeatureFlag).not.toHaveBeenCalled();
    expect(toastErrorMock).toHaveBeenCalledWith(
      'Target user IDs must be positive integers',
      expect.stringContaining('abc')
    );
  });

  it('submits rollout percent, deduped target list, and variant value', async () => {
    const user = userEvent.setup();
    render(<FlagsPage />);

    await screen.findByText('Feature Flags');
    await user.type(screen.getByLabelText('Flag Key'), 'new.experiment');
    await user.clear(screen.getByLabelText('Rollout %'));
    await user.type(screen.getByLabelText('Rollout %'), '65');
    await user.type(screen.getByLabelText('Target User IDs (comma-separated)'), '1, 3, 3');
    await user.type(screen.getByLabelText('Variant Value (optional)'), 'variant_b');
    await user.click(screen.getByRole('button', { name: 'Save Flag' }));

    await waitFor(() => {
      expect(apiMock.upsertFeatureFlag).toHaveBeenCalledWith(
        'new.experiment',
        expect.objectContaining({
          rollout_percent: 65,
          target_user_ids: [1, 3],
          variant_value: 'variant_b',
        })
      );
    });
  });

  it('preserves existing flag configuration when toggling enabled state', async () => {
    const user = userEvent.setup();
    render(<FlagsPage />);

    const toggleButton = await screen.findByRole('button', { name: 'Enabled' });
    expect(toggleButton.getAttribute('aria-pressed')).toBe('true');
    await user.click(toggleButton);

    await waitFor(() => {
      expect(apiMock.upsertFeatureFlag).toHaveBeenCalledWith(
        'checkout.redesign',
        expect.objectContaining({
          scope: 'global',
          enabled: false,
          description: 'Checkout redesign rollout',
          target_user_ids: [11, 42],
          rollout_percent: 37,
          variant_value: 'variant_a',
        })
      );
    });
  });
});
