/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { act, cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import ApiKeysPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());
const setPageMock = vi.hoisted(() => vi.fn());
const setPageSizeMock = vi.hoisted(() => vi.fn());
const resetPaginationMock = vi.hoisted(() => vi.fn());

const createDeferred = <T,>() => {
  let resolve!: (value: T | PromiseLike<T>) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
};

vi.mock('next/link', () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a>,
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
}));

vi.mock('@/components/ResponsiveLayout', () => ({
  ResponsiveLayout: ({ children }: { children: ReactNode }) => (
    <div data-testid="layout">{children}</div>
  ),
}));

vi.mock('@/components/OrgContextSwitcher', () => ({
  useOrgContext: () => ({ selectedOrg: null }),
  OrgContextSwitcher: () => <div data-testid="org-switcher" />,
}));

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => confirmMock,
}));

vi.mock('@/components/ui/toast', () => ({
  useToast: () => ({
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/use-url-state', () => ({
  useUrlState: (_key: string, options?: { defaultValue?: string }) => [options?.defaultValue ?? '', vi.fn()],
  useUrlPagination: () => ({
    page: 1,
    pageSize: 25,
    setPage: setPageMock,
    setPageSize: setPageSizeMock,
    resetPagination: resetPaginationMock,
  }),
}));

vi.mock('@/lib/api-client', () => ({
  api: {
    getUsersPage: vi.fn(),
    getUserApiKeys: vi.fn(),
    rotateApiKey: vi.fn(),
    createApiKey: vi.fn(),
    getTopApiKeyUsage: vi.fn(),
    revokeApiKey: vi.fn(),
    getAuditLogs: vi.fn(),
  },
}));

type ApiMock = {
  getUsersPage: ReturnType<typeof vi.fn>;
  getUserApiKeys: ReturnType<typeof vi.fn>;
  rotateApiKey: ReturnType<typeof vi.fn>;
  createApiKey: ReturnType<typeof vi.fn>;
  getTopApiKeyUsage: ReturnType<typeof vi.fn>;
  revokeApiKey: ReturnType<typeof vi.fn>;
  getAuditLogs: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  confirmMock.mockResolvedValue(true);
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getUsersPage.mockResolvedValue({
    items: [
      { id: 1, username: 'alice', email: 'alice@example.com' },
      { id: 2, username: 'bob', email: 'bob@example.com' },
      { id: 3, username: 'charlie', email: 'charlie@example.com' },
    ],
    total: 3,
    page: 1,
    pages: 1,
    limit: 100,
  });

  apiMock.getUserApiKeys.mockImplementation(async (userId: string) => {
    if (userId === '1') {
      return [{
        id: 101,
        key_prefix: 'sk-alice',
        status: 'active',
        created_at: '2026-02-15T00:00:00Z',
        expires_at: null,
        last_used_at: '2026-02-16T00:00:00Z',
      }];
    }
    if (userId === '2') {
      return [{
        id: 202,
        key_prefix: 'sk-bob',
        status: 'active',
        created_at: '2026-02-14T00:00:00Z',
        expires_at: null,
        last_used_at: '2026-02-16T00:00:00Z',
      }];
    }
    return [];
  });

  apiMock.rotateApiKey.mockResolvedValue({ status: 'stored' });
  apiMock.createApiKey.mockResolvedValue({ key: 'sk-new-test-key-123' });
  apiMock.getTopApiKeyUsage.mockResolvedValue({ items: [] });
  apiMock.revokeApiKey.mockResolvedValue({ status: 'revoked' });
  apiMock.getAuditLogs.mockResolvedValue({ items: [] });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('ApiKeysPage', () => {
  it('supports bulk key rotation with selection and confirmation', async () => {
    const user = userEvent.setup();

    render(<ApiKeysPage />);

    expect(await screen.findByLabelText('Select key sk-alice')).toBeInTheDocument();
    expect(screen.getByLabelText('Select key sk-bob')).toBeInTheDocument();

    await user.click(screen.getByLabelText('Select all keys'));
    expect(screen.getByRole('button', { name: 'Rotate Selected (2)' })).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Rotate Selected (2)' }));

    await waitFor(() => {
      expect(confirmMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Rotate selected keys',
          message: 'Rotate 2 key(s)? Existing keys will stop working immediately.',
        })
      );
    });

    await waitFor(() => {
      expect(apiMock.rotateApiKey).toHaveBeenCalledWith('1', '101');
      expect(apiMock.rotateApiKey).toHaveBeenCalledWith('2', '202');
    });

    expect(toastSuccessMock).toHaveBeenCalledWith(
      'Bulk rotation complete',
      '2 key(s) rotated successfully.'
    );

    await waitFor(() => {
      expect(screen.getByRole('button', { name: 'Rotate Selected (0)' })).toBeDisabled();
    });
  });

  it('renders a Create Key button that opens a dialog', async () => {
    const user = userEvent.setup();
    render(<ApiKeysPage />);

    const createButton = await screen.findByRole('button', { name: /Create Key/i });
    expect(createButton).toBeInTheDocument();

    await user.click(createButton);

    await waitFor(() => {
      expect(screen.getByText('Create API Key')).toBeInTheDocument();
      expect(screen.getByText('Select a user and configure the new API key.')).toBeInTheDocument();
    });
  });

  it('shows users without existing keys in the Create Key owner list', async () => {
    const user = userEvent.setup();
    render(<ApiKeysPage />);

    await user.click(await screen.findByRole('button', { name: /Create Key/i }));

    const ownerSelect = await screen.findByLabelText('User');
    expect(within(ownerSelect).getByRole('option', { name: 'alice' })).toBeInTheDocument();
    expect(within(ownerSelect).getByRole('option', { name: 'bob' })).toBeInTheDocument();
    expect(within(ownerSelect).getByRole('option', { name: 'charlie' })).toBeInTheDocument();
  });

  it('shows expiring-soon warning banner when keys expire within 7 days', async () => {
    const nearFuture = new Date(Date.now() + 3 * 24 * 60 * 60 * 1000).toISOString();

    apiMock.getUserApiKeys.mockImplementation(async (userId: string) => {
      if (userId === '1') {
        return [{
          id: 101,
          key_prefix: 'sk-expiring',
          status: 'active',
          created_at: '2026-02-15T00:00:00Z',
          expires_at: nearFuture,
          last_used_at: '2026-02-16T00:00:00Z',
        }];
      }
      return [{
        id: 202,
        key_prefix: 'sk-bob',
        status: 'active',
        created_at: '2026-02-14T00:00:00Z',
        expires_at: null,
        last_used_at: '2026-02-16T00:00:00Z',
      }];
    });

    render(<ApiKeysPage />);

    await waitFor(() => {
      expect(screen.getByText(/1 key expires within 7 days/i)).toBeInTheDocument();
    });

    expect(screen.getByRole('button', { name: /View expiring keys/i })).toBeInTheDocument();
  });

  it('disables both bulk actions while a bulk rotation is in progress', async () => {
    const user = userEvent.setup();
    const pendingRotations = [
      createDeferred<{ status: string }>(),
      createDeferred<{ status: string }>(),
    ];
    let rotationIndex = 0;
    apiMock.rotateApiKey.mockImplementation(() => pendingRotations[rotationIndex++].promise);

    render(<ApiKeysPage />);

    expect(await screen.findByLabelText('Select key sk-alice')).toBeInTheDocument();
    await user.click(screen.getByLabelText('Select all keys'));

    const rotateButton = screen.getByRole('button', { name: 'Rotate Selected (2)' });
    const revokeButton = screen.getByRole('button', { name: 'Revoke Selected (2)' });

    await user.click(rotateButton);

    await waitFor(() => {
      expect(apiMock.rotateApiKey).toHaveBeenCalledTimes(2);
    });
    await waitFor(() => {
      expect(rotateButton).toBeDisabled();
      expect(revokeButton).toBeDisabled();
    });

    await user.click(revokeButton);
    expect(confirmMock).toHaveBeenCalledTimes(1);
    expect(apiMock.revokeApiKey).not.toHaveBeenCalled();

    await act(async () => {
      pendingRotations.forEach((deferred) => deferred.resolve({ status: 'stored' }));
      await Promise.resolve();
    });
  });
});
