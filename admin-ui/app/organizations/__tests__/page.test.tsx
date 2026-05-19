/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import OrganizationsPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const privilegedActionMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());
const setPageMock = vi.hoisted(() => vi.fn());
const setPageSizeMock = vi.hoisted(() => vi.fn());
const resetPaginationMock = vi.hoisted(() => vi.fn());

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

vi.mock('@/components/ui/confirm-dialog', () => ({
  useConfirm: () => confirmMock,
}));

vi.mock('@/components/ui/privileged-action-dialog', () => ({
  usePrivilegedActionDialog: () => privilegedActionMock,
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
    getOrganizations: vi.fn(),
    getSubscriptions: vi.fn(),
    createOrganization: vi.fn(),
    updateOrganization: vi.fn(),
    getOrgMembers: vi.fn(),
    deleteOrganization: vi.fn(),
  },
}));

type ApiMock = {
  getOrganizations: ReturnType<typeof vi.fn>;
  getSubscriptions: ReturnType<typeof vi.fn>;
  createOrganization: ReturnType<typeof vi.fn>;
  updateOrganization: ReturnType<typeof vi.fn>;
  getOrgMembers: ReturnType<typeof vi.fn>;
  deleteOrganization: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  confirmMock.mockResolvedValue(true);
  privilegedActionMock.mockResolvedValue({ reason: 'test audit reason', adminPassword: '' });
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getOrganizations.mockResolvedValue({
    items: [
      {
        id: 1,
        name: 'Acme Inc',
        slug: 'acme-inc',
        created_at: '2026-01-10T00:00:00Z',
      },
    ],
    total: 1,
  });
  apiMock.getSubscriptions.mockResolvedValue([]);
  apiMock.getOrgMembers.mockResolvedValue([
    { user_id: 11 },
    { user_id: 12 },
  ]);
  apiMock.updateOrganization.mockResolvedValue({});
  apiMock.deleteOrganization.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('OrganizationsPage critical CRUD controls', () => {
  it('shows slug guidance and blocks invalid slug updates', async () => {
    const user = userEvent.setup();
    render(<OrganizationsPage />);

    const rowLabel = await screen.findByText('Acme Inc');
    const row = rowLabel.closest('tr');
    expect(row).not.toBeNull();

    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Edit' }));

    expect(screen.getByText('Slug must be unique across organizations.')).toBeInTheDocument();

    const slugInput = screen.getByLabelText('Slug');
    await user.clear(slugInput);
    await user.type(slugInput, 'Invalid Slug');
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    expect(await screen.findByText('Slug must be lowercase letters, numbers, and hyphens.')).toBeInTheDocument();
    expect(apiMock.updateOrganization).not.toHaveBeenCalled();
  });

  it('acknowledges member count before delete and calls delete API', async () => {
    const user = userEvent.setup();
    render(<OrganizationsPage />);

    const rowLabel = await screen.findByText('Acme Inc');
    const row = rowLabel.closest('tr');
    expect(row).not.toBeNull();

    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(privilegedActionMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Delete organization',
          message: expect.stringContaining('This organization has 2 members'),
        })
      );
    });

    await waitFor(() => {
      expect(apiMock.deleteOrganization).toHaveBeenCalledWith('1');
    });
  });
});
