/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TeamsPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const privilegedActionMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());
const setPageMock = vi.hoisted(() => vi.fn());
const setPageSizeMock = vi.hoisted(() => vi.fn());
const resetPaginationMock = vi.hoisted(() => vi.fn());
const setSelectedOrgIdMock = vi.hoisted(() => vi.fn());

let currentSelectedOrgId = '10';

vi.mock('next/link', () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a>,
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
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
    warning: vi.fn(),
    success: toastSuccessMock,
    error: toastErrorMock,
  }),
}));

vi.mock('@/lib/use-url-state', () => ({
  useUrlState: (key: string, options?: { defaultValue?: string }) => {
    if (key === 'org') return [currentSelectedOrgId, setSelectedOrgIdMock];
    return [options?.defaultValue ?? '', vi.fn()];
  },
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
    getTeams: vi.fn(),
    createTeam: vi.fn(),
    updateTeam: vi.fn(),
    getTeamMembers: vi.fn(),
    deleteTeam: vi.fn(),
  },
}));

type ApiMock = {
  getOrganizations: ReturnType<typeof vi.fn>;
  getTeams: ReturnType<typeof vi.fn>;
  createTeam: ReturnType<typeof vi.fn>;
  updateTeam: ReturnType<typeof vi.fn>;
  getTeamMembers: ReturnType<typeof vi.fn>;
  deleteTeam: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  currentSelectedOrgId = '10';
  confirmMock.mockResolvedValue(true);
  privilegedActionMock.mockResolvedValue({ reason: 'test', adminPassword: '' });
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();
  setSelectedOrgIdMock.mockClear();

  apiMock.getOrganizations.mockResolvedValue([
    {
      id: 10,
      name: 'Acme',
      slug: 'acme',
    },
  ]);
  apiMock.getTeams.mockResolvedValue([
    {
      id: 5,
      org_id: 10,
      name: 'Team One',
      description: 'Team one description',
      created_at: '2026-01-01T00:00:00Z',
      updated_at: '2026-01-01T00:00:00Z',
    },
  ]);
  apiMock.createTeam.mockResolvedValue({});
  apiMock.updateTeam.mockResolvedValue({});
  apiMock.getTeamMembers.mockResolvedValue([{ user_id: 1 }, { user_id: 2 }, { user_id: 3 }]);
  apiMock.deleteTeam.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('TeamsPage edit and delete flows', () => {
  it('updates and deletes a team with member-count acknowledgment', async () => {
    const user = userEvent.setup();
    render(<TeamsPage />);

    const rowLabel = await screen.findByText('Team One');
    const row = rowLabel.closest('tr');
    expect(row).not.toBeNull();

    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Edit' }));

    const teamNameInput = screen.getByLabelText('Team Name');
    await user.clear(teamNameInput);
    await user.type(teamNameInput, 'Platform Team');
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(apiMock.updateTeam).toHaveBeenCalledWith('10', '5', {
        name: 'Platform Team',
        description: 'Team one description',
      });
    });

    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(confirmMock).toHaveBeenCalledWith(
        expect.objectContaining({
          title: 'Delete team',
          message: expect.stringContaining('This team has 3 members'),
        })
      );
    });

    await waitFor(() => {
      expect(apiMock.deleteTeam).toHaveBeenCalledWith('10', '5');
    });
  });

  it('bulk delete uses privileged action dialog and passes real org_id', async () => {
    const user = userEvent.setup();
    render(<TeamsPage />);

    await screen.findByText('Team One');

    const checkboxes = screen.getAllByRole('checkbox');
    const teamCheckbox = checkboxes.find(
      (cb) => cb.closest('tr')?.textContent?.includes('Team One'),
    );
    expect(teamCheckbox).toBeDefined();
    await user.click(teamCheckbox!);

    const deleteButtons = await screen.findAllByRole('button', { name: /delete/i });
    const bulkDeleteBtn = deleteButtons.find((btn) => btn.textContent?.match(/delete/i) && !btn.closest('tr'));
    expect(bulkDeleteBtn).toBeDefined();
    await user.click(bulkDeleteBtn!);

    await waitFor(() => {
      expect(privilegedActionMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Delete selected teams' }),
      );
    });

    await waitFor(() => {
      expect(apiMock.deleteTeam).toHaveBeenCalledWith('10', '5');
    });
  });

  it('keeps the all-organizations sentinel stable and aggregates teams across orgs', async () => {
    currentSelectedOrgId = '__all__';
    apiMock.getOrganizations.mockResolvedValue([
      {
        id: 10,
        name: 'Acme',
        slug: 'acme',
      },
      {
        id: 12,
        name: 'Beta',
        slug: 'beta',
      },
    ]);
    apiMock.getTeams.mockImplementation(async (orgId: string) => {
      if (orgId === '10') {
        return [
          {
            id: 5,
            org_id: 10,
            name: 'Team One',
            description: 'Team one description',
            created_at: '2026-01-01T00:00:00Z',
            updated_at: '2026-01-01T00:00:00Z',
          },
        ];
      }
      if (orgId === '12') {
        return [
          {
            id: 7,
            org_id: 12,
            name: 'Team Two',
            description: 'Team two description',
            created_at: '2026-01-02T00:00:00Z',
            updated_at: '2026-01-02T00:00:00Z',
          },
        ];
      }
      return [];
    });

    render(<TeamsPage />);

    await screen.findByText('Team One');
    expect(await screen.findByText('Team Two')).toBeInTheDocument();
    expect(screen.getByText('Teams across all organizations')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'New Team' })).toBeDisabled();
    expect(screen.getByPlaceholderText('Search teams by name or description...').getAttribute('disabled')).toBeNull();
    expect(apiMock.getTeams).toHaveBeenCalledWith('10');
    expect(apiMock.getTeams).toHaveBeenCalledWith('12');
  });

  it('uses the row org_id for edit and delete in the all-organizations view', async () => {
    currentSelectedOrgId = '__all__';
    apiMock.getOrganizations.mockResolvedValue([
      { id: 10, name: 'Acme', slug: 'acme' },
      { id: 42, name: 'Beta', slug: 'beta' },
    ]);
    apiMock.getTeams.mockImplementation(async (orgId: string) => {
      if (orgId === '42') {
        return [{
          id: 5,
          org_id: 42,
          name: 'Cross Org Team',
          description: 'Team from beta',
          created_at: '2026-01-01T00:00:00Z',
          updated_at: '2026-01-01T00:00:00Z',
        }];
      }
      return [];
    });

    const user = userEvent.setup();
    render(<TeamsPage />);

    const rowLabel = await screen.findByText('Cross Org Team');
    const row = rowLabel.closest('tr');
    expect(row).not.toBeNull();

    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Edit' }));
    await user.clear(screen.getByLabelText('Team Name'));
    await user.type(screen.getByLabelText('Team Name'), 'Cross Org Team Updated');
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(apiMock.updateTeam).toHaveBeenCalledWith('42', '5', {
        name: 'Cross Org Team Updated',
        description: 'Team from beta',
      });
    });

    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Delete' }));

    await waitFor(() => {
      expect(apiMock.deleteTeam).toHaveBeenCalledWith('42', '5');
    });
  });
});
