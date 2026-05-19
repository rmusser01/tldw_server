/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { cleanup, render, screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import TeamDetailPage from '../page';
import { api } from '@/lib/api-client';

const confirmMock = vi.hoisted(() => vi.fn());
const pushMock = vi.hoisted(() => vi.fn());
const loggerErrorMock = vi.hoisted(() => vi.fn());

vi.mock('next/link', () => ({
  default: ({ href, children }: { href: string; children: ReactNode }) => <a href={href}>{children}</a>,
}));

vi.mock('next/navigation', () => ({
  useParams: () => ({ id: '5' }),
  useRouter: () => ({
    push: pushMock,
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
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

vi.mock('@/lib/api-client', () => ({
  api: {
    getTeam: vi.fn(),
    getTeamMembers: vi.fn(),
    addTeamMember: vi.fn(),
    updateTeamMemberRole: vi.fn(),
    removeTeamMember: vi.fn(),
    updateTeam: vi.fn(),
    deleteTeam: vi.fn(),
  },
}));

vi.mock('@/lib/logger', () => ({
  logger: {
    error: loggerErrorMock,
  },
}));

type ApiMock = {
  getTeam: ReturnType<typeof vi.fn>;
  getTeamMembers: ReturnType<typeof vi.fn>;
  addTeamMember: ReturnType<typeof vi.fn>;
  updateTeamMemberRole: ReturnType<typeof vi.fn>;
  removeTeamMember: ReturnType<typeof vi.fn>;
  updateTeam: ReturnType<typeof vi.fn>;
  deleteTeam: ReturnType<typeof vi.fn>;
};

const apiMock = api as unknown as ApiMock;

beforeEach(() => {
  confirmMock.mockResolvedValue(true);
  pushMock.mockClear();
  loggerErrorMock.mockClear();

  apiMock.getTeam.mockResolvedValue({
    id: 5,
    org_id: 10,
    name: 'Team One',
    description: 'Demo team',
  });
  apiMock.getTeamMembers.mockResolvedValue([
    {
      user_id: 2,
      role: 'member',
      joined_at: '2026-01-01T00:00:00Z',
      user: {
        id: 2,
        username: 'Bob',
        email: 'bob@example.com',
      },
    },
  ]);
  apiMock.addTeamMember.mockResolvedValue({});
  apiMock.updateTeamMemberRole.mockResolvedValue({});
  apiMock.removeTeamMember.mockResolvedValue({});
  apiMock.updateTeam.mockResolvedValue({});
  apiMock.deleteTeam.mockResolvedValue({});
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('TeamDetailPage member role updates', () => {
  it('updates a team member role from the inline dropdown', async () => {
    const user = userEvent.setup();
    render(<TeamDetailPage />);

    const rowLabel = await screen.findByText('Bob');
    const row = rowLabel.closest('tr');
    expect(row).not.toBeNull();

    const roleSelect = within(row as HTMLElement).getByLabelText(/team role for bob/i);
    await user.selectOptions(roleSelect, 'admin');
    await user.click(within(row as HTMLElement).getByRole('button', { name: 'Save' }));

    await waitFor(() => {
      expect(apiMock.updateTeamMemberRole).toHaveBeenCalledWith('5', 2, { role: 'admin' });
    });
  });

  it('logs an error when updating the team details fails', async () => {
    apiMock.updateTeam.mockRejectedValueOnce(new Error('backend exploded'));
    const user = userEvent.setup();
    render(<TeamDetailPage />);

    await screen.findByText('Team One');
    await user.click(screen.getByRole('button', { name: 'Edit Team' }));
    await user.clear(screen.getByLabelText('Team Name'));
    await user.type(screen.getByLabelText('Team Name'), 'Platform Team');
    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(loggerErrorMock).toHaveBeenCalledWith(
        'Failed to update team',
        expect.objectContaining({
          component: 'TeamDetailPage',
          error: 'backend exploded',
        })
      );
    });
    expect(await screen.findByText('backend exploded')).toBeInTheDocument();
  });
});
