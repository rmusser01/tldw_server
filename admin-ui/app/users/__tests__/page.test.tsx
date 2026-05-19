/* @vitest-environment jsdom */
import type { ReactNode } from 'react';
import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { render, screen, waitFor, cleanup, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import UsersPage from '../page';
import { api } from '@/lib/api-client';
import { formatAxeViolations, getCriticalAndSeriousAxeViolations } from '@/test-utils/axe';
import { getScopedItem } from '@/lib/scoped-storage';

const confirmMock = vi.hoisted(() => vi.fn());
const privilegedActionMock = vi.hoisted(() => vi.fn());
const toastSuccessMock = vi.hoisted(() => vi.fn());
const toastErrorMock = vi.hoisted(() => vi.fn());

const permissionUser = vi.hoisted(() => ({
  id: 1,
  uuid: 'user-1',
  username: 'Alice',
  email: 'alice@example.com',
  role: 'admin',
  is_active: true,
  is_verified: true,
  mfa_enabled: true,
  storage_quota_mb: 100,
  storage_used_mb: 12,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
}));

vi.mock('@/components/PermissionGuard', () => ({
  PermissionGuard: ({ children }: { children: ReactNode }) => <>{children}</>,
  default: ({ children }: { children: ReactNode }) => <>{children}</>,
  usePermissions: () => ({
    user: permissionUser,
    permissions: [],
    permissionHints: [],
    roles: ['admin'],
    loading: false,
    hasPermission: () => true,
    hasRole: () => true,
    hasAnyPermission: () => true,
    hasAllPermissions: () => true,
    isAdmin: () => true,
    isSuperAdmin: () => false,
    refresh: async () => {},
  }),
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

vi.mock('next/navigation', () => ({
  useRouter: () => ({
    push: vi.fn(),
    replace: vi.fn(),
    prefetch: vi.fn(),
  }),
  usePathname: () => '/users',
  useSearchParams: () => new URLSearchParams(),
}));

vi.mock('@/components/OrgContextSwitcher', () => ({
  useOrgContext: () => ({ selectedOrg: null }),
  OrgContextSwitcher: () => <div data-testid="org-switcher" />,
}));

vi.mock('@/lib/use-url-state', async () => {
  const React = await import('react');
  return {
    useUrlState: (_key: string, options?: { defaultValue?: unknown }) => {
      const [value, setValue] = React.useState(options?.defaultValue);
      return [value, setValue];
    },
    useUrlPagination: () => {
      const [page, setPage] = React.useState(1);
      const [pageSize, setPageSize] = React.useState(25);
      return {
        page,
        pageSize,
        setPage,
        setPageSize,
        resetPagination: () => setPage(1),
      };
    },
  };
});

vi.mock('@/lib/api-client', () => ({
  api: {
    getUsers: vi.fn(),
    getOrganizations: vi.fn(),
    getOrgInvites: vi.fn(),
    getUserMfaStatusBulk: vi.fn(),
    deleteUser: vi.fn(),
    updateUser: vi.fn(),
    createUser: vi.fn(),
    resetUserPassword: vi.fn(),
    setUserMfaRequirement: vi.fn(),
    getUserMfaStatus: vi.fn(),
    inviteUser: vi.fn(),
    getInvitations: vi.fn(),
    revokeInvitation: vi.fn(),
    resendInvitation: vi.fn(),
  },
}));

const apiMock = vi.mocked(api);

const makeUser = (overrides: Partial<Record<string, unknown>> = {}) => ({
  id: 1,
  uuid: 'user-1',
  username: 'Alice',
  email: 'alice@example.com',
  role: 'admin',
  is_active: true,
  is_verified: true,
  mfa_enabled: false,
  storage_quota_mb: 100,
  storage_used_mb: 12,
  created_at: '2024-01-01T00:00:00Z',
  updated_at: '2024-01-01T00:00:00Z',
  ...overrides,
});

beforeEach(() => {
  confirmMock.mockResolvedValue(true);
  privilegedActionMock.mockResolvedValue({
    reason: 'Customer requested this change',
    adminPassword: 'AdminPass123!',
  });
  toastSuccessMock.mockClear();
  toastErrorMock.mockClear();

  apiMock.getUsers.mockResolvedValue([
    makeUser({ id: 1, username: 'Alice', email: 'alice@example.com', role: 'admin', is_active: true, is_verified: true, mfa_enabled: true }),
    makeUser({ id: 2, uuid: 'user-2', username: 'Bob', email: 'bob@example.com', role: 'user', is_active: false, is_verified: true, mfa_enabled: false }),
    makeUser({ id: 3, uuid: 'user-3', username: 'Carol', email: 'carol@example.com', role: 'service', is_active: true, is_verified: false, mfa_enabled: true }),
  ]);
  apiMock.getOrganizations.mockResolvedValue([
    { id: 11, name: 'Core Org' },
  ]);
  apiMock.getOrgInvites.mockResolvedValue({
    items: [
      {
        id: 1001,
        org_id: 11,
        org_name: 'Core Org',
        role_to_grant: 'member',
        email: 'pending@example.com',
        created_by: 1,
        created_at: '2026-02-17T10:00:00Z',
        expires_at: '2099-02-20T10:00:00Z',
        uses_count: 0,
      },
      {
        id: 1002,
        org_id: 11,
        org_name: 'Core Org',
        role_to_grant: 'admin',
        email: 'accepted@example.com',
        created_by: 1,
        created_at: '2026-02-16T10:00:00Z',
        expires_at: '2099-02-20T10:00:00Z',
        uses_count: 1,
      },
      {
        id: 1003,
        org_id: 11,
        org_name: 'Core Org',
        role_to_grant: 'member',
        email: 'expired@example.com',
        created_by: 1,
        created_at: '2026-01-01T10:00:00Z',
        expires_at: '2026-01-05T10:00:00Z',
        uses_count: 0,
      },
    ],
  });
  apiMock.getUserMfaStatus.mockImplementation(async (userId: string) => {
    if (userId === '2') return { enabled: false };
    if (userId === '3') return { enabled: true };
    return { enabled: true };
  });
  apiMock.getUserMfaStatusBulk.mockResolvedValue({
    mfa_status: { '1': true, '2': false, '3': true },
    failed_user_ids: [],
  });
  apiMock.deleteUser.mockResolvedValue({});
  apiMock.updateUser.mockResolvedValue({});
  apiMock.createUser.mockResolvedValue({});
  apiMock.resetUserPassword.mockResolvedValue({});
  apiMock.setUserMfaRequirement.mockResolvedValue({});
  apiMock.getUserMfaStatus.mockResolvedValue({ enabled: false });
  apiMock.getInvitations.mockResolvedValue({
    items: [
      {
        id: 'inv-001',
        email: 'invited@example.com',
        role: 'user',
        status: 'pending',
        invited_by: 'Alice',
        created_at: '2026-03-20T10:00:00Z',
        expires_at: '2026-03-27T10:00:00Z',
        email_sent: true,
        email_error: null,
      },
    ],
    total: 1,
  });
  apiMock.inviteUser.mockResolvedValue({
    id: 'inv-new',
    email: 'new@example.com',
    role: 'user',
    status: 'pending',
    email_sent: true,
    email_error: null,
  });
  apiMock.revokeInvitation.mockResolvedValue({
    id: 'inv-001',
    status: 'revoked',
  });
  apiMock.resendInvitation.mockResolvedValue({
    id: 'inv-001',
    email: 'invited@example.com',
    role: 'user',
    status: 'pending',
    email_sent: true,
    email_error: null,
    resend_count: 1,
    last_resent_at: '2026-03-27T10:00:00Z',
  });
});

afterEach(() => {
  cleanup();
  vi.resetAllMocks();
});

describe('UsersPage', () => {
  it('has no critical/serious axe violations in the default state', async () => {
    const { container } = render(<UsersPage />);

    await screen.findByText('Organization Invitations');
    const violations = await getCriticalAndSeriousAxeViolations(container);
    expect(violations, formatAxeViolations(violations)).toEqual([]);
  });

  it('renders invitation funnel metrics and mixed invitation statuses', async () => {
    render(<UsersPage />);

    await screen.findByText('Organization Invitations');
    expect(screen.getByTestId('invitation-total-sent').textContent).toContain('3');
    expect(screen.getByTestId('invitation-total-accepted').textContent).toContain('1');
    expect(screen.getByTestId('invitation-conversion-rate').textContent).toContain('33.3%');

    expect(screen.getByText('pending@example.com')).toBeInTheDocument();
    expect(screen.getByText('accepted@example.com')).toBeInTheDocument();
    expect(screen.getByText('expired@example.com')).toBeInTheDocument();
    expect(screen.getByText('sent')).toBeInTheDocument();
    expect(screen.getByText('accepted')).toBeInTheDocument();
    expect(screen.getByText('expired')).toBeInTheDocument();
  });

  it('disables selection and deletion for the current user', async () => {
    render(<UsersPage />);

    const checkbox = await screen.findByRole('checkbox', {
      name: /select user alice/i,
    });
    const row = checkbox.closest('tr');
    expect(row).not.toBeNull();

    const currentUserCheckbox = within(row as HTMLElement).getByRole('checkbox', {
      name: /select user alice/i,
    });
    expect(currentUserCheckbox).toBeDisabled();

    const deleteButton = within(row as HTMLElement).getByTitle('Cannot delete yourself');
    expect(deleteButton).toBeDisabled();
  });

  it('does not show the dormant badge for users who have never logged in', async () => {
    render(<UsersPage />);

    await screen.findByText('Bob');
    expect(screen.queryAllByText('Dormant')).toHaveLength(0);
  });

  it('sends combined search and filter params to user API', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    await screen.findByText('Bob');
    await user.type(screen.getByLabelText(/search users by username, email, or role/i), 'bob');
    await user.selectOptions(screen.getByLabelText(/filter by user status/i), 'inactive');
    await user.selectOptions(screen.getByLabelText(/filter by verification state/i), 'verified');
    await user.selectOptions(screen.getByLabelText(/filter by mfa status/i), 'disabled');

    await waitFor(() => {
      expect(apiMock.getUsers).toHaveBeenLastCalledWith(
        expect.objectContaining({
          limit: '200',
          search: 'bob',
          is_active: 'false',
          is_verified: 'true',
          mfa_enabled: 'false',
        })
      );
    });
  });

  it('applies search and filters together with AND logic', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    await screen.findByText('Bob');
    await user.type(screen.getByLabelText(/search users by username, email, or role/i), 'bob');
    await user.selectOptions(screen.getByLabelText(/filter by user status/i), 'inactive');
    await user.selectOptions(screen.getByLabelText(/filter by verification state/i), 'verified');
    await user.selectOptions(screen.getByLabelText(/filter by mfa status/i), 'disabled');

    expect(await screen.findByText('Bob')).toBeInTheDocument();
    await waitFor(() => {
      expect(screen.queryByRole('checkbox', { name: /select user alice/i })).not.toBeInTheDocument();
      expect(screen.queryByRole('checkbox', { name: /select user carol/i })).not.toBeInTheDocument();
    });
  });

  it('supports bulk role assignment from the selected role dropdown', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    await screen.findByText('Bob');
    await user.click(screen.getByRole('checkbox', { name: /select user bob/i }));
    await user.selectOptions(screen.getByLabelText(/bulk role selection/i), 'admin');
    await user.click(screen.getByRole('button', { name: 'Assign Role' }));

    await waitFor(() => {
      expect(privilegedActionMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Assign role to selected users' })
      );
    });
    await waitFor(() => {
      expect(apiMock.updateUser).toHaveBeenCalledWith('2', {
        role: 'admin',
        reason: 'Customer requested this change',
        admin_password: 'AdminPass123!',
      });
    });
  });

  it('does not offer bulk password reset from the list view', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    await screen.findByText('Bob');
    await user.click(screen.getByRole('checkbox', { name: /select user bob/i }));

    expect(screen.queryByRole('button', { name: 'Reset Passwords' })).not.toBeInTheDocument();
  });

  it('supports bulk MFA requirement updates for selected users', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    await screen.findByText('Bob');
    await user.click(screen.getByRole('checkbox', { name: /select user bob/i }));
    await user.click(screen.getByRole('button', { name: 'Require MFA' }));

    await waitFor(() => {
      expect(privilegedActionMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Require MFA for selected users' })
      );
    });
    await waitFor(() => {
      expect(apiMock.setUserMfaRequirement).toHaveBeenCalledWith('2', {
        require_mfa: true,
        reason: 'Customer requested this change',
        admin_password: 'AdminPass123!',
      });
    });
    await waitFor(() => {
      expect(apiMock.getUsers).toHaveBeenCalledTimes(2);
    });
  });

  it('round-trips a saved view through save, apply, and delete', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    const searchInput = screen.getByLabelText(/search users by username, email, or role/i) as HTMLInputElement;

    await screen.findByText('Bob');
    await user.type(searchInput, 'bob');
    await user.click(screen.getByRole('button', { name: 'Save view' }));
    await user.type(screen.getByLabelText('View name'), 'Bob only');
    await user.click(within(screen.getByRole('dialog')).getByRole('button', { name: 'Save view' }));

    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalledWith('Saved view', 'Bob only has been added.');
    });
    expect(getScopedItem('admin_users_saved_views')).toContain('Bob only');

    await user.clear(searchInput);
    await waitFor(() => {
      expect(searchInput.value).toBe('');
    });

    await user.selectOptions(
      screen.getByLabelText('Saved views'),
      await screen.findByRole('option', { name: 'Bob only' })
    );

    await waitFor(() => {
      expect(searchInput.value).toBe('bob');
    });
    expect(await screen.findByText('Bob')).toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Delete view' }));

    await waitFor(() => {
      expect(confirmMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Delete saved view' })
      );
    });
    await waitFor(() => {
      expect(getScopedItem('admin_users_saved_views') ?? '').not.toContain('Bob only');
    });
  });

  it('renders the Pending Invitations section with direct invitations', async () => {
    render(<UsersPage />);

    await screen.findByText('Pending Invitations');
    expect(screen.getByText('invited@example.com')).toBeInTheDocument();
    expect(screen.getByTestId('direct-invitation-row-inv-001')).toBeInTheDocument();
  });

  it('shows the Invite User button and opens the invite dialog', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    const inviteButton = await screen.findByRole('button', { name: /invite user/i });
    expect(inviteButton).toBeInTheDocument();

    await user.click(inviteButton);
    expect(await screen.findByText('Invite user')).toBeInTheDocument();
    expect(screen.getByLabelText('Email address')).toBeInTheDocument();
    expect(screen.getByLabelText('Role')).toBeInTheDocument();
  });

  it('sends an invitation when the invite form is submitted', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    await user.click(await screen.findByRole('button', { name: /invite user/i }));
    await user.type(screen.getByLabelText('Email address'), 'new@example.com');
    await user.click(screen.getByRole('button', { name: /send invitation/i }));

    await waitFor(() => {
      expect(apiMock.inviteUser).toHaveBeenCalledWith({
        email: 'new@example.com',
        role: 'user',
      });
    });
    await waitFor(() => {
      expect(toastSuccessMock).toHaveBeenCalled();
    });
  });

  it('revokes an invitation when the revoke button is clicked', async () => {
    const user = userEvent.setup();
    render(<UsersPage />);

    await screen.findByText('Pending Invitations');
    const row = screen.getByTestId('direct-invitation-row-inv-001');
    const revokeButton = within(row).getByTitle('Revoke invitation');

    await user.click(revokeButton);

    await waitFor(() => {
      expect(confirmMock).toHaveBeenCalledWith(
        expect.objectContaining({ title: 'Revoke invitation' })
      );
    });
    await waitFor(() => {
      expect(apiMock.revokeInvitation).toHaveBeenCalledWith('inv-001');
    });
  });

  it('shows empty state when no direct invitations exist', async () => {
    apiMock.getInvitations.mockResolvedValue({ items: [], total: 0 });
    render(<UsersPage />);

    await screen.findByText('Pending Invitations');
    expect(screen.getByText('No invitations')).toBeInTheDocument();
  });

  it('renders a resend button on pending invitation rows', async () => {
    apiMock.getInvitations.mockResolvedValue({
      items: [
        {
          id: 'inv-resend',
          email: 'resendme@example.com',
          role: 'user',
          status: 'pending',
          invited_by: 'Alice',
          created_at: '2026-03-20T10:00:00Z',
          expires_at: '2099-03-27T10:00:00Z',
          email_sent: true,
          email_error: null,
          resend_count: 0,
        },
      ],
      total: 1,
    });
    render(<UsersPage />);

    const row = await screen.findByTestId('direct-invitation-row-inv-resend');
    const resendBtn = within(row).getByLabelText('Resend invitation');
    expect(resendBtn).toBeInTheDocument();
    expect(resendBtn).not.toBeDisabled();
  });

  it('disables resend button when resend_count >= 3', async () => {
    apiMock.getInvitations.mockResolvedValue({
      items: [
        {
          id: 'inv-maxed',
          email: 'maxed@example.com',
          role: 'user',
          status: 'pending',
          invited_by: 'Alice',
          created_at: '2026-03-20T10:00:00Z',
          expires_at: '2099-03-27T10:00:00Z',
          email_sent: true,
          email_error: null,
          resend_count: 3,
        },
      ],
      total: 1,
    });
    render(<UsersPage />);

    const row = await screen.findByTestId('direct-invitation-row-inv-maxed');
    const resendBtn = within(row).getByLabelText('Resend invitation');
    expect(resendBtn).toBeDisabled();
  });

  it('calls resendInvitation API when resend button is clicked', async () => {
    const user = userEvent.setup();
    apiMock.getInvitations.mockResolvedValue({
      items: [
        {
          id: 'inv-click',
          email: 'click@example.com',
          role: 'user',
          status: 'pending',
          invited_by: 'Alice',
          created_at: '2026-03-20T10:00:00Z',
          expires_at: '2099-03-27T10:00:00Z',
          email_sent: false,
          email_error: 'delivery failed',
          resend_count: 1,
        },
      ],
      total: 1,
    });
    apiMock.resendInvitation.mockResolvedValue({
      id: 'inv-click',
      email: 'click@example.com',
      role: 'user',
      status: 'pending',
      email_sent: true,
      email_error: null,
      resend_count: 2,
      last_resent_at: '2026-03-27T12:00:00Z',
    });
    render(<UsersPage />);

    const row = await screen.findByTestId('direct-invitation-row-inv-click');
    const resendBtn = within(row).getByLabelText('Resend invitation');
    await user.click(resendBtn);

    await waitFor(() => {
      expect(apiMock.resendInvitation).toHaveBeenCalledWith('inv-click');
    });
  });
});
