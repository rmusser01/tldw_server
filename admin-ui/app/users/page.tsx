'use client';

import { useCallback, useEffect, useMemo, useState, Suspense } from 'react';
import { useRouter } from 'next/navigation';
import { useForm, FormProvider } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { PermissionGuard, usePermissions } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/empty-state';
import { Pagination } from '@/components/ui/pagination';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Checkbox } from '@/components/ui/checkbox';
import { Form, FormCheckbox, FormInput, FormSelect } from '@/components/ui/form';
import {
  Eye,
  Key,
  Mail,
  RefreshCw,
  Search,
  Plus,
  Trash2,
  UserCheck,
  UserX,
  XCircle,
  BookmarkPlus,
  BookmarkX,
} from 'lucide-react';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import { api } from '@/lib/api-client';
import { isSingleUserMode } from '@/lib/auth';
import { Organization, User } from '@/types';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportUsers, ExportFormat } from '@/lib/export';
import { Skeleton, TableSkeleton } from '@/components/ui/skeleton';
import { useUrlPagination } from '@/lib/use-url-state';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import { useOrgContext } from '@/components/OrgContextSwitcher';
import { useResourceState } from '@/lib/use-resource-state';
import { UserBulkActions } from './components/UserBulkActions';
import {
  useUserFilters,
  type UserMfaFilter,
  type UserStatusFilter,
  type UserVerifiedFilter,
} from './hooks/use-user-filters';
import { logger } from '@/lib/logger';

type BulkActionType =
  | 'activate'
  | 'deactivate'
  | 'delete'
  | 'assign-role'
  | 'mfa-require'
  | 'mfa-clear'
  | null;

const createUserSchema = z.object({
  username: z.string().min(1, 'Username is required'),
  email: z.string().email('Enter a valid email address'),
  password: z.string().min(10, 'Password must be at least 10 characters'),
  role: z.enum(['user', 'admin', 'service']),
  is_active: z.boolean(),
  is_verified: z.boolean(),
});

type CreateUserFormData = z.infer<typeof createUserSchema>;

type InvitationStatus = 'sent' | 'accepted' | 'expired';

type OrgInviteRecord = Record<string, unknown>;

type InvitationRow = {
  id: string;
  status: InvitationStatus;
  email: string;
  invitedBy: string;
  role: string;
  org: string;
  sentAt: string | null;
  expiresAt: string | null;
};

const toRecord = (value: unknown): Record<string, unknown> | null =>
  value && typeof value === 'object'
    ? (value as Record<string, unknown>)
    : null;

const pickString = (...values: unknown[]): string | null => {
  for (const value of values) {
    if (typeof value === 'string' && value.trim()) {
      return value.trim();
    }
  }
  return null;
};

const pickNumber = (...values: unknown[]): number | null => {
  for (const value of values) {
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
    if (typeof value === 'string' && value.trim()) {
      const parsed = Number.parseInt(value.trim(), 10);
      if (Number.isFinite(parsed)) return parsed;
    }
  }
  return null;
};

const parseOrganizationsResponse = (value: unknown): Organization[] => {
  if (Array.isArray(value)) {
    return value as Organization[];
  }
  const payload = toRecord(value);
  if (payload && Array.isArray(payload.items)) {
    return payload.items as Organization[];
  }
  return [];
};

const parseOrgInvitesResponse = (value: unknown): OrgInviteRecord[] => {
  if (Array.isArray(value)) {
    return value.filter((item) => toRecord(item) !== null) as OrgInviteRecord[];
  }
  const payload = toRecord(value);
  if (payload && Array.isArray(payload.items)) {
    return payload.items.filter((item) => toRecord(item) !== null) as OrgInviteRecord[];
  }
  return [];
};

const resolveInvitationStatus = (invite: OrgInviteRecord, nowMs: number = Date.now()): InvitationStatus => {
  const expiresAtRaw = pickString(invite.expires_at);
  const expiresAtMs = expiresAtRaw ? Date.parse(expiresAtRaw) : Number.NaN;
  if (Number.isFinite(expiresAtMs) && expiresAtMs < nowMs) {
    return 'expired';
  }
  const usesCount = pickNumber(invite.uses_count) ?? 0;
  if (usesCount > 0) {
    return 'accepted';
  }
  return 'sent';
};

const invitationStatusBadgeVariant = (status: InvitationStatus): 'default' | 'secondary' | 'destructive' => {
  if (status === 'accepted') return 'default';
  if (status === 'expired') return 'destructive';
  return 'secondary';
};

function UsersPageContent() {
  const router = useRouter();
  const confirm = useConfirm();
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();
  const { selectedOrg } = useOrgContext();
  const { user: currentUser } = usePermissions();
  const requirePasswordReauth = !isSingleUserMode();
  const currentUserId = currentUser?.id;
  const [bulkAction, setBulkAction] = useState<BulkActionType>(null);
  const [selectedUserIds, setSelectedUserIds] = useState<Set<number>>(new Set());
  const [bulkRole, setBulkRole] = useState('user');
  const [showCreateUserDialog, setShowCreateUserDialog] = useState(false);
  const [createUserError, setCreateUserError] = useState('');
  const [creatingUser, setCreatingUser] = useState(false);
  const [deletingUserIds, setDeletingUserIds] = useState<Set<number>>(new Set());
  const [showInviteUserDialog, setShowInviteUserDialog] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState('user');
  const [invitingUser, setInvitingUser] = useState(false);
  const [inviteUserError, setInviteUserError] = useState('');
  type DirectInvitation = {
    id: string;
    email: string;
    role: string;
    status: string;
    invited_by: string | null;
    created_at: string | null;
    expires_at: string | null;
    email_sent: boolean;
    email_error: string | null;
    resend_count?: number;
    last_resent_at?: string | null;
  };
  const [directInvitations, setDirectInvitations] = useState<DirectInvitation[]>([]);
  const [directInvitesLoading, setDirectInvitesLoading] = useState(true);
  const [directInvitesError, setDirectInvitesError] = useState('');
  const [revokingInviteIds, setRevokingInviteIds] = useState<Set<string>>(new Set());
  const [resendingInviteIds, setResendingInviteIds] = useState<Set<string>>(new Set());
  const [mfaByUserId, setMfaByUserId] = useState<Record<number, boolean | null>>({});
  const [mfaLoading, setMfaLoading] = useState(false);
  const [orgInvites, setOrgInvites] = useState<OrgInviteRecord[]>([]);
  const [invitesLoading, setInvitesLoading] = useState(true);
  const [invitesError, setInvitesError] = useState('');
  const createUserForm = useForm<CreateUserFormData>({
    resolver: zodResolver(createUserSchema),
    defaultValues: {
      username: '',
      email: '',
      password: '',
      role: 'user',
      is_active: true,
      is_verified: true,
    },
  });

  // URL state for search + filters
  // URL state for pagination
  const { page: currentPage, pageSize, setPage: setCurrentPage, setPageSize, resetPagination } = useUrlPagination();
  const {
    savedViews,
    showSaveViewDialog,
    saveViewName,
    saveViewError,
    searchQuery,
    statusFilter,
    verifiedFilter,
    mfaFilter,
    activeViewId,
    hasActiveFilters,
    setShowSaveViewDialog,
    setSaveViewName,
    clearSaveViewForm,
    handleSearchChange,
    handleStatusFilterChange,
    handleVerifiedFilterChange,
    handleMfaFilterChange,
    handleClearFilters,
    handleApplySavedView,
    saveCurrentView,
    removeSavedView,
  } = useUserFilters({ resetPagination });

  useEffect(() => {
    if (!showCreateUserDialog) {
      createUserForm.reset();
      setCreateUserError('');
    }
  }, [createUserForm, showCreateUserDialog]);

  useEffect(() => {
    if (!showInviteUserDialog) {
      setInviteEmail('');
      setInviteRole('user');
      setInviteUserError('');
    }
  }, [showInviteUserDialog]);

  const loadDirectInvitations = useCallback(async () => {
    try {
      setDirectInvitesLoading(true);
      setDirectInvitesError('');
      const response = await api.getInvitations() as { items?: DirectInvitation[] };
      setDirectInvitations(Array.isArray(response?.items) ? response.items : []);
    } catch (err: unknown) {
      logger.error('Failed to load direct invitations', { component: 'UsersPage', error: err instanceof Error ? err.message : String(err) });
      setDirectInvitesError(err instanceof Error ? err.message : 'Failed to load invitations');
      setDirectInvitations([]);
    } finally {
      setDirectInvitesLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadDirectInvitations();
  }, [loadDirectInvitations]);

  const handleInviteUser = async () => {
    setInviteUserError('');
    if (!inviteEmail.trim()) {
      setInviteUserError('Email is required');
      return;
    }
    try {
      setInvitingUser(true);
      const result = await api.inviteUser({
        email: inviteEmail.trim(),
        role: inviteRole,
      }) as { email_sent?: boolean; email_error?: string | null };
      if (result.email_sent) {
        success('Invitation sent', `Invite email sent to ${inviteEmail.trim()}.`);
      } else {
        success(
          'Invitation created',
          `Invitation created for ${inviteEmail.trim()}, but email could not be sent. ${result.email_error || 'Check email configuration.'}`,
        );
      }
      setShowInviteUserDialog(false);
      void loadDirectInvitations();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to send invitation';
      setInviteUserError(message);
      showError('Invite failed', message);
    } finally {
      setInvitingUser(false);
    }
  };

  const handleRevokeInvitation = async (invitationId: string) => {
    const confirmed = await confirm({
      title: 'Revoke invitation',
      message: 'Revoke this pending invitation? The invite link will no longer work.',
      confirmText: 'Revoke',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setRevokingInviteIds((prev) => new Set(prev).add(invitationId));
      await api.revokeInvitation(invitationId);
      success('Invitation revoked', 'The invitation has been revoked.');
      void loadDirectInvitations();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to revoke invitation';
      showError('Revoke failed', message);
    } finally {
      setRevokingInviteIds((prev) => {
        const next = new Set(prev);
        next.delete(invitationId);
        return next;
      });
    }
  };

  const handleResendInvitation = async (inv: DirectInvitation) => {
    if ((inv.resend_count ?? 0) >= 3) {
      showError('Resend limit', 'This invitation has reached the maximum resend limit (3).');
      return;
    }
    try {
      setResendingInviteIds((prev) => new Set(prev).add(inv.id));
      const result = await api.resendInvitation(inv.id) as DirectInvitation;
      if (result.email_sent) {
        success('Invitation resent', `Invite email resent to ${inv.email}.`);
      } else {
        success(
          'Invitation renewed',
          `Token renewed for ${inv.email}, but email could not be sent. ${result.email_error || 'Check email configuration.'}`,
        );
      }
      void loadDirectInvitations();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to resend invitation';
      showError('Resend failed', message);
    } finally {
      setResendingInviteIds((prev) => {
        const next = new Set(prev);
        next.delete(inv.id);
        return next;
      });
    }
  };

  const loadUsersResource = useCallback(async () => {
    const params: Record<string, string> = { limit: '200' };
    if (selectedOrg) params.org_id = String(selectedOrg.id);
    if (searchQuery) params.search = searchQuery;
    if (statusFilter === 'active') params.is_active = 'true';
    if (statusFilter === 'inactive') params.is_active = 'false';
    if (verifiedFilter === 'verified') params.is_verified = 'true';
    if (verifiedFilter === 'unverified') params.is_verified = 'false';
    if (mfaFilter === 'enabled') params.mfa_enabled = 'true';
    if (mfaFilter === 'disabled') params.mfa_enabled = 'false';
    try {
      return await api.getUsers(params);
    } catch (err: unknown) {
      logger.error('Failed to load users', { component: 'UsersPage', error: err instanceof Error ? err.message : String(err) });
      throw err instanceof Error
        ? err
        : new Error('Failed to load users');
    }
  }, [mfaFilter, searchQuery, selectedOrg, statusFilter, verifiedFilter]);

  const {
    value: users,
    loading,
    error,
    reload: loadUsers,
  } = useResourceState<User[]>({
    load: loadUsersResource,
    deps: [selectedOrg?.id, searchQuery, statusFilter, verifiedFilter, mfaFilter],
    initialValue: [],
    defaultError: 'Failed to load users',
    resetOnError: true,
  });

  const loadInvitations = useCallback(async () => {
    try {
      setInvitesLoading(true);
      setInvitesError('');

      const organizations = selectedOrg
        ? [{ id: selectedOrg.id, name: selectedOrg.name || `Org #${selectedOrg.id}` }] as Array<Pick<Organization, 'id' | 'name'>>
        : parseOrganizationsResponse(await api.getOrganizations({ limit: '200' }));

      if (organizations.length === 0) {
        setOrgInvites([]);
        return;
      }

      const inviteResults = await Promise.allSettled(
        organizations.map((org) =>
          api.getOrgInvites(String(org.id), {
            include_expired: 'true',
            include_inactive: 'true',
            limit: '100',
          }),
        ),
      );

      const invites: OrgInviteRecord[] = [];
      let successCount = 0;
      inviteResults.forEach((result, index) => {
        if (result.status !== 'fulfilled') return;
        successCount += 1;
        const org = organizations[index];
        parseOrgInvitesResponse(result.value).forEach((invite) => {
          invites.push({
            ...invite,
            org_id: pickNumber(invite.org_id) ?? org.id,
            org_name: pickString(invite.org_name) ?? org.name,
          });
        });
      });

      if (successCount === 0) {
        const firstError = inviteResults.find((result) => result.status === 'rejected');
        throw (firstError && firstError.status === 'rejected') ? firstError.reason : new Error('Failed to load invites');
      }
      if (successCount < inviteResults.length) {
        setInvitesError('Some organization invitations could not be loaded.');
      }

      setOrgInvites(invites);
    } catch (err: unknown) {
      logger.error('Failed to load invitations', { component: 'UsersPage', error: err instanceof Error ? err.message : String(err) });
      setInvitesError(err instanceof Error && err.message ? err.message : 'Failed to load invitations');
      setOrgInvites([]);
    } finally {
      setInvitesLoading(false);
    }
  }, [selectedOrg]);

  useEffect(() => {
    void loadInvitations();
  }, [loadInvitations]);

  useEffect(() => {
    setSelectedUserIds((prev) => {
      if (prev.size === 0) return prev;
      const available = new Set(users.map((user) => user.id));
      const next = new Set<number>();
      prev.forEach((id) => {
        if (available.has(id) && id !== currentUserId) {
          next.add(id);
        }
      });
      return next;
    });
  }, [currentUserId, users]);

  useEffect(() => {
    if (mfaFilter === 'all') return;
    const missingIds = users
      .map((user) => user.id)
      .filter((id) => mfaByUserId[id] === undefined);
    if (missingIds.length === 0) return;

    let cancelled = false;
    const loadMfaStatus = async () => {
      try {
        setMfaLoading(true);
        // Use bulk endpoint instead of N+1 individual calls
        const bulkResult = await api.getUserMfaStatusBulk(missingIds);
        if (cancelled) return;
        setMfaByUserId((prev) => {
          const responseEntries = Object.entries(bulkResult.mfa_status ?? {});
          const returnedIds = new Set<number>();
          const failedIds = new Set((bulkResult.failed_user_ids ?? []).map((id) => Number(id)));
          let changed = false;
          const next = { ...prev };
          for (const [uid, enabled] of responseEntries) {
            const userId = Number(uid);
            if (!Number.isFinite(userId)) continue;
            returnedIds.add(userId);
            if (next[userId] === enabled) continue;
            next[userId] = enabled;
            changed = true;
          }
          for (const userId of missingIds) {
            if (returnedIds.has(userId)) continue;
            if (!failedIds.has(userId) && bulkResult.failed_user_ids && bulkResult.failed_user_ids.length > 0) {
              continue;
            }
            if (next[userId] !== null) {
              next[userId] = null;
              changed = true;
            }
          }
          return changed ? next : prev;
        });
      } catch (err) {
        console.error('Failed to load MFA status for users:', err);
      } finally {
        if (!cancelled) {
          setMfaLoading(false);
        }
      }
    };
    void loadMfaStatus();

    return () => {
      cancelled = true;
    };
  }, [mfaByUserId, mfaFilter, users]);


  const filteredUsers = users.filter((user) => {
    const query = (searchQuery || '').toLowerCase();
    if (
      query
      && !(
        user.username?.toLowerCase().includes(query)
        || user.email?.toLowerCase().includes(query)
        || user.role?.toLowerCase().includes(query)
      )
    ) {
      return false;
    }

    if (statusFilter === 'active' && !user.is_active) return false;
    if (statusFilter === 'inactive' && user.is_active) return false;

    if (verifiedFilter === 'verified' && !user.is_verified) return false;
    if (verifiedFilter === 'unverified' && user.is_verified) return false;

    if (mfaFilter !== 'all') {
      const hasMfa = mfaByUserId[user.id];
      if (hasMfa === undefined || hasMfa === null) return false;
      if (mfaFilter === 'enabled' && hasMfa !== true) return false;
      if (mfaFilter === 'disabled' && hasMfa !== false) return false;
    }

    return true;
  });

  const userDisplayById = useMemo(() => {
    const mapping = new Map<number, string>();
    users.forEach((user) => {
      mapping.set(user.id, user.username || user.email || `User #${user.id}`);
    });
    return mapping;
  }, [users]);

  const invitationRows = useMemo<InvitationRow[]>(() => {
    const nowMs = Date.now();
    return orgInvites
      .map((invite, index) => {
        const status = resolveInvitationStatus(invite, nowMs);
        const createdBy = pickNumber(invite.created_by);
        const orgId = pickNumber(invite.org_id);
        const allowedEmailDomain = pickString(invite.allowed_email_domain);
        const email = pickString(invite.email, invite.invited_email, invite.invitee_email)
          ?? (allowedEmailDomain
            ? `Any ${allowedEmailDomain.startsWith('@') ? allowedEmailDomain : `@${allowedEmailDomain}`}`
            : '—');

        return {
          id: `${orgId ?? 'org'}-${pickNumber(invite.id) ?? index}`,
          status,
          email,
          invitedBy: pickString(invite.invited_by, invite.created_by_name)
            ?? (createdBy !== null ? userDisplayById.get(createdBy) ?? `User #${createdBy}` : '—'),
          role: pickString(invite.role_to_grant, invite.role) ?? 'member',
          org: pickString(invite.org_name) ?? (orgId !== null ? `Org #${orgId}` : '—'),
          sentAt: pickString(invite.created_at, invite.sent_at),
          expiresAt: pickString(invite.expires_at),
        };
      })
      .sort((a, b) => {
        const aMs = a.sentAt ? Date.parse(a.sentAt) : 0;
        const bMs = b.sentAt ? Date.parse(b.sentAt) : 0;
        return bMs - aMs;
      });
  }, [orgInvites, userDisplayById]);

  const invitationFunnel = useMemo(() => {
    const totalSent = invitationRows.length;
    const totalAccepted = invitationRows.filter((row) => row.status === 'accepted').length;
    const totalPending = invitationRows.filter((row) => row.status === 'sent').length;
    const totalExpired = invitationRows.filter((row) => row.status === 'expired').length;
    const conversionRate = totalSent > 0 ? (totalAccepted / totalSent) * 100 : 0;
    return {
      totalSent,
      totalAccepted,
      totalPending,
      totalExpired,
      conversionRate,
    };
  }, [invitationRows]);

  // Pagination calculations
  const totalItems = filteredUsers.length;
  const totalPages = Math.ceil(totalItems / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const paginatedUsers = filteredUsers.slice(startIndex, startIndex + pageSize);
  const selectableUsers = currentUserId
    ? paginatedUsers.filter((user) => user.id !== currentUserId)
    : paginatedUsers;
  const allVisibleSelected = selectableUsers.length > 0
    && selectableUsers.every((user) => selectedUserIds.has(user.id));
  const selectedCount = selectedUserIds.size;
  const bulkBusy = bulkAction !== null;
  const bulkRoleOptions = useMemo(() => {
    const roleSet = new Set<string>(['user', 'admin', 'service']);
    users.forEach((user) => {
      if (typeof user.role === 'string' && user.role.trim()) {
        roleSet.add(user.role);
      }
    });
    return Array.from(roleSet);
  }, [users]);

  useEffect(() => {
    if (bulkRoleOptions.length === 0) return;
    if (!bulkRoleOptions.includes(bulkRole)) {
      setBulkRole(bulkRoleOptions[0]);
    }
  }, [bulkRole, bulkRoleOptions]);

  const handlePageChange = (page: number) => {
    setCurrentPage(page);
  };

  const handlePageSizeChange = (size: number) => {
    setPageSize(size);
    resetPagination();
  };

  const handleToggleSelectUser = (userId: number, checked: boolean) => {
    if (currentUserId && userId === currentUserId) return;
    setSelectedUserIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(userId);
      } else {
        next.delete(userId);
      }
      return next;
    });
  };

  const handleToggleSelectAllVisible = (checked: boolean) => {
    setSelectedUserIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        paginatedUsers.forEach((user) => {
          if (currentUserId && user.id === currentUserId) return;
          next.add(user.id);
        });
      } else {
        paginatedUsers.forEach((user) => {
          if (currentUserId && user.id === currentUserId) return;
          next.delete(user.id);
        });
      }
      return next;
    });
  };

  const handleClearSelection = () => {
    setSelectedUserIds(new Set());
  };

  const handleBulkToggleActive = async (nextState: boolean) => {
    const ids = Array.from(selectedUserIds);
    if (ids.length === 0) return;
    const approval = await promptPrivilegedAction({
      title: nextState ? 'Activate selected users' : 'Deactivate selected users',
      message: `${nextState ? 'Activate' : 'Deactivate'} ${ids.length} selected user${ids.length !== 1 ? 's' : ''}? Reauthentication is required.`,
      confirmText: nextState ? 'Activate' : 'Deactivate',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      setBulkAction(nextState ? 'activate' : 'deactivate');
      const results = await Promise.allSettled(
        ids.map((id) => api.updateUser(id.toString(), {
          is_active: nextState,
          reason: approval.reason,
          admin_password: approval.adminPassword,
        }))
      );
      const failures = results.filter((result) => result.status === 'rejected').length;
      if (failures > 0) {
        showError(
          'Bulk update incomplete',
          `${ids.length - failures} updated, ${failures} failed.`
        );
      } else {
        success(
          'Users updated',
          `${ids.length} user${ids.length !== 1 ? 's' : ''} ${nextState ? 'activated' : 'deactivated'}.`
        );
      }
      handleClearSelection();
      void loadUsers();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to update users';
      showError('Bulk update failed', message);
    } finally {
      setBulkAction(null);
    }
  };

  const handleBulkDelete = async () => {
    const ids = Array.from(selectedUserIds);
    if (ids.length === 0) return;
    if (currentUser && ids.includes(currentUser.id)) {
      showError('Cannot delete yourself', 'Remove your account from the selection to continue.');
      return;
    }
    const approval = await promptPrivilegedAction({
      title: 'Delete selected users',
      message: `Delete ${ids.length} selected user${ids.length !== 1 ? 's' : ''}? This cannot be undone.`,
      confirmText: 'Delete',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      setBulkAction('delete');
      const results = await Promise.allSettled(
        ids.map((id) => api.deleteUser(id.toString(), {
          reason: approval.reason,
          admin_password: approval.adminPassword,
        }))
      );
      const failures = results.filter((result) => result.status === 'rejected').length;
      if (failures > 0) {
        showError(
          'Bulk delete incomplete',
          `${ids.length - failures} deleted, ${failures} failed.`
        );
      } else {
        success(
          'Users deleted',
          `${ids.length} user${ids.length !== 1 ? 's' : ''} removed.`
        );
      }
      handleClearSelection();
      void loadUsers();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete users';
      showError('Bulk delete failed', message);
    } finally {
      setBulkAction(null);
    }
  };

  const handleBulkAssignRole = async () => {
    const ids = Array.from(selectedUserIds);
    if (ids.length === 0 || !bulkRole) return;
    const approval = await promptPrivilegedAction({
      title: 'Assign role to selected users',
      message: `Assign "${bulkRole}" role to ${ids.length} selected user${ids.length !== 1 ? 's' : ''}? Reauthentication is required.`,
      confirmText: 'Assign role',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      setBulkAction('assign-role');
      const results = await Promise.allSettled(
        ids.map((id) => api.updateUser(id.toString(), {
          role: bulkRole,
          reason: approval.reason,
          admin_password: approval.adminPassword,
        }))
      );
      const failures = results.filter((result) => result.status === 'rejected').length;
      if (failures > 0) {
        showError(
          'Role assignment incomplete',
          `${ids.length - failures} updated, ${failures} failed.`
        );
      } else {
        success(
          'Roles assigned',
          `${ids.length} user${ids.length !== 1 ? 's' : ''} updated to ${bulkRole}.`
        );
      }
      handleClearSelection();
      void loadUsers();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to assign role';
      showError('Role assignment failed', message);
    } finally {
      setBulkAction(null);
    }
  };

  const handleBulkSetMfaRequirement = async (requireMfa: boolean) => {
    const ids = Array.from(selectedUserIds);
    if (ids.length === 0) return;
    const approval = await promptPrivilegedAction({
      title: requireMfa ? 'Require MFA for selected users' : 'Clear MFA requirement for selected users',
      message: `${requireMfa ? 'Require MFA for' : 'Clear MFA requirement for'} ${ids.length} selected user${ids.length !== 1 ? 's' : ''}?`,
      confirmText: requireMfa ? 'Require MFA' : 'Clear requirement',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      setBulkAction(requireMfa ? 'mfa-require' : 'mfa-clear');
      const results = await Promise.allSettled(
        ids.map((id) => api.setUserMfaRequirement(id.toString(), {
          require_mfa: requireMfa,
          reason: approval.reason,
          admin_password: approval.adminPassword,
        }))
      );
      const failures = results.filter((result) => result.status === 'rejected').length;
      const successIds: number[] = [];
      results.forEach((result, index) => {
        if (result.status === 'fulfilled') {
          successIds.push(ids[index]);
        }
      });

      if (failures > 0) {
        showError(
          'Bulk MFA update incomplete',
          `${ids.length - failures} updated, ${failures} failed.`
        );
      } else {
        success(
          'MFA requirements updated',
          `${ids.length} user${ids.length !== 1 ? 's' : ''} ${requireMfa ? 'now require' : 'no longer require'} MFA.`
        );
      }
      handleClearSelection();
      if (successIds.length > 0) {
        void loadUsers();
      }
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to update MFA requirement';
      showError('Bulk MFA update failed', message);
    } finally {
      setBulkAction(null);
    }
  };

  const getRoleBadgeVariant = (role: string) => {
    switch (role) {
      case 'admin':
      case 'super_admin':
      case 'owner':
        return 'default';
      default:
        return 'secondary';
    }
  };

  const formatStorageUsage = (usedMb: number, quotaMb: number) => {
    const percentage = quotaMb > 0 ? (usedMb / quotaMb) * 100 : 0;
    return {
      text: `${usedMb.toFixed(1)} / ${quotaMb} MB`,
      percentage: Math.min(percentage, 100),
    };
  };

  const handleExport = (format: ExportFormat) => {
    exportUsers(filteredUsers, format);
  };

  const handleSaveView = () => {
    const result = saveCurrentView();
    if (!result.ok) return;
    success('Saved view', `${result.view.name} has been added.`);
  };

  const handleDeleteView = async () => {
    if (!activeViewId) return;
    const view = savedViews.find((item) => item.id === activeViewId);
    if (!view) return;
    const confirmed = await confirm({
      title: 'Delete saved view',
      message: `Delete "${view.name}"?`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;
    const removedView = removeSavedView(activeViewId);
    if (!removedView) return;
    success('Saved view removed', `"${removedView.name}" deleted.`);
  };

  const handleCreateUserSubmit = createUserForm.handleSubmit(async (data) => {
    setCreateUserError('');
    try {
      setCreatingUser(true);
      await api.createUser({
        username: data.username,
        email: data.email,
        password: data.password,
        role: data.role,
        is_active: data.is_active,
        is_verified: data.is_verified,
      });
      success('User created', `${data.username} added.`);
      setShowCreateUserDialog(false);
      createUserForm.reset();
      void loadUsers();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to create user';
      setCreateUserError(message);
      showError('Create user failed', message);
    } finally {
      setCreatingUser(false);
    }
  });

  const handleToggleActive = async (user: User) => {
    const nextState = !user.is_active;
    const approval = await promptPrivilegedAction({
      title: nextState ? 'Activate User' : 'Deactivate User',
      message: `${nextState ? 'Activate' : 'Deactivate'} ${user.username || user.email}? Reauthentication is required.`,
      confirmText: nextState ? 'Activate' : 'Deactivate',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      await api.updateUser(user.id.toString(), {
        is_active: nextState,
        reason: approval.reason,
        admin_password: approval.adminPassword,
      });
      success('User updated', `${user.username || user.email} ${nextState ? 'activated' : 'deactivated'}.`);
      void loadUsers();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to update user';
      showError('Update failed', message);
    }
  };

  const handleDeleteUser = async (user: User) => {
    if (currentUser && user.id === currentUser.id) {
      showError('Cannot delete yourself', 'You cannot delete your own account.');
      return;
    }
    const userId = user.id;
    if (deletingUserIds.has(userId)) return;
    const approval = await promptPrivilegedAction({
      title: 'Delete User',
      message: `Delete ${user.username || user.email}? This cannot be undone.`,
      confirmText: 'Delete',
      requirePassword: requirePasswordReauth,
    });
    if (!approval) return;

    try {
      setDeletingUserIds((prev) => {
        const next = new Set(prev);
        next.add(userId);
        return next;
      });
      await api.deleteUser(String(userId), {
        reason: approval.reason,
        admin_password: approval.adminPassword,
      });
      success('User deleted', `${user.username || user.email} removed.`);
      void loadUsers();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete user';
      showError('Delete failed', message);
    } finally {
      setDeletingUserIds((prev) => {
        const next = new Set(prev);
        next.delete(userId);
        return next;
      });
    }
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8 flex flex-col sm:flex-row sm:items-center justify-between gap-4">
              <div>
                <h1 className="text-3xl font-bold">Users</h1>
                <p className="text-muted-foreground">Manage system users and their access</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <ExportMenu
                  onExport={handleExport}
                  disabled={filteredUsers.length === 0}
                />
                <Dialog open={showCreateUserDialog} onOpenChange={setShowCreateUserDialog}>
                  <DialogTrigger asChild>
                    <Button>
                      <Plus className="mr-2 h-4 w-4" />
                      Create User
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Create user</DialogTitle>
                      <DialogDescription>Create a user with a temporary password.</DialogDescription>
                    </DialogHeader>
                    {createUserError && (
                      <Alert variant="destructive">
                        <AlertDescription>{createUserError}</AlertDescription>
                      </Alert>
                    )}
                    <FormProvider {...createUserForm}>
                      <Form onSubmit={handleCreateUserSubmit}>
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                          <FormInput<CreateUserFormData>
                            name="username"
                            label="Username"
                            required
                          />
                          <FormInput<CreateUserFormData>
                            name="email"
                            label="Email"
                            type="email"
                            required
                          />
                        </div>
                        <FormInput<CreateUserFormData>
                          name="password"
                          label="Password"
                          type="password"
                          description="Minimum 10 characters."
                          required
                        />
                        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                          <FormSelect<CreateUserFormData>
                            name="role"
                            label="Role"
                            options={[
                              { value: 'user', label: 'User' },
                              { value: 'admin', label: 'Admin' },
                              { value: 'service', label: 'Service' },
                            ]}
                          />
                          <div className="space-y-2">
                            <Label className="block">Status</Label>
                            <div className="space-y-2">
                              <FormCheckbox<CreateUserFormData>
                                name="is_active"
                                label="Active"
                              />
                              <FormCheckbox<CreateUserFormData>
                                name="is_verified"
                                label="Verified"
                              />
                            </div>
                          </div>
                        </div>
                        <DialogFooter className="gap-2 sm:gap-0">
                          <Button
                            type="button"
                            variant="outline"
                            onClick={() => setShowCreateUserDialog(false)}
                            disabled={creatingUser}
                          >
                            Cancel
                          </Button>
                          <Button type="submit" loading={creatingUser} loadingText="Creating...">
                            Create user
                          </Button>
                        </DialogFooter>
                      </Form>
                    </FormProvider>
                  </DialogContent>
                </Dialog>
                <Dialog open={showInviteUserDialog} onOpenChange={setShowInviteUserDialog}>
                  <DialogTrigger asChild>
                    <Button variant="outline">
                      <Mail className="mr-2 h-4 w-4" />
                      Invite User
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Invite user</DialogTitle>
                      <DialogDescription>Send an invitation email to a new user.</DialogDescription>
                    </DialogHeader>
                    {inviteUserError && (
                      <Alert variant="destructive">
                        <AlertDescription>{inviteUserError}</AlertDescription>
                      </Alert>
                    )}
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <Label htmlFor="invite-email">Email address</Label>
                        <Input
                          id="invite-email"
                          type="email"
                          placeholder="user@example.com"
                          value={inviteEmail}
                          onChange={(e) => setInviteEmail(e.target.value)}
                          disabled={invitingUser}
                        />
                      </div>
                      <div className="space-y-2">
                        <Label htmlFor="invite-role">Role</Label>
                        <Select
                          id="invite-role"
                          value={inviteRole}
                          onChange={(e) => setInviteRole(e.target.value)}
                          disabled={invitingUser}
                        >
                          <option value="user">User</option>
                          <option value="admin">Admin</option>
                          <option value="viewer">Viewer</option>
                          <option value="service">Service</option>
                        </Select>
                      </div>
                    </div>
                    <DialogFooter className="gap-2 sm:gap-0">
                      <Button
                        type="button"
                        variant="outline"
                        onClick={() => setShowInviteUserDialog(false)}
                        disabled={invitingUser}
                      >
                        Cancel
                      </Button>
                      <Button
                        onClick={handleInviteUser}
                        loading={invitingUser}
                        loadingText="Sending..."
                      >
                        Send invitation
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            </div>

            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {/* Search */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="space-y-4">
                  <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="relative max-w-md w-full">
                      <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" aria-hidden="true" />
                      <label htmlFor="users-search" className="sr-only">
                        Search users by username, email, or role
                      </label>
                      <Input
                        id="users-search"
                        placeholder="Search by username, email, or role..."
                        value={searchQuery || ''}
                        onChange={(e) => handleSearchChange(e.target.value)}
                        className="pl-10"
                      />
                    </div>
                    <div className="flex flex-wrap items-center gap-2">
                      <Select
                        id="users-saved-view-filter"
                        value={activeViewId}
                        onChange={(event) => handleApplySavedView(event.target.value)}
                        className="min-w-[200px]"
                        aria-label="Saved views"
                        disabled={savedViews.length === 0}
                      >
                        <option value="">Saved views</option>
                        {savedViews.map((view) => (
                          <option key={view.id} value={view.id}>
                            {view.name}
                          </option>
                        ))}
                      </Select>
                      <Dialog open={showSaveViewDialog} onOpenChange={(open) => {
                        setShowSaveViewDialog(open);
                        if (!open) {
                          clearSaveViewForm();
                        }
                      }}>
                        <DialogTrigger asChild>
                          <Button variant="outline">
                            <BookmarkPlus className="mr-2 h-4 w-4" />
                            Save view
                          </Button>
                        </DialogTrigger>
                        <DialogContent>
                          <DialogHeader>
                            <DialogTitle>Save view</DialogTitle>
                            <DialogDescription>Store the current search for quick reuse.</DialogDescription>
                          </DialogHeader>
                          {saveViewError && (
                            <Alert variant="destructive">
                              <AlertDescription>{saveViewError}</AlertDescription>
                            </Alert>
                          )}
                          <div className="space-y-2">
                            <Label htmlFor="saved-view-name">View name</Label>
                            <Input
                              id="saved-view-name"
                              value={saveViewName}
                              onChange={(event) => setSaveViewName(event.target.value)}
                              placeholder="e.g., Inactive admins"
                            />
                            <p className="text-xs text-muted-foreground">
                              Current search: {searchQuery || 'All users'}
                            </p>
                          </div>
                          <DialogFooter>
                            <Button variant="outline" onClick={() => setShowSaveViewDialog(false)}>
                              Cancel
                            </Button>
                            <Button onClick={handleSaveView}>
                              Save view
                            </Button>
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>
                      <Button
                        variant="outline"
                        onClick={handleDeleteView}
                        disabled={!activeViewId}
                      >
                        <BookmarkX className="mr-2 h-4 w-4" />
                        Delete view
                      </Button>
                    </div>
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Label htmlFor="users-status-filter" className="sr-only">Filter by user status</Label>
                    <Select
                      id="users-status-filter"
                      className="min-w-[160px]"
                      value={statusFilter || 'all'}
                      onChange={(event) => handleStatusFilterChange(event.target.value as UserStatusFilter)}
                    >
                      <option value="all">Status: All</option>
                      <option value="active">Status: Active</option>
                      <option value="inactive">Status: Inactive</option>
                    </Select>
                    <Label htmlFor="users-verified-filter" className="sr-only">Filter by verification state</Label>
                    <Select
                      id="users-verified-filter"
                      className="min-w-[160px]"
                      value={verifiedFilter || 'all'}
                      onChange={(event) => handleVerifiedFilterChange(event.target.value as UserVerifiedFilter)}
                    >
                      <option value="all">Verified: All</option>
                      <option value="verified">Verified: Yes</option>
                      <option value="unverified">Verified: No</option>
                    </Select>
                    <Label htmlFor="users-mfa-filter" className="sr-only">Filter by MFA status</Label>
                    <Select
                      id="users-mfa-filter"
                      className="min-w-[160px]"
                      value={mfaFilter || 'all'}
                      onChange={(event) => handleMfaFilterChange(event.target.value as UserMfaFilter)}
                    >
                      <option value="all">MFA: All</option>
                      <option value="enabled">MFA: Enabled</option>
                      <option value="disabled">MFA: Disabled</option>
                    </Select>
                    <Button
                      variant="ghost"
                      size="sm"
                      onClick={handleClearFilters}
                      disabled={!hasActiveFilters}
                    >
                      Clear filters
                    </Button>
                  </div>
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Pending Invitations</CardTitle>
                <CardDescription>
                  Direct email invitations sent to prospective users.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {directInvitesError && (
                  <Alert variant="destructive">
                    <AlertDescription>{directInvitesError}</AlertDescription>
                  </Alert>
                )}
                {directInvitesLoading ? (
                  <div className="text-sm text-muted-foreground">Loading invitations...</div>
                ) : directInvitations.length === 0 ? (
                  <EmptyState
                    icon={Mail}
                    title="No invitations"
                    description="Use the Invite User button to send email invitations."
                  />
                ) : (
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Status</TableHead>
                          <TableHead>Email</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Invited by</TableHead>
                          <TableHead>Email Sent</TableHead>
                          <TableHead>Created</TableHead>
                          <TableHead>Expires</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {directInvitations.map((inv) => (
                          <TableRow key={inv.id} data-testid={`direct-invitation-row-${inv.id}`}>
                            <TableCell>
                              <Badge variant={
                                inv.status === 'accepted' ? 'default'
                                : inv.status === 'revoked' || inv.status === 'expired' ? 'destructive'
                                : 'secondary'
                              }>
                                {inv.status}
                              </Badge>
                            </TableCell>
                            <TableCell>{inv.email}</TableCell>
                            <TableCell>{inv.role}</TableCell>
                            <TableCell>{inv.invited_by || '\u2014'}</TableCell>
                            <TableCell>
                              {inv.email_sent ? (
                                <Badge variant="default">Sent</Badge>
                              ) : (
                                <Badge variant="destructive" title={inv.email_error || 'Not sent'}>
                                  Not sent
                                </Badge>
                              )}
                            </TableCell>
                            <TableCell>
                              {inv.created_at ? new Date(inv.created_at).toLocaleDateString() : '\u2014'}
                            </TableCell>
                            <TableCell>
                              {inv.expires_at ? new Date(inv.expires_at).toLocaleDateString() : '\u2014'}
                            </TableCell>
                            <TableCell className="text-right">
                              {inv.status === 'pending' && (
                                <span className="inline-flex gap-1">
                                  <AccessibleIconButton
                                    icon={RefreshCw}
                                    label="Resend invitation"
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleResendInvitation(inv)}
                                    disabled={resendingInviteIds.has(inv.id) || (inv.resend_count ?? 0) >= 3}
                                    loading={resendingInviteIds.has(inv.id)}
                                    title={(inv.resend_count ?? 0) >= 3 ? 'Resend limit reached (3)' : `Resend (${inv.resend_count ?? 0}/3)`}
                                  />
                                  <AccessibleIconButton
                                    icon={XCircle}
                                    label="Revoke invitation"
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleRevokeInvitation(inv.id)}
                                    disabled={revokingInviteIds.has(inv.id)}
                                    loading={revokingInviteIds.has(inv.id)}
                                    className="text-destructive hover:text-destructive"
                                  />
                                </span>
                              )}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Organization Invitations</CardTitle>
                <CardDescription>
                  Onboarding invitation visibility across organizations.
                </CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                {invitesError && (
                  <Alert variant="destructive">
                    <AlertDescription>{invitesError}</AlertDescription>
                  </Alert>
                )}
                <div className="grid gap-3 sm:grid-cols-4">
                  <div className="rounded-lg border p-3">
                    <p className="text-xs uppercase text-muted-foreground">Total Sent</p>
                    <p className="text-2xl font-semibold" data-testid="invitation-total-sent">
                      {invitationFunnel.totalSent}
                    </p>
                  </div>
                  <div className="rounded-lg border p-3">
                    <p className="text-xs uppercase text-muted-foreground">Accepted</p>
                    <p className="text-2xl font-semibold" data-testid="invitation-total-accepted">
                      {invitationFunnel.totalAccepted}
                    </p>
                  </div>
                  <div className="rounded-lg border p-3">
                    <p className="text-xs uppercase text-muted-foreground">Pending</p>
                    <p className="text-2xl font-semibold">{invitationFunnel.totalPending}</p>
                  </div>
                  <div className="rounded-lg border p-3">
                    <p className="text-xs uppercase text-muted-foreground">Conversion Rate</p>
                    <p className="text-2xl font-semibold" data-testid="invitation-conversion-rate">
                      {invitationFunnel.conversionRate.toFixed(1)}%
                    </p>
                  </div>
                </div>

                {invitesLoading ? (
                  <div className="text-sm text-muted-foreground">Loading invitations…</div>
                ) : invitationRows.length === 0 ? (
                  <div className="text-sm text-muted-foreground">No invitations found.</div>
                ) : (
                  <div className="rounded-md border">
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Status</TableHead>
                          <TableHead>Email</TableHead>
                          <TableHead>Invited by</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Org</TableHead>
                          <TableHead>Sent</TableHead>
                          <TableHead>Expires</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {invitationRows.map((invite) => (
                          <TableRow key={invite.id} data-testid={`invitation-row-${invite.id}`}>
                            <TableCell>
                              <Badge variant={invitationStatusBadgeVariant(invite.status)}>
                                {invite.status}
                              </Badge>
                            </TableCell>
                            <TableCell>{invite.email}</TableCell>
                            <TableCell>{invite.invitedBy}</TableCell>
                            <TableCell>{invite.role}</TableCell>
                            <TableCell>{invite.org}</TableCell>
                            <TableCell>
                              {invite.sentAt ? new Date(invite.sentAt).toLocaleDateString() : '—'}
                            </TableCell>
                            <TableCell>
                              {invite.expiresAt ? new Date(invite.expiresAt).toLocaleDateString() : '—'}
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                )}
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Users List</CardTitle>
                <CardDescription>
                  {totalItems} user{totalItems !== 1 ? 's' : ''} found
                </CardDescription>
              </CardHeader>
              <CardContent>
                <UserBulkActions
                  selectedCount={selectedCount}
                  bulkRole={bulkRole}
                  bulkRoleOptions={bulkRoleOptions}
                  bulkBusy={bulkBusy}
                  bulkAction={bulkAction}
                  onBulkRoleChange={setBulkRole}
                  onAssignRole={handleBulkAssignRole}
                  onActivate={() => handleBulkToggleActive(true)}
                  onDeactivate={() => handleBulkToggleActive(false)}
                  onRequireMfa={() => handleBulkSetMfaRequirement(true)}
                  onClearMfa={() => handleBulkSetMfaRequirement(false)}
                  onDelete={handleBulkDelete}
                  onClearSelection={handleClearSelection}
                />
                {loading ? (
                  <div className="py-4">
                    <TableSkeleton rows={5} columns={10} />
                  </div>
                ) : filteredUsers.length === 0 ? (
                  <EmptyState
                    icon={UserCheck}
                    title={searchQuery || hasActiveFilters ? 'No users match your filters' : 'No users found'}
                    description={
                      searchQuery || hasActiveFilters
                        ? 'Try adjusting search terms or clearing filters.'
                        : 'Create your first user to start onboarding.'
                    }
                    actions={[
                      searchQuery || hasActiveFilters
                        ? {
                            label: 'Clear filters',
                            onClick: () => {
                              handleSearchChange('');
                              handleClearFilters();
                            },
                          }
                        : {
                            label: 'Create user',
                            onClick: () => setShowCreateUserDialog(true),
                          },
                    ]}
                  />
                ) : (
                  <>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-10">
                            <Checkbox
                              checked={allVisibleSelected}
                              onCheckedChange={handleToggleSelectAllVisible}
                              aria-label="Select all visible users"
                            />
                          </TableHead>
                          <TableHead>ID</TableHead>
                          <TableHead>Username</TableHead>
                          <TableHead>Email</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Status</TableHead>
                          <TableHead>MFA</TableHead>
                          <TableHead>Storage</TableHead>
                          <TableHead>Created</TableHead>
                          <TableHead>Last Login</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {paginatedUsers.map((user) => {
                          const storage = formatStorageUsage(
                            user.storage_used_mb || 0,
                            user.storage_quota_mb || 0
                          );
                          const isCurrentUser = currentUserId === user.id;
                          const isDeleting = deletingUserIds.has(user.id);
                          return (
                            <TableRow key={user.id}>
                              <TableCell>
                                <Checkbox
                                  checked={selectedUserIds.has(user.id)}
                                  onCheckedChange={(checked) => handleToggleSelectUser(user.id, checked)}
                                  aria-label={`Select user ${user.username || user.email || user.id}`}
                                  disabled={isCurrentUser}
                                />
                              </TableCell>
                              <TableCell className="font-mono text-sm">{user.id}</TableCell>
                              <TableCell className="font-medium">{user.username}</TableCell>
                              <TableCell>{user.email}</TableCell>
                              <TableCell>
                                <Badge variant={getRoleBadgeVariant(user.role)}>
                                  {user.role}
                                </Badge>
                              </TableCell>
                              <TableCell>
                                <Badge variant={user.is_active ? 'default' : 'destructive'}>
                                  {user.is_active ? 'Active' : 'Inactive'}
                                </Badge>
                                {user.is_verified && (
                                  <Badge variant="outline" className="ml-1">
                                    Verified
                                  </Badge>
                                )}
                              </TableCell>
                              <TableCell>
                                {user.mfa_enabled ? (
                                  <span className="text-green-600" title="MFA enabled" aria-label="MFA enabled">&#10003;</span>
                                ) : (
                                  <span className="text-gray-400" title="MFA disabled" aria-label="MFA disabled">&#8212;</span>
                                )}
                              </TableCell>
                              <TableCell>
                                <div className="space-y-1">
                                  <div className="text-xs">{storage.text}</div>
                                  <div
                                    className="w-20 bg-gray-200 rounded-full h-1.5"
                                    role="progressbar"
                                    aria-valuenow={Math.round(storage.percentage)}
                                    aria-valuemin={0}
                                    aria-valuemax={100}
                                    aria-label={`Storage usage: ${Math.round(storage.percentage)}%${
                                      storage.percentage > 90 ? ', critical' :
                                      storage.percentage > 70 ? ', warning' : ''
                                    }`}
                                  >
                                    <div
                                      className={`h-1.5 rounded-full ${
                                        storage.percentage > 90 ? 'bg-red-500' :
                                        storage.percentage > 70 ? 'bg-yellow-500' :
                                        'bg-green-500'
                                      }`}
                                      style={{ width: `${storage.percentage}%` }}
                                    />
                                  </div>
                                </div>
                              </TableCell>
                              <TableCell className="text-muted-foreground text-sm">
                                {user.created_at
                                  ? new Date(user.created_at).toLocaleDateString()
                                  : '—'}
                              </TableCell>
                              <TableCell className="text-muted-foreground text-sm">
                                <div className="flex items-center gap-1.5">
                                  <span>
                                    {user.last_login
                                      ? new Date(user.last_login).toLocaleDateString()
                                      : 'Never'}
                                  </span>
                                  {(() => {
                                    const DORMANT_THRESHOLD_DAYS = 90;
                                    const now = Date.now();
                                    const lastLoginMs = user.last_login ? Date.parse(user.last_login) : 0;
                                    const daysSinceLogin = !user.last_login || !Number.isFinite(lastLoginMs)
                                      ? Infinity
                                      : (now - lastLoginMs) / (1000 * 60 * 60 * 24);
                                    return daysSinceLogin > DORMANT_THRESHOLD_DAYS ? (
                                      <Badge variant="destructive" className="text-[10px] px-1.5 py-0">
                                        Dormant
                                      </Badge>
                                    ) : null;
                                  })()}
                                </div>
                              </TableCell>
                              <TableCell className="text-right">
                                <div className="flex justify-end gap-1">
                                  <AccessibleIconButton
                                    icon={Eye}
                                    label="View user details"
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => router.push(`/users/${user.id}`)}
                                  />
                                  <AccessibleIconButton
                                    icon={Key}
                                    label="Manage API keys"
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => router.push(`/users/${user.id}/api-keys`)}
                                  />
                                  <AccessibleIconButton
                                    icon={user.is_active ? UserX : UserCheck}
                                    label={user.is_active ? 'Deactivate user' : 'Activate user'}
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleToggleActive(user)}
                                  />
                                  <AccessibleIconButton
                                    icon={Trash2}
                                    label={isDeleting ? 'Deleting user' : isCurrentUser ? 'Cannot delete yourself' : 'Delete user'}
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleDeleteUser(user)}
                                    disabled={isCurrentUser || isDeleting}
                                    loading={isDeleting}
                                    className="text-destructive hover:text-destructive"
                                  />
                                </div>
                              </TableCell>
                            </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>

                    <Pagination
                      currentPage={currentPage}
                      totalPages={totalPages}
                      totalItems={totalItems}
                      pageSize={pageSize}
                      onPageChange={handlePageChange}
                      onPageSizeChange={handlePageSizeChange}
                    />
                  </>
                )}
              </CardContent>
            </Card>
          </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Wrap with Suspense for useSearchParams
export default function UsersPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8">
              <Skeleton className="h-8 w-32 mb-2" />
              <Skeleton className="h-4 w-64" />
            </div>
            <TableSkeleton rows={5} columns={9} />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <UsersPageContent />
    </Suspense>
  );
}
