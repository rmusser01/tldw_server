'use client';

import { useCallback, useEffect, useMemo, useState, type ReactNode } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import { ArrowLeft, Key, Building2, Users, Trash2, Clock, ShieldCheck, Plus, X } from 'lucide-react';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import { api, ApiError } from '@/lib/api-client';
import { isSingleUserMode } from '@/lib/auth';
import { parseOptionalInt } from '@/lib/number';
import {
  deriveLimitPerMinute,
  getDerivedLimitPerMin,
  normalizeRateLimitValue,
  validateRateLimitInputs,
} from '@/lib/rate-limits';
import { canEditFromMemberships } from '@/lib/permissions';
import { User, Permission, OrgMembership, TeamMembership } from '@/types';
import {
  UserProfileCard,
  type UserProfileFormData,
} from './components/UserProfileCard';
import { UserSecurityCard } from './components/UserSecurityCard';
import { UserMembershipDialogs } from './components/UserMembershipDialogs';
import { useUserSecurity } from './hooks/use-user-security';
import Link from 'next/link';
import { logger } from '@/lib/logger';

type UserRateLimits = {
  requests_per_minute?: number | null;
  requests_per_hour?: number | null;
  requests_per_day?: number | null;
};

type RateLimitUpsertPayload = {
  resource: string;
  limit_per_min: number | null;
  burst: number | null;
};

const DEFAULT_RATE_LIMIT_RESOURCE = 'api.default';

type PermissionOverride = {
  id: number;
  permission_id: number;
  permission_name: string;
  grant: boolean;
};

type EffectivePermissionSource = 'role' | 'override' | 'inherited';

type EffectivePermission = {
  id: number;
  name: string;
  source: EffectivePermissionSource;
  sourceLabel: string;
};

const roleOptions = [
  { value: 'user', label: 'User' },
  { value: 'admin', label: 'Admin' },
  { value: 'service', label: 'Service' },
] as const;

type UserRole = (typeof roleOptions)[number]['value'];

const isValidRole = (role: string): role is UserRole => roleOptions.some((option) => option.value === role);

type RateLimitRecord = {
  resource?: string;
  limit_per_min?: number | null;
  burst?: number | null;
  requests_per_minute?: number | null;
  requests_per_hour?: number | null;
  requests_per_day?: number | null;
};

const toOptionalNumber = (input: unknown): number | null => {
  if (typeof input !== 'number' || !Number.isFinite(input)) return null;
  return input > 0 ? input : null;
};

const selectRateLimitRecord = (items: RateLimitRecord[]): RateLimitRecord | null => {
  if (!items.length) return null;
  return items.find((item) => item.resource === DEFAULT_RATE_LIMIT_RESOURCE) ?? items[0];
};

const normalizeRateLimits = (value: unknown): UserRateLimits | null => {
  if (!value) return null;
  if (Array.isArray(value)) {
    const record = selectRateLimitRecord(value as RateLimitRecord[]);
    return record ? normalizeRateLimits(record) : null;
  }
  if (typeof value !== 'object') return null;

  const container = value as RateLimitRecord & { rate_limits?: unknown; items?: unknown };
  if (container.rate_limits) {
    return normalizeRateLimits(container.rate_limits);
  }
  if (container.items) {
    return normalizeRateLimits(container.items);
  }

  const rpm = toOptionalNumber(container.requests_per_minute ?? container.limit_per_min);
  const rph = toOptionalNumber(container.requests_per_hour);
  const rpd = toOptionalNumber(container.requests_per_day);

  if (rpm === null && rph === null && rpd === null) return null;
  return {
    requests_per_minute: rpm,
    requests_per_hour: rph,
    requests_per_day: rpd,
  };
};

const isForbiddenError = (err: unknown): boolean => {
  if (err instanceof ApiError) {
    return err.status === 403;
  }
  if (typeof err === 'object' && err !== null && 'status' in err) {
    return (err as { status?: number }).status === 403;
  }
  if (err instanceof Error) {
    return /not authorized|forbidden|permission/i.test(err.message);
  }
  return false;
};

const normalizePermissionOverrideList = (value: unknown): PermissionOverride[] => {
  if (!Array.isArray(value)) return [];
  return value
    .map((item) => {
      const record = item as Record<string, unknown>;
      const id = Number(record.id);
      const permissionId = Number(record.permission_id ?? record.permissionId ?? record.id);
      const permissionName = String(record.permission_name ?? record.permissionName ?? '').trim();
      const grantValue = record.grant;
      const grant = typeof grantValue === 'boolean' ? grantValue : true;
      if (!permissionName || !Number.isFinite(permissionId)) return null;
      return {
        id: Number.isFinite(id) ? id : permissionId,
        permission_id: permissionId,
        permission_name: permissionName,
        grant,
      };
    })
    .filter((entry): entry is PermissionOverride => entry !== null);
};

const normalizeEffectivePermissionList = (
  value: unknown,
  overrideEntries: PermissionOverride[],
  roleName?: string
): EffectivePermission[] => {
  if (!Array.isArray(value)) return [];
  const overrideGrantNames = new Set(
    overrideEntries
      .filter((entry) => entry.grant)
      .map((entry) => entry.permission_name)
  );
  return value
    .map((item, index) => {
      const roleLabel = roleName || '';
      if (typeof item === 'string') {
        const permissionName = item.trim();
        if (!permissionName) return null;
        if (overrideGrantNames.has(permissionName)) {
          return {
            id: index,
            name: permissionName,
            source: 'override' as const,
            sourceLabel: 'Direct override',
          };
        }
        return {
          id: index,
          name: permissionName,
          source: 'role' as const,
          sourceLabel: roleLabel || 'role',
        };
      }

      if (!item || typeof item !== 'object') return null;
      const record = item as Record<string, unknown>;
      const permissionName = String(
        record.name ?? record.permission_name ?? record.permission ?? ''
      ).trim();
      if (!permissionName) return null;
      const sourceText = String(record.source ?? '').toLowerCase();
      const explicitRoleName = String(record.role_name ?? record.source_role ?? '').trim();
      const idValue = Number(record.id ?? index);

      let source: EffectivePermissionSource = 'inherited';
      let sourceLabel = 'Inherited';
      if (sourceText.includes('override') || overrideGrantNames.has(permissionName)) {
        source = 'override';
        sourceLabel = 'Direct override';
      } else if (sourceText.includes('inherit')) {
        source = 'inherited';
        sourceLabel = 'Inherited';
      } else if (sourceText.includes('role') || explicitRoleName) {
        source = 'role';
        sourceLabel = explicitRoleName || roleLabel || 'role';
      } else if (roleLabel) {
        source = 'role';
        sourceLabel = roleLabel;
      }

      return {
        id: Number.isFinite(idValue) ? idValue : index,
        name: permissionName,
        source,
        sourceLabel,
      };
    })
    .filter((entry): entry is EffectivePermission => entry !== null);
};

export default function UserDetailPage() {
  const params = useParams();
  const router = useRouter();
  const userId = typeof params.id === 'string' ? params.id : params.id?.[0] ?? '';
  const confirm = useConfirm();
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { success: toastSuccess, error: showError } = useToast();
  const requirePasswordReauth = !isSingleUserMode();

  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [isAuthorized, setIsAuthorized] = useState(true);
  const [showOrgMembershipsDialog, setShowOrgMembershipsDialog] = useState(false);
  const [showTeamMembershipsDialog, setShowTeamMembershipsDialog] = useState(false);
  const [orgMemberships, setOrgMemberships] = useState<OrgMembership[]>([]);
  const [teamMemberships, setTeamMemberships] = useState<TeamMembership[]>([]);
  const [orgMembershipsLoading, setOrgMembershipsLoading] = useState(false);
  const [teamMembershipsLoading, setTeamMembershipsLoading] = useState(false);
  const [orgMembershipsError, setOrgMembershipsError] = useState('');
  const [teamMembershipsError, setTeamMembershipsError] = useState('');

  const [formData, setFormData] = useState<UserProfileFormData>({
    username: '',
    email: '',
    role: 'user',
    is_active: true,
    storage_quota_mb: 0,
  });

  // Rate Limits
  const [rateLimits, setRateLimits] = useState<UserRateLimits>({});
  const [editRpm, setEditRpm] = useState('');
  const [editRph, setEditRph] = useState('');
  const [editRpd, setEditRpd] = useState('');
  const [rateLimitsSaving, setRateLimitsSaving] = useState(false);

  // Permission Overrides
  const [permissionOverrides, setPermissionOverrides] = useState<PermissionOverride[]>([]);
  const [effectivePermissions, setEffectivePermissions] = useState<EffectivePermission[]>([]);
  const [permFilter, setPermFilter] = useState('');
  const [allPermissions, setAllPermissions] = useState<Permission[]>([]);
  const [permissionsLoading, setPermissionsLoading] = useState(false);
  const [showAddOverride, setShowAddOverride] = useState(false);
  const [newOverridePermissionId, setNewOverridePermissionId] = useState('');
  const [newOverrideGrant, setNewOverrideGrant] = useState(true);
  const [permSearchQuery, setPermSearchQuery] = useState('');
  const [permSourceFilter, setPermSourceFilter] = useState<'all' | EffectivePermissionSource>('all');

  const filteredPerms = useMemo(() =>
    effectivePermissions.filter((perm) => {
      const matchesSearch = !permSearchQuery || perm.name.toLowerCase().includes(permSearchQuery.toLowerCase());
      const matchesSource = permSourceFilter === 'all' || perm.source === permSourceFilter;
      return matchesSearch && matchesSource;
    }),
    [effectivePermissions, permSearchQuery, permSourceFilter],
  );

  const applyRateLimits = useCallback((limits?: UserRateLimits | null) => {
    if (!limits) {
      setRateLimits({});
      setEditRpm('');
      setEditRph('');
      setEditRpd('');
      return;
    }
    setRateLimits(limits);
    setEditRpm(limits.requests_per_minute?.toString() || '');
    setEditRph(limits.requests_per_hour?.toString() || '');
    setEditRpd(limits.requests_per_day?.toString() || '');
  }, []);

  const loadUser = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      setIsAuthorized(true);
      const data = await api.getUser(userId);
      const userValue = data as User & { rate_limits?: UserRateLimits; metadata?: Record<string, unknown> };
      const roleValue = typeof userValue.role === 'string' && userValue.role.trim()
        ? userValue.role
        : 'user';
      setUser(userValue);
      setFormData({
        username: userValue.username || '',
        email: userValue.email || '',
        role: roleValue,
        is_active: userValue.is_active ?? true,
        storage_quota_mb: userValue.storage_quota_mb || 0,
      });
      const normalizedRateLimits = normalizeRateLimits(userValue.rate_limits);
      applyRateLimits(normalizedRateLimits);

      try {
        const currentUser = await api.getCurrentUser();
        const [adminMemberships, targetMemberships] = await Promise.all([
          api.getUserOrgMemberships(currentUser.id.toString()),
          api.getUserOrgMemberships(userId),
        ]);

        const adminList = Array.isArray(adminMemberships) ? adminMemberships : [];
        const targetList = Array.isArray(targetMemberships) ? targetMemberships : [];

        if (!canEditFromMemberships(adminList, targetList)) {
          setIsAuthorized(false);
          setError('You are not authorized to edit this user.');
        }
      } catch (scopeErr: unknown) {
        if (isForbiddenError(scopeErr)) {
          setIsAuthorized(false);
          setError('You are not authorized to edit this user.');
        } else {
          logger.error('Failed to verify user scope', { component: 'UserDetailPage', error: scopeErr instanceof Error ? scopeErr.message : String(scopeErr) });
        }
      }
    } catch (err: unknown) {
      if (isForbiddenError(err)) {
        setIsAuthorized(false);
        setError('You are not authorized to view or edit this user.');
        setUser(null);
        return;
      }
      const message =
        err instanceof Error ? err.message : typeof err === 'string' ? err : 'Failed to load user';
      logger.error('Failed to load user', { component: 'UserDetailPage', error: message });
      setError(message);
    } finally {
      setLoading(false);
    }
  }, [applyRateLimits, userId]);
  const userSecurity = useUserSecurity({
    userId,
    user,
    requirePasswordReauth,
    promptPrivilegedAction,
    onUserRefresh: loadUser,
    showError,
    toastSuccess,
  });
  const { loadSecurity, loadLoginHistory } = userSecurity;

  const loadOrgMemberships = useCallback(async () => {
    if (!userId) return;
    try {
      setOrgMembershipsLoading(true);
      setOrgMembershipsError('');
      const response = await api.getUserOrgMemberships(userId);
      const normalized = response
        .filter((item) => Number.isFinite(item.org_id) && typeof item.role === 'string' && item.role.trim().length > 0)
        .map((item) => ({
          org_id: item.org_id,
          role: item.role.trim(),
          org_name: typeof item.org_name === 'string' ? item.org_name : undefined,
        }));
      setOrgMemberships(normalized);
    } catch (err: unknown) {
      logger.error('Failed to load user organizations', { component: 'UserDetailPage', error: err instanceof Error ? err.message : String(err) });
      setOrgMemberships([]);
      setOrgMembershipsError('Failed to load user organizations.');
    } finally {
      setOrgMembershipsLoading(false);
    }
  }, [userId]);

  const loadTeamMemberships = useCallback(async () => {
    if (!userId) return;
    try {
      setTeamMembershipsLoading(true);
      setTeamMembershipsError('');
      const response = await api.getUserTeamMemberships(userId);
      const normalized = response
        .filter(
          (item) =>
            Number.isFinite(item.team_id)
            && Number.isFinite(item.org_id)
            && typeof item.role === 'string'
            && item.role.trim().length > 0
        )
        .map((item) => ({
          team_id: item.team_id,
          org_id: item.org_id,
          role: item.role.trim(),
          team_name: typeof item.team_name === 'string' ? item.team_name : undefined,
          org_name: typeof item.org_name === 'string' ? item.org_name : undefined,
        }));
      setTeamMemberships(normalized);
    } catch (err: unknown) {
      logger.error('Failed to load user teams', { component: 'UserDetailPage', error: err instanceof Error ? err.message : String(err) });
      setTeamMemberships([]);
      setTeamMembershipsError('Failed to load user teams.');
    } finally {
      setTeamMembershipsLoading(false);
    }
  }, [userId]);

  const loadPermissions = useCallback(async () => {
    if (!userId) return;
    try {
      setPermissionsLoading(true);
      const results = await Promise.allSettled([
        api.getUserEffectivePermissions(userId),
        api.getUserPermissionOverrides(userId),
        api.getPermissions(),
        api.getUserRateLimits(userId),
      ]);
      const [effectiveResult, overridesResult, allPermsResult, rateLimitsResult] = results;
      const hasRejected = results.some((result) => result.status === 'rejected');
      if (hasRejected) {
        setEffectivePermissions([]);
        setPermissionOverrides([]);
        setAllPermissions([]);
        applyRateLimits(normalizeRateLimits({}));
        return;
      }

      let normalizedOverrides: PermissionOverride[] = [];
      if (overridesResult.status === 'fulfilled') {
        const data = overridesResult.value as { overrides?: unknown; items?: unknown };
        const rawOverrides = Array.isArray(data.overrides) ? data.overrides :
          Array.isArray(data.items) ? data.items :
          Array.isArray(data) ? data : [];
        normalizedOverrides = normalizePermissionOverrideList(rawOverrides);
      }
      setPermissionOverrides(normalizedOverrides);

      if (effectiveResult.status === 'fulfilled') {
        const data = effectiveResult.value as { permissions?: unknown; items?: unknown };
        const rawPermissions = Array.isArray(data.permissions) ? data.permissions :
          Array.isArray(data.items) ? data.items :
          Array.isArray(data) ? data : [];
        setEffectivePermissions(
          normalizeEffectivePermissionList(rawPermissions, normalizedOverrides, user?.role)
        );
      } else {
        setEffectivePermissions([]);
      }

      if (allPermsResult.status === 'fulfilled') {
        setAllPermissions(Array.isArray(allPermsResult.value) ? allPermsResult.value : []);
      }

      if (rateLimitsResult.status === 'fulfilled') {
        const normalizedRateLimits = normalizeRateLimits(rateLimitsResult.value);
        applyRateLimits(normalizedRateLimits);
      }
    } catch (err: unknown) {
      logger.error('Failed to load permissions', { component: 'UserDetailPage', error: err instanceof Error ? err.message : String(err) });
    } finally {
      setPermissionsLoading(false);
    }
  }, [applyRateLimits, user?.role, userId]);

  useEffect(() => {
    loadUser();
  }, [loadUser]);

  useEffect(() => {
    if (user && isAuthorized) {
      void loadSecurity();
      void loadPermissions();
      void loadLoginHistory();
    }
  }, [user, isAuthorized, loadLoginHistory, loadPermissions, loadSecurity]);

  const handleSave = async () => {
    if (!isAuthorized) {
      setError('You are not authorized to update this user.');
      return;
    }
    const requiresPrivilegedApproval = Boolean(
      user && (
        formData.role !== user.role
        || formData.is_active !== user.is_active
      )
    );
    try {
      setSaving(true);
      setError('');
      setSuccess('');
      const payload: Record<string, unknown> = {};
      const normalizedEmail = formData.email.trim();
      const currentEmail = typeof user?.email === 'string' ? user.email.trim() : '';
      if (normalizedEmail !== currentEmail) {
        payload.email = normalizedEmail;
      }
      if (user && formData.role !== user.role && isValidRole(formData.role)) {
        payload.role = formData.role;
      }
      if (user && formData.is_active !== user.is_active) {
        payload.is_active = formData.is_active;
      }
      if (user && formData.storage_quota_mb !== user.storage_quota_mb) {
        payload.storage_quota_mb = formData.storage_quota_mb;
      }
      if (Object.keys(payload).length === 0) {
        setSuccess('No changes to save');
        setSaving(false);
        return;
      }
      if (requiresPrivilegedApproval) {
        const approval = await promptPrivilegedAction({
          title: 'Apply privileged user changes',
          message: 'Changing role or activation state requires a reason and reauthentication.',
          confirmText: 'Apply changes',
          requirePassword: requirePasswordReauth,
        });
        if (!approval) {
          setSaving(false);
          return;
        }
        Object.assign(payload, {
          ...payload,
          reason: approval.reason,
          admin_password: approval.adminPassword,
        });
      }
      await api.updateUser(userId, payload);
      setSuccess('User updated successfully');
      void loadUser();
    } catch (err: unknown) {
      if (isForbiddenError(err)) {
        setIsAuthorized(false);
        setError('You are not authorized to update this user.');
        return;
      }
      const message = err instanceof Error ? err.message : String(err);
      logger.error('Failed to update user', { component: 'UserDetailPage', error: message });
      setError(message);
    } finally {
      setSaving(false);
    }
  };

  const handleViewOrganizations = () => {
    setShowOrgMembershipsDialog(true);
    void loadOrgMemberships();
  };

  const handleViewTeams = () => {
    setShowTeamMembershipsDialog(true);
    void loadTeamMemberships();
  };

  const buildRateLimitPayload = (
    rpm: number | null,
    rph: number | null,
    rpd: number | null
  ): RateLimitUpsertPayload => {
    const limitPerMin = getDerivedLimitPerMin(rpm, rph, rpd);
    const burstPerMinute = rph != null
      ? deriveLimitPerMinute(rph, 60)
      : rpd != null
        ? deriveLimitPerMinute(rpd, 1440)
        : null;
    const burstMultiplier = burstPerMinute != null && limitPerMin
      ? Math.max(1, burstPerMinute / limitPerMin)
      : 1;

    return {
      resource: DEFAULT_RATE_LIMIT_RESOURCE,
      limit_per_min: normalizeRateLimitValue(limitPerMin),
      burst: normalizeRateLimitValue(burstMultiplier),
    };
  };

  const handleSaveRateLimits = async () => {
    try {
      setRateLimitsSaving(true);
      setError('');
      const data: UserRateLimits = {};
      const rpm = parseOptionalInt(editRpm);
      const rph = parseOptionalInt(editRph);
      const rpd = parseOptionalInt(editRpd);
      const normalizedRpm = normalizeRateLimitValue(rpm);
      const normalizedRph = normalizeRateLimitValue(rph);
      const normalizedRpd = normalizeRateLimitValue(rpd);
      if (normalizedRpm !== null) data.requests_per_minute = normalizedRpm;
      if (normalizedRph !== null) data.requests_per_hour = normalizedRph;
      if (normalizedRpd !== null) data.requests_per_day = normalizedRpd;

      const { error: rateLimitError } = validateRateLimitInputs(
        normalizedRpm,
        normalizedRph,
        normalizedRpd
      );
      if (rateLimitError) {
        setRateLimitsSaving(false);
        setError(rateLimitError);
        return;
      }

      const payload = buildRateLimitPayload(normalizedRpm, normalizedRph, normalizedRpd);
      await api.setUserRateLimits(userId, payload);
      let normalizedRateLimits: UserRateLimits | null = null;
      try {
        const updated = await api.getUserRateLimits(userId);
        normalizedRateLimits = normalizeRateLimits(updated);
      } catch (err: unknown) {
        logger.error('Failed to reload rate limits after update', { component: 'UserDetailPage', error: err instanceof Error ? err.message : String(err) });
        normalizedRateLimits = data;
      }
      applyRateLimits(normalizedRateLimits);
      toastSuccess('Rate limits updated', 'User rate limits have been saved.');
    } catch (err: unknown) {
      logger.error('Failed to update rate limits', { component: 'UserDetailPage', error: err instanceof Error ? err.message : String(err) });
      const message = err instanceof Error ? err.message : 'Failed to update rate limits';
      showError('Save failed', message);
    } finally {
      setRateLimitsSaving(false);
    }
  };

  const handleClearRateLimits = async () => {
    const confirmed = await confirm({
      title: 'Clear Rate Limits',
      message: 'Remove all custom rate limits for this user? They will inherit role or default limits.',
      confirmText: 'Clear',
      variant: 'danger',
    });
    if (!confirmed) return;

    try {
      setRateLimitsSaving(true);
      await api.setUserRateLimits(userId, {
        resource: DEFAULT_RATE_LIMIT_RESOURCE,
        limit_per_min: null,
        burst: null,
      });
      setRateLimits({});
      setEditRpm('');
      setEditRph('');
      setEditRpd('');
      toastSuccess('Rate limits cleared', 'Custom rate limits have been removed.');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to clear rate limits';
      showError('Clear failed', message);
    } finally {
      setRateLimitsSaving(false);
    }
  };

  const handleAddPermissionOverride = async () => {
    if (!newOverridePermissionId) {
      showError('Select permission', 'Please select a permission to override.');
      return;
    }

    try {
      setPermissionsLoading(true);
      await api.addUserPermissionOverride(userId, {
        permission_id: parseInt(newOverridePermissionId, 10),
        grant: newOverrideGrant,
      });
      toastSuccess('Override added', 'Permission override has been added.');
      setShowAddOverride(false);
      setNewOverridePermissionId('');
      setNewOverrideGrant(true);
      void loadPermissions();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to add override';
      showError('Add failed', message);
    } finally {
      setPermissionsLoading(false);
    }
  };

  const handleRemovePermissionOverride = async (override: PermissionOverride) => {
    const confirmed = await confirm({
      title: 'Remove Override',
      message: `Remove the override for "${override.permission_name}"? The user will inherit this permission from their role.`,
      confirmText: 'Remove',
      variant: 'warning',
    });
    if (!confirmed) return;

    try {
      setPermissionsLoading(true);
      await api.removeUserPermissionOverride(userId, override.id.toString());
      toastSuccess('Override removed', 'Permission override has been removed.');
      void loadPermissions();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to remove override';
      showError('Remove failed', message);
    } finally {
      setPermissionsLoading(false);
    }
  };

  let content: ReactNode = null;

  if (loading) {
    content = (
      <div className="p-4 lg:p-8">
        <div className="text-center text-muted-foreground py-8">Loading user...</div>
      </div>
    );
  } else if (!user) {
    if (!isAuthorized) {
      content = (
        <div className="p-4 lg:p-8">
          <Alert variant="destructive">
            <AlertDescription>You are not authorized to view this user.</AlertDescription>
          </Alert>
          <Button onClick={() => router.push('/users')} className="mt-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Users
          </Button>
        </div>
      );
    } else {
      content = (
        <div className="p-4 lg:p-8">
          <Alert variant="destructive">
            <AlertDescription>User not found</AlertDescription>
          </Alert>
          <Button onClick={() => router.push('/users')} className="mt-4">
            <ArrowLeft className="mr-2 h-4 w-4" />
            Back to Users
          </Button>
        </div>
      );
    }
  } else {
    content = (
      <div className="p-4 lg:p-8">
            {/* Header */}
            <div className="mb-8 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <AccessibleIconButton
                  icon={ArrowLeft}
                  label="Go back to users list"
                  variant="ghost"
                  onClick={() => router.push('/users')}
                />
                <div>
                  <h1 className="text-3xl font-bold">{user.username}</h1>
                  <p className="text-muted-foreground">{user.email}</p>
                </div>
                <Badge variant={user.is_active ? 'default' : 'destructive'}>
                  {user.is_active ? 'Active' : 'Inactive'}
                </Badge>
                <Badge variant="outline">{user.role}</Badge>
              </div>
              <div className="flex gap-2">
                <Link href={`/users/${userId}/api-keys`}>
                  <Button variant="outline">
                    <Key className="mr-2 h-4 w-4" />
                    API Keys
                  </Button>
                </Link>
              </div>
            </div>

            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {success && (
              <Alert className="mb-6 bg-green-50 border-green-200">
                <AlertDescription className="text-green-800">{success}</AlertDescription>
              </Alert>
            )}

            <div className="grid gap-6 lg:grid-cols-2">
              <UserProfileCard
                user={user}
                formData={formData}
                roleOptions={roleOptions}
                isAuthorized={isAuthorized}
                saving={saving}
                isValidRole={isValidRole}
                onFormDataChange={setFormData}
                onSave={handleSave}
              />

              <UserSecurityCard
                isAuthorized={isAuthorized}
                securityLoading={userSecurity.securityLoading}
                securityError={userSecurity.securityError}
                mfaStatus={userSecurity.mfaStatus}
                sessions={userSecurity.sessions}
                loginHistory={userSecurity.loginHistory}
                loginHistoryLoading={userSecurity.loginHistoryLoading}
                loginHistoryError={userSecurity.loginHistoryError}
                forcePasswordChangeOnNextLogin={userSecurity.forcePasswordChangeOnNextLogin}
                passwordResetLoading={userSecurity.passwordResetLoading}
                resetPasswordValue={userSecurity.resetPasswordValue}
                onForcePasswordChangeOnNextLogin={userSecurity.setForcePasswordChangeOnNextLogin}
                onResetPasswordValueChange={userSecurity.setResetPasswordValue}
                onResetPassword={userSecurity.handleResetPassword}
                onDisableMfa={userSecurity.handleDisableMfa}
                onRefreshSecurity={loadSecurity}
                onRevokeAllSessions={userSecurity.handleRevokeAllSessions}
                onRevokeSession={userSecurity.handleRevokeSession}
                onRefreshLoginHistory={loadLoginHistory}
              />

              {/* Rate Limits */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Clock className="h-5 w-5" />
                    Rate Limits
                  </CardTitle>
                  <CardDescription>
                    Set custom rate limits for this user (overrides role defaults)
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  <div className="grid gap-4 sm:grid-cols-3">
                    <div className="space-y-1">
                      <Label htmlFor="user-rate-rpm">Requests/Min</Label>
                      <Input
                        id="user-rate-rpm"
                        type="number"
                        min="0"
                        placeholder="e.g., 60"
                        value={editRpm}
                        onChange={(e) => setEditRpm(e.target.value)}
                        disabled={!isAuthorized}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="user-rate-rph">Requests/Hour</Label>
                      <Input
                        id="user-rate-rph"
                        type="number"
                        min="0"
                        placeholder="e.g., 1000"
                        value={editRph}
                        onChange={(e) => setEditRph(e.target.value)}
                        disabled={!isAuthorized}
                      />
                    </div>
                    <div className="space-y-1">
                      <Label htmlFor="user-rate-rpd">Requests/Day</Label>
                      <Input
                        id="user-rate-rpd"
                        type="number"
                        min="0"
                        placeholder="e.g., 10000"
                        value={editRpd}
                        onChange={(e) => setEditRpd(e.target.value)}
                        disabled={!isAuthorized}
                      />
                    </div>
                  </div>
                  <div className="flex gap-2">
                    <Button
                      onClick={handleSaveRateLimits}
                      disabled={rateLimitsSaving || !isAuthorized}
                      loading={rateLimitsSaving}
                      loadingText="Saving..."
                    >
                      Save Limits
                    </Button>
                    {(rateLimits.requests_per_minute || rateLimits.requests_per_hour || rateLimits.requests_per_day) && (
                      <Button
                        variant="outline"
                        onClick={handleClearRateLimits}
                        disabled={rateLimitsSaving || !isAuthorized}
                      >
                        Clear
                      </Button>
                    )}
                  </div>
                  {(rateLimits.requests_per_minute || rateLimits.requests_per_hour || rateLimits.requests_per_day) && (
                    <p className="text-sm text-muted-foreground">
                      Current limits:{' '}
                      {rateLimits.requests_per_minute && `${rateLimits.requests_per_minute}/min`}
                      {rateLimits.requests_per_minute && rateLimits.requests_per_hour && ', '}
                      {rateLimits.requests_per_hour && `${rateLimits.requests_per_hour}/hr`}
                      {(rateLimits.requests_per_minute || rateLimits.requests_per_hour) && rateLimits.requests_per_day && ', '}
                      {rateLimits.requests_per_day && `${rateLimits.requests_per_day}/day`}
                    </p>
                  )}
                </CardContent>
              </Card>

              {/* Permission Overrides */}
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <ShieldCheck className="h-5 w-5" />
                    Permission Overrides
                  </CardTitle>
                  <CardDescription>
                    Grant or deny specific permissions beyond role defaults
                  </CardDescription>
                </CardHeader>
                <CardContent className="space-y-4">
                  {/* Add Override Form */}
                  {showAddOverride ? (
                    <div className="p-4 border rounded-lg space-y-3">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">Add Permission Override</span>
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => setShowAddOverride(false)}
                          aria-label="Close add permission override"
                        >
                          <X className="h-4 w-4" />
                        </Button>
                      </div>
                      <div className="grid gap-3 sm:grid-cols-2">
                        <div className="space-y-1">
                          <Label htmlFor="override-permission">Permission</Label>
                          <Select
                            id="override-permission"
                            value={newOverridePermissionId}
                            onChange={(e) => setNewOverridePermissionId(e.target.value)}
                          >
                            <option value="">Select permission...</option>
                            {allPermissions.map((perm) => (
                              <option key={perm.id} value={perm.id}>
                                {perm.name}
                              </option>
                            ))}
                          </Select>
                        </div>
                        <div className="space-y-1">
                          <Label htmlFor="override-grant">Action</Label>
                          <Select
                            id="override-grant"
                            value={newOverrideGrant ? 'grant' : 'deny'}
                            onChange={(e) => setNewOverrideGrant(e.target.value === 'grant')}
                          >
                            <option value="grant">Grant</option>
                            <option value="deny">Deny</option>
                          </Select>
                        </div>
                      </div>
                      <div className="flex gap-2">
                        <Button
                          onClick={handleAddPermissionOverride}
                          disabled={permissionsLoading || !newOverridePermissionId}
                        >
                          Add Override
                        </Button>
                        <Button variant="outline" onClick={() => setShowAddOverride(false)}>
                          Cancel
                        </Button>
                      </div>
                    </div>
                  ) : (
                    <Button
                      variant="outline"
                      onClick={() => setShowAddOverride(true)}
                      disabled={!isAuthorized}
                    >
                      <Plus className="mr-2 h-4 w-4" />
                      Add Override
                    </Button>
                  )}

                  {/* Current Overrides */}
                  {permissionOverrides.length > 0 && (
                    <div className="space-y-2">
                      <span className="text-sm font-medium">Active Overrides</span>
                      <div className="space-y-1">
                        {permissionOverrides.map((override) => (
                          <div
                            key={override.id}
                            className={`flex items-center justify-between p-2 rounded text-sm ${
                              override.grant ? 'bg-green-50 dark:bg-green-900/20' : 'bg-red-50 dark:bg-red-900/20'
                            }`}
                          >
                            <div className="flex items-center gap-2">
                              <Badge variant={override.grant ? 'default' : 'destructive'}>
                                {override.grant ? 'Grant' : 'Deny'}
                              </Badge>
                              <code className="font-mono text-xs">{override.permission_name}</code>
                            </div>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={() => handleRemovePermissionOverride(override)}
                              disabled={!isAuthorized || permissionsLoading}
                              aria-label={`Remove override for ${override.permission_name}`}
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  )}

                  {/* Effective Permissions */}
                  {effectivePermissions.length > 0 && (
                    <details className="text-sm">
                      <summary className="cursor-pointer text-muted-foreground hover:text-foreground">
                        View effective permissions ({effectivePermissions.length})
                      </summary>
                      <div className="mt-2 space-y-2">
                        <div className="flex gap-2">
                          <Input
                            placeholder="Search permissions..."
                            value={permSearchQuery}
                            onChange={(e) => setPermSearchQuery(e.target.value)}
                            className="h-8 text-xs"
                          />
                          <Select
                            value={permSourceFilter}
                            onChange={(e) => setPermSourceFilter(e.target.value as 'all' | EffectivePermissionSource)}
                            className="h-8 text-xs w-40"
                          >
                            <option value="all">All sources</option>
                            <option value="role">Role-derived</option>
                            <option value="override">Direct overrides</option>
                            <option value="inherited">Inherited only</option>
                          </Select>
                        </div>
                        <div className="max-h-48 overflow-y-auto space-y-1">
                          {filteredPerms.length === 0 ? (
                            <div className="text-xs text-muted-foreground py-2 text-center">
                              No permissions match your filter.
                            </div>
                          ) : filteredPerms.map((perm, index) => (
                            <div
                              key={perm.id || `perm-${index}`}
                              className="flex items-center justify-between p-2 rounded bg-muted/30"
                            >
                              <code className="font-mono text-xs">{perm.name}</code>
                              {perm.source === 'override' ? (
                                <Badge variant="secondary" className="text-xs">
                                  Direct override
                                </Badge>
                              ) : perm.source === 'role' ? (
                                <Badge variant="outline" className="text-xs">
                                  Role: {perm.sourceLabel || user.role || 'role'}
                                </Badge>
                              ) : (
                                <Badge variant="outline" className="text-xs">
                                  Inherited
                                </Badge>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    </details>
                  )}
                </CardContent>
              </Card>

              {/* Quick Actions */}
              <Card className="lg:col-span-2">
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-4">
                    <Link href={`/users/${userId}/api-keys`}>
                      <Button variant="outline">
                        <Key className="mr-2 h-4 w-4" />
                        Manage API Keys
                      </Button>
                    </Link>
                    <Button variant="outline" onClick={handleViewOrganizations}>
                      <Building2 className="mr-2 h-4 w-4" />
                      View Organizations
                    </Button>
                    <Button variant="outline" onClick={handleViewTeams}>
                      <Users className="mr-2 h-4 w-4" />
                      View Teams
                    </Button>
                  </div>
                </CardContent>
              </Card>

              <UserMembershipDialogs
                showOrgMembershipsDialog={showOrgMembershipsDialog}
                showTeamMembershipsDialog={showTeamMembershipsDialog}
                orgMemberships={orgMemberships}
                teamMemberships={teamMemberships}
                orgMembershipsLoading={orgMembershipsLoading}
                teamMembershipsLoading={teamMembershipsLoading}
                orgMembershipsError={orgMembershipsError}
                teamMembershipsError={teamMembershipsError}
                onOpenOrgMembershipsChange={setShowOrgMembershipsDialog}
                onOpenTeamMembershipsChange={setShowTeamMembershipsDialog}
              />
            </div>
          </div>
    );
  }

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        {content}
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
