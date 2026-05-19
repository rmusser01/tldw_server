'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { ArrowLeft, Shield, Lock, Save, Users, Trash2, Check, X, Clock, Wrench } from 'lucide-react';
import { api } from '@/lib/api-client';
import { parseOptionalInt } from '@/lib/number';
import {
  deriveLimitPerMinute,
  getDerivedLimitPerMin,
  normalizeRateLimitValue,
  validateRateLimitInputs,
} from '@/lib/rate-limits';
import { Role, Permission, User } from '@/types';
import { Label } from '@/components/ui/label';
import { FormSkeleton } from '@/components/ui/skeleton';
import Link from 'next/link';
import { logger } from '@/lib/logger';

type RateLimits = {
  requests_per_minute?: number | null;
  requests_per_hour?: number | null;
  requests_per_day?: number | null;
};

type RateLimitUpsertPayload = {
  resource: string;
  limit_per_min: number | null;
  burst: number | null;
};

type RateLimitUpsertResponse = {
  limit_per_min?: number | null;
  burst?: number | null;
  rate_limits?: RateLimits | null;
};

const DEFAULT_RATE_LIMIT_RESOURCE = 'api.default';

type ToolPermission = {
  id?: number;
  tool_name: string;
  granted: boolean;
};

const MAX_DESCRIPTION_LENGTH = 500;
const USER_LIST_LIMIT = 10;

type PermissionItemProps = {
  perm: Permission;
  isChecked: boolean;
  onToggle: () => void;
  disabled: boolean;
};

const PermissionItem = ({ perm, isChecked, onToggle, disabled }: PermissionItemProps) => (
  <div
    className={`flex items-center justify-between p-3 rounded-lg border ${
      isChecked ? 'bg-primary/5 border-primary/20' : 'bg-muted/30'
    }`}
  >
    <div className="flex items-center gap-3">
      <Checkbox checked={isChecked} onCheckedChange={onToggle} disabled={disabled} />
      <div>
        <code className="text-sm font-mono">{perm.name}</code>
        {perm.description && (
          <p className="text-xs text-muted-foreground">{perm.description}</p>
        )}
      </div>
    </div>
    {isChecked && <Check className="h-4 w-4 text-green-500" />}
  </div>
);

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

const mergeRateLimitsFromResponse = (
  base: RateLimits,
  response: unknown,
  options: { normalizedRpm: number | null; normalizedRph: number | null; normalizedRpd: number | null }
): RateLimits => {
  if (!response || typeof response !== 'object') return base;

  const candidate = response as RateLimitUpsertResponse;
  if (candidate.rate_limits && typeof candidate.rate_limits === 'object') {
    return candidate.rate_limits;
  }

  const merged: RateLimits = { ...base };
  const limitPerMin = candidate.limit_per_min;
  const burstMultiplier = candidate.burst;

  if (options.normalizedRpm !== null && (typeof limitPerMin === 'number' || limitPerMin === null)) {
    merged.requests_per_minute = limitPerMin;
  }

  if (typeof burstMultiplier === 'number' || burstMultiplier === null) {
    const burstPerMinute =
      limitPerMin != null && burstMultiplier != null
        ? limitPerMin * burstMultiplier
        : null;

    if (options.normalizedRph !== null) {
      merged.requests_per_hour = burstPerMinute != null ? Math.round(burstPerMinute * 60) : null;
    }
    if (options.normalizedRpd !== null) {
      merged.requests_per_day = burstPerMinute != null ? Math.round(burstPerMinute * 1440) : null;
    }
  }

  return merged;
};

export default function RoleDetailPage() {
  const params = useParams();
  const router = useRouter();
  const confirm = useConfirm();
  const roleId = params.id as string;

  const [role, setRole] = useState<Role | null>(null);
  const [allPermissions, setAllPermissions] = useState<Permission[]>([]);
  const [rolePermissions, setRolePermissions] = useState<Set<number>>(new Set());
  const [roleUsers, setRoleUsers] = useState<User[]>([]);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [isDeletingRole, setIsDeletingRole] = useState(false);
  const [error, setError] = useState('');
  const [warning, setWarning] = useState('');
  const [success, setSuccess] = useState('');

  // Edit mode
  const [editMode, setEditMode] = useState(false);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');

  // Rate Limits
  const [rateLimits, setRateLimits] = useState<RateLimits | null>(null);
  const [editRpm, setEditRpm] = useState('');
  const [editRph, setEditRph] = useState('');
  const [editRpd, setEditRpd] = useState('');
  const [rateLimitsSaving, setRateLimitsSaving] = useState(false);

  // Tool Permissions
  const [toolPermissions, setToolPermissions] = useState<ToolPermission[]>([]);
  const [toolPermissionsLoading, setToolPermissionsLoading] = useState(false);
  const [toolPermissionsError, setToolPermissionsError] = useState('');
  const [newToolPrefix, setNewToolPrefix] = useState('');

  const hasRateLimits = Boolean(
    rateLimits
    && (rateLimits.requests_per_minute != null
      || rateLimits.requests_per_hour != null
      || rateLimits.requests_per_day != null)
  );

  useEffect(() => {
    if (!success) {
      return;
    }
    const timer = setTimeout(() => setSuccess(''), 2000);
    return () => clearTimeout(timer);
  }, [success]);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      setWarning('');
      setToolPermissionsError('');

      const [roleData, allPermsData, rolePermsData, usersData, toolPermsData] = await Promise.allSettled([
        api.getRole(roleId),
        api.getPermissions(),
        api.getRolePermissions(roleId),
        api.getRoleUsers(roleId),
        api.getRoleToolPermissions(roleId),
      ]);

      const getErrorMessage = (reason: unknown, fallback: string) =>
        reason instanceof Error && reason.message ? reason.message : fallback;
      const formatReason = (reason: unknown) => {
        if (reason instanceof Error && reason.message) {
          return `${reason.name}: ${reason.message}`;
        }
        if (typeof reason === 'string') {
          return reason;
        }
        try {
          return JSON.stringify(reason);
        } catch {
          return String(reason);
        }
      };

      if (roleData.status === 'rejected') {
        setError(getErrorMessage(roleData.reason, 'Failed to load role details'));
        return;
      }

      if (roleData.status === 'fulfilled') {
        const roleValue = roleData.value as Role & { rate_limits?: RateLimits };
        setRole(roleValue);
        setEditName(roleValue.name || '');
        setEditDescription(roleValue.description || '');
        // Set rate limits from role data if available
        if (roleValue.rate_limits) {
          setRateLimits(roleValue.rate_limits);
          setEditRpm(roleValue.rate_limits.requests_per_minute?.toString() || '');
          setEditRph(roleValue.rate_limits.requests_per_hour?.toString() || '');
          setEditRpd(roleValue.rate_limits.requests_per_day?.toString() || '');
        } else {
          setRateLimits(null);
          setEditRpm('');
          setEditRph('');
          setEditRpd('');
        }
      }

      const errors: string[] = [];

      if (allPermsData.status === 'fulfilled') {
        setAllPermissions(Array.isArray(allPermsData.value) ? allPermsData.value : []);
      } else {
        errors.push(getErrorMessage(allPermsData.reason, 'Failed to load permissions'));
      }

      if (rolePermsData.status === 'fulfilled') {
        const perms = Array.isArray(rolePermsData.value) ? rolePermsData.value : [];
        const permIds = new Set(perms.map((p: Permission) => p.id));
        setRolePermissions(permIds);
      } else {
        errors.push(getErrorMessage(rolePermsData.reason, 'Failed to load role permissions'));
      }

      if (usersData.status === 'fulfilled') {
        setRoleUsers(Array.isArray(usersData.value) ? usersData.value : []);
      } else {
        errors.push(getErrorMessage(usersData.reason, 'Failed to load role users'));
      }

      if (toolPermsData.status === 'fulfilled') {
        const tools = toolPermsData.value as { tools?: ToolPermission[]; items?: ToolPermission[] };
        setToolPermissions(Array.isArray(tools.tools) ? tools.tools : Array.isArray(tools.items) ? tools.items : []);
      } else {
        const reason = formatReason(toolPermsData.reason);
        setToolPermissionsError(reason);
        logger.warn('Failed to load tool permissions', { component: 'RoleDetailPage', error: toolPermsData.reason instanceof Error ? toolPermsData.reason.message : String(toolPermsData.reason) });
      }

      if (errors.length > 0) {
        setWarning(errors.join(' | '));
      }
    } catch (err: unknown) {
      logger.error('Failed to load role data', { component: 'RoleDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to load role data');
    } finally {
      setLoading(false);
    }
  }, [roleId]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  const handleTogglePermission = async (permissionId: number) => {
    if (role?.is_system) {
      setError('Cannot modify permissions for system roles');
      return;
    }

    const hasPermission = rolePermissions.has(permissionId);

    try {
      setError('');
      if (hasPermission) {
        await api.removePermissionFromRole(roleId, permissionId.toString());
        setRolePermissions((prev) => {
          const newSet = new Set(prev);
          newSet.delete(permissionId);
          return newSet;
        });
        setSuccess('Permission removed');
      } else {
        await api.assignPermissionToRole(roleId, permissionId.toString());
        setRolePermissions((prev) => {
          const newSet = new Set(prev);
          newSet.add(permissionId);
          return newSet;
        });
        setSuccess('Permission assigned');
      }
    } catch (err: unknown) {
      logger.error('Failed to update permission', { component: 'RoleDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to update permission');
    }
  };

  const handleSaveRole = async () => {
    if (!editName.trim()) {
      setError('Role name is required');
      return;
    }
    const trimmedDescription = editDescription.trim();
    if (trimmedDescription.length > MAX_DESCRIPTION_LENGTH) {
      setError(`Description must be ${MAX_DESCRIPTION_LENGTH} characters or less`);
      return;
    }

    try {
      setSaving(true);
      setError('');
      await api.updateRole(roleId, {
        name: editName.trim(),
        description: trimmedDescription || undefined,
      });
      setSuccess('Role updated successfully');
      setEditMode(false);
      await loadData();
    } catch (err: unknown) {
      logger.error('Failed to update role', { component: 'RoleDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to update role');
    } finally {
      setSaving(false);
    }
  };

  const handleCancelEdit = () => {
    setEditName(role?.name || '');
    setEditDescription(role?.description || '');
    setEditMode(false);
  };

  const handleDeleteRole = async () => {
    if (isDeletingRole) return;
    const confirmed = await confirm({
      title: 'Delete Role',
      message: `Delete role "${role?.name || 'this role'}"? This cannot be undone.`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setIsDeletingRole(true);
      await api.deleteRole(roleId);
      router.push('/roles');
    } catch (err: unknown) {
      setError(err instanceof Error && err.message ? err.message : 'Failed to delete role');
    } finally {
      setIsDeletingRole(false);
    }
  };

  const handleSaveRateLimits = async () => {
    try {
      setRateLimitsSaving(true);
      setError('');
      const data: RateLimits = {};
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
      const response = await api.setRoleRateLimits(roleId, payload);
      const nextRateLimits = mergeRateLimitsFromResponse(data, response, {
        normalizedRpm,
        normalizedRph,
        normalizedRpd,
      });
      setRateLimits(nextRateLimits);
      if (normalizedRpm !== null) {
        setEditRpm(nextRateLimits.requests_per_minute?.toString() || '');
      }
      if (normalizedRph !== null) {
        setEditRph(nextRateLimits.requests_per_hour?.toString() || '');
      }
      if (normalizedRpd !== null) {
        setEditRpd(nextRateLimits.requests_per_day?.toString() || '');
      }
      setSuccess('Rate limits updated');
    } catch (err: unknown) {
      logger.error('Failed to update rate limits', { component: 'RoleDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to update rate limits');
    } finally {
      setRateLimitsSaving(false);
    }
  };

  const handleClearRateLimits = async () => {
    const confirmed = await confirm({
      title: 'Clear Rate Limits',
      message: 'This will remove all custom rate limits for this role. Users will fall back to default limits.',
      confirmText: 'Clear',
      variant: 'danger',
    });
    if (!confirmed) return;

    try {
      setRateLimitsSaving(true);
      setError('');
      await api.clearRoleRateLimits(roleId);
      setRateLimits(null);
      setEditRpm('');
      setEditRph('');
      setEditRpd('');
      setSuccess('Rate limits cleared');
    } catch (err: unknown) {
      logger.error('Failed to clear rate limits', { component: 'RoleDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to clear rate limits');
    } finally {
      setRateLimitsSaving(false);
    }
  };

  const handleGrantToolsByPrefix = async () => {
    const prefix = newToolPrefix.trim();
    if (!prefix) {
      setError('Please enter a tool prefix (e.g., "read:" or "mcp:")');
      return;
    }

    try {
      setToolPermissionsLoading(true);
      setError('');
      await api.grantToolPermissionsByPrefix(roleId, prefix);
      setToolPermissionsError('');
      setSuccess(`Tool permissions granted for prefix "${prefix}"`);
      setNewToolPrefix('');
      // Reload tool permissions
      try {
        const toolPermsData = await api.getRoleToolPermissions(roleId);
        const tools = toolPermsData as { tools?: ToolPermission[]; items?: ToolPermission[] };
        setToolPermissions(Array.isArray(tools.tools) ? tools.tools : Array.isArray(tools.items) ? tools.items : []);
      } catch (err: unknown) {
        const message = err instanceof Error && err.message
          ? `Failed to reload tool permissions: ${err.message}`
          : 'Failed to reload tool permissions';
        setToolPermissionsError(message);
        setError(message);
      }
    } catch (err: unknown) {
      logger.error('Failed to grant tool permissions', { component: 'RoleDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to grant tool permissions');
    } finally {
      setToolPermissionsLoading(false);
    }
  };

  const handleRevokeToolsByPrefix = async () => {
    const prefix = newToolPrefix.trim();
    if (!prefix) {
      setError('Please enter a tool prefix to revoke (e.g., "read:" or "mcp:")');
      return;
    }

    const confirmed = await confirm({
      title: 'Revoke Tool Permissions',
      message: `Revoke all tool permissions matching prefix "${prefix}" from this role?`,
      confirmText: 'Revoke',
      variant: 'danger',
    });
    if (!confirmed) return;

    try {
      setToolPermissionsLoading(true);
      setError('');
      await api.revokeToolPermissionsByPrefix(roleId, prefix);
      setToolPermissionsError('');
      setSuccess(`Tool permissions revoked for prefix "${prefix}"`);
      setNewToolPrefix('');
      // Reload tool permissions
      try {
        const toolPermsData = await api.getRoleToolPermissions(roleId);
        const tools = toolPermsData as { tools?: ToolPermission[]; items?: ToolPermission[] };
        setToolPermissions(Array.isArray(tools.tools) ? tools.tools : Array.isArray(tools.items) ? tools.items : []);
      } catch (err: unknown) {
        const message = err instanceof Error && err.message
          ? `Failed to reload tool permissions: ${err.message}`
          : 'Failed to reload tool permissions';
        setToolPermissionsError(message);
        setError(message);
      }
    } catch (err: unknown) {
      logger.error('Failed to revoke tool permissions', { component: 'RoleDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to revoke tool permissions');
    } finally {
      setToolPermissionsLoading(false);
    }
  };

  // Group permissions by category (based on naming convention like "read:users", "write:media")
  const groupedPermissions = useMemo(
    () => allPermissions.reduce((acc, perm) => {
      const parts = perm.name.split(':');
      const category = parts.length > 1 ? parts[0] : 'general';
      if (!acc[category]) acc[category] = [];
      acc[category].push(perm);
      return acc;
    }, {} as Record<string, Permission[]>),
    [allPermissions]
  );

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          {loading ? (
            <FormSkeleton />
          ) : !role ? (
            <>
              <Alert variant="destructive">
                <AlertDescription>Role not found</AlertDescription>
              </Alert>
              <Button onClick={() => router.push('/roles')} className="mt-4">
                <ArrowLeft className="mr-2 h-4 w-4" />
                Back to Roles
              </Button>
            </>
          ) : (
            <>
              {/* Header */}
              <div className="mb-8 flex items-center justify-between">
                <div className="flex items-center gap-4">
                  <Button variant="ghost" onClick={() => router.push('/roles')}>
                    <ArrowLeft className="h-4 w-4" />
                  </Button>
                  <div>
                    <div className="flex items-center gap-3">
                      <Shield className="h-8 w-8 text-primary" />
                      {editMode ? (
                        <Input
                          value={editName}
                          onChange={(e) => setEditName(e.target.value)}
                          className="text-2xl font-bold h-auto py-1"
                          placeholder="Role name"
                        />
                      ) : (
                        <h1 className="text-3xl font-bold">{role.name}</h1>
                      )}
                      {role.is_system && (
                        <Badge variant="secondary">
                          <Lock className="mr-1 h-3 w-3" />
                          System
                        </Badge>
                      )}
                    </div>
                    {editMode ? (
                      <>
                        <Input
                          value={editDescription}
                          onChange={(e) => setEditDescription(e.target.value)}
                          className="mt-2 text-sm"
                          placeholder="Description (optional)"
                        />
                        <div className="text-xs text-muted-foreground mt-1">
                          {editDescription.length}/{MAX_DESCRIPTION_LENGTH} characters
                        </div>
                      </>
                    ) : (
                      <p className="text-muted-foreground mt-1">
                        {role.description || 'No description'}
                      </p>
                    )}
                  </div>
                </div>
                <div className="flex gap-2">
                  {editMode ? (
                    <>
                      <Button variant="outline" onClick={handleCancelEdit}>
                        <X className="mr-2 h-4 w-4" />
                        Cancel
                      </Button>
                      <Button onClick={handleSaveRole} disabled={saving} loading={saving} loadingText="Saving...">
                        <Save className="mr-2 h-4 w-4" />
                        Save
                      </Button>
                    </>
                  ) : (
                    <Button
                      variant="outline"
                      onClick={() => setEditMode(true)}
                      disabled={role.is_system}
                    >
                      Edit Role
                    </Button>
                  )}
                </div>
              </div>

              {error && (
                <Alert variant="destructive" className="mb-6">
                  <AlertDescription>{error}</AlertDescription>
                </Alert>
              )}

              {warning && (
                <Alert className="mb-6 border-yellow-200 bg-yellow-50 text-yellow-900 dark:border-yellow-800 dark:bg-yellow-900/20 dark:text-yellow-200">
                  <AlertDescription className="flex items-start justify-between gap-4">
                    <span>{warning}</span>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={() => setWarning('')}
                      aria-label="Dismiss warning"
                    >
                      <X className="h-4 w-4" />
                    </Button>
                  </AlertDescription>
                </Alert>
              )}

              {success && (
                <Alert className="mb-6 bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800">
                  <AlertDescription className="text-green-800 dark:text-green-200">
                    {success}
                  </AlertDescription>
                </Alert>
              )}

              <div className="grid gap-6 lg:grid-cols-3">
                {/* Permissions Section */}
                <Card className="lg:col-span-2">
                  <CardHeader>
                    <CardTitle>Permissions</CardTitle>
                    <CardDescription>
                      {rolePermissions.size} of {allPermissions.length} permissions assigned
                      {role.is_system && ' (read-only for system roles)'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {allPermissions.length === 0 ? (
                      <div className="text-center text-muted-foreground py-8">
                        No permissions defined in the system.
                        <Link href="/roles" className="block mt-2 text-primary hover:underline">
                          Create permissions first
                        </Link>
                      </div>
                    ) : Object.keys(groupedPermissions).length > 1 ? (
                      // Show grouped view if there are multiple categories
                      <div className="space-y-6">
                        {Object.entries(groupedPermissions).map(([category, perms]) => (
                          <div key={category}>
                            <h4 className="text-sm font-semibold text-muted-foreground uppercase mb-3">
                              {category}
                            </h4>
                            <div className="grid gap-2">
                              {perms.map((perm) => {
                                const isChecked = rolePermissions.has(perm.id);
                                return (
                                  <PermissionItem
                                    key={perm.id}
                                    perm={perm}
                                    isChecked={isChecked}
                                    onToggle={() => handleTogglePermission(perm.id)}
                                    disabled={role.is_system}
                                  />
                                );
                              })}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      // Show flat list if only one category
                      <div className="grid gap-2">
                        {allPermissions.map((perm) => {
                          const isChecked = rolePermissions.has(perm.id);
                          return (
                            <PermissionItem
                              key={perm.id}
                              perm={perm}
                              isChecked={isChecked}
                              onToggle={() => handleTogglePermission(perm.id)}
                              disabled={role.is_system}
                            />
                          );
                        })}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Users with this role */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Users className="h-5 w-5" />
                      Users
                    </CardTitle>
                    <CardDescription>
                      {roleUsers.length} user{roleUsers.length !== 1 ? 's' : ''} with this role
                    </CardDescription>
                  </CardHeader>
                  <CardContent>
                    {roleUsers.length === 0 ? (
                      <div className="text-center text-muted-foreground py-8">
                        No users have this role assigned.
                      </div>
                    ) : (
                      <div className="space-y-2">
                        {roleUsers.slice(0, USER_LIST_LIMIT).map((user) => (
                          <Link
                            key={user.id}
                            href={`/users/${user.id}`}
                            className="flex items-center gap-3 p-2 rounded-lg hover:bg-muted transition-colors"
                          >
                            <div className="h-8 w-8 rounded-full bg-primary/10 flex items-center justify-center">
                              <span className="text-xs font-medium">
                                {user.username?.charAt(0).toUpperCase() || '?'}
                              </span>
                            </div>
                            <div className="flex-1 min-w-0">
                              <div className="font-medium truncate">{user.username}</div>
                              <div className="text-xs text-muted-foreground truncate">
                                {user.email}
                              </div>
                            </div>
                          </Link>
                        ))}
                        {roleUsers.length > USER_LIST_LIMIT && (
                          <p className="text-sm text-muted-foreground text-center pt-2">
                            +{roleUsers.length - USER_LIST_LIMIT} more users
                          </p>
                        )}
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Rate Limits and Tool Permissions */}
              <div className="grid gap-6 lg:grid-cols-2 mt-6">
                {/* Rate Limits Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Clock className="h-5 w-5" />
                      Rate Limits
                    </CardTitle>
                    <CardDescription>
                      Set custom rate limits for users with this role
                      {role.is_system && ' (read-only for system roles)'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="grid gap-4">
                      <div className="space-y-1">
                        <Label htmlFor="rate-rpm">Requests per Minute</Label>
                        <Input
                          id="rate-rpm"
                          type="number"
                          min="0"
                          placeholder="e.g., 60"
                          value={editRpm}
                          onChange={(e) => setEditRpm(e.target.value)}
                          disabled={role.is_system}
                        />
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="rate-rph">Requests per Hour</Label>
                        <Input
                          id="rate-rph"
                          type="number"
                          min="0"
                          placeholder="e.g., 1000"
                          value={editRph}
                          onChange={(e) => setEditRph(e.target.value)}
                          disabled={role.is_system}
                        />
                      </div>
                      <div className="space-y-1">
                        <Label htmlFor="rate-rpd">Requests per Day</Label>
                        <Input
                          id="rate-rpd"
                          type="number"
                          min="0"
                          placeholder="e.g., 10000"
                          value={editRpd}
                          onChange={(e) => setEditRpd(e.target.value)}
                          disabled={role.is_system}
                        />
                      </div>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={handleSaveRateLimits}
                        disabled={rateLimitsSaving || role.is_system}
                        loading={rateLimitsSaving}
                        loadingText="Saving..."
                      >
                        Save Rate Limits
                      </Button>
                      {hasRateLimits && (
                        <Button
                          variant="outline"
                          onClick={handleClearRateLimits}
                          disabled={rateLimitsSaving || role.is_system}
                        >
                          Clear
                        </Button>
                      )}
                    </div>
                    {hasRateLimits && (
                      <div className="text-sm text-muted-foreground mt-2">
                        Current limits:{' '}
                        {rateLimits?.requests_per_minute != null && `${rateLimits.requests_per_minute}/min`}
                        {rateLimits?.requests_per_minute != null && rateLimits?.requests_per_hour != null && ', '}
                        {rateLimits?.requests_per_hour != null && `${rateLimits.requests_per_hour}/hr`}
                        {(rateLimits?.requests_per_minute != null || rateLimits?.requests_per_hour != null) && rateLimits?.requests_per_day != null && ', '}
                        {rateLimits?.requests_per_day != null && `${rateLimits.requests_per_day}/day`}
                      </div>
                    )}
                  </CardContent>
                </Card>

                {/* Tool Permissions Card */}
                <Card>
                  <CardHeader>
                    <CardTitle className="flex items-center gap-2">
                      <Wrench className="h-5 w-5" />
                      Tool Permissions
                    </CardTitle>
                    <CardDescription>
                      Manage MCP and API tool access for this role
                      {role.is_system && ' (read-only for system roles)'}
                    </CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="flex gap-2">
                      <Input
                        placeholder="Tool prefix (e.g., mcp:, read:, write:)"
                        value={newToolPrefix}
                        onChange={(e) => setNewToolPrefix(e.target.value)}
                        disabled={role.is_system || toolPermissionsLoading}
                      />
                      <Button
                        onClick={handleGrantToolsByPrefix}
                        disabled={role.is_system || toolPermissionsLoading || !newToolPrefix.trim()}
                      >
                        Grant
                      </Button>
                      <Button
                        variant="outline"
                        onClick={handleRevokeToolsByPrefix}
                        disabled={role.is_system || toolPermissionsLoading || !newToolPrefix.trim()}
                      >
                        Revoke
                      </Button>
                    </div>
                    {toolPermissionsError ? (
                      <Alert variant="destructive">
                        <AlertDescription>
                          Failed to load tool permissions: {toolPermissionsError}
                        </AlertDescription>
                      </Alert>
                    ) : toolPermissions.length > 0 ? (
                      <div className="max-h-48 overflow-y-auto space-y-1">
                        {toolPermissions.map((tool, index) => (
                          <div
                            key={tool.id || `tool-${index}`}
                            className={`flex items-center justify-between p-2 rounded text-sm ${
                              tool.granted ? 'bg-green-50 dark:bg-green-900/20' : 'bg-muted/30'
                            }`}
                          >
                            <code className="font-mono text-xs">{tool.tool_name}</code>
                            <Badge variant={tool.granted ? 'default' : 'secondary'}>
                              {tool.granted ? 'Granted' : 'Denied'}
                            </Badge>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-sm text-muted-foreground text-center py-4">
                        No tool permissions configured. Use prefix patterns to grant access.
                      </div>
                    )}
                  </CardContent>
                </Card>
              </div>

              {/* Quick Actions */}
              <Card className="mt-6">
                <CardHeader>
                  <CardTitle>Quick Actions</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex flex-wrap gap-4">
                    <Button
                      variant="outline"
                      onClick={() => router.push('/roles/matrix')}
                    >
                      View Permission Matrix
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => router.push('/roles/compare')}
                    >
                      Compare with...
                    </Button>
                    <Button
                      variant="outline"
                      onClick={() => router.push('/roles')}
                    >
                      Manage All Roles
                    </Button>
                    {!role.is_system && (
                      <Button
                        variant="outline"
                        className="text-red-500 hover:text-red-600"
                        onClick={handleDeleteRole}
                        disabled={isDeletingRole}
                        loading={isDeletingRole}
                        loadingText="Deleting..."
                      >
                        <Trash2 className="mr-2 h-4 w-4" />
                        Delete Role
                      </Button>
                    )}
                  </div>
                </CardContent>
              </Card>
            </>
          )}
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
