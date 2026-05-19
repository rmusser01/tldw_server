'use client';

import React, { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { ArrowLeft, ChevronDown, ChevronRight, RefreshCw, Search, Shield, Save, X } from 'lucide-react';
import { api } from '@/lib/api-client';
import { TableSkeleton } from '@/components/ui/skeleton';
import { Role, Permission } from '@/types';
import { CardSkeleton } from '@/components/ui/skeleton';
import { logger } from '@/lib/logger';

type RolePermissionMap = Record<number, Set<number>>;

export default function PermissionMatrixPage() {
  const router = useRouter();
  const [roles, setRoles] = useState<Role[]>([]);
  const [permissions, setPermissions] = useState<Permission[]>([]);
  const [rolePermissions, setRolePermissions] = useState<RolePermissionMap>({});
  const [pendingChanges, setPendingChanges] = useState<Record<string, boolean>>({});
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [searchQuery, setSearchQuery] = useState('');
  const [collapsedGroups, setCollapsedGroups] = useState<Set<string>>(new Set());
  const [showDifferencesOnly, setShowDifferencesOnly] = useState(false);
  const [permSearch, setPermSearch] = useState('');
  const [differencesOnly, setDifferencesOnly] = useState(false);

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError('');

      // Load roles and permissions
      const [rolesData, permsData] = await Promise.all([
        api.getRoles(),
        api.getPermissions(),
      ]);

      const rolesArray = Array.isArray(rolesData) ? rolesData : [];
      const permsArray = Array.isArray(permsData) ? permsData : [];

      setRoles(rolesArray);
      setPermissions(permsArray);

      // Load permissions for each role
      const permMap: RolePermissionMap = {};
      await Promise.all(
        rolesArray.map(async (role) => {
          try {
            const rolePerms = await api.getRolePermissions(role.id.toString());
            permMap[role.id] = new Set(
              (Array.isArray(rolePerms) ? rolePerms : []).map((p: Permission) => p.id)
            );
          } catch (err: unknown) {
            logger.warn(`Failed to load permissions for role ${role.id}`, { component: 'PermissionMatrixPage', error: err instanceof Error ? err.message : String(err) });
            permMap[role.id] = new Set();
          }
        })
      );

      setRolePermissions(permMap);
      setPendingChanges({});
    } catch (err: unknown) {
      logger.error('Failed to load data', { component: 'PermissionMatrixPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error ? err.message : 'Failed to load roles and permissions');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (!success) return;
    const timer = setTimeout(() => setSuccess(''), 2500);
    return () => clearTimeout(timer);
  }, [success]);

  const getCellKey = (roleId: number, permissionId: number): string => `${roleId}:${permissionId}`;

  const baseHasPermission = (roleId: number, permissionId: number): boolean => {
    return rolePermissions[roleId]?.has(permissionId) || false;
  };

  const hasPermission = (roleId: number, permissionId: number): boolean => {
    const key = getCellKey(roleId, permissionId);
    if (key in pendingChanges) {
      return pendingChanges[key];
    }
    return baseHasPermission(roleId, permissionId);
  };

  const pendingChangeCount = useMemo(() => Object.keys(pendingChanges).length, [pendingChanges]);

  /** Extract namespace prefix from permission name (e.g. "read:" from "read:users") */
  const getNamespace = (permName: string): string => {
    const colonIndex = permName.indexOf(':');
    return colonIndex > 0 ? permName.substring(0, colonIndex + 1) : 'other';
  };

  /** Check whether all roles have the same value for a given permission (considering pending changes) */
  const isUniformPermission = useCallback((perm: Permission): boolean => {
    if (roles.length <= 1) return true;
    const firstValue = hasPermission(roles[0].id, perm.id);
    return roles.every((role) => hasPermission(role.id, perm.id) === firstValue);
  }, [roles, rolePermissions, pendingChanges]); // eslint-disable-line react-hooks/exhaustive-deps

  /** Filtered permissions based on search and differences toggle */
  const filteredPermissions = useMemo(() => {
    let filtered = permissions;

    // Apply search filter
    if (searchQuery.trim()) {
      const query = searchQuery.toLowerCase();
      filtered = filtered.filter((perm) =>
        perm.name.toLowerCase().includes(query)
        || (perm.description && perm.description.toLowerCase().includes(query))
      );
    }

    // Apply differences-only filter
    if (showDifferencesOnly) {
      filtered = filtered.filter((perm) => !isUniformPermission(perm));
    }

    return filtered;
  }, [permissions, searchQuery, showDifferencesOnly, isUniformPermission]);

  /** Group filtered permissions by namespace */
  const groupedPermissions = useMemo(() => {
    const groups: { namespace: string; permissions: Permission[] }[] = [];
    const groupMap = new Map<string, Permission[]>();

    filteredPermissions.forEach((perm) => {
      const ns = getNamespace(perm.name);
      if (!groupMap.has(ns)) {
        groupMap.set(ns, []);
      }
      groupMap.get(ns)!.push(perm);
    });

    // Sort namespaces alphabetically, but "other" goes last
    const sortedKeys = Array.from(groupMap.keys()).sort((a, b) => {
      if (a === 'other') return 1;
      if (b === 'other') return -1;
      return a.localeCompare(b);
    });

    sortedKeys.forEach((ns) => {
      groups.push({ namespace: ns, permissions: groupMap.get(ns)! });
    });

    return groups;
  }, [filteredPermissions]);

  const toggleGroupCollapse = (namespace: string) => {
    setCollapsedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(namespace)) {
        next.delete(namespace);
      } else {
        next.add(namespace);
      }
      return next;
    });
  };

  const isDirtyCell = (roleId: number, permissionId: number): boolean => {
    return getCellKey(roleId, permissionId) in pendingChanges;
  };

  const togglePermission = (role: Role, permissionId: number) => {
    if (role.is_system || saving) return;

    const roleId = role.id;
    const key = getCellKey(roleId, permissionId);
    const baseline = baseHasPermission(roleId, permissionId);
    const nextValue = !hasPermission(roleId, permissionId);

    setPendingChanges((prev) => {
      if (nextValue === baseline) {
        if (!(key in prev)) return prev;
        const next = { ...prev };
        delete next[key];
        return next;
      }
      return {
        ...prev,
        [key]: nextValue,
      };
    });
  };

  const handleDiscardChanges = () => {
    if (pendingChangeCount === 0) return;
    setPendingChanges({});
    setSuccess('Unsaved changes discarded.');
  };

  const handleSaveChanges = async () => {
    const entries = Object.entries(pendingChanges);
    if (entries.length === 0) return;

    try {
      setSaving(true);
      setError('');

      for (const [cellKey, shouldHavePermission] of entries) {
        const [roleIdPart, permissionIdPart] = cellKey.split(':');
        const roleId = Number(roleIdPart);
        const permissionId = Number(permissionIdPart);
        if (!Number.isFinite(roleId) || !Number.isFinite(permissionId)) {
          continue;
        }
        const currentlyHasPermission = baseHasPermission(roleId, permissionId);
        if (shouldHavePermission === currentlyHasPermission) {
          continue;
        }
        if (shouldHavePermission) {
          await api.assignPermissionToRole(String(roleId), String(permissionId));
        } else {
          await api.removePermissionFromRole(String(roleId), String(permissionId));
        }
      }

      setRolePermissions((prev) => {
        const next: RolePermissionMap = {};
        Object.entries(prev).forEach(([roleId, permissionSet]) => {
          next[Number(roleId)] = new Set(permissionSet);
        });

        entries.forEach(([cellKey, shouldHavePermission]) => {
          const [roleIdPart, permissionIdPart] = cellKey.split(':');
          const roleId = Number(roleIdPart);
          const permissionId = Number(permissionIdPart);
          if (!Number.isFinite(roleId) || !Number.isFinite(permissionId)) {
            return;
          }
          if (!next[roleId]) {
            next[roleId] = new Set();
          }
          if (shouldHavePermission) {
            next[roleId].add(permissionId);
          } else {
            next[roleId].delete(permissionId);
          }
        });

        return next;
      });

      setPendingChanges({});
      setSuccess(`Saved ${entries.length} permission change${entries.length === 1 ? '' : 's'}.`);
    } catch (err: unknown) {
      logger.error('Failed to save matrix changes', { component: 'PermissionMatrixPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error ? err.message : 'Failed to save matrix changes');
    } finally {
      setSaving(false);
    }
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            {/* Header */}
            <div className="mb-8 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Button variant="ghost" onClick={() => router.push('/roles')}>
                  <ArrowLeft className="h-4 w-4" />
                </Button>
                <div>
                  <h1 className="text-3xl font-bold">Permission Matrix</h1>
                  <p className="text-muted-foreground">
                    Multi-edit role-permission assignments with batch save
                  </p>
                </div>
              </div>
              <div className="flex gap-2">
                <Button
                  variant="outline"
                  onClick={() => router.push('/roles/compare')}
                >
                  Compare Roles
                </Button>
                {pendingChangeCount > 0 && (
                  <Button
                    variant="outline"
                    onClick={handleDiscardChanges}
                    disabled={saving}
                  >
                    <X className="mr-2 h-4 w-4" />
                    Discard ({pendingChangeCount})
                  </Button>
                )}
                {pendingChangeCount > 0 && (
                  <Button
                    onClick={handleSaveChanges}
                    disabled={saving}
                    loading={saving}
                    loadingText="Saving..."
                  >
                    <Save className="mr-2 h-4 w-4" />
                    Save Changes
                  </Button>
                )}
                <Button
                  variant="outline"
                  onClick={loadData}
                  disabled={loading || saving}
                  loading={loading}
                  loadingText="Refreshing..."
                >
                  <RefreshCw className="mr-2 h-4 w-4" />
                  Refresh
                </Button>
              </div>
            </div>

            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {success && (
              <Alert className="mb-6 bg-green-50 border-green-200 dark:bg-green-900/20 dark:border-green-800">
                <AlertDescription className="text-green-800 dark:text-green-200">
                  {success}
                </AlertDescription>
              </Alert>
            )}

            {/* Info Card */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="flex items-start gap-4">
                  <Shield className="h-8 w-8 text-primary mt-1" />
                  <div>
                    <h3 className="font-semibold">Permission Matrix View</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Toggle cells to stage permission grants/removals, then save all changes in one batch.
                      Unsaved cells are highlighted. System roles remain read-only.
                    </p>
                    {pendingChangeCount > 0 && (
                      <p className="text-sm text-amber-700 dark:text-amber-300 mt-2 font-medium">
                        {pendingChangeCount} unsaved change{pendingChangeCount === 1 ? '' : 's'}.
                      </p>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Search & Filters */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="relative max-w-md w-full">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" aria-hidden="true" />
                    <label htmlFor="matrix-search" className="sr-only">
                      Search permissions by name
                    </label>
                    <Input
                      id="matrix-search"
                      placeholder="Search permissions..."
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  <div className="flex items-center gap-4">
                    <label className="flex items-center gap-2 text-sm cursor-pointer">
                      <Checkbox
                        checked={showDifferencesOnly}
                        onCheckedChange={(checked) => setShowDifferencesOnly(checked)}
                        aria-label="Show differences only"
                      />
                      <span className="text-muted-foreground">Show differences only</span>
                    </label>
                    {(searchQuery || showDifferencesOnly) && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={() => {
                          setSearchQuery('');
                          setShowDifferencesOnly(false);
                        }}
                      >
                        Clear filters
                      </Button>
                    )}
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Matrix Table */}
            <Card>
              <CardHeader>
                <CardTitle>Role-Permission Matrix</CardTitle>
                <CardDescription>
                  {filteredPermissions.length} of {permissions.length} permissions
                  {' '}&times; {roles.length} roles
                  {groupedPermissions.length > 1 && ` in ${groupedPermissions.length} groups`}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {loading ? (
                  <TableSkeleton rows={5} columns={5} />
                ) : roles.length === 0 || permissions.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    {roles.length === 0 ? 'No roles found. ' : ''}
                    {permissions.length === 0 ? 'No permissions found. ' : ''}
                    Create some first on the Roles page.
                  </div>
                ) : filteredPermissions.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    No permissions match your search
                    {showDifferencesOnly ? ' and difference filter' : ''}.
                  </div>
                ) : (
                  <>
                  <div className="flex items-center gap-3 mb-4">
                    <div className="relative flex-1 max-w-xs">
                      <Search className="absolute left-2.5 top-2.5 h-4 w-4 text-muted-foreground" />
                      <Input
                        placeholder="Filter permissions..."
                        value={permSearch}
                        onChange={(e) => setPermSearch(e.target.value)}
                        className="pl-9"
                        aria-label="Filter permissions"
                      />
                    </div>
                    <Button
                      variant={differencesOnly ? 'default' : 'outline'}
                      size="sm"
                      onClick={() => setDifferencesOnly(!differencesOnly)}
                      aria-pressed={differencesOnly}
                    >
                      Differences only
                    </Button>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="text-left p-3 bg-muted font-semibold sticky left-0 z-10 border-b">
                            Permission
                          </th>
                          {roles.map((role) => (
                            <th key={role.id} className="text-center p-3 bg-muted font-semibold border-b min-w-[100px]">
                              <div>
                                <span>{role.name}</span>
                                {role.is_system && (
                                  <Badge variant="secondary" className="ml-1 text-xs">System</Badge>
                                )}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {groupedPermissions.map((group) => {
                          const isCollapsed = collapsedGroups.has(group.namespace);
                          return (
                            <React.Fragment key={group.namespace}>
                              {/* Namespace group header */}
                              {groupedPermissions.length > 1 && (
                                <tr className="bg-muted/50">
                                  <td
                                    colSpan={1 + roles.length}
                                    className="p-2 border-b cursor-pointer select-none sticky left-0"
                                    onClick={() => toggleGroupCollapse(group.namespace)}
                                    role="button"
                                    aria-expanded={!isCollapsed}
                                    aria-label={`Toggle ${group.namespace} group`}
                                  >
                                    <div className="flex items-center gap-2">
                                      {isCollapsed ? (
                                        <ChevronRight className="h-4 w-4 text-muted-foreground" />
                                      ) : (
                                        <ChevronDown className="h-4 w-4 text-muted-foreground" />
                                      )}
                                      <span className="text-sm font-semibold">
                                        {group.namespace === 'other' ? 'Other' : group.namespace}
                                      </span>
                                      <Badge variant="secondary" className="text-xs">
                                        {group.permissions.length}
                                      </Badge>
                                    </div>
                                  </td>
                                </tr>
                              )}
                              {/* Permission rows (hidden when collapsed) */}
                              {!isCollapsed && group.permissions.map((perm, index) => (
                                <tr key={perm.id} className={index % 2 === 0 ? 'bg-background' : 'bg-muted/30'}>
                                  <td className="p-3 sticky left-0 bg-inherit border-b">
                                    <div>
                                      <code className="text-sm font-mono">{perm.name}</code>
                                      {perm.description && (
                                        <p className="text-xs text-muted-foreground">{perm.description}</p>
                                      )}
                                    </div>
                                  </td>
                                  {roles.map((role) => (
                                    <td
                                      key={`${role.id}-${perm.id}`}
                                      className={`p-3 text-center border-b ${
                                        isDirtyCell(role.id, perm.id)
                                          ? 'bg-amber-50 dark:bg-amber-900/20'
                                          : ''
                                      }`}
                                    >
                                      <Checkbox
                                        checked={hasPermission(role.id, perm.id)}
                                        onCheckedChange={() => togglePermission(role, perm.id)}
                                        aria-label={`Toggle ${perm.name} for ${role.name}`}
                                        disabled={role.is_system || saving}
                                        className="mx-auto"
                                      />
                                    </td>
                                  ))}
                                </tr>
                              ))}
                            </React.Fragment>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                  </>
                )}
              </CardContent>
            </Card>

            {/* Legend */}
            <Card className="mt-6">
              <CardContent className="pt-6">
                <div className="flex flex-wrap gap-6 text-sm">
                  <div className="flex items-center gap-2">
                    <Checkbox checked disabled className="opacity-50" />
                    <span className="text-muted-foreground">Permission granted</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Checkbox disabled className="opacity-50" />
                    <span className="text-muted-foreground">Permission not granted</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <Badge variant="secondary">System</Badge>
                    <span className="text-muted-foreground">System roles cannot be modified</span>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className="inline-block h-3 w-3 rounded-sm bg-amber-200 dark:bg-amber-700" />
                    <span className="text-muted-foreground">Unsaved matrix change</span>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
