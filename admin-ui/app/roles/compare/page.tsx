'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { EmptyState } from '@/components/ui/empty-state';
import { ArrowLeft, Check, X } from 'lucide-react';
import { api } from '@/lib/api-client';
import { Role, Permission } from '@/types';
import { deriveComparisonCellState } from '@/lib/role-comparison';
import { logger } from '@/lib/logger';

type RolePermissionMap = Record<number, Set<number>>;

export default function RoleComparisonPage() {
  const router = useRouter();
  const [roles, setRoles] = useState<Role[]>([]);
  const [permissions, setPermissions] = useState<Permission[]>([]);
  const [rolePermissions, setRolePermissions] = useState<RolePermissionMap>({});
  const [selectedRoleIds, setSelectedRoleIds] = useState<number[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError('');

      const [rolesData, permissionsData] = await Promise.all([
        api.getRoles(),
        api.getPermissions(),
      ]);

      const roleItems = Array.isArray(rolesData) ? rolesData : [];
      const permissionItems = Array.isArray(permissionsData) ? permissionsData : [];
      setRoles(roleItems);
      setPermissions(permissionItems);

      const permissionMap: RolePermissionMap = {};
      await Promise.all(
        roleItems.map(async (role) => {
          try {
            const rolePerms = await api.getRolePermissions(String(role.id));
            permissionMap[role.id] = new Set(
              (Array.isArray(rolePerms) ? rolePerms : []).map((permission: Permission) => permission.id)
            );
          } catch (err: unknown) {
            logger.warn(`Failed to load permissions for role ${role.id}`, { component: 'RoleComparisonPage', error: err instanceof Error ? err.message : String(err) });
            permissionMap[role.id] = new Set();
          }
        })
      );
      setRolePermissions(permissionMap);
      setSelectedRoleIds(roleItems.slice(0, 2).map((role) => role.id));
    } catch (err: unknown) {
      logger.error('Failed to load role comparison data', { component: 'RoleComparisonPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error ? err.message : 'Failed to load role comparison data');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  const selectedRoles = useMemo(
    () => selectedRoleIds
      .map((roleId) => roles.find((role) => role.id === roleId))
      .filter((role): role is Role => Boolean(role)),
    [roles, selectedRoleIds]
  );

  const toggleRoleSelection = (roleId: number) => {
    setSelectedRoleIds((prev) => {
      const alreadySelected = prev.includes(roleId);
      if (alreadySelected) {
        if (prev.length <= 2) return prev;
        return prev.filter((id) => id !== roleId);
      }
      if (prev.length >= 3) return prev;
      return [...prev, roleId];
    });
  };

  const getCellStateClasses = (state: ReturnType<typeof deriveComparisonCellState>): string => {
    if (state === 'only-has') {
      return 'bg-emerald-50 text-emerald-800 dark:bg-emerald-900/30 dark:text-emerald-200';
    }
    if (state === 'only-missing') {
      return 'bg-rose-50 text-rose-800 dark:bg-rose-900/30 dark:text-rose-200';
    }
    return 'bg-background text-foreground';
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8">
          <div className="mb-8 flex items-center justify-between">
            <div className="flex items-center gap-4">
              <Button variant="ghost" onClick={() => router.push('/roles')}>
                <ArrowLeft className="h-4 w-4" />
              </Button>
              <div>
                <h1 className="text-3xl font-bold">Compare Roles</h1>
                <p className="text-muted-foreground">
                  Select 2-3 roles to compare permission differences side-by-side.
                </p>
              </div>
            </div>
          </div>

          {error && (
            <Alert variant="destructive" className="mb-6">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          <Card className="mb-6">
            <CardHeader>
              <CardTitle>Role Selection</CardTitle>
              <CardDescription>
                Select at least 2 roles (up to 3).
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-muted-foreground">Loading roles...</div>
              ) : roles.length === 0 ? (
                <EmptyState
                  title="No roles available for comparison."
                  description="Create roles first, then select two or three roles to compare."
                />
              ) : (
                <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
                  {roles.map((role) => {
                    const selected = selectedRoleIds.includes(role.id);
                    const disableSelection = !selected && selectedRoleIds.length >= 3;
                    return (
                      <label
                        key={role.id}
                        className={`flex items-center gap-3 rounded-lg border p-3 ${
                          selected ? 'border-primary bg-primary/5' : 'border-border'
                        }`}
                      >
                        <Checkbox
                          checked={selected}
                          onCheckedChange={() => toggleRoleSelection(role.id)}
                          disabled={disableSelection}
                          aria-label={`Select role ${role.name}`}
                        />
                        <span className="font-medium">{role.name}</span>
                        {role.is_system && <Badge variant="secondary">System</Badge>}
                      </label>
                    );
                  })}
                </div>
              )}
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle>Permission Comparison</CardTitle>
              <CardDescription>
                {selectedRoles.length < 2
                  ? 'Select at least two roles to render a comparison.'
                  : `${selectedRoles.length} roles x ${permissions.length} permissions`}
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="text-muted-foreground">Loading permission comparison...</div>
              ) : selectedRoles.length < 2 ? (
                <div className="text-muted-foreground">
                  Choose two or three roles above to compare.
                </div>
              ) : permissions.length === 0 ? (
                <EmptyState
                  title="No permissions available."
                  description="Create permissions to render a role-by-role comparison matrix."
                />
              ) : (
                <>
                  <div className="overflow-x-auto">
                    <table className="w-full border-collapse">
                      <thead>
                        <tr>
                          <th className="text-left p-3 bg-muted sticky left-0 z-10 border-b">Permission</th>
                          {selectedRoles.map((role) => (
                            <th key={role.id} className="text-center p-3 bg-muted border-b min-w-[120px]">
                              <div className="inline-flex items-center gap-2">
                                <span>{role.name}</span>
                                {role.is_system && <Badge variant="secondary">System</Badge>}
                              </div>
                            </th>
                          ))}
                        </tr>
                      </thead>
                      <tbody>
                        {permissions.map((permission, rowIndex) => {
                          const values = selectedRoles.map((role) =>
                            rolePermissions[role.id]?.has(permission.id) || false
                          );
                          return (
                            <tr
                              key={permission.id}
                              className={rowIndex % 2 === 0 ? 'bg-background' : 'bg-muted/20'}
                            >
                              <td className="p-3 sticky left-0 bg-inherit border-b">
                                <code className="text-sm font-mono">{permission.name}</code>
                              </td>
                              {selectedRoles.map((role, roleIndex) => {
                                const state = deriveComparisonCellState(values, roleIndex);
                                return (
                                  <td
                                    key={`${role.id}-${permission.id}`}
                                    data-testid={`compare-cell-${role.id}-${permission.id}`}
                                    className={`p-3 text-center border-b ${getCellStateClasses(state)}`}
                                  >
                                    {values[roleIndex] ? (
                                      <Check className="mx-auto h-4 w-4" />
                                    ) : (
                                      <X className="mx-auto h-4 w-4" />
                                    )}
                                  </td>
                                );
                              })}
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>

                  <div className="mt-4 flex flex-wrap items-center gap-4 text-sm">
                    <div className="inline-flex items-center gap-2">
                      <span className="inline-block h-3 w-3 rounded-sm bg-emerald-200 dark:bg-emerald-700" />
                      <span className="text-muted-foreground">Only selected role has permission</span>
                    </div>
                    <div className="inline-flex items-center gap-2">
                      <span className="inline-block h-3 w-3 rounded-sm bg-rose-200 dark:bg-rose-700" />
                      <span className="text-muted-foreground">Selected role missing permission</span>
                    </div>
                    <div className="inline-flex items-center gap-2">
                      <span className="inline-block h-3 w-3 rounded-sm bg-muted" />
                      <span className="text-muted-foreground">Shared state</span>
                    </div>
                  </div>
                </>
              )}
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
