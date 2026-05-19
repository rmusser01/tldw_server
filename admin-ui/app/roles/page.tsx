'use client';

import { useCallback, useEffect, useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useToast } from '@/components/ui/toast';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { TableSkeleton } from '@/components/ui/skeleton';
import { Form, FormInput } from '@/components/ui/form';
import { Shield, Plus, Trash2, Lock, Settings, ArrowLeftRight } from 'lucide-react';
import { api } from '@/lib/api-client';
import { Role, Permission } from '@/types';
import Link from 'next/link';
import { logger } from '@/lib/logger';

const roleSchema = z.object({
  name: z.string().min(1, 'Role name is required'),
  description: z.string().optional(),
});

const permissionSchema = z.object({
  name: z.string().min(1, 'Permission name is required'),
  description: z.string().optional(),
});

type RoleFormData = z.infer<typeof roleSchema>;
type PermissionFormData = z.infer<typeof permissionSchema>;

export default function RolesPage() {
  const [roles, setRoles] = useState<Role[]>([]);
  const [permissions, setPermissions] = useState<Permission[]>([]);
  const [loading, setLoading] = useState(true);
  const { success, error: showError } = useToast();
  const confirm = useConfirm();

  // Create role dialog
  const [showCreateRole, setShowCreateRole] = useState(false);
  const [creatingRole, setCreatingRole] = useState(false);
  const [deletingRoleIds, setDeletingRoleIds] = useState<Set<string>>(() => new Set());
  const roleForm = useForm<RoleFormData>({
    resolver: zodResolver(roleSchema),
    defaultValues: {
      name: '',
      description: '',
    },
  });

  // Create permission dialog
  const [showCreatePermission, setShowCreatePermission] = useState(false);
  const [creatingPermission, setCreatingPermission] = useState(false);
  const [deletingPermissionIds, setDeletingPermissionIds] = useState<Set<string>>(() => new Set());
  const permissionForm = useForm<PermissionFormData>({
    resolver: zodResolver(permissionSchema),
    defaultValues: {
      name: '',
      description: '',
    },
  });

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      const [rolesData, permsData] = await Promise.all([
        api.getRoles(),
        api.getPermissions(),
      ]);
      setRoles(Array.isArray(rolesData) ? rolesData : []);
      setPermissions(Array.isArray(permsData) ? permsData : []);
    } catch (err: unknown) {
      logger.error('Failed to load data', { component: 'RolesPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to load data', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setLoading(false);
    }
  }, [showError]);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    if (!showCreateRole) {
      roleForm.reset();
    }
  }, [roleForm, showCreateRole]);

  useEffect(() => {
    if (!showCreatePermission) {
      permissionForm.reset();
    }
  }, [permissionForm, showCreatePermission]);

  const handleCreateRole = roleForm.handleSubmit(async (data) => {
    try {
      setCreatingRole(true);
      await api.createRole({
        name: data.name.trim(),
        description: data.description?.trim() || undefined,
      });
      success('Role Created', `Role "${data.name}" has been created`);
      setShowCreateRole(false);
      roleForm.reset();
      loadData();
    } catch (err: unknown) {
      logger.error('Failed to create role', { component: 'RolesPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to create role', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setCreatingRole(false);
    }
  });

  const handleDeleteRole = async (role: Role) => {
    if (role.is_system) {
      showError('Cannot Delete', 'System roles cannot be deleted');
      return;
    }
    const roleId = role.id.toString();
    if (deletingRoleIds.has(roleId)) return;

    const confirmed = await confirm({
      title: 'Delete Role',
      message: `Are you sure you want to delete the role "${role.name}"? This action cannot be undone.`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setDeletingRoleIds((prev) => new Set(prev).add(roleId));
      await api.deleteRole(roleId);
      success('Role Deleted', `Role "${role.name}" has been deleted`);
      loadData();
    } catch (err: unknown) {
      logger.error('Failed to delete role', { component: 'RolesPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to delete role', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setDeletingRoleIds((prev) => {
        if (!prev.has(roleId)) return prev;
        const next = new Set(prev);
        next.delete(roleId);
        return next;
      });
    }
  };

  const handleCreatePermission = permissionForm.handleSubmit(async (data) => {
    try {
      setCreatingPermission(true);
      await api.createPermission({
        name: data.name.trim(),
        description: data.description?.trim() || undefined,
      });
      success('Permission Created', `Permission "${data.name}" has been created`);
      setShowCreatePermission(false);
      permissionForm.reset();
      loadData();
    } catch (err: unknown) {
      logger.error('Failed to create permission', { component: 'RolesPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to create permission', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setCreatingPermission(false);
    }
  });

  const handleDeletePermission = async (perm: Permission) => {
    const permissionId = perm.id.toString();
    if (deletingPermissionIds.has(permissionId)) return;
    const confirmed = await confirm({
      title: 'Delete Permission',
      message: `Are you sure you want to delete the permission "${perm.name}"?`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setDeletingPermissionIds((prev) => new Set(prev).add(permissionId));
      await api.deletePermission(permissionId);
      success('Permission Deleted', `Permission "${perm.name}" has been deleted`);
      loadData();
    } catch (err: unknown) {
      logger.error('Failed to delete permission', { component: 'RolesPage', error: err instanceof Error ? err.message : String(err) });
      showError('Failed to delete permission', err instanceof Error ? err.message : 'Please try again.');
    } finally {
      setDeletingPermissionIds((prev) => {
        if (!prev.has(permissionId)) return prev;
        const next = new Set(prev);
        next.delete(permissionId);
        return next;
      });
    }
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8">
              <h1 className="text-3xl font-bold">Roles & Permissions</h1>
              <p className="text-muted-foreground">
                Manage user roles and their associated permissions
              </p>
            </div>

            {/* Info Card */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="flex items-start gap-4">
                  <Shield className="h-8 w-8 text-primary mt-1" />
                  <div>
                    <h3 className="font-semibold">Role-Based Access Control (RBAC)</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Roles define what actions users can perform in the system. Each role can have
                      multiple permissions assigned to it. System roles cannot be deleted.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            <div className="grid gap-6 lg:grid-cols-2">
              {/* Roles Section */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <CardTitle>Roles</CardTitle>
                    <CardDescription>
                      {roles.length} role{roles.length !== 1 ? 's' : ''} defined
                    </CardDescription>
                  </div>
                  <Dialog open={showCreateRole} onOpenChange={setShowCreateRole}>
                    <DialogTrigger asChild>
                      <Button size="sm">
                        <Plus className="mr-2 h-4 w-4" />
                        Add Role
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Create New Role</DialogTitle>
                        <DialogDescription>
                          Add a new role to the system. You can assign permissions after creation.
                        </DialogDescription>
                      </DialogHeader>
                      <FormProvider {...roleForm}>
                        <Form onSubmit={handleCreateRole}>
                          <div className="space-y-4 py-4">
                            <FormInput<RoleFormData>
                              name="name"
                              label="Role Name"
                              placeholder="e.g., editor, viewer, moderator"
                              required
                            />
                            <FormInput<RoleFormData>
                              name="description"
                              label="Description (optional)"
                              placeholder="What this role is for..."
                            />
                          </div>
                          <DialogFooter>
                            <Button type="button" variant="outline" onClick={() => setShowCreateRole(false)}>Cancel</Button>
                            <Button type="submit" loading={creatingRole} loadingText="Creating...">
                              Create Role
                            </Button>
                          </DialogFooter>
                        </Form>
                      </FormProvider>
                    </DialogContent>
                  </Dialog>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="py-4">
                      <TableSkeleton rows={4} columns={3} />
                    </div>
                  ) : roles.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8">
                      No roles found. Create one to get started.
                    </div>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Role</TableHead>
                          <TableHead>Type</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {roles.map((role) => {
                          const roleId = role.id.toString();
                          const isDeleting = deletingRoleIds.has(roleId);
                          return (
                          <TableRow key={role.id}>
                            <TableCell>
                              <div>
                                <div className="font-medium">{role.name}</div>
                                {role.description && (
                                  <div className="text-xs text-muted-foreground">{role.description}</div>
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              {role.is_system ? (
                                <Badge variant="secondary">
                                  <Lock className="mr-1 h-3 w-3" />
                                  System
                                </Badge>
                              ) : (
                                <Badge variant="outline">Custom</Badge>
                              )}
                            </TableCell>
                            <TableCell className="text-right">
                              <div className="flex justify-end gap-1">
                                <Link href={`/roles/${role.id}`}>
                                  <Button
                                    variant="outline"
                                    size="sm"
                                    title="Manage role"
                                  >
                                    <Settings className="mr-2 h-4 w-4" />
                                    Manage
                                  </Button>
                                </Link>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  onClick={() => handleDeleteRole(role)}
                                  disabled={role.is_system || isDeleting}
                                  title={
                                    role.is_system
                                      ? 'Cannot delete system roles'
                                      : isDeleting
                                        ? 'Deleting role'
                                        : 'Delete role'
                                  }
                                  aria-label={
                                    role.is_system
                                      ? 'Cannot delete system roles'
                                      : isDeleting
                                        ? 'Deleting role'
                                        : 'Delete role'
                                  }
                                  loading={isDeleting}
                                  className={role.is_system ? 'text-muted-foreground' : 'text-red-500 hover:text-red-500'}
                                >
                                  <Trash2 className="h-4 w-4" />
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>

              {/* Permissions Section */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <CardTitle>Permissions</CardTitle>
                    <CardDescription>
                      {permissions.length} permission{permissions.length !== 1 ? 's' : ''} available
                    </CardDescription>
                  </div>
                  <Dialog open={showCreatePermission} onOpenChange={setShowCreatePermission}>
                    <DialogTrigger asChild>
                      <Button size="sm">
                        <Plus className="mr-2 h-4 w-4" />
                        Add Permission
                      </Button>
                    </DialogTrigger>
                    <DialogContent>
                      <DialogHeader>
                        <DialogTitle>Create New Permission</DialogTitle>
                        <DialogDescription>
                          Add a new permission that can be assigned to roles.
                        </DialogDescription>
                      </DialogHeader>
                      <FormProvider {...permissionForm}>
                        <Form onSubmit={handleCreatePermission}>
                          <div className="space-y-4 py-4">
                            <FormInput<PermissionFormData>
                              name="name"
                              label="Permission Name"
                              placeholder="e.g., read:users, write:media"
                              required
                            />
                            <FormInput<PermissionFormData>
                              name="description"
                              label="Description (optional)"
                              placeholder="What this permission allows..."
                            />
                          </div>
                          <DialogFooter>
                            <Button type="button" variant="outline" onClick={() => setShowCreatePermission(false)}>Cancel</Button>
                            <Button type="submit" loading={creatingPermission} loadingText="Creating...">
                              Create Permission
                            </Button>
                          </DialogFooter>
                        </Form>
                      </FormProvider>
                    </DialogContent>
                  </Dialog>
                </CardHeader>
                <CardContent>
                  {loading ? (
                    <div className="py-4">
                      <TableSkeleton rows={4} columns={2} />
                    </div>
                  ) : permissions.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8">
                      No permissions found. Create one to get started.
                    </div>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Permission</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {permissions.map((perm) => {
                          const permissionId = perm.id.toString();
                          const isDeleting = deletingPermissionIds.has(permissionId);
                          return (
                          <TableRow key={perm.id}>
                            <TableCell>
                              <div>
                                <div className="font-medium font-mono text-sm">{perm.name}</div>
                                {perm.description && (
                                  <div className="text-xs text-muted-foreground">{perm.description}</div>
                                )}
                              </div>
                            </TableCell>
                            <TableCell className="text-right">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleDeletePermission(perm)}
                                title={isDeleting ? 'Deleting permission' : 'Delete permission'}
                                aria-label={isDeleting ? 'Deleting permission' : 'Delete permission'}
                                disabled={isDeleting}
                                loading={isDeleting}
                                className="text-red-500 hover:text-red-500"
                              >
                                <Trash2 className="h-4 w-4" />
                              </Button>
                            </TableCell>
                          </TableRow>
                          );
                        })}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
            </div>

            {/* Permission Matrix Link */}
            <Card className="mt-6">
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Settings className="h-5 w-5" />
                  Advanced Configuration
                </CardTitle>
              </CardHeader>
              <CardContent>
                <p className="text-sm text-muted-foreground mb-4">
                  For advanced role-permission mapping and bulk assignment, use the permission matrix view.
                </p>
                <div className="flex gap-2">
                  <Link href="/roles/compare">
                    <Button variant="outline">
                      <ArrowLeftRight className="mr-2 h-4 w-4" />
                      Compare Roles
                    </Button>
                  </Link>
                  <Link href="/roles/matrix">
                    <Button variant="outline">
                      Open Permission Matrix
                    </Button>
                  </Link>
                </div>
              </CardContent>
            </Card>
          </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
