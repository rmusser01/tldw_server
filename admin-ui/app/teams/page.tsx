'use client';

import { useCallback, useEffect, useState, Suspense } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { EmptyState } from '@/components/ui/empty-state';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { useToast } from '@/components/ui/toast';
import { Pagination } from '@/components/ui/pagination';
import { TableSkeleton } from '@/components/ui/skeleton';
import { Form, FormInput } from '@/components/ui/form';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { Checkbox } from '@/components/ui/checkbox';
import { Badge } from '@/components/ui/badge';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { Plus, Users, Building2, Search, Pencil, Trash2 } from 'lucide-react';
import { Team, Organization } from '@/types';
import { api } from '@/lib/api-client';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportTeams, ExportFormat } from '@/lib/export';
import Link from 'next/link';
import { useUrlPagination, useUrlState } from '@/lib/use-url-state';
import { logger } from '@/lib/logger';

const teamSchema = z.object({
  name: z.string().min(1, 'Team name is required'),
  description: z.string().optional(),
});

type TeamFormData = z.infer<typeof teamSchema>;

const ALL_ORGANIZATIONS_SENTINEL = '__all__';

function TeamsPageContent() {
  const [teams, setTeams] = useState<Team[]>([]);
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [selectedOrgId, setSelectedOrgId] = useUrlState<string>('org');
  const [searchQuery, setSearchQuery] = useUrlState<string>('q', { defaultValue: '' });
  const {
    page: currentPage,
    pageSize,
    setPage: setCurrentPage,
    setPageSize,
    resetPagination,
  } = useUrlPagination();
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [creatingTeam, setCreatingTeam] = useState(false);
  const [editingTeam, setEditingTeam] = useState<Team | null>(null);
  const [editName, setEditName] = useState('');
  const [editDescription, setEditDescription] = useState('');
  const [editError, setEditError] = useState('');
  const [updatingTeam, setUpdatingTeam] = useState(false);
  const [deletingTeamId, setDeletingTeamId] = useState<number | null>(null);
  const [selectedTeamIds, setSelectedTeamIds] = useState<Set<number>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const confirm = useConfirm();
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { warning, success, error: showError } = useToast();
  const isAllOrganizations = selectedOrgId === ALL_ORGANIZATIONS_SENTINEL;
  const scopedOrganizationId = isAllOrganizations ? undefined : selectedOrgId;
  const canBrowseTeams = Boolean(scopedOrganizationId || isAllOrganizations);
  const teamForm = useForm<TeamFormData>({
    resolver: zodResolver(teamSchema),
    defaultValues: {
      name: '',
      description: '',
    },
  });

  const loadOrganizations = useCallback(async () => {
    try {
      const data = await api.getOrganizations();
      const orgs = Array.isArray(data) ? data : [];
      setOrganizations(orgs);
    } catch (error) {
      logger.error('Failed to load organizations', { component: 'TeamsPage', error: error instanceof Error ? error.message : String(error) });
      setOrganizations([]);
    }
  }, []);

  const loadTeams = useCallback(async (orgId: string) => {
    try {
      setLoading(true);
      if (orgId === '__all__') {
        // Load teams from all organizations
        const results = await Promise.allSettled(
          organizations.map((org) => api.getTeams(String(org.id)))
        );
        const allTeams: Team[] = [];
        const failedOrgs: string[] = [];
        results.forEach((result, idx) => {
          if (result.status === 'fulfilled' && Array.isArray(result.value)) {
            allTeams.push(...result.value);
          } else if (result.status === 'rejected') {
            failedOrgs.push(String(organizations[idx]?.name ?? organizations[idx]?.id ?? idx));
          }
        });
        setTeams(allTeams);
        if (failedOrgs.length > 0) {
          warning(`Failed to load teams for: ${failedOrgs.join(', ')}`);
        }
      } else {
        const data = await api.getTeams(orgId);
        setTeams(Array.isArray(data) ? data : []);
      }
    } catch (error) {
      logger.error('Failed to load teams', { component: 'TeamsPage', error: error instanceof Error ? error.message : String(error) });
      setTeams([]);
    } finally {
      setLoading(false);
    }
  }, [organizations]);

  useEffect(() => {
    loadOrganizations();
  }, [loadOrganizations]);

  useEffect(() => {
    if (selectedOrgId === undefined && organizations.length > 0) {
      setSelectedOrgId(String(organizations[0].id));
    }
  }, [organizations, selectedOrgId, setSelectedOrgId]);

  useEffect(() => {
    if (!showCreateForm) {
      teamForm.reset();
    }
  }, [showCreateForm, teamForm]);

  useEffect(() => {
    if (scopedOrganizationId) {
      loadTeams(scopedOrganizationId);
    } else if (isAllOrganizations && organizations.length > 0) {
      // "All Organizations" mode: fetch teams for each org
      setLoading(true);
      Promise.allSettled(organizations.map(org => api.getTeams(String(org.id))))
        .then(results => {
          const allTeams = results.flatMap(r => r.status === 'fulfilled' && Array.isArray(r.value) ? r.value : []);
          setTeams(allTeams);
        })
        .finally(() => setLoading(false));
    } else {
      setTeams([]);
    }
  }, [isAllOrganizations, loadTeams, organizations, scopedOrganizationId]);

  const handleSubmit = teamForm.handleSubmit(async (data) => {
    if (!scopedOrganizationId) {
      warning('Select organization', 'Please select an organization first.');
      return;
    }
    try {
      setCreatingTeam(true);
      await api.createTeam(scopedOrganizationId, {
        name: data.name,
        description: data.description?.trim() || undefined,
      });
      setShowCreateForm(false);
      teamForm.reset();
      success('Team created', `${data.name} has been created.`);
      await loadTeams(scopedOrganizationId);
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Please try again.';
      logger.error('Failed to create team', { component: 'TeamsPage', error: error instanceof Error ? error.message : String(error) });
      showError('Failed to create team', message);
    } finally {
      setCreatingTeam(false);
    }
  });

  const handleSearchChange = (value: string) => {
    setSearchQuery(value || undefined);
    resetPagination();
  };

  const handleOrgChange = (value: string) => {
    setSelectedOrgId(value || undefined);
    resetPagination();
  };

  const openEditTeamDialog = (team: Team) => {
    setEditingTeam(team);
    setEditName(team.name || '');
    setEditDescription(team.description || '');
    setEditError('');
  };

  const closeEditTeamDialog = () => {
    setEditingTeam(null);
    setEditName('');
    setEditDescription('');
    setEditError('');
  };

  const handleUpdateTeam = async () => {
    const targetOrgId = isAllOrganizations ? String(editingTeam?.org_id ?? '') : scopedOrganizationId;
    if (!editingTeam || !targetOrgId) return;
    const trimmedName = editName.trim();
    if (!trimmedName) {
      setEditError('Team name is required.');
      return;
    }

    try {
      setUpdatingTeam(true);
      setEditError('');
      await api.updateTeam(targetOrgId, String(editingTeam.id), {
        name: trimmedName,
        description: editDescription.trim() || undefined,
      });
      success('Team updated', `${trimmedName} has been updated.`);
      closeEditTeamDialog();
      if (isAllOrganizations) {
        setTeams((currentTeams) =>
          currentTeams.map((team) =>
            team.id === editingTeam.id
              ? {
                  ...team,
                  name: trimmedName,
                  description: editDescription.trim() || undefined,
                }
              : team
          )
        );
      } else {
        await loadTeams(targetOrgId);
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Failed to update team.';
      setEditError(message);
      showError('Update team failed', message);
    } finally {
      setUpdatingTeam(false);
    }
  };

  const handleDeleteTeam = async (team: Team) => {
    const targetOrgId = isAllOrganizations ? String(team.org_id ?? '') : scopedOrganizationId;
    if (!targetOrgId) return;
    try {
      const membersResponse = await api.getTeamMembers(String(team.id));
      const memberCount = Array.isArray(membersResponse)
        ? membersResponse.length
        : (
          membersResponse &&
          typeof membersResponse === 'object' &&
          'items' in (membersResponse as Record<string, unknown>) &&
          Array.isArray((membersResponse as { items?: unknown[] }).items)
        )
          ? ((membersResponse as { items: unknown[] }).items).length
          : 0;
      const confirmed = await confirm({
        title: 'Delete team',
        message: `Delete "${team.name}"? This team has ${memberCount} member${memberCount === 1 ? '' : 's'}.`,
        confirmText: 'Delete',
        variant: 'danger',
        icon: 'delete',
      });
      if (!confirmed) return;

      setDeletingTeamId(team.id);
      await api.deleteTeam(targetOrgId, String(team.id));
      success('Team deleted', `${team.name} has been deleted.`);
      if (isAllOrganizations) {
        setTeams((currentTeams) => currentTeams.filter((currentTeam) => currentTeam.id !== team.id));
      } else {
        await loadTeams(targetOrgId);
      }
    } catch (error: unknown) {
      const message = error instanceof Error ? error.message : 'Failed to delete team.';
      showError('Delete team failed', message);
    } finally {
      setDeletingTeamId((current) => (current === team.id ? null : current));
    }
  };

  // Clear selection when team list changes
  useEffect(() => {
    setSelectedTeamIds((prev) => {
      if (prev.size === 0) return prev;
      const available = new Set(teams.map((t) => t.id));
      const next = new Set<number>();
      prev.forEach((id) => {
        if (available.has(id)) next.add(id);
      });
      return next;
    });
  }, [teams]);

  const handleToggleSelectTeam = (teamId: number, checked: boolean) => {
    setSelectedTeamIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(teamId);
      } else {
        next.delete(teamId);
      }
      return next;
    });
  };

  const handleToggleSelectAllVisibleTeams = (checked: boolean, visibleTeams: Team[]) => {
    setSelectedTeamIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        visibleTeams.forEach((t) => next.add(t.id));
      } else {
        visibleTeams.forEach((t) => next.delete(t.id));
      }
      return next;
    });
  };

  const handleClearTeamSelection = () => {
    setSelectedTeamIds(new Set());
  };

  const handleBulkDeleteTeams = async () => {
    if (!selectedOrgId || selectedOrgId === '__all__') return;
    const ids = Array.from(selectedTeamIds);
    if (ids.length === 0) return;
    const approval = await promptPrivilegedAction({
      title: 'Delete selected teams',
      message: `Delete ${ids.length} team${ids.length !== 1 ? 's' : ''}? This cannot be undone.`,
      confirmText: 'Delete',
      requirePassword: false,
    });
    if (!approval) return;

    try {
      setBulkDeleting(true);
      const results = await Promise.allSettled(
        ids.map((id) => api.deleteTeam(selectedOrgId, String(id)))
      );
      const failures = results.filter((r) => r.status === 'rejected').length;
      if (failures > 0) {
        showError(
          'Bulk delete incomplete',
          `${ids.length - failures} deleted, ${failures} failed.`
        );
      } else {
        success(
          'Teams deleted',
          `${ids.length} team${ids.length !== 1 ? 's' : ''} removed.`
        );
      }
      handleClearTeamSelection();
      await loadTeams(selectedOrgId);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete teams';
      showError('Bulk delete failed', message);
    } finally {
      setBulkDeleting(false);
    }
  };

  const selectedTeamCount = selectedTeamIds.size;

  const filteredTeams = teams.filter((team) => {
    if (!searchQuery) return true;
    const query = searchQuery.toLowerCase();
    return (
      team.name?.toLowerCase().includes(query) ||
      team.description?.toLowerCase().includes(query)
    );
  });

  const totalItems = filteredTeams.length;
  const totalPages = Math.ceil(totalItems / pageSize);
  const startIndex = (currentPage - 1) * pageSize;
  const paginatedTeams = filteredTeams.slice(startIndex, startIndex + pageSize);
  const allVisibleTeamsSelected = paginatedTeams.length > 0
    && paginatedTeams.every((t) => selectedTeamIds.has(t.id));

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8 flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">Teams</h1>
                <p className="text-muted-foreground">Manage teams within organizations</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <ExportMenu
                  onExport={(format: ExportFormat) => exportTeams(filteredTeams, format)}
                  disabled={filteredTeams.length === 0}
                />
                <Button
                  onClick={() => setShowCreateForm(!showCreateForm)}
                  disabled={!selectedOrgId || selectedOrgId === '__all__'}
                >
                  <Plus className="mr-2 h-4 w-4" />
                  New Team
                </Button>
              </div>
            </div>

            {/* Organization Selector */}
            <Card className="mb-6">
              <CardHeader>
                <CardTitle className="text-lg flex items-center gap-2">
                  <Building2 className="h-5 w-5" />
                  Select Organization
                </CardTitle>
              </CardHeader>
              <CardContent>
                {organizations.length === 0 ? (
                  <EmptyState
                    icon={Building2}
                    title="No organizations found"
                    description="Create an organization before creating teams."
                    actions={[
                      {
                        label: 'Go to organizations',
                        onClick: () => {
                          window.location.href = '/organizations';
                        },
                      },
                    ]}
                    className="py-6"
                  />
                ) : (
                  <Select
                    value={selectedOrgId ?? ''}
                    onChange={(e) => handleOrgChange(e.target.value)}
                    className="max-w-md"
                  >
                    <option value={ALL_ORGANIZATIONS_SENTINEL}>All Organizations</option>
                    {organizations.map((org) => (
                      <option key={org.id} value={org.id}>
                        {org.name} ({org.slug})
                      </option>
                    ))}
                  </Select>
                )}
              </CardContent>
            </Card>

            {showCreateForm && scopedOrganizationId && (
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle>Create Team</CardTitle>
                  <CardDescription>Add a new team to the selected organization</CardDescription>
                </CardHeader>
                <CardContent>
                  <FormProvider {...teamForm}>
                    <Form onSubmit={handleSubmit}>
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <FormInput<TeamFormData>
                          name="name"
                          label="Team Name"
                          placeholder="e.g., Engineering"
                          required
                        />
                        <FormInput<TeamFormData>
                          name="description"
                          label="Description"
                          placeholder="e.g., Core engineering team"
                        />
                      </div>

                      <div className="flex gap-2">
                        <Button type="submit" loading={creatingTeam} loadingText="Creating..." disabled={creatingTeam}>
                          Create Team
                        </Button>
                        <Button type="button" variant="outline" onClick={() => setShowCreateForm(false)}>
                          Cancel
                        </Button>
                      </div>
                    </Form>
                  </FormProvider>
                </CardContent>
              </Card>
            )}

            {/* Search */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="relative max-w-md">
                  <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
                  <Input
                    placeholder="Search teams by name or description..."
                    value={searchQuery || ''}
                    onChange={(e) => handleSearchChange(e.target.value)}
                    className="pl-10"
                    disabled={!canBrowseTeams}
                  />
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Teams List</CardTitle>
                <CardDescription>
                  {isAllOrganizations
                    ? 'Teams across all organizations'
                    : scopedOrganizationId
                      ? `Teams in ${organizations.find(o => String(o.id) === scopedOrganizationId)?.name || 'selected organization'}`
                      : 'Select an organization to view teams'}
                </CardDescription>
              </CardHeader>
              <CardContent>
                {!canBrowseTeams ? (
                  <EmptyState
                    icon={Users}
                    title="Select an organization"
                    description="Choose an organization above to view or manage teams."
                    actions={[
                      {
                        label: 'Use first organization',
                        onClick: () => {
                          if (organizations.length > 0) {
                            handleOrgChange(String(organizations[0].id));
                          }
                        },
                      },
                    ]}
                  />
                ) : loading ? (
                  <div className="py-4">
                    <TableSkeleton rows={4} columns={6} />
                  </div>
                ) : filteredTeams.length === 0 ? (
                  <EmptyState
                    icon={Users}
                    title={searchQuery ? 'No teams match your search.' : 'No teams found in this organization'}
                    description={
                      searchQuery
                        ? 'Adjust the query or clear search to see all teams.'
                        : 'Create a team to start organizing members.'
                    }
                    actions={[
                      searchQuery
                        ? {
                            label: 'Clear search',
                            onClick: () => handleSearchChange(''),
                          }
                        : {
                            label: 'Create team',
                            onClick: () => {
                              if (scopedOrganizationId) {
                                setShowCreateForm(true);
                              }
                            },
                          },
                    ]}
                  />
                ) : (
                  <>
                    {selectedTeamCount > 0 && (
                      <div className="mb-4 flex flex-col gap-3 rounded-md border bg-muted/20 p-4 sm:flex-row sm:items-center sm:justify-between">
                        <div className="flex flex-wrap items-center gap-2">
                          <Badge variant="outline">{selectedTeamCount} selected</Badge>
                          <span className="text-sm text-muted-foreground">
                            Bulk actions apply to selected teams.
                          </span>
                        </div>
                        <div className="flex flex-wrap gap-2">
                          <Button
                            variant="outline"
                            size="sm"
                            onClick={handleBulkDeleteTeams}
                            loading={bulkDeleting}
                            loadingText="Deleting..."
                            disabled={bulkDeleting}
                          >
                            <Trash2 className="mr-2 h-4 w-4 text-destructive" />
                            Delete
                          </Button>
                          <Button
                            variant="ghost"
                            size="sm"
                            onClick={handleClearTeamSelection}
                            disabled={bulkDeleting}
                          >
                            Clear selection
                          </Button>
                        </div>
                      </div>
                    )}
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-10">
                            <Checkbox
                              checked={allVisibleTeamsSelected}
                              onCheckedChange={(checked) => handleToggleSelectAllVisibleTeams(checked, paginatedTeams)}
                              aria-label="Select all visible teams"
                            />
                          </TableHead>
                          <TableHead>ID</TableHead>
                          <TableHead>Name</TableHead>
                          <TableHead>Description</TableHead>
                          <TableHead>Created</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {paginatedTeams.map((team) => (
                          <TableRow key={team.id}>
                            <TableCell>
                              <Checkbox
                                checked={selectedTeamIds.has(team.id)}
                                onCheckedChange={(checked) => handleToggleSelectTeam(team.id, checked)}
                                aria-label={`Select team ${team.name}`}
                              />
                            </TableCell>
                            <TableCell className="font-mono text-sm">{team.id}</TableCell>
                            <TableCell className="font-medium">{team.name}</TableCell>
                            <TableCell className="text-muted-foreground">
                              {team.description || '-'}
                            </TableCell>
                            <TableCell>{new Date(team.created_at).toLocaleDateString()}</TableCell>
                            <TableCell className="text-right">
                              <div className="flex justify-end gap-2">
                                <Link href={`/teams/${team.id}`}>
                                  <Button variant="outline" size="sm">
                                    <Users className="mr-2 h-4 w-4" />
                                    Manage
                                  </Button>
                                </Link>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => openEditTeamDialog(team)}
                                >
                                  <Pencil className="mr-2 h-4 w-4" />
                                  Edit
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleDeleteTeam(team)}
                                  disabled={deletingTeamId === team.id}
                                  loading={deletingTeamId === team.id}
                                  loadingText="Deleting..."
                                >
                                  <Trash2 className="mr-2 h-4 w-4" />
                                  Delete
                                </Button>
                              </div>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>

                    <Pagination
                      currentPage={currentPage}
                      totalPages={totalPages}
                      totalItems={totalItems}
                      pageSize={pageSize}
                      onPageChange={setCurrentPage}
                      onPageSizeChange={(size) => {
                        setPageSize(size);
                        resetPagination();
                      }}
                    />
                  </>
                )}
              </CardContent>
            </Card>

            <Dialog
              open={Boolean(editingTeam)}
              onOpenChange={(open) => {
                if (!open) closeEditTeamDialog();
              }}
            >
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Edit Team</DialogTitle>
                  <DialogDescription>
                    Update team name and description.
                  </DialogDescription>
                </DialogHeader>
                {editError && (
                  <p className="text-sm text-red-600">{editError}</p>
                )}
                <div className="space-y-4 py-2">
                  <div className="space-y-2">
                    <label htmlFor="edit-team-name" className="text-sm font-medium">Team Name</label>
                    <Input
                      id="edit-team-name"
                      value={editName}
                      onChange={(event) => setEditName(event.target.value)}
                      placeholder="Team name"
                    />
                  </div>
                  <div className="space-y-2">
                    <label htmlFor="edit-team-description" className="text-sm font-medium">Description</label>
                    <Input
                      id="edit-team-description"
                      value={editDescription}
                      onChange={(event) => setEditDescription(event.target.value)}
                      placeholder="Team description"
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={closeEditTeamDialog}>
                    Cancel
                  </Button>
                  <Button
                    onClick={handleUpdateTeam}
                    disabled={updatingTeam}
                    loading={updatingTeam}
                    loadingText="Saving..."
                  >
                    Save Changes
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>
          </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Wrap with Suspense for useSearchParams
export default function TeamsPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8">
              <div className="h-8 w-32 bg-muted rounded animate-pulse mb-2" />
              <div className="h-4 w-64 bg-muted rounded animate-pulse" />
            </div>
            <div className="h-96 bg-muted rounded animate-pulse" />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <TeamsPageContent />
    </Suspense>
  );
}
