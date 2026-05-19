'use client';

import { useCallback, useEffect, useMemo, useState, Suspense } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { z } from 'zod';
import { zodResolver } from '@hookform/resolvers/zod';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Checkbox } from '@/components/ui/checkbox';
import { EmptyState } from '@/components/ui/empty-state';
import { Pagination } from '@/components/ui/pagination';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { Select } from '@/components/ui/select';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import { Form, FormField } from '@/components/ui/form';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportOrganizations, ExportFormat } from '@/lib/export';
import { exportData } from '@/lib/export';
import { Plus, Eye, Search, BookmarkPlus, BookmarkX, Pencil, Trash2 } from 'lucide-react';
import type { Organization, PlanTier, Subscription } from '@/types';
import { api } from '@/lib/api-client';
import {
  EMPTY_BILLING_CELL_PLACEHOLDER,
  isBillingEnabled,
  type OrganizationPlanMap,
} from '@/lib/billing';
import { PlanBadge } from '@/components/PlanBadge';
import { TableSkeleton } from '@/components/ui/skeleton';
import Link from 'next/link';
import { useUrlPagination, useUrlState } from '@/lib/use-url-state';
import { getScopedItem, setScopedItem } from '@/lib/scoped-storage';
import { logger } from '@/lib/logger';

type SavedOrgView = {
  id: string;
  name: string;
  query: string;
};

const SAVED_VIEWS_STORAGE_KEY = 'admin_org_saved_views';

const organizationSchema = z.object({
  name: z.string().min(1, 'Organization name is required'),
  slug: z.string()
    .min(1, 'Slug is required')
    .regex(/^[a-z0-9]+(?:-[a-z0-9]+)*$/, 'Slug must be lowercase letters, numbers, and hyphens'),
});

type OrganizationFormData = z.infer<typeof organizationSchema>;

function OrganizationsPageContent() {
  const confirm = useConfirm();
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();
  const [organizations, setOrganizations] = useState<Organization[]>([]);
  const [showCreateForm, setShowCreateForm] = useState(false);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [createError, setCreateError] = useState('');
  const [creatingOrganization, setCreatingOrganization] = useState(false);
  const [editingOrganization, setEditingOrganization] = useState<Organization | null>(null);
  const [editName, setEditName] = useState('');
  const [editSlug, setEditSlug] = useState('');
  const [editError, setEditError] = useState('');
  const [updatingOrganization, setUpdatingOrganization] = useState(false);
  const [deletingOrganizationId, setDeletingOrganizationId] = useState<number | null>(null);
  const [selectedOrgIds, setSelectedOrgIds] = useState<Set<number>>(new Set());
  const [bulkDeleting, setBulkDeleting] = useState(false);
  const [totalItems, setTotalItems] = useState(0);
  const [orgPlans, setOrgPlans] = useState<OrganizationPlanMap>({});
  const [slugTouched, setSlugTouched] = useState(false);
  const [searchQuery, setSearchQuery] = useUrlState<string>('q', { defaultValue: '' });
  const [savedViews, setSavedViews] = useState<SavedOrgView[]>([]);
  const activeViewId = useMemo(() => {
    const match = savedViews.find((view) => view.query === (searchQuery || ''));
    return match ? match.id : '';
  }, [savedViews, searchQuery]);
  const [showSaveViewDialog, setShowSaveViewDialog] = useState(false);
  const [saveViewName, setSaveViewName] = useState('');
  const [saveViewError, setSaveViewError] = useState('');
  const organizationForm = useForm<OrganizationFormData>({
    resolver: zodResolver(organizationSchema),
    defaultValues: {
      name: '',
      slug: '',
    },
  });
  const {
    page: currentPage,
    pageSize,
    setPage: setCurrentPage,
    setPageSize,
    resetPagination,
  } = useUrlPagination();

  useEffect(() => {
    try {
      const stored = getScopedItem(SAVED_VIEWS_STORAGE_KEY);
      if (!stored) return;
      const parsed = JSON.parse(stored);
      if (Array.isArray(parsed)) {
        setSavedViews(parsed as SavedOrgView[]);
      }
    } catch (err) {
      logger.warn('Failed to load saved organization views', { component: 'OrganizationsPage', error: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  const persistSavedViews = useCallback((views: SavedOrgView[]) => {
    setSavedViews(views);
    try {
      setScopedItem(SAVED_VIEWS_STORAGE_KEY, JSON.stringify(views));
    } catch (err) {
      logger.warn('Failed to persist saved organization views', { component: 'OrganizationsPage', error: err instanceof Error ? err.message : String(err) });
    }
  }, []);

  const loadOrganizations = useCallback(async () => {
    try {
      setLoading(true);
      setLoadError('');
      const params: Record<string, string> = {
        limit: String(pageSize),
        offset: String((currentPage - 1) * pageSize),
      };
      if (searchQuery) {
        params.q = searchQuery;
      }
      const data = await api.getOrganizations(params);
      if (data && typeof data === 'object' && 'items' in data) {
        const items = Array.isArray((data as { items?: Organization[] }).items)
          ? (data as { items: Organization[] }).items
          : [];
        const total = (data as { total?: number }).total;
        setOrganizations(items);
        setTotalItems(typeof total === 'number' ? total : items.length);
      } else if (Array.isArray(data)) {
        setOrganizations(data);
        setTotalItems(data.length);
      } else {
        setOrganizations([]);
        setTotalItems(0);
      }
      if (isBillingEnabled()) {
        try {
          const subs = await api.getSubscriptions();
          const planMap: Record<number, { tier: PlanTier; status: string }> = {};
          if (Array.isArray(subs)) {
            subs.forEach((sub: Subscription) => {
              planMap[sub.org_id] = { tier: (sub.plan?.tier ?? 'free') as PlanTier, status: sub.status };
            });
          }
          setOrgPlans(planMap);
        } catch (err) {
          logger.warn('Failed to load subscription data for organizations', { component: 'OrganizationsPage', error: err instanceof Error ? err.message : String(err) });
        }
      }
    } catch (error: unknown) {
      setOrganizations([]);
      setTotalItems(0);
      setLoadError(
        error instanceof Error && error.message
          ? error.message
          : 'Failed to load organizations'
      );
    } finally {
      setLoading(false);
    }
  }, [currentPage, pageSize, searchQuery]);

  useEffect(() => {
    loadOrganizations();
  }, [loadOrganizations]);

  useEffect(() => {
    const visibleIds = new Set(organizations.map((org) => org.id));
    setSelectedOrgIds((prev) => {
      const next = new Set([...prev].filter((id) => visibleIds.has(id)));
      return next.size === prev.size ? prev : next;
    });
  }, [organizations]);

  useEffect(() => {
    if (!showCreateForm) {
      organizationForm.reset();
      setCreateError('');
      setSlugTouched(false);
    }
  }, [organizationForm, showCreateForm]);

  const handleSubmit = organizationForm.handleSubmit(async (data) => {
    setCreateError('');
    setCreatingOrganization(true);
    try {
      await api.createOrganization(data);
      setShowCreateForm(false);
      organizationForm.reset();
      setSlugTouched(false);
      loadOrganizations();
    } catch (error: unknown) {
      logger.error('Failed to create organization', { component: 'OrganizationsPage', error: error instanceof Error ? error.message : String(error) });
      const message =
        error instanceof Error && error.message
          ? error.message
          : 'Failed to create organization';
      setCreateError(message);
      showError('Create organization failed', message);
    } finally {
      setCreatingOrganization(false);
    }
  });

  const handleNameChange = (value: string) => {
    if (slugTouched) return;
    const slug = value
      .toLowerCase()
      .replace(/[^a-z0-9]+/g, '-')
      .replace(/^-|-$/g, '');
    organizationForm.setValue('slug', slug, { shouldDirty: true, shouldValidate: true });
  };

  const nameField = organizationForm.register('name');
  const slugField = organizationForm.register('slug');
  const nameError = organizationForm.formState.errors.name;
  const slugError = organizationForm.formState.errors.slug;

  const handleSearchChange = (value: string) => {
    setSearchQuery(value || undefined);
    resetPagination();
  };

  const handleApplySavedView = (viewId: string) => {
    if (!viewId) {
      setSearchQuery(undefined);
      resetPagination();
      return;
    }
    const view = savedViews.find((item) => item.id === viewId);
    if (!view) return;
    setSearchQuery(view.query || undefined);
    resetPagination();
  };

  const handleSaveView = () => {
    const name = saveViewName.trim();
    if (!name) {
      setSaveViewError('Provide a name for this view.');
      return;
    }
    const query = searchQuery || '';
    const newView: SavedOrgView = {
      id: `${Date.now()}`,
      name,
      query,
    };
    persistSavedViews([newView, ...savedViews]);
    setSaveViewName('');
    setSaveViewError('');
    setShowSaveViewDialog(false);
    success('Saved view', `${name} has been added.`);
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
    const next = savedViews.filter((item) => item.id !== activeViewId);
    persistSavedViews(next);
    success('Saved view removed', `"${view.name}" deleted.`);
  };

  const openEditOrganizationDialog = (organization: Organization) => {
    setEditingOrganization(organization);
    setEditName(organization.name || '');
    setEditSlug(organization.slug || '');
    setEditError('');
  };

  const closeEditOrganizationDialog = () => {
    setEditingOrganization(null);
    setEditName('');
    setEditSlug('');
    setEditError('');
  };

  const validateSlug = (slug: string): string | null => {
    if (!slug) return 'Slug is required.';
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
      return 'Slug must be lowercase letters, numbers, and hyphens.';
    }
    return null;
  };

  const handleUpdateOrganization = async () => {
    if (!editingOrganization) return;
    const trimmedName = editName.trim();
    const trimmedSlug = editSlug.trim();

    if (!trimmedName) {
      setEditError('Organization name is required.');
      return;
    }
    const slugValidationError = validateSlug(trimmedSlug);
    if (slugValidationError) {
      setEditError(slugValidationError);
      return;
    }

    try {
      setUpdatingOrganization(true);
      setEditError('');
      await api.updateOrganization(String(editingOrganization.id), {
        name: trimmedName,
        slug: trimmedSlug,
      });
      success('Organization updated', `${trimmedName} has been updated.`);
      closeEditOrganizationDialog();
      await loadOrganizations();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to update organization.';
      setEditError(message);
      showError('Update organization failed', message);
    } finally {
      setUpdatingOrganization(false);
    }
  };

  const handleDeleteOrganization = async (organization: Organization) => {
    try {
      const members = await api.getOrgMembers(String(organization.id));
      const memberCount = Array.isArray(members) ? members.length : 0;
      const approval = await promptPrivilegedAction({
        title: 'Delete organization',
        message: `Delete "${organization.name}"? This organization has ${memberCount} member${memberCount === 1 ? '' : 's'}.`,
        confirmText: 'Delete',
        requirePassword: true,
      });
      if (!approval) return;

      setDeletingOrganizationId(organization.id);
      await api.deleteOrganization(String(organization.id));
      success('Organization deleted', `${organization.name} has been deleted.`);
      await loadOrganizations();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete organization.';
      showError('Delete organization failed', message);
    } finally {
      setDeletingOrganizationId((current) => (current === organization.id ? null : current));
    }
  };

  // Clear selection when org list changes
  useEffect(() => {
    setSelectedOrgIds((prev) => {
      if (prev.size === 0) return prev;
      const available = new Set(organizations.map((org) => org.id));
      const next = new Set<number>();
      prev.forEach((id) => {
        if (available.has(id)) next.add(id);
      });
      return next;
    });
  }, [organizations]);

  const handleToggleSelectOrg = (orgId: number, checked: boolean) => {
    setSelectedOrgIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        next.add(orgId);
      } else {
        next.delete(orgId);
      }
      return next;
    });
  };

  const handleToggleSelectAllVisible = (checked: boolean) => {
    setSelectedOrgIds((prev) => {
      const next = new Set(prev);
      if (checked) {
        organizations.forEach((org) => next.add(org.id));
      } else {
        organizations.forEach((org) => next.delete(org.id));
      }
      return next;
    });
  };

  const handleClearSelection = () => {
    setSelectedOrgIds(new Set());
  };

  const handleBulkDeleteOrganizations = async () => {
    const ids = Array.from(selectedOrgIds);
    if (ids.length === 0) return;
    const approval = await promptPrivilegedAction({
      title: 'Delete selected organizations',
      message: `Delete ${ids.length} organization${ids.length !== 1 ? 's' : ''}? This cannot be undone.`,
      confirmText: 'Delete',
      requirePassword: false,
    });
    if (!approval) return;

    try {
      setBulkDeleting(true);
      const results = await Promise.allSettled(
        ids.map((id) => api.deleteOrganization(String(id)))
      );
      const failures = results.filter((r) => r.status === 'rejected').length;
      if (failures > 0) {
        showError(
          'Bulk delete incomplete',
          `${ids.length - failures} deleted, ${failures} failed.`
        );
      } else {
        success(
          'Organizations deleted',
          `${ids.length} organization${ids.length !== 1 ? 's' : ''} removed.`
        );
      }
      handleClearSelection();
      await loadOrganizations();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete organizations';
      showError('Bulk delete failed', message);
    } finally {
      setBulkDeleting(false);
    }
  };

  const selectedOrgCount = selectedOrgIds.size;
  const allVisibleSelected = organizations.length > 0
    && organizations.every((org) => selectedOrgIds.has(org.id));

  const totalPages = Math.ceil(totalItems / pageSize);
  const paginatedOrganizations = organizations;

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8 flex items-center justify-between">
              <div>
                <h1 className="text-3xl font-bold">Organizations</h1>
                <p className="text-muted-foreground">Manage organizations and their members</p>
              </div>
              <div className="flex flex-wrap gap-2">
                <ExportMenu
                  onExport={(format: ExportFormat) => exportOrganizations(organizations, format)}
                  disabled={organizations.length === 0}
                />
                <Button onClick={() => setShowCreateForm(!showCreateForm)}>
                  <Plus className="mr-2 h-4 w-4" />
                  New Organization
                </Button>
              </div>
            </div>

            {showCreateForm && (
              <Card className="mb-6">
                <CardHeader>
                  <CardTitle>Create Organization</CardTitle>
                  <CardDescription>Add a new organization to the system</CardDescription>
                </CardHeader>
                <CardContent>
                  <FormProvider {...organizationForm}>
                    <Form onSubmit={handleSubmit}>
                      {createError && (
                        <Alert variant="destructive">
                          <AlertDescription>{createError}</AlertDescription>
                        </Alert>
                      )}
                      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                        <FormField<OrganizationFormData> name="name" label="Organization Name" required>
                          <Input
                            id="name"
                            placeholder="e.g., Acme Corporation"
                            aria-invalid={nameError ? 'true' : undefined}
                            aria-describedby={nameError ? 'name-error' : undefined}
                            {...nameField}
                            onChange={(e) => {
                              nameField.onChange(e);
                              handleNameChange(e.target.value);
                            }}
                          />
                        </FormField>

                        <FormField<OrganizationFormData> name="slug" label="Slug" required>
                          <Input
                            id="slug"
                            placeholder="e.g., acme-corp"
                            aria-invalid={slugError ? 'true' : undefined}
                            aria-describedby={slugError ? 'slug-error' : undefined}
                            {...slugField}
                            onChange={(e) => {
                              slugField.onChange(e);
                              setSlugTouched(true);
                            }}
                          />
                          <p className="text-xs text-muted-foreground">
                            Unique URL-friendly identifier
                          </p>
                        </FormField>
                      </div>

                      <div className="flex gap-2">
                        <Button type="submit" loading={creatingOrganization} loadingText="Creating...">
                          Create Organization
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

            {loadError && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{loadError}</AlertDescription>
              </Alert>
            )}

            {/* Search */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="flex flex-col gap-4 sm:flex-row sm:items-center sm:justify-between">
                  <div className="relative max-w-md w-full">
                    <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" aria-hidden="true" />
                    <label htmlFor="orgs-search" className="sr-only">
                      Search organizations by name or slug
                    </label>
                    <Input
                      id="orgs-search"
                      placeholder="Search organizations by name or slug..."
                      value={searchQuery || ''}
                      onChange={(e) => handleSearchChange(e.target.value)}
                      className="pl-10"
                    />
                  </div>
                  <div className="flex flex-wrap items-center gap-2">
                    <Select
                      value={activeViewId}
                      onChange={(event) => handleApplySavedView(event.target.value)}
                      className="min-w-[200px]"
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
                        setSaveViewError('');
                        setSaveViewName('');
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
                          <Label htmlFor="saved-org-view-name">View name</Label>
                          <Input
                            id="saved-org-view-name"
                            value={saveViewName}
                            onChange={(event) => setSaveViewName(event.target.value)}
                            placeholder="e.g., Active orgs"
                          />
                          <p className="text-xs text-muted-foreground">
                            Current search: {searchQuery || 'All organizations'}
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
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Organizations List</CardTitle>
                <CardDescription>
                  {totalItems} organization{totalItems !== 1 ? 's' : ''} found
                </CardDescription>
              </CardHeader>
              <CardContent>
                {selectedOrgCount > 0 && (
                  <div className="mb-4 flex flex-col gap-3 rounded-md border bg-muted/20 p-4 sm:flex-row sm:items-center sm:justify-between">
                    <div className="flex flex-wrap items-center gap-2">
                      <Badge variant="outline">{selectedOrgCount} selected</Badge>
                      <span className="text-sm text-muted-foreground">
                        Bulk actions apply to selected organizations.
                      </span>
                    </div>
                    <div className="flex flex-wrap gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        onClick={handleBulkDeleteOrganizations}
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
                        onClick={handleClearSelection}
                        disabled={bulkDeleting}
                      >
                        Clear selection
                      </Button>
                    </div>
                  </div>
                )}
                {loading ? (
                  <div className="py-4">
                    <TableSkeleton rows={3} columns={isBillingEnabled() ? 8 : 6} />
                  </div>
                ) : organizations.length === 0 ? (
                  <EmptyState
                    icon={Plus}
                    title={searchQuery ? 'No organizations match your search' : 'No organizations found'}
                    description={
                      searchQuery
                        ? 'Try a different query or clear the search.'
                        : 'Create your first organization to get started.'
                    }
                    actions={[
                      searchQuery
                        ? {
                            label: 'Clear search',
                            onClick: () => {
                              setSearchQuery(undefined);
                              resetPagination();
                            },
                          }
                        : {
                            label: 'Create organization',
                            onClick: () => setShowCreateForm(true),
                          },
                    ]}
                  />
                ) : (
                  <>
                    {selectedOrgIds.size > 0 && (
                      <div className="mb-3 flex items-center gap-2 rounded-md border p-2">
                        <Badge variant="secondary">{selectedOrgIds.size} selected</Badge>
                        <Button
                          variant="destructive"
                          size="sm"
                          onClick={async () => {
                            const result = await promptPrivilegedAction({
                              title: 'Bulk Delete Organizations',
                              message: `Delete ${selectedOrgIds.size} organization(s)? This cannot be undone.`,
                              confirmText: 'Delete All',
                              requirePassword: true,
                            });
                            if (!result) return;
                            const ids = [...selectedOrgIds];
                            const results = await Promise.allSettled(ids.map(id => api.deleteOrganization(String(id))));
                            const okCount = results.filter(r => r.status === 'fulfilled').length;
                            const failCount = results.length - okCount;
                            setSelectedOrgIds(new Set());
                            await loadOrganizations();
                            if (okCount > 0) success('Bulk Delete', `${okCount} organization(s) deleted.`);
                            if (failCount > 0) showError('Partial Failure', `${failCount} organization(s) failed to delete.`);
                          }}
                        >
                          Delete Selected
                        </Button>
                        <Button variant="ghost" size="sm" onClick={() => setSelectedOrgIds(new Set())}>
                          Clear
                        </Button>
                      </div>
                    )}
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead className="w-10">
                            <Checkbox
                              checked={allVisibleSelected}
                              onCheckedChange={handleToggleSelectAllVisible}
                              aria-label="Select all visible organizations"
                            />
                          </TableHead>
                          <TableHead>ID</TableHead>
                          <TableHead>Name</TableHead>
                          <TableHead>Slug</TableHead>
                          <TableHead>Created</TableHead>
                          {isBillingEnabled() && <TableHead>Plan</TableHead>}
                          {isBillingEnabled() && <TableHead>Status</TableHead>}
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {paginatedOrganizations.map((org) => (
                          <TableRow key={org.id}>
                            <TableCell>
                              <Checkbox
                                checked={selectedOrgIds.has(org.id)}
                                onCheckedChange={(checked) => handleToggleSelectOrg(org.id, checked)}
                                aria-label={`Select organization ${org.name}`}
                              />
                            </TableCell>
                            <TableCell className="font-mono text-sm">{org.id}</TableCell>
                            <TableCell className="font-medium">{org.name}</TableCell>
                            <TableCell>
                              <Badge variant="secondary">{org.slug}</Badge>
                            </TableCell>
                            <TableCell>{new Date(org.created_at).toLocaleDateString()}</TableCell>
                            {isBillingEnabled() && (
                              <TableCell>
                                <PlanBadge tier={orgPlans[org.id]?.tier ?? 'free'} />
                              </TableCell>
                            )}
                            {isBillingEnabled() && (
                              <TableCell className="capitalize text-sm">
                                {orgPlans[org.id]?.status ?? EMPTY_BILLING_CELL_PLACEHOLDER}
                              </TableCell>
                            )}
                            <TableCell className="text-right">
                              <div className="flex justify-end gap-2">
                                <Link href={`/organizations/${org.id}`}>
                                  <Button variant="outline" size="sm">
                                    <Eye className="mr-2 h-4 w-4" />
                                    Manage
                                  </Button>
                                </Link>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => openEditOrganizationDialog(org)}
                                >
                                  <Pencil className="mr-2 h-4 w-4" />
                                  Edit
                                </Button>
                                <Button
                                  variant="outline"
                                  size="sm"
                                  onClick={() => handleDeleteOrganization(org)}
                                  disabled={deletingOrganizationId === org.id}
                                  loading={deletingOrganizationId === org.id}
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
              open={Boolean(editingOrganization)}
              onOpenChange={(open) => {
                if (!open) closeEditOrganizationDialog();
              }}
            >
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Edit Organization</DialogTitle>
                  <DialogDescription>
                    Update organization name and slug.
                  </DialogDescription>
                </DialogHeader>
                {editError && (
                  <Alert variant="destructive">
                    <AlertDescription>{editError}</AlertDescription>
                  </Alert>
                )}
                <div className="space-y-4 py-2">
                  <div className="space-y-2">
                    <Label htmlFor="edit-org-name">Organization Name</Label>
                    <Input
                      id="edit-org-name"
                      value={editName}
                      onChange={(event) => setEditName(event.target.value)}
                      placeholder="Organization name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="edit-org-slug">Slug</Label>
                    <Input
                      id="edit-org-slug"
                      value={editSlug}
                      onChange={(event) => setEditSlug(event.target.value)}
                      placeholder="organization-slug"
                    />
                    <p className="text-xs text-muted-foreground">
                      Slug must be unique across organizations.
                    </p>
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={closeEditOrganizationDialog}>
                    Cancel
                  </Button>
                  <Button
                    onClick={handleUpdateOrganization}
                    disabled={updatingOrganization}
                    loading={updatingOrganization}
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
export default function OrganizationsPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <div className="mb-8">
              <div className="h-8 w-48 bg-muted rounded animate-pulse mb-2" />
              <div className="h-4 w-64 bg-muted rounded animate-pulse" />
            </div>
            <div className="h-96 bg-muted rounded animate-pulse" />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <OrganizationsPageContent />
    </Suspense>
  );
}
