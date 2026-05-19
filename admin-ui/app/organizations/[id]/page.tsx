'use client';

import { Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { useParams, useRouter, useSearchParams } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Checkbox } from '@/components/ui/checkbox';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Select } from '@/components/ui/select';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useConfirm } from '@/components/ui/confirm-dialog';
import {
  ArrowLeft, Building2, Users, UserPlus, Mail, Trash2, Key, Shield, Copy, Plus, Eye, EyeOff, ListChecks, Pencil, Search
} from 'lucide-react';
import { api } from '@/lib/api-client';
import { Organization, OrgMember, Team, ProviderSecret, User, WatchlistSettings, Subscription, OrgUsageSummary, Invoice } from '@/types';
import Link from 'next/link';
import { UserPicker } from '@/components/users/UserPicker';
import { PlanBadge } from '@/components/PlanBadge';
import { CardSkeleton, FormSkeleton } from '@/components/ui/skeleton';
import { UsageMeter } from '@/components/UsageMeter';
import { InvoiceTable } from '@/components/InvoiceTable';
import {
  formatBillingDate,
  isBillingEnabled,
  resolveOrganizationBillingSnapshot,
} from '@/lib/billing';
import { Pagination } from '@/components/ui/pagination';
import { logger } from '@/lib/logger';

const VALID_TABS = ['members', 'teams', 'keys', 'billing'] as const;
type OrgTab = (typeof VALID_TABS)[number];

function OrganizationDetailPageInner() {
  const params = useParams();
  const router = useRouter();
  const searchParams = useSearchParams();
  const confirm = useConfirm();
  const orgId = params.id as string;

  // Tab state from URL search param (?tab=members)
  const tabFromUrl = searchParams.get('tab') as OrgTab | null;
  const initialTab: OrgTab = tabFromUrl && VALID_TABS.includes(tabFromUrl) ? tabFromUrl : 'members';

  const handleTabChange = useCallback(
    (value: string) => {
      const next = new URLSearchParams(searchParams.toString());
      if (value === 'members') {
        next.delete('tab');
      } else {
        next.set('tab', value);
      }
      const qs = next.toString();
      router.replace(`/organizations/${orgId}${qs ? `?${qs}` : ''}`, { scroll: false });
    },
    [router, orgId, searchParams],
  );

  const [org, setOrg] = useState<Organization | null>(null);
  const [members, setMembers] = useState<OrgMember[]>([]);
  const [memberSearch, setMemberSearch] = useState('');
  const [memberRoleSelections, setMemberRoleSelections] = useState<Record<number, string>>({});
  const [teams, setTeams] = useState<Team[]>([]);
  const [byokKeys, setByokKeys] = useState<ProviderSecret[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const successTimerRef = useRef<number | null>(null);
  const loadRequestRef = useRef(0);

  // Add member dialog
  const [showAddMember, setShowAddMember] = useState(false);
  const [newMemberUserId, setNewMemberUserId] = useState('');
  const [newMemberRole, setNewMemberRole] = useState('member');
  const [selectedMember, setSelectedMember] = useState<User | null>(null);

  // Invite dialog
  const [showInvite, setShowInvite] = useState(false);
  const [inviteEmail, setInviteEmail] = useState('');
  const [inviteRole, setInviteRole] = useState('member');
  const [inviteLink, setInviteLink] = useState('');

  // BYOK dialog
  const [showAddByok, setShowAddByok] = useState(false);
  const [byokProvider, setByokProvider] = useState('');
  const [byokApiKey, setByokApiKey] = useState('');
  const [showByokApiKey, setShowByokApiKey] = useState(false);
  const [deletingByokProvider, setDeletingByokProvider] = useState<string | null>(null);

  // Billing / Subscription
  const [subscription, setSubscription] = useState<Subscription | null>(null);
  const [usageSummary, setUsageSummary] = useState<OrgUsageSummary | null>(null);
  const [invoices, setInvoices] = useState<Invoice[]>([]);

  // Watchlist Settings
  const [watchlistLoading, setWatchlistLoading] = useState(false);
  const [watchlistSaving, setWatchlistSaving] = useState(false);
  const [editWatchlistEnabled, setEditWatchlistEnabled] = useState(false);
  const [editWatchlistThreshold, setEditWatchlistThreshold] = useState('');
  const [editWatchlistAlertOnBreach, setEditWatchlistAlertOnBreach] = useState(false);
  const [showEditOrgDialog, setShowEditOrgDialog] = useState(false);
  const [editOrgName, setEditOrgName] = useState('');
  const [editOrgSlug, setEditOrgSlug] = useState('');
  const [editOrgError, setEditOrgError] = useState('');
  const [updatingOrg, setUpdatingOrg] = useState(false);
  const [deletingOrg, setDeletingOrg] = useState(false);

  const loadData = useCallback(async () => {
    const requestId = loadRequestRef.current + 1;
    loadRequestRef.current = requestId;
    const isCurrentRequest = () => loadRequestRef.current === requestId;

    setLoading(true);
    setError('');
    setSubscription(null);
    setUsageSummary(null);
    setInvoices([]);

    const [orgData, membersData, teamsData, byokData] = await Promise.allSettled([
      api.getOrganization(orgId),
      api.getOrgMembers(orgId),
      api.getTeams(orgId),
      api.getOrgByokKeys(orgId),
    ]);

    if (!isCurrentRequest()) {
      return;
    }

    if (orgData.status === 'fulfilled') {
      setOrg(orgData.value);
      setEditOrgName(orgData.value.name || '');
      setEditOrgSlug(orgData.value.slug || '');
      setEditOrgError('');
    } else {
      setError('Failed to load organization');
    }

    if (membersData.status === 'fulfilled') {
      const nextMembers = Array.isArray(membersData.value) ? membersData.value : [];
      setMembers(nextMembers);
      setMemberRoleSelections(() => {
        const nextSelections: Record<number, string> = {};
        nextMembers.forEach((member) => {
          nextSelections[member.user_id] = member.role;
        });
        return nextSelections;
      });
    }

    if (teamsData.status === 'fulfilled') {
      setTeams(Array.isArray(teamsData.value) ? teamsData.value : []);
    }

    if (byokData.status === 'fulfilled') {
      setByokKeys(Array.isArray(byokData.value) ? byokData.value : []);
    }

    // Load watchlist settings separately (may not be available)
    try {
      setWatchlistLoading(true);
      const watchData = await api.getOrgWatchlistSettings(orgId);
      if (!isCurrentRequest()) {
        return;
      }
      const settings = watchData as WatchlistSettings;
      setEditWatchlistEnabled(settings?.watchlists_enabled ?? false);
      setEditWatchlistThreshold(settings?.default_threshold?.toString() || '100');
      setEditWatchlistAlertOnBreach(settings?.alert_on_breach ?? true);
    } catch {
      if (!isCurrentRequest()) {
        return;
      }
      // Set defaults if endpoint unavailable
      setEditWatchlistEnabled(false);
      setEditWatchlistThreshold('100');
      setEditWatchlistAlertOnBreach(true);
    } finally {
      if (isCurrentRequest()) {
        setWatchlistLoading(false);
      }
    }

    // Load billing data when enabled
    if (isBillingEnabled()) {
      const [subResult, usageResult, invoiceResult] = await Promise.allSettled([
        api.getOrgSubscription(Number(orgId)),
        api.getOrgUsageSummary(Number(orgId)),
        api.getOrgInvoices(Number(orgId)),
      ]);
      if (!isCurrentRequest()) {
        return;
      }
      const billingSnapshot = resolveOrganizationBillingSnapshot(
        subResult,
        usageResult,
        invoiceResult
      );
      setSubscription(billingSnapshot.subscription);
      setUsageSummary(billingSnapshot.usageSummary);
      setInvoices(billingSnapshot.invoices);
    }

    if (isCurrentRequest()) {
      setLoading(false);
    }
  }, [orgId]);

  useEffect(() => {
    void loadData();
    return () => {
      loadRequestRef.current += 1;
    };
  }, [loadData]);

  useEffect(() => {
    return () => {
      if (successTimerRef.current !== null) {
        window.clearTimeout(successTimerRef.current);
        successTimerRef.current = null;
      }
    };
  }, []);

  const handleAddMember = async () => {
    if (!newMemberUserId) {
      setError('Select a user to add');
      return;
    }
    const userId = parseInt(newMemberUserId, 10);
    if (Number.isNaN(userId) || userId <= 0) {
      setError('User ID must be a valid positive number');
      return;
    }

    try {
      setError('');
      await api.addOrgMember(orgId, {
        user_id: userId,
        role: newMemberRole,
      });
      setSuccess('Member added successfully');
      setShowAddMember(false);
      setNewMemberUserId('');
      setNewMemberRole('member');
      setSelectedMember(null);
      void loadData();
    } catch (err: unknown) {
      logger.error('Failed to add member', { component: 'OrganizationDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to add member');
    }
  };

  const handleRemoveMember = async (userId: number, username?: string) => {
    const confirmed = await confirm({
      title: 'Remove Member',
      message: `Remove ${username || `user ${userId}`} from this organization?`,
      confirmText: 'Remove',
      variant: 'danger',
      icon: 'remove-user',
    });
    if (!confirmed) return;

    try {
      setError('');
      await api.removeOrgMember(orgId, userId.toString());
      setSuccess('Member removed successfully');
      void loadData();
    } catch (err: unknown) {
      logger.error('Failed to remove member', { component: 'OrganizationDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to remove member');
    }
  };

  const handleUpdateMemberRole = async (userId: number, newRole: string) => {
    try {
      setError('');
      await api.updateOrgMemberRole(orgId, userId.toString(), { role: newRole });
      setSuccess('Member role updated');
      void loadData();
      return true;
    } catch (err: unknown) {
      logger.error('Failed to update member role', { component: 'OrganizationDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to update member role');
      return false;
    }
  };

  const handleCreateInvite = async () => {
    if (!inviteEmail) {
      setError('Email is required');
      return;
    }
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    if (!emailRegex.test(inviteEmail)) {
      setError('Please enter a valid email address');
      return;
    }

    try {
      setError('');
      const result = await api.createOrgInvite(orgId, {
        email: inviteEmail,
        role: inviteRole,
      }) as { invite_url?: string; link?: string };
      setInviteLink(result.invite_url || result.link || 'Invite created - check email');
      setSuccess('Invite created successfully');
      setInviteEmail('');
      setInviteRole('member');
    } catch (err: unknown) {
      logger.error('Failed to create invite', { component: 'OrganizationDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to create invite');
    }
  };

  const handleAddByokKey = async () => {
    if (!byokProvider || !byokApiKey) {
      setError('Provider and API key are required');
      return;
    }

    try {
      setError('');
      await api.createOrgByokKey(orgId, {
        provider: byokProvider,
        api_key: byokApiKey,
      });
      setSuccess('Provider key added successfully');
      setShowAddByok(false);
      setByokProvider('');
      setByokApiKey('');
      setShowByokApiKey(false);
      void loadData();
    } catch (err: unknown) {
      logger.error('Failed to add BYOK key', { component: 'OrganizationDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to add provider key');
    }
  };

  const handleDeleteByokKey = async (provider: string) => {
    if (deletingByokProvider === provider) return;
    const confirmed = await confirm({
      title: 'Remove API Key',
      message: `Remove the API key for ${provider}?`,
      confirmText: 'Remove',
      variant: 'danger',
      icon: 'key',
    });
    if (!confirmed) return;

    try {
      setError('');
      setDeletingByokProvider(provider);
      await api.deleteOrgByokKey(orgId, provider);
      setSuccess('Provider key removed');
      void loadData();
    } catch (err: unknown) {
      logger.error('Failed to delete BYOK key', { component: 'OrganizationDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error && err.message ? err.message : 'Failed to delete provider key');
    } finally {
      setDeletingByokProvider((prev) => (prev === provider ? null : prev));
    }
  };

  const handleSaveWatchlistSettings = async () => {
    const threshold = parseInt(editWatchlistThreshold, 10);
    if (Number.isNaN(threshold) || threshold < 1) {
      setError('Threshold must be at least 1');
      return;
    }
    try {
      setWatchlistSaving(true);
      setError('');
      await api.updateOrgWatchlistSettings(orgId, {
        watchlists_enabled: editWatchlistEnabled,
        default_threshold: threshold,
        alert_on_breach: editWatchlistAlertOnBreach,
      });
      setSuccess('Watchlist settings saved');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to save watchlist settings';
      setError(message);
    } finally {
      setWatchlistSaving(false);
    }
  };

  const validateSlug = (slug: string): string | null => {
    if (!slug) return 'Slug is required.';
    if (!/^[a-z0-9]+(?:-[a-z0-9]+)*$/.test(slug)) {
      return 'Slug must be lowercase letters, numbers, and hyphens.';
    }
    return null;
  };

  const handleUpdateOrganization = async () => {
    const trimmedName = editOrgName.trim();
    const trimmedSlug = editOrgSlug.trim();
    if (!trimmedName) {
      setEditOrgError('Organization name is required.');
      return;
    }
    const slugError = validateSlug(trimmedSlug);
    if (slugError) {
      setEditOrgError(slugError);
      return;
    }
    try {
      setUpdatingOrg(true);
      setEditOrgError('');
      await api.updateOrganization(orgId, {
        name: trimmedName,
        slug: trimmedSlug,
      });
      setShowEditOrgDialog(false);
      setSuccess('Organization updated successfully');
      await loadData();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to update organization';
      setEditOrgError(message);
    } finally {
      setUpdatingOrg(false);
    }
  };

  const handleDeleteOrganization = async () => {
    if (!org) return;
    const memberCount = members.length;
    const confirmed = await confirm({
      title: 'Delete Organization',
      message: `Delete "${org.name}"? This organization has ${memberCount} member${memberCount === 1 ? '' : 's'}.`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;
    try {
      setDeletingOrg(true);
      await api.deleteOrganization(orgId);
      router.push('/organizations');
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'Failed to delete organization';
      setError(message);
    } finally {
      setDeletingOrg(false);
    }
  };

  const copyToClipboard = async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setSuccess('Copied to clipboard!');
      if (successTimerRef.current !== null) {
        window.clearTimeout(successTimerRef.current);
      }
      successTimerRef.current = window.setTimeout(() => {
        setSuccess('');
        successTimerRef.current = null;
      }, 2000);
    } catch (err: unknown) {
      logger.error('Failed to copy to clipboard', { component: 'OrganizationDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError('Failed to copy to clipboard');
    }
  };

  if (loading) {
    return (
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <CardSkeleton />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    );
  }

  if (!org) {
    return (
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <Alert variant="destructive">
              <AlertDescription>Organization not found</AlertDescription>
            </Alert>
            <Button onClick={() => router.push('/organizations')} className="mt-4">
              <ArrowLeft className="mr-2 h-4 w-4" />
              Back to Organizations
            </Button>
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    );
  }

  const filteredMembers = members.filter((member) => {
    if (!memberSearch) return true;
    const query = memberSearch.toLowerCase();
    return (
      member.user?.username?.toLowerCase().includes(query) ||
      member.user?.email?.toLowerCase().includes(query) ||
      String(member.user_id).includes(query)
    );
  });

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            {/* Header */}
            <div className="mb-8 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Button variant="ghost" onClick={() => router.push('/organizations')}>
                  <ArrowLeft className="h-4 w-4" />
                </Button>
                <div>
                  <div className="flex items-center gap-3">
                    <Building2 className="h-8 w-8 text-primary" />
                    <h1 className="text-3xl font-bold">{org.name}</h1>
                  </div>
                  <p className="text-muted-foreground mt-1">
                    <Badge variant="secondary">{org.slug}</Badge>
                    <span className="ml-3">Created {new Date(org.created_at).toLocaleDateString()}</span>
                  </p>
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  onClick={() => {
                    setShowEditOrgDialog(true);
                    setEditOrgError('');
                    setEditOrgName(org.name);
                    setEditOrgSlug(org.slug);
                  }}
                >
                  <Pencil className="mr-2 h-4 w-4" />
                  Edit Organization
                </Button>
                <Button
                  variant="outline"
                  onClick={handleDeleteOrganization}
                  disabled={deletingOrg}
                  loading={deletingOrg}
                  loadingText="Deleting..."
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete Organization
                </Button>
              </div>
            </div>

            <Dialog open={showEditOrgDialog} onOpenChange={setShowEditOrgDialog}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Edit Organization</DialogTitle>
                  <DialogDescription>
                    Update organization name and slug.
                  </DialogDescription>
                </DialogHeader>
                {editOrgError && (
                  <Alert variant="destructive">
                    <AlertDescription>{editOrgError}</AlertDescription>
                  </Alert>
                )}
                <div className="space-y-4 py-2">
                  <div className="space-y-2">
                    <Label htmlFor="editOrgName">Organization Name</Label>
                    <Input
                      id="editOrgName"
                      value={editOrgName}
                      onChange={(event) => setEditOrgName(event.target.value)}
                      placeholder="Organization name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="editOrgSlug">Slug</Label>
                    <Input
                      id="editOrgSlug"
                      value={editOrgSlug}
                      onChange={(event) => setEditOrgSlug(event.target.value)}
                      placeholder="organization-slug"
                    />
                    <p className="text-xs text-muted-foreground">
                      Slug must be unique across organizations.
                    </p>
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setShowEditOrgDialog(false)}>
                    Cancel
                  </Button>
                  <Button
                    onClick={handleUpdateOrganization}
                    disabled={updatingOrg}
                    loading={updatingOrg}
                    loadingText="Saving..."
                  >
                    Save Changes
                  </Button>
                </DialogFooter>
              </DialogContent>
            </Dialog>

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

            <Tabs defaultValue={initialTab} onValueChange={handleTabChange} className="space-y-4">
              <TabsList>
                <TabsTrigger value="members">
                  <Users className="mr-2 h-4 w-4" />
                  Members
                </TabsTrigger>
                <TabsTrigger value="teams">
                  <Shield className="mr-2 h-4 w-4" />
                  Teams
                </TabsTrigger>
                <TabsTrigger value="keys">
                  <Key className="mr-2 h-4 w-4" />
                  Keys &amp; Settings
                </TabsTrigger>
                {isBillingEnabled() && (
                  <TabsTrigger value="billing">
                    Billing
                  </TabsTrigger>
                )}
              </TabsList>

              <TabsContent value="members">
              {/* Members Section */}
              <Card>
                <CardHeader className="flex flex-row items-center justify-between">
                  <div>
                    <CardTitle className="flex items-center gap-2">
                      <Users className="h-5 w-5" />
                      Members
                    </CardTitle>
                    <CardDescription>
                      {members.length} member{members.length !== 1 ? 's' : ''}
                    </CardDescription>
                    <Label htmlFor="member-search" className="sr-only">
                      Search members
                    </Label>
                    <Input
                      id="member-search"
                      placeholder="Search members..."
                      value={memberSearch}
                      onChange={(e) => setMemberSearch(e.target.value)}
                      className="mt-2 max-w-xs"
                    />
                  </div>
                  <div className="flex gap-2">
                    <Dialog
                      open={showInvite}
                      onOpenChange={(nextOpen) => {
                        setShowInvite(nextOpen);
                        if (!nextOpen) {
                          setInviteEmail('');
                          setInviteRole('member');
                          setInviteLink('');
                        }
                      }}
                    >
                      <DialogTrigger asChild>
                        <Button variant="outline" size="sm">
                          <Mail className="mr-2 h-4 w-4" />
                          Invite
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Invite Member</DialogTitle>
                          <DialogDescription>
                            Send an email invitation to join this organization
                          </DialogDescription>
                        </DialogHeader>
                        <div className="space-y-4 py-4">
                          <div className="space-y-2">
                            <Label htmlFor="inviteEmail">Email Address</Label>
                            <Input
                              id="inviteEmail"
                              type="email"
                              placeholder="user@example.com"
                              value={inviteEmail}
                              onChange={(e) => setInviteEmail(e.target.value)}
                            />
                          </div>
                          <div className="space-y-2">
                            <Label htmlFor="inviteRole">Role</Label>
                            <Select
                              id="inviteRole"
                              value={inviteRole}
                              onChange={(e) => setInviteRole(e.target.value)}
                            >
                              <option value="member">Member</option>
                              <option value="admin">Admin</option>
                            </Select>
                          </div>
                          {inviteLink && (
                            <div className="space-y-2">
                              <Label>Invite Link</Label>
                              <div className="flex gap-2">
                                <Input value={inviteLink} readOnly className="font-mono text-sm" />
                                <Button
                                  variant="outline"
                                  size="icon"
                                  onClick={() => copyToClipboard(inviteLink)}
                                  aria-label="Copy invite link"
                                  title="Copy invite link"
                                >
                                  <Copy className="h-4 w-4" />
                                </Button>
                              </div>
                            </div>
                          )}
                        </div>
                        <DialogFooter>
                          <Button variant="outline" onClick={() => setShowInvite(false)}>
                            Close
                          </Button>
                          <Button onClick={handleCreateInvite}>Send Invite</Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>

                    <Dialog open={showAddMember} onOpenChange={setShowAddMember}>
                      <DialogTrigger asChild>
                        <Button size="sm">
                          <UserPlus className="mr-2 h-4 w-4" />
                          Add
                        </Button>
                      </DialogTrigger>
                      <DialogContent>
                        <DialogHeader>
                          <DialogTitle>Add Member</DialogTitle>
                          <DialogDescription>
                            Add an existing user to this organization
                          </DialogDescription>
                        </DialogHeader>
                        <div className="space-y-4 py-4">
                          <UserPicker
                            label="User"
                            value={selectedMember}
                            helperText="Search by username or email to add a member."
                            onSelect={(user) => {
                              setSelectedMember(user);
                              setNewMemberUserId(String(user.id));
                              setError('');
                            }}
                            onClear={() => {
                              setSelectedMember(null);
                              setNewMemberUserId('');
                            }}
                          />
                          <div className="space-y-2">
                            <Label htmlFor="memberRole">Role</Label>
                            <Select
                              id="memberRole"
                              value={newMemberRole}
                              onChange={(e) => setNewMemberRole(e.target.value)}
                            >
                              <option value="member">Member</option>
                              <option value="admin">Admin</option>
                              <option value="owner">Owner</option>
                            </Select>
                          </div>
                        </div>
                        <DialogFooter>
                          <Button variant="outline" onClick={() => setShowAddMember(false)}>Cancel</Button>
                          <Button onClick={handleAddMember}>Add Member</Button>
                        </DialogFooter>
                      </DialogContent>
                    </Dialog>
                  </div>
                </CardHeader>
                <CardContent>
                  {members.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8">
                      No members yet. Add or invite members to get started.
                    </div>
                  ) : filteredMembers.length === 0 ? (
                    <div className="text-center text-muted-foreground py-8">
                      No members match your search.
                    </div>
                  ) : (
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>User</TableHead>
                          <TableHead>Role</TableHead>
                          <TableHead>Joined</TableHead>
                          <TableHead className="text-right">Actions</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {filteredMembers.map((member) => (
                          <TableRow key={member.user_id}>
                            <TableCell>
                              <div>
                                <div className="font-medium">{member.user?.username || `User ${member.user_id}`}</div>
                                {member.user?.email && (
                                  <div className="text-xs text-muted-foreground">{member.user.email}</div>
                                )}
                              </div>
                            </TableCell>
                            <TableCell>
                              <Select
                                value={memberRoleSelections[member.user_id] ?? member.role}
                                onChange={async (event) => {
                                  const newRole = event.target.value;
                                  const previousRole = memberRoleSelections[member.user_id] ?? member.role;
                                  setMemberRoleSelections((prev) => ({
                                    ...prev,
                                    [member.user_id]: newRole,
                                  }));
                                  const displayName = member.user?.username || `User ${member.user_id}`;
                                  const confirmed = await confirm({
                                    title: 'Change Role',
                                    message: `Change role for ${displayName} to ${newRole}?`,
                                    confirmText: 'Change Role',
                                    variant: 'default',
                                  });
                                  if (confirmed) {
                                    const updated = await handleUpdateMemberRole(member.user_id, newRole);
                                    if (!updated) {
                                      setMemberRoleSelections((prev) => ({
                                        ...prev,
                                        [member.user_id]: previousRole,
                                      }));
                                    }
                                  } else {
                                    setMemberRoleSelections((prev) => ({
                                      ...prev,
                                      [member.user_id]: previousRole,
                                    }));
                                  }
                                }}
                                className="h-8"
                              >
                                <option value="member">Member</option>
                                <option value="admin">Admin</option>
                                <option value="owner">Owner</option>
                              </Select>
                            </TableCell>
                            <TableCell className="text-sm text-muted-foreground">
                              {new Date(member.joined_at).toLocaleDateString()}
                            </TableCell>
                            <TableCell className="text-right">
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleRemoveMember(member.user_id, member.user?.username)}
                                aria-label={`Remove ${member.user?.username || `User ${member.user_id}`} from organization`}
                                title="Remove member"
                              >
                                <Trash2 className="h-4 w-4 text-red-500" />
                              </Button>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  )}
                </CardContent>
              </Card>
              </TabsContent>

              {/* Teams Tab */}
              <TabsContent value="teams">
                <Card>
                  <CardHeader className="flex flex-row items-center justify-between">
                    <div>
                      <CardTitle className="flex items-center gap-2">
                        <Shield className="h-5 w-5" />
                        Teams
                      </CardTitle>
                      <CardDescription>
                        {teams.length} team{teams.length !== 1 ? 's' : ''}
                      </CardDescription>
                    </div>
                    <Link href={`/teams?org=${orgId}`}>
                      <Button size="sm">
                        <Plus className="mr-2 h-4 w-4" />
                        New Team
                      </Button>
                    </Link>
                  </CardHeader>
                  <CardContent>
                    {teams.length === 0 ? (
                      <div className="text-center text-muted-foreground py-8">
                        No teams in this organization yet.
                      </div>
                    ) : (
                      <Table>
                        <TableHeader>
                          <TableRow>
                            <TableHead>Name</TableHead>
                            <TableHead>Description</TableHead>
                            <TableHead className="text-right">Actions</TableHead>
                          </TableRow>
                        </TableHeader>
                        <TableBody>
                          {teams.map((team) => (
                            <TableRow key={team.id}>
                              <TableCell className="font-medium">{team.name}</TableCell>
                              <TableCell className="text-muted-foreground text-sm">
                                {team.description || '-'}
                              </TableCell>
                              <TableCell className="text-right">
                                <Link href={`/teams/${team.id}`}>
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    aria-label={`View team ${team.name}`}
                                    title={`View team ${team.name}`}
                                  >
                                    <Eye className="h-4 w-4" />
                                  </Button>
                                </Link>
                              </TableCell>
                            </TableRow>
                          ))}
                        </TableBody>
                      </Table>
                    )}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Keys & Settings Tab */}
              <TabsContent value="keys">
                <div className="space-y-6">
                  {/* BYOK Provider Keys Section */}
                  <Card>
                    <CardHeader className="flex flex-row items-center justify-between">
                      <div>
                        <CardTitle className="flex items-center gap-2">
                          <Key className="h-5 w-5" />
                          Provider API Keys (BYOK)
                        </CardTitle>
                        <CardDescription>
                          Organization-level API keys for LLM providers. These keys are shared by all members.
                        </CardDescription>
                      </div>
                        <Dialog
                          open={showAddByok}
                          onOpenChange={(nextOpen) => {
                            setShowAddByok(nextOpen);
                            if (!nextOpen) {
                              setShowByokApiKey(false);
                              setByokProvider('');
                              setByokApiKey('');
                            }
                          }}
                        >
                        <DialogTrigger asChild>
                          <Button size="sm">
                            <Plus className="mr-2 h-4 w-4" />
                            Add Key
                          </Button>
                        </DialogTrigger>
                        <DialogContent>
                          <DialogHeader>
                            <DialogTitle>Add Provider API Key</DialogTitle>
                            <DialogDescription>
                              Add an API key for an LLM provider. This key will be used for all organization members.
                            </DialogDescription>
                          </DialogHeader>
                          <div className="space-y-4 py-4">
                            <div className="space-y-2">
                              <Label htmlFor="byokProvider">Provider</Label>
                              <Select
                                id="byokProvider"
                                value={byokProvider}
                                onChange={(e) => setByokProvider(e.target.value)}
                              >
                                <option value="">Select provider...</option>
                                <option value="openai">OpenAI</option>
                                <option value="anthropic">Anthropic</option>
                                <option value="google">Google AI</option>
                                <option value="cohere">Cohere</option>
                                <option value="groq">Groq</option>
                                <option value="mistral">Mistral AI</option>
                                <option value="deepseek">DeepSeek</option>
                                <option value="openrouter">OpenRouter</option>
                              </Select>
                            </div>
                            <div className="space-y-2">
                              <Label htmlFor="byokApiKey">API Key</Label>
                              <div className="relative">
                                <Input
                                  id="byokApiKey"
                                  type={showByokApiKey ? 'text' : 'password'}
                                  placeholder="sk-..."
                                  value={byokApiKey}
                                  onChange={(e) => setByokApiKey(e.target.value)}
                                  className="pr-10"
                                />
                                <Button
                                  type="button"
                                  variant="ghost"
                                  size="icon"
                                  onClick={() => setShowByokApiKey((prev) => !prev)}
                                  className="absolute right-1 top-1/2 h-8 w-8 -translate-y-1/2"
                                  title={showByokApiKey ? 'Hide API key' : 'Show API key'}
                                  aria-label={showByokApiKey ? 'Hide API key' : 'Show API key'}
                                >
                                  {showByokApiKey ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                                </Button>
                              </div>
                              <p className="text-xs text-muted-foreground">
                                This key will be encrypted and stored securely.
                              </p>
                            </div>
                          </div>
                          <DialogFooter>
                            <Button variant="outline" onClick={() => setShowAddByok(false)}>Cancel</Button>
                            <Button onClick={handleAddByokKey}>Add Key</Button>
                          </DialogFooter>
                        </DialogContent>
                      </Dialog>
                    </CardHeader>
                    <CardContent>
                      {byokKeys.length === 0 ? (
                        <div className="text-center text-muted-foreground py-8">
                          No provider keys configured. Add keys to allow members to use their own API accounts.
                        </div>
                      ) : (
                        <Table>
                          <TableHeader>
                            <TableRow>
                              <TableHead>Provider</TableHead>
                              <TableHead>Added</TableHead>
                              <TableHead>Last Updated</TableHead>
                              <TableHead className="text-right">Actions</TableHead>
                            </TableRow>
                          </TableHeader>
                          <TableBody>
                            {byokKeys.map((key) => {
                              const isDeleting = deletingByokProvider === key.provider;
                              return (
                              <TableRow key={key.provider}>
                                <TableCell>
                                  <Badge variant="outline" className="capitalize">
                                    {key.provider}
                                  </Badge>
                                </TableCell>
                                <TableCell className="text-sm text-muted-foreground">
                                  {new Date(key.created_at).toLocaleDateString()}
                                </TableCell>
                                <TableCell className="text-sm text-muted-foreground">
                                  {new Date(key.updated_at).toLocaleDateString()}
                                </TableCell>
                                <TableCell className="text-right">
                                  <Button
                                    variant="ghost"
                                    size="sm"
                                    onClick={() => handleDeleteByokKey(key.provider)}
                                    disabled={isDeleting}
                                    title={isDeleting ? 'Removing API key' : 'Remove API key'}
                                    aria-label={isDeleting ? 'Removing API key' : 'Remove API key'}
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

                  {/* Watchlist Settings */}
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        <ListChecks className="h-5 w-5" />
                        Watchlist Settings
                      </CardTitle>
                      <CardDescription>
                        Configure usage watchlists and alerts for this organization
                      </CardDescription>
                    </CardHeader>
                    <CardContent>
                      {watchlistLoading ? (
                        <FormSkeleton />
                      ) : (
                        <div className="space-y-4">
                          <div className="grid gap-4 sm:grid-cols-3">
                            <div className="flex items-center justify-between p-3 border rounded-lg">
                              <div>
                                <Label htmlFor="watchlist-enabled">Enable Watchlists</Label>
                                <p className="text-xs text-muted-foreground">Track usage and spending</p>
                              </div>
                              <Checkbox
                                id="watchlist-enabled"
                                checked={editWatchlistEnabled}
                                onCheckedChange={setEditWatchlistEnabled}
                              />
                            </div>
                            <div className="space-y-1 p-3 border rounded-lg">
                              <Label htmlFor="watchlist-threshold">Default Threshold</Label>
                              <Input
                                id="watchlist-threshold"
                                type="number"
                                min="1"
                                value={editWatchlistThreshold}
                                onChange={(e) => setEditWatchlistThreshold(e.target.value)}
                                disabled={!editWatchlistEnabled}
                              />
                              <p className="text-xs text-muted-foreground">Usage limit before alerts</p>
                            </div>
                            <div className="flex items-center justify-between p-3 border rounded-lg">
                              <div>
                                <Label htmlFor="watchlist-alert">Alert on Breach</Label>
                                <p className="text-xs text-muted-foreground">Send notifications when exceeded</p>
                              </div>
                              <Checkbox
                                id="watchlist-alert"
                                checked={editWatchlistAlertOnBreach}
                                onCheckedChange={setEditWatchlistAlertOnBreach}
                                disabled={!editWatchlistEnabled}
                              />
                            </div>
                          </div>
                          <Button onClick={handleSaveWatchlistSettings} disabled={watchlistSaving} loading={watchlistSaving} loadingText="Saving...">
                            Save Settings
                          </Button>
                        </div>
                      )}
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              {/* Billing Tab */}
              {isBillingEnabled() && (
                <TabsContent value="billing">
                  <Card>
                    <CardHeader>
                      <CardTitle className="flex items-center gap-2">
                        Subscription
                        {subscription?.plan && <PlanBadge tier={subscription.plan.tier} />}
                      </CardTitle>
                    </CardHeader>
                    <CardContent className="space-y-4">
                      {subscription ? (
                        <>
                          <div className="grid grid-cols-2 gap-4 text-sm">
                            <div>
                              <p className="text-muted-foreground">Status</p>
                              <p className="font-medium capitalize">{subscription.status}</p>
                            </div>
                            <div>
                              <p className="text-muted-foreground">Current Period</p>
                              <p className="font-medium">
                                {formatBillingDate(subscription.current_period_start)} —{' '}
                                {formatBillingDate(subscription.current_period_end)}
                              </p>
                            </div>
                          </div>
                          {usageSummary && (
                            <div>
                              <p className="text-sm font-medium mb-2">Token Usage</p>
                              <UsageMeter
                                used={usageSummary.tokens_used}
                                included={usageSummary.tokens_included}
                                overageCostCents={usageSummary.overage_cost_cents}
                              />
                            </div>
                          )}
                          {invoices.length > 0 && (
                            <div>
                              <p className="text-sm font-medium mb-2">Invoices</p>
                              <InvoiceTable invoices={invoices} />
                            </div>
                          )}
                        </>
                      ) : (
                        <p className="text-muted-foreground">No active subscription.</p>
                      )}
                    </CardContent>
                  </Card>
                </TabsContent>
              )}
            </Tabs>
          </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Wrap with Suspense for useSearchParams
export default function OrganizationDetailPage() {
  return (
    <Suspense fallback={
      <PermissionGuard variant="route" requireAuth role="admin">
        <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            <CardSkeleton />
          </div>
        </ResponsiveLayout>
      </PermissionGuard>
    }>
      <OrganizationDetailPageInner />
    </Suspense>
  );
}
