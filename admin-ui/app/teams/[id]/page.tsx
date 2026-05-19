'use client';

import { useCallback, useEffect, useState } from 'react';
import { useParams, useRouter } from 'next/navigation';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Button } from '@/components/ui/button';
import { Select } from '@/components/ui/select';
import { Label } from '@/components/ui/label';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Dialog, DialogContent, DialogDescription, DialogFooter, DialogHeader, DialogTitle, DialogTrigger } from '@/components/ui/dialog';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { Input } from '@/components/ui/input';
import { ArrowLeft, Users, UserPlus, Trash2, Shield, Building2, Pencil } from 'lucide-react';
import { api } from '@/lib/api-client';
import { Team, TeamMember, User } from '@/types';
import { CardSkeleton } from '@/components/ui/skeleton';
import Link from 'next/link';
import { UserPicker } from '@/components/users/UserPicker';
import { logger } from '@/lib/logger';

export default function TeamDetailPage() {
  const params = useParams();
  const router = useRouter();
  const confirm = useConfirm();
  const teamId = typeof params.id === 'string' ? params.id : params.id?.[0] ?? '';

  const [team, setTeam] = useState<Team | null>(null);
  const [members, setMembers] = useState<TeamMember[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');

  // Add member dialog
  const [showAddMember, setShowAddMember] = useState(false);
  const [newMemberUserId, setNewMemberUserId] = useState('');
  const [newMemberRole, setNewMemberRole] = useState('member');
  const [selectedMember, setSelectedMember] = useState<User | null>(null);
  const [addingMember, setAddingMember] = useState(false);
  const [removingMemberId, setRemovingMemberId] = useState<number | null>(null);
  const [memberRoleEdits, setMemberRoleEdits] = useState<Record<number, string>>({});
  const [updatingMemberRoleId, setUpdatingMemberRoleId] = useState<number | null>(null);
  const [showEditTeam, setShowEditTeam] = useState(false);
  const [editTeamName, setEditTeamName] = useState('');
  const [editTeamDescription, setEditTeamDescription] = useState('');
  const [editTeamError, setEditTeamError] = useState('');
  const [updatingTeam, setUpdatingTeam] = useState(false);
  const [deletingTeam, setDeletingTeam] = useState(false);

  const formatJoinedAt = (value?: string | null) => {
    if (!value) {
      return 'N/A';
    }
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) {
      return 'N/A';
    }
    return date.toLocaleDateString();
  };

  const isTeamResponse = (value: unknown): value is Team => {
    if (!value || typeof value !== 'object' || Array.isArray(value)) {
      return false;
    }
    const teamValue = value as Record<string, unknown>;
    return typeof teamValue.id === 'number'
      && typeof teamValue.org_id === 'number'
      && typeof teamValue.name === 'string';
  };

  const loadData = useCallback(async () => {
    try {
      setLoading(true);
      setError('');
      setTeam(null);

      const [teamData, membersData] = await Promise.all([
        api.getTeam(teamId),
        api.getTeamMembers(teamId),
      ]);

      if (!isTeamResponse(teamData)) {
        throw new Error('Unexpected team response');
      }

      if (!Array.isArray(membersData)) {
        throw new Error('Unexpected team members response');
      }
      setTeam(teamData);
      setEditTeamName(teamData.name || '');
      setEditTeamDescription(teamData.description || '');
      setEditTeamError('');
      setMembers(membersData);
      setMemberRoleEdits(
        membersData.reduce((acc, member) => {
          acc[member.user_id] = member.role;
          return acc;
        }, {} as Record<number, string>)
      );
    } catch (err: unknown) {
      logger.error('Failed to load team data', { component: 'TeamDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(err instanceof Error ? err.message : 'Failed to load team data');
    } finally {
      setLoading(false);
    }
  }, [teamId]);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  useEffect(() => {
    if (!success) {
      return;
    }

    const timer = setTimeout(() => setSuccess(''), 3000);
    return () => clearTimeout(timer);
  }, [success]);

  const handleAddMember = async () => {
    if (!newMemberUserId) {
      setError('Select a user to add');
      return;
    }

    setError('');
    try {
      const userId = parseInt(newMemberUserId, 10);
      if (Number.isNaN(userId)) {
        setError('Please enter a valid numeric User ID');
        return;
      }
      if (members.some((member) => member.user_id === userId)) {
        setError('This user is already a member of the team');
        return;
      }
      setAddingMember(true);
      await api.addTeamMember(teamId, {
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
      const msg = err instanceof Error ? err.message : String(err ?? 'Failed to add member');
      logger.error('Failed to add member', { component: 'TeamDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(msg);
    } finally {
      setAddingMember(false);
    }
  };

  const handleRemoveMember = async (userId: number, username?: string) => {
    const confirmed = await confirm({
      title: 'Remove Team Member',
      message: `Remove ${username || `user ${userId}`} from this team?`,
      confirmText: 'Remove',
      variant: 'danger',
      icon: 'remove-user',
    });
    if (!confirmed) return;

    try {
      setError('');
      setRemovingMemberId(userId);
      await api.removeTeamMember(teamId, userId);
      setSuccess('Member removed successfully');
      void loadData();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to remove member';
      logger.error('Failed to remove member', { component: 'TeamDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(msg);
    } finally {
      setRemovingMemberId((current) => (current === userId ? null : current));
    }
  };

  const handleUpdateMemberRole = async (member: TeamMember) => {
    const nextRole = memberRoleEdits[member.user_id] || member.role;
    if (nextRole === member.role) return;
    try {
      setError('');
      setUpdatingMemberRoleId(member.user_id);
      await api.updateTeamMemberRole(teamId, member.user_id, { role: nextRole });
      setSuccess('Member role updated successfully');
      void loadData();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to update member role';
      logger.error('Failed to update member role', { component: 'TeamDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(msg);
    } finally {
      setUpdatingMemberRoleId((current) => (current === member.user_id ? null : current));
    }
  };

  const handleUpdateTeam = async () => {
    if (!team) return;
    const trimmedName = editTeamName.trim();
    if (!trimmedName) {
      setEditTeamError('Team name is required.');
      return;
    }
    try {
      setUpdatingTeam(true);
      setEditTeamError('');
      await api.updateTeam(String(team.org_id), String(team.id), {
        name: trimmedName,
        description: editTeamDescription.trim() || undefined,
      });
      setShowEditTeam(false);
      setSuccess('Team updated successfully');
      void loadData();
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to update team';
      logger.error('Failed to update team', { component: 'TeamDetailPage', error: err instanceof Error ? err.message : String(err) });
      setEditTeamError(msg);
    } finally {
      setUpdatingTeam(false);
    }
  };

  const handleDeleteTeam = async () => {
    if (!team) return;
    const confirmed = await confirm({
      title: 'Delete Team',
      message: `Delete "${team.name}"? This team has ${members.length} member${members.length === 1 ? '' : 's'}.`,
      confirmText: 'Delete',
      variant: 'danger',
      icon: 'delete',
    });
    if (!confirmed) return;

    try {
      setDeletingTeam(true);
      setError('');
      await api.deleteTeam(String(team.org_id), String(team.id));
      router.push(team.org_id ? `/teams?org=${team.org_id}` : '/teams');
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'Failed to delete team';
      logger.error('Failed to delete team', { component: 'TeamDetailPage', error: err instanceof Error ? err.message : String(err) });
      setError(msg);
    } finally {
      setDeletingTeam(false);
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

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
          <div className="p-4 lg:p-8">
            {/* Header */}
            <div className="mb-8 flex items-center justify-between">
              <div className="flex items-center gap-4">
                <Button variant="ghost" onClick={() => router.push('/teams')}>
                  <ArrowLeft className="h-4 w-4" />
                </Button>
                <div>
                  <div className="flex items-center gap-3">
                    <Shield className="h-8 w-8 text-primary" />
                    <h1 className="text-3xl font-bold">{team?.name || `Team ${teamId}`}</h1>
                  </div>
                  {team?.description && (
                    <p className="text-muted-foreground mt-1">{team.description}</p>
                  )}
                  {team?.org_id && (
                    <Link href={`/organizations/${team.org_id}`} className="text-sm text-primary hover:underline mt-1 inline-flex items-center gap-1">
                      <Building2 className="h-3 w-3" />
                      View Organization
                    </Link>
                  )}
                </div>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant="outline"
                  onClick={() => {
                    setShowEditTeam(true);
                    setEditTeamError('');
                    setEditTeamName(team?.name || '');
                    setEditTeamDescription(team?.description || '');
                  }}
                >
                  <Pencil className="mr-2 h-4 w-4" />
                  Edit Team
                </Button>
                <Button
                  variant="outline"
                  onClick={handleDeleteTeam}
                  disabled={deletingTeam}
                  loading={deletingTeam}
                  loadingText="Deleting..."
                >
                  <Trash2 className="mr-2 h-4 w-4" />
                  Delete Team
                </Button>
                <Dialog open={showAddMember} onOpenChange={setShowAddMember}>
                  <DialogTrigger asChild>
                    <Button>
                      <UserPlus className="mr-2 h-4 w-4" />
                      Add Member
                    </Button>
                  </DialogTrigger>
                  <DialogContent>
                    <DialogHeader>
                      <DialogTitle>Add Team Member</DialogTitle>
                      <DialogDescription>
                        Add an existing user to this team
                      </DialogDescription>
                    </DialogHeader>
                    <div className="space-y-4 py-4">
                      <UserPicker
                        label="User"
                        value={selectedMember}
                        helperText="Search by username or email. The user must belong to this organization."
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
                          <option value="lead">Lead</option>
                          <option value="admin">Admin</option>
                        </Select>
                      </div>
                    </div>
                    <DialogFooter>
                      <Button
                        variant="outline"
                        onClick={() => {
                          setShowAddMember(false);
                          setError('');
                        }}
                      >
                        Cancel
                      </Button>
                      <Button onClick={handleAddMember} disabled={addingMember} loading={addingMember} loadingText="Adding...">
                        Add Member
                      </Button>
                    </DialogFooter>
                  </DialogContent>
                </Dialog>
              </div>
            </div>

            <Dialog open={showEditTeam} onOpenChange={setShowEditTeam}>
              <DialogContent>
                <DialogHeader>
                  <DialogTitle>Edit Team</DialogTitle>
                  <DialogDescription>
                    Update team name and description.
                  </DialogDescription>
                </DialogHeader>
                {editTeamError && (
                  <Alert variant="destructive">
                    <AlertDescription>{editTeamError}</AlertDescription>
                  </Alert>
                )}
                <div className="space-y-4 py-2">
                  <div className="space-y-2">
                    <Label htmlFor="editTeamName">Team Name</Label>
                    <Input
                      id="editTeamName"
                      value={editTeamName}
                      onChange={(event) => setEditTeamName(event.target.value)}
                      placeholder="Team name"
                    />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="editTeamDescription">Description</Label>
                    <Input
                      id="editTeamDescription"
                      value={editTeamDescription}
                      onChange={(event) => setEditTeamDescription(event.target.value)}
                      placeholder="Team description"
                    />
                  </div>
                </div>
                <DialogFooter>
                  <Button variant="outline" onClick={() => setShowEditTeam(false)}>
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

            {error && (
              <Alert variant="destructive" className="mb-6">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {success && (
              <Alert variant="success" className="mb-6">
                <AlertDescription>{success}</AlertDescription>
              </Alert>
            )}

            {/* Team Info */}
            <Card className="mb-6">
              <CardContent className="pt-6">
                <div className="flex items-start gap-4">
                  <Users className="h-8 w-8 text-primary mt-1" />
                  <div>
                    <h3 className="font-semibold">Team Management</h3>
                    <p className="text-sm text-muted-foreground mt-1">
                      Teams are groups within an organization that can have their own members and permissions.
                      Add users from the parent organization to this team to collaborate on projects.
                    </p>
                  </div>
                </div>
              </CardContent>
            </Card>

            {/* Members Section */}
            <Card>
              <CardHeader>
                <CardTitle className="flex items-center gap-2">
                  <Users className="h-5 w-5" />
                  Team Members
                </CardTitle>
                <CardDescription>
                  {members.length} member{members.length !== 1 ? 's' : ''} in this team
                </CardDescription>
              </CardHeader>
              <CardContent>
                {members.length === 0 ? (
                  <div className="text-center text-muted-foreground py-8">
                    No members in this team yet. Add members to get started.
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
                      {members.map((member) => (
                        <TableRow key={member.user_id}>
                          <TableCell>
                            <div>
                              <div className="font-medium">
                                {member.user?.username || `User ${member.user_id}`}
                              </div>
                              {member.user?.email && (
                                <div className="text-xs text-muted-foreground">{member.user.email}</div>
                              )}
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="space-y-2">
                              <Label htmlFor={`team-member-role-${member.user_id}`} className="sr-only">
                                Team role for {member.user?.username || `user ${member.user_id}`}
                              </Label>
                              <Select
                                id={`team-member-role-${member.user_id}`}
                                value={memberRoleEdits[member.user_id] || member.role}
                                onChange={(event) => {
                                  const role = event.target.value;
                                  setMemberRoleEdits((prev) => ({ ...prev, [member.user_id]: role }));
                                }}
                                disabled={updatingMemberRoleId === member.user_id}
                                className="max-w-[140px]"
                              >
                                <option value="member">Member</option>
                                <option value="lead">Lead</option>
                                <option value="admin">Admin</option>
                              </Select>
                            </div>
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatJoinedAt(member.joined_at)}
                          </TableCell>
                          <TableCell className="text-right">
                            <div className="flex justify-end gap-1">
                              <Link href={`/users/${member.user_id}`}>
                                <Button
                                  variant="ghost"
                                  size="sm"
                                  title="View user"
                                  aria-label={`View user ${member.user?.username || member.user_id}`}
                                >
                                  <Users className="h-4 w-4" />
                                </Button>
                              </Link>
                              <Button
                                variant="outline"
                                size="sm"
                                onClick={() => handleUpdateMemberRole(member)}
                                disabled={
                                  updatingMemberRoleId === member.user_id
                                  || (memberRoleEdits[member.user_id] || member.role) === member.role
                                }
                                loading={updatingMemberRoleId === member.user_id}
                                loadingText="Saving..."
                              >
                                Save
                              </Button>
                              <Button
                                variant="ghost"
                                size="sm"
                                onClick={() => handleRemoveMember(member.user_id, member.user?.username)}
                                title="Remove from team"
                                aria-label={`Remove ${member.user?.username || member.user_id} from team`}
                                disabled={removingMemberId === member.user_id}
                                loading={removingMemberId === member.user_id}
                                loadingText="Removing..."
                              >
                                <Trash2 className="h-4 w-4 text-red-500" />
                              </Button>
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
