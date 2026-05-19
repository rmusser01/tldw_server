'use client';

import { useCallback, useEffect, useRef, useState } from 'react';
import Link from 'next/link';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { EmptyState } from '@/components/ui/empty-state';
import { Input } from '@/components/ui/input';
import { Select } from '@/components/ui/select';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Pause, Play, RefreshCw, MessageSquare, XCircle, Wifi, WifiOff, Gauge } from 'lucide-react';
import { ExportMenu } from '@/components/ui/export-menu';
import { exportACPSessions, ExportFormat } from '@/lib/export';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import { api, ApiError } from '@/lib/api-client';
import { formatDateTime, formatTokens } from '@/lib/format';
import { formatDistanceToNow } from 'date-fns';

type UserInfo = { id: number; username?: string; email?: string };

interface ACPSession {
  session_id: string;
  user_id: number;
  agent_type: string;
  name: string;
  status: string;
  created_at: string;
  last_activity_at: string | null;
  message_count: number;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
  tags: string[];
  has_websocket: boolean;
  model: string | null;
  estimated_cost_usd: number | null;
  token_budget?: number | null;
  auto_terminate_at_budget?: boolean;
  budget_exhausted?: boolean;
  budget_remaining?: number | null;
  agent_budget?: number | null;
}

interface ACPSessionListResponse {
  sessions: ACPSession[];
  total: number;
}

export default function ACPSessionsPage() {
  const [sessions, setSessions] = useState<ACPSession[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [statusFilterDraft, setStatusFilterDraft] = useState<string>('');
  const [agentTypeFilterDraft, setAgentTypeFilterDraft] = useState('');
  const [userIdFilterDraft, setUserIdFilterDraft] = useState('');
  const [appliedFilters, setAppliedFilters] = useState<{
    status: string;
    agentType: string;
    userId: string;
  }>({
    status: '',
    agentType: '',
    userId: '',
  });
  const loadingRef = useRef(false);
  const latestRequestIdRef = useRef(0);
  const [autoRefreshEnabled, setAutoRefreshEnabled] = useState(true);
  const [lastRefreshed, setLastRefreshed] = useState<Date | null>(null);
  const [userMap, setUserMap] = useState<Record<number, UserInfo>>({});
  const confirm = useConfirm();
  const toast = useToast();

  // Load user map for resolving IDs to names
  useEffect(() => {
    let cancelled = false;
    const loadUsers = async () => {
      try {
        const items: UserInfo[] = [];
        let page = 1;
        let pages = 1;

        while (!cancelled && page <= pages) {
          const response = await api.getUsersPage({ page: String(page), limit: '200' });
          items.push(...(Array.isArray(response?.items) ? response.items : []));
          pages = Math.max(1, response?.pages ?? 1);
          if (!response?.items?.length) {
            break;
          }
          page += 1;
        }

        if (cancelled) return;
        const map: Record<number, UserInfo> = {};
        items.forEach((u) => { map[u.id] = u; });
        setUserMap(map);
      } catch {
        // Best-effort — keep numeric IDs as fallback
      }
    };
    void loadUsers();
    return () => { cancelled = true; };
  }, []);

  const resolveUserLabel = (userId: number) => {
    const info = userMap[userId];
    return info?.username || info?.email || String(userId);
  };

  const loadSessions = useCallback(async () => {
    const requestId = latestRequestIdRef.current + 1;
    latestRequestIdRef.current = requestId;
    loadingRef.current = true;
    setLoading(true);
    setError('');
    try {
      const params: Record<string, string> = {};
      if (appliedFilters.status) params.status = appliedFilters.status;
      if (appliedFilters.agentType) params.agent_type = appliedFilters.agentType;
      if (appliedFilters.userId) params.user_id = appliedFilters.userId;
      const response = await api.getACPSessions(params) as ACPSessionListResponse;
      if (latestRequestIdRef.current !== requestId) return;
      setSessions(response.sessions || []);
      setTotal(response.total || 0);
      setLastRefreshed(new Date());
    } catch (err) {
      if (latestRequestIdRef.current !== requestId) return;
      const message = err instanceof ApiError ? err.message : 'Failed to load ACP sessions';
      setError(message);
    } finally {
      if (latestRequestIdRef.current === requestId) {
        loadingRef.current = false;
        setLoading(false);
      }
    }
  }, [appliedFilters]);

  useEffect(() => {
    loadSessions();
  }, [loadSessions]);

  // Auto-refresh: 15-second interval, paused when tab is not visible
  useEffect(() => {
    if (!autoRefreshEnabled) return;

    const intervalId = setInterval(() => {
      if (document.visibilityState === 'visible' && !loadingRef.current) {
        void loadSessions();
      }
    }, 15_000);

    return () => clearInterval(intervalId);
  }, [autoRefreshEnabled, loadSessions]);

  // Re-render the "Last updated X ago" text every 15 seconds so it stays fresh
  const [, setTick] = useState(0);
  useEffect(() => {
    if (!lastRefreshed) return;
    const tickId = setInterval(() => setTick((t) => t + 1), 15_000);
    return () => clearInterval(tickId);
  }, [lastRefreshed]);

  const lastUpdatedLabel = lastRefreshed
    ? `Updated ${formatDistanceToNow(lastRefreshed, { addSuffix: true })}`
    : null;

  const handleApplyFilters = useCallback(() => {
    setAppliedFilters({
      status: statusFilterDraft,
      agentType: agentTypeFilterDraft.trim(),
      userId: userIdFilterDraft.trim(),
    });
  }, [statusFilterDraft, agentTypeFilterDraft, userIdFilterDraft]);

  const handleCloseSession = useCallback(async (sessionId: string) => {
    const ok = await confirm({
      title: 'Close ACP Session',
      message: `Are you sure you want to force-close session ${sessionId.slice(0, 8)}...?`,
      confirmText: 'Close Session',
      variant: 'danger',
    });
    if (!ok) return;

    try {
      await api.closeACPSession(sessionId);
      toast.success('Session closed successfully');
      loadSessions();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to close session';
      toast.error(message);
    }
  }, [confirm, toast, loadSessions]);

  // -- Set Budget dialog state --
  const [budgetDialogOpen, setBudgetDialogOpen] = useState(false);
  const [budgetSessionId, setBudgetSessionId] = useState<string | null>(null);
  const [budgetValue, setBudgetValue] = useState('');
  const [budgetAutoTerminate, setBudgetAutoTerminate] = useState(true);
  const [budgetSaving, setBudgetSaving] = useState(false);

  const openBudgetDialog = useCallback((session: ACPSession) => {
    setBudgetSessionId(session.session_id);
    setBudgetValue(session.token_budget != null ? String(session.token_budget) : '');
    setBudgetAutoTerminate(session.auto_terminate_at_budget ?? true);
    setBudgetDialogOpen(true);
  }, []);

  const handleSaveBudget = useCallback(async () => {
    if (!budgetSessionId) return;
    const parsed = parseInt(budgetValue, 10);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      toast.error('Token budget must be a positive number');
      return;
    }
    setBudgetSaving(true);
    try {
      await api.setSessionBudget(budgetSessionId, {
        token_budget: parsed,
        auto_terminate_at_budget: budgetAutoTerminate,
      });
      toast.success('Token budget updated');
      setBudgetDialogOpen(false);
      loadSessions();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to set budget';
      toast.error(message);
    } finally {
      setBudgetSaving(false);
    }
  }, [budgetSessionId, budgetValue, budgetAutoTerminate, toast, loadSessions]);

  const getStatusBadge = (status: string) => {
    switch (status) {
      case 'active':
        return <Badge variant="default">Active</Badge>;
      case 'closed':
        return <Badge variant="secondary">Closed</Badge>;
      case 'error':
        return <Badge variant="destructive">Error</Badge>;
      case 'budget_exceeded':
        return <Badge variant="destructive">Budget Exceeded</Badge>;
      default:
        return <Badge variant="outline">{status}</Badge>;
    }
  };

  const formatCost = (usd: number | null | undefined): string => {
    if (usd == null) return '\u2014';
    return `$${usd.toFixed(usd < 0.01 ? 4 : 2)}`;
  };

  const getBudgetDisplay = (session: ACPSession) => {
    if (session.token_budget == null) {
      return <span className="text-xs text-muted-foreground">No budget</span>;
    }
    const used = session.usage.total_tokens;
    const budget = session.token_budget;
    const pct = budget > 0 ? Math.min((used / budget) * 100, 100) : 0;

    if (session.budget_exhausted) {
      return (
        <div className="flex flex-col gap-0.5 min-w-[80px]" data-testid="budget-exhausted">
          <Badge variant="destructive" className="text-[10px] px-1.5 py-0">Exhausted</Badge>
          <span className="text-[10px] text-muted-foreground">
            {formatTokens(used)} / {formatTokens(budget)}
          </span>
        </div>
      );
    }

    const barColor =
      pct > 80 ? 'bg-red-500' :
      pct > 60 ? 'bg-yellow-500' :
      'bg-green-500';

    const textColor =
      pct > 80 ? 'text-red-600' :
      pct > 60 ? 'text-yellow-600' :
      'text-green-600';

    return (
      <div className="flex flex-col gap-0.5 min-w-[80px]" data-testid="budget-progress">
        <div className="flex items-center gap-1">
          <div className="flex-1 h-1.5 bg-muted rounded-full overflow-hidden" role="progressbar" aria-valuenow={Math.round(pct)} aria-valuemin={0} aria-valuemax={100}>
            <div className={`h-full rounded-full transition-all ${barColor}`} style={{ width: `${pct}%` }} />
          </div>
          <span className={`text-[10px] font-medium ${textColor}`}>{Math.round(pct)}%</span>
        </div>
        <span className="text-[10px] text-muted-foreground">
          {formatTokens(used)} / {formatTokens(budget)}
          {session.auto_terminate_at_budget && (
            <span title="Auto-terminates when budget is exhausted"> (auto)</span>
          )}
        </span>
      </div>
    );
  };
  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8 space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <div className="flex items-center gap-2">
                <h1 className="text-2xl font-bold">ACP Sessions</h1>
                {autoRefreshEnabled && (
                  <Badge variant="default" className="bg-green-600 animate-pulse text-xs">Live</Badge>
                )}
              </div>
              <p className="text-muted-foreground">Monitor and manage Agent Client Protocol sessions across all users</p>
            </div>
            <div className="flex items-center gap-2">
              {lastUpdatedLabel && (
                <span className="text-xs text-muted-foreground" data-testid="last-updated-label">
                  {lastUpdatedLabel}
                </span>
              )}
              <ExportMenu
                onExport={(format: ExportFormat) => exportACPSessions(sessions, format)}
                disabled={sessions.length === 0}
              />
              <Button
                variant="ghost"
                size="icon"
                onClick={() => setAutoRefreshEnabled((prev) => !prev)}
                aria-label={autoRefreshEnabled ? 'Pause auto-refresh' : 'Resume auto-refresh'}
                data-testid="auto-refresh-toggle"
                title={autoRefreshEnabled ? 'Pause auto-refresh' : 'Resume auto-refresh'}
              >
                {autoRefreshEnabled ? (
                  <Pause className="h-4 w-4" />
                ) : (
                  <Play className="h-4 w-4" />
                )}
              </Button>
              <AccessibleIconButton
                icon={RefreshCw}
                label="Refresh"
                onClick={loadSessions}
                disabled={loading}
                className={loading ? 'animate-spin' : ''}
              />
            </div>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Filters */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Filters</CardTitle>
            </CardHeader>
            <CardContent>
              <div className="flex flex-wrap gap-3">
                <Select
                  value={statusFilterDraft}
                  onChange={(e) => setStatusFilterDraft(e.target.value)}
                >
                  <option value="">All Statuses</option>
                  <option value="active">Active</option>
                  <option value="closed">Closed</option>
                  <option value="error">Error</option>
                </Select>
                <Input
                  placeholder="Agent type..."
                  value={agentTypeFilterDraft}
                  onChange={(e) => setAgentTypeFilterDraft(e.target.value)}
                  className="w-40"
                />
                <Input
                  placeholder="User ID..."
                  value={userIdFilterDraft}
                  onChange={(e) => setUserIdFilterDraft(e.target.value)}
                  className="w-32"
                />
                <Button variant="outline" size="sm" onClick={handleApplyFilters}>Apply</Button>
              </div>
            </CardContent>
          </Card>

          {/* Session Table */}
          <Card>
            <CardHeader>
              <CardTitle className="text-base">Sessions ({total})</CardTitle>
              <CardDescription>All ACP agent sessions across the platform</CardDescription>
            </CardHeader>
            <CardContent>
              {sessions.length === 0 && !loading ? (
                <EmptyState
                  icon={MessageSquare}
                  title="No ACP Sessions"
                  description="No agent sessions found matching your filters."
                />
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Session</TableHead>
                        <TableHead>User</TableHead>
                        <TableHead>Agent</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Messages</TableHead>
                        <TableHead>Tokens</TableHead>
                        <TableHead title="Estimated at blended $3/M tokens — actual cost varies by model">
                          Est. Cost <span className="text-muted-foreground cursor-help">&#9432;</span>
                        </TableHead>
                        <TableHead>Budget</TableHead>
                        <TableHead>WS</TableHead>
                        <TableHead>Created</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {sessions.map((session) => (
                        <TableRow key={session.session_id}>
                          <TableCell className="font-mono text-xs">
                            <div className="flex flex-col">
                              <span className="font-medium">{session.name || 'Unnamed'}</span>
                              <span className="text-muted-foreground">{session.session_id.slice(0, 12)}...</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Link href={`/users/${session.user_id}`} className="text-primary hover:underline">
                              {resolveUserLabel(session.user_id)}
                            </Link>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">{session.agent_type}</Badge>
                          </TableCell>
                          <TableCell>{getStatusBadge(session.status)}</TableCell>
                          <TableCell>{session.message_count}</TableCell>
                          <TableCell>
                            {session.agent_budget ? (
                              <div className="flex flex-col gap-1">
                                <span className="text-xs font-mono">
                                  {formatTokens(session.usage.total_tokens)} / {formatTokens(session.agent_budget)}
                                </span>
                                <div
                                  role="progressbar"
                                  aria-valuenow={Math.round((session.usage.total_tokens / session.agent_budget) * 100)}
                                  aria-valuemin={0}
                                  aria-valuemax={100}
                                  aria-label="Token budget usage"
                                  className="h-1.5 w-full rounded-full bg-muted overflow-hidden"
                                >
                                  <div
                                    className={`h-full rounded-full transition-all ${
                                      session.usage.total_tokens / session.agent_budget > 0.9 ? 'bg-red-500' :
                                      session.usage.total_tokens / session.agent_budget > 0.7 ? 'bg-yellow-500' :
                                      'bg-green-500'
                                    }`}
                                    style={{ width: `${Math.min(100, (session.usage.total_tokens / session.agent_budget) * 100)}%` }}
                                  />
                                </div>
                              </div>
                            ) : (
                              <span className="text-xs font-mono" title={`Prompt: ${session.usage.prompt_tokens} | Completion: ${session.usage.completion_tokens}`}>
                                {formatTokens(session.usage.total_tokens)}
                              </span>
                            )}
                          </TableCell>
                          <TableCell>
                            <span className="text-xs font-mono" title={session.model || undefined}>
                              {formatCost(session.estimated_cost_usd)}
                            </span>
                            {session.model && (
                              <div className="text-[10px] text-muted-foreground truncate max-w-[100px]" title={session.model}>
                                {session.model}
                              </div>
                            )}
                          </TableCell>
                          <TableCell>
                            {getBudgetDisplay(session)}
                          </TableCell>
                          <TableCell>
                            {session.has_websocket ? (
                              <Wifi className="h-4 w-4 text-green-500" />
                            ) : (
                              <WifiOff className="h-4 w-4 text-muted-foreground" />
                            )}
                          </TableCell>
                          <TableCell className="text-xs">{formatDateTime(session.created_at)}</TableCell>
                          <TableCell>
                            <div className="flex gap-1">
                              {session.status === 'active' && (
                                <>
                                  <AccessibleIconButton
                                    icon={Gauge}
                                    label="Set budget"
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => openBudgetDialog(session)}
                                  />
                                  <AccessibleIconButton
                                    icon={XCircle}
                                    label="Close session"
                                    size="sm"
                                    variant="ghost"
                                    onClick={() => handleCloseSession(session.session_id)}
                                  />
                                </>
                              )}
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
          {/* Set Budget Dialog */}
          <Dialog open={budgetDialogOpen} onOpenChange={setBudgetDialogOpen}>
            <DialogContent className="max-w-sm">
              <DialogHeader>
                <DialogTitle>Set Token Budget</DialogTitle>
                <DialogDescription>
                  Set a token budget for session {budgetSessionId?.slice(0, 12)}...
                </DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-2">
                <div className="space-y-2">
                  <Label htmlFor="budget-tokens">Token Budget</Label>
                  <Input
                    id="budget-tokens"
                    type="number"
                    min={1}
                    placeholder="e.g. 100000"
                    value={budgetValue}
                    onChange={(e) => setBudgetValue(e.target.value)}
                    data-testid="budget-input"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="budget-auto-terminate"
                    checked={budgetAutoTerminate}
                    onCheckedChange={(checked) => setBudgetAutoTerminate(checked === true)}
                    data-testid="budget-auto-terminate"
                  />
                  <Label htmlFor="budget-auto-terminate">Auto-terminate when budget is exhausted</Label>
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" size="sm" onClick={() => setBudgetDialogOpen(false)}>Cancel</Button>
                <Button size="sm" onClick={handleSaveBudget} disabled={budgetSaving} loading={budgetSaving} loadingText="Saving...">
                  Set Budget
                </Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
