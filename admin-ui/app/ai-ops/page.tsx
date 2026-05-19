'use client';

import { useCallback, useEffect, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { EmptyState } from '@/components/ui/empty-state';
import { api } from '@/lib/api-client';
import { formatDateTime } from '@/lib/format';
import { Activity, Bot, DollarSign, Hash, RefreshCw } from 'lucide-react';

interface AgentMetrics {
  agent_type: string;
  session_count: number;
  active_sessions: number;
  total_prompt_tokens: number;
  total_completion_tokens: number;
  total_tokens: number;
  total_messages: number;
  last_used_at: string | null;
  total_estimated_cost_usd: number | null;
}

interface RealtimeStats {
  active_sessions: number;
  tokens_today: {
    prompt: number;
    completion: number;
    total: number;
  };
}

interface SessionItem {
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
  estimated_cost_usd: number | null;
  model: string | null;
}

function formatTokens(n: number): string {
  if (n === 0) return '0';
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
}

function formatCost(usd: number | null | undefined): string {
  if (usd == null) return '\u2014';
  return `$${usd.toFixed(usd < 0.01 ? 4 : 2)}`;
}

function formatRelativeTime(iso: string | null): string {
  if (!iso) return '\u2014';
  const now = Date.now();
  const then = new Date(iso).getTime();
  const diffMs = now - then;
  if (diffMs < 0) return 'just now';
  const seconds = Math.floor(diffMs / 1000);
  if (seconds < 60) return 'just now';
  const minutes = Math.floor(seconds / 60);
  if (minutes < 60) return `${minutes}m ago`;
  const hours = Math.floor(minutes / 60);
  if (hours < 24) return `${hours}h ago`;
  const days = Math.floor(hours / 24);
  return `${days}d ago`;
}

const getStatusBadge = (status: string) => {
  switch (status) {
    case 'active':
      return <Badge variant="default">Active</Badge>;
    case 'closed':
      return <Badge variant="secondary">Closed</Badge>;
    case 'error':
      return <Badge variant="destructive">Error</Badge>;
    default:
      return <Badge variant="outline">{status}</Badge>;
  }
};

export default function AIOpsPage() {
  const [agentMetrics, setAgentMetrics] = useState<AgentMetrics[]>([]);
  const [realtimeStats, setRealtimeStats] = useState<RealtimeStats | null>(null);
  const [recentSessions, setRecentSessions] = useState<SessionItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  const loadData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [metricsRes, statsRes, sessionsRes] = await Promise.allSettled([
        api.getACPAgentMetrics(),
        api.getRealtimeStats(),
        api.getACPSessions({ limit: '10', sort: 'created_at:desc' }),
      ]);
      const partialFailures: string[] = [];

      if (metricsRes.status === 'fulfilled') {
        const data = metricsRes.value as { items: AgentMetrics[] };
        setAgentMetrics(Array.isArray(data.items) ? data.items : []);
      } else {
        setAgentMetrics([]);
        partialFailures.push('agent metrics');
      }

      if (statsRes.status === 'fulfilled') {
        setRealtimeStats(statsRes.value as RealtimeStats);
      } else {
        setRealtimeStats(null);
        partialFailures.push('realtime stats');
      }

      if (sessionsRes.status === 'fulfilled') {
        const data = sessionsRes.value as { sessions: SessionItem[]; total: number };
        setRecentSessions(Array.isArray(data.sessions) ? data.sessions.slice(0, 10) : []);
      } else {
        setRecentSessions([]);
        partialFailures.push('recent sessions');
      }

      if (partialFailures.length > 0) {
        setError(`Some AI Ops data could not be loaded: ${partialFailures.join(', ')}.`);
      }
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to load AI operations data';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void loadData();
  }, [loadData]);

  // Compute KPIs
  const totalSpend = agentMetrics.reduce(
    (sum, m) => sum + (m.total_estimated_cost_usd ?? 0),
    0
  );
  const totalTokensConsumed = agentMetrics.reduce(
    (sum, m) => sum + m.total_tokens,
    0
  );
  const activeSessionsCount = realtimeStats?.active_sessions ?? 0;
  const activeAgentsCount = agentMetrics.filter((m) => m.active_sessions > 0).length;

  // Sort agents by cost descending
  const agentsByCost = [...agentMetrics].sort(
    (a, b) => (b.total_estimated_cost_usd ?? 0) - (a.total_estimated_cost_usd ?? 0)
  );

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="flex flex-col gap-6 p-4 lg:p-8">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold">AI Operations</h1>
              <p className="text-muted-foreground">
                Aggregated view of AI spend, token consumption, and agent activity.
              </p>
            </div>
            <Button variant="outline" onClick={loadData} disabled={loading}>
              <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
              Refresh
            </Button>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* KPI Row */}
          <div className="grid gap-4 md:grid-cols-4">
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardDescription>Total AI Spend</CardDescription>
                <DollarSign className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatCost(totalSpend)}</div>
                <p className="text-xs text-muted-foreground">Across all agents</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardDescription>Active Sessions</CardDescription>
                <Activity className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{activeSessionsCount}</div>
                <p className="text-xs text-muted-foreground">
                  {realtimeStats
                    ? `${formatTokens(realtimeStats.tokens_today.total)} tokens today`
                    : 'Loading...'}
                </p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardDescription>Total Tokens</CardDescription>
                <Hash className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{formatTokens(totalTokensConsumed)}</div>
                <p className="text-xs text-muted-foreground">All-time consumption</p>
              </CardContent>
            </Card>
            <Card>
              <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
                <CardDescription>Active Agents</CardDescription>
                <Bot className="h-4 w-4 text-muted-foreground" />
              </CardHeader>
              <CardContent>
                <div className="text-2xl font-bold">{activeAgentsCount}</div>
                <p className="text-xs text-muted-foreground">
                  {agentMetrics.length} total configured
                </p>
              </CardContent>
            </Card>
          </div>

          {/* Top Agents by Cost */}
          <Card>
            <CardHeader>
              <CardTitle>Top Agents by Cost</CardTitle>
              <CardDescription>
                Agent types ranked by total estimated cost.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="py-8 text-center text-muted-foreground">Loading agent metrics...</div>
              ) : agentsByCost.length === 0 ? (
                <EmptyState
                  icon={Bot}
                  title="No agent metrics available"
                  description="Agent metrics will appear here once agents have been used."
                />
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Agent Type</TableHead>
                        <TableHead className="text-right">Sessions</TableHead>
                        <TableHead className="text-right">Active</TableHead>
                        <TableHead className="text-right">Messages</TableHead>
                        <TableHead className="text-right">Tokens</TableHead>
                        <TableHead className="text-right">Est. Cost</TableHead>
                        <TableHead>Last Used</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {agentsByCost.map((agent) => (
                        <TableRow key={agent.agent_type}>
                          <TableCell className="font-medium">
                            <Badge variant="outline">{agent.agent_type}</Badge>
                          </TableCell>
                          <TableCell className="text-right">{agent.session_count}</TableCell>
                          <TableCell className="text-right">
                            {agent.active_sessions > 0 ? (
                              <Badge variant="default" className="text-xs">
                                {agent.active_sessions}
                              </Badge>
                            ) : (
                              <span className="text-muted-foreground">0</span>
                            )}
                          </TableCell>
                          <TableCell className="text-right">{agent.total_messages}</TableCell>
                          <TableCell className="text-right font-mono text-xs">
                            {formatTokens(agent.total_tokens)}
                          </TableCell>
                          <TableCell className="text-right font-mono text-xs">
                            {formatCost(agent.total_estimated_cost_usd)}
                          </TableCell>
                          <TableCell className="text-xs text-muted-foreground">
                            {formatRelativeTime(agent.last_used_at)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Recent Sessions */}
          <Card>
            <CardHeader>
              <CardTitle>Recent Sessions</CardTitle>
              <CardDescription>
                Last 10 agent sessions with status, tokens, and cost.
              </CardDescription>
            </CardHeader>
            <CardContent>
              {loading ? (
                <div className="py-8 text-center text-muted-foreground">Loading sessions...</div>
              ) : recentSessions.length === 0 ? (
                <EmptyState
                  icon={Activity}
                  title="No recent sessions"
                  description="Agent sessions will appear here once activity begins."
                />
              ) : (
                <div className="overflow-x-auto">
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Session</TableHead>
                        <TableHead>Agent</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>User</TableHead>
                        <TableHead className="text-right">Messages</TableHead>
                        <TableHead className="text-right">Tokens</TableHead>
                        <TableHead className="text-right">Est. Cost</TableHead>
                        <TableHead>Created</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {recentSessions.map((session) => (
                        <TableRow key={session.session_id}>
                          <TableCell className="font-mono text-xs">
                            <div className="flex flex-col">
                              <span className="font-medium">{session.name || 'Unnamed'}</span>
                              <span className="text-muted-foreground">
                                {session.session_id.slice(0, 12)}...
                              </span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <Badge variant="outline">{session.agent_type}</Badge>
                          </TableCell>
                          <TableCell>{getStatusBadge(session.status)}</TableCell>
                          <TableCell>{session.user_id}</TableCell>
                          <TableCell className="text-right">{session.message_count}</TableCell>
                          <TableCell className="text-right font-mono text-xs">
                            {formatTokens(session.usage.total_tokens)}
                          </TableCell>
                          <TableCell className="text-right font-mono text-xs">
                            {formatCost(session.estimated_cost_usd)}
                          </TableCell>
                          <TableCell className="text-xs">
                            {formatDateTime(session.created_at)}
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                </div>
              )}
            </CardContent>
          </Card>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
