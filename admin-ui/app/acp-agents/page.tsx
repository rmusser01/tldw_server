'use client';

import { useCallback, useEffect, useState } from 'react';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { EmptyState } from '@/components/ui/empty-state';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Checkbox } from '@/components/ui/checkbox';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Dialog, DialogContent, DialogDescription, DialogHeader, DialogTitle } from '@/components/ui/dialog';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Badge } from '@/components/ui/badge';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { useToast } from '@/components/ui/toast';
import { RefreshCw, Bot, Plus, Pencil, Trash2, Shield } from 'lucide-react';
import { AccessibleIconButton } from '@/components/ui/accessible-icon-button';
import { api, ApiError } from '@/lib/api-client';
import { TagInput } from '@/components/ui/tag-input';

interface AgentConfig {
  id: number;
  type: string;
  name: string;
  description: string;
  system_prompt: string | null;
  allowed_tools: string[] | null;
  denied_tools: string[] | null;
  parameters: Record<string, unknown>;
  requires_api_key: string | null;
  org_id: number | null;
  team_id: number | null;
  enabled: boolean;
  is_configured: boolean;
  created_at: string;
  updated_at: string | null;
  max_token_budget: number | null;
}

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

function formatTokens(n: number): string {
  if (n === 0) return '0';
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`;
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return String(n);
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
  if (days < 30) return `${days}d ago`;
  const months = Math.floor(days / 30);
  return `${months}mo ago`;
}

function formatCost(usd: number | null | undefined): string {
  if (usd == null) return '\u2014';
  return `$${usd.toFixed(usd < 0.01 ? 4 : 2)}`;
}

interface PermissionPolicy {
  id: number;
  name: string;
  description: string;
  rules: Array<{ tool_pattern: string; tier: string }>;
  org_id: number | null;
  team_id: number | null;
  priority: number;
  created_at: string;
  updated_at: string | null;
}

const defaultAgentForm = {
  type: '',
  name: '',
  description: '',
  system_prompt: '',
  allowed_tools: '',
  denied_tools: '',
  temperature: '0.7',
  model: '',
  max_tokens: '',
  max_token_budget: '',
  requires_api_key: '',
  enabled: true,
  default_token_budget: '',
  default_auto_terminate_at_budget: true,
};

const defaultPolicyForm = {
  name: '',
  description: '',
  rules: '[]',
  priority: '0',
};

function extractToolNames(response: unknown): string[] {
  const rawTools: unknown[] = Array.isArray(response)
    ? response
    : (
      response
      && typeof response === 'object'
      && 'tools' in response
      && Array.isArray((response as Record<string, unknown>).tools)
    )
      ? (response as Record<string, unknown>).tools as unknown[]
      : [];

  return rawTools.flatMap((tool) => {
    if (typeof tool === 'string' && tool.trim()) {
      return [tool];
    }
    if (tool && typeof tool === 'object' && 'name' in tool) {
      const name = String((tool as Record<string, unknown>).name).trim();
      return name ? [name] : [];
    }
    return [];
  });
}

export default function ACPAgentsPage() {
  const [agents, setAgents] = useState<AgentConfig[]>([]);
  const [policies, setPolicies] = useState<PermissionPolicy[]>([]);
  const [agentMetrics, setAgentMetrics] = useState<Map<string, AgentMetrics>>(new Map());
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  // Agent dialog state
  const [agentDialogOpen, setAgentDialogOpen] = useState(false);
  const [editingAgent, setEditingAgent] = useState<AgentConfig | null>(null);
  const [agentForm, setAgentForm] = useState(defaultAgentForm);

  // Policy dialog state
  const [policyDialogOpen, setPolicyDialogOpen] = useState(false);
  const [editingPolicy, setEditingPolicy] = useState<PermissionPolicy | null>(null);
  const [policyForm, setPolicyForm] = useState(defaultPolicyForm);

  const [activeTab, setActiveTab] = useState<'agents' | 'policies'>('agents');

  // Agent usage metrics (optional — fetched separately)
  const [agentUsage, setAgentUsage] = useState<Record<string, {
    invocation_count: number; total_tokens: number;
    estimated_cost_usd: number; error_count: number;
  }>>({});

  const [availableTools, setAvailableTools] = useState<string[]>([]);

  const confirm = useConfirm();
  const toast = useToast();

  const loadData = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [agentsRes, policiesRes] = await Promise.all([
        api.getACPAgentConfigs() as Promise<{ agents: AgentConfig[]; total: number }>,
        api.getACPPermissionPolicies() as Promise<{ policies: PermissionPolicy[]; total: number }>,
      ]);
      setAgents(agentsRes.agents || []);
      setPolicies(policiesRes.policies || []);

      // Fetch metrics separately — non-blocking; failures are silently ignored
      api.getACPAgentMetrics().then(({ items }) => {
        const map = new Map(items.map(m => [m.agent_type, m]));
        setAgentMetrics(map);
      }).catch(() => {});
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to load data';
      setError(message);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    loadData();
  }, [loadData]);

  useEffect(() => {
    api.getACPAgentUsage(7).then((res) => {
      const map: Record<string, typeof res.agents[0]> = {};
      for (const a of res.agents) map[a.agent_type] = a;
      setAgentUsage(map);
    }).catch(() => { /* usage is optional — don't block page */ });

    api.getMCPTools().then((res) => {
      setAvailableTools(extractToolNames(res));
    }).catch(() => { /* tools list is optional */ });
  }, []);

  // -- Agent CRUD --

  const openCreateAgent = () => {
    setEditingAgent(null);
    setAgentForm(defaultAgentForm);
    setAgentDialogOpen(true);
  };

  const openEditAgent = (agent: AgentConfig) => {
    setEditingAgent(agent);
    setAgentForm({
      type: agent.type,
      name: agent.name,
      description: agent.description,
      system_prompt: agent.system_prompt || '',
      allowed_tools: agent.allowed_tools?.join(', ') || '',
      denied_tools: agent.denied_tools?.join(', ') || '',
      temperature: String((agent.parameters as Record<string, unknown>).temperature ?? '0.7'),
      model: String((agent.parameters as Record<string, unknown>).model ?? ''),
      max_tokens: String((agent.parameters as Record<string, unknown>).max_tokens ?? ''),
      requires_api_key: agent.requires_api_key || '',
      max_token_budget: agent.max_token_budget != null ? String(agent.max_token_budget) : '',
      enabled: agent.enabled,
      default_token_budget: String((agent.parameters as Record<string, unknown>).default_token_budget ?? ''),
      default_auto_terminate_at_budget: (agent.parameters as Record<string, unknown>).default_auto_terminate_at_budget !== false,
    });
    setAgentDialogOpen(true);
  };

  const handleSaveAgent = async () => {
    try {
      const payload: Record<string, unknown> = {
        type: agentForm.type,
        name: agentForm.name,
        description: agentForm.description,
        system_prompt: agentForm.system_prompt || null,
        allowed_tools: agentForm.allowed_tools ? agentForm.allowed_tools.split(',').map(s => s.trim()).filter(Boolean) : null,
        denied_tools: agentForm.denied_tools ? agentForm.denied_tools.split(',').map(s => s.trim()).filter(Boolean) : null,
        parameters: {
          ...(agentForm.temperature ? { temperature: parseFloat(agentForm.temperature) } : {}),
          ...(agentForm.model ? { model: agentForm.model } : {}),
          ...(agentForm.max_tokens ? { max_tokens: parseInt(agentForm.max_tokens) } : {}),
          ...(agentForm.default_token_budget ? { default_token_budget: parseInt(agentForm.default_token_budget) } : {}),
          ...(agentForm.default_token_budget ? { default_auto_terminate_at_budget: agentForm.default_auto_terminate_at_budget } : {}),
        },
        requires_api_key: agentForm.requires_api_key || null,
        enabled: agentForm.enabled,
        max_token_budget: agentForm.max_token_budget ? parseInt(agentForm.max_token_budget) : null,
      };

      if (editingAgent) {
        await api.updateACPAgentConfig(editingAgent.id, payload);
        toast.success('Agent configuration updated');
      } else {
        await api.createACPAgentConfig(payload);
        toast.success('Agent configuration created');
      }
      setAgentDialogOpen(false);
      loadData();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to save agent configuration';
      toast.error(message);
    }
  };

  const handleToggleAgent = async (agent: AgentConfig) => {
    const newEnabled = !agent.enabled;
    // Optimistic: update UI immediately
    setAgents((prev) =>
      prev.map((a) => (a.id === agent.id ? { ...a, enabled: newEnabled } : a))
    );
    try {
      await api.updateACPAgentConfig(agent.id, { enabled: newEnabled });
    } catch (err: unknown) {
      // Rollback on failure
      setAgents((prev) =>
        prev.map((a) => (a.id === agent.id ? { ...a, enabled: agent.enabled } : a))
      );
      const message = err instanceof ApiError ? err.message : 'Failed to toggle agent';
      toast.error(message);
    }
  };

  const handleDeleteAgent = async (agent: AgentConfig) => {
    const ok = await confirm({
      title: 'Delete Agent Configuration',
      message: `Are you sure you want to delete "${agent.name}"?`,
      confirmText: 'Delete',
      variant: 'danger',
    });
    if (!ok) return;
    try {
      await api.deleteACPAgentConfig(agent.id);
      toast.success('Agent configuration deleted');
      loadData();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to delete agent';
      toast.error(message);
    }
  };

  // -- Policy CRUD --

  const openCreatePolicy = () => {
    setEditingPolicy(null);
    setPolicyForm(defaultPolicyForm);
    setPolicyDialogOpen(true);
  };

  const openEditPolicy = (policy: PermissionPolicy) => {
    setEditingPolicy(policy);
    setPolicyForm({
      name: policy.name,
      description: policy.description,
      rules: JSON.stringify(policy.rules, null, 2),
      priority: String(policy.priority),
    });
    setPolicyDialogOpen(true);
  };

  const handleSavePolicy = async () => {
    try {
      let rules;
      try {
        rules = JSON.parse(policyForm.rules);
      } catch {
        toast.error('Invalid JSON in rules');
        return;
      }
      const payload = {
        name: policyForm.name,
        description: policyForm.description,
        rules,
        priority: parseInt(policyForm.priority) || 0,
      };
      if (editingPolicy) {
        await api.updateACPPermissionPolicy(editingPolicy.id, payload);
        toast.success('Permission policy updated');
      } else {
        await api.createACPPermissionPolicy(payload);
        toast.success('Permission policy created');
      }
      setPolicyDialogOpen(false);
      loadData();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to save policy';
      toast.error(message);
    }
  };

  const handleDeletePolicy = async (policy: PermissionPolicy) => {
    const ok = await confirm({
      title: 'Delete Permission Policy',
      message: `Are you sure you want to delete "${policy.name}"?`,
      confirmText: 'Delete',
      variant: 'danger',
    });
    if (!ok) return;
    try {
      await api.deleteACPPermissionPolicy(policy.id);
      toast.success('Permission policy deleted');
      loadData();
    } catch (err) {
      const message = err instanceof ApiError ? err.message : 'Failed to delete policy';
      toast.error(message);
    }
  };

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="p-4 lg:p-8 space-y-6">
          {/* Header */}
          <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
            <div>
              <h1 className="text-2xl font-bold">ACP Agent Configuration</h1>
              <p className="text-muted-foreground">Manage custom agent profiles and tool permission policies</p>
            </div>
            <AccessibleIconButton
              icon={RefreshCw}
              label="Refresh"
              onClick={loadData}
              disabled={loading}
              className={loading ? 'animate-spin' : ''}
            />
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {/* Tab Navigation */}
          <div className="flex gap-2 border-b pb-2">
            <Button
              variant={activeTab === 'agents' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('agents')}
            >
              <Bot className="h-4 w-4 mr-2" />
              Agents ({agents.length})
            </Button>
            <Button
              variant={activeTab === 'policies' ? 'default' : 'ghost'}
              size="sm"
              onClick={() => setActiveTab('policies')}
            >
              <Shield className="h-4 w-4 mr-2" />
              Permission Policies ({policies.length})
            </Button>
          </div>

          {/* Agents Tab */}
          {activeTab === 'agents' && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base">Custom Agent Configurations</CardTitle>
                  <CardDescription>Define agent profiles with system prompts, tool access, and parameters</CardDescription>
                </div>
                <Button size="sm" onClick={openCreateAgent}>
                  <Plus className="h-4 w-4 mr-2" />
                  New Agent
                </Button>
              </CardHeader>
              <CardContent>
                {agents.length === 0 ? (
                  <EmptyState
                    icon={Bot}
                    title="No Custom Agents"
                    description="Create custom agent configurations to control agent behavior and tool access."
                  />
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Type</TableHead>
                        <TableHead>Status</TableHead>
                        <TableHead>Model</TableHead>
                        <TableHead>Tools</TableHead>
                        <TableHead>Sessions</TableHead>
                        <TableHead className="text-right">Invocations</TableHead>
                        <TableHead className="text-right">Tokens</TableHead>
                        <TableHead className="text-right">Cost</TableHead>
                        <TableHead>Last Used</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {agents.map((agent) => (
                        <TableRow key={agent.id}>
                          <TableCell>
                            <div className="flex flex-col">
                              <span className="font-medium">{agent.name}</span>
                              <span className="text-xs text-muted-foreground">{agent.description}</span>
                            </div>
                          </TableCell>
                          <TableCell><Badge variant="outline">{agent.type}</Badge></TableCell>
                          <TableCell>
                            <div className="flex gap-1">
                              <Button
                                variant="ghost"
                                size="sm"
                                type="button"
                                onClick={() => void handleToggleAgent(agent)}
                                className="h-auto p-0 hover:bg-transparent hover:opacity-80"
                                title={`Click to ${agent.enabled ? 'disable' : 'enable'}`}
                                aria-pressed={agent.enabled}
                              >
                                {agent.enabled ? (
                                  <Badge variant="default">Enabled</Badge>
                                ) : (
                                  <Badge variant="secondary">Disabled</Badge>
                                )}
                              </Button>
                              {agent.is_configured ? (
                                <Badge variant="default">Configured</Badge>
                              ) : (
                                <Badge variant="destructive">Needs Key</Badge>
                              )}
                            </div>
                          </TableCell>
                          <TableCell className="text-xs font-mono">
                            {(agent.parameters as Record<string, unknown>).model as string || '-'}
                          </TableCell>
                          <TableCell className="text-xs">
                            {agent.allowed_tools ? `${agent.allowed_tools.length} allowed` : 'All'}
                            {agent.denied_tools ? `, ${agent.denied_tools.length} denied` : ''}
                          </TableCell>
                          {(() => {
                            const metrics = agentMetrics.get(agent.type);
                            return (
                              <>
                                <TableCell>
                                  {metrics ? (
                                    <div className="flex items-center gap-1.5">
                                      <span>{metrics.session_count}</span>
                                      {metrics.active_sessions > 0 && (
                                        <Badge variant="default" className="text-xs px-1.5 py-0">
                                          {metrics.active_sessions} active
                                        </Badge>
                                      )}
                                    </div>
                                  ) : (
                                    <span className="text-muted-foreground">{'\u2014'}</span>
                                  )}
                                </TableCell>
                              </>
                            );
                          })()}
                          <TableCell className="text-right font-mono text-sm">
                            {agentUsage[agent.type]?.invocation_count ?? '—'}
                          </TableCell>
                          <TableCell className="text-right font-mono text-sm">
                            {agentUsage[agent.type]
                              ? formatTokens(agentUsage[agent.type].total_tokens)
                              : (() => {
                                  const metrics = agentMetrics.get(agent.type);
                                  return metrics ? formatTokens(metrics.total_tokens) : '—';
                                })()}
                          </TableCell>
                          <TableCell className="text-right font-mono text-sm">
                            {agentUsage[agent.type]?.estimated_cost_usd != null
                              ? `$${agentUsage[agent.type].estimated_cost_usd.toFixed(2)}`
                              : (() => {
                                  const metrics = agentMetrics.get(agent.type);
                                  return metrics?.total_estimated_cost_usd != null
                                    ? formatCost(metrics.total_estimated_cost_usd)
                                    : '—';
                                })()}
                          </TableCell>
                          {(() => {
                            const metrics = agentMetrics.get(agent.type);
                            return (
                              <TableCell className="text-xs text-muted-foreground">
                                {metrics ? formatRelativeTime(metrics.last_used_at) : '\u2014'}
                              </TableCell>
                            );
                          })()}
                          <TableCell>
                            <div className="flex gap-1">
                              <AccessibleIconButton
                                icon={Pencil}
                                label="Edit"
                                size="sm"
                                variant="ghost"
                                onClick={() => openEditAgent(agent)}
                              />
                              <AccessibleIconButton
                                icon={Trash2}
                                label="Delete"
                                size="sm"
                                variant="ghost"
                                onClick={() => handleDeleteAgent(agent)}
                              />
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          )}

          {/* Policies Tab */}
          {activeTab === 'policies' && (
            <Card>
              <CardHeader className="flex flex-row items-center justify-between">
                <div>
                  <CardTitle className="text-base">Tool Permission Policies</CardTitle>
                  <CardDescription>Define rules for auto-approving, batching, or requiring individual approval for tools</CardDescription>
                </div>
                <Button size="sm" onClick={openCreatePolicy}>
                  <Plus className="h-4 w-4 mr-2" />
                  New Policy
                </Button>
              </CardHeader>
              <CardContent>
                {policies.length === 0 ? (
                  <EmptyState
                    icon={Shield}
                    title="No Permission Policies"
                    description="Create policies to control which tools require approval."
                  />
                ) : (
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Name</TableHead>
                        <TableHead>Rules</TableHead>
                        <TableHead>Priority</TableHead>
                        <TableHead>Actions</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {policies.map((policy) => (
                        <TableRow key={policy.id}>
                          <TableCell>
                            <div className="flex flex-col">
                              <span className="font-medium">{policy.name}</span>
                              <span className="text-xs text-muted-foreground">{policy.description}</span>
                            </div>
                          </TableCell>
                          <TableCell>
                            <div className="flex flex-wrap gap-1">
                              {policy.rules.map((rule, i) => (
                                <Badge key={i} variant="outline" className="text-xs">
                                  {rule.tool_pattern}: {rule.tier}
                                </Badge>
                              ))}
                            </div>
                          </TableCell>
                          <TableCell>{policy.priority}</TableCell>
                          <TableCell>
                            <div className="flex gap-1">
                              <AccessibleIconButton
                                icon={Pencil}
                                label="Edit"
                                size="sm"
                                variant="ghost"
                                onClick={() => openEditPolicy(policy)}
                              />
                              <AccessibleIconButton
                                icon={Trash2}
                                label="Delete"
                                size="sm"
                                variant="ghost"
                                onClick={() => handleDeletePolicy(policy)}
                              />
                            </div>
                          </TableCell>
                        </TableRow>
                      ))}
                    </TableBody>
                  </Table>
                )}
              </CardContent>
            </Card>
          )}

          {/* Agent Config Dialog */}
          <Dialog open={agentDialogOpen} onOpenChange={setAgentDialogOpen}>
            <DialogContent className="max-w-2xl max-h-[80vh] overflow-y-auto">
              <DialogHeader>
                <DialogTitle>{editingAgent ? 'Edit Agent Configuration' : 'Create Agent Configuration'}</DialogTitle>
                <DialogDescription>Define agent behavior, tool access, and model parameters</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="agent-type">Type Identifier</Label>
                    <Input id="agent-type" value={agentForm.type} onChange={(e) => setAgentForm(f => ({ ...f, type: e.target.value }))} placeholder="e.g., my_custom_agent" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="agent-name">Name</Label>
                    <Input id="agent-name" value={agentForm.name} onChange={(e) => setAgentForm(f => ({ ...f, name: e.target.value }))} placeholder="My Custom Agent" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-desc">Description</Label>
                  <Input id="agent-desc" value={agentForm.description} onChange={(e) => setAgentForm(f => ({ ...f, description: e.target.value }))} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-prompt">System Prompt</Label>
                  <textarea
                    id="agent-prompt"
                    value={agentForm.system_prompt}
                    onChange={(e) => setAgentForm(f => ({ ...f, system_prompt: e.target.value }))}
                    className="w-full min-h-[120px] rounded-md border border-input bg-background px-3 py-2 text-sm"
                    placeholder="You are a helpful coding assistant..."
                  />
                </div>
                <div className="grid grid-cols-3 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="agent-model">Model</Label>
                    <Input id="agent-model" value={agentForm.model} onChange={(e) => setAgentForm(f => ({ ...f, model: e.target.value }))} placeholder="claude-opus-4-6" />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="agent-temp">Temperature</Label>
                    <Input id="agent-temp" type="number" step="0.1" min="0" max="2" value={agentForm.temperature} onChange={(e) => setAgentForm(f => ({ ...f, temperature: e.target.value }))} />
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="agent-tokens">Max Tokens</Label>
                    <Input id="agent-tokens" type="number" value={agentForm.max_tokens} onChange={(e) => setAgentForm(f => ({ ...f, max_tokens: e.target.value }))} placeholder="4096" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-budget">Session Token Budget</Label>
                  <Input id="agent-budget" type="number" value={agentForm.max_token_budget} onChange={(e) => setAgentForm(f => ({ ...f, max_token_budget: e.target.value }))} placeholder="Leave empty for unlimited" />
                  <p className="text-xs text-muted-foreground">Maximum total tokens per session. Sessions are auto-terminated when exceeded.</p>
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="agent-allowed">Allowed Tools</Label>
                    <TagInput id="agent-allowed" value={agentForm.allowed_tools} onChange={(v) => setAgentForm(f => ({ ...f, allowed_tools: v }))} placeholder="Type tool name, press Enter" />
                    {availableTools.length > 0 && (
                      <p className="text-xs text-muted-foreground">{availableTools.length} backend tools available. Enter exact tool names to allow.</p>
                    )}
                  </div>
                  <div className="space-y-2">
                    <Label htmlFor="agent-denied">Denied Tools</Label>
                    <TagInput id="agent-denied" value={agentForm.denied_tools} onChange={(v) => setAgentForm(f => ({ ...f, denied_tools: v }))} placeholder="Type tool name, press Enter" />
                  </div>
                </div>
                <div className="space-y-2">
                  <Label htmlFor="agent-key">Required API Key (env var name)</Label>
                  <Input id="agent-key" value={agentForm.requires_api_key} onChange={(e) => setAgentForm(f => ({ ...f, requires_api_key: e.target.value }))} placeholder="ANTHROPIC_API_KEY" />
                </div>
                <div className="grid grid-cols-2 gap-4">
                  <div className="space-y-2">
                    <Label htmlFor="agent-default-budget">Default Token Budget</Label>
                    <Input id="agent-default-budget" type="number" min={0} value={agentForm.default_token_budget} onChange={(e) => setAgentForm(f => ({ ...f, default_token_budget: e.target.value }))} placeholder="e.g. 100000 (empty = no budget)" />
                  </div>
                  <div className="flex items-end pb-2">
                    <div className="flex items-center gap-2">
                      <Checkbox
                        id="agent-auto-terminate"
                        checked={agentForm.default_auto_terminate_at_budget}
                        onCheckedChange={(checked) => setAgentForm(f => ({ ...f, default_auto_terminate_at_budget: checked === true }))}
                        disabled={!agentForm.default_token_budget}
                      />
                      <Label htmlFor="agent-auto-terminate">Auto-terminate at budget</Label>
                    </div>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  <Checkbox
                    id="agent-enabled"
                    checked={agentForm.enabled}
                    onCheckedChange={(checked) => setAgentForm(f => ({ ...f, enabled: checked === true }))}
                  />
                  <Label htmlFor="agent-enabled">Enabled</Label>
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setAgentDialogOpen(false)}>Cancel</Button>
                <Button onClick={handleSaveAgent}>{editingAgent ? 'Update' : 'Create'}</Button>
              </div>
            </DialogContent>
          </Dialog>

          {/* Policy Dialog */}
          <Dialog open={policyDialogOpen} onOpenChange={setPolicyDialogOpen}>
            <DialogContent className="max-w-lg">
              <DialogHeader>
                <DialogTitle>{editingPolicy ? 'Edit Permission Policy' : 'Create Permission Policy'}</DialogTitle>
                <DialogDescription>Define tool permission rules</DialogDescription>
              </DialogHeader>
              <div className="space-y-4 py-4">
                <div className="space-y-2">
                  <Label htmlFor="policy-name">Name</Label>
                  <Input id="policy-name" value={policyForm.name} onChange={(e) => setPolicyForm(f => ({ ...f, name: e.target.value }))} />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="policy-desc">Description</Label>
                  <Input id="policy-desc" value={policyForm.description} onChange={(e) => setPolicyForm(f => ({ ...f, description: e.target.value }))} />
                </div>
                <div className="space-y-2">
                  <Label>Rules</Label>
                  <PolicyRuleBuilder
                    value={policyForm.rules}
                    onChange={(rules) => setPolicyForm(f => ({ ...f, rules }))}
                  />
                </div>
                <div className="space-y-2">
                  <Label htmlFor="policy-priority">Priority (higher takes precedence)</Label>
                  <Input id="policy-priority" type="number" value={policyForm.priority} onChange={(e) => setPolicyForm(f => ({ ...f, priority: e.target.value }))} />
                </div>
              </div>
              <div className="flex justify-end gap-2">
                <Button variant="outline" onClick={() => setPolicyDialogOpen(false)}>Cancel</Button>
                <Button onClick={handleSavePolicy}>{editingPolicy ? 'Update' : 'Create'}</Button>
              </div>
            </DialogContent>
          </Dialog>
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}

// Structured rule builder that serializes to/from JSON
function PolicyRuleBuilder({ value, onChange }: { value: string; onChange: (v: string) => void }) {
  const TIER_OPTIONS = ['auto_approve', 'require_approval', 'deny'];

  let rules: Array<{ tool_pattern: string; tier: string }> = [];
  try {
    const parsed = JSON.parse(value);
    if (Array.isArray(parsed)) rules = parsed;
  } catch { /* invalid JSON — start fresh */ }

  const updateRules = (updated: typeof rules) => {
    onChange(JSON.stringify(updated, null, 2));
  };

  const addRule = () => {
    updateRules([...rules, { tool_pattern: '', tier: 'auto_approve' }]);
  };

  const removeRule = (index: number) => {
    updateRules(rules.filter((_, i) => i !== index));
  };

  const updateRule = (index: number, field: 'tool_pattern' | 'tier', val: string) => {
    updateRules(rules.map((r, i) => i === index ? { ...r, [field]: val } : r));
  };

  return (
    <div className="space-y-2">
      {rules.map((rule, idx) => (
        <div key={idx} className="flex items-center gap-2">
          <Input
            placeholder="Tool pattern (e.g., read_*)"
            value={rule.tool_pattern}
            onChange={(e) => updateRule(idx, 'tool_pattern', e.target.value)}
            className="flex-1"
          />
          <select
            value={rule.tier}
            onChange={(e) => updateRule(idx, 'tier', e.target.value)}
            className="h-10 rounded-md border border-input bg-background px-3 text-sm"
          >
            {TIER_OPTIONS.map(t => <option key={t} value={t}>{t.replace(/_/g, ' ')}</option>)}
          </select>
          <Button variant="ghost" size="sm" onClick={() => removeRule(idx)} className="text-destructive">
            <Trash2 className="h-4 w-4" />
          </Button>
        </div>
      ))}
      <Button variant="outline" size="sm" onClick={addRule}>
        <Plus className="mr-1 h-3 w-3" /> Add Rule
      </Button>
      <details className="text-xs">
        <summary className="cursor-pointer text-muted-foreground">JSON preview</summary>
        <pre className="mt-1 p-2 bg-muted rounded text-xs overflow-auto max-h-24">{value}</pre>
      </details>
    </div>
  );
}
