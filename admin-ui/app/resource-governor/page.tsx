'use client';

import { useCallback, useEffect, useMemo, useState } from 'react';
import { useForm, FormProvider } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { PermissionGuard } from '@/components/PermissionGuard';
import { ResponsiveLayout } from '@/components/ResponsiveLayout';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Button } from '@/components/ui/button';
import { Select } from '@/components/ui/select';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Alert, AlertDescription } from '@/components/ui/alert';
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table';
import { Badge } from '@/components/ui/badge';
import { useConfirm } from '@/components/ui/confirm-dialog';
import { usePrivilegedActionDialog } from '@/components/ui/privileged-action-dialog';
import { useToast } from '@/components/ui/toast';
import { Form, FormCheckbox, FormInput, FormSelect } from '@/components/ui/form';
import { api } from '@/lib/api-client';
import { formatDateTime } from '@/lib/format';
import { parseOptionalInt } from '@/lib/number';
import { RateLimitEvent, normalizeRateLimitEventsPayload, parseRateLimitEventsFromMetricsText } from '@/lib/rate-limit-events';
import Link from 'next/link';
import { RefreshCw, Trash2, Edit2, Plus, Gauge, X } from 'lucide-react';

type ResourcePolicy = {
  id?: string | number;
  name: string;
  scope: 'global' | 'org' | 'user' | 'role';
  scope_id?: string | number | null;
  resource_type: string;
  max_requests_per_minute?: number | null;
  max_requests_per_hour?: number | null;
  max_requests_per_day?: number | null;
  max_tokens_per_request?: number | null;
  max_concurrent_requests?: number | null;
  priority?: number;
  enabled: boolean;
  description?: string | null;
  created_at?: string | null;
  updated_at?: string | null;
};

type PoliciesResponse = {
  policies?: ResourcePolicy[];
  items?: ResourcePolicy[];
};

type UsersPageResponse = {
  items?: ScopeUser[];
  total?: number;
  page?: number;
  pages?: number;
  limit?: number;
};

type ScopeUser = {
  id: number;
  username?: string;
  email?: string;
  role?: string;
  roles?: string[];
};

type OrgMembershipItem = {
  org_id?: number | string;
};

type LlmUsageLogItem = {
  user_id?: number | null;
};

type LlmUsageLogResponse = {
  items?: LlmUsageLogItem[];
  total?: number;
  limit?: number;
  page?: number;
};

type PolicySimulationResult = {
  affectedUsers: number;
  affectedRequests24h: number;
  source: 'endpoint' | 'client_estimate';
};

type PolicyResolutionStep = {
  policy: ResourcePolicy;
  priority: number;
  reason: string;
};

type PolicyResolutionResult = {
  user: ScopeUser;
  resourceType: string;
  steps: PolicyResolutionStep[];
  winner: PolicyResolutionStep | null;
};

type RateLimitEventsSource = 'endpoint' | 'metrics_text' | 'unavailable';

const RESOURCE_TYPES = [
  'llm',
  'embedding',
  'transcription',
  'tts',
  'rag',
  'media_processing',
  'all',
] as const;

const POLICY_SCOPES = ['global', 'org', 'user', 'role'] as const;
const USERS_PAGE_LIMIT = 100;
const USAGE_LOOKBACK_HOURS = 24;

const optionalNonNegativeIntegerString = z
  .string()
  .trim()
  .refine((value) => value === '' || /^\d+$/.test(value), {
    message: 'Enter a non-negative whole number',
  });

const policyFormSchema = z
  .object({
    name: z.string().trim().min(1, 'Policy name is required'),
    scope: z.enum(POLICY_SCOPES),
    scope_id: z.string().default(''),
    resource_type: z.enum(RESOURCE_TYPES),
    max_requests_per_minute: optionalNonNegativeIntegerString,
    max_requests_per_hour: optionalNonNegativeIntegerString,
    max_requests_per_day: optionalNonNegativeIntegerString,
    max_tokens_per_request: optionalNonNegativeIntegerString,
    max_concurrent_requests: optionalNonNegativeIntegerString,
    priority: optionalNonNegativeIntegerString,
    enabled: z.boolean(),
    description: z.string().default(''),
  })
  .superRefine((data, ctx) => {
    if (data.scope === 'global' || data.scope_id.trim()) {
      return;
    }

    const scopeLabel = data.scope === 'org'
      ? 'Organization'
      : data.scope === 'user'
        ? 'User'
        : 'Role';

    ctx.addIssue({
      code: z.ZodIssueCode.custom,
      path: ['scope_id'],
      message: `${scopeLabel} ID is required for ${data.scope} scope`,
    });
  });

type PolicyFormInput = z.input<typeof policyFormSchema>;
type PolicyFormData = z.output<typeof policyFormSchema>;

const defaultPolicyFormValues: PolicyFormInput = {
  name: '',
  scope: 'global',
  scope_id: '',
  resource_type: 'llm',
  max_requests_per_minute: '',
  max_requests_per_hour: '',
  max_requests_per_day: '',
  max_tokens_per_request: '',
  max_concurrent_requests: '',
  priority: '0',
  enabled: true,
  description: '',
};

const formatPolicyDate = (value?: string | null) =>
  formatDateTime(value, { fallback: '—' });

const toIsoHoursAgo = (hours: number) => {
  const date = new Date(Date.now() - hours * 60 * 60 * 1000);
  return date.toISOString();
};

const normalizeUsersPageItems = (payload: unknown): ScopeUser[] => {
  if (Array.isArray(payload)) return payload as ScopeUser[];
  if (payload && typeof payload === 'object' && Array.isArray((payload as UsersPageResponse).items)) {
    return (payload as UsersPageResponse).items ?? [];
  }
  return [];
};

const normalizeMembershipItems = (payload: unknown): OrgMembershipItem[] => {
  if (Array.isArray(payload)) return payload as OrgMembershipItem[];
  if (payload && typeof payload === 'object' && Array.isArray((payload as { items?: OrgMembershipItem[] }).items)) {
    return (payload as { items: OrgMembershipItem[] }).items;
  }
  return [];
};

const normalizeLlmUsageItems = (payload: unknown): LlmUsageLogItem[] => {
  if (Array.isArray(payload)) return payload as LlmUsageLogItem[];
  if (payload && typeof payload === 'object' && Array.isArray((payload as LlmUsageLogResponse).items)) {
    return (payload as LlmUsageLogResponse).items ?? [];
  }
  return [];
};

const getPolicyPriority = (policy: ResourcePolicy) =>
  Number.isFinite(policy.priority) ? Number(policy.priority) : 0;

const policyResourceMatches = (policy: ResourcePolicy, resourceType: string) =>
  (policy.resource_type || 'all') === 'all' || policy.resource_type === resourceType;

const getPolicyRowKey = (policy: ResourcePolicy, fallbackIndex: number) => {
  if (policy.id != null) return `policy-${policy.id}`;
  return `policy-${policy.name}-${policy.scope}-${String(policy.scope_id ?? 'global')}-${policy.resource_type}-${fallbackIndex}`;
};

export default function ResourceGovernorPage() {
  const confirm = useConfirm();
  const promptPrivilegedAction = usePrivilegedActionDialog();
  const { success, error: showError } = useToast();

  // Policies state
  const [allPolicies, setAllPolicies] = useState<ResourcePolicy[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  // Filter state
  const [scopeFilter, setScopeFilter] = useState('');
  const [resourceTypeFilter, setResourceTypeFilter] = useState('');

  // Form state for create/edit
  const [showForm, setShowForm] = useState(false);
  const [editingPolicy, setEditingPolicy] = useState<ResourcePolicy | null>(null);
  const [formSaving, setFormSaving] = useState(false);
  const [deletingPolicyId, setDeletingPolicyId] = useState<string | null>(null);
  const [scopeUsers, setScopeUsers] = useState<ScopeUser[]>([]);
  const [orgMembershipsByUserId, setOrgMembershipsByUserId] = useState<Record<string, Set<string>>>({});
  const [requests24hByUserId, setRequests24hByUserId] = useState<Record<string, number>>({});
  const [scopeContextLoading, setScopeContextLoading] = useState(false);
  const [scopeContextError, setScopeContextError] = useState('');
  const [simulationResult, setSimulationResult] = useState<PolicySimulationResult | null>(null);
  const [simulationLoading, setSimulationLoading] = useState(false);
  const [resolutionUserId, setResolutionUserId] = useState('');
  const [userSuggestions, setUserSuggestions] = useState<Array<{ id: number; username: string }>>([]);
  const [resolutionResourceType, setResolutionResourceType] = useState<PolicyFormData['resource_type']>('llm');
  const [resolutionResult, setResolutionResult] = useState<PolicyResolutionResult | null>(null);
  const [resolutionError, setResolutionError] = useState('');
  const [resolutionLoading, setResolutionLoading] = useState(false);
  const [rateLimitEvents, setRateLimitEvents] = useState<RateLimitEvent[]>([]);
  const [rateLimitEventsLoading, setRateLimitEventsLoading] = useState(false);
  const [rateLimitEventsError, setRateLimitEventsError] = useState('');
  const [rateLimitEventsSource, setRateLimitEventsSource] = useState<RateLimitEventsSource>('unavailable');
  const [rateLimitSummary, setRateLimitSummary] = useState<{
    total_throttle_events: number;
    period: string;
    top_throttled_entities: Array<{ entity: string; rejections: number; last_rejected_at: string | null }>;
    policy_headroom: Array<{ policy_id: string; resource_type: string | null; utilization_pct: number; total_denials: number; total_decisions: number }>;
  } | null>(null);
  const [rateLimitSummaryLoading, setRateLimitSummaryLoading] = useState(false);

  const policyForm = useForm<PolicyFormInput, unknown, PolicyFormData>({
    resolver: zodResolver(policyFormSchema),
    defaultValues: defaultPolicyFormValues,
  });

  const watchedScope = policyForm.watch('scope');

  const fetchAllUsers = useCallback(async (): Promise<ScopeUser[]> => {
    const users: ScopeUser[] = [];
    let page = 1;
    let pages = 1;
    while (page <= pages) {
      const payload = await api.getUsersPage({
        page: String(page),
        limit: String(USERS_PAGE_LIMIT),
      });
      const items = normalizeUsersPageItems(payload);
      users.push(...items);

      if (payload && typeof payload === 'object') {
        const pageCount = Number((payload as UsersPageResponse).pages);
        const total = Number((payload as UsersPageResponse).total);
        const limit = Number((payload as UsersPageResponse).limit || USERS_PAGE_LIMIT);
        if (Number.isFinite(pageCount) && pageCount > 0) {
          pages = pageCount;
        } else if (Number.isFinite(total) && total > 0) {
          pages = Math.max(1, Math.ceil(total / Math.max(limit, 1)));
        }
      }

      if (items.length === 0) break;
      page += 1;
    }
    return users;
  }, []);

  const fetchRequests24hByUser = useCallback(async (): Promise<Record<string, number>> => {
    const requestsByUser: Record<string, number> = {};
    const start = toIsoHoursAgo(USAGE_LOOKBACK_HOURS);
    let page = 1;
    let totalPages = 1;
    const limit = 500;

    while (page <= totalPages) {
      const payload = await api.getLlmUsage({
        start,
        end: new Date().toISOString(),
        page: String(page),
        limit: String(limit),
      });
      const items = normalizeLlmUsageItems(payload);
      items.forEach((item) => {
        if (item.user_id == null) return;
        const userKey = String(item.user_id);
        requestsByUser[userKey] = (requestsByUser[userKey] || 0) + 1;
      });

      let resolvedTotalPages = totalPages;
      if (payload && typeof payload === 'object') {
        const total = Number((payload as LlmUsageLogResponse).total);
        const responseLimit = Number((payload as LlmUsageLogResponse).limit || limit);
        if (Number.isFinite(total) && total > 0) {
          resolvedTotalPages = Math.max(1, Math.ceil(total / Math.max(responseLimit, 1)));
        }
      }
      totalPages = resolvedTotalPages;
      if (items.length < limit) break;
      page += 1;
    }

    return requestsByUser;
  }, []);

  const refreshScopeContext = useCallback(async () => {
    try {
      setScopeContextLoading(true);
      setScopeContextError('');

      const users = await fetchAllUsers();
      setScopeUsers(users);

      const membershipsResults = await Promise.allSettled(
        users.map(async (user) => ({
          userId: user.id,
          memberships: normalizeMembershipItems(await api.getUserOrgMemberships(String(user.id))),
        }))
      );
      const membershipsByUser: Record<string, Set<string>> = {};
      membershipsResults.forEach((result) => {
        if (result.status !== 'fulfilled') return;
        const orgSet = new Set<string>();
        result.value.memberships.forEach((membership) => {
          if (membership.org_id != null) {
            orgSet.add(String(membership.org_id));
          }
        });
        membershipsByUser[String(result.value.userId)] = orgSet;
      });
      setOrgMembershipsByUserId(membershipsByUser);

      const requests = await fetchRequests24hByUser();
      setRequests24hByUserId(requests);
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to load scope context';
      setScopeContextError(message);
      setScopeUsers([]);
      setOrgMembershipsByUserId({});
      setRequests24hByUserId({});
    } finally {
      setScopeContextLoading(false);
    }
  }, [fetchAllUsers, fetchRequests24hByUser]);

  const loadRateLimitEvents = useCallback(async () => {
    try {
      setRateLimitEventsLoading(true);
      setRateLimitEventsError('');

      try {
        const endpointPayload = await api.getRateLimitEvents({
          hours: String(USAGE_LOOKBACK_HOURS),
        });
        const endpointEvents = normalizeRateLimitEventsPayload(endpointPayload);
        setRateLimitEvents(endpointEvents);
        setRateLimitEventsSource('endpoint');
        return;
      } catch {
        // Endpoint may be unavailable in older backends. Fall back to metrics parsing.
      }

      const metricsText = await api.getMetricsText();
      const fallbackEvents = parseRateLimitEventsFromMetricsText(metricsText);
      setRateLimitEvents(fallbackEvents);
      setRateLimitEventsSource('metrics_text');
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to load rate limit events';
      setRateLimitEvents([]);
      setRateLimitEventsError(message);
      setRateLimitEventsSource('unavailable');
    } finally {
      setRateLimitEventsLoading(false);
    }
  }, []);

  const loadRateLimitSummary = useCallback(async () => {
    try {
      setRateLimitSummaryLoading(true);
      const result = await api.getRateLimitSummary({ hours: String(USAGE_LOOKBACK_HOURS) });
      if (result && typeof result === 'object') {
        const record = result as Record<string, unknown>;
        setRateLimitSummary({
          total_throttle_events: typeof record.total_throttle_events === 'number' ? record.total_throttle_events : 0,
          period: typeof record.period === 'string' ? record.period : '24h',
          top_throttled_entities: Array.isArray(record.top_throttled_entities)
            ? (record.top_throttled_entities as Array<{ entity: string; rejections: number; last_rejected_at: string | null }>)
            : [],
          policy_headroom: Array.isArray(record.policy_headroom)
            ? (record.policy_headroom as Array<{ policy_id: string; resource_type: string | null; utilization_pct: number; total_denials: number; total_decisions: number }>)
            : [],
        });
      }
    } catch {
      // Summary is non-critical; silently ignore failures
      setRateLimitSummary(null);
    } finally {
      setRateLimitSummaryLoading(false);
    }
  }, []);

  const loadPolicies = useCallback(async (signal?: AbortSignal) => {
    try {
      setLoading(true);
      setError('');
      const data = (await api.getResourceGovernorPolicy({ include_ids: true }, signal)) as PoliciesResponse;
      const items = data.policies || data.items || [];
      setAllPolicies(Array.isArray(items) ? items : []);
      await Promise.all([
        refreshScopeContext(),
        loadRateLimitEvents(),
        loadRateLimitSummary(),
      ]);
    } catch (err: unknown) {
      if (err instanceof Error && err.name === 'AbortError') return;
      const message = err instanceof Error && err.message ? err.message : 'Failed to load policies';
      setError(message);
      setAllPolicies([]);
    } finally {
      setLoading(false);
    }
  }, [loadRateLimitEvents, loadRateLimitSummary, refreshScopeContext]);

  const policies = useMemo(() => {
    let filtered = allPolicies;
    if (scopeFilter) {
      filtered = filtered.filter((policy) => policy.scope === scopeFilter);
    }
    if (resourceTypeFilter) {
      filtered = filtered.filter((policy) => policy.resource_type === resourceTypeFilter);
    }
    return filtered;
  }, [allPolicies, scopeFilter, resourceTypeFilter]);

  const doesPolicyMatchUserScope = useCallback((policy: ResourcePolicy, user: ScopeUser) => {
    const scopeId = policy.scope_id != null ? String(policy.scope_id).trim() : '';
    if (policy.scope === 'global') return true;
    if (!scopeId) return false;
    if (policy.scope === 'user') {
      return String(user.id) === scopeId;
    }
    if (policy.scope === 'org') {
      const orgIds = orgMembershipsByUserId[String(user.id)];
      return orgIds?.has(scopeId) ?? false;
    }
    if (policy.scope === 'role') {
      const normalizedScope = scopeId.toLowerCase();
      const roles = [user.role, ...(user.roles || [])]
        .filter((value): value is string => Boolean(value))
        .map((value) => value.trim().toLowerCase())
        .filter(Boolean);
      return roles.includes(normalizedScope);
    }
    return false;
  }, [orgMembershipsByUserId]);

  const getPolicyMatchReason = useCallback((policy: ResourcePolicy, user: ScopeUser) => {
    const scopeId = policy.scope_id != null ? String(policy.scope_id).trim() : '';
    if (policy.scope === 'global') {
      return 'Global scope matches all users.';
    }
    if (policy.scope === 'user') {
      return `User scope ${scopeId} matches user ${user.id}.`;
    }
    if (policy.scope === 'org') {
      return `Org scope ${scopeId} matches the user organization membership.`;
    }
    if (policy.scope === 'role') {
      return `Role scope ${scopeId} matches one of the user roles.`;
    }
    return 'Scope matched.';
  }, []);

  const getMatchingUsersForPolicy = useCallback((policy: ResourcePolicy): ScopeUser[] => {
    return scopeUsers.filter((user) => doesPolicyMatchUserScope(policy, user));
  }, [doesPolicyMatchUserScope, scopeUsers]);

  const estimatePolicyImpact = useCallback((policy: ResourcePolicy): PolicySimulationResult => {
    const matchedUsers = getMatchingUsersForPolicy(policy);
    const affectedRequests24h = matchedUsers.reduce((total, user) => {
      return total + (requests24hByUserId[String(user.id)] || 0);
    }, 0);
    return {
      affectedUsers: matchedUsers.length,
      affectedRequests24h,
      source: 'client_estimate',
    };
  }, [getMatchingUsersForPolicy, requests24hByUserId]);

  const affectedUsersByPolicyKey = useMemo(() => {
    const counts: Record<string, number> = {};
    allPolicies.forEach((policy, index) => {
      const key = getPolicyRowKey(policy, index);
      counts[key] = getMatchingUsersForPolicy(policy).length;
    });
    return counts;
  }, [allPolicies, getMatchingUsersForPolicy]);

  useEffect(() => {
    const controller = new AbortController();
    void loadPolicies(controller.signal);
    return () => controller.abort();
  }, [loadPolicies]);

  const handleRefresh = useCallback(() => {
    void loadPolicies();
  }, [loadPolicies]);

  const resetForm = () => {
    policyForm.reset(defaultPolicyFormValues);
    setEditingPolicy(null);
    setSimulationResult(null);
  };

  const handleNewPolicy = () => {
    resetForm();
    setShowForm(true);
  };

  const handleEditPolicy = (policy: ResourcePolicy) => {
    setEditingPolicy(policy);
    setSimulationResult(null);
    policyForm.reset({
      name: policy.name || '',
      scope: policy.scope || 'global',
      scope_id: policy.scope_id?.toString() || '',
      resource_type: (policy.resource_type || 'llm') as PolicyFormData['resource_type'],
      max_requests_per_minute: policy.max_requests_per_minute?.toString() || '',
      max_requests_per_hour: policy.max_requests_per_hour?.toString() || '',
      max_requests_per_day: policy.max_requests_per_day?.toString() || '',
      max_tokens_per_request: policy.max_tokens_per_request?.toString() || '',
      max_concurrent_requests: policy.max_concurrent_requests?.toString() || '',
      priority: policy.priority?.toString() || '0',
      enabled: policy.enabled ?? true,
      description: policy.description || '',
    });
    setShowForm(true);
  };

  const handleCancelForm = () => {
    setShowForm(false);
    resetForm();
  };

  const handleSimulatePolicy = async () => {
    const isValid = await policyForm.trigger();
    if (!isValid) return;

    const formData = policyForm.getValues();
    const scopeId = formData.scope_id?.trim() ?? '';
    const description = formData.description?.trim() ?? '';
    const simulatedPolicy: ResourcePolicy = {
      id: editingPolicy?.id,
      name: formData.name.trim(),
      scope: formData.scope,
      scope_id: formData.scope === 'global' ? null : scopeId,
      resource_type: formData.resource_type,
      enabled: formData.enabled,
      priority: parseOptionalInt(formData.priority) ?? 0,
      max_requests_per_minute: parseOptionalInt(formData.max_requests_per_minute),
      max_requests_per_hour: parseOptionalInt(formData.max_requests_per_hour),
      max_requests_per_day: parseOptionalInt(formData.max_requests_per_day),
      max_tokens_per_request: parseOptionalInt(formData.max_tokens_per_request),
      max_concurrent_requests: parseOptionalInt(formData.max_concurrent_requests),
      description: description || null,
    };

    try {
      setSimulationLoading(true);
      let nextResult: PolicySimulationResult | null = null;

      try {
        const response = await api.simulateResourceGovernorPolicy({
          policy: simulatedPolicy,
          lookback_hours: USAGE_LOOKBACK_HOURS,
        });
        if (response && typeof response === 'object') {
          const record = response as Record<string, unknown>;
          const affectedUsers = Number(
            record.affected_users ?? record.users ?? record.users_affected
          );
          const affectedRequests24h = Number(
            record.affected_requests_24h ?? record.requests_24h ?? record.requests_affected
          );
          if (Number.isFinite(affectedUsers) && Number.isFinite(affectedRequests24h)) {
            nextResult = {
              affectedUsers: Math.max(0, affectedUsers),
              affectedRequests24h: Math.max(0, affectedRequests24h),
              source: 'endpoint',
            };
          }
        }
      } catch {
        // Best-effort endpoint attempt; fallback to client estimation below.
      }

      if (!nextResult) {
        nextResult = estimatePolicyImpact(simulatedPolicy);
      }

      setSimulationResult(nextResult);
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to simulate policy';
      showError(message);
      setSimulationResult(null);
    } finally {
      setSimulationLoading(false);
    }
  };

  const handleResolvePolicy = async () => {
    try {
      setResolutionLoading(true);
      setResolutionError('');
      setResolutionResult(null);

      const userIdValue = resolutionUserId.trim();
      if (!userIdValue) {
        setResolutionError('Enter a user ID to resolve policy scope.');
        return;
      }

      const normalizedUserValue = userIdValue.toLowerCase();
      const matchingSuggestion = userSuggestions.find(
        (entry) =>
          String(entry.id) === userIdValue ||
          entry.username?.toLowerCase() === normalizedUserValue
      );
      const resolvedUserId = matchingSuggestion ? String(matchingSuggestion.id) : userIdValue;
      const user = scopeUsers.find(
        (entry) =>
          String(entry.id) === resolvedUserId ||
          entry.username?.toLowerCase() === normalizedUserValue
      );
      if (!user) {
        setResolutionError(`User ${userIdValue} was not found in the current admin scope.`);
        return;
      }

      const matchingSteps = allPolicies
        .filter((policy) => policy.enabled)
        .filter((policy) => policyResourceMatches(policy, resolutionResourceType))
        .filter((policy) => doesPolicyMatchUserScope(policy, user))
        .map((policy) => ({
          policy,
          priority: getPolicyPriority(policy),
          reason: getPolicyMatchReason(policy, user),
        }))
        .sort((a, b) => {
          if (a.priority !== b.priority) return a.priority - b.priority;
          return a.policy.name.localeCompare(b.policy.name);
        });

      const winner = matchingSteps.length > 0
        ? [...matchingSteps].sort((a, b) => b.priority - a.priority)[0]
        : null;

      setResolutionResult({
        user,
        resourceType: resolutionResourceType,
        steps: matchingSteps,
        winner,
      });
    } finally {
      setResolutionLoading(false);
    }
  };

  const handleSavePolicy = policyForm.handleSubmit(async (data) => {
    try {
      setFormSaving(true);
      const payload: Record<string, unknown> = {
        name: data.name.trim(),
        scope: data.scope,
        resource_type: data.resource_type,
        enabled: data.enabled,
      };

      if (data.scope !== 'global') {
        payload.scope_id = data.scope_id.trim();
      }

      const maxRpm = parseOptionalInt(data.max_requests_per_minute);
      const maxRph = parseOptionalInt(data.max_requests_per_hour);
      const maxRpd = parseOptionalInt(data.max_requests_per_day);
      const maxTokens = parseOptionalInt(data.max_tokens_per_request);
      const maxConcurrent = parseOptionalInt(data.max_concurrent_requests);
      const priority = parseOptionalInt(data.priority);
      const isEditMode = editingPolicy?.id != null;

      if (maxRpm !== null) payload.max_requests_per_minute = maxRpm;
      else if (isEditMode) payload.max_requests_per_minute = null;

      if (maxRph !== null) payload.max_requests_per_hour = maxRph;
      else if (isEditMode) payload.max_requests_per_hour = null;

      if (maxRpd !== null) payload.max_requests_per_day = maxRpd;
      else if (isEditMode) payload.max_requests_per_day = null;

      if (maxTokens !== null) payload.max_tokens_per_request = maxTokens;
      else if (isEditMode) payload.max_tokens_per_request = null;

      if (maxConcurrent !== null) payload.max_concurrent_requests = maxConcurrent;
      else if (isEditMode) payload.max_concurrent_requests = null;

      if (priority !== null) payload.priority = priority;
      else if (isEditMode) payload.priority = null;

      const trimmedDescription = data.description.trim();
      if (isEditMode) {
        payload.description = trimmedDescription || null;
      } else if (trimmedDescription) {
        payload.description = trimmedDescription;
      }

      if (editingPolicy?.id != null) {
        payload.id = editingPolicy.id;
      }

      await api.updateResourceGovernorPolicy(payload);
      success(isEditMode ? 'Policy updated' : 'Policy created');
      setShowForm(false);
      resetForm();
      await loadPolicies();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to save policy';
      showError(message);
    } finally {
      setFormSaving(false);
    }
  });

  const handleDeletePolicy = async (policy: ResourcePolicy) => {
    if (policy.id == null) {
      showError('Cannot delete policy without ID');
      return;
    }
    const policyId = String(policy.id);
    if (deletingPolicyId === policyId) return;

    const approval = await promptPrivilegedAction({
      title: `Delete policy "${policy.name}"?`,
      message: 'This will remove the resource governance policy. This action cannot be undone.',
      confirmText: 'Delete',
      requirePassword: true,
    });

    if (!approval) return;

    try {
      setDeletingPolicyId(policyId);
      await api.deleteResourceGovernorPolicy(policyId);
      success('Policy deleted');
      await loadPolicies();
    } catch (err: unknown) {
      const message = err instanceof Error && err.message ? err.message : 'Failed to delete policy';
      showError(message);
    } finally {
      setDeletingPolicyId((prev) => (prev === policyId ? null : prev));
    }
  };

  const getScopeDisplay = (policy: ResourcePolicy) => {
    if (policy.scope === 'global') return 'Global';
    if (policy.scope === 'org') return `Org ${policy.scope_id}`;
    if (policy.scope === 'user') return `User ${policy.scope_id}`;
    if (policy.scope === 'role') return `Role ${policy.scope_id}`;
    return policy.scope;
  };

  const getLimitsDisplay = (policy: ResourcePolicy) => {
    const parts: string[] = [];
    if (policy.max_requests_per_minute != null) parts.push(`${policy.max_requests_per_minute}/min`);
    if (policy.max_requests_per_hour != null) parts.push(`${policy.max_requests_per_hour}/hr`);
    if (policy.max_requests_per_day != null) parts.push(`${policy.max_requests_per_day}/day`);
    if (policy.max_concurrent_requests != null) parts.push(`${policy.max_concurrent_requests} concurrent`);
    if (policy.max_tokens_per_request != null) parts.push(`${policy.max_tokens_per_request} tokens/req`);
    return parts.length > 0 ? parts.join(', ') : 'No limits set';
  };

  const scopeLabelPrefix = watchedScope === 'org'
    ? 'Organization'
    : watchedScope === 'user'
      ? 'User'
      : 'Role';

  const resolutionChainSummary = useMemo(() => {
    if (!resolutionResult || resolutionResult.steps.length === 0 || !resolutionResult.winner) return '';
    const chain = resolutionResult.steps
      .map((step) => {
        const scopeLabel = step.policy.scope.charAt(0).toUpperCase() + step.policy.scope.slice(1);
        return `${scopeLabel} policy "${step.policy.name}" (priority ${step.priority})`;
      })
      .join(' → ');
    return `${chain} → Winner: ${resolutionResult.winner.policy.name}`;
  }, [resolutionResult]);

  const rateLimitEventsSourceLabel = rateLimitEventsSource === 'endpoint'
    ? 'Endpoint data'
    : rateLimitEventsSource === 'metrics_text'
      ? 'Metrics fallback'
      : 'Unavailable';

  return (
    <PermissionGuard variant="route" requireAuth role="admin">
      <ResponsiveLayout>
        <div className="flex flex-col gap-6 p-6">
          <div className="flex items-center justify-between">
            <div>
              <h1 className="text-2xl font-bold flex items-center gap-2">
                <Gauge className="h-6 w-6" />
                Resource Governor
              </h1>
              <p className="text-muted-foreground">
                Manage rate limits and resource quotas for API operations.
              </p>
            </div>
            <div className="flex gap-2">
              <Button variant="outline" onClick={handleRefresh} disabled={loading}>
                <RefreshCw className={`mr-2 h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
                Refresh
              </Button>
              <Button onClick={handleNewPolicy}>
                <Plus className="mr-2 h-4 w-4" />
                New Policy
              </Button>
            </div>
          </div>

          {error && (
            <Alert variant="destructive">
              <AlertDescription>{error}</AlertDescription>
            </Alert>
          )}

          {scopeContextError && (
            <Alert>
              <AlertDescription>
                Scope context unavailable: {scopeContextError}
              </AlertDescription>
            </Alert>
          )}

          {/* Create/Edit Form */}
          {showForm && (
            <Card>
              <CardHeader>
                <div className="flex items-center justify-between">
                  <div>
                    <CardTitle>{editingPolicy ? 'Edit Policy' : 'New Policy'}</CardTitle>
                    <CardDescription>
                      {editingPolicy ? 'Update the resource governance policy.' : 'Create a new resource governance policy.'}
                    </CardDescription>
                  </div>
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleCancelForm}
                    aria-label="Close policy form"
                    title="Close policy form"
                    disabled={formSaving}
                  >
                    <X className="h-4 w-4" />
                  </Button>
                </div>
              </CardHeader>
              <CardContent>
                <FormProvider {...policyForm}>
                  <Form onSubmit={handleSavePolicy} className="grid gap-4">
                    <div className="grid gap-4 md:grid-cols-2">
                      <FormInput<PolicyFormData>
                        name="name"
                        label="Policy Name"
                        placeholder="e.g., Default LLM Rate Limit"
                        required
                      />
                      <FormSelect<PolicyFormData>
                        name="scope"
                        label="Scope"
                        required
                        options={POLICY_SCOPES.map((scope) => ({
                          value: scope,
                          label: scope.charAt(0).toUpperCase() + scope.slice(1),
                        }))}
                      />
                      {watchedScope !== 'global' && (
                        <FormInput<PolicyFormData>
                          name="scope_id"
                          label={`${scopeLabelPrefix} ID`}
                          placeholder={`Enter ${watchedScope} ID`}
                          required
                        />
                      )}
                      <FormSelect<PolicyFormData>
                        name="resource_type"
                        label="Resource Type"
                        required
                        options={RESOURCE_TYPES.map((type) => ({
                          value: type,
                          label: type.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase()),
                        }))}
                      />
                    </div>

                    <div className="grid gap-4 md:grid-cols-5">
                      <FormInput<PolicyFormData>
                        name="max_requests_per_minute"
                        label="Max Requests/Min"
                        type="number"
                        placeholder="e.g., 60"
                      />
                      <FormInput<PolicyFormData>
                        name="max_requests_per_hour"
                        label="Max Requests/Hour"
                        type="number"
                        placeholder="e.g., 1000"
                      />
                      <FormInput<PolicyFormData>
                        name="max_requests_per_day"
                        label="Max Requests/Day"
                        type="number"
                        placeholder="e.g., 10000"
                      />
                      <FormInput<PolicyFormData>
                        name="max_tokens_per_request"
                        label="Max Tokens/Request"
                        type="number"
                        placeholder="e.g., 4096"
                      />
                      <FormInput<PolicyFormData>
                        name="max_concurrent_requests"
                        label="Max Concurrent"
                        type="number"
                        placeholder="e.g., 10"
                      />
                    </div>

                    <div className="grid gap-4 md:grid-cols-3">
                      <FormInput<PolicyFormData>
                        name="priority"
                        label="Priority (higher = more important)"
                        type="number"
                        placeholder="0"
                      />
                      <div className="pt-6">
                        <FormCheckbox<PolicyFormData>
                          name="enabled"
                          label="Enabled"
                        />
                      </div>
                    </div>

                    <FormInput<PolicyFormData>
                      name="description"
                      label="Description"
                      placeholder="Optional description for this policy"
                    />

                    <div className="flex flex-wrap gap-2">
                      <Button type="submit" loading={formSaving} loadingText="Saving...">
                        {editingPolicy ? 'Update Policy' : 'Create Policy'}
                      </Button>
                      <Button
                        type="button"
                        variant="secondary"
                        onClick={handleSimulatePolicy}
                        disabled={formSaving}
                        loading={simulationLoading}
                        loadingText="Simulating..."
                      >
                        Simulate Impact
                      </Button>
                      <Button type="button" variant="outline" onClick={handleCancelForm} disabled={formSaving}>
                        Cancel
                      </Button>
                    </div>
                    {simulationResult ? (
                      <Alert>
                        <AlertDescription>
                          Would affect {simulationResult.affectedUsers} users / {simulationResult.affectedRequests24h} requests in last 24h.
                          {' '}
                          {simulationResult.source === 'client_estimate'
                            ? '(Client estimate)'
                            : '(Backend simulation)'}
                        </AlertDescription>
                      </Alert>
                    ) : null}
                  </Form>
                </FormProvider>
              </CardContent>
            </Card>
          )}

          <Card>
            <CardHeader>
              <CardTitle>Policy Resolution</CardTitle>
              <CardDescription>
                Enter a user and resource type to see which policy applies and why.
              </CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-4 md:grid-cols-3">
                <div className="space-y-1">
                  <Label htmlFor="resolution-user-id">User</Label>
                  {scopeUsers.length > 0 ? (
                    <Select
                      id="resolution-user-id"
                      value={resolutionUserId}
                      onChange={(event) => setResolutionUserId(event.target.value)}
                    >
                      <option value="">Select a user...</option>
                      {scopeUsers.map((user) => (
                        <option key={user.id} value={String(user.id)}>
                          {user.username || user.email || `User ${user.id}`} (ID: {user.id})
                        </option>
                      ))}
                    </Select>
                  ) : (
                    <Input
                      id="resolution-user-id"
                      placeholder="e.g., 42"
                      value={resolutionUserId}
                      onChange={(event) => setResolutionUserId(event.target.value)}
                    />
                  )}
                </div>
                <div className="space-y-1">
                  <Label htmlFor="resolution-resource-type">Resource Type</Label>
                  <Select
                    id="resolution-resource-type"
                    value={resolutionResourceType}
                    onChange={(event) => setResolutionResourceType(event.target.value as PolicyFormData['resource_type'])}
                  >
                    {RESOURCE_TYPES.map((type) => (
                      <option key={type} value={type}>
                        {type.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="flex items-end">
                  <Button
                    type="button"
                    onClick={handleResolvePolicy}
                    loading={resolutionLoading}
                    loadingText="Resolving..."
                    disabled={scopeContextLoading}
                  >
                    Resolve Policy
                  </Button>
                </div>
              </div>

              {scopeContextLoading ? (
                <div className="text-sm text-muted-foreground">Loading scope context...</div>
              ) : null}

              {resolutionError ? (
                <Alert variant="destructive">
                  <AlertDescription>{resolutionError}</AlertDescription>
                </Alert>
              ) : null}

              {resolutionResult && resolutionResult.steps.length > 0 ? (
                <div className="grid gap-3">
                  <Alert>
                    <AlertDescription>{resolutionChainSummary}</AlertDescription>
                  </Alert>
                  <Table>
                    <TableHeader>
                      <TableRow>
                        <TableHead>Policy</TableHead>
                        <TableHead>Scope</TableHead>
                        <TableHead>Priority</TableHead>
                        <TableHead>Reason</TableHead>
                      </TableRow>
                    </TableHeader>
                    <TableBody>
                      {resolutionResult.steps.map((step, index) => {
                        const isWinner = resolutionResult.winner?.policy === step.policy;
                        return (
                          <TableRow key={`${step.policy.name}-${index}`}>
                            <TableCell className="font-medium">
                              {step.policy.name}
                              {isWinner ? (
                                <Badge variant="default" className="ml-2">Winner</Badge>
                              ) : null}
                            </TableCell>
                            <TableCell>{getScopeDisplay(step.policy)}</TableCell>
                            <TableCell>{step.priority}</TableCell>
                            <TableCell className="text-sm text-muted-foreground">{step.reason}</TableCell>
                          </TableRow>
                        );
                      })}
                    </TableBody>
                  </Table>
                </div>
              ) : null}

              {resolutionResult && resolutionResult.steps.length === 0 ? (
                <Alert>
                  <AlertDescription>
                    No enabled policy matched user {resolutionResult.user.id} for resource type {resolutionResult.resourceType}.
                  </AlertDescription>
                </Alert>
              ) : null}
            </CardContent>
          </Card>

          {/* Rate Limit Analytics Summary */}
          {rateLimitSummary && (rateLimitSummary.total_throttle_events > 0 || rateLimitSummary.policy_headroom.length > 0) ? (
            <Card data-testid="rate-limit-analytics-card">
              <CardHeader>
                <CardTitle>Rate Limit Analytics</CardTitle>
                <CardDescription>
                  Throttle summary and policy utilization for the last {rateLimitSummary.period}.
                </CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid gap-4 md:grid-cols-3 mb-4">
                  <div className="rounded-lg bg-muted/50 p-4 text-center">
                    <div className="text-2xl font-bold tabular-nums" data-testid="total-throttle-events">
                      {rateLimitSummary.total_throttle_events}
                    </div>
                    <div className="text-sm text-muted-foreground">Total Throttle Events</div>
                  </div>
                  <div className="rounded-lg bg-muted/50 p-4 text-center">
                    <div className="text-2xl font-bold tabular-nums">
                      {rateLimitSummary.top_throttled_entities.length}
                    </div>
                    <div className="text-sm text-muted-foreground">Affected Entities</div>
                  </div>
                  <div className="rounded-lg bg-muted/50 p-4 text-center">
                    <div className="text-2xl font-bold tabular-nums">
                      {rateLimitSummary.policy_headroom.length}
                    </div>
                    <div className="text-sm text-muted-foreground">Active Policies</div>
                  </div>
                </div>

                {rateLimitSummary.top_throttled_entities.length > 0 ? (
                  <div className="mb-4">
                    <h4 className="text-sm font-medium mb-2">Top Throttled Entities</h4>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Entity</TableHead>
                          <TableHead className="text-right">Rejections</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {rateLimitSummary.top_throttled_entities.slice(0, 5).map((entity, index) => (
                          <TableRow key={`${entity.entity}-${index}`}>
                            <TableCell className="font-medium">{entity.entity}</TableCell>
                            <TableCell className="text-right tabular-nums">{entity.rejections}</TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : null}

                {rateLimitSummary.policy_headroom.length > 0 ? (
                  <div>
                    <h4 className="text-sm font-medium mb-2">Policy Utilization</h4>
                    <Table>
                      <TableHeader>
                        <TableRow>
                          <TableHead>Policy</TableHead>
                          <TableHead>Resource</TableHead>
                          <TableHead className="text-right">Denials</TableHead>
                          <TableHead className="text-right">Utilization</TableHead>
                        </TableRow>
                      </TableHeader>
                      <TableBody>
                        {rateLimitSummary.policy_headroom.slice(0, 5).map((policy, index) => (
                          <TableRow key={`${policy.policy_id}-${index}`}>
                            <TableCell className="font-mono text-sm">{policy.policy_id}</TableCell>
                            <TableCell>
                              {policy.resource_type ? (
                                <Badge variant="outline">
                                  {policy.resource_type.replaceAll('_', ' ')}
                                </Badge>
                              ) : '--'}
                            </TableCell>
                            <TableCell className="text-right tabular-nums">{policy.total_denials}</TableCell>
                            <TableCell className="text-right tabular-nums">
                              <Badge variant={policy.utilization_pct > 50 ? 'destructive' : policy.utilization_pct > 20 ? 'secondary' : 'default'}>
                                {policy.utilization_pct.toFixed(1)}%
                              </Badge>
                            </TableCell>
                          </TableRow>
                        ))}
                      </TableBody>
                    </Table>
                  </div>
                ) : null}
              </CardContent>
            </Card>
          ) : rateLimitSummaryLoading ? (
            <Card>
              <CardContent className="py-6">
                <div className="text-sm text-muted-foreground text-center">Loading rate limit analytics...</div>
              </CardContent>
            </Card>
          ) : null}

          <Card>
            <CardHeader>
              <div className="flex items-center justify-between gap-3">
                <div>
                  <CardTitle>Rate Limit Events</CardTitle>
                  <CardDescription>
                    Recent rate-limit rejections by user/role and policy over the last {USAGE_LOOKBACK_HOURS} hours.
                  </CardDescription>
                </div>
                <div className="flex items-center gap-2">
                  {!rateLimitEventsLoading && rateLimitEvents.length > 0 ? (
                    <Badge variant="destructive" className="text-xs" data-testid="throttle-event-count">
                      {rateLimitEvents.reduce((sum, e) => sum + (e.rejections24h ?? 0), 0)} rejections
                    </Badge>
                  ) : null}
                  <Badge variant="outline" className="text-xs">
                    {rateLimitEventsSourceLabel}
                  </Badge>
                </div>
              </div>
              {!rateLimitEventsLoading && rateLimitEvents.length > 0 ? (
                <Link
                  href="/audit?action=rate_limit"
                  className="mt-1 inline-block text-xs text-primary hover:underline"
                  data-testid="throttle-audit-link"
                >
                  View in audit log
                </Link>
              ) : null}
            </CardHeader>
            <CardContent className="grid gap-4">
              {rateLimitEventsLoading ? (
                <div className="text-sm text-muted-foreground">Loading rate limit events...</div>
              ) : null}

              {rateLimitEventsError ? (
                <Alert variant="destructive">
                  <AlertDescription>{rateLimitEventsError}</AlertDescription>
                </Alert>
              ) : null}

              {!rateLimitEventsLoading && rateLimitEvents.length === 0 ? (
                <div className="text-sm text-muted-foreground">
                  No rate limit rejections were found for the current data source.
                </div>
              ) : null}

              {!rateLimitEventsLoading && rateLimitEvents.length > 0 ? (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>User/Role</TableHead>
                      <TableHead>Policy</TableHead>
                      <TableHead className="text-right">Rejections (24h)</TableHead>
                      <TableHead>Last Rejection</TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {rateLimitEvents.map((event, index) => (
                      <TableRow key={`${event.actor}-${event.policy}-${index}`}>
                        <TableCell className="font-medium">{event.actor}</TableCell>
                        <TableCell>
                          <div>{event.policy}</div>
                          {(event.resourceType || event.reason) ? (
                            <div className="text-xs text-muted-foreground">
                              {[event.resourceType, event.reason].filter(Boolean).join(' • ')}
                            </div>
                          ) : null}
                        </TableCell>
                        <TableCell className="text-right tabular-nums">{event.rejections24h}</TableCell>
                        <TableCell className="text-sm text-muted-foreground">
                          {formatPolicyDate(event.lastRejectedAt)}
                        </TableCell>
                      </TableRow>
                    ))}
                  </TableBody>
                </Table>
              ) : null}
            </CardContent>
          </Card>

          {/* Policies List */}
          <Card>
            <CardHeader>
              <CardTitle>Policies</CardTitle>
              <CardDescription>Resource governance policies in order of priority.</CardDescription>
            </CardHeader>
            <CardContent className="grid gap-4">
              <div className="grid gap-4 md:grid-cols-2">
                <div className="space-y-1">
                  <Label htmlFor="scope-filter">Filter by Scope</Label>
                  <Select
                    id="scope-filter"
                    value={scopeFilter}
                    onChange={(e) => setScopeFilter(e.target.value)}
                  >
                    <option value="">All Scopes</option>
                    {POLICY_SCOPES.map((scope) => (
                      <option key={scope} value={scope}>
                        {scope.charAt(0).toUpperCase() + scope.slice(1)}
                      </option>
                    ))}
                  </Select>
                </div>
                <div className="space-y-1">
                  <Label htmlFor="resource-filter">Filter by Resource Type</Label>
                  <Select
                    id="resource-filter"
                    value={resourceTypeFilter}
                    onChange={(e) => setResourceTypeFilter(e.target.value)}
                  >
                    <option value="">All Resources</option>
                    {RESOURCE_TYPES.map((type) => (
                      <option key={type} value={type}>
                        {type.replaceAll('_', ' ').replace(/\b\w/g, (letter) => letter.toUpperCase())}
                      </option>
                    ))}
                  </Select>
                </div>
              </div>

              {loading ? (
                <div className="py-8 text-center text-muted-foreground">Loading policies...</div>
              ) : policies.length === 0 ? (
                <div className="py-8 text-center text-muted-foreground">
                  No policies found. Click &quot;New Policy&quot; to create one.
                </div>
              ) : (
                <Table>
                  <TableHeader>
                    <TableRow>
                      <TableHead>Name</TableHead>
                      <TableHead>Scope</TableHead>
                      <TableHead>Resource</TableHead>
                      <TableHead>Affected Users</TableHead>
                      <TableHead>Limits</TableHead>
                      <TableHead>Priority</TableHead>
                      <TableHead>Status</TableHead>
                      <TableHead>Updated</TableHead>
                      <TableHead></TableHead>
                    </TableRow>
                  </TableHeader>
                  <TableBody>
                    {policies.map((policy, index) => {
                      const rowKey = getPolicyRowKey(policy, index);
                      const policyId = policy.id != null ? String(policy.id) : '';
                      const isDeleting = policyId !== '' && deletingPolicyId === policyId;
                      return (
                        <TableRow key={rowKey}>
                          <TableCell className="font-medium">
                            <div>{policy.name}</div>
                            {policy.description && (
                              <div className="text-xs text-muted-foreground">{policy.description}</div>
                            )}
                          </TableCell>
                          <TableCell>{getScopeDisplay(policy)}</TableCell>
                          <TableCell>
                            <Badge variant="outline">
                              {policy.resource_type?.replaceAll('_', ' ') || 'all'}
                            </Badge>
                          </TableCell>
                          <TableCell>
                            {scopeContextLoading ? '…' : (affectedUsersByPolicyKey[rowKey] ?? '—')}
                          </TableCell>
                          <TableCell className="text-sm">{getLimitsDisplay(policy)}</TableCell>
                          <TableCell>{policy.priority ?? 0}</TableCell>
                          <TableCell>
                            <Badge variant={policy.enabled ? 'default' : 'secondary'}>
                              {policy.enabled ? 'Enabled' : 'Disabled'}
                            </Badge>
                          </TableCell>
                          <TableCell className="text-sm text-muted-foreground">
                            {formatPolicyDate(policy.updated_at)}
                          </TableCell>
                          <TableCell>
                            <div className="flex gap-1">
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => handleEditPolicy(policy)}
                                title="Edit policy"
                                aria-label={`Edit policy ${policy.name}`}
                              >
                                <Edit2 className="h-4 w-4" />
                              </Button>
                              <Button
                                variant="ghost"
                                size="icon"
                                onClick={() => handleDeletePolicy(policy)}
                                title={isDeleting ? 'Deleting policy' : 'Delete policy'}
                                aria-label={isDeleting ? 'Deleting policy' : 'Delete policy'}
                                disabled={isDeleting}
                                loading={isDeleting}
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
        </div>
      </ResponsiveLayout>
    </PermissionGuard>
  );
}
