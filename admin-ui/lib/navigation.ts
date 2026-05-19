import type { LucideIcon } from 'lucide-react';
import {
  Activity,
  BarChart3,
  Bot,
  Brain,
  Bug,
  Building2,
  Cpu,
  CreditCard,
  Database,
  PlugZap,
  FileText,
  Flag,
  Gauge,
  Grid3X3,
  AlertTriangle,
  MessageSquare,
  Mic,
  Receipt,
  ScrollText,
  Server,
  Key,
  KeyRound,
  LayoutDashboard,
  Mail,
  ListChecks,
  Settings,
  Shield,
  ShieldAlert,
  ShieldCheck,
  UserCog,
  UserPlus,
  Users,
  Wallet,
  Webhook,
} from 'lucide-react';

export type NavigationItem = {
  name: string;
  href: string;
  icon: LucideIcon;
  permission?: string;
  role?: string[];
  keywords?: string[];
  billingOnly?: boolean;
};

export type NavigationSection = {
  title: string;
  items: NavigationItem[];
};

export type BreadcrumbItem = {
  label: string;
  href?: string;
  current: boolean;
};

const navigationItems = {
  dashboard: { name: 'Dashboard', href: '/', icon: LayoutDashboard, keywords: ['overview', 'home', 'stats'] },
  users: { name: 'Users', href: '/users', icon: Users, permission: 'read:users', keywords: ['accounts', 'people'] },
  organizations: { name: 'Organizations', href: '/organizations', icon: Building2, permission: 'read:orgs', keywords: ['orgs', 'tenants'] },
  teams: { name: 'Teams', href: '/teams', icon: UserCog, permission: 'read:teams', keywords: ['groups'] },
  rolesPermissions: { name: 'Roles & Permissions', href: '/roles', icon: Shield, role: ['admin', 'super_admin', 'owner'], keywords: ['rbac', 'authz'] },
  registrationCodes: { name: 'Registration Codes', href: '/users/registration', icon: KeyRound, permission: 'read:users', keywords: ['registration', 'invite', 'onboarding', 'codes'] },
  apiKeys: { name: 'API Keys', href: '/api-keys', icon: Key, permission: 'read:api_keys', keywords: ['credentials', 'tokens'] },
  registration: { name: 'Registration', href: '/users/registration', icon: UserPlus, role: ['admin', 'super_admin', 'owner'], keywords: ['invite', 'onboard', 'codes'] },
  invitations: { name: 'Invitations', href: '/invitations', icon: Mail, role: ['admin', 'super_admin', 'owner'], keywords: ['invite', 'org invite', 'invite code'] },
  byok: { name: 'BYOK', href: '/byok', icon: KeyRound, role: ['admin', 'super_admin', 'owner'], keywords: ['provider keys', 'bring your own key'] },
  aiOverview: { name: 'AI Overview', href: '/ai-overview', icon: Brain, role: ['admin', 'super_admin', 'owner'], keywords: ['ai', 'spend', 'tokens', 'agents', 'governance'] },
  providers: { name: 'LLM Providers', href: '/providers', icon: Cpu, permission: 'read:config', keywords: ['models', 'inference'] },
  resourceGovernor: { name: 'Resource Governor', href: '/resource-governor', icon: Gauge, role: ['admin', 'super_admin', 'owner'], keywords: ['limits', 'quotas', 'rate limits'] },
  compliance: { name: 'Compliance', href: '/compliance', icon: ShieldCheck, role: ['admin', 'super_admin', 'owner'], keywords: ['compliance', 'posture', 'mfa adoption', 'key rotation', 'score'] },
  security: { name: 'Security', href: '/security', icon: ShieldAlert, role: ['admin', 'super_admin', 'owner'], keywords: ['risk', 'mfa', 'sessions'] },
  auditLogs: { name: 'Audit Logs', href: '/audit', icon: FileText, permission: 'read:audit', keywords: ['audit', 'events', 'history'] },
  monitoring: { name: 'Monitoring', href: '/monitoring', icon: Activity, role: ['admin', 'super_admin', 'owner'], keywords: ['health', 'alerts', 'metrics'] },
  dependencies: { name: 'Dependencies', href: '/dependencies', icon: PlugZap, role: ['admin', 'super_admin', 'owner'], keywords: ['providers', 'connectivity', 'health checks'] },
  jobs: { name: 'Jobs', href: '/jobs', icon: ListChecks, role: ['admin', 'super_admin', 'owner'], keywords: ['queue', 'workers', 'tasks'] },
  usage: { name: 'Usage', href: '/usage', icon: BarChart3, role: ['admin', 'super_admin', 'owner'], keywords: ['analytics', 'consumption'] },
  budgets: { name: 'Budgets', href: '/budgets', icon: Wallet, role: ['admin', 'super_admin', 'owner'], keywords: ['cost', 'spend'] },
  dataOps: { name: 'Data Ops', href: '/data-ops', icon: Database, role: ['admin', 'super_admin', 'owner'], keywords: ['backup', 'retention', 'export'] },
  logs: { name: 'Logs', href: '/logs', icon: ScrollText, role: ['admin', 'super_admin', 'owner'], keywords: ['system logs'] },
  flags: { name: 'Flags', href: '/flags', icon: Flag, role: ['admin', 'super_admin', 'owner'], keywords: ['feature flags', 'maintenance'] },
  incidents: { name: 'Incidents', href: '/incidents', icon: AlertTriangle, role: ['admin', 'super_admin', 'owner'], keywords: ['outages', 'response'] },
  aiOps: { name: 'AI Operations', href: '/ai-ops', icon: Activity, role: ['admin', 'super_admin', 'owner'], keywords: ['ai', 'spend', 'tokens', 'agents', 'cost', 'operations'] },
  acpSessions: { name: 'ACP Sessions', href: '/acp-sessions', icon: MessageSquare, role: ['admin', 'super_admin', 'owner'], keywords: ['agent', 'chat', 'sessions', 'acp'] },
  acpAgents: { name: 'ACP Agents', href: '/acp-agents', icon: Bot, role: ['admin', 'super_admin', 'owner'], keywords: ['agent', 'config', 'custom agents'] },
  mcpServers: { name: 'MCP Servers', href: '/mcp-servers', icon: Server, role: ['admin', 'super_admin', 'owner'], keywords: ['mcp', 'tools', 'servers', 'model context protocol'] },
  voiceCommands: { name: 'Voice Commands', href: '/voice-commands', icon: Mic, role: ['admin', 'super_admin', 'owner'], keywords: ['speech', 'commands'] },
  debug: { name: 'Debug', href: '/debug', icon: Bug, role: ['super_admin', 'owner'], keywords: ['diagnostics'] },
  configuration: { name: 'Configuration', href: '/config', icon: Settings, role: ['admin', 'super_admin', 'owner'], keywords: ['settings', 'system config'] },
  plans: { name: 'Plans', href: '/plans', icon: CreditCard, role: ['admin', 'super_admin', 'owner'], keywords: ['billing', 'pricing', 'subscription', 'tiers'], billingOnly: true },
  subscriptions: { name: 'Subscriptions', href: '/subscriptions', icon: Receipt, role: ['admin', 'super_admin', 'owner'], keywords: ['billing', 'payments', 'invoices'], billingOnly: true },
  revenueAnalytics: { name: 'Revenue Analytics', href: '/billing/analytics', icon: BarChart3, role: ['admin', 'super_admin', 'owner'], keywords: ['billing', 'mrr', 'revenue', 'metrics', 'analytics'], billingOnly: true },
  featureRegistry: { name: 'Feature Registry', href: '/feature-registry', icon: Grid3X3, role: ['admin', 'super_admin', 'owner'], keywords: ['gating', 'entitlements', 'open core'], billingOnly: true },
  webhooks: { name: 'Webhooks', href: '/webhooks', icon: Webhook, role: ['admin', 'super_admin', 'owner'], keywords: ['hooks', 'notifications', 'events', 'integrations'] },
} satisfies Record<string, NavigationItem>;

// Grouped navigation for sidebar sections
export const navigationSections: NavigationSection[] = [
  {
    title: 'Overview',
    items: [
      navigationItems.dashboard,
    ],
  },
  {
    title: 'Identity & Access',
    items: [
      navigationItems.users,
      navigationItems.registrationCodes,
      navigationItems.organizations,
      navigationItems.teams,
      navigationItems.rolesPermissions,
      navigationItems.apiKeys,
      navigationItems.registration,
      navigationItems.invitations,
    ],
  },
  {
    title: 'AI & Models',
    items: [
      navigationItems.aiOps,
      navigationItems.aiOverview,
      navigationItems.providers,
      navigationItems.byok,
      navigationItems.acpSessions,
      navigationItems.acpAgents,
      navigationItems.mcpServers,
      navigationItems.voiceCommands,
    ],
  },
  {
    title: 'Operations',
    items: [
      navigationItems.monitoring,
      navigationItems.dependencies,
      navigationItems.incidents,
      navigationItems.webhooks,
      navigationItems.jobs,
      navigationItems.auditLogs,
      navigationItems.logs,
    ],
  },
  {
    title: 'Security & Compliance',
    items: [
      navigationItems.security,
      navigationItems.compliance,
      navigationItems.resourceGovernor,
      navigationItems.flags,
      navigationItems.dataOps,
    ],
  },
  {
    title: 'Cost & Usage',
    items: [
      navigationItems.usage,
      navigationItems.budgets,
      navigationItems.plans,
      navigationItems.subscriptions,
      navigationItems.revenueAnalytics,
      navigationItems.featureRegistry,
    ],
  },
  {
    title: 'Integrations',
    items: [
      navigationItems.webhooks,
    ],
  },
  {
    title: 'Advanced',
    items: [
      navigationItems.configuration,
      navigationItems.debug,
    ],
  },
];

export const matchesNavigationQuery = (
  item: NavigationItem,
  sectionTitle: string,
  query: string
): boolean => {
  const normalizedQuery = query.trim().toLowerCase();
  if (!normalizedQuery) return true;
  const searchableParts = [
    item.name,
    sectionTitle,
    ...(item.keywords ?? []),
  ];
  return searchableParts.some((part) => part.toLowerCase().includes(normalizedQuery));
};

// Flat navigation list for backwards compatibility
export const navigation: NavigationItem[] = navigationSections.flatMap((section) => section.items);

const routeLabelOverrides: Record<string, string> = {
  '/roles/matrix': 'Permission Matrix',
  '/roles/compare': 'Role Comparison',
};

const staticRouteLabels: Record<string, string> = navigation.reduce<Record<string, string>>(
  (accumulator, item) => ({
    ...accumulator,
    [item.href]: item.name,
  }),
  {
    '/': 'Dashboard',
    ...routeLabelOverrides,
  }
);

const normalizePathname = (pathname: string): string => {
  if (!pathname) return '/';
  const withoutQuery = pathname.split('?')[0]?.split('#')[0] ?? '/';
  const trimmed = withoutQuery.replace(/\/+$/, '');
  return trimmed || '/';
};

const titleCase = (value: string): string =>
  value
    .split(' ')
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(' ');

const humanizeSegment = (segment: string): string => {
  const decoded = decodeURIComponent(segment);
  return titleCase(decoded.replace(/[-_]+/g, ' '));
};

const resolveDynamicPathLabel = (segments: string[]): string | null => {
  if (segments.length < 2) return null;
  const [root, idOrSlug] = segments;
  if (!idOrSlug) return null;

  if (root === 'users') {
    if (segments.length === 2 && idOrSlug === 'registration') return 'Registration Codes';
    if (segments.length === 2) return `User ${decodeURIComponent(idOrSlug)}`;
    if (segments.length === 3 && segments[2] === 'api-keys') return 'API Keys';
  }
  if (root === 'organizations' && segments.length === 2) {
    return `Organization ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'teams' && segments.length === 2) {
    return `Team ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'roles' && segments.length === 2) {
    return `Role ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'voice-commands' && segments.length === 2) {
    return `Command ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'acp-sessions' && segments.length === 2) {
    return `Session ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'acp-agents' && segments.length === 2) {
    return `Agent ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'plans' && segments.length === 2) {
    return `Plan ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'subscriptions' && segments.length === 2) {
    return `Subscription ${decodeURIComponent(idOrSlug)}`;
  }
  if (root === 'billing' && idOrSlug === 'analytics') {
    return 'Revenue Analytics';
  }
  return null;
};

const resolveRouteLabel = (path: string): string => {
  const normalized = normalizePathname(path);
  const staticLabel = staticRouteLabels[normalized];
  if (staticLabel) return staticLabel;

  const segments = normalized.split('/').filter(Boolean);
  const dynamicLabel = resolveDynamicPathLabel(segments);
  if (dynamicLabel) return dynamicLabel;

  const lastSegment = segments[segments.length - 1];
  return lastSegment ? humanizeSegment(lastSegment) : 'Dashboard';
};

export const buildBreadcrumbs = (pathname: string): BreadcrumbItem[] => {
  const normalized = normalizePathname(pathname);
  if (normalized === '/') {
    return [{ label: 'Dashboard', current: true }];
  }

  const segments = normalized.split('/').filter(Boolean);
  const items: BreadcrumbItem[] = [{ label: 'Dashboard', href: '/', current: false }];

  segments.forEach((_, index) => {
    const href = `/${segments.slice(0, index + 1).join('/')}`;
    const current = index === segments.length - 1;
    items.push({
      label: resolveRouteLabel(href),
      href: current ? undefined : href,
      current,
    });
  });

  return items;
};

export const getPageTitleForPath = (pathname: string): string => {
  const breadcrumbs = buildBreadcrumbs(pathname);
  const currentLabel = breadcrumbs[breadcrumbs.length - 1]?.label || 'Dashboard';
  return `${currentLabel} | Admin Dashboard`;
};
