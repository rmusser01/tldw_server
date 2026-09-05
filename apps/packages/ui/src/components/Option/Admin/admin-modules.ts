/**
 * Single source of truth for the admin surface.
 *
 * Every admin module (route, operator-facing label, one-line purpose, group)
 * is declared here once. The admin overview page, the cross-module nav, and
 * document titles all derive from this list — never maintain a second,
 * partial list of admin routes elsewhere (2026-09 UX audit finding S1).
 */

export type AdminModuleGroup =
  | "Server"
  | "Access & security"
  | "Data"
  | "Local models"
  | "Observability"
  | "Workspace"

export interface AdminModule {
  route: string
  label: string
  description: string
  group: AdminModuleGroup
  /** Route exists but the surface is a placeholder - badge it so operators
   *  know before clicking (#2897). */
  comingSoon?: boolean
}

export const ADMIN_MODULE_GROUPS: AdminModuleGroup[] = [
  "Server",
  "Access & security",
  "Data",
  "Local models",
  "Observability",
  "Workspace"
]

export const ADMIN_MODULES: AdminModule[] = [
  {
    route: "/admin/server",
    label: "Server Admin",
    description:
      "Server health, users, roles, storage, sessions, and media budget diagnostics.",
    group: "Server"
  },
  {
    route: "/admin/maintenance",
    label: "Maintenance",
    description:
      "Maintenance mode, feature flags, and incident tracking for this server.",
    group: "Server"
  },
  {
    route: "/admin/runtime-config",
    label: "Runtime Config",
    description: "Inspect and adjust runtime configuration values.",
    group: "Server"
  },
  {
    route: "/admin/api-keys",
    label: "API Keys",
    description: "Create, rotate, and revoke the API keys users authenticate with.",
    group: "Access & security"
  },
  {
    route: "/admin/rbac",
    label: "Roles & Permissions",
    description: "Role-based access control: the permission matrix and role grants.",
    group: "Access & security"
  },
  {
    route: "/admin/orgs",
    label: "Organizations & Teams",
    description: "Organization and team structure for multi-user deployments.",
    group: "Access & security"
  },
  {
    route: "/admin/rate-limiting",
    label: "Rate Limiting",
    description: "Resource governor policy and endpoint rate-limit coverage.",
    group: "Access & security"
  },
  {
    route: "/admin/data-ops",
    label: "Data Operations",
    description:
      "Backups and schedules, data subject requests, retention policies, and bundles.",
    group: "Data"
  },
  {
    route: "/admin/sources",
    label: "Sources",
    description:
      "Local folders and archive snapshots that sync into notes or media.",
    group: "Data"
  },
  {
    route: "/admin/llamacpp",
    label: "Llama.cpp",
    description:
      "Manage the llama.cpp inference server: launch, models, profiles, assets.",
    group: "Local models"
  },
  {
    route: "/admin/mlx",
    label: "MLX LM",
    description: "Load and monitor MLX models on Apple Silicon.",
    group: "Local models"
  },
  {
    route: "/admin/monitoring",
    label: "Monitoring",
    description: "Health metrics, alerts, and operations telemetry for this server.",
    group: "Observability"
  },
  {
    route: "/admin/usage",
    label: "Usage Analytics",
    description: "Request, storage, and LLM usage over time, with CSV export.",
    group: "Observability"
  },
  {
    route: "/admin/billing",
    label: "Billing",
    description:
      "Subscriptions and billing events (multi-user deployments only).",
    group: "Observability"
  },
  {
    route: "/admin/integrations",
    label: "Workspace Integrations",
    description:
      "Slack, Discord, and Telegram workspace policy and linked actors.",
    group: "Workspace"
  },
  {
    route: "/admin/watchlists-items",
    label: "Watchlist Items",
    description: "Review collected watchlist updates, matches, and briefings.",
    group: "Workspace"
  },
  {
    route: "/admin/watchlists-runs",
    label: "Watchlist Runs",
    description: "Run history and job inspection for watchlists (coming soon).",
    group: "Workspace",
    comingSoon: true
  }
]

const normalizeAdminPath = (path: string): string => {
  const withoutQuery = path.split(/[?#]/)[0] ?? ""
  return withoutQuery.length > 1 && withoutQuery.endsWith("/")
    ? withoutQuery.slice(0, -1)
    : withoutQuery
}

export const adminModuleForRoute = (path: string): AdminModule | undefined => {
  const normalized = normalizeAdminPath(path)
  return ADMIN_MODULES.find((module) => module.route === normalized)
}

export const isAdminRoute = (path: string): boolean => {
  const normalized = normalizeAdminPath(path)
  return normalized === "/admin" || normalized.startsWith("/admin/")
}
