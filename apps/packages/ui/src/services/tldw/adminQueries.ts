import { useQuery } from "@tanstack/react-query"
import { tldwClient } from "./TldwApiClient"

/**
 * Query keys + thin hooks for admin reference data (admin perf B-S4 / F11).
 *
 * Reference data (system stats, users pages, roles, permissions) is fetched
 * once per staleTime window no matter how many admin surfaces mount: every
 * hook and the module-signal probes share these keys and the admin
 * QueryClient provided by `AdminQueryProvider` (mounted by AdminRouteShell).
 *
 * Mutations elsewhere must invalidate via
 * `queryClient.invalidateQueries({ queryKey: adminKeys... })` after edits so
 * every mounted consumer refetches.
 */
export const adminKeys = {
  /** Prefix covering every admin query (used to drop the cache when the
   *  connection target changes). */
  all: ["admin"] as const,
  systemStats: ["admin", "system-stats"] as const,
  usersPage: (page: number, limit: number, search?: string) =>
    ["admin", "users", page, limit, search ?? ""] as const,
  roles: ["admin", "roles"] as const,
  permissions: ["admin", "permissions"] as const,
  // Module-signal probes (admin-module-signals.ts). systemStats is shared
  // with the pages above so the overview probe and the dashboards reuse one
  // fetch; the remaining probes are overview-cheap endpoints that get the
  // same stale-window treatment.
  securityAlertStatus: ["admin", "security-alert-status"] as const,
  backups: ["admin", "backups"] as const,
  llamacppStatus: ["admin", "llamacpp-status"] as const,
  mlxStatus: ["admin", "mlx-status"] as const,
  governorCoverage: ["admin", "governor-coverage"] as const
}

export const useSystemStats = (options?: { timeoutMs?: number }) =>
  useQuery({
    queryKey: adminKeys.systemStats,
    queryFn: () => tldwClient.getSystemStats(options)
  })

export const useUsersPage = (page: number, limit: number, search?: string) =>
  useQuery({
    queryKey: adminKeys.usersPage(page, limit, search),
    queryFn: () => tldwClient.listAdminUsers({ page, limit, search })
  })

export const useAdminRoles = () =>
  useQuery({
    queryKey: adminKeys.roles,
    queryFn: () => tldwClient.listAdminRoles()
  })

export const useAdminPermissions = () =>
  useQuery({
    queryKey: adminKeys.permissions,
    queryFn: () => tldwClient.listPermissions()
  })
