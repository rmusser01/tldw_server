import { getRouteMetadata } from "./route-metadata"

export type OperationsRouteConcept =
  | "admin"
  | "mcp"
  | "source"
  | "connector"
  | "integration"
  | "schedule"
  | "watchlist"
  | "workflow"
  | "skill"

export type OperationsCapabilityMode =
  | "frontend_state"
  | "existing_probe"
  | "placeholder"
  | "backend_gate"

export type OperationsRouteJob = {
  route: string
  concept: OperationsRouteConcept
  label: string
  primaryJob: string
  primaryActionLabel: string
  capabilityMode: OperationsCapabilityMode
  diagnosticsPolicy: "disclosed" | "not_applicable"
  implementationOwner: "shared_route" | "next_page"
  relatedRoutes?: readonly string[]
}

const metadataLabel = (route: string, fallback: string): string =>
  getRouteMetadata(route)?.label ?? fallback

export const OPERATIONS_ROUTE_JOBS: OperationsRouteJob[] = [
  {
    route: "/admin",
    concept: "admin",
    label: metadataLabel("/admin", "Admin"),
    primaryJob: "Review operations status and choose an admin module",
    primaryActionLabel: "Open server admin",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "next_page",
    relatedRoutes: [
      "/admin/server",
      "/admin/integrations",
      "/admin/sources",
      "/admin/monitoring"
    ]
  },
  {
    route: "/admin/server",
    concept: "admin",
    label: "Server Admin",
    primaryJob: "Inspect server health, users, roles, and media budgets",
    primaryActionLabel: "Refresh stats",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/admin"]
  },
  {
    route: "/admin/integrations",
    concept: "integration",
    label: "Workspace Integrations",
    primaryJob: "Manage workspace-level integration policy",
    primaryActionLabel: "Review policies",
    capabilityMode: "existing_probe",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/admin", "/integrations"]
  },
  {
    route: "/admin/sources",
    concept: "source",
    label: "Admin Sources",
    primaryJob: "Manage source availability and sync state as an admin",
    primaryActionLabel: "New source",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/admin", "/sources"]
  },
  {
    route: "/admin/monitoring",
    concept: "admin",
    label: "Monitoring",
    primaryJob: "Inspect monitoring and operations metrics",
    primaryActionLabel: "Refresh metrics",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/admin"]
  },
  {
    route: "/mcp-hub",
    concept: "mcp",
    label: metadataLabel("/mcp-hub", "MCP Hub"),
    primaryJob: "Manage external tool servers and governance workflows",
    primaryActionLabel: "Check servers",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  },
  {
    route: "/sources",
    concept: "source",
    label: metadataLabel("/sources", "Sources"),
    primaryJob: "Manage ingestion sources and sync status",
    primaryActionLabel: "New source",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/admin/sources"]
  },
  {
    route: "/connectors",
    concept: "connector",
    label: metadataLabel("/connectors", "Connectors"),
    primaryJob: "Understand connector availability and current alternatives",
    primaryActionLabel: "Open settings",
    capabilityMode: "placeholder",
    diagnosticsPolicy: "not_applicable",
    implementationOwner: "next_page",
    relatedRoutes: [
      "/connectors/browse",
      "/connectors/jobs",
      "/connectors/sources",
      "/integrations",
      "/sources",
      "/scheduled-tasks"
    ]
  },
  {
    route: "/connectors/browse",
    concept: "connector",
    label: "Connector Catalog",
    primaryJob: "Explain planned connector catalog availability",
    primaryActionLabel: "Back to connectors",
    capabilityMode: "placeholder",
    diagnosticsPolicy: "not_applicable",
    implementationOwner: "next_page",
    relatedRoutes: ["/connectors", "/integrations"]
  },
  {
    route: "/connectors/jobs",
    concept: "connector",
    label: "Connector Jobs",
    primaryJob: "Explain planned connector job orchestration availability",
    primaryActionLabel: "Open scheduled tasks",
    capabilityMode: "placeholder",
    diagnosticsPolicy: "not_applicable",
    implementationOwner: "next_page",
    relatedRoutes: ["/connectors", "/scheduled-tasks", "/watchlists"]
  },
  {
    route: "/connectors/sources",
    concept: "connector",
    label: "Connector Sources",
    primaryJob: "Explain planned connector-source workflow availability",
    primaryActionLabel: "Open sources",
    capabilityMode: "placeholder",
    diagnosticsPolicy: "not_applicable",
    implementationOwner: "next_page",
    relatedRoutes: ["/connectors", "/sources"]
  },
  {
    route: "/integrations",
    concept: "integration",
    label: metadataLabel("/integrations", "Integrations"),
    primaryJob: "Manage personal Slack and Discord connections",
    primaryActionLabel: "Refresh all",
    capabilityMode: "existing_probe",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/admin/integrations"]
  },
  {
    route: "/scheduled-tasks",
    concept: "schedule",
    label: metadataLabel("/scheduled-tasks", "Scheduled Tasks"),
    primaryJob: "Manage reminder tasks and endpoint availability",
    primaryActionLabel: "Create reminder",
    capabilityMode: "existing_probe",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/connectors/jobs", "/watchlists"]
  },
  {
    route: "/watchlists",
    concept: "watchlist",
    label: metadataLabel("/watchlists", "Watchlists"),
    primaryJob: "Monitor feeds, jobs, runs, articles, reports, and templates",
    primaryActionLabel: "Set up feeds",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route",
    relatedRoutes: ["/scheduled-tasks", "/connectors/jobs"]
  },
  {
    route: "/workflow-editor",
    concept: "workflow",
    label: metadataLabel("/workflow-editor", "Workflow Editor"),
    primaryJob: "Build and validate visual workflows",
    primaryActionLabel: "Add node",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  },
  {
    route: "/skills",
    concept: "skill",
    label: metadataLabel("/skills", "Skills"),
    primaryJob: "Manage server-backed skill definitions",
    primaryActionLabel: "Open skills",
    capabilityMode: "frontend_state",
    diagnosticsPolicy: "disclosed",
    implementationOwner: "shared_route"
  }
]

const normalizeRoutePath = (route: string): string => {
  const [pathname] = route.split("?")
  if (!pathname || pathname === "/") {
    return "/"
  }
  return pathname.endsWith("/") ? pathname.slice(0, -1) : pathname
}

const routeJobByPath = new Map(
  OPERATIONS_ROUTE_JOBS.map((job) => [normalizeRoutePath(job.route), job])
)

export function getOperationsRouteJob(
  route: string
): OperationsRouteJob | undefined {
  return routeJobByPath.get(normalizeRoutePath(route))
}

export function getOperationsRouteJobsByCapabilityMode(
  capabilityMode: OperationsCapabilityMode
): OperationsRouteJob[] {
  return OPERATIONS_ROUTE_JOBS.filter(
    (job) => job.capabilityMode === capabilityMode
  )
}
