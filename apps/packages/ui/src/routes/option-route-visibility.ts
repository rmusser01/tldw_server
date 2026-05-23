import { normalizeRoutePath, ROUTE_METADATA } from "./route-metadata"

export const HOSTED_VISIBLE_OPTION_PATHS = new Set(
  ROUTE_METADATA.filter(
    (metadata) => metadata.hostedOptionVisibility === "visible"
  ).map((metadata) => normalizeRoutePath(metadata.path))
)

export const isHostedVisibleOptionPath = (path: string) =>
  HOSTED_VISIBLE_OPTION_PATHS.has(normalizeRoutePath(path))
