/** Canonical UUIDs and existing opaque workspace IDs must remain one URL segment. */
export const isWorkspaceTargetId = (value: unknown): value is string =>
  typeof value === "string" && /^[A-Za-z0-9_-]{1,128}$/.test(value)
