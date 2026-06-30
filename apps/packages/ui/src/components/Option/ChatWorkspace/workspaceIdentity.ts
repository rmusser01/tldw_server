export const normalizeWorkspaceId = (
  workspaceId?: string | null
): string | null => {
  const normalized = workspaceId?.trim()
  return normalized ? normalized : null
}
