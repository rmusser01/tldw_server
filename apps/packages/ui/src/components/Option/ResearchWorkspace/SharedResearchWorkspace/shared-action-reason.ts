export const formatSharedActionReason = (
  reasonCode: string | null | undefined
): string | null => reasonCode?.replaceAll("_", " ") ?? null
