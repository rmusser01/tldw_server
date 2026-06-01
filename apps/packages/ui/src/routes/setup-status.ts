export const setupRequiredStatuses = new Set([
  "not_started",
  "in_progress",
  "blocked",
  "first_chat_complete"
])

export const isSetupStatusRequiringWizard = (
  status: string | null | undefined
): boolean => !status || setupRequiredStatuses.has(status)
