type PersistOutcome = {
  saved: true
  assistantMessageId?: string
  version?: number
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

export const resolveSavedDegradedCharacterPersist = (
  error: unknown
): PersistOutcome | null => {
  if (!isRecord(error) || error.status !== 503) {
    return null
  }

  const details = isRecord(error.details) ? error.details : null
  const detail = details && isRecord(details.detail) ? details.detail : null
  if (
    !detail ||
    detail.code !== "persist_validation_degraded" ||
    detail.saved !== true
  ) {
    return null
  }

  return {
    saved: true,
    assistantMessageId:
      typeof detail.assistant_message_id === "string"
        ? detail.assistant_message_id
        : undefined,
    version: typeof detail.version === "number" ? detail.version : undefined
  }
}
