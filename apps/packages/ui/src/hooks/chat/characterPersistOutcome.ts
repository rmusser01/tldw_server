type PersistOutcome = {
  saved: true
  assistantMessageId?: string
  version?: number
}

const isRecord = (value: unknown): value is Record<string, unknown> =>
  typeof value === "object" && value !== null

const resolvePersistDetail = (
  error: Record<string, unknown>
): Record<string, unknown> | null => {
  if (isRecord(error.detail)) {
    return error.detail
  }

  const details = isRecord(error.details) ? error.details : null
  if (details && isRecord(details.detail)) {
    return details.detail
  }
  if (details && typeof details.code !== "undefined") {
    return details
  }
  return null
}

export const resolveSavedDegradedCharacterPersist = (
  error: unknown
): PersistOutcome | null => {
  const status = isRecord(error) ? Number(error.status) : Number.NaN
  if (!isRecord(error) || status !== 503) {
    return null
  }

  const detail = resolvePersistDetail(error)
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
