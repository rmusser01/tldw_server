const getStatusCode = (error: unknown): number | null => {
  if (!error || typeof error !== 'object' || !('status' in error)) return null
  const status = (error as { status?: unknown }).status
  if (typeof status === 'number' && Number.isFinite(status)) return status
  if (typeof status === 'string' && /^\d{3}$/.test(status.trim())) {
    return Number.parseInt(status, 10)
  }
  return null
}

export const shouldReportMediaDetailFetchError = (error: unknown): boolean => {
  const statusCode = getStatusCode(error)
  return statusCode !== 404 && statusCode !== 410
}
