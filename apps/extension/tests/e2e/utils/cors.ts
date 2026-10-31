/** CORS for disposable service-prompt protocol fixtures. */
export const fixtureCorsHeaders = (
  origin: string | undefined,
  allowedOrigins: ReadonlySet<string>
): Record<string, string> => {
  if (!origin || !allowedOrigins.has(origin)) return { vary: "Origin" }
  return {
    "access-control-allow-origin": origin,
    "access-control-allow-credentials": "true",
    vary: "Origin"
  }
}
