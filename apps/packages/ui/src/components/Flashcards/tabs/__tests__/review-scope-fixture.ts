// Presentation tests use a verified inert lease; real owner/transport behavior has its own integration suite.
export const loadServicePromptSnapshot = async (_ids: unknown, { signal }: { signal: AbortSignal }) => ({
  requestScope: { config: { serverUrl: "https://study.test", authMode: "multi-user" }, userId: 7 },
  scopeSignal: signal,
  scopeInvalidatedSignal: new AbortController().signal,
  release: () => {}
})
