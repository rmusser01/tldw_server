// Shared capability probe: answers "does this server expose path X" from the
// server's openapi.json. The spec (historically ~1.4 MB - see
// ./openapi-guard.ts) is fetched at most once per server URL per session;
// in-flight requests are deduplicated by storing the promise, and failures
// are held against a short TTL so an unreachable server is not hammered.

const specCache = new Map<string, Promise<Set<string>>>()   // serverUrl -> paths
const failures = new Map<string, number>()                  // serverUrl -> monotonic-ish ms
const FAILURE_RETRY_MS = 60_000

// Tri-state answer: true = spec advertises the path; false = spec was fetched
// and definitively lacks the path; null = the probe itself failed (network
// error / non-ok response), so support is UNKNOWN - callers must not present
// null as "not available on this server".
export async function serverSupportsPath(serverUrl: string, path: string): Promise<boolean | null> {
  const cached = specCache.get(serverUrl)
  if (cached) {
    try { return (await cached).has(path) } catch { return null }
  }
  if (Date.now() - (failures.get(serverUrl) ?? 0) < FAILURE_RETRY_MS) return null
  const p = (async () => {
    const res = await fetch(`${serverUrl}/openapi.json`)
    if (!res.ok) throw new Error(`openapi probe ${res.status}`)
    const spec = await res.json()
    return new Set(Object.keys(spec?.paths ?? {}))
  })()
  specCache.set(serverUrl, p)
  try { return (await p).has(path) } catch { specCache.delete(serverUrl); failures.set(serverUrl, Date.now()); return null }
}
export function clearCapabilityProbeCacheForTests(): void { specCache.clear(); failures.clear() }
