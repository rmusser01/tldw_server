import type { BrowserContext, Request, Response } from "@playwright/test"

export type SkillsRelayEntry = {
  method: string
  path: string
  worker_owned: boolean
  outcome: "pending" | "success" | "redirect" | "http_error" | "failed"
  status?: number
}

type InternalEntry = {
  entry: SkillsRelayEntry
  dryRun: boolean | undefined
}

const SKILLS_ROOT = "/api/v1/skills"
const UNEXPECTED_SKILLS_PATH = `${SKILLS_ROOT}/:unexpected`

const canonicalizeSkillsPath = (url: string): string | null => {
  let pathname: string
  try {
    pathname = new URL(url).pathname
  } catch {
    return null
  }

  if (pathname === SKILLS_ROOT || pathname === `${SKILLS_ROOT}/`)
    return SKILLS_ROOT
  if (!pathname.startsWith(`${SKILLS_ROOT}/`)) return null

  const remainder = pathname.slice(SKILLS_ROOT.length)
  if (remainder.includes("//")) return UNEXPECTED_SKILLS_PATH

  const segments = remainder.slice(1).split("/")
  if (segments.length === 1 && segments[0] === "trash")
    return `${SKILLS_ROOT}/trash`
  if (segments.length === 1) return `${SKILLS_ROOT}/:name`
  if (
    segments.length === 2 &&
    segments[0] !== "trash" &&
    ["execute", "restore", "purge"].includes(segments[1])
  ) {
    return `${SKILLS_ROOT}/:name/${segments[1]}`
  }
  return UNEXPECTED_SKILLS_PATH
}

const isDryRun = (request: Request, path: string): boolean | undefined => {
  if (request.method() !== "POST" || path !== `${SKILLS_ROOT}/:name/execute`)
    return undefined
  try {
    return JSON.parse(request.postData() || "").dry_run === true
  } catch {
    return false
  }
}

export function createSkillsRelayObserver(
  context: Pick<BrowserContext, "on" | "off">,
  expectedWorkerUrl: string
): {
  entries: SkillsRelayEntry[]
  assertValid(): void
  dispose(): void
} {
  const entries: SkillsRelayEntry[] = []
  const tracked = new WeakMap<Request, InternalEntry>()
  const successfulExecuteDryRuns: boolean[] = []

  const onRequest = (request: Request) => {
    const path = canonicalizeSkillsPath(request.url())
    if (!path) return
    const entry: SkillsRelayEntry = {
      method: request.method(),
      path,
      worker_owned: request.serviceWorker()?.url() === expectedWorkerUrl,
      outcome: "pending"
    }
    entries.push(entry)
    tracked.set(request, { entry, dryRun: isDryRun(request, path) })
  }

  const onResponse = (response: Response) => {
    const internal = tracked.get(response.request())
    if (!internal) return
    const status = response.status()
    internal.entry.status = status
    internal.entry.outcome =
      status >= 200 && status < 300
        ? "success"
        : status >= 300 && status < 400
          ? "redirect"
          : "http_error"
    if (
      internal.entry.outcome === "success" &&
      internal.entry.method === "POST" &&
      internal.entry.path === `${SKILLS_ROOT}/:name/execute`
    ) {
      successfulExecuteDryRuns.push(internal.dryRun === true)
    }
  }

  const onRequestFailed = (request: Request) => {
    const internal = tracked.get(request)
    if (internal) internal.entry.outcome = "failed"
  }

  context.on("request", onRequest)
  context.on("response", onResponse)
  context.on("requestfailed", onRequestFailed)

  let disposed = false
  const assertValid = () => {
    const pending = entries.find((entry) => entry.outcome === "pending")
    if (pending) {
      throw new Error(
        `Skills request is still pending: ${pending.method} ${pending.path}`
      )
    }

    const invalid = entries.find((entry) => !entry.worker_owned)
    if (invalid)
      throw new Error(
        `Skills request was page-owned: ${invalid.method} ${invalid.path}`
      )

    const failed = entries.find(
      (entry) => entry.outcome === "failed" || entry.outcome === "http_error"
    )
    if (failed)
      throw new Error(
        `Skills request failed or returned HTTP error: ${failed.method} ${failed.path}`
      )

    const unexpected = entries.find(
      (entry) => entry.path === UNEXPECTED_SKILLS_PATH
    )
    if (unexpected) {
      throw new Error(
        `Unexpected Skills route: ${unexpected.method} ${unexpected.path}`
      )
    }

    const terminalMutations = entries.filter(
      (entry) => entry.outcome === "success" && entry.method !== "GET"
    )
    const expected = [
      ["POST", SKILLS_ROOT, 201, 1],
      ["POST", `${SKILLS_ROOT}/:name/execute`, 200, 1],
      ["DELETE", `${SKILLS_ROOT}/:name`, 204, 2],
      ["POST", `${SKILLS_ROOT}/:name/restore`, 200, 1],
      ["DELETE", `${SKILLS_ROOT}/:name/purge`, 204, 1]
    ] as const

    for (const [method, path, status, count] of expected) {
      const actual = terminalMutations.filter(
        (entry) =>
          entry.method === method &&
          entry.path === path &&
          entry.status === status
      ).length
      if (actual !== count)
        throw new Error(
          `Expected ${method} ${path} status ${status} x${count}, got x${actual}`
        )
    }

    const expectedCount = expected.reduce(
      (total, [, , , count]) => total + count,
      0
    )
    if (terminalMutations.length !== expectedCount) {
      throw new Error("Found extra successful mutation in Skills relay ledger")
    }

    if (
      successfulExecuteDryRuns.length !== 1 ||
      successfulExecuteDryRuns[0] !== true
    ) {
      throw new Error("Skills execute request must use dry_run true")
    }
  }

  return {
    entries,
    assertValid,
    dispose: () => {
      if (disposed) return
      disposed = true
      context.off("request", onRequest)
      context.off("response", onResponse)
      context.off("requestfailed", onRequestFailed)
    }
  }
}
