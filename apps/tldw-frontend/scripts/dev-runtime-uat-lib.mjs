export const MAX_RSS_BYTES = 16 * 2 ** 30
export const MAX_IDLE_GROWTH_BYTES = 2 * 2 ** 30

/**
 * @typedef {{
 *   pid: number,
 *   parentPid: number,
 *   rssKibibytes: number,
 *   cpuPercent: number,
 *   command: string,
 * }} ProcessRow
 */

/**
 * @typedef {{
 *   phase: string,
 *   rssBytes: number,
 *   responsive: boolean,
 * }} RuntimeSample
 */

/**
 * Parse output from `ps -axo pid=,ppid=,rss=,%cpu=,command=`.
 *
 * @param {string} text
 * @returns {ProcessRow[]}
 */
export function parseProcessTable(text) {
  if (!text.trim()) return []

  return text.split(/\r?\n/).flatMap((line, index) => {
    if (!line.trim()) return []
    const match = line.match(/^\s*(\d+)\s+(\d+)\s+(\d+)\s+([\d.]+)\s+(.+?)\s*$/)
    if (!match) {
      throw new Error(`Malformed process table row ${index + 1}: ${line}`)
    }

    return [{
      pid: Number(match[1]),
      parentPid: Number(match[2]),
      rssKibibytes: Number(match[3]),
      cpuPercent: Number(match[4]),
      command: match[5],
    }]
  })
}

/**
 * Sum resource usage for a root process and all of its descendants.
 *
 * @param {ProcessRow[]} rows
 * @param {number} rootPid
 * @returns {{ rssBytes: number, cpuPercent: number, pids: number[] }}
 */
export function descendantUsage(rows, rootPid) {
  const byPid = new Map(rows.map((row) => [row.pid, row]))
  if (!byPid.has(rootPid)) {
    throw new Error(`Root process ${rootPid} not found`)
  }

  const childrenByParent = new Map()
  for (const row of rows) {
    const children = childrenByParent.get(row.parentPid) ?? []
    children.push(row.pid)
    childrenByParent.set(row.parentPid, children)
  }

  const pending = [rootPid]
  const descendants = new Set()
  while (pending.length > 0) {
    const pid = pending.pop()
    if (descendants.has(pid)) continue
    descendants.add(pid)
    pending.push(...(childrenByParent.get(pid) ?? []))
  }

  const pids = [...descendants].sort((left, right) => left - right)
  const usageRows = pids.map((pid) => byPid.get(pid))
  const rssBytes = usageRows.reduce(
    (total, row) => total + row.rssKibibytes * 1024,
    0,
  )
  const cpuPercent = Number(usageRows.reduce(
    (total, row) => total + row.cpuPercent,
    0,
  ).toFixed(3))

  return { rssBytes, cpuPercent, pids }
}

/**
 * Evaluate the measured runtime against the UAT-host qualification guardrails.
 *
 * @param {RuntimeSample[]} samples
 * @returns {{ qualified: boolean, reasons: string[] }}
 */
export function evaluateRuntime(samples) {
  const postTraversal = samples.find((sample) => sample.phase === "post-traversal")
  const postIdle = samples.find((sample) => sample.phase === "post-idle")
  const secondPass = samples.find((sample) => sample.phase === "second-pass")
  const reasons = []

  if (!samples.every((sample) => sample.responsive)) reasons.push("unresponsive")
  if (!postTraversal) reasons.push("post_traversal_missing")
  if (!postIdle) reasons.push("post_idle_missing")
  if (!secondPass) reasons.push("second_pass_missing")
  if (samples.some((sample) => sample.rssBytes >= MAX_RSS_BYTES)) {
    reasons.push("rss_limit")
  }
  if (
    postTraversal &&
    postIdle &&
    postIdle.rssBytes - postTraversal.rssBytes > MAX_IDLE_GROWTH_BYTES
  ) {
    reasons.push("idle_rss_growth")
  }

  return { qualified: reasons.length === 0, reasons }
}
