const RULES = new Set(["literal-color", "palette-color", "custom-radius", "layout-value"])
const CATEGORIES = new Set(["theme-definition", "data-visualization", "user-controlled-styling", "media-editor-rendering", "layout-geometry"])
const compareText = (a, b) => a < b ? -1 : a > b ? 1 : 0
const valueKey = value => String(value).trim().replace(/\s+/g, " ")
const exceptionKey = entry => JSON.stringify([entry.path, entry.rule, valueKey(entry.value), entry.owner])
const compareLocations = (a, b) => compareText(a.path, b.path) || (a.line ?? a.start ?? 0) - (b.line ?? b.start ?? 0) || (a.column ?? 0) - (b.column ?? 0) || compareText(a.rule ?? a.code ?? "", b.rule ?? b.code ?? "") || compareText(JSON.stringify(a), JSON.stringify(b))
const sorted = values => [...values].sort(compareLocations)
const isObject = value => value !== null && typeof value === "object" && !Array.isArray(value)
const isText = value => typeof value === "string" && value.trim().length > 0
const isRepoPath = value => isText(value) && !value.startsWith("/") && !value.includes("\\") && !value.split("/").some(part => part === ".." || part === "." || part === "") && !/^[A-Za-z]:/.test(value)

function readExceptions(manifest, diagnostics) {
  const invalid = message => diagnostics.push({ path: "apps/packages/ui/scripts/design-system-drift-exceptions.json", code: "invalid-exceptions", message, severity: "error" })
  if (!isObject(manifest) || manifest.schemaVersion !== 1 || !Array.isArray(manifest.exceptions)) {
    invalid("Expected exception manifest schemaVersion 1 and an exceptions array")
    return []
  }
  const entries = []
  for (const [index, entry] of manifest.exceptions.entries()) {
    if (!isObject(entry) || !isRepoPath(entry.path) || !RULES.has(entry.rule) || !isText(entry.value) || !(entry.owner === null || isText(entry.owner)) || !CATEGORIES.has(entry.category) || !isText(entry.reason)) {
      invalid(`Invalid exception record at index ${index}`)
      continue
    }
    entries.push(entry)
  }
  return entries
}

/** Assemble an advisory report without hiding existing debt behind a baseline. */
export function assembleDriftReport({ analyses, exceptions, coverage, provenance, diagnostics = [] }) {
  const allDiagnostics = [...diagnostics, ...analyses.flatMap(analysis => analysis.diagnostics)]
  const entries = readExceptions(exceptions, allDiagnostics)
  const grouped = new Map()
  for (const entry of entries) {
    const key = exceptionKey(entry)
    grouped.set(key, [...(grouped.get(key) ?? []), entry])
  }
  const allFindings = sorted(analyses.flatMap(analysis => analysis.findings))
  const matched = new Map()
  const findings = []
  for (const finding of allFindings) {
    const key = exceptionKey(finding)
    if (grouped.get(key)?.length === 1) {
      matched.set(key, [...(matched.get(key) ?? []), finding])
    } else {
      findings.push(finding)
    }
  }
  const acceptedExceptions = []
  const staleExceptions = []
  for (const [key, group] of grouped) {
    if (group.length > 1) {
      allDiagnostics.push({ path: group[0].path, code: "exception-ambiguous", severity: "coverage", message: `Multiple exception records match the same path, rule, value and owner (${group.length} records)` })
    } else if (matched.has(key)) {
      acceptedExceptions.push({ exception: group[0], findings: matched.get(key), matchedCount: matched.get(key).length })
    } else {
      staleExceptions.push(group[0])
    }
  }
  if (provenance.sourceCommit === null || provenance.dirtySource === null) {
    allDiagnostics.push({ path: "", code: "missing-provenance", severity: "error", message: "Source provenance is unavailable" })
  }
  return {
    schemaVersion: 1,
    status: allDiagnostics.some(diagnostic => diagnostic.severity === "error") ? "incomplete" : "complete",
    ...provenance,
    coverage: {
      roots: [...coverage.roots].sort((a, b) => compareText(a.path, b.path)),
      excluded: [...coverage.excluded].sort((a, b) => compareText(a.reason, b.reason)),
      limitations: [...coverage.limitations].sort(compareText)
    },
    findings,
    definitions: sorted(analyses.flatMap(analysis => analysis.definitions)),
    acceptedExceptions: acceptedExceptions.sort((a, b) => compareText(exceptionKey(a.exception), exceptionKey(b.exception))),
    staleExceptions: staleExceptions.sort((a, b) => compareText(exceptionKey(a), exceptionKey(b))),
    diagnostics: sorted(allDiagnostics)
  }
}

const safeText = value => String(value).replace(/[\r\n]/g, " ").replace(/`/g, "'").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/\|/g, "\\|")

/** Keep the displayed inventory bounded while retaining every finding in JSON. */
export function formatDriftSummary(report, { limit = 25, reportPath }) {
  const layoutCount = report.findings.filter(finding => finding.classification === "layout-review").length
  const sourceState = report.dirtySource === null ? "unknown source state" : report.dirtySource ? "modified working files (not reproducible from SHA alone)" : "clean source"
  const lines = [
    `## UI Design-System Drift Report: ${report.status === "complete" ? "COMPLETE" : "INCOMPLETE"}`,
    "",
    "Advisory only. Findings do not block builds or merges.",
    `Source: ${safeText(report.sourceCommit ?? "unavailable")} (${sourceState})`,
    `Untracked source files: ${report.untrackedSourceFileCount}`,
    `Token/radius review findings: ${report.findings.length - layoutCount}`,
    `Layout review candidates: ${layoutCount}`,
    `Accepted exceptions: ${report.acceptedExceptions.length}; stale exceptions: ${report.staleExceptions.length}; diagnostics: ${report.diagnostics.length}`,
    `Full JSON report: ${safeText(reportPath)}`,
    "",
    ...report.coverage.roots.map(root => `${safeText(root.path)}: ${root.analyzed}/${root.discovered} files analyzed`),
    ...report.coverage.excluded.map(entry => `Excluded (${safeText(entry.reason)}): ${entry.count}`),
    ""
  ]
  const shown = report.findings.slice(0, Math.max(0, limit))
  for (const finding of shown) lines.push(`- ${safeText(finding.path)}:${finding.line}:${finding.column} [${finding.rule}] ${safeText(finding.value)}${finding.suggestion ? `; suggested token: ${safeText(finding.suggestion)}` : ""}`)
  const omitted = report.findings.length - shown.length
  if (omitted) lines.push(`${omitted} finding${omitted === 1 ? "" : "s"} omitted from this summary; see the full JSON report.`)
  for (const entry of report.acceptedExceptions.slice(0, limit)) lines.push(`- Accepted: ${safeText(entry.exception.path)} [${entry.exception.rule}], ${entry.matchedCount} matches: ${safeText(entry.exception.reason)}`)
  for (const entry of report.staleExceptions.slice(0, limit)) lines.push(`- Stale exception: ${safeText(entry.path)} [${entry.rule}]: ${safeText(entry.reason)}`)
  for (const diagnostic of report.diagnostics.slice(0, limit)) lines.push(`- ${diagnostic.severity}: ${safeText(diagnostic.path)} [${safeText(diagnostic.code)}] ${safeText(diagnostic.message)}`)
  if (report.diagnostics.length > limit || report.acceptedExceptions.length > limit || report.staleExceptions.length > limit) lines.push("Additional exception/diagnostic entries are in the full JSON report.")
  lines.push("", "Coverage limitations:", ...report.coverage.limitations.map(value => `- ${safeText(value)}`))
  return lines.join("\n") + "\n"
}
