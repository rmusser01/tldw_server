import { isDeepStrictEqual } from "node:util"
import { assertPlaywrightNoSkips } from "../assert-playwright-no-skips.mjs"

function emptyProjectResult() {
  return { passed: 0, failed: 0, skipped: 0, interrupted: 0, elapsedMs: 0 }
}

export function parseListOutput(output) {
  const counts = {}
  for (const line of String(output).split(/\r?\n/)) {
    const match = line.match(/^\s*\[([^\]]+)\]\s+›/)
    if (match) counts[match[1]] = (counts[match[1]] ?? 0) + 1
  }
  return counts
}

function visitSpecs(node, callback, parents = []) {
  // The file-level suite is redundant with spec.file; describe titles are not.
  const isFileSuite = node?.line === 0 && node?.title === node?.file
  const titles = node?.title && !isFileSuite ? [...parents, node.title] : parents
  for (const spec of node?.specs ?? []) callback(spec, [...titles, spec.title])
  for (const suite of node?.suites ?? []) visitSpecs(suite, callback, titles)
}

export function collectPlaywrightCases(report) {
  const cases = []
  visitSpecs(report, (spec, titlePath) => {
    for (const test of spec.tests ?? []) {
      cases.push({ project: test.projectName, file: spec.file, titlePath, playwrightId: spec.id })
    }
  })
  return cases
}

const caseKey = ({ project, file, titlePath }) => JSON.stringify([project, file, titlePath])

export function summarizePlaywrightReport(report) {
  const summary = {}
  visitSpecs(report, (spec) => {
    for (const test of spec.tests ?? []) {
      const project = test.projectName ?? "unknown"
      summary[project] ??= emptyProjectResult()
      const result = test.results?.at(-1)
      const status = result?.status ?? test.status
      summary[project].elapsedMs += result?.duration ?? 0
      if (test.status === "skipped" || status === "skipped") {
        summary[project].skipped += 1
      } else if (status === "passed" && test.status !== "unexpected") {
        summary[project].passed += 1
      } else if (status === "interrupted") {
        summary[project].interrupted += 1
      } else {
        summary[project].failed += 1
      }
    }
  })
  return summary
}

export function assertProjectAccounting({
  projects,
  listed,
  results,
  allowSkips = false,
  requiredCases = undefined,
  report = undefined,
  runContext = undefined,
}) {
  if (requiredCases !== undefined) assertRequiredCases({ requiredCases, report, runContext })
  for (const project of projects) {
    const expected = listed[project] ?? 0
    const result = results[project] ?? emptyProjectResult()
    const accounted = result.passed + result.failed + result.skipped + result.interrupted
    if (accounted !== expected) {
      throw new Error(
        `Playwright project ${project} listed ${expected} test(s) but accounted ${accounted}`
      )
    }
    if (result.failed || result.interrupted) {
      throw new Error(
        `Playwright project ${project} reported failed ${result.failed}, interrupted ${result.interrupted}`
      )
    }
    if (!allowSkips && result.skipped) {
      throw new Error(
        `Playwright project ${project} reported skipped ${result.skipped} in strict mode`
      )
    }
  }

  const unexpectedProjects = Object.keys(results).filter(
    (project) => !projects.includes(project)
  )
  if (unexpectedProjects.length) {
    throw new Error(
      `Playwright JSON contained unexpected project(s): ${unexpectedProjects.join(", ")}`
    )
  }
}

export function collectSkippedTests(report) {
  const skipped = []
  visitSpecs(report, (spec) => {
    for (const test of spec.tests ?? []) {
      const result = test.results?.at(-1)
      if (test.status !== "skipped" && result?.status !== "skipped") continue
      const annotation = (test.annotations ?? []).find((entry) => entry.type === "skip")
      skipped.push({
        project: test.projectName ?? "unknown",
        title: spec.title ?? "untitled test",
        reason: annotation?.description ?? "No skip reason recorded",
      })
    }
  })
  return skipped
}

function interceptedCounts(inventory) {
  const byProject = {}
  for (const entry of inventory ?? []) {
    byProject[entry.project] ??= new Set()
    byProject[entry.project].add(entry.test ?? `${entry.file}:${entry.line}`)
  }
  return Object.fromEntries(
    Object.entries(byProject).map(([project, values]) => [project, values.size])
  )
}

function displayPath(value) {
  return value ? `\`${value}\`` : "not produced"
}

export function renderMarkdownReport({
  runId,
  commit = "unknown",
  listed = {},
  results = {},
  inventory = [],
  health = {},
  artifacts = {},
  certification = true,
  skippedTests = [],
  error = null,
}) {
  const intercepted = interceptedCounts(inventory)
  const projects = [...new Set([...Object.keys(listed), ...Object.keys(results)])].sort()
  const lines = [
    "# Tier 1-3 Live-Backend UAT Results",
    "",
    `Run: \`${runId}\``,
    `Commit: \`${commit}\``,
    `Certification run: ${certification ? "yes" : "no"}`,
    "Offline fallback: disabled",
    "Retries: 0",
    "",
    "| Project | Listed | Passed | Failed | Skipped | Intercepted | Live |",
    "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
  ]

  for (const project of projects) {
    const projectResult = results[project] ?? emptyProjectResult()
    const listedCount = listed[project] ?? 0
    const interceptedCount = Math.min(intercepted[project] ?? 0, listedCount)
    lines.push(
      `| ${project} | ${listedCount} | ${projectResult.passed} | ${projectResult.failed} | ${projectResult.skipped} | ${interceptedCount} | ${Math.max(0, listedCount - interceptedCount)} |`
    )
    if (projectResult.interrupted) {
      lines.push(`<!-- ${project}: ${projectResult.interrupted} interrupted -->`)
    }
  }

  lines.push(
    "",
    `Health before tests: ${health.before ? "healthy" : "unhealthy"}`,
    `Health after tests: ${health.after ? "healthy" : "unhealthy"}`,
    `Spawned services stopped: ${health.stopped ? "yes" : "not verified"}`,
    `Artifacts: ${displayPath(artifacts.root)}`,
    `Playwright JSON: ${displayPath(artifacts.playwrightJson)}`,
    "",
    "## API interception inventory",
    ""
  )

  if (!inventory.length) {
    lines.push("No fulfilling or aborting API routes were found in the selected sources.")
  } else {
    lines.push("| Project | Test | Source | Matcher | Evidence |", "| --- | --- | --- | --- | --- |")
    for (const entry of inventory) {
      lines.push(
        `| ${entry.project} | ${entry.test ?? "file-level helper"} | \`${entry.file}:${entry.line}\` | \`${String(entry.matcher).replaceAll("|", "\\|")}\` | UI/contract (intercepted) |`
      )
    }
  }

  lines.push("", "## Skipped tests", "")
  if (!skippedTests.length) {
    lines.push("None.")
  } else {
    lines.push("| Project | Test | Reason |", "| --- | --- | --- |")
    for (const skipped of skippedTests) {
      lines.push(
        `| ${skipped.project} | ${String(skipped.title).replaceAll("|", "\\|")} | ${String(skipped.reason).replaceAll("|", "\\|")} |`
      )
    }
  }

  if (error) lines.push("", "## Runner error", "", String(error))
  return `${lines.join("\n")}\n`
}

// The runner must independently verify these source/artifact hashes against its
// owned runtime before recording runContext in Playwright metadata. This check
// binds receipts; it does not prove a server is running the claimed artifact.
function assertRequiredCases({ requiredCases, report, runContext }) {
  const stats = assertPlaywrightNoSkips(report)
  const sha256 = (value) => typeof value === "string" && /^[a-f0-9]{64}$/.test(value)
  if (!runContext || !["sqlite-single", "sqlite-multi", "pg-single", "pg-multi"].includes(runContext.cell) ||
      !["web", "extension-options", "extension-sidepanel", "cross-surface"].includes(runContext.surface) ||
      !["clean-install", "fresh-state", "supported-upgrade"].includes(runContext.phase) ||
      !/^[a-f0-9]{40}$/.test(runContext.revision ?? "") || !sha256(runContext.sourceSha256) ||
      !sha256(runContext.artifacts?.backend) ||
      (runContext.surface !== "web" && !sha256(runContext.artifacts?.extension)) ||
      (["web", "cross-surface"].includes(runContext.surface) && !sha256(runContext.artifacts?.web))) {
    throw new Error("Required-case accounting needs complete candidate/cell/surface/phase provenance")
  }
  if (!isDeepStrictEqual(runContext, report.config?.metadata?.uat)) {
    throw new Error("Playwright candidate provenance differs from the planned run")
  }
  if (!Array.isArray(requiredCases) || !requiredCases.length) {
    throw new Error("Required case manifest must not be empty")
  }
  const planned = new Set()
  const caseIds = new Set()
  for (const entry of requiredCases) {
    if (![entry.id, entry.project, entry.file, entry.workflow, entry.variant].every(value => typeof value === "string" && value.trim()) ||
        !Array.isArray(entry.titlePath) || !entry.titlePath.length ||
        !entry.titlePath.every(value => typeof value === "string" && value.trim())) {
      throw new Error("Each required case needs id, project, file, full titlePath, workflow and variant")
    }
    const identity = caseKey(entry)
    if (planned.has(identity) || caseIds.has(entry.id)) throw new Error(`Duplicate required case: ${entry.id}`)
    planned.add(identity)
    caseIds.add(entry.id)
    const project = report.config?.projects?.find(p => p.name === entry.project)
    if (!project || project.retries !== 0 || project.repeatEach !== 1) {
      throw new Error(`Required project ${entry.project} must use retries0 and repeatEach1`)
    }
  }
  const observed = new Set()
  visitSpecs(report, (spec, titlePath) => {
    for (const test of spec.tests ?? []) {
      const identity = caseKey({ project: test.projectName, file: spec.file, titlePath })
      if (!planned.has(identity)) throw new Error(`Unexpected required-case result: ${identity}`)
      if (observed.has(identity)) throw new Error(`Duplicate required-case result: ${identity}`)
      observed.add(identity)
      if (test.status !== "expected" || test.expectedStatus !== "passed" || test.results?.length !== 1 ||
          test.results[0].status !== "passed" || test.results[0].retry !== 0) {
        throw new Error(`Required case did not pass its first attempt: ${identity}`)
      }
    }
  })
  const missing = [...planned].filter(identity => !observed.has(identity))
  if (missing.length) throw new Error(`Missing required cases: ${missing.join(", ")}`)
  if (stats.expected !== observed.size) throw new Error("Playwright statistics disagree with exact case results")
}
