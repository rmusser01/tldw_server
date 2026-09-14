function emptyProjectResult() {
  return { passed: 0, failed: 0, skipped: 0, interrupted: 0, elapsedMs: 0 }
}

export function parseListOutput(output) {
  const counts = {}
  for (const line of String(output).split(/\r?\n/)) {
    const match = line.match(/^\s*\[(tier-[123])\]\s+›/)
    if (match) counts[match[1]] = (counts[match[1]] ?? 0) + 1
  }
  return counts
}

function visitSpecs(node, callback) {
  for (const spec of node?.specs ?? []) callback(spec)
  for (const suite of node?.suites ?? []) visitSpecs(suite, callback)
}

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
}) {
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
