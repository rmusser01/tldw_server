#!/usr/bin/env node

import fs from "node:fs"

const reportPath = process.argv[2]

if (!reportPath) {
  console.error("[playwright-no-skips] Usage: node scripts/assert-playwright-no-skips.mjs <report.json>")
  process.exit(2)
}

if (!fs.existsSync(reportPath)) {
  console.error(`[playwright-no-skips] Report not found: ${reportPath}`)
  process.exit(2)
}

let report

try {
  report = JSON.parse(fs.readFileSync(reportPath, "utf8"))
} catch (error) {
  const message = error instanceof Error ? error.message : String(error)
  console.error(`[playwright-no-skips] Unable to parse Playwright JSON report: ${message}`)
  process.exit(2)
}

const stats = report?.stats || {}

const expected = Number(stats.expected || 0)
const skipped = Number(stats.skipped || 0)
const unexpected = Number(stats.unexpected || 0)
const flaky = Number(stats.flaky || 0)
const executed = expected + skipped + unexpected + flaky

console.log(
  `[playwright-no-skips] executed=${executed} expected=${expected} skipped=${skipped} unexpected=${unexpected} flaky=${flaky}`
)

if (executed <= 0) {
  console.error("[playwright-no-skips] No tests executed. Expected at least one executed test.")
  process.exit(1)
}

if (skipped > 0) {
  console.error(`[playwright-no-skips] Found ${skipped} skipped test(s). Skips are not allowed.`)
  process.exit(1)
}

if (unexpected > 0) {
  console.error(`[playwright-no-skips] Found ${unexpected} unexpected failure(s).`)
  process.exit(1)
}

if (flaky > 0) {
  console.error(`[playwright-no-skips] Found ${flaky} flaky test(s).`)
  process.exit(1)
}
