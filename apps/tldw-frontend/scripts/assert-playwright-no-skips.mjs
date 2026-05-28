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

const report = JSON.parse(fs.readFileSync(reportPath, "utf8"))
const stats = report?.stats || {}

const passed = Number(stats.expected || 0)
const skipped = Number(stats.skipped || 0)
const unexpected = Number(stats.unexpected || 0)
const flaky = Number(stats.flaky || 0)

console.log(
  `[playwright-no-skips] passed=${passed} skipped=${skipped} unexpected=${unexpected} flaky=${flaky}`
)

if (passed <= 0) {
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
