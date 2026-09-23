#!/usr/bin/env node

import fs from "node:fs"
import { pathToFileURL } from "node:url"

export function assertPlaywrightNoSkips(report) {
  const stats = report?.stats || {}
  const counts = Object.fromEntries(
    ["expected", "skipped", "unexpected", "flaky"].map((key) => {
      const value = Number(stats[key] || 0)
      if (!Number.isSafeInteger(value) || value < 0) {
        throw new Error(`Invalid Playwright ${key} count`)
      }
      return [key, value]
    })
  )
  const executed = Object.values(counts).reduce((sum, value) => sum + value, 0)
  if (executed <= 0) throw new Error("No tests executed. Expected at least one executed test.")
  if (counts.skipped) throw new Error(`Found ${counts.skipped} skipped test(s). Skips are not allowed.`)
  if (counts.unexpected) throw new Error(`Found ${counts.unexpected} unexpected failure(s).`)
  if (counts.flaky) throw new Error(`Found ${counts.flaky} flaky test(s).`)
  if (report.errors?.length) throw new Error("Playwright reported global errors.")
  return { ...counts, executed }
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  const reportPath = process.argv[2]
  if (!reportPath || !fs.existsSync(reportPath)) {
    console.error("[playwright-no-skips] Usage: node scripts/assert-playwright-no-skips.mjs <existing-report.json>")
    process.exitCode = 2
  } else {
    let report
    try {
      report = JSON.parse(fs.readFileSync(reportPath, "utf8"))
    } catch (error) {
      console.error(`[playwright-no-skips] Unable to parse Playwright JSON report: ${error.message}`)
      process.exitCode = 2
    }
    if (!process.exitCode) {
      try {
        const counts = assertPlaywrightNoSkips(report)
        console.log(`[playwright-no-skips] ${Object.entries(counts).map(([key, value]) => `${key}=${value}`).join(" ")}`)
      } catch (error) {
        console.error(`[playwright-no-skips] ${error.message}`)
        process.exitCode = 1
      }
    }
  }
}
