#!/usr/bin/env node

import { readFileSync } from "node:fs"
import { pathToFileURL } from "node:url"
import { assertProjectAccounting, summarizePlaywrightReport } from "./live-tier-uat/report.mjs"

export function assertReleaseUat(manifest, report) {
  const listed = {}
  for (const entry of manifest.cases ?? []) {
    listed[entry.project] = (listed[entry.project] ?? 0) + 1
  }
  assertProjectAccounting({
    projects: Object.keys(listed),
    listed,
    results: summarizePlaywrightReport(report),
    requiredCases: manifest.cases ?? [],
    report,
    runContext: manifest.context,
  })
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  if (process.argv.length !== 4) {
    console.error("Usage: node scripts/assert-release-uat.mjs <required-manifest.json> <report.json>")
    process.exitCode = 2
  } else {
    try {
      const manifest = JSON.parse(readFileSync(process.argv[2], "utf8"))
      const report = JSON.parse(readFileSync(process.argv[3], "utf8"))
      assertReleaseUat(manifest, report)
      console.log(`[release-uat] ${manifest.cases.length} exact required cases passed on their first attempt`)
    } catch (error) {
      console.error(`[release-uat] ${error.message}`)
      process.exitCode = 1
    }
  }
}
