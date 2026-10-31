#!/usr/bin/env node

import { readFileSync } from "node:fs"
import { pathToFileURL } from "node:url"
import { verifyArtifactSeal, writeArtifactSeal } from "./live-tier-uat/artifact-integrity.mjs"
import { assertReleaseCatalog } from "./live-tier-uat/release-catalog.mjs"
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
  const args = process.argv.slice(2)
  const catalogMode = args[0] === "--catalog"
  const artifactMode = ["--seal-artifacts", "--verify-artifacts"].includes(args[0])
  const paths = catalogMode || artifactMode ? args.slice(1) : args
  if (paths.length !== 2) {
    console.error("Usage: node scripts/assert-release-uat.mjs [--catalog] <manifest.json> <report.json>\n       node scripts/assert-release-uat.mjs <--seal-artifacts|--verify-artifacts> <directory> <seal.json>")
    process.exitCode = 2
  } else {
    try {
      if (artifactMode) {
        const seal = args[0] === "--seal-artifacts"
          ? writeArtifactSeal(paths[0], paths[1])
          : verifyArtifactSeal(paths[0], JSON.parse(readFileSync(paths[1], "utf8")))
        console.log(JSON.stringify({
          kind: "artifact-integrity-only", certifiesRelease: false,
          action: args[0] === "--seal-artifacts" ? "sealed" : "verified",
          root: seal.root, sha256: seal.sha256, entries: seal.entries.length,
          limitations: "Quiescent-directory integrity only; no build success, clean installation, loaded runtime, or release certification.",
        }))
      } else {
        const manifest = JSON.parse(readFileSync(paths[0], "utf8"))
        const report = JSON.parse(readFileSync(paths[1], "utf8"))
        if (catalogMode) {
          const result = assertReleaseCatalog(manifest, report)
          console.log(JSON.stringify({
            kind: "catalog-plan", certifiesRelease: false,
            requirements: result.requirements.length, executions: result.executions.length,
            humanReviewsPending: result.humanReviews.filter(entry => entry.humanReview.status !== "completed").length,
            exclusions: result.exclusions.length, scopeExclusions: result.scopeExclusions.length,
          }))
        } else {
          assertReleaseUat(manifest, report)
          console.log(`[release-uat] ${manifest.cases.length} exact required cases passed on their first attempt; runtime artifacts and whole-release coverage are not certified by this receipt check`)
        }
      }
    } catch (error) {
      console.error(`[release-uat] ${error.message}`)
      process.exitCode = 1
    }
  }
}
