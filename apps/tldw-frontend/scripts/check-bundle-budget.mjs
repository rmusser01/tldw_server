#!/usr/bin/env node
/**
 * Fail the build when the shared app-shell bundle grows past its budget.
 *
 * Every page downloads the shared `_app` chunk before it can render anything,
 * so growth there is paid by every route, including /login. It reached 680 KB
 * gzip -- against 1-15 KB of route-specific code -- because two things drifted
 * into it unnoticed: all 13 English locale files, and the API client with every
 * domain module it re-exports.
 *
 * Nothing caught that, and Turbopack no longer prints a size table at the end of
 * `next build`, so there is no longer any incidental signal either. This reads
 * the emitted manifest directly.
 *
 * Raising SHARED_BUDGET_BYTES is a deliberate act: it means every page in the
 * app got heavier. Record why in the commit message.
 *
 * Usage: node scripts/check-bundle-budget.mjs [--dir .next] [--json]
 */

import { gzipSync } from "node:zlib"
import fs from "node:fs"
import path from "node:path"

// Measured 558.4 KB after moving the API client out of the shell. The ceiling
// leaves modest headroom for ordinary growth while still catching another
// 100 KB-scale regression.
//
// The English locale bundles (~139 KB gzip) are still inside this number. They
// were briefly split out, but the app awaited all of them before rendering, so
// the bytes still transferred on first load and only the measurement moved.
// Loading them per route would take this budget down again.
const SHARED_BUDGET_BYTES = 600 * 1024

// A route should be mostly shell plus its own code. A route far above the shell
// is carrying something that belongs in an async chunk.
const ROUTE_BUDGET_BYTES = 900 * 1024

const args = process.argv.slice(2)
const dirFlag = args.indexOf("--dir")
const distDir = dirFlag === -1 ? ".next" : args[dirFlag + 1]
const asJson = args.includes("--json")

const manifestPath = path.join(distDir, "build-manifest.json")
if (!fs.existsSync(manifestPath)) {
  console.error(
    `[bundle-budget] no build manifest at ${manifestPath}. ` +
      `Run the production build first.`
  )
  process.exit(2)
}

const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"))
const gzipCache = new Map()

const gzipSize = (file) => {
  if (!gzipCache.has(file)) {
    const full = path.join(distDir, file)
    try {
      gzipCache.set(file, gzipSync(fs.readFileSync(full), { level: 6 }).length)
    } catch {
      gzipCache.set(file, 0)
    }
  }
  return gzipCache.get(file)
}

const pages = manifest.pages ?? {}
const shared = new Set([
  ...(pages["/_app"] ?? []),
  ...(manifest.rootMainFiles ?? []),
])

const sharedBytes = [...shared].reduce((sum, f) => sum + gzipSize(f), 0)

const routes = Object.entries(pages)
  .filter(([route]) => !["/_app", "/_error", "/_document"].includes(route))
  .map(([route, files]) => {
    const all = new Set([...files, ...shared])
    return { route, bytes: [...all].reduce((sum, f) => sum + gzipSize(f), 0) }
  })
  .sort((a, b) => b.bytes - a.bytes)

const kb = (bytes) => (bytes / 1024).toFixed(1)

if (asJson) {
  console.log(
    JSON.stringify(
      {
        sharedBytes,
        sharedBudget: SHARED_BUDGET_BYTES,
        heaviestRoute: routes[0],
        routeBudget: ROUTE_BUDGET_BYTES,
      },
      null,
      2
    )
  )
}

const failures = []

console.log(
  `[bundle-budget] shared _app: ${kb(sharedBytes)} KB gzip ` +
    `(budget ${kb(SHARED_BUDGET_BYTES)} KB, ${shared.size} files)`
)
if (sharedBytes > SHARED_BUDGET_BYTES) {
  failures.push(
    `shared app-shell bundle is ${kb(sharedBytes)} KB gzip, over the ` +
      `${kb(SHARED_BUDGET_BYTES)} KB budget by ` +
      `${kb(sharedBytes - SHARED_BUDGET_BYTES)} KB. Every page pays this. ` +
      `Check whether something now reaches _app statically that should be ` +
      `behind a dynamic import.`
  )
}

const over = routes.filter((r) => r.bytes > ROUTE_BUDGET_BYTES)
for (const { route, bytes } of over) {
  failures.push(
    `route ${route} first-load JS is ${kb(bytes)} KB gzip, over the ` +
      `${kb(ROUTE_BUDGET_BYTES)} KB budget`
  )
}

if (routes.length) {
  console.log(
    `[bundle-budget] heaviest route: ${routes[0].route} ` +
      `${kb(routes[0].bytes)} KB gzip`
  )
}

if (failures.length) {
  console.error("\n[bundle-budget] FAILED:")
  for (const failure of failures) console.error(`  - ${failure}`)
  process.exit(1)
}

console.log("[bundle-budget] ok")
