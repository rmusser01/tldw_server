#!/usr/bin/env node
/**
 * Copy PDF.js worker to public folder
 *
 * This script copies the PDF.js worker file from node_modules to the public
 * directory so it can be served locally instead of from a CDN.
 *
 * This improves:
 * - Security: No external CDN dependencies
 * - Reliability: Works offline / in air-gapped environments
 * - Performance: Same-origin requests are faster
 */

import { copyFileSync, existsSync, mkdirSync } from "fs"
import { dirname, join } from "path"
import { fileURLToPath } from "url"
import { createRequire } from "module"

const __dirname = dirname(fileURLToPath(import.meta.url))
const projectRoot = join(__dirname, "..")
const require = createRequire(import.meta.url)

// Destination: public/pdf.worker.min.mjs
const workerDest = join(projectRoot, "public", "pdf.worker.min.mjs")

function resolveWorkerCandidates() {
  try {
    // The worker must match the PDF.js instance exported by the shared viewer.
    const sharedRequire = createRequire(require.resolve("@tldw/ui/package.json"))
    const viewerRequire = createRequire(sharedRequire.resolve("react-pdf/package.json"))
    const packageRoot = dirname(viewerRequire.resolve("pdfjs-dist/package.json"))
    return [
      join(packageRoot, "legacy", "build", "pdf.worker.min.mjs"),
      join(packageRoot, "build", "pdf.worker.min.mjs")
    ]
  } catch {
    return []
  }
}

function resolveWorkerSource() {
  for (const candidate of resolveWorkerCandidates()) {
    if (existsSync(candidate)) {
      return candidate
    }
  }
  return null
}

function copyPdfWorker() {
  const workerSource = resolveWorkerSource()

  // Ensure source exists
  if (!workerSource) {
    console.error("PDF.js worker not found in any expected location.")
    console.error("Searched paths:")
    for (const candidate of resolveWorkerCandidates()) {
      console.error(`- ${candidate}`)
    }
    console.error("Make sure pdfjs-dist is installed: bun install")
    process.exit(1)
  }

  // Ensure public directory exists
  const publicDir = dirname(workerDest)
  if (!existsSync(publicDir)) {
    mkdirSync(publicDir, { recursive: true })
  }

  // Copy the worker file
  try {
    copyFileSync(workerSource, workerDest)
    console.log(`Copied PDF.js worker to: ${workerDest}`)
  } catch (error) {
    console.error(`Failed to copy PDF.js worker: ${error.message}`)
    process.exit(1)
  }
}

copyPdfWorker()
