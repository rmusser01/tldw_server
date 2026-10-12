#!/usr/bin/env node
import { prepareMonacoAssets } from '../../scripts/prepare-monaco-assets.mjs';

prepareMonacoAssets();

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
  const candidates = []

  // Workspace-local install (bun and some npm layouts).
  candidates.push(join(projectRoot, "node_modules", "pdfjs-dist", "build", "pdf.worker.min.mjs"))
  candidates.push(join(projectRoot, "node_modules", "pdfjs-dist", "legacy", "build", "pdf.worker.min.mjs"))

  // Workspace-hoisted install (`apps/node_modules` when workdir is `apps/`).
  candidates.push(join(projectRoot, "..", "node_modules", "pdfjs-dist", "build", "pdf.worker.min.mjs"))
  candidates.push(join(projectRoot, "..", "node_modules", "pdfjs-dist", "legacy", "build", "pdf.worker.min.mjs"))

  // Resolution-based lookup for layouts npm/bun may create.
  try {
    const pkgJsonPath = require.resolve("pdfjs-dist/package.json", { paths: [projectRoot] })
    const packageRoot = dirname(pkgJsonPath)
    candidates.unshift(join(packageRoot, "build", "pdf.worker.min.mjs"))
    candidates.unshift(join(packageRoot, "legacy", "build", "pdf.worker.min.mjs"))
  } catch {
    // Keep explicit candidates above.
  }

  return [...new Set(candidates)]
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
