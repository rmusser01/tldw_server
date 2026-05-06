#!/usr/bin/env node
import fs from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { runGuardOnSources } from "./design-system-product-state-rules.mjs"

const here = path.dirname(fileURLToPath(import.meta.url))
const packageRoot = path.resolve(here, "..")
const srcRoot = path.resolve(packageRoot, "src")
const baselinePath = path.resolve(
  here,
  "design-system-product-state-baseline.json"
)
const SKIPPED_DIRECTORIES = new Set(["node_modules", "dist", "build"])

async function walkSourceFiles(dir) {
  const entries = await fs.readdir(dir, { withFileTypes: true })
  const files = []

  for (const entry of entries.sort((left, right) =>
    left.name.localeCompare(right.name)
  )) {
    const fullPath = path.join(dir, entry.name)

    if (entry.isDirectory()) {
      if (SKIPPED_DIRECTORIES.has(entry.name)) {
        continue
      }

      files.push(...(await walkSourceFiles(fullPath)))
      continue
    }

    if (entry.isFile() && /\.(?:ts|tsx)$/.test(entry.name)) {
      files.push(fullPath)
    }
  }

  return files
}

function toPackageRelativePath(filePath) {
  return path.relative(packageRoot, filePath).replaceAll(path.sep, "/")
}

async function main() {
  const [filePaths, baselineSource] = await Promise.all([
    walkSourceFiles(srcRoot),
    fs.readFile(baselinePath, "utf8")
  ])
  const baseline = JSON.parse(baselineSource)
  const sources = await Promise.all(
    filePaths.map(async (filePath) => ({
      relativePath: toPackageRelativePath(filePath),
      source: await fs.readFile(filePath, "utf8")
    }))
  )
  const result = await runGuardOnSources({ sources, baseline })

  console.log(result.report)
  process.exitCode = result.exitCode
}

main().catch((error) => {
  console.error(error)
  process.exitCode = 1
})
