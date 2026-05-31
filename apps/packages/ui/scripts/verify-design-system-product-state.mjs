#!/usr/bin/env node
import fs from "node:fs/promises"
import path from "node:path"
import { fileURLToPath } from "node:url"
import { runGuardOnSources } from "./design-system-product-state-rules.mjs"

const scriptPath = fileURLToPath(import.meta.url)
const here = path.dirname(fileURLToPath(import.meta.url))
const packageRoot = path.resolve(here, "..")
const srcRoot = path.resolve(packageRoot, "src")
const baselinePath = path.resolve(
  here,
  "design-system-product-state-baseline.json"
)
const SKIPPED_DIRECTORIES = new Set(["node_modules", "dist", "build"])
const DEFAULT_SOURCE_READ_CONCURRENCY = 16

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

export async function readSourcesInOrder({
  filePaths,
  concurrency = DEFAULT_SOURCE_READ_CONCURRENCY,
  readFile = fs.readFile,
  toRelativePath = toPackageRelativePath
}) {
  const sources = new Array(filePaths.length)
  const workerCount = Math.max(1, Math.min(concurrency, filePaths.length))
  let nextIndex = 0

  await Promise.all(
    Array.from({ length: workerCount }, async () => {
      while (nextIndex < filePaths.length) {
        const index = nextIndex
        nextIndex += 1
        const filePath = filePaths[index]

        sources[index] = {
          relativePath: toRelativePath(filePath),
          source: await readFile(filePath, "utf8")
        }
      }
    })
  )

  return sources
}

async function main() {
  const [filePaths, baselineSource] = await Promise.all([
    walkSourceFiles(srcRoot),
    fs.readFile(baselinePath, "utf8")
  ])
  const baseline = JSON.parse(baselineSource)
  const sources = await readSourcesInOrder({ filePaths })
  const result = await runGuardOnSources({ sources, baseline })

  console.log(result.report)
  process.exitCode = result.exitCode
}

if (process.argv[1] && path.resolve(process.argv[1]) === scriptPath) {
  main().catch((error) => {
    console.error(error)
    process.exitCode = 1
  })
}
