import fs from "node:fs"
import path from "node:path"
import { describe, expect, it } from "vitest"

// AX-19 (#3124): a behaviour passed from script ("smooth") wins over the CSS
// reduced-motion block, so scripted scrolls must ask smoothScrollBehavior().
// Every scanned file is converted except the four below, which open PRs #3435
// (KnowledgeQAProvider, ResearchWorkspace ChatPane) and #3071 (all four)
// rewrite; convert them once both have landed, to avoid conflicts. Remove a
// file from this list when you convert it; never add one.
const REMAINING = new Set([
  "components/Option/KnowledgeQA/KnowledgeQAProvider.tsx",
  "components/Option/KnowledgeQA/SourceList.tsx",
  "components/Option/ResearchWorkspace/ChatPane/index.tsx",
  "components/Option/ResearchWorkspace/SourcesPane/index.tsx"
])

const LITERAL_SMOOTH = /behavior:\s*["']smooth["']/

const srcRoot = path.resolve(__dirname, "../..")
// The web app's extension routes (the quick chat popout) use the same shared
// UI but live outside packages/ui; they are reported relative to apps/.
const appsRoot = path.resolve(srcRoot, "../../..")
const extraRoots = [path.join(appsRoot, "tldw-frontend/extension/routes")]

const relativeName = (file: string) => {
  const root = file.startsWith(srcRoot + path.sep) ? srcRoot : appsRoot
  return path.relative(root, file).split(path.sep).join("/")
}

function collect(dir: string, out: string[]) {
  for (const entry of fs.readdirSync(dir, { withFileTypes: true })) {
    if (entry.name === "__tests__" || entry.name === "node_modules") continue
    const full = path.join(dir, entry.name)
    if (entry.isDirectory()) collect(full, out)
    else if (/\.(tsx?|jsx?)$/.test(entry.name) && !/\.(test|spec)\./.test(entry.name)) out.push(full)
  }
}

describe("scripted smooth scrolling", () => {
  it("asks smoothScrollBehavior() instead of passing a literal smooth behaviour", () => {
    const files: string[] = []
    collect(srcRoot, files)
    for (const root of extraRoots) collect(root, files)
    expect(files.some((file) => file.startsWith(extraRoots[0]))).toBe(true)
    const offenders = files
      .filter((file) => !REMAINING.has(relativeName(file)))
      .filter((file) => LITERAL_SMOOTH.test(fs.readFileSync(file, "utf8")))
      .map(relativeName)
    expect(offenders).toEqual([])
  })

  it("keeps the remaining list current", () => {
    const stale = [...REMAINING].filter((relative) => {
      const file = path.join(srcRoot, relative)
      return !fs.existsSync(file) || !LITERAL_SMOOTH.test(fs.readFileSync(file, "utf8"))
    })
    expect(stale).toEqual([])
  })
})
