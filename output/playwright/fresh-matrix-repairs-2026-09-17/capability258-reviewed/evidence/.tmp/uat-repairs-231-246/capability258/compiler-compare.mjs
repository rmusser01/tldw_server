import { execFileSync } from "node:child_process"
import fs from "node:fs"
import path from "node:path"
import { createRequire } from "node:module"

const root = process.cwd()
const outputDirectory = path.join(
  root,
  ".tmp/uat-repairs-231-246/capability258"
)
const files = [
  "apps/packages/ui/src/services/tldw/TldwApiClient.ts",
  "apps/packages/ui/src/services/tldw/server-capabilities.ts",
  "apps/packages/ui/src/services/__tests__/server-capabilities.test.ts"
]
const require = createRequire(
  path.join(root, "apps/tldw-frontend/package.json")
)
const ts = require("typescript")
const configPath = path.join(root, "apps/tldw-frontend/tsconfig.json")
const config = ts.readConfigFile(configPath, ts.sys.readFile)
const parsed = ts.parseJsonConfigFileContent(
  config.config,
  ts.sys,
  path.dirname(configPath)
)
const baseline = new Map(
  files.map((file) => [
    path.join(root, file),
    execFileSync("git", ["show", `HEAD:${file}`], { encoding: "utf8" })
  ])
)

const getDiagnostics = (useBaseline) => {
  const host = ts.createCompilerHost({ ...parsed.options, incremental: false })
  const readFile = host.readFile
  if (useBaseline) {
    host.readFile = (file) => baseline.get(path.resolve(file)) ?? readFile(file)
  }
  return ts
    .getPreEmitDiagnostics(
      ts.createProgram(parsed.fileNames, { ...parsed.options, incremental: false }, host)
    )
    .map((diagnostic) => ({
      file: diagnostic.file ? path.relative(root, diagnostic.file.fileName) : "",
      code: diagnostic.code,
      message: ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")
    }))
}

const baselineDiagnostics = getDiagnostics(true)
const currentDiagnostics = getDiagnostics(false)
const key = (diagnostic) => JSON.stringify(diagnostic)
const baselineSet = new Set(baselineDiagnostics.map(key))
const currentSet = new Set(currentDiagnostics.map(key))
const report = {
  baselineCount: baselineDiagnostics.length,
  currentCount: currentDiagnostics.length,
  added: currentDiagnostics.filter((diagnostic) => !baselineSet.has(key(diagnostic))),
  removed: baselineDiagnostics.filter((diagnostic) => !currentSet.has(key(diagnostic)))
}

fs.writeFileSync(
  path.join(outputDirectory, "compiler-comparison.json"),
  JSON.stringify(report, null, 2) + "\n"
)
console.log(
  JSON.stringify({
    baselineCount: report.baselineCount,
    currentCount: report.currentCount,
    added: report.added.length,
    removed: report.removed.length
  })
)
