import fs from "node:fs"
import path from "node:path"
import { createRequire } from "node:module"

const root = process.cwd()
const frontend = path.join(root, "apps/tldw-frontend")
const evidence = path.join(root, ".tmp/uat-repairs-231-246/sources264")
const require = createRequire(path.join(frontend, "package.json"))
const ts = require("typescript")
const config = ts.readConfigFile(path.join(frontend, "tsconfig.json"), ts.sys.readFile)
const parsed = ts.parseJsonConfigFileContent(config.config, ts.sys, frontend)
const diagnostics = ts.getPreEmitDiagnostics(
  ts.createProgram(parsed.fileNames, { ...parsed.options, incremental: false, noEmit: true })
).map(diagnostic => ({
  file: diagnostic.file ? path.relative(root, diagnostic.file.fileName) : null,
  code: diagnostic.code,
  message: ts.flattenDiagnosticMessageText(diagnostic.messageText, "\n")
}))
const baseline = JSON.parse(fs.readFileSync(
  path.join(root, ".tmp/uat-repairs-231-246/character248/tsc-baseline.json"),
  "utf8"
)).map(({ location, ...diagnostic }) => diagnostic)
const key = diagnostic => JSON.stringify(diagnostic)
const baselineKeys = new Set(baseline.map(key))
const currentKeys = new Set(diagnostics.map(key))
const report = {
  baselineCount: baseline.length,
  currentCount: diagnostics.length,
  added: diagnostics.filter(diagnostic => !baselineKeys.has(key(diagnostic))),
  removed: baseline.filter(diagnostic => !currentKeys.has(key(diagnostic)))
}
fs.writeFileSync(path.join(evidence, "tsc-comparison.json"), JSON.stringify(report, null, 2) + "\n")
console.log(JSON.stringify({
  baselineCount: report.baselineCount,
  currentCount: report.currentCount,
  added: report.added.length,
  removed: report.removed.length
}))
process.exitCode = report.added.length || report.removed.length ? 1 : 0
