import { readdirSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { ESLint } from 'eslint'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const scopes = ['packages/ui/src', 'tldw-frontend/pages']
const hookRules = ['react-hooks/purity', 'react-hooks/static-components', 'react-hooks/use-memo']

function sourceFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name)
    if (entry.isDirectory()) return sourceFiles(entryPath)
    return /\.(?:[cm]?js|jsx|tsx?)$/.test(entry.name) ? [entryPath] : []
  })
}

// General frontend lint remains a separate required step. This gate extends the
// three enabled compiler rules to every source file in both declared UI scopes.
export async function checkSharedHooks({
  appsRoot = path.dirname(frontendRoot),
  configFile = path.join(frontendRoot, 'eslint.config.mjs')
} = {}) {
  const files = scopes.flatMap((scope) => {
    const found = sourceFiles(path.join(appsRoot, scope))
    if (!found.length) throw new Error(`No source files in required scope: ${scope}`)
    return found
  })
  const eslint = new ESLint({ cwd: appsRoot, overrideConfigFile: configFile })
  const failures = []
  for (const filePath of files) {
    const config = await eslint.calculateConfigForFile(filePath)
    if (!config) {
      failures.push({ filePath, message: 'Source file is not covered by ESLint' })
      continue
    }
    for (const ruleId of hookRules) {
      if (config.rules?.[ruleId]?.[0] !== 2) {
        failures.push({ filePath, ruleId, message: 'Hook rule must remain enabled as an error' })
      }
    }
  }
  const results = await eslint.lintFiles(files)
  let otherErrorCount = 0
  for (const result of results) {
    for (const message of result.messages) {
      if (
        message.fatal ||
        (!message.ruleId && message.severity === 2) ||
        message.message.startsWith('Definition for rule ') ||
        hookRules.includes(message.ruleId)
      ) {
        failures.push({ filePath: result.filePath, ...message })
      } else if (message.severity === 2) {
        otherErrorCount += 1
      }
    }
  }
  return { fileCount: files.length, failures, otherErrorCount }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = await checkSharedHooks()
    for (const failure of result.failures) {
      console.error(
        `${failure.filePath}:${failure.line ?? 0}:${failure.column ?? 0} ${failure.ruleId ?? 'ESLint'}: ${failure.message}`
      )
    }
    console.log(
      `Shared hook gate: ${result.fileCount} files in ${scopes.join(' and ')}; ${result.failures.length} failures; ${result.otherErrorCount} other ESLint errors outside this three-rule gate.`
    )
    process.exitCode = result.failures.length ? 1 : 0
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  }
}
