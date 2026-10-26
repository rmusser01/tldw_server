import { readdirSync, readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { ESLint } from 'eslint'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const scopes = ['packages/ui/src', 'tldw-frontend/pages']
// Vendored, minified assets; eslint.config.mjs ignores the same directory.
const vendored = ['packages/ui/src/public']
const hookRules = ['react-hooks/purity', 'react-hooks/static-components', 'react-hooks/use-memo']
// Written by `bun run lint:ui --suppress-all` (ESLint 9 bulk suppressions), keyed by
// file path relative to apps/ and rule id.
const suppressionsName = 'tldw-frontend/eslint-suppressions.packages-ui.json'
const pruneHint = 'run `bun run lint:ui --prune-suppressions` from apps/tldw-frontend and commit the file'
function sourceFiles(directory, appsRoot) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const entryPath = path.join(directory, entry.name)
    if (entry.isDirectory()) {
      const relative = path.relative(appsRoot, entryPath).split(path.sep).join('/')
      return vendored.includes(relative) ? [] : sourceFiles(entryPath, appsRoot)
    }
    return /\.(?:[cm]?js|jsx|tsx?)$/.test(entry.name) ? [entryPath] : []
  })
}

function loadSuppressions(filePath) {
  try {
    return JSON.parse(readFileSync(filePath, 'utf8'))
  } catch (error) {
    if (error.code === 'ENOENT') return {}
    throw error
  }
}

function fixHint(ruleId) {
  return `fix it, or if it must stay, disable that line with "// eslint-disable-next-line ${ruleId} -- <reason>"`
}

// Every ESLint error in both scopes fails unless the suppressions file freezes it. This
// mirrors ESLint's own applySuppressions: a file's errors for one rule pass while their
// count is at most the frozen count, and a frozen count above the current one is stale.
// Unlike ESLint, the three hook rules and parser errors are never suppressible, and an
// entry for a file that is no longer linted is stale too.
export async function checkSharedHooks({
  appsRoot = path.dirname(frontendRoot),
  configFile = path.join(frontendRoot, 'eslint.config.mjs'),
  suppressionsFile = path.join(appsRoot, suppressionsName)
} = {}) {
  const files = scopes.flatMap((scope) => {
    const found = sourceFiles(path.join(appsRoot, scope), appsRoot)
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
  const suppressions = loadSuppressions(suppressionsFile)
  const linted = new Set()
  let suppressedCount = 0
  for (const result of await eslint.lintFiles(files)) {
    const key = path.relative(appsRoot, result.filePath).split(path.sep).join('/')
    linted.add(key)
    const frozen = suppressions[key] ?? {}
    const errors = {}
    for (const message of result.messages) {
      if (
        message.fatal ||
        (!message.ruleId && message.severity === 2) ||
        message.message.startsWith('Definition for rule ') ||
        hookRules.includes(message.ruleId)
      ) {
        failures.push({ filePath: result.filePath, ...message })
      } else if (message.ruleId && message.severity === 2) {
        ;(errors[message.ruleId] ??= []).push(message)
      }
    }
    for (const [ruleId, messages] of Object.entries(errors)) {
      const limit = frozen[ruleId]?.count ?? 0
      if (messages.length <= limit) {
        suppressedCount += messages.length
        continue
      }
      for (const message of messages) {
        failures.push({ filePath: result.filePath, ...message, hint: fixHint(ruleId) })
      }
    }
    for (const [ruleId, { count }] of Object.entries(frozen)) {
      const found = errors[ruleId]?.length ?? 0
      if (hookRules.includes(ruleId)) {
        failures.push({ filePath: result.filePath, ruleId, message: `${suppressionsName} must not freeze hook rules; fix the hook error instead` })
      } else if (found < count) {
        failures.push({ filePath: result.filePath, ruleId, message: `Stale suppression: ${found} errors left of ${count} frozen; ${pruneHint}` })
      }
    }
  }
  for (const key of Object.keys(suppressions)) {
    if (!linted.has(key)) {
      failures.push({ filePath: path.join(appsRoot, key), message: `Stale suppression: file is gone or no longer linted; ${pruneHint}` })
    }
  }
  return { fileCount: files.length, failures, suppressedCount }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = await checkSharedHooks()
    for (const failure of result.failures) {
      console.error(
        `${failure.filePath}:${failure.line ?? 0}:${failure.column ?? 0} ${failure.ruleId ?? 'ESLint'}: ${failure.message}${failure.hint ? ` (to fix: ${failure.hint})` : ''}`
      )
    }
    console.log(
      `Shared UI lint: ${result.fileCount} files in ${scopes.join(' and ')}; ${result.failures.length} failures; ${result.suppressedCount} pre-existing errors frozen in ${suppressionsName}.`
    )
    process.exitCode = result.failures.length ? 1 : 0
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  }
}
