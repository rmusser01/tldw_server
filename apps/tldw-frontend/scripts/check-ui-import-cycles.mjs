import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const scriptName = 'scripts/check-ui-import-cycles.mjs'
const howToBreak =
  'Break it by moving the shared code down a layer, by making the import `import type` when only types are used, or by loading it lazily with import(); the cycles are listed below.'

// Runtime file-level import cycles in packages/ui/src (TASK-13950.10), measured by this
// script on dev 39093fc827. The cycle count is exact: when it drops, lower it here in the
// same PR. The largest cycle and the files in all cycles are ceilings, so a small cycle
// cannot grow unnoticed either. Raising any of them is a reviewed exception: change the
// number and add a "bump: <reason>" comment on that line.
export const BASELINE = { cycles: 4, largestCycle: 4, filesInCycles: 12 }

// Static imports and re-exports that survive compilation. `import type`, `export type` and
// specifier lists that are all `type` are erased; import() is lazy, so it closes no cycle.
function runtimeSpecifiers(sourceFile) {
  const specifiers = []
  for (const statement of sourceFile.statements) {
    let clause
    if (ts.isImportDeclaration(statement)) {
      const importClause = statement.importClause
      if (importClause?.isTypeOnly) continue
      clause = importClause && !importClause.name ? importClause.namedBindings : undefined
    } else if (ts.isExportDeclaration(statement) && statement.moduleSpecifier) {
      if (statement.isTypeOnly) continue
      clause = statement.exportClause
    } else {
      continue
    }
    const elements = clause && (ts.isNamedImports(clause) || ts.isNamedExports(clause)) ? clause.elements : []
    if (elements.length && elements.every((element) => element.isTypeOnly)) continue
    if (ts.isStringLiteral(statement.moduleSpecifier)) specifiers.push(statement.moduleSpecifier.text)
  }
  return specifiers
}

// Strongly connected components of size > 1 (or a file importing itself), by an iterative
// Tarjan walk, since the shared UI graph is deep enough to overflow a recursive one.
function cycles(graph) {
  let next = 0
  const index = new Map()
  const low = new Map()
  const stack = []
  const onStack = new Set()
  const found = []
  const visit = (node) => {
    index.set(node, next)
    low.set(node, next++)
    stack.push(node)
    onStack.add(node)
    return [node, graph.get(node).values()]
  }
  for (const root of graph.keys()) {
    if (index.has(root)) continue
    const work = [visit(root)]
    while (work.length) {
      const [node, edges] = work.at(-1)
      const edge = edges.next()
      if (!edge.done) {
        if (!index.has(edge.value)) work.push(visit(edge.value))
        else if (onStack.has(edge.value)) low.set(node, Math.min(low.get(node), index.get(edge.value)))
        continue
      }
      work.pop()
      if (work.length) {
        const parent = work.at(-1)[0]
        low.set(parent, Math.min(low.get(parent), low.get(node)))
      }
      if (low.get(node) !== index.get(node)) continue
      const members = []
      let member
      do {
        member = stack.pop()
        onStack.delete(member)
        members.push(member)
      } while (member !== node)
      if (members.length > 1 || graph.get(node).has(node)) found.push(members.sort())
    }
  }
  return found
}

// Resolves every runtime import the way tsc does with packages/ui's tsconfig (so @/ and ~/
// aliases count) and keeps the edges between non-declaration source files.
export function findImportCycles(uiRoot = path.join(path.dirname(frontendRoot), 'packages/ui')) {
  const configPath = path.join(uiRoot, 'tsconfig.json')
  const { config, error } = ts.readConfigFile(configPath, ts.sys.readFile)
  if (error) throw new Error(ts.flattenDiagnosticMessageText(error.messageText, '\n'))
  const { options, fileNames } = ts.parseJsonConfigFileContent(config, ts.sys, uiRoot)
  const sourceRoot = path.join(uiRoot, 'src') + path.sep
  const files = fileNames
    .map((file) => path.resolve(file))
    .filter((file) => file.startsWith(sourceRoot) && !file.endsWith('.d.ts'))
  if (!files.length) throw new Error(`No source files under ${sourceRoot}`)
  const graph = new Map(files.map((file) => [file, new Set()]))
  const cache = ts.createModuleResolutionCache(uiRoot, (name) => name, options)
  for (const file of files) {
    const sourceFile = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest)
    for (const specifier of runtimeSpecifiers(sourceFile)) {
      const resolved = ts.resolveModuleName(specifier, file, options, ts.sys, cache).resolvedModule
      const target = resolved && path.resolve(resolved.resolvedFileName)
      if (target && graph.has(target)) graph.get(file).add(target)
    }
  }
  return {
    fileCount: files.length,
    cycles: cycles(graph).map((members) => members.map((file) => path.relative(sourceRoot, file)))
  }
}

export function checkImportCycles(found, baseline = BASELINE) {
  const count = found.length
  const largest = Math.max(0, ...found.map((members) => members.length))
  const files = found.reduce((total, members) => total + members.length, 0)
  const failures = []
  if (count > baseline.cycles) {
    failures.push(
      `${count} import cycles, up from ${baseline.cycles}: an import added in this change closes a new cycle. ${howToBreak}`
    )
  } else if (count < baseline.cycles) {
    failures.push(
      `Stale baseline: ${count} import cycles, down from ${baseline.cycles}. Lower BASELINE.cycles to ${count} in ${scriptName} in this PR.`
    )
  }
  if (largest > baseline.largestCycle) {
    failures.push(
      `The largest import cycle grew from ${baseline.largestCycle} to ${largest} files: an import added in this change pulls more files into it. ${howToBreak}`
    )
  }
  if (files > baseline.filesInCycles) {
    failures.push(
      `${files} files are in import cycles, up from ${baseline.filesInCycles}: an import added in this change pulls more files into a cycle. ${howToBreak}`
    )
  }
  return { count, largest, files, failures }
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const { fileCount, cycles: found } = findImportCycles()
    const { count, largest, files, failures } = checkImportCycles(found)
    for (const failure of failures) console.error(failure)
    if (failures.length) {
      for (const members of found) console.error(`Cycle of ${members.length} files:\n  ${members.join('\n  ')}`)
    }
    console.log(
      `Shared UI import cycles: ${fileCount} files; ${count} runtime cycles (baseline ${BASELINE.cycles}); largest ${largest} files (ceiling ${BASELINE.largestCycle}); ${files} files in cycles (ceiling ${BASELINE.filesInCycles}).`
    )
    process.exitCode = failures.length ? 1 : 0
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  }
}
