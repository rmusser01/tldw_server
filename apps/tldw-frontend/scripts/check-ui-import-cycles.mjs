import { readFileSync } from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import ts from 'typescript'

const frontendRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..')
const scriptName = 'scripts/check-ui-import-cycles.mjs'
const howToBreak =
  'Break it by moving the shared code down a layer, by making the import `import type` when only types are used, or by loading it lazily with import(). Accepting it instead is a reviewed exception: add or extend its BASELINE entry with a "// bump: <reason>" comment.'
// Asset imports that bundlers handle and tsc does not resolve.
const assetSpecifier =
  /\.(?:css|scss|sass|less|svg|png|jpe?g|gif|webp|avif|ico|bmp|woff2?|ttf|otf|eot|mp3|wav|ogg|mp4|webm|json|md|txt|html|wasm)(?:\?.*)?$|\?(?:raw|url|inline|worker)$/
// Relative paths and the @/ and ~/ aliases, which must resolve to a file.
const localSpecifier = /^(?:\.\.?|[@~])(?:\/|$)/

// The runtime import cycles in packages/ui/src (TASK-13950.10), keyed by their member files
// as this script found them on dev 39093fc827. A cycle may shrink or split, but one that is
// not inside a single entry fails, so trading one cycle for another fails too. When a cycle
// is gone, delete its entry in the same PR. Adding or extending an entry is a reviewed
// exception: put a "// bump: <reason>" comment on it.
export const BASELINE = [
  ['components/Review/ViewMediaPage.tsx', 'components/Review/hooks/useMediaNavigationState.ts'],
  ['db/dexie/helpers.ts', 'db/dexie/server-chat-mirror.ts'],
  // bump: #3298 wave 3 chat-model-outcome (2c60650739). TldwApiClient -> chat-model-outcome ->
  // utils/chat-model-readiness -> utils/chat-model-id -> utils/resolve-api-provider ->
  // services/tldw (index) -> TldwApiClient; every edge is a value import, so `import type`
  // cannot cut it.
  [
    'services/tldw/TldwApiClient.ts',
    'services/tldw/TldwAuth.ts',
    'services/tldw/TldwChat.ts',
    'services/tldw/TldwModels.ts',
    'services/tldw/auth-fetch.ts',
    'services/tldw/chat-model-outcome.ts',
    'services/tldw/domains/chat-rag.ts',
    'services/tldw/domains/prototype-workspaces.ts',
    'services/tldw/index.ts',
    'utils/chat-model-id.ts',
    'utils/chat-model-readiness.ts',
    'utils/resolve-api-provider.ts'
  ],
  [
    'store/workspace-slices/sources-slice.ts',
    'store/workspace-slices/studio-slice.ts',
    'store/workspace-slices/workspace-list-slice.ts',
    'store/workspace.ts'
  ]
]

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
  const unresolved = []
  for (const file of files) {
    const sourceFile = ts.createSourceFile(file, readFileSync(file, 'utf8'), ts.ScriptTarget.Latest)
    for (const specifier of runtimeSpecifiers(sourceFile)) {
      const resolved = ts.resolveModuleName(specifier, file, options, ts.sys, cache).resolvedModule
      const target = resolved && path.resolve(resolved.resolvedFileName)
      if (target && graph.has(target)) graph.get(file).add(target)
      else if (!target && localSpecifier.test(specifier) && !assetSpecifier.test(specifier)) {
        unresolved.push(`${path.relative(sourceRoot, file)} imports "${specifier}"`)
      }
    }
  }
  return {
    fileCount: files.length,
    cycles: cycles(graph).map((members) => members.map((file) => path.relative(sourceRoot, file))),
    unresolved
  }
}

export function checkImportCycles({ cycles: found, unresolved }, baseline = BASELINE) {
  const failures = unresolved.map(
    (edge) =>
      `${edge}, which resolves to no file: fix the path. An import this script cannot resolve is an edge it cannot check.`
  )
  for (const members of found) {
    if (baseline.some((entry) => members.every((file) => entry.includes(file)))) continue
    const listed = members.map((file) =>
      baseline.some((entry) => entry.includes(file)) ? file : `${file}  (not in BASELINE)`
    )
    failures.push(
      `New or grown import cycle of ${members.length} files, not inside any BASELINE entry in ${scriptName}:\n  ${listed.join('\n  ')}\n${howToBreak}`
    )
  }
  for (const entry of baseline) {
    if (found.some((members) => members.some((file) => entry.includes(file)))) continue
    failures.push(
      `Stale baseline: this cycle is gone; delete its BASELINE entry in ${scriptName} in this PR:\n  ${entry.join('\n  ')}`
    )
  }
  return failures
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try {
    const result = findImportCycles()
    const failures = checkImportCycles(result)
    for (const failure of failures) console.error(failure)
    console.log(
      `Shared UI import cycles: ${result.fileCount} files; ${result.cycles.length} runtime cycles against ${BASELINE.length} baselined; ${failures.length} failures.`
    )
    process.exitCode = failures.length ? 1 : 0
  } catch (error) {
    console.error(error)
    process.exitCode = 1
  }
}
