import { readdirSync, readFileSync } from "node:fs"
import path from "node:path"
import ts from "typescript"

const projectDirectories = Object.freeze({
  "tier-1": "e2e/workflows/tier-1-critical",
  "tier-2": "e2e/workflows/tier-2-features",
  "tier-3": "e2e/workflows/tier-3-automation",
})

function stringValue(node, sourceFile) {
  if (!node) return "<unknown>"
  if (ts.isStringLiteralLike(node)) return node.text
  return node.getText(sourceFile)
}

function callPropertyName(node) {
  if (!ts.isCallExpression(node) || !ts.isPropertyAccessExpression(node.expression)) {
    return null
  }
  return node.expression.name.text
}

function isRouteCall(node) {
  return callPropertyName(node) === "route" && node.arguments.length >= 1
}

function containsDirectFulfillOrAbort(node) {
  let found = false
  const visit = (child) => {
    if (found) return
    const property = callPropertyName(child)
    if (property === "fulfill" || property === "abort") {
      found = true
      return
    }
    ts.forEachChild(child, visit)
  }
  visit(node)
  return found
}

function namedFunctionBodies(sourceFile) {
  const bodies = new Map()
  const visit = (node) => {
    if (ts.isFunctionDeclaration(node) && node.name && node.body) {
      bodies.set(node.name.text, node.body)
    } else if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.initializer &&
      (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))
    ) {
      bodies.set(node.name.text, node.initializer.body)
    }
    ts.forEachChild(node, visit)
  }
  visit(sourceFile)
  return bodies
}

function invokedFunctionNames(node) {
  const names = new Set()
  const visit = (child) => {
    if (ts.isCallExpression(child) && ts.isIdentifier(child.expression)) {
      names.add(child.expression.text)
    }
    ts.forEachChild(child, visit)
  }
  visit(node)
  return names
}

function fulfillmentHelpers(sourceFile) {
  const bodies = namedFunctionBodies(sourceFile)
  const helpers = new Set(
    [...bodies].filter(([, body]) => containsDirectFulfillOrAbort(body)).map(([name]) => name)
  )
  let changed = true
  while (changed) {
    changed = false
    for (const [name, body] of bodies) {
      if (helpers.has(name)) continue
      if ([...invokedFunctionNames(body)].some((calledName) => helpers.has(calledName))) {
        helpers.add(name)
        changed = true
      }
    }
  }
  return helpers
}

function containsFulfillOrAbort(node, helperNames) {
  if (containsDirectFulfillOrAbort(node)) return true
  return [...invokedFunctionNames(node)].some((name) => helperNames.has(name))
}

function testCallKind(node) {
  if (!ts.isCallExpression(node)) return null
  const expression = node.expression
  if (ts.isIdentifier(expression) && ["test", "it"].includes(expression.text)) {
    return "runnable"
  }
  if (
    ts.isPropertyAccessExpression(expression) &&
    ts.isIdentifier(expression.expression) &&
    ["test", "it"].includes(expression.expression.text)
  ) {
    if (expression.name.text === "only") return "runnable"
    if (["skip", "fixme"].includes(expression.name.text)) return "skipped"
  }
  return null
}

function testTitle(node) {
  if (testCallKind(node) !== "runnable") return null
  if (!node.arguments.length || !ts.isStringLiteralLike(node.arguments[0])) {
    return null
  }
  return node.arguments[0].text
}

function routeEntriesWithin(node, sourceFile, helperNames) {
  const entries = []
  const visit = (child) => {
    if (isRouteCall(child) && containsFulfillOrAbort(child, helperNames)) {
      const position = sourceFile.getLineAndCharacterOfPosition(child.getStart(sourceFile))
      entries.push({
        line: position.line + 1,
        matcher: stringValue(child.arguments[0], sourceFile),
      })
    }
    ts.forEachChild(child, visit)
  }
  ts.forEachChild(node, visit)
  return entries
}

function namedRouteHelpers(sourceFile, helperNames) {
  const helpers = new Map()
  const visit = (node) => {
    let name = null
    let body = null
    if (ts.isFunctionDeclaration(node) && node.name && node.body) {
      name = node.name.text
      body = node.body
    } else if (
      ts.isVariableDeclaration(node) &&
      ts.isIdentifier(node.name) &&
      node.initializer &&
      (ts.isArrowFunction(node.initializer) || ts.isFunctionExpression(node.initializer))
    ) {
      name = node.name.text
      body = node.initializer.body
    }
    if (name && body) {
      const routes = routeEntriesWithin(body, sourceFile, helperNames)
      if (routes.length) helpers.set(name, routes)
    }
    ts.forEachChild(node, visit)
  }
  visit(sourceFile)
  return helpers
}

function invokedHelpers(node, helpers) {
  const names = new Set()
  const visit = (child) => {
    if (ts.isCallExpression(child) && ts.isIdentifier(child.expression) && helpers.has(child.expression.text)) {
      names.add(child.expression.text)
    }
    ts.forEachChild(child, visit)
  }
  ts.forEachChild(node, visit)
  return names
}

/**
 * Inventory API routes that fulfill or abort requests. Passive response waits
 * remain live evidence and are deliberately excluded.
 */
export function inventorySource(source, { project = "unknown", file = "unknown" } = {}) {
  const sourceFile = ts.createSourceFile(file, source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TS)
  const fulfillHelperNames = fulfillmentHelpers(sourceFile)
  const helpers = namedRouteHelpers(sourceFile, fulfillHelperNames)
  const entries = []
  const helperRouteKeysUsedByTests = new Set()
  const skippedRouteKeys = new Set()

  const visitTests = (node) => {
    if (testCallKind(node) === "skipped") {
      for (const route of routeEntriesWithin(node, sourceFile, fulfillHelperNames)) {
        skippedRouteKeys.add(`${route.line}:${route.matcher}`)
      }
      return
    }
    const title = testTitle(node)
    if (title) {
      for (const route of routeEntriesWithin(node, sourceFile, fulfillHelperNames)) {
        entries.push({ project, file, ...route, kind: "intercepted", test: title })
      }
      for (const helperName of invokedHelpers(node, helpers)) {
        for (const route of helpers.get(helperName)) {
          helperRouteKeysUsedByTests.add(`${route.line}:${route.matcher}`)
          entries.push({ project, file, ...route, kind: "intercepted", test: title })
        }
      }
      return
    }
    ts.forEachChild(node, visitTests)
  }
  visitTests(sourceFile)

  for (const route of routeEntriesWithin(sourceFile, sourceFile, fulfillHelperNames)) {
    const key = `${route.line}:${route.matcher}`
    if (skippedRouteKeys.has(key)) continue
    if (entries.some((entry) => entry.line === route.line && entry.matcher === route.matcher)) {
      continue
    }
    if (!helperRouteKeysUsedByTests.has(key)) {
      entries.push({ project, file, ...route, kind: "intercepted", test: null })
    }
  }

  return entries.filter((entry, index, all) =>
    all.findIndex((candidate) =>
      candidate.project === entry.project &&
      candidate.file === entry.file &&
      candidate.line === entry.line &&
      candidate.matcher === entry.matcher &&
      candidate.test === entry.test
    ) === index
  )
}

function specFiles(directory) {
  return readdirSync(directory, { withFileTypes: true }).flatMap((entry) => {
    const resolved = path.join(directory, entry.name)
    if (entry.isDirectory()) return specFiles(resolved)
    return entry.isFile() && entry.name.endsWith(".spec.ts") ? [resolved] : []
  })
}

export function inventoryProjects(frontendRoot, projects) {
  return projects.flatMap((project) => {
    const relativeDirectory = projectDirectories[project]
    if (!relativeDirectory) throw new Error(`Unknown Tier project: ${project}`)
    const directory = path.join(frontendRoot, relativeDirectory)
    return specFiles(directory).flatMap((filePath) =>
      inventorySource(readFileSync(filePath, "utf8"), {
        project,
        file: path.relative(frontendRoot, filePath),
      })
    )
  })
}
