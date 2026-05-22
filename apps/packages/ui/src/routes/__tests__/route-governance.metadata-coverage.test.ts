import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import * as ts from "typescript"
import { describe, expect, it } from "vitest"

import { PAGES } from "../../../../../tldw-frontend/e2e/smoke/page-inventory"
import { getRouteMetadata, ROUTE_METADATA } from "../route-metadata"
import * as routePathExports from "../route-paths"

const sorted = (values: string[]): string[] => [...values].sort()

const pageEntryByPath = new Map(PAGES.map((entry) => [entry.path, entry]))
const testDir = path.dirname(fileURLToPath(import.meta.url))
const routePathConstants = routePathExports as Record<string, unknown>

const routeRegistryPath = [
  path.resolve(testDir, "../route-registry.tsx"),
  "src/routes/route-registry.tsx",
  "packages/ui/src/routes/route-registry.tsx",
  "../packages/ui/src/routes/route-registry.tsx",
  "apps/packages/ui/src/routes/route-registry.tsx"
].find((candidate) => existsSync(candidate))

if (!routeRegistryPath) {
  throw new Error("Unable to locate route-registry.tsx for governance test")
}

const getPropertyNameText = (name: ts.PropertyName): string | undefined => {
  if (
    ts.isIdentifier(name) ||
    ts.isStringLiteral(name) ||
    ts.isNumericLiteral(name)
  ) {
    return name.text
  }

  return undefined
}

const getObjectProperty = (
  objectLiteral: ts.ObjectLiteralExpression,
  propertyName: string
): ts.PropertyAssignment | undefined =>
  objectLiteral.properties.find(
    (property): property is ts.PropertyAssignment =>
      ts.isPropertyAssignment(property) &&
      getPropertyNameText(property.name) === propertyName
  )

const readRoutePathExpression = (
  expression: ts.Expression,
  context: string
): string => {
  if (
    ts.isStringLiteral(expression) ||
    ts.isNoSubstitutionTemplateLiteral(expression)
  ) {
    return expression.text
  }

  if (ts.isIdentifier(expression)) {
    const value = routePathConstants[expression.text]

    if (typeof value === "string") {
      return value
    }

    throw new Error(
      `Unable to resolve route path constant ${expression.text} in ${context}`
    )
  }

  throw new Error(
    `Unsupported route path expression ${expression.getText()} in ${context}`
  )
}

const extractRoutePathsFromRouteObjects = (
  source: string,
  fileName: string
): string[] => {
  const sourceFile = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX
  )
  const routePaths: string[] = []

  const visit = (node: ts.Node) => {
    if (ts.isObjectLiteralExpression(node)) {
      const pathProperty = getObjectProperty(node, "path")
      const kindProperty = getObjectProperty(node, "kind")

      if (
        pathProperty &&
        kindProperty &&
        ts.isStringLiteral(kindProperty.initializer) &&
        kindProperty.initializer.text === "options"
      ) {
        const { line } = sourceFile.getLineAndCharacterOfPosition(
          pathProperty.getStart(sourceFile)
        )
        routePaths.push(
          readRoutePathExpression(
            pathProperty.initializer,
            `${fileName}:${line + 1}`
          )
        )
      }
    }

    ts.forEachChild(node, visit)
  }

  visit(sourceFile)

  return Array.from(new Set(routePaths)).sort()
}

const optionRegistryPaths = extractRoutePathsFromRouteObjects(
  readFileSync(routeRegistryPath, "utf8"),
  routeRegistryPath
)

describe("route governance metadata coverage", () => {
  it("does not define duplicate smoke inventory paths", () => {
    const paths = PAGES.map((entry) => entry.path)
    const duplicatePaths = paths.filter(
      (path, index) => paths.indexOf(path) !== index
    )

    expect(sorted(duplicatePaths)).toEqual([])
  })

  it("covers every shared option route", () => {
    const missingMetadata = optionRegistryPaths
      .filter((path) => !path.includes(":"))
      .filter((path) => !getRouteMetadata(path))

    expect(sorted(missingMetadata)).toEqual([])
  })

  it("requires every active smoke inventory route to have metadata", () => {
    const missingMetadata = PAGES
      .filter((entry) => !entry.skip)
      .filter((entry) => !getRouteMetadata(entry.path))
      .map((entry) => entry.path)

    expect(sorted(missingMetadata)).toEqual([])
  })

  it("requires skipped smoke inventory routes to have metadata and reasons", () => {
    const invalidSkippedRoutes = PAGES
      .filter((entry) => entry.skip)
      .filter((entry) => {
        const metadata = getRouteMetadata(entry.path)

        return !metadata || !entry.skip?.trim() || metadata.smoke === "include"
      })
      .map((entry) => entry.path)

    expect(sorted(invalidSkippedRoutes)).toEqual([])
  })

  it("keeps included web smoke routes active in the page inventory", () => {
    const missingIncludedRoutes = ROUTE_METADATA
      .filter((metadata) => metadata.availability.includes("web"))
      .filter((metadata) => metadata.smoke === "include")
      .filter((metadata) => {
        const pageEntry = pageEntryByPath.get(metadata.path)

        return !pageEntry || Boolean(pageEntry.skip)
      })
      .map((metadata) => metadata.path)

    expect(sorted(missingIncludedRoutes)).toEqual([])
  })

  it("does not run smoke-excluded routes as active page inventory entries", () => {
    const activeExcludedRoutes = PAGES
      .filter((entry) => !entry.skip)
      .filter((entry) => getRouteMetadata(entry.path)?.smoke === "exclude")
      .map((entry) => entry.path)

    expect(sorted(activeExcludedRoutes)).toEqual([])
  })
})
