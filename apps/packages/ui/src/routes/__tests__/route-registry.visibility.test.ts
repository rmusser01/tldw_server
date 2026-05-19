import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import * as ts from "typescript"
import { describe, expect, it } from "vitest"

import { getRouteMetadata, ROUTE_METADATA } from "../route-metadata"
import * as routePathExports from "../route-paths"

const isDynamicRoutePath = (routePath: string): boolean =>
  routePath.includes(":") || routePath.includes("*")

const testDir = path.dirname(fileURLToPath(import.meta.url))
const routePathConstants = routePathExports as Record<string, unknown>

const routeRegistryPathCandidates = [
  path.resolve(testDir, "../route-registry.tsx"),
  "src/routes/route-registry.tsx",
  "packages/ui/src/routes/route-registry.tsx",
  "../packages/ui/src/routes/route-registry.tsx",
  "apps/packages/ui/src/routes/route-registry.tsx"
]

const routeRegistryPath = routeRegistryPathCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!routeRegistryPath) {
  throw new Error("Unable to locate route-registry.tsx for visibility test")
}

const routeRegistrySource = readFileSync(routeRegistryPath, "utf8")

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

      if (pathProperty) {
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
  routeRegistrySource,
  routeRegistryPath
)

const nonDynamicOptionRegistryPaths = optionRegistryPaths.filter(
  (routePath) => !isDynamicRoutePath(routePath)
)

const frontendPagesRootCandidates = [
  path.resolve(process.cwd(), "../../tldw-frontend/pages"),
  path.resolve(process.cwd(), "tldw-frontend/pages"),
  path.resolve(process.cwd(), "apps/tldw-frontend/pages"),
  path.resolve(testDir, "../../../../../tldw-frontend/pages")
]

const frontendPagesRoot = frontendPagesRootCandidates.find((candidate) =>
  existsSync(candidate)
)

if (!frontendPagesRoot) {
  throw new Error("Unable to locate tldw-frontend/pages for visibility test")
}

const routePathToPageCandidates = (routePath: string): string[] => {
  const normalizedPath = routePath === "/" ? "/index" : routePath
  const pagePath = normalizedPath.replace(/^\//, "")

  return [
    path.join(frontendPagesRoot, `${pagePath}.tsx`),
    path.join(frontendPagesRoot, pagePath, "index.tsx")
  ]
}

const hasNextPageFile = (routePath: string): boolean =>
  routePathToPageCandidates(routePath).some((candidate) => existsSync(candidate))

const isRegistryBackedRoute = (routePath: string): boolean =>
  optionRegistryPaths.includes(routePath)

describe("route registry visibility metadata", () => {
  it("defines metadata for every non-dynamic option registry route", () => {
    const missingMetadata = nonDynamicOptionRegistryPaths.filter(
      (routePath) => !getRouteMetadata(routePath)
    )

    expect(missingMetadata).toEqual([])
  })

  it("does not claim web availability for unknown routes", () => {
    for (const metadata of ROUTE_METADATA) {
      if (!metadata.availability.includes("web")) {
        continue
      }

      const hasRouteOwner =
        isRegistryBackedRoute(metadata.path) ||
        hasNextPageFile(metadata.path) ||
        Boolean(metadata.redirectsTo)

      expect(hasRouteOwner, metadata.path).toBe(true)
    }
  })

  it("keeps internal QA and debug routes out of primary navigation", () => {
    for (const metadata of ROUTE_METADATA) {
      if (metadata.surface !== "internal_qa_debug") {
        continue
      }

      expect(metadata.nav, metadata.path).not.toBe("primary")
      expect(metadata.commandPalette, metadata.path).toBe("hide")
    }
  })

  it("keeps hosted-only routes out of default self-hosted navigation", () => {
    for (const metadata of ROUTE_METADATA) {
      if (metadata.surface !== "hosted_only") {
        continue
      }

      expect(metadata.nav, metadata.path).not.toBe("primary")
    }
  })
})
