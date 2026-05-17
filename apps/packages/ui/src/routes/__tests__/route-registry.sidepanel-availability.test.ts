import { existsSync, readFileSync } from "node:fs"
import path from "node:path"
import { fileURLToPath } from "node:url"
import * as ts from "typescript"
import { describe, expect, it } from "vitest"

import {
  getRouteMetadata,
  isRouteVisibleForSurface
} from "../route-metadata"
import * as routePathExports from "../route-paths"

const readFirstExistingSource = (
  candidates: string[],
  label: string
): string => {
  const sourcePath = candidates.find((candidate) => existsSync(candidate))

  if (!sourcePath) {
    throw new Error(`Unable to locate ${label}`)
  }

  return readFileSync(sourcePath, "utf8")
}

const testDir = path.dirname(fileURLToPath(import.meta.url))
const routePathConstants = routePathExports as Record<string, unknown>

const sharedSidepanelRegistrySource = readFirstExistingSource(
  [
    path.resolve(testDir, "../sidepanel-route-registry.tsx"),
    "src/routes/sidepanel-route-registry.tsx",
    "packages/ui/src/routes/sidepanel-route-registry.tsx",
    "../packages/ui/src/routes/sidepanel-route-registry.tsx",
    "apps/packages/ui/src/routes/sidepanel-route-registry.tsx"
  ],
  "shared sidepanel-route-registry.tsx"
)

const extensionSidepanelRegistrySource = readFirstExistingSource(
  [
    path.resolve(
      testDir,
      "../../../../../tldw-frontend/extension/routes/sidepanel-route-registry.tsx"
    ),
    "../../tldw-frontend/extension/routes/sidepanel-route-registry.tsx",
    "tldw-frontend/extension/routes/sidepanel-route-registry.tsx",
    "apps/tldw-frontend/extension/routes/sidepanel-route-registry.tsx"
  ],
  "extension sidepanel-route-registry.tsx"
)

const extensionOptionRegistrySource = readFirstExistingSource(
  [
    path.resolve(
      testDir,
      "../../../../../tldw-frontend/extension/routes/route-registry.tsx"
    ),
    "../../tldw-frontend/extension/routes/route-registry.tsx",
    "tldw-frontend/extension/routes/route-registry.tsx",
    "apps/tldw-frontend/extension/routes/route-registry.tsx"
  ],
  "extension route-registry.tsx"
)

const uniqueSorted = (values: string[]): string[] =>
  Array.from(new Set(values)).sort()

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
  fileName: string,
  options: { requireNav?: boolean } = {}
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
      const navProperty = getObjectProperty(node, "nav")

      if (pathProperty && (!options.requireNav || navProperty)) {
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

  return uniqueSorted(routePaths)
}

const sidepanelRoutePaths = uniqueSorted([
  ...extractRoutePathsFromRouteObjects(
    sharedSidepanelRegistrySource,
    "shared sidepanel-route-registry.tsx"
  ),
  ...extractRoutePathsFromRouteObjects(
    extensionSidepanelRegistrySource,
    "extension sidepanel-route-registry.tsx"
  )
])

const extensionOptionNavPaths = uniqueSorted(
  extractRoutePathsFromRouteObjects(
    extensionOptionRegistrySource,
    "extension route-registry.tsx",
    { requireNav: true }
  ).filter(
    (routePath) =>
      !routePath.includes(":") && !sidepanelRoutePaths.includes(routePath)
  )
)

describe("sidepanel route availability metadata", () => {
  it("keeps extension sidepanel chat reachable at both root and /chat", () => {
    expect(extensionSidepanelRegistrySource).toMatch(/path\s*:\s*["']\/["']/)
    expect(extensionSidepanelRegistrySource).toMatch(/path\s*:\s*["']\/chat["']/)
    expect(extensionOptionRegistrySource).toMatch(/path\s*:\s*["']\/["']/)
    expect(extensionOptionRegistrySource).toMatch(/path\s*:\s*["']\/chat["']/)
  })

  it("declares sidepanel availability for every shared or extension sidepanel route", () => {
    const routesMissingSidepanelAvailability = sidepanelRoutePaths.filter(
      (routePath) =>
        !getRouteMetadata(routePath)?.availability.includes("extension_sidepanel")
    )

    expect(routesMissingSidepanelAvailability).toEqual([])
  })

  it("defines metadata labels and groups for extension option routes that appear in nav", () => {
    const routesMissingNavMetadata = extensionOptionNavPaths.filter((routePath) => {
      const metadata = getRouteMetadata(routePath)

      return !metadata?.label || !metadata.group
    })

    expect(routesMissingNavMetadata).toEqual([])
  })

  it("marks sidepanel debug routes as internal QA/debug routes", () => {
    const debugRoutes = [
      "/error-boundary-test",
      "/__debug__/sidepanel-chat",
      "/__debug__/sidepanel-error-boundary"
    ]

    for (const routePath of debugRoutes) {
      const metadata = getRouteMetadata(routePath)

      expect(metadata?.surface, routePath).toBe("internal_qa_debug")
      expect(metadata?.nav, routePath).toBe("hidden")
      expect(metadata?.commandPalette, routePath).toBe("hide")
    }
  })

  it("does not infer sidepanel availability from web or extension options availability", () => {
    expect(isRouteVisibleForSurface("/chat", "extension_sidepanel")).toBe(true)
    expect(isRouteVisibleForSurface("/media", "extension_sidepanel")).toBe(false)
    expect(isRouteVisibleForSurface("/settings/model", "extension_sidepanel")).toBe(
      false
    )
  })
})
