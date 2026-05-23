import { existsSync, readFileSync } from "node:fs"
import * as ts from "typescript"

import * as routePathExports from "../route-paths"

const routePathConstants = routePathExports as Record<string, unknown>

export type RouteRegistrySource = {
  path: string
  source: string
}

export type RouteObjectExtractionOptions = {
  kind?: string
  requireNav?: boolean
}

export const uniqueSorted = (values: string[]): string[] =>
  Array.from(new Set(values)).sort()

export const resolveFirstExistingPath = (
  candidates: string[],
  label: string
): string => {
  const sourcePath = candidates.find((candidate) => existsSync(candidate))

  if (!sourcePath) {
    throw new Error(`Unable to locate ${label}`)
  }

  return sourcePath
}

export const readFirstExistingSource = (
  candidates: string[],
  label: string
): RouteRegistrySource => {
  const sourcePath = resolveFirstExistingPath(candidates, label)

  return {
    path: sourcePath,
    source: readFileSync(sourcePath, "utf8")
  }
}

export const readOptionalFirstExistingSource = (
  candidates: string[]
): RouteRegistrySource | undefined => {
  const sourcePath = candidates.find((candidate) => existsSync(candidate))

  if (!sourcePath) {
    return undefined
  }

  return {
    path: sourcePath,
    source: readFileSync(sourcePath, "utf8")
  }
}

export const getPropertyNameText = (
  name: ts.PropertyName
): string | undefined => {
  if (
    ts.isIdentifier(name) ||
    ts.isStringLiteral(name) ||
    ts.isNumericLiteral(name)
  ) {
    return name.text
  }

  return undefined
}

export const getObjectProperty = (
  objectLiteral: ts.ObjectLiteralExpression,
  propertyName: string
): ts.PropertyAssignment | undefined =>
  objectLiteral.properties.find(
    (property): property is ts.PropertyAssignment =>
      ts.isPropertyAssignment(property) &&
      getPropertyNameText(property.name) === propertyName
  )

export const readRoutePathExpression = (
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

export const extractRoutePathsFromRouteObjects = (
  source: string,
  fileName: string,
  options: RouteObjectExtractionOptions = {}
): string[] => {
  const sourceFile = ts.createSourceFile(
    fileName,
    source,
    ts.ScriptTarget.Latest,
    true,
    ts.ScriptKind.TSX
  )
  const routePaths: string[] = []

  const matchesKind = (objectLiteral: ts.ObjectLiteralExpression): boolean => {
    if (!options.kind) {
      return true
    }

    const kindProperty = getObjectProperty(objectLiteral, "kind")

    if (!kindProperty || !ts.isStringLiteral(kindProperty.initializer)) {
      return false
    }

    return kindProperty.initializer.text === options.kind
  }

  const visit = (node: ts.Node) => {
    if (ts.isObjectLiteralExpression(node)) {
      const pathProperty = getObjectProperty(node, "path")
      const navProperty = getObjectProperty(node, "nav")

      if (
        pathProperty &&
        matchesKind(node) &&
        (!options.requireNav || navProperty)
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

  return uniqueSorted(routePaths)
}
