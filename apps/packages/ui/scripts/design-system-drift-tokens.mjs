import ts from "typescript"
import postcss from "postcss"
import defaultTheme from "tailwindcss/defaultTheme.js"

/** @typedef {"shared-ui" | "webui" | "extension" | "admin-ui"} Surface */
/** @typedef {{path: string, source: string}} Source */
/** @typedef {{path: string, code: string, message: string, severity: "error" | "coverage", line?: number, column?: number}} Diagnostic */
/** @typedef {{path: string, start: number, end: number, owner: string, kind: string}} Definition */
/** @typedef {{semanticColors: Set<string>, cssVariables: Set<string>, radiusClasses: Set<string>, radiusValues: Set<string>, definitionPaths: string[]}} SurfaceAuthority */
/** @typedef {{surfaces: Record<Surface, SurfaceAuthority>, definitionScopes: Definition[], diagnostics: Diagnostic[]}} Authority */

const WEB_CONFIG = "apps/tldw-frontend/tailwind.config.js"
const EXT_CONFIG = "apps/extension/tailwind.config.js"
const ADMIN_CONFIG = "admin-ui/tailwind.config.ts"
const SHARED_CSS = "apps/packages/ui/src/assets/tailwind-shared.css"
const ADMIN_CSS = "admin-ui/app/globals.css"
const TOKEN_ROOT = "apps/packages/ui/src/styles/tokens/"
const THEME_ROOT = "apps/packages/ui/src/themes/"

// Pin declarations, not directories: new exports/custom properties need an ownership review.
const TOKEN_EXPORTS = {
  "spacing.ts": ["spacing", "padding", "paddingX", "paddingY", "margin"],
  "typography.ts": ["fontFamily", "fontSize", "fontWeight", "lineHeight", "letterSpacing"],
  "colors.ts": ["bgColor", "textColor", "borderColor", "statusColors", "temporaryChatIndicator"],
  "motion.ts": ["duration", "easing", "transition", "motionSafe"],
  "layout.ts": ["zIndex", "container", "sidebarWidth", "touchTarget"],
  "borders.ts": ["borderRadius", "borderWidth", "borderStyle"],
  "shadows.ts": ["shadow", "innerShadow"],
  "focus.ts": ["focusRing"],
  "components.ts": ["buttonBase", "buttonSize", "iconButton", "iconSize", "badge", "inputBase", "cardBase"],
  "index.ts": [],
}
const THEME_NAMES = {
  "presets.ts": ["defaultTheme", "solarizedTheme", "nordTheme", "highContrastTheme", "rosePineTheme"],
  "primer-preset.ts": ["primerTheme"],
}

export const REQUIRED_DEFINITION_PATHS = Object.freeze([
  WEB_CONFIG, EXT_CONFIG, ADMIN_CONFIG, SHARED_CSS, ADMIN_CSS,
])
export const TOKEN_SOURCE_PATHS = Object.freeze([
  ...REQUIRED_DEFINITION_PATHS,
  ...Object.keys(TOKEN_EXPORTS).map(name => TOKEN_ROOT + name),
  ...Object.keys(THEME_NAMES).map(name => THEME_ROOT + name),
])

const SHARED_VARIABLES = new Set([
  "color-bg", "color-surface", "color-surface-2", "color-elevated", "color-primary",
  "color-primary-foreground", "color-primary-strong", "color-primary-strong-foreground",
  "color-accent", "color-success", "color-warn", "color-danger", "color-muted",
  "color-border", "color-border-strong", "color-text", "color-text-muted", "color-text-subtle", "color-focus",
  "state-ready", "state-unavailable", "state-setup-required", "state-auth-required", "state-permission-denied",
  "state-degraded", "state-retrying", "state-blocked", "state-empty", "state-loading", "state-error",
  "shadow-sm", "shadow-md", "font-family", "font-family-mono", "font-size-body", "font-size-message",
  "font-size-caption", "font-size-label", "radius-sm", "radius-md", "radius-lg", "radius-xl", "surface-blur",
  "sidebar-width", "sidebar-collapsed-width", "header-height", "content-max-width", "spacing-unit", "duration-base",
  "moderation-hero-start", "moderation-hero-end", "moderation-hero-border", "moderation-hero-shadow",
  "moderation-hero-grid-1", "moderation-hero-grid-2", "moderation-hero-grid-opacity",
  "glow-primary", "glow-accent", "glow-success",
].map(name => `--${name}`))
const ADMIN_VARIABLES = new Set([
  "background", "foreground", "card", "card-foreground", "primary", "primary-foreground",
  "secondary", "secondary-foreground", "muted", "muted-foreground", "accent", "accent-foreground",
  "destructive", "destructive-foreground", "border", "input", "ring", "popover", "popover-foreground",
  "chart-1", "chart-2", "radius",
].map(name => `--${name}`))
const THEME_PROPERTIES = new Set([
  "colors", "borderRadius", "fontFamily", "fontSize", "boxShadow", "backgroundImage", "maskImage", "keyframes", "animation",
])

/** Unwrap syntax only; never evaluate calls, getters, imports or computed keys. */
function unwrap(node) {
  while (node && (ts.isAsExpression(node) || ts.isSatisfiesExpression(node) ||
    ts.isParenthesizedExpression(node) || ts.isTypeAssertionExpression(node))) node = node.expression
  return node
}

function key(node) {
  return node && (ts.isIdentifier(node) || ts.isStringLiteral(node) || ts.isNumericLiteral(node)) ? node.text : null
}

function variables(value) {
  return [...value.matchAll(/var\(\s*(--[\w-]+)/g)].map(match => match[1])
}

function emptySurface() {
  return { semanticColors: new Set(), cssVariables: new Set(), radiusClasses: new Set(), radiusValues: new Set(), definitionPaths: [] }
}

/** Build surface-specific authority from supplied repository syntax; performs no IO.
 * @param {{sources: Source[]}} options
 * @returns {Authority}
 */
export function buildTokenAuthority({ sources }) {
  const definitionScopes = []
  const diagnostics = []
  const inputs = new Map(sources.map(source => [source.path, source.source]))
  const parsed = new Map()
  const cssValues = new Map()
  const surfaces = { "shared-ui": emptySurface(), webui: emptySurface(), extension: emptySurface(), "admin-ui": emptySurface() }
  const addDiagnostic = (path, code, message, severity = "coverage", ast, node) => {
    const diagnostic = { path, code, message, severity }
    if (ast && node) {
      const position = ast.getLineAndCharacterOfPosition(node.getStart(ast))
      diagnostic.line = position.line + 1
      diagnostic.column = position.character + 1
    }
    diagnostics.push(diagnostic)
  }
  const scope = (path, ast, node, owner, kind) => {
    definitionScopes.push({ path, start: node.getStart(ast), end: node.end, owner, kind })
  }

  for (const path of REQUIRED_DEFINITION_PATHS) {
    if (!inputs.has(path)) addDiagnostic(path, "missing-definition-input", "Required token/configuration input is missing.", "error")
  }
  for (const path of TOKEN_SOURCE_PATHS) {
    if (!inputs.has(path)) continue
    const source = inputs.get(path)
    if (path.endsWith(".css")) {
      const allowed = path === SHARED_CSS ? SHARED_VARIABLES : ADMIN_VARIABLES
      const values = new Map()
      cssValues.set(path, values)
      try {
        const css = postcss.parse(source, { from: path })
        css.walkDecls(decl => {
          if (!allowed.has(decl.prop) || decl.parent.type !== "rule" ||
            ![":root", ".dark"].includes(decl.parent.selector.trim())) return
          if (!values.has(decl.prop)) values.set(decl.prop, new Set())
          values.get(decl.prop).add(decl.value)
          const start = decl.source.start.offset
          definitionScopes.push({ path, start, end: start + decl.toString().length, owner: decl.parent.selector, kind: "css-variable" })
        })
        if (!values.size) addDiagnostic(path, "missing-token-declarations",
          "Required stylesheet contains no recognized token declarations.", "error")
      } catch (error) {
        values.clear()
        diagnostics.push({ path, code: "definition-parse-error", message: error.reason ?? error.message,
          severity: "error", line: error.line, column: error.column })
      }
    } else {
      const ast = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true,
        path.endsWith(".ts") ? ts.ScriptKind.TS : ts.ScriptKind.JS)
      for (const diagnostic of ast.parseDiagnostics) {
        const position = ast.getLineAndCharacterOfPosition(diagnostic.start ?? 0)
        diagnostics.push({ path, code: "definition-parse-error", message: ts.flattenDiagnosticMessageText(diagnostic.messageText, " "),
          severity: "error", line: position.line + 1, column: position.character + 1 })
      }
      if (ast.parseDiagnostics.length) continue
      parsed.set(path, ast)
      const names = TOKEN_EXPORTS[path.slice(TOKEN_ROOT.length)]
      const themes = THEME_NAMES[path.slice(THEME_ROOT.length)]
      for (const statement of ast.statements) {
        if (!ts.isVariableStatement(statement)) continue
        const exported = statement.modifiers?.some(modifier => modifier.kind === ts.SyntaxKind.ExportKeyword)
        for (const declaration of statement.declarationList.declarations) {
          const name = key(declaration.name)
          const value = unwrap(declaration.initializer)
          if (path.startsWith(TOKEN_ROOT) && names?.includes(name) && exported &&
            value && (ts.isObjectLiteralExpression(value) || ts.isStringLiteral(value))) {
            scope(path, ast, declaration, name, "token-export")
          }
          if (path.startsWith(THEME_ROOT) && themes?.includes(name) && value && ts.isObjectLiteralExpression(value)) {
            scope(path, ast, declaration, name, "theme")
          }
        }
      }
    }
  }

  function readConfig(path) {
    const ast = parsed.get(path)
    if (!ast) return null
    const constantTables = new Map()
    const constantsFor = file => {
      if (!constantTables.has(file)) {
        const constants = new Map()
        for (const statement of file.statements) {
          if (ts.isVariableStatement(statement) && (statement.declarationList.flags & ts.NodeFlags.Const)) {
            for (const declaration of statement.declarationList.declarations) {
              if (ts.isIdentifier(declaration.name)) constants.set(declaration.name.text, declaration.initializer)
            }
          }
        }
        constantTables.set(file, constants)
      }
      return constantTables.get(file)
    }
    const exportedFrom = file => {
      let exported
      for (const statement of file?.statements ?? []) {
        if (ts.isExportAssignment(statement) && !statement.isExportEquals) exported = statement.expression
        if (ts.isExpressionStatement(statement) && ts.isBinaryExpression(statement.expression) &&
          statement.expression.operatorToken.kind === ts.SyntaxKind.EqualsToken &&
          statement.expression.left.getText(file) === "module.exports") exported = statement.expression.right
      }
      return exported
    }
    const unsupported = (node, message) => {
      const file = node?.getSourceFile() ?? ast
      addDiagnostic(file.fileName, "unsupported-config-expression", message, "coverage", file, node ?? file)
    }
    const unavailable = message => addDiagnostic(path, "unusable-config-authority", message, "error", ast, ast)
    const valueOf = prop => {
      if (!prop) return undefined
      if (ts.isPropertyAssignment(prop)) return prop.initializer
      if (ts.isShorthandPropertyAssignment(prop)) return prop.name
      unsupported(prop, `Cannot statically read ${key(prop.name)} accessor or method.`)
      return undefined
    }
    const resolve = (node, visited = new Set()) => {
      node = unwrap(node)
      if (!node || visited.has(node)) return node
      const next = new Set([...visited, node])
      const file = node.getSourceFile()
      if (ts.isIdentifier(node) && constantsFor(file).has(node.text)) {
        return resolve(constantsFor(file).get(node.text), next)
      }
      if (ts.isPropertyAccessExpression(node)) {
        const container = object(node.expression, "referenced theme object", next)
        return resolve(valueOf(container?.get(node.name.text)), next)
      }
      if (file.fileName === EXT_CONFIG && ts.isCallExpression(node) &&
        ts.isIdentifier(node.expression) && node.expression.text === "require" &&
        node.arguments.length === 1 && ts.isStringLiteral(node.arguments[0]) &&
        node.arguments[0].text === "../tldw-frontend/tailwind.config.js") {
        return resolve(exportedFrom(parsed.get(WEB_CONFIG)), next)
      }
      return node
    }
    const object = (node, label, visited = new Set()) => {
      const resolved = resolve(node, visited)
      if (!resolved || !ts.isObjectLiteralExpression(resolved) || visited.has(resolved)) {
        unsupported(node, `Cannot statically read ${label}.`)
        return null
      }
      const next = new Set([...visited, resolved])
      const entries = new Map()
      for (const prop of resolved.properties) {
        if (ts.isSpreadAssignment(prop)) {
          for (const [name, inherited] of object(prop.expression, label, next) ?? []) entries.set(name, inherited)
        } else if (key(prop.name)) {
          entries.set(key(prop.name), prop)
        } else {
          unsupported(prop, `Dynamic ${label} property is not statically resolved.`)
        }
      }
      return entries
    }
    const exported = exportedFrom(ast)
    const root = object(exported, "exported Tailwind config")
    if (!root) {
      unavailable("Required exported Tailwind configuration cannot be read statically.")
      return null
    }
    const themeProp = root.get("theme")
    const theme = themeProp ? object(valueOf(themeProp), "theme") : null
    if (!theme) {
      unavailable("Required Tailwind theme is missing or cannot be read statically.")
      return null
    }
    const extendProp = theme.get("extend")
    const extend = extendProp ? object(valueOf(extendProp), "theme.extend") : null
    if (![theme, extend].some(container => container?.has("colors") || container?.has("borderRadius"))) {
      unavailable("Required Tailwind theme has no readable color or radius definition properties.")
      return null
    }
    for (const container of [theme, extend]) {
      for (const [name, prop] of container ?? []) {
        if (prop.getSourceFile() === ast && ts.isPropertyAssignment(prop) && THEME_PROPERTIES.has(name)) {
          scope(path, ast, prop, `theme.${container === extend ? "extend." : ""}${name}`, "tailwind-property")
        }
      }
    }
    const leaves = (node, label, prefix = "", result = new Map(), visited = new Set()) => {
      node = resolve(node)
      if (visited.has(node)) {
        unsupported(node, `Cyclic ${label} value is not statically resolved.`)
        return result
      }
      const obj = object(node, label)
      if (!obj) return result
      const next = new Set([...visited, node])
      for (const [name, prop] of obj) {
        const fullName = name === "DEFAULT" ? prefix : [prefix, name].filter(Boolean).join("-")
        const value = resolve(valueOf(prop))
        if (value && ts.isObjectLiteralExpression(value)) leaves(value, label, fullName, result, next)
        else if (value && (ts.isStringLiteral(value) || ts.isNoSubstitutionTemplateLiteral(value))) result.set(fullName, value.text)
        else unsupported(prop, `Dynamic ${label} value is not statically resolved.`)
      }
      return result
    }
    let configuredEntries = 0
    const readScale = (name, defaults) => {
      const direct = theme.get(name)
      const extra = extend?.get(name)
      const directValues = direct ? leaves(valueOf(direct), name) : null
      const extraValues = extra ? leaves(valueOf(extra), name) : new Map()
      configuredEntries += (directValues?.size ?? 0) + extraValues.size
      return new Map([
        ...(directValues ?? defaults),
        ...extraValues,
      ])
    }
    const colors = readScale("colors", [])
    const radii = readScale("borderRadius", Object.entries(defaultTheme.borderRadius))
    if (!configuredEntries) {
      unavailable("Required Tailwind theme contains no statically readable color or radius entries.")
      return null
    }
    return {
      path: themeProp.getSourceFile().fileName,
      colors,
      radii,
    }
  }

  const web = readConfig(WEB_CONFIG)
  const extension = readConfig(EXT_CONFIG)
  const admin = readConfig(ADMIN_CONFIG)
  function fill(surface, config, cssPath, paths) {
    const target = surfaces[surface]
    target.definitionPaths = paths.filter(path => inputs.has(path)).sort()
    const values = cssValues.get(cssPath) ?? new Map()
    target.cssVariables = new Set([...values.keys()].sort())
    if (!config) return
    for (const [name, value] of config.colors) {
      const refs = variables(value)
      if (refs.length && refs.every(ref => values.has(ref))) target.semanticColors.add(name)
      else if (refs.length) addDiagnostic(config.path,
        "unresolved-semantic-variable", `Color ${name} references a variable without an established declaration.`)
    }
    for (const [name, value] of config.radii) {
      target.radiusClasses.add(name === "" || name === "DEFAULT" ? "rounded" : `rounded-${name}`)
      target.radiusValues.add(value)
      const refs = variables(value)
      for (const ref of refs) {
        if (!values.has(ref)) continue
        // Only a direct variable aliases its values; calc() expressions remain symbolic.
        if (/^var\(\s*--[\w-]+\s*(?:,[^()]*)?\)$/.test(value)) {
          target.radiusValues.add(`var(${ref})`)
          for (const literal of values.get(ref)) target.radiusValues.add(literal)
        }
      }
      if (refs.length) addDiagnostic(config.path,
        "symbolic-radius", `Radius ${name || "DEFAULT"} remains symbolic; unresolved runtime values or unit comparisons require coverage review.`)
    }
    for (const name of ["semanticColors", "radiusClasses", "radiusValues"]) target[name] = new Set([...target[name]].sort())
  }
  const sharedPaths = TOKEN_SOURCE_PATHS.filter(path => path !== EXT_CONFIG && path !== ADMIN_CONFIG && path !== ADMIN_CSS)
  fill("shared-ui", web, SHARED_CSS, sharedPaths)
  fill("webui", web, SHARED_CSS, sharedPaths)
  fill("extension", extension, SHARED_CSS, [...sharedPaths, EXT_CONFIG])
  fill("admin-ui", admin, ADMIN_CSS, [ADMIN_CONFIG, ADMIN_CSS])
  const compare = (a, b) => a < b ? -1 : a > b ? 1 : 0
  definitionScopes.sort((a, b) => compare(a.path, b.path) || a.start - b.start || a.end - b.end)
  const uniqueDiagnostics = [...new Map(diagnostics.map(diagnostic => [JSON.stringify(diagnostic), diagnostic])).values()]
  uniqueDiagnostics.sort((a, b) => compare(a.path, b.path) || (a.line ?? 0) - (b.line ?? 0) ||
    (a.column ?? 0) - (b.column ?? 0) || compare(a.code, b.code) || compare(a.message, b.message))
  return { surfaces, definitionScopes, diagnostics: uniqueDiagnostics }
}
