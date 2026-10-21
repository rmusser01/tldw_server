import ts from "typescript"
import postcss from "postcss"
import colors from "tailwindcss/colors.js"
import { JSDOM } from "jsdom"

/** @typedef {"shared-ui" | "webui" | "extension" | "admin-ui"} Surface */
/** @typedef {{path:string, code:string, message:string, severity:"error"|"coverage", line?:number, column?:number}} Diagnostic */
/** @typedef {{path:string, start:number, end:number, owner:string, kind:string}} Definition */
/** @typedef {{path:string, surface:Surface, line:number, column:number, rule:"literal-color"|"palette-color"|"custom-radius"|"layout-value", value:string, owner:string|null, classification:"token-review"|"radius-review"|"layout-review", reason:string, suggestion?:string}} Finding */
/** @typedef {{findings:Finding[], definitions:Definition[], diagnostics:Diagnostic[]}} Analysis */
/** @typedef {{surfaces:Record<Surface, {semanticColors:Set<string>, cssVariables:Set<string>, radiusClasses:Set<string>, radiusValues:Set<string>, definitionPaths:string[]}>, definitionScopes:Definition[], diagnostics:Diagnostic[]}} Authority */

const colorProbe = new JSDOM("").window.document.createElement("span").style
const namedColors = new Map()
const allowed = new Set([
  "transparent", "currentcolor", "inherit", "initial", "unset", "revert", "revert-layer",
  "accentcolor", "accentcolortext", "activetext", "buttonborder", "buttonface", "buttontext",
  "canvas", "canvastext", "field", "fieldtext", "graytext", "highlight", "highlighttext",
  "linktext", "mark", "marktext", "selecteditem", "selecteditemtext", "visitedtext",
  // Deprecated system colors still carry platform semantics in legacy stylesheets.
  "activeborder", "activecaption", "appworkspace", "background", "buttonhighlight", "buttonshadow",
  "captiontext", "inactiveborder", "inactivecaption", "inactivecaptiontext", "infobackground", "infotext",
  "menu", "menutext", "scrollbar", "threeddarkshadow", "threedface", "threedhighlight",
  "threedlightshadow", "threedshadow", "window", "windowframe", "windowtext"
])
const colorFunctions = new Set(["rgb", "rgba", "hsl", "hsla", "hwb", "lab", "lch", "oklab", "oklch", "color", "color-mix", "light-dark"])
const compositions = new Set(["cn", "clsx", "classNames", "cva"])
const layoutProperty = /^(?:(?:min-|max-)?(?:width|height|inline-size|block-size)|(?:margin|padding|inset)(?:-(?:top|right|bottom|left|inline|block)(?:-(?:start|end))?)?|(?:row-|column-)?gap|top|right|bottom|left|z-index|grid-template(?:-columns|-rows|-areas)?)$/
const radiusProperty = /^border-(?:(?:top-left|top-right|bottom-left|bottom-right|start-start|start-end|end-start|end-end)-)?radius$/
const colorProperty = /^(?:color|background(?:-color|-image)?|(?:border(?:-(?:top|right|bottom|left|inline|block)(?:-(?:start|end))?)?|outline|text-decoration|column-rule)(?:-color)?|(?:box|text)-shadow|fill|stroke|(?:caret|accent|stop|flood|lighting)-color|text-emphasis(?:-color)?)$/
const colorUtility = /^(text|bg|border(?:-[trblxyse])?|ring(?:-offset)?|outline|divide(?:-[xy])?|placeholder|decoration|fill|stroke|shadow|from|via|to)-(.+)$/
const layoutUtility = /^(?:-)?(?:[mp][trblxyse]?|space-[xy]|gap(?:-[xy])?|(?:min-|max-)?[wh]|size|inset(?:-[xy])?|top|right|bottom|left|start|end|z|grid-cols|grid-rows|auto-cols|auto-rows)-\[/

/** Split only outside brackets, parentheses and quotes, preserving source offsets. */
function split(value, separators = /\s/) {
  const parts = []
  let start = 0
  let depth = 0
  let quote = null
  for (let i = 0; i <= value.length; i++) {
    const char = value[i]
    if (char === "\\") { i++; continue }
    if (quote) { if (char === quote) quote = null; continue }
    if (char === '"' || char === "'") { quote = char; continue }
    if (char === "[" || char === "(") depth++
    if (char === "]" || char === ")") depth--
    if (i === value.length || (depth === 0 && separators.test(char))) {
      if (i > start) parts.push({ value: value.slice(start, i), start, end: i })
      start = i + 1
    }
  }
  return parts
}

function isNamedColor(value) {
  if (!namedColors.has(value)) {
    colorProbe.color = ""
    colorProbe.color = value
    namedColors.set(value, colorProbe.color !== "" && !allowed.has(value))
  }
  return namedColors.get(value)
}

function kebab(value) {
  return value.replace(/[A-Z]/g, char => `-${char.toLowerCase()}`).replace(/^ms-/, "-ms-")
}

function normalizeRadius(value) {
  return value.trim().toLowerCase().replace(/^(-?(?:\d+\.?\d*|\.\d+))([a-z%]*)$/, (_, number, unit) => Number(number) === 0 ? "0" : `${Number(number)}${unit}`)
}

function maskCssComments(value) {
  return value.replace(/"(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\/\*[\s\S]*?\*\//g, token => token.startsWith("/*") ? token.replace(/[^\r\n]/g, " ") : token)
}

/** Analyze styling syntax only; analyzed application code is never evaluated.
 * @param {{path:string, source:string, authority:Authority}} input
 * @returns {Analysis}
 */
export function analyzeDriftSource({ path, source, authority }) {
  const surface = path.startsWith("apps/tldw-frontend/") ? "webui" : path.startsWith("apps/extension/") ? "extension" : path.startsWith("admin-ui/") ? "admin-ui" : "shared-ui"
  const tokens = authority.surfaces[surface]
  const definitions = authority.definitionScopes.filter(scope => scope.path === path)
  /** @type {Analysis} */
  const result = { findings: [], definitions: [...definitions].sort((a, b) => a.start - b.start || a.end - b.end), diagnostics: [] }
  const seen = new Set()
  const lineStarts = [0]
  for (let i = 0; i < source.length; i++) if (source[i] === "\n") lineStarts.push(i + 1)
  function location(offset) {
    let low = 0
    let high = lineStarts.length
    while (low + 1 < high) {
      const mid = (low + high) >>> 1
      if (lineStarts[mid] <= offset) low = mid
      else high = mid
    }
    return { line: low + 1, column: offset - lineStarts[low] + 1 }
  }
  function diagnostic(offset, code, message, severity = "coverage") {
    const key = `${code}:${offset}:${message}`
    if (seen.has(key)) return
    seen.add(key)
    result.diagnostics.push({ path, code, message, severity, ...location(offset) })
  }
  function finding(rule, value, offset, owner) {
    if (definitions.some(scope => offset >= scope.start && offset < scope.end)) return
    const key = `${rule}:${offset}:${value}`
    if (seen.has(key)) return
    seen.add(key)
    const classification = rule === "layout-value" ? "layout-review" : rule === "custom-radius" ? "radius-review" : "token-review"
    const reason = rule === "layout-value" ? "Explicit layout geometry needs contextual review, not automatic replacement." : rule === "custom-radius" ? "Radius is outside the surface's documented scale." : rule === "palette-color" ? "Direct palette color bypasses semantic color tokens." : "Literal color needs semantic token review."
    result.findings.push({ path, surface, ...location(offset), rule, value, owner, classification, reason })
  }
  const radii = new Set([...tokens.radiusValues].map(normalizeRadius))
  const positionAt = offset => typeof offset === "function" ? offset : index => offset + index

  function variables(value, offset) {
    const at = positionAt(offset)
    for (const match of value.matchAll(/var\(\s*(--[\w-]+)/g)) {
      if (!tokens.cssVariables.has(match[1])) diagnostic(at(match.index), "unresolved-variable", "CSS variable is not in this surface's token authority.")
    }
  }

  // Walk CSS values without looking inside strings, comments, or URL payloads.
  function colorsIn(value, offset, owner, utility, named = true) {
    const at = positionAt(offset)
    const emit = (color, start) => finding("literal-color", utility ?? color, at(utility ? 0 : start), owner)
    for (let i = 0; i < value.length;) {
      const char = value[i]
      if (char === "\\") { i += 2; continue }
      if (char === '"' || char === "'") {
        const quote = char
        for (i++; i < value.length; i++) {
          if (value[i] === "\\") i++
          else if (value[i] === quote) { i++; break }
        }
        continue
      }
      if (value.startsWith("/*", i)) {
        const end = value.indexOf("*/", i + 2)
        i = end < 0 ? value.length : end + 2
        continue
      }
      const hex = value.slice(i).match(/^#[\da-f]{3,8}(?![\w-])/i)
      if (hex && [4, 5, 7, 9].includes(hex[0].length)) { emit(hex[0], i); i += hex[0].length; continue }
      const identifier = value.slice(i).match(/^-?[_a-zA-Z][\w-]*/)
      if (!identifier) { i++; continue }
      const word = identifier[0]
      const start = i
      i += word.length
      if (value[i] === "(") {
        let depth = 1
        let quote = null
        let end = i + 1
        for (; end < value.length && depth; end++) {
          if (value[end] === "\\") { end++; continue }
          if (quote) { if (value[end] === quote) quote = null; continue }
          if (value[end] === '"' || value[end] === "'") { quote = value[end]; continue }
          if (value[end] === "(") depth++
          if (value[end] === ")") depth--
        }
        const name = word.toLowerCase()
        const body = value.slice(i + 1, end - 1)
        if (name === "url") { i = end; continue }
        if (name === "var") {
          const args = split(body, /,/)
          const variable = args[0]?.value.trim()
          if (named && !tokens.cssVariables.has(variable)) diagnostic(at(start), "unresolved-variable", "CSS variable is not in this surface's token authority.")
          const comma = args[1]?.start
          if (comma !== undefined) colorsIn(body.slice(comma), index => at(utility ? 0 : i + 1 + comma + index), owner, utility, named)
          i = end
          continue
        }
        let channels = /^(?:rgba?|hsla?|hwb|lab|lch|oklab|oklch|color)$/.test(name) ? split(body, /\//)[0]?.value ?? "" : body
        if (/^(?:rgba?|hsla?)$/.test(name)) channels = split(channels, /,/).slice(0, 3).map(part => part.value).join(",")
        if (colorFunctions.has(name) && !/\bvar\s*\(/.test(channels) && name !== "light-dark") {
          emit(value.slice(start, end), start)
          i = end
          continue
        }
        // Other functions (gradients, semantic hsl/rgb wrappers) expose their arguments.
        continue
      }
      if (named && !allowed.has(word.toLowerCase()) && isNamedColor(word.toLowerCase())) emit(word, start)
    }
  }

  function radius(value, offset, owner, utility) {
    const at = positionAt(offset)
    for (const part of split(value, /[\s/]/)) {
      const normalized = normalizeRadius(part.value)
      if (radii.has(normalized) || allowed.has(normalized)) continue
      const variable = part.value.match(/^var\(\s*(--[\w-]+)\s*\)$/)
      if (variable && tokens.cssVariables.has(variable[1])) continue
      const literal = normalized.match(/^-?(?:\d*\.)?\d+([a-z%]*)$/i)
      const differentUnits = literal && [...radii].some(scale => {
        const entry = scale.match(/^-?(?:\d*\.)?\d+([a-z%]+)$/i)
        return entry && entry[1] !== literal[1]
      })
      if (literal && !differentUnits) finding("custom-radius", utility ?? part.value, at(utility ? 0 : part.start), owner)
      else diagnostic(at(part.start), "unresolved-radius", "Radius cannot be compared statically with the surface scale.")
    }
  }

  function declaration(property, value, offset, owner, utility) {
    property = kebab(property)
    if (property.startsWith("--")) colorsIn(value, offset, owner, utility, false)
    else if (colorProperty.test(property)) colorsIn(value, offset, owner, utility)
    else if (radiusProperty.test(property)) radius(value, offset, owner, utility)
    else if (layoutProperty.test(property)) {
      variables(value, offset)
      if (!allowed.has(value.toLowerCase()) && !/^var\(\s*--[\w-]+\s*\)$/.test(value)) finding("layout-value", utility ?? value, positionAt(offset)(0), owner)
    }
  }

  function classes(value, offsetAt, owner, fragment = {}) {
    for (const part of split(value)) {
      if ((fragment.left && part.start === 0) || (fragment.right && part.end === value.length)) continue
      const raw = part.value
      const offset = offsetAt(part.start)
      const variants = split(raw, /:/)
      const base = variants.at(-1)?.value.replace(/^!|!$/g, "") ?? ""
      if (layoutUtility.test(base)) {
        const value = base.slice(base.indexOf("[") + 1, base.lastIndexOf("]")).replace(/_/g, " ")
        variables(value, () => offset)
        if (!/^var\(\s*--[\w-]+\s*\)$/.test(value)) finding("layout-value", raw, offset, owner)
        continue
      }
      if (/^rounded(?:-[trblse]{1,2})?-\[/.test(base)) {
        radius(base.slice(base.indexOf("[") + 1, base.lastIndexOf("]")).replace(/_/g, " "), offset, owner, raw)
        continue
      }
      if (/^rounded(?:-|$)/.test(base)) {
        const canonical = base.replace(/^rounded-(?:tl|tr|bl|br|ss|se|es|ee|t|r|b|l|s|e)(?=-|$)/, "rounded")
        if (!tokens.radiusClasses.has(canonical)) diagnostic(offset, "unresolved-radius-class", "Radius utility is not in the surface scale.")
        continue
      }
      if (base.startsWith("[") && base.endsWith("]")) {
        const body = base.slice(1, -1)
        const colon = body.indexOf(":")
        if (colon > 0) declaration(body.slice(0, colon), body.slice(colon + 1).replace(/_/g, " "), offset, owner, raw)
        continue
      }
      const match = base.match(colorUtility)
      if (!match) continue
      const color = split(match[2], /\//)[0]?.value ?? ""
      if (tokens.semanticColors.has(color) || allowed.has(color.toLowerCase())) continue
      if (/^\[.*\]$/.test(color)) {
        let body = color.slice(1, -1).replace(/_/g, " ")
        const hint = body.match(/^([\w-]+):/)
        if (hint && !["color", "length", "image", "position", "size", "number", "percentage"].includes(hint[1])) continue
        if (hint && hint[1] !== "color") continue
        if (hint) body = body.slice(hint[0].length)
        colorsIn(body, offset, owner, raw)
      } else if (color === "black" || color === "white" || /^([a-z]+)-(\d{2,3})$/.test(color) && Object.hasOwn(colors, color.replace(/-\d+$/, ""))) {
        finding("palette-color", raw, offset, owner)
      }
    }
  }

  if (path.endsWith(".css")) {
    try {
      const root = postcss.parse(source, { from: path })
      const cssOwner = node => {
        for (let parent = node.parent; parent; parent = parent.parent) if (parent.type === "rule") return parent.selector
        return null
      }
      root.walkDecls(node => {
        const start = node.source.start.offset + node.prop.length + node.raws.between.length
        declaration(node.prop, maskCssComments(node.raws.value?.raw ?? node.value), start, cssOwner(node))
      })
      root.walkAtRules("apply", node => {
        const start = node.source.start.offset + 1 + node.name.length + (node.raws.afterName?.length ?? 0)
        classes(maskCssComments(node.raws.params?.raw ?? node.params), index => start + index, cssOwner(node))
      })
    } catch (error) {
      result.diagnostics.push({ path, code: "parse-error", message: error.reason ?? "CSS parse failed.", severity: "error", line: error.line ?? 1, column: error.column ?? 1 })
    }
  } else {
    const scriptKind = /\.(?:tsx|jsx|js)$/.test(path) ? ts.ScriptKind.TSX : ts.ScriptKind.TS
    const ast = ts.createSourceFile(path, source, ts.ScriptTarget.Latest, true, scriptKind)
    for (const error of ast.parseDiagnostics) diagnostic(error.start ?? 0, "parse-error", ts.flattenDiagnosticMessageText(error.messageText, " "), "error")
    if (ast.parseDiagnostics.length) return result
    const mutated = new Set()

    function owner(node) {
      for (let parent = node.parent; parent; parent = parent.parent) {
        if ((ts.isFunctionLike(parent) || ts.isClassDeclaration(parent) || ts.isVariableDeclaration(parent) || ts.isPropertyAssignment(parent) && ts.isFunctionLike(unwrap(parent.initializer))) && parent.name && !ts.isComputedPropertyName(parent.name)) return parent.name.text ?? null
      }
      return null
    }
    function unwrap(node) {
      while (node && (ts.isParenthesizedExpression(node) || ts.isAsExpression(node) || ts.isTypeAssertionExpression(node) || ts.isSatisfiesExpression(node) || ts.isNonNullExpression(node))) node = node.expression
      return node
    }
    // Lexical lookup avoids confusing an API field with a class map or a shadowed prop.
    function lookup(identifier) {
      const name = identifier.text
      for (let scope = identifier.parent; scope; scope = scope.parent) {
        if (ts.isFunctionLike(scope) && scope.parameters.some(param => boundName(param.name, name))) return null
        const statements = ts.isSourceFile(scope) || ts.isBlock(scope) ? scope.statements : null
        if (!statements) continue
        for (const statement of statements) {
          if (!ts.isVariableStatement(statement)) continue
          for (const decl of statement.declarationList.declarations) {
            if (boundName(decl.name, name)) return ts.isIdentifier(decl.name) && (statement.declarationList.flags & ts.NodeFlags.Const) ? decl.initializer : null
          }
        }
      }
      return null
    }
    function boundName(node, name) {
      return ts.isIdentifier(node) ? node.text === name : node.elements.some(item => !ts.isOmittedExpression(item) && boundName(item.name, name))
    }
    function resolve(node, visited) {
      node = unwrap(node)
      if (!node || visited.has(node)) return null
      if (ts.isIdentifier(node)) {
        const next = lookup(node)
        if (!next || mutated.has(unwrap(next))) return null
        return resolve(next, new Set([...visited, node]))
      }
      if (ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) {
        const object = resolve(node.expression, new Set([...visited, node]))
        const key = ts.isPropertyAccessExpression(node) ? node.name.text : node.argumentExpression && (ts.isStringLiteralLike(node.argumentExpression) || ts.isNumericLiteral(node.argumentExpression)) ? node.argumentExpression.text : null
        if (!object || !ts.isObjectLiteralExpression(object) || key === null) return null
        for (const property of [...object.properties].reverse()) {
          if (ts.isSpreadAssignment(property) || property.name && ts.isComputedPropertyName(property.name)) return null
          if (ts.isPropertyAssignment(property) && property.name.getText(ast).replace(/^['"]|['"]$/g, "") === key) return resolve(property.initializer, new Set([...visited, node]))
        }
        return null
      }
      return node
    }
    function collectMutations(node) {
      let target = ts.isBinaryExpression(node) && node.operatorToken.kind >= ts.SyntaxKind.FirstAssignment && node.operatorToken.kind <= ts.SyntaxKind.LastAssignment ? node.left : ts.isDeleteExpression(node) ? node.expression : (ts.isPrefixUnaryExpression(node) || ts.isPostfixUnaryExpression(node)) && [ts.SyntaxKind.PlusPlusToken, ts.SyntaxKind.MinusMinusToken].includes(node.operator) ? node.operand : null
      while (target && (ts.isPropertyAccessExpression(target) || ts.isElementAccessExpression(target))) target = target.expression
      if (target && ts.isIdentifier(target)) {
        const resolved = resolve(target, new Set())
        if (resolved) mutated.add(resolved)
      }
      ts.forEachChild(node, collectMutations)
    }
    collectMutations(ast)

    // Map cooked string indices back through escapes to the original source.
    function stringContent(node) {
      const start = node.getStart(ast)
      const raw = source.slice(start + 1, node.end - (ts.isTemplateHead(node) || ts.isTemplateMiddle(node) ? 2 : 1))
      let value = ""
      const offsets = []
      for (let i = 0; i < raw.length; i++) {
        const original = i
        let decoded = raw[i]
        if (raw[i] === "\\") {
          const escape = raw[++i]
          if (escape === "\n") continue
          if (escape === "\r") { if (raw[i + 1] === "\n") i++; continue }
          const unicode = escape === "u" ? raw.slice(i + 1).match(/^\{([\da-f]+)\}|^([\da-f]{4})/i) : null
          const hex = escape === "x" ? raw.slice(i + 1).match(/^[\da-f]{2}/i) : null
          if (unicode) { decoded = String.fromCodePoint(parseInt(unicode[1] ?? unicode[2], 16)); i += unicode[0].length }
          else if (hex) { decoded = String.fromCharCode(parseInt(hex[0], 16)); i += 2 }
          else decoded = ({n: "\n", r: "\r", t: "\t", b: "\b", f: "\f", v: "\v", 0: "\0"})[escape] ?? escape
        }
        value += decoded
        for (let j = 0; j < decoded.length; j++) offsets.push(start + 1 + original)
      }
      return { value, offsetAt: index => offsets[index] ?? node.end - 1 }
    }

    function unresolved(node, context) {
      diagnostic(node.getStart(ast), "dynamic-styling", `Unresolved ${context} expression; runtime styling is outside static coverage.`)
    }
    function classExpression(node, visited = new Set(), mode = "classes") {
      node = unwrap(node)
      if (!node) return
      if (visited.has(node)) { unresolved(node, "class"); return }
      const nextVisited = new Set([...visited, node])
      if (ts.isStringLiteralLike(node)) {
        const text = stringContent(node)
        classes(text.value, text.offsetAt, owner(node))
      } else if (ts.isIdentifier(node) || ts.isPropertyAccessExpression(node) || ts.isElementAccessExpression(node)) {
        if (ts.isIdentifier(node) && node.text === "undefined") return
        const resolved = resolve(node, visited)
        if (resolved && resolved !== node) classExpression(resolved, nextVisited, mode)
        else unresolved(node, "class")
      } else if (ts.isCallExpression(node)) {
        const name = ts.isIdentifier(node.expression) ? node.expression.text : ""
        if (compositions.has(name)) {
          if (name === "cva") {
            if (node.arguments[0]) classExpression(node.arguments[0], nextVisited)
            if (node.arguments[1]) classExpression(node.arguments[1], nextVisited, "options")
          } else for (const arg of node.arguments) classExpression(arg, nextVisited)
        } else {
          const factory = ts.isIdentifier(node.expression) && resolve(node.expression, visited)
          if (factory && ts.isCallExpression(factory) && ts.isIdentifier(factory.expression) && factory.expression.text === "cva") classExpression(factory, nextVisited)
          else unresolved(node, "class")
        }
      } else if (ts.isConditionalExpression(node)) {
        classExpression(node.whenTrue, nextVisited, mode)
        classExpression(node.whenFalse, nextVisited, mode)
      } else if (ts.isBinaryExpression(node) && [ts.SyntaxKind.AmpersandAmpersandToken, ts.SyntaxKind.BarBarToken, ts.SyntaxKind.QuestionQuestionToken].includes(node.operatorToken.kind)) {
        if (node.operatorToken.kind !== ts.SyntaxKind.AmpersandAmpersandToken) classExpression(node.left, nextVisited, mode)
        classExpression(node.right, nextVisited, mode)
      } else if (ts.isArrayLiteralExpression(node)) {
        for (const element of node.elements) classExpression(element, nextVisited, mode)
      } else if (ts.isObjectLiteralExpression(node)) {
        for (const prop of node.properties) {
          if (ts.isSpreadAssignment(prop)) { classExpression(prop.expression, nextVisited, mode); continue }
          if (!ts.isPropertyAssignment(prop) && !ts.isShorthandPropertyAssignment(prop)) { unresolved(prop, "class map"); continue }
          const value = ts.isPropertyAssignment(prop) ? prop.initializer : prop.name
          if (mode === "options") {
            if (prop.name.text === "variants") classExpression(value, nextVisited, "variants")
            else if (prop.name.text === "compoundVariants") classExpression(value, nextVisited, "compound")
            else if (ts.isComputedPropertyName(prop.name)) unresolved(prop.name, "class options key")
          } else if (mode === "variants") classExpression(value, nextVisited, "variant-values")
          else if (mode === "variant-values") classExpression(value, nextVisited)
          else if (mode === "compound") {
            if (["class", "className"].includes(prop.name.text)) classExpression(value, nextVisited)
            else if (ts.isComputedPropertyName(prop.name)) unresolved(prop.name, "compound class key")
          } else if (ts.isComputedPropertyName(prop.name)) classExpression(prop.name.expression, nextVisited)
          else {
            const key = prop.name
            if (ts.isStringLiteralLike(key)) classExpression(key, nextVisited)
            else classes(key.text, i => key.getStart(ast) + i, owner(key))
          }
        }
      } else if (ts.isTemplateExpression(node)) {
        const fragments = [node.head, ...node.templateSpans.map(span => span.literal)]
        for (let i = 0; i < fragments.length; i++) {
          const text = stringContent(fragments[i])
          classes(text.value, text.offsetAt, owner(node), {left: i > 0, right: i < fragments.length - 1})
        }
        let partial = false
        for (let i = 0; i < node.templateSpans.length; i++) {
          const before = stringContent(fragments[i]).value
          const after = stringContent(fragments[i + 1]).value
          if ((i === 0 && !before || /\s$/.test(before)) && (i === node.templateSpans.length - 1 && !after || /^\s/.test(after))) classExpression(node.templateSpans[i].expression, nextVisited)
          else partial = true
        }
        if (partial) unresolved(node, "class template")
      } else if (![ts.SyntaxKind.FalseKeyword, ts.SyntaxKind.TrueKeyword, ts.SyntaxKind.NullKeyword, ts.SyntaxKind.NumericLiteral].includes(node.kind)) unresolved(node, "class")
    }

    function styleValue(node, property, visited) {
      const value = resolve(node, visited)
      if (value && ts.isConditionalExpression(value)) {
        const next = new Set([...visited, value])
        styleValue(value.whenTrue, property, next)
        styleValue(value.whenFalse, property, next)
      } else if (value && ts.isStringLiteralLike(value)) {
        const text = stringContent(value)
        declaration(property, text.value, text.offsetAt, owner(value))
      } else if (value && (ts.isNumericLiteral(value) || ts.isPrefixUnaryExpression(value) && ts.isNumericLiteral(value.operand))) {
        const number = value.getText(ast)
        declaration(property, property === "z-index" || Number(number) === 0 ? number : `${number}px`, value.getStart(ast), owner(value))
      } else unresolved(node, "style value")
    }

    function styleExpression(node, visited = new Set()) {
      node = unwrap(node)
      if (!node) return
      const resolved = resolve(node, visited)
      if (!resolved || visited.has(resolved)) { unresolved(node, "style"); return }
      const nextVisited = new Set([...visited, resolved])
      if (ts.isConditionalExpression(resolved)) {
        styleExpression(resolved.whenTrue, nextVisited)
        styleExpression(resolved.whenFalse, nextVisited)
        return
      }
      if (!ts.isObjectLiteralExpression(resolved)) { unresolved(resolved, "style"); return }
      for (const prop of resolved.properties) {
        if (ts.isSpreadAssignment(prop)) { styleExpression(prop.expression, nextVisited); continue }
        if (!ts.isPropertyAssignment(prop) || ts.isComputedPropertyName(prop.name)) { unresolved(prop, "style property"); continue }
        const property = kebab(prop.name.text)
        if (!property.startsWith("--") && !colorProperty.test(property) && !radiusProperty.test(property) && !layoutProperty.test(property)) continue
        styleValue(prop.initializer, property, nextVisited)
      }
    }
    function visit(node) {
      if (ts.isJsxAttribute(node)) {
        const name = node.name.getText(ast)
        if (name === "className" || name === "class") {
          if (node.initializer && ts.isJsxExpression(node.initializer)) classExpression(node.initializer.expression)
          else if (node.initializer) classExpression(node.initializer)
        } else if (name === "style" && node.initializer) {
          if (ts.isJsxExpression(node.initializer)) styleExpression(node.initializer.expression)
          else unresolved(node.initializer, "style")
        }
      } else if (ts.isCallExpression(node) && ts.isIdentifier(node.expression) && compositions.has(node.expression.text)) classExpression(node)
      ts.forEachChild(node, visit)
    }
    visit(ast)
  }
  const compare = (a, b) => a.line - b.line || a.column - b.column || (a.rule < b.rule ? -1 : a.rule > b.rule ? 1 : 0)
  result.findings.sort(compare)
  result.diagnostics.sort((a, b) => (a.line ?? 0) - (b.line ?? 0) || (a.column ?? 0) - (b.column ?? 0) || (a.code < b.code ? -1 : a.code > b.code ? 1 : 0))
  return result
}
