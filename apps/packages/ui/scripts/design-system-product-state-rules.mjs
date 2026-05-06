import ts from "typescript"

export const PRODUCT_STATE_ANTD_NAMES = new Set([
  "Alert",
  "Badge",
  "Empty",
  "Result",
  "Spin",
  "Tag"
])

export const VALID_BASELINE_STATES = new Set([
  "allowed_legacy_exception",
  "active_migration_target"
])

export const CANONICAL_ROOTS = [
  "src/components/ui/primitives/Alert.tsx",
  "src/components/ui/primitives/Badge.tsx",
  "src/components/ui/feedback/EmptyState.tsx",
  "src/components/ui/feedback/LoadingState.tsx",
  "src/components/ui/layout/ModalFooter.tsx",
  "src/components/ui/state/ActionGroup.tsx",
  "src/components/ui/state/DiagnosticRow.tsx",
  "src/components/ui/state/PermissionNotice.tsx",
  "src/components/ui/state/RecoveryCallout.tsx",
  "src/components/ui/state/SetupRequiredPanel.tsx",
  "src/components/ui/state/StatePanel.tsx",
  "src/design-system/states.ts",
  "src/design-system/index.ts",
  "src/assets/tailwind.css",
  "src/assets/tailwind-shared.css"
]

const CANONICAL_STATE_LABELS = [
  "Unavailable",
  "Setup required",
  "Sign in required",
  "Permission denied",
  "Degraded",
  "Retrying",
  "Blocked",
  "Ready",
  "Loading"
]

const PRODUCT_STATE_WORDS = [
  "unavailable",
  "degraded",
  "retrying",
  "blocked",
  "setup",
  "sign in",
  "permission denied",
  "retry",
  "diagnostics",
  "reconnect",
  "disconnected",
  "loading",
  "failed"
]

const RECOVERY_COMPONENT_PATTERN =
  /(Error|Connection|Unavailable|Recovery|Offline|Readiness|Permission)Banner$/
const EMPTY_COMPONENT_PATTERN = /(EmptyState|Empty)$/
const LOADING_COMPONENT_PATTERN = /(LoadingState|Loading|Spinner)$/
const STATUS_COMPONENT_PATTERN = /(StatusBadge|StatusTag|StatusChip|StatusDot)$/

const PRODUCT_STATE_NAME_PATTERN =
  /(Status|Error|Empty|Loading|Recovery|Connection|Unavailable|Readiness|Permission|Offline)/
const SEVERITY_PROP_NAMES = new Set(["color", "status", "type", "severity"])
const SEVERITY_PROP_VALUES = new Set([
  "danger",
  "error",
  "info",
  "processing",
  "success",
  "warning"
])
const RECOVERY_ACTION_WORDS = [
  "copy diagnostics",
  "diagnostics",
  "open setup",
  "open settings",
  "reconnect",
  "reload",
  "retry",
  "switch server"
]

const RULE_REPLACEMENTS = {
  "antd-product-state-import": "tldw design-system state primitive",
  "local-recovery-banner": "RecoveryCallout or StatePanel",
  "local-empty-state": "EmptyState",
  "local-loading-state": "LoadingState",
  "local-status-badge": "Badge with design-system state mapping",
  "canonical-state-label": "design-system state registry"
}

export function createFindingId(rule, relativePath, subject) {
  return `${rule}:${normalizePath(relativePath)}:${subject}`
}

export function analyzeSource({ relativePath, source }) {
  const normalizedPath = normalizePath(relativePath)

  if (isExcludedPath(normalizedPath)) {
    return []
  }

  const sourceFile = ts.createSourceFile(
    normalizedPath,
    source,
    ts.ScriptTarget.Latest,
    true,
    normalizedPath.endsWith(".tsx") ? ts.ScriptKind.TSX : ts.ScriptKind.TS
  )

  const findings = []
  const localAntdNames = collectAntdProductStateImports(sourceFile)
  const componentNames = collectComponentNames(sourceFile)
  const fileSubject = subjectFromPath(normalizedPath)

  for (const subject of new Set([fileSubject, ...componentNames])) {
    pushLocalComponentFinding(findings, normalizedPath, subject)
  }

  pushCanonicalLabelFindings(findings, normalizedPath, sourceFile)
  pushAntdFindings(findings, normalizedPath, sourceFile, localAntdNames, {
    componentNames,
    fileSubject
  })

  return dedupeFindings(findings)
}

function isExcludedPath(relativePath) {
  if (CANONICAL_ROOTS.includes(relativePath)) {
    return true
  }

  return (
    /^src\/components\/ui(?:\/.*)?\/index\.ts$/.test(relativePath) ||
    /^src\/.*\.test\.tsx?$/.test(relativePath) ||
    /^src\/.*\/__tests__\/.*/.test(relativePath) ||
    relativePath.startsWith("src/assets/locale/") ||
    relativePath.startsWith("src/public/_locales/")
  )
}

function collectAntdProductStateImports(sourceFile) {
  const localNames = new Map()

  for (const statement of sourceFile.statements) {
    if (
      !ts.isImportDeclaration(statement) ||
      !ts.isStringLiteral(statement.moduleSpecifier) ||
      statement.moduleSpecifier.text !== "antd"
    ) {
      continue
    }

    const namedBindings = statement.importClause?.namedBindings
    if (!namedBindings || !ts.isNamedImports(namedBindings)) {
      continue
    }

    for (const element of namedBindings.elements) {
      const importedName = element.propertyName?.text ?? element.name.text
      if (PRODUCT_STATE_ANTD_NAMES.has(importedName)) {
        localNames.set(element.name.text, importedName)
      }
    }
  }

  return localNames
}

function collectComponentNames(sourceFile) {
  const names = new Set()

  walk(sourceFile, (node) => {
    if (ts.isFunctionDeclaration(node) && node.name) {
      names.add(node.name.text)
      return
    }

    if (ts.isClassDeclaration(node) && node.name) {
      names.add(node.name.text)
      return
    }

    if (!ts.isVariableDeclaration(node) || !ts.isIdentifier(node.name)) {
      return
    }

    if (
      node.initializer &&
      (ts.isArrowFunction(node.initializer) ||
        ts.isFunctionExpression(node.initializer))
    ) {
      names.add(node.name.text)
    }
  })

  return names
}

function pushLocalComponentFinding(findings, relativePath, subject) {
  if (!subject) {
    return
  }

  if (RECOVERY_COMPONENT_PATTERN.test(subject)) {
    pushFinding(findings, {
      relativePath,
      rule: "local-recovery-banner",
      subject,
      message: `${subject} duplicates recovery banner product-state UI.`
    })
  }

  if (EMPTY_COMPONENT_PATTERN.test(subject)) {
    pushFinding(findings, {
      relativePath,
      rule: "local-empty-state",
      subject,
      message: `${subject} should use the shared EmptyState primitive.`
    })
  }

  if (LOADING_COMPONENT_PATTERN.test(subject)) {
    pushFinding(findings, {
      relativePath,
      rule: "local-loading-state",
      subject,
      message: `${subject} should use the shared LoadingState primitive.`
    })
  }

  if (STATUS_COMPONENT_PATTERN.test(subject)) {
    pushFinding(findings, {
      relativePath,
      rule: "local-status-badge",
      subject,
      message: `${subject} should map status through the design system.`
    })
  }
}

function pushCanonicalLabelFindings(findings, relativePath, sourceFile) {
  const labels = new Map()

  walk(sourceFile, (node) => {
    if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isJsxText(node)
    ) {
      const label = canonicalLabelFromLiteral(node.getText(sourceFile), node.text)
      if (label && !labels.has(label)) {
        labels.set(label, node)
      }
    }
  })

  for (const [subject, node] of labels) {
    pushFinding(findings, {
      relativePath,
      rule: "canonical-state-label",
      subject,
      message: `"${subject}" should come from the design-system state registry.`,
      line: lineForNode(sourceFile, node)
    })
  }
}

function pushAntdFindings(
  findings,
  relativePath,
  sourceFile,
  localAntdNames,
  context
) {
  if (localAntdNames.size === 0) {
    return
  }

  walk(sourceFile, (node) => {
    if (!ts.isJsxSelfClosingElement(node) && !ts.isJsxOpeningElement(node)) {
      return
    }

    const localName = jsxTagName(node.tagName)
    const importedName = localName ? localAntdNames.get(localName) : undefined
    if (!importedName) {
      return
    }

    const useContext = collectJsxUseContext(node, sourceFile)
    if (!isProductStateAntdUse(importedName, useContext, context)) {
      return
    }

    pushFinding(findings, {
      relativePath,
      rule: "antd-product-state-import",
      subject: importedName,
      message: `${importedName} from AntD is rendering product-state UI directly.`,
      line: lineForNode(sourceFile, node)
    })
  })
}

function isProductStateAntdUse(importedName, useContext, context) {
  if (
    useContext.hasProductStateText ||
    useContext.hasCanonicalLabel ||
    useContext.hasSeverityProp ||
    useContext.hasRecoveryAction
  ) {
    return true
  }

  if (importedName === "Tag") {
    return false
  }

  return [context.fileSubject, ...context.componentNames].some((name) =>
    PRODUCT_STATE_NAME_PATTERN.test(name)
  )
}

function collectJsxUseContext(node, sourceFile) {
  const texts = []
  const attributes = getJsxAttributes(node)

  for (const attribute of attributes) {
    const value = jsxAttributeValue(attribute, sourceFile)
    if (value) {
      texts.push(value)
    }
  }

  if (ts.isJsxOpeningElement(node)) {
    const parent = node.parent
    if (ts.isJsxElement(parent)) {
      texts.push(...jsxChildTexts(parent, sourceFile))
    }
  }

  const normalizedTexts = texts.map((text) => normalizeTextSignal(text))
  const hasCanonicalLabel = texts.some((text) =>
    CANONICAL_STATE_LABELS.includes(text.trim())
  )
  const hasProductStateText = normalizedTexts.some((text) =>
    PRODUCT_STATE_WORDS.some((word) => text.includes(word))
  )
  const hasRecoveryAction = normalizedTexts.some((text) =>
    RECOVERY_ACTION_WORDS.some((word) => text.includes(word))
  )
  const hasSeverityProp = attributes.some((attribute) => {
    const propName = attribute.name.text
    const propValue = jsxAttributeValue(attribute, sourceFile)

    return (
      SEVERITY_PROP_NAMES.has(propName) &&
      typeof propValue === "string" &&
      SEVERITY_PROP_VALUES.has(propValue.toLowerCase())
    )
  })

  return {
    hasCanonicalLabel,
    hasProductStateText,
    hasRecoveryAction,
    hasSeverityProp
  }
}

function getJsxAttributes(node) {
  if (!("attributes" in node)) {
    return []
  }

  return node.attributes.properties.filter(ts.isJsxAttribute)
}

function jsxAttributeValue(attribute, sourceFile) {
  if (!attribute.initializer) {
    return undefined
  }

  if (ts.isStringLiteral(attribute.initializer)) {
    return attribute.initializer.text
  }

  if (
    ts.isJsxExpression(attribute.initializer) &&
    attribute.initializer.expression
  ) {
    const expression = attribute.initializer.expression
    if (ts.isStringLiteral(expression) || ts.isNoSubstitutionTemplateLiteral(expression)) {
      return expression.text
    }

    return expression.getText(sourceFile)
  }

  return undefined
}

function jsxChildTexts(node, sourceFile) {
  const texts = []

  for (const child of node.children) {
    if (ts.isJsxText(child)) {
      texts.push(child.text)
      continue
    }

    if (
      ts.isJsxExpression(child) &&
      child.expression &&
      (ts.isStringLiteral(child.expression) ||
        ts.isNoSubstitutionTemplateLiteral(child.expression))
    ) {
      texts.push(child.expression.text)
      continue
    }

    if (ts.isJsxElement(child)) {
      texts.push(...jsxChildTexts(child, sourceFile))
    }
  }

  return texts
}

function jsxTagName(tagName) {
  return ts.isIdentifier(tagName) ? tagName.text : undefined
}

function canonicalLabelFromLiteral(rawText, valueText) {
  const text = (valueText ?? rawText).trim()
  return CANONICAL_STATE_LABELS.find((label) => text === label)
}

function pushFinding(findings, { relativePath, rule, subject, message, line }) {
  findings.push({
    id: createFindingId(rule, relativePath, subject),
    path: relativePath,
    rule,
    subject,
    message,
    ...(typeof line === "number" ? { line } : {}),
    replacement: RULE_REPLACEMENTS[rule]
  })
}

function dedupeFindings(findings) {
  const seen = new Set()
  const deduped = []

  for (const finding of findings) {
    if (seen.has(finding.id)) {
      continue
    }

    seen.add(finding.id)
    deduped.push(finding)
  }

  return deduped
}

function lineForNode(sourceFile, node) {
  return sourceFile.getLineAndCharacterOfPosition(node.getStart(sourceFile)).line + 1
}

function normalizeTextSignal(text) {
  return text.toLowerCase().replace(/\s+/g, " ").trim()
}

function subjectFromPath(relativePath) {
  const filename = relativePath.split("/").pop() ?? ""
  return filename.replace(/\.[^.]+$/, "")
}

function normalizePath(path) {
  return path.replaceAll("\\", "/")
}

function walk(node, visitor) {
  visitor(node)
  ts.forEachChild(node, (child) => walk(child, visitor))
}
