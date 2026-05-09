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
  /(Error|Connection|Unavailable|Recovery|Offline|Readiness|Permission).*Banner$/
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
const TEXT_BEARING_PROP_NAMES = new Set([
  "action",
  "content",
  "description",
  "emptyText",
  "label",
  "message",
  "placeholder",
  "subTitle",
  "text",
  "title"
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
const DUPLICATE_FINDING_ID_SUFFIX_PATTERN = /#[-a-z0-9]+$/

const RULE_REPLACEMENTS = {
  "antd-product-state-import": "tldw design-system state primitive",
  "local-recovery-banner": "RecoveryCallout or StatePanel",
  "local-empty-state": "EmptyState",
  "local-loading-state": "LoadingState",
  "local-status-badge": "Badge with design-system state registry mapping",
  "canonical-state-label": "design-system state registry"
}

const REQUIRED_BASELINE_FIELDS = [
  "id",
  "path",
  "rule",
  "subject",
  "state",
  "owner",
  "reason",
  "replacement",
  "migrationQueue"
]

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
  const fileSubject = subjectFromPath(normalizedPath)
  const canonicalUsage = collectCanonicalDesignSystemUsage(
    sourceFile,
    fileSubject
  )
  const localComponentSubjects = normalizedPath.endsWith(".tsx")
    ? collectLocalComponentSubjects(sourceFile, fileSubject)
    : []

  for (const subject of localComponentSubjects) {
    pushLocalComponentFinding(findings, normalizedPath, subject, canonicalUsage)
  }

  pushCanonicalLabelFindings(findings, normalizedPath, sourceFile)
  pushAntdFindings(findings, normalizedPath, sourceFile, localAntdNames, {
    fileSubject
  })

  return disambiguateFindingIds(findings)
}

export function validateBaseline(baseline) {
  if (!Array.isArray(baseline)) {
    return ["baseline must be a JSON array"]
  }

  const errors = []
  const seenIds = new Set()

  baseline.forEach((entry, index) => {
    const prefix = `baseline[${index}]`

    if (!isRecord(entry)) {
      errors.push(`${prefix} must be an object`)
      return
    }

    for (const field of REQUIRED_BASELINE_FIELDS) {
      if (!isPresentString(entry[field])) {
        errors.push(`${prefix} ${field} is required`)
      }
    }

    if (isPresentString(entry.id)) {
      if (seenIds.has(entry.id)) {
        errors.push(`${prefix} duplicate baseline id ${entry.id}`)
      } else {
        seenIds.add(entry.id)
      }
    }

    if (
      isPresentString(entry.state) &&
      !VALID_BASELINE_STATES.has(entry.state)
    ) {
      errors.push(
        `${prefix} state must be allowed_legacy_exception or active_migration_target`
      )
    }

    if (
      isPresentString(entry.id) &&
      isPresentString(entry.rule) &&
      isPresentString(entry.path) &&
      isPresentString(entry.subject)
    ) {
      const expectedId = createFindingId(entry.rule, entry.path, entry.subject)
      if (
        entry.id !== expectedId &&
        !isDuplicateFindingId(entry.id, expectedId)
      ) {
        errors.push(
          `${prefix} id must match rule/path/subject: expected ${expectedId}`
        )
      }
    }
  })

  return errors
}

export function applyBaseline({ findings, baseline }) {
  const liveFindings = Array.isArray(findings) ? findings : []
  const baselineErrors = validateBaseline(baseline)
  const result = {
    blocked: [],
    activeMigrationTargets: [],
    allowedLegacy: [],
    staleBaseline: [],
    baselineErrors
  }

  if (baselineErrors.length > 0) {
    result.blocked = liveFindings.map(markBlocked)
    return result
  }

  const baselineById = new Map()
  for (const entry of baseline) {
    baselineById.set(entry.id, entry)
  }

  const matchedBaselineIds = new Set()
  for (const finding of liveFindings) {
    const baselineEntry = baselineById.get(finding.id)

    if (!baselineEntry) {
      result.blocked.push(markBlocked(finding))
      continue
    }

    matchedBaselineIds.add(baselineEntry.id)
    const allowedFinding = {
      ...finding,
      state: baselineEntry.state,
      owner: baselineEntry.owner,
      reason: baselineEntry.reason,
      replacement: baselineEntry.replacement,
      migrationQueue: baselineEntry.migrationQueue
    }

    if (baselineEntry.state === "active_migration_target") {
      result.activeMigrationTargets.push(allowedFinding)
      continue
    }

    result.allowedLegacy.push(allowedFinding)
  }

  result.staleBaseline = baseline.filter(
    (entry) => !matchedBaselineIds.has(entry.id)
  )

  return result
}

export async function runGuardOnSources({ sources, baseline }) {
  const findings = sources.flatMap(({ relativePath, source }) =>
    analyzeSource({ relativePath, source })
  )
  const result = applyBaseline({ findings, baseline })
  const report = formatReport(result)
  const exitCode =
    result.blocked.length > 0 || result.baselineErrors.length > 0 ? 1 : 0

  return { ...result, findings, report, exitCode }
}

export function formatReport(result) {
  const baselineErrors = result?.baselineErrors ?? []
  const blocked = sortedEntries(result?.blocked ?? [])
  const activeMigrationTargets = sortedEntries(
    result?.activeMigrationTargets ?? []
  )
  const allowedLegacy = sortedEntries(result?.allowedLegacy ?? [])
  const staleBaseline = sortedEntries(result?.staleBaseline ?? [])

  if (
    baselineErrors.length === 0 &&
    blocked.length === 0 &&
    activeMigrationTargets.length === 0 &&
    allowedLegacy.length === 0 &&
    staleBaseline.length === 0
  ) {
    return "No product-state guard issues found"
  }

  const sections = []

  if (baselineErrors.length > 0) {
    sections.push(formatTextList("Invalid baseline entries", baselineErrors))
  }

  if (blocked.length > 0) {
    sections.push(formatEntryList("Blocked product-state findings", blocked))
  }

  if (activeMigrationTargets.length > 0) {
    sections.push(
      formatEntryList(
        "Active product-state migration targets",
        activeMigrationTargets
      )
    )
  }

  if (allowedLegacy.length > 0) {
    sections.push(
      formatEntryList(
        "Allowed legacy product-state exceptions",
        allowedLegacy
      )
    )
  }

  if (staleBaseline.length > 0) {
    sections.push(formatEntryList("Stale baseline entries", staleBaseline))
  }

  const remainingBaseline = sortedEntries([
    ...activeMigrationTargets,
    ...allowedLegacy
  ])
  if (remainingBaseline.length > 0) {
    sections.push(formatBaselineTotals(remainingBaseline))
  }

  return sections.join("\n\n")
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

function collectCanonicalDesignSystemUsage(sourceFile, fileSubject) {
  const imports = {
    emptyState: new Set(),
    loadingState: new Set(),
    badge: new Set(),
    stateRegistry: new Set()
  }

  for (const statement of sourceFile.statements) {
    if (
      !ts.isImportDeclaration(statement) ||
      !ts.isStringLiteral(statement.moduleSpecifier)
    ) {
      continue
    }

    const importPath = statement.moduleSpecifier.text
    const isCanonicalComponentImport =
      importPath.startsWith("@/components/ui") ||
      importPath.startsWith("@tldw/ui/components/ui")
    const isDesignSystemImport =
      importPath.startsWith("@/design-system") ||
      importPath.startsWith("@tldw/ui/design-system")

    if (!isCanonicalComponentImport && !isDesignSystemImport) {
      continue
    }

    const importClause = statement.importClause
    const namedBindings = importClause?.namedBindings
    if (namedBindings && ts.isNamedImports(namedBindings)) {
      for (const element of namedBindings.elements) {
        const importedName = element.propertyName?.text ?? element.name.text
        if (isCanonicalComponentImport && importedName === "EmptyState") {
          imports.emptyState.add(element.name.text)
        }
        if (isCanonicalComponentImport && importedName === "LoadingState") {
          imports.loadingState.add(element.name.text)
        }
        if (isCanonicalComponentImport && importedName === "Badge") {
          imports.badge.add(element.name.text)
        }
        if (isDesignSystemImport && importedName === "getDesignSystemState") {
          imports.stateRegistry.add(element.name.text)
        }
      }
    }

    if (
      isCanonicalComponentImport &&
      importClause?.name &&
      /\/EmptyState$/.test(importPath)
    ) {
      imports.emptyState.add(importClause.name.text)
    }
    if (
      isCanonicalComponentImport &&
      importClause?.name &&
      /\/LoadingState$/.test(importPath)
    ) {
      imports.loadingState.add(importClause.name.text)
    }
    if (
      isCanonicalComponentImport &&
      importClause?.name &&
      /\/Badge$/.test(importPath)
    ) {
      imports.badge.add(importClause.name.text)
    }
  }

  const stateRegistryOwners = collectCallExpressionOwners(
    sourceFile,
    imports.stateRegistry,
    fileSubject
  )
  const returnedBadgeOwners = collectReturnedJsxTreeTagOwners(
    sourceFile,
    imports.badge,
    fileSubject
  )

  return {
    emptyStateOwners: collectJsxTagOwners(
      sourceFile,
      imports.emptyState,
      fileSubject
    ),
    loadingStateOwners: collectReturnedJsxTagOwners(
      sourceFile,
      imports.loadingState,
      fileSubject
    ),
    statusBadgeOwners: intersectSets(returnedBadgeOwners, stateRegistryOwners)
  }
}

function collectLocalComponentSubjects(sourceFile, fileSubject) {
  const subjects = new Set([fileSubject])

  walk(sourceFile, (node) => {
    if (ts.isFunctionDeclaration(node) && node.name) {
      if (
        isLikelyComponentName(node.name.text) &&
        (functionLikeReturnsJsx(node) || hasExportModifier(node))
      ) {
        subjects.add(node.name.text)
      }
      return
    }

    if (ts.isClassDeclaration(node) && node.name) {
      if (isLikelyComponentName(node.name.text)) {
        subjects.add(node.name.text)
      }
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
      if (
        isLikelyComponentName(node.name.text) &&
        (functionLikeReturnsJsx(node.initializer) ||
          hasExportModifier(node.parent?.parent))
      ) {
        subjects.add(node.name.text)
      }
    }
  })

  return subjects
}

function pushLocalComponentFinding(
  findings,
  relativePath,
  subject,
  canonicalUsage = {}
) {
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

  if (
    EMPTY_COMPONENT_PATTERN.test(subject) &&
    !canonicalUsage.emptyStateOwners?.has(subject)
  ) {
    pushFinding(findings, {
      relativePath,
      rule: "local-empty-state",
      subject,
      message: `${subject} should use the shared EmptyState primitive.`
    })
  }

  if (
    LOADING_COMPONENT_PATTERN.test(subject) &&
    !canonicalUsage.loadingStateOwners?.has(subject)
  ) {
    pushFinding(findings, {
      relativePath,
      rule: "local-loading-state",
      subject,
      message: `${subject} should use the shared LoadingState primitive.`
    })
  }

  if (
    STATUS_COMPONENT_PATTERN.test(subject) &&
    !canonicalUsage.statusBadgeOwners?.has(subject)
  ) {
    pushFinding(findings, {
      relativePath,
      rule: "local-status-badge",
      subject,
      message: `${subject} should map status through the design system.`
    })
  }
}

function collectJsxTagOwners(sourceFile, localNames, fallbackOwner) {
  const owners = new Set()

  if (!localNames || localNames.size === 0) {
    return owners
  }

  walk(sourceFile, (node) => {
    if (!ts.isJsxSelfClosingElement(node) && !ts.isJsxOpeningElement(node)) {
      return
    }

    const localName = jsxTagName(node.tagName)
    if (!localName || !localNames.has(localName)) {
      return
    }

    const ownerName = findNearestOwnerName(node) ?? fallbackOwner
    if (ownerName) {
      owners.add(ownerName)
    }
  })

  return owners
}

function collectCallExpressionOwners(sourceFile, localNames, fallbackOwner) {
  const owners = new Set()

  if (!localNames || localNames.size === 0) {
    return owners
  }

  walk(sourceFile, (node) => {
    if (!ts.isIdentifier(node) || !localNames.has(node.text)) {
      return
    }

    if (
      isImportIdentifier(node) ||
      isTypeQueryIdentifier(node) ||
      !isCallExpressionCallee(node)
    ) {
      return
    }

    const ownerName = findNearestOwnerName(node) ?? fallbackOwner
    if (ownerName) {
      owners.add(ownerName)
    }
  })

  return owners
}

function isImportIdentifier(node) {
  return Boolean(
    node.parent &&
      (ts.isImportSpecifier(node.parent) ||
        ts.isImportClause(node.parent) ||
        ts.isNamespaceImport(node.parent))
  )
}

function isTypeQueryIdentifier(node) {
  return Boolean(node.parent && ts.isTypeQueryNode(node.parent))
}

function isCallExpressionCallee(node) {
  return Boolean(
    node.parent &&
      ts.isCallExpression(node.parent) &&
      node.parent.expression === node
  )
}

function intersectSets(left, right) {
  const intersection = new Set()

  for (const value of left) {
    if (right.has(value)) {
      intersection.add(value)
    }
  }

  return intersection
}

function collectReturnedJsxTagOwners(sourceFile, localNames, fallbackOwner) {
  const owners = new Set()

  if (!localNames || localNames.size === 0) {
    return owners
  }

  walk(sourceFile, (node) => {
    if (!isFunctionLikeNode(node)) {
      return
    }

    const ownerName = returnedJsxOwnerName(node, fallbackOwner)
    if (!ownerName) {
      return
    }

    const returnsMatchingTag = collectOwnReturnExpressions(node).some(
      (expression) => expressionDirectlyReturnsJsxTag(expression, localNames)
    )
    if (returnsMatchingTag) {
      owners.add(ownerName)
    }
  })

  return owners
}

function collectReturnedJsxTreeTagOwners(sourceFile, localNames, fallbackOwner) {
  const owners = new Set()

  if (!localNames || localNames.size === 0) {
    return owners
  }

  walk(sourceFile, (node) => {
    if (!isFunctionLikeNode(node)) {
      return
    }

    const ownerName = returnedJsxOwnerName(node, fallbackOwner)
    if (!ownerName) {
      return
    }

    const returnsMatchingTag = collectOwnReturnExpressions(node).some(
      (expression) => returnedExpressionContainsJsxTag(expression, localNames)
    )
    if (returnsMatchingTag) {
      owners.add(ownerName)
    }
  })

  return owners
}

function returnedJsxOwnerName(functionNode, fallbackOwner) {
  if (functionNode.name && ts.isIdentifier(functionNode.name)) {
    return functionNode.name.text
  }

  if (
    ts.isArrowFunction(functionNode) ||
    ts.isFunctionExpression(functionNode)
  ) {
    const variable = findVariableDeclarationOwner(functionNode)
    if (
      variable &&
      ts.isIdentifier(variable.name) &&
      isLikelyComponentName(variable.name.text)
    ) {
      return variable.name.text
    }
  }

  if (isDefaultExportFunction(functionNode)) {
    return fallbackOwner
  }

  return undefined
}

function isDefaultExportFunction(functionNode) {
  return (
    Boolean(
      functionNode.modifiers?.some(
        (modifier) => modifier.kind === ts.SyntaxKind.DefaultKeyword
      )
    ) ||
    Boolean(functionNode.parent && ts.isExportAssignment(functionNode.parent))
  )
}

function collectOwnReturnExpressions(functionNode) {
  if (!functionNode.body) {
    return []
  }

  if (!ts.isBlock(functionNode.body)) {
    return [functionNode.body]
  }

  const expressions = []
  walkOwnReturnStatements(functionNode.body, (returnStatement) => {
    if (returnStatement.expression) {
      expressions.push(returnStatement.expression)
    }
  })

  return expressions
}

function walkOwnReturnStatements(node, visitor) {
  const visit = (child) => {
    if (
      child !== node &&
      (isFunctionLikeNode(child) ||
        ts.isClassDeclaration(child) ||
        ts.isClassExpression(child))
    ) {
      return
    }

    if (ts.isReturnStatement(child)) {
      visitor(child)
      return
    }

    ts.forEachChild(child, visit)
  }

  visit(node)
}

function expressionDirectlyReturnsJsxTag(expression, localNames) {
  const unwrapped = unwrapReturnedExpression(expression)

  if (ts.isJsxSelfClosingElement(unwrapped)) {
    const localName = jsxTagName(unwrapped.tagName)
    return Boolean(localName && localNames.has(localName))
  }

  if (ts.isJsxElement(unwrapped)) {
    const localName = jsxTagName(unwrapped.openingElement.tagName)
    return Boolean(localName && localNames.has(localName))
  }

  if (ts.isConditionalExpression(unwrapped)) {
    return (
      expressionDirectlyReturnsJsxTag(unwrapped.whenTrue, localNames) ||
      expressionDirectlyReturnsJsxTag(unwrapped.whenFalse, localNames)
    )
  }

  return false
}

function returnedExpressionContainsJsxTag(expression, localNames) {
  const unwrapped = unwrapReturnedExpression(expression)

  if (ts.isConditionalExpression(unwrapped)) {
    return (
      returnedExpressionContainsJsxTag(unwrapped.whenTrue, localNames) ||
      returnedExpressionContainsJsxTag(unwrapped.whenFalse, localNames)
    )
  }

  let found = false

  const visit = (node) => {
    if (found) {
      return
    }

    if (node !== unwrapped && isFunctionLikeNode(node)) {
      return
    }

    if (ts.isJsxSelfClosingElement(node)) {
      const localName = jsxTagName(node.tagName)
      if (localName && localNames.has(localName)) {
        found = true
      }
      return
    }

    if (ts.isJsxOpeningElement(node)) {
      const localName = jsxTagName(node.tagName)
      if (localName && localNames.has(localName)) {
        found = true
      }
      return
    }

    ts.forEachChild(node, visit)
  }

  visit(unwrapped)
  return found
}

function unwrapReturnedExpression(expression) {
  let current = expression

  while (true) {
    if (
      ts.isParenthesizedExpression(current) ||
      ts.isAsExpression(current) ||
      ts.isTypeAssertionExpression(current) ||
      ts.isNonNullExpression(current)
    ) {
      current = current.expression
      continue
    }

    if (
      typeof ts.isSatisfiesExpression === "function" &&
      ts.isSatisfiesExpression(current)
    ) {
      current = current.expression
      continue
    }

    return current
  }
}

function isFunctionLikeNode(node) {
  return (
    ts.isFunctionDeclaration(node) ||
    ts.isFunctionExpression(node) ||
    ts.isArrowFunction(node) ||
    ts.isMethodDeclaration(node) ||
    ts.isConstructorDeclaration(node) ||
    ts.isGetAccessorDeclaration(node) ||
    ts.isSetAccessorDeclaration(node)
  )
}

function pushCanonicalLabelFindings(findings, relativePath, sourceFile) {
  walk(sourceFile, (node) => {
    if (
      ts.isStringLiteral(node) ||
      ts.isNoSubstitutionTemplateLiteral(node) ||
      ts.isJsxText(node)
    ) {
      const label = canonicalLabelFromLiteral(node.getText(sourceFile), node.text)
      if (label) {
        pushFinding(findings, {
          relativePath,
          rule: "canonical-state-label",
          subject: label,
          message: `"${label}" should come from the design-system state registry.`,
          line: lineForNode(sourceFile, node),
          identityHint: [
            "label",
            label,
            findNearestOwnerName(node) ?? subjectFromPath(relativePath)
          ].join("|")
        })
      }
    }
  })
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
    const ownerName = findNearestOwnerName(node)
    if (
      !isProductStateAntdUse(importedName, useContext, {
        ownerName,
        fileSubject: context.fileSubject
      })
    ) {
      return
    }

    pushFinding(findings, {
      relativePath,
      rule: "antd-product-state-import",
      subject: importedName,
      message: `${importedName} from AntD is rendering product-state UI directly.`,
      line: lineForNode(sourceFile, node),
      identityHint: antdUseIdentityHint({
        importedName,
        ownerName: ownerName ?? context.fileSubject,
        useContext
      })
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

  const scopedName = context.ownerName ?? context.fileSubject
  return scopedName ? PRODUCT_STATE_NAME_PATTERN.test(scopedName) : false
}

function collectJsxUseContext(node, sourceFile) {
  const texts = []
  const attributes = getJsxAttributes(node)

  for (const attribute of attributes) {
    if (TEXT_BEARING_PROP_NAMES.has(jsxAttributeName(attribute))) {
      texts.push(...jsxRenderedAttributeTexts(attribute, sourceFile))
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
    const propName = jsxAttributeName(attribute)
    const propValue = jsxLiteralAttributeValue(attribute)

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
    hasSeverityProp,
    severityValues: attributes
      .map((attribute) => [
        jsxAttributeName(attribute),
        jsxLiteralAttributeValue(attribute)
      ])
      .filter(
        ([propName, propValue]) =>
          SEVERITY_PROP_NAMES.has(propName) && typeof propValue === "string"
      )
      .map(([propName, propValue]) => `${propName}:${propValue.toLowerCase()}`),
    texts
  }
}

function getJsxAttributes(node) {
  if (!("attributes" in node)) {
    return []
  }

  return node.attributes.properties.filter(ts.isJsxAttribute)
}

function jsxAttributeName(attribute) {
  return ts.isIdentifier(attribute.name) ? attribute.name.text : undefined
}

function jsxLiteralAttributeValue(attribute) {
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
    if (
      ts.isStringLiteral(expression) ||
      ts.isNoSubstitutionTemplateLiteral(expression)
    ) {
      return expression.text
    }
  }

  return undefined
}

function jsxRenderedAttributeTexts(attribute, sourceFile) {
  if (!attribute.initializer) {
    return []
  }

  if (ts.isStringLiteral(attribute.initializer)) {
    return [attribute.initializer.text]
  }

  if (
    ts.isJsxExpression(attribute.initializer) &&
    attribute.initializer.expression
  ) {
    const expression = attribute.initializer.expression

    if (
      ts.isStringLiteral(expression) ||
      ts.isNoSubstitutionTemplateLiteral(expression)
    ) {
      return [expression.text]
    }

    if (ts.isJsxElement(expression) || ts.isJsxFragment(expression)) {
      return jsxChildTexts(expression, sourceFile)
    }
  }

  return []
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
      continue
    }

    if (ts.isJsxFragment(child)) {
      texts.push(...jsxChildTexts(child, sourceFile))
    }
  }

  return texts
}

function jsxTagName(tagName) {
  return ts.isIdentifier(tagName) ? tagName.text : undefined
}

function findNearestOwnerName(node) {
  let current = node.parent

  while (current) {
    if (
      ts.isFunctionDeclaration(current) ||
      ts.isClassDeclaration(current) ||
      ts.isMethodDeclaration(current)
    ) {
      return current.name && ts.isIdentifier(current.name)
        ? current.name.text
        : undefined
    }

    if (ts.isArrowFunction(current) || ts.isFunctionExpression(current)) {
      const variable = findVariableDeclarationOwner(current)
      if (
        variable &&
        ts.isIdentifier(variable.name) &&
        isLikelyComponentName(variable.name.text)
      ) {
        return variable.name.text
      }
    }

    current = current.parent
  }

  return undefined
}

function findVariableDeclarationOwner(functionNode) {
  let current = functionNode.parent

  while (current) {
    if (ts.isVariableDeclaration(current)) {
      return current
    }

    if (
      ts.isFunctionDeclaration(current) ||
      ts.isClassDeclaration(current) ||
      ts.isMethodDeclaration(current)
    ) {
      return undefined
    }

    current = current.parent
  }

  return undefined
}

function canonicalLabelFromLiteral(rawText, valueText) {
  const text = (valueText ?? rawText).trim()
  return CANONICAL_STATE_LABELS.find((label) => text === label)
}

function pushFinding(
  findings,
  { relativePath, rule, subject, message, line, identityHint }
) {
  findings.push({
    id: createFindingId(rule, relativePath, subject),
    path: relativePath,
    rule,
    subject,
    message,
    ...(typeof line === "number" ? { line } : {}),
    replacement: RULE_REPLACEMENTS[rule],
    ...(identityHint ? { identityHint } : {})
  })
}

function disambiguateFindingIds(findings) {
  const groups = new Map()

  for (const finding of findings) {
    const group = groups.get(finding.id) ?? []
    group.push(finding)
    groups.set(finding.id, group)
  }

  const disambiguated = new Map()

  for (const group of groups.values()) {
    if (group.length === 1) {
      const { identityHint, ...publicFinding } = group[0]
      disambiguated.set(group[0], publicFinding)
      continue
    }

    const suffixCounts = new Map()

    group.forEach((finding, index) => {
      const { identityHint, ...publicFinding } = finding
      const suffixBase = duplicateFindingSuffix(
        occurrenceIdentityHint(finding, identityHint, index)
      )
      const suffixCount = suffixCounts.get(suffixBase) ?? 0
      suffixCounts.set(suffixBase, suffixCount + 1)

      const suffix =
        suffixCount === 0 ? suffixBase : `${suffixBase}-${suffixCount + 1}`
      disambiguated.set(finding, {
        ...publicFinding,
        id: `${publicFinding.id}#${suffix}`
      })
    })
  }

  return findings.map((finding) => disambiguated.get(finding))
}

function occurrenceIdentityHint(finding, identityHint, index) {
  return [
    identityHint ?? `occurrence-${index + 1}`,
    typeof finding.line === "number" ? `line:${finding.line}` : undefined
  ]
    .filter(Boolean)
    .join("|")
}

function antdUseIdentityHint({ importedName, ownerName, useContext }) {
  return [
    "antd",
    importedName,
    ownerName,
    ...useContext.severityValues,
    ...useContext.texts.map(normalizeTextSignal).filter(Boolean)
  ]
    .filter(Boolean)
    .join("|")
}

function duplicateFindingSuffix(identityHint) {
  const normalized = identityHint
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
  const readable = normalized.slice(0, 56).replace(/-+$/g, "")
  const hash = stableHash(identityHint)

  return `${readable || "occurrence"}-${hash}`
}

function isDuplicateFindingId(id, expectedId) {
  if (!id.startsWith(`${expectedId}#`)) {
    return false
  }

  return DUPLICATE_FINDING_ID_SUFFIX_PATTERN.test(id.slice(expectedId.length))
}

function stableHash(value) {
  let hash = 0x811c9dc5

  for (let index = 0; index < value.length; index += 1) {
    hash ^= value.charCodeAt(index)
    hash = Math.imul(hash, 0x01000193)
  }

  return (hash >>> 0).toString(36)
}

function markBlocked(finding) {
  return {
    ...finding,
    state: "blocked"
  }
}

function sortedEntries(entries) {
  return [...entries].sort((left, right) =>
    [
      left.rule,
      left.path,
      left.subject,
      left.id
    ]
      .join("\0")
      .localeCompare(
        [
          right.rule,
          right.path,
          right.subject,
          right.id
        ].join("\0")
      )
  )
}

function formatTextList(heading, items) {
  return [heading, ...items.map((item) => `- ${item}`)].join("\n")
}

function formatEntryList(heading, entries) {
  return [heading, ...entries.flatMap(formatEntry)].join("\n")
}

function formatEntry(entry) {
  const lines = [
    `- ${entry.rule}: ${entry.path} (${entry.subject})`
  ]

  appendEntryDetail(lines, "id", entry.id)
  appendEntryDetail(lines, "line", entry.line)
  appendEntryDetail(lines, "message", entry.message)
  appendEntryDetail(lines, "owner", entry.owner)
  appendEntryDetail(lines, "replacement", entry.replacement)
  appendEntryDetail(lines, "migrationQueue", entry.migrationQueue)
  appendEntryDetail(lines, "reason", entry.reason)

  return lines
}

function appendEntryDetail(lines, label, value) {
  if (value === undefined || value === null || value === "") {
    return
  }

  lines.push(`  ${label}: ${value}`)
}

function formatBaselineTotals(entries) {
  return [
    `Baseline exceptions: ${entries.length}`,
    "By rule:",
    ...formatCountGroup(entries, "rule"),
    "By migration queue:",
    ...formatCountGroup(entries, "migrationQueue")
  ].join("\n")
}

function formatCountGroup(entries, field) {
  const counts = new Map()

  for (const entry of entries) {
    if (!entry[field]) {
      continue
    }

    counts.set(entry[field], (counts.get(entry[field]) ?? 0) + 1)
  }

  return [...counts.entries()]
    .sort(([left], [right]) => left.localeCompare(right))
    .map(([label, count]) => `- ${label}: ${count}`)
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

function hasExportModifier(node) {
  return Boolean(
    node?.modifiers?.some(
      (modifier) => modifier.kind === ts.SyntaxKind.ExportKeyword
    )
  )
}

function isLikelyComponentName(name) {
  return /^[A-Z][A-Za-z0-9]*$/.test(name)
}

function functionLikeReturnsJsx(functionNode) {
  if (!functionNode.body) {
    return false
  }

  if (!ts.isBlock(functionNode.body)) {
    return containsJsx(functionNode.body)
  }

  let returnsJsx = false
  walkUntil(functionNode.body, (node) => {
    if (!ts.isReturnStatement(node) || !node.expression) {
      return false
    }

    returnsJsx = containsJsx(node.expression)
    return returnsJsx
  })

  return returnsJsx
}

function containsJsx(node) {
  let hasJsx = false

  walkUntil(node, (child) => {
    if (
      ts.isJsxElement(child) ||
      ts.isJsxSelfClosingElement(child) ||
      ts.isJsxFragment(child)
    ) {
      hasJsx = true
      return true
    }

    return false
  })

  return hasJsx
}

function normalizePath(path) {
  return path.replaceAll("\\", "/")
}

function isRecord(value) {
  return Boolean(value) && typeof value === "object" && !Array.isArray(value)
}

function isPresentString(value) {
  return typeof value === "string" && value.trim().length > 0
}

function walk(node, visitor) {
  visitor(node)
  ts.forEachChild(node, (child) => walk(child, visitor))
}

function walkUntil(node, visitor) {
  if (visitor(node)) {
    return true
  }

  let shouldStop = false
  ts.forEachChild(node, (child) => {
    if (shouldStop) {
      return
    }

    shouldStop = walkUntil(child, visitor)
  })

  return shouldStop
}
