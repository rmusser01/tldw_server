import { isDeepStrictEqual } from "node:util"
import { collectPlaywrightCases } from "./report.mjs"
import { CELLS, PHASES, SURFACES, RELEASE_UAT_CATALOG } from "./release-catalog-data.mjs"

export { RELEASE_UAT_CATALOG }

const nonempty = value => typeof value === "string" && value.trim().length > 0
const caseKey = entry => JSON.stringify([entry.project, entry.file, entry.titlePath])
const contextKey = context => JSON.stringify([context.cell, context.browser, context.surface, context.phase, context.upgradeFrom ?? null])
const requirementKey = entry => JSON.stringify([entry.workflow, entry.variant, entry.contextId, entry.mode, entry.instance ?? null])

function disposition(entry, label) {
  if (!["required", "excluded", "not-applicable"].includes(entry?.status)) {
    throw new Error(`Invalid disposition: ${label}`)
  }
  if (entry.status !== "required" && !nonempty(entry.reason)) {
    throw new Error(`Explicit exclusion needs a reason: ${label}`)
  }
}

function scopeAxis(entries, expected, name) {
  if (!Array.isArray(entries) || entries.length !== expected.length ||
      new Set(entries.map(entry => entry.id)).size !== expected.length ||
      entries.some(entry => !expected.includes(entry.id))) {
    throw new Error(`Release scope.${name} must explicitly account for ${expected.join(", ")}`)
  }
  for (const entry of entries) disposition(entry, `scope.${name}.${entry.id}`)
  return entries.filter(entry => entry.status === "required").map(entry => entry.id)
}

function assertInstanceInventories(scope) {
  scopeAxis(scope?.instances, RELEASE_UAT_CATALOG.instanceInventories.map(group => group.id), "instances")
  for (const group of scope.instances) {
    if (!Array.isArray(group.instances) || group.instances.some(id => !nonempty(id) || id !== id.trim()) ||
        new Set(group.instances).size !== group.instances.length ||
        (group.status === "required" ? !group.instances.length : group.instances.length !== 0)) {
      throw new Error(`Instance inventory ${group.id} needs unique named instances, or an explicit no-applicable-instance disposition and reason`)
    }
  }
  return new Map(scope.instances.map(group => [group.id, group.instances]))
}

function assertContexts(manifest) {
  const scope = manifest.scope ?? {}
  const cells = scopeAxis(scope.cells, CELLS, "cells")
  const surfaces = scopeAxis(scope.surfaces, SURFACES, "surfaces")
  const phases = scopeAxis(scope.phases, PHASES, "phases")
  scopeAxis(scope.upgradePaths, ["previous-public", "oldest-supported"], "upgradePaths")
  if (!cells.length || !surfaces.includes("web") || !phases.includes("fresh-state")) {
    throw new Error("Release scope requires at least one cell and fresh-state web coverage")
  }
  const upgradePaths = scope.upgradePaths.filter(entry => entry.status === "required")
  if (upgradePaths.some(entry => !nonempty(entry.from)) ||
      phases.includes("supported-upgrade") !== Boolean(upgradePaths.length)) {
    throw new Error("Required supported-upgrade phase needs explicitly versioned upgradePaths")
  }
  if (!Array.isArray(scope.browsers) || !scope.browsers.length ||
      new Set(scope.browsers.map(browser => browser.id)).size !== scope.browsers.length ||
      scope.browsers.some(browser => !nonempty(browser.id) || !Array.isArray(browser.surfaces) || !browser.surfaces.length ||
        new Set(browser.surfaces).size !== browser.surfaces.length || browser.surfaces.some(surface => !surfaces.includes(surface))) ||
      !scope.browsers[0].surfaces.includes("web") ||
      surfaces.some(surface => !scope.browsers.some(browser => browser.surfaces.includes(surface)))) {
    throw new Error("Release scope.browsers must explicitly cover each required surface with a primary web browser")
  }
  const expected = new Set()
  for (const cell of cells) for (const browser of scope.browsers) for (const surface of browser.surfaces) for (const phase of phases) {
    const starts = phase === "supported-upgrade" ? [...new Set(upgradePaths.map(entry => entry.from))] : [undefined]
    for (const upgradeFrom of starts) expected.add(contextKey({ cell, browser: browser.id, surface, phase, upgradeFrom }))
  }
  if (!Array.isArray(manifest.contexts)) throw new Error("Missing release contexts")
  const ids = new Set()
  const observed = new Set()
  for (const context of manifest.contexts) {
    const key = contextKey(context)
    if (!nonempty(context.id) || ids.has(context.id) || observed.has(key)) throw new Error(`Duplicate or invalid context: ${context.id}`)
    if (!expected.has(key)) throw new Error(`Unknown or excluded context: ${key}`)
    ids.add(context.id)
    observed.add(key)
  }
  const missing = [...expected].filter(key => !observed.has(key))
  if (missing.length) throw new Error(`Missing required context(s): ${missing.join(", ")}`)
}

function assertListOnly(node) {
  for (const spec of node.specs ?? []) for (const test of spec.tests ?? []) {
    if (test.results?.length) throw new Error("Catalog mapping requires list-only collection, not execution results")
  }
  for (const suite of node.suites ?? []) assertListOnly(suite)
}

/**
 * Validate a planned catalog against list-only Playwright registrations.
 *
 * Every supported axis and applicable variant/context/mode needs an explicit
 * disposition. scope.instances declares each fixed instance inventory using
 * { id, status: "required", instances: ["format-or-path-id", ...] }, or an
 * excluded/not-applicable disposition with instances: [] and a reason. Variants
 * naming instanceSet require a separate entry.instance for every listed ID,
 * including recovery and human review; an omitted instance is a coverage gap.
 * A required D/L mapping declares the exact catalog assertions and
 * a rationale; this function verifies registration identity, not the sufficiency
 * of the test's assertions or any execution outcome. Required U rows carry
 * a pending human review or separately attributed completed human evidence,
 * never an automated mapping. Evidence
 * paths/hashes are declarations here, not verified artifacts or UX approval.
 *
 * requiredCases has one row per explicit requirement. executions deduplicates
 * shared mappings within each context, retaining all requirements; use those
 * rows for per-context receipt accounting. Retries and failed attempts stay in
 * the existing execution validator and are neither read nor rewritten here.
 */
export function assertReleaseCatalog(manifest, collectionReport) {
  if (manifest?.catalogVersion !== RELEASE_UAT_CATALOG.version) throw new Error("Unsupported release catalog version")
  assertContexts(manifest)
  const instanceInventories = assertInstanceInventories(manifest.scope)
  if (!collectionReport || !Array.isArray(collectionReport.errors) || collectionReport.errors.length) {
    throw new Error("A successful list-only collection report is required")
  }
  assertListOnly(collectionReport)
  const registered = new Set()
  for (const entry of collectPlaywrightCases(collectionReport)) {
    const key = caseKey(entry)
    if (registered.has(key)) throw new Error(`Duplicate collection registration: ${key}`)
    registered.add(key)
  }
  if (!registered.size) throw new Error("Empty Playwright collection cannot establish catalog mappings")

  const requirements = new Map()
  for (const context of manifest.contexts) for (const workflow of RELEASE_UAT_CATALOG.workflows) for (const variant of workflow.variants) {
    if (!variant.cells.includes(context.cell) || !variant.surfaces.includes(context.surface) || !variant.phases.includes(context.phase)) continue
    const instances = variant.instanceSet ? instanceInventories.get(variant.instanceSet) : [undefined]
    for (const instance of instances) for (const mode of variant.modes) {
      const requirement = { workflow: workflow.id, variant: variant.id, contextId: context.id, mode, instance,
        assertions: variant.assertions, needs: variant.needs, fixtures: variant.fixtures }
      requirements.set(requirementKey(requirement), requirement)
    }
  }
  if (!Array.isArray(manifest.entries)) throw new Error("Missing release catalog entries")
  const observed = new Set()
  const requiredCases = []
  const humanReviews = []
  const exclusions = []
  const executions = new Map()
  const caseIds = new Map()
  for (const entry of manifest.entries) {
    const key = requirementKey(entry)
    const requirement = requirements.get(key)
    if (!requirement) throw new Error(`Unknown workflow/variant/context/mode mapping: ${key}`)
    if (observed.has(key)) throw new Error(`Duplicate requirement mapping: ${key}`)
    observed.add(key)
    disposition(entry, key)
    if (entry.status !== "required") {
      if (entry.mapping || entry.humanReview) throw new Error(`Excluded requirement cannot carry a passing mapping or human evidence: ${key}`)
      exclusions.push(entry)
      continue
    }
    if (entry.mode === "U") {
      const review = entry.humanReview
      const pending = review?.status === "pending"
      const completed = review?.status === "completed"
      if (entry.mapping || (!pending && !completed) ||
          (pending && (review.evidence !== undefined || (review.reviewer !== undefined && !nonempty(review.reviewer)))) ||
          (completed && (!nonempty(review.reviewer) || !Array.isArray(review.evidence) || !review.evidence.length ||
            review.evidence.some(evidence => !nonempty(evidence.path) || !/^[a-f0-9]{64}$/.test(evidence.sha256 ?? ""))))) {
        throw new Error(`Required human UX review must be pending or have completed reviewer/evidence declarations, never a Playwright mapping: ${key}`)
      }
      humanReviews.push(entry)
      continue
    }
    const mapping = entry.mapping
    if (entry.humanReview || ![mapping?.id, mapping?.project, mapping?.file].every(nonempty) ||
        !Array.isArray(mapping.titlePath) || !mapping.titlePath.length || !mapping.titlePath.every(nonempty)) {
      throw new Error(`Required automated mapping needs case ID, project, file and full titlePath: ${key}`)
    }
    if (!isDeepStrictEqual(mapping.assertions, requirement.assertions) || !nonempty(mapping.rationale)) {
      throw new Error(`Mapping must explicitly declare catalog assertions and their test rationale: ${key}`)
    }
    if (entry.workflow === "S-06" && (!nonempty(entry.details?.input) || !nonempty(entry.details?.oracle) ||
        !Array.isArray(entry.details?.steps) || !entry.details.steps.length || !entry.details.steps.every(nonempty))) {
      throw new Error(`Specialized variant requires pinned input, oracle and steps: ${key}`)
    }
    const identity = caseKey(mapping)
    if (!registered.has(identity)) throw new Error(`Mapped test is not registered in list-only collection: ${identity}`)
    const executionKey = JSON.stringify([entry.contextId, identity])
    const idKey = JSON.stringify([entry.contextId, mapping.id])
    if ((caseIds.has(idKey) && caseIds.get(idKey) !== identity) ||
        (executions.has(executionKey) && executions.get(executionKey).id !== mapping.id)) {
      throw new Error(`Conflicting case ID for mapped execution: ${mapping.id}`)
    }
    caseIds.set(idKey, identity)
    const requiredCase = { ...mapping, workflow: entry.workflow, variant: entry.variant, contextId: entry.contextId, mode: entry.mode, instance: entry.instance }
    requiredCases.push(requiredCase)
    if (!executions.has(executionKey)) executions.set(executionKey, { ...requiredCase, requirements: [] })
    executions.get(executionKey).requirements.push({ workflow: entry.workflow, variant: entry.variant, mode: entry.mode, instance: entry.instance })
  }
  const missing = [...requirements.keys()].filter(key => !observed.has(key))
  if (missing.length) {
    const error = new Error(`Missing ${missing.length} catalog requirement(s): ${missing.slice(0, 20).join(", ")}${missing.length > 20 ? " (see missingRequirements)" : ""}`)
    error.missingRequirements = missing.map(key => requirements.get(key))
    throw error
  }
  if (!requiredCases.length) throw new Error("No required automated cases in release catalog plan")
  const scopeExclusions = ["cells", "surfaces", "phases", "upgradePaths", "instances"].flatMap(axis =>
    manifest.scope[axis].filter(entry => entry.status !== "required").map(entry => ({ axis, ...entry })))
  return { catalogVersion: RELEASE_UAT_CATALOG.version, requirements: [...requirements.values()], requiredCases,
    executions: [...executions.values()], humanReviews,
    pendingHumanReviews: humanReviews.filter(entry => entry.humanReview.status === "pending"), exclusions, scopeExclusions }
}
