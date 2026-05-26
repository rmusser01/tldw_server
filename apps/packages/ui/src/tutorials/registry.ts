/**
 * Tutorial Registry
 * Central registry for all page tutorials using React Joyride
 */

import type { Placement } from "react-joyride"
import type { LucideIcon } from "lucide-react"

// ─────────────────────────────────────────────────────────────────────────────
// Type Definitions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * A single step in a tutorial
 */
export interface TutorialStep {
  /** CSS selector or data-testid for the target element */
  target: string
  /** i18n key for the step title */
  titleKey: string
  /** Fallback title if i18n key not found */
  titleFallback: string
  /** i18n key for the step content */
  contentKey: string
  /** Fallback content if i18n key not found */
  contentFallback: string
  /** Placement of the tooltip relative to the target */
  placement?: Placement
  /** Whether to disable the beacon (pulsing dot) for this step */
  disableBeacon?: boolean
  /** Whether to allow clicks on the spotlight area */
  spotlightClicks?: boolean
  /** Whether the step target is fixed positioned */
  isFixed?: boolean
}

/**
 * A complete tutorial definition
 */
export interface TutorialSequence {
  /** Explicit next tutorial in a multi-page sequence */
  nextTutorialId: string
  /** Route to navigate to before starting the next tutorial */
  nextRoute?: string
  /** i18n key for a continue/progression label */
  nextLabelKey?: string
  /** Fallback progression label */
  nextLabelFallback?: string
}

export interface TutorialDefinition {
  /** Unique identifier for this tutorial */
  id: string
  /** Route pattern to match (supports wildcards like /options/*) */
  routePattern: string
  /** i18n key for the tutorial name displayed in the help modal */
  labelKey: string
  /** Fallback label if i18n key not found */
  labelFallback: string
  /** i18n key for the tutorial description */
  descriptionKey: string
  /** Fallback description if i18n key not found */
  descriptionFallback: string
  /** Optional icon to display in the tutorial list */
  icon?: LucideIcon
  /** The steps that make up this tutorial */
  steps: TutorialStep[]
  /** IDs of other tutorials that should be completed first (optional) */
  prerequisites?: string[]
  /** Optional sequence metadata for multi-page tutorial flows */
  sequence?: TutorialSequence
  /** Priority for ordering in the tutorial list (lower = higher priority) */
  priority?: number
}

// ─────────────────────────────────────────────────────────────────────────────
// Tutorial Registry
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Import tutorial definitions from individual files
 */
import { playgroundTutorials } from "./definitions/playground"
import { researchWorkspaceTutorials } from "./definitions/research-workspace"
import { mediaTutorials } from "./definitions/media"
import { knowledgeTutorials } from "./definitions/knowledge"
import { charactersTutorials } from "./definitions/characters"
import { promptsTutorials } from "./definitions/prompts"
import { evaluationsTutorials } from "./definitions/evaluations"
import { notesTutorials } from "./definitions/notes"
import { flashcardsTutorials } from "./definitions/flashcards"
import { worldBooksTutorials } from "./definitions/world-books"
import { gettingStartedTutorials } from "./definitions/getting-started"
import { moderationTutorials } from "./definitions/moderation"
import { mcpHubTutorials } from "./definitions/mcp-hub"
import { quizTutorials } from "./definitions/quiz"
import { ttsTutorials } from "./definitions/tts"
import { sttTutorials } from "./definitions/stt"
import { watchlistsTutorials } from "./definitions/watchlists"
import { monitoringTutorials } from "./definitions/monitoring"
import { documentWorkspaceTutorials } from "./definitions/document-workspace"

/**
 * Central registry of all available tutorials
 */
export const TUTORIAL_REGISTRY: TutorialDefinition[] = [
  ...gettingStartedTutorials,
  ...playgroundTutorials,
  ...researchWorkspaceTutorials,
  ...mediaTutorials,
  ...knowledgeTutorials,
  ...charactersTutorials,
  ...promptsTutorials,
  ...evaluationsTutorials,
  ...notesTutorials,
  ...flashcardsTutorials,
  ...worldBooksTutorials,
  ...documentWorkspaceTutorials,
  ...moderationTutorials,
  ...mcpHubTutorials,
  ...quizTutorials,
  ...ttsTutorials,
  ...sttTutorials,
  ...watchlistsTutorials,
  ...monitoringTutorials
]

// ─────────────────────────────────────────────────────────────────────────────
// Helper Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Match a route pattern against a pathname
 * Supports:
 * - Exact matches: "/options/playground" matches "/options/playground"
 * - Wildcards: "/options/*" matches "/options/playground", "/options/media"
 * - Partial wildcards: "/options/prompt/*" matches "/options/prompt/studio"
 */
function matchRoute(pattern: string, pathname: string): boolean {
  const normalizedPattern = normalizeTutorialRoute(pattern)
  const normalizedPathname = normalizeTutorialRoute(pathname)

  // Exact match
  if (normalizedPattern === normalizedPathname) {
    return true
  }

  // Wildcard matching
  if (normalizedPattern.includes("*")) {
    const regexPattern = normalizedPattern
      .replace(/[.*+?^${}()|[\]\\]/g, "\\$&") // Escape special regex chars
      .replace(/\\\*/g, ".*") // Replace escaped * with .*
    const regex = new RegExp(`^${regexPattern}$`)
    return regex.test(normalizedPathname)
  }

  return false
}

const LEGACY_ROUTE_ALIASES: Record<string, string> = {
  "/options/playground": "/chat",
  "/options/chat": "/chat",
  "/options/media": "/media",
  "/options/knowledge": "/knowledge",
  "/options/characters": "/characters",
  "/options/research-workspace": "/research-workspace",
  "/options/prompts": "/prompts",
  "/options/evaluations": "/evaluations",
  "/options/notes": "/notes",
  "/options/flashcards": "/flashcards",
  "/options/world-books": "/world-books",
  "/options/document-workspace": "/document-workspace"
}

const PREFIX_ROUTE_ALIASES: Array<{ pattern: RegExp; canonical: string }> = [
  { pattern: /^\/knowledge\/thread(?:\/.*)?$/i, canonical: "/knowledge" },
  { pattern: /^\/knowledge\/shared(?:\/.*)?$/i, canonical: "/knowledge" }
]

/**
 * Tutorials are only meant for the options surface. Sidepanel hosts can share
 * route-like paths (for example "/chat") that should not trigger options tours.
 */
export function isTutorialRuntimeSuppressed(
  runtimePathname?: string | null
): boolean {
  const pathname =
    runtimePathname ??
    (typeof window !== "undefined" ? window.location.pathname : "")
  return /sidepanel/i.test(pathname || "")
}

/**
 * Normalize route-like inputs so tutorial matching works across:
 * - Canonical in-app paths (e.g. /chat)
 * - Legacy paths (e.g. /options/playground)
 * - Hash/extension URLs (e.g. chrome-extension://.../options.html#/chat?tab=...)
 */
export function normalizeTutorialRoute(routeLike: string): string {
  let route = routeLike.trim()
  if (!route) {
    return "/"
  }

  if (/^https?:\/\//i.test(route) || /^chrome-extension:\/\//i.test(route)) {
    try {
      const parsed = new URL(route)
      route = parsed.hash ? parsed.hash.slice(1) : parsed.pathname
    } catch {
      // keep original route value
    }
  }

  const optionsHashIndex = route.indexOf("options.html#")
  if (optionsHashIndex >= 0) {
    route = route.slice(optionsHashIndex + "options.html#".length)
  }

  if (route.startsWith("#")) {
    route = route.slice(1)
  }

  route = route.split("?")[0] ?? route
  route = route.split("#")[0] ?? route
  route = route.trim()

  if (!route.startsWith("/")) {
    route = `/${route}`
  }

  route = route.replace(/\/{2,}/g, "/")
  if (route.length > 1 && route.endsWith("/")) {
    route = route.slice(0, -1)
  }

  const legacyAlias = LEGACY_ROUTE_ALIASES[route]
  if (legacyAlias) {
    return legacyAlias
  }

  for (const alias of PREFIX_ROUTE_ALIASES) {
    if (alias.pattern.test(route)) {
      return alias.canonical
    }
  }

  return route
}

/**
 * Get all tutorials available for a given route
 * @param pathname - The current route pathname (e.g., "/options/playground")
 * @returns Array of tutorials matching the route, sorted by priority
 */
type GetTutorialsForRouteOptions = {
  ignoreRuntimeSuppression?: boolean
  completedTutorialIds?: CompletedTutorialIds
  includeLocked?: boolean
}

type CompletedTutorialIds = readonly string[] | ReadonlySet<string>

const getCompletedTutorialSet = (
  completedTutorialIds: CompletedTutorialIds = []
): ReadonlySet<string> =>
  completedTutorialIds instanceof Set
    ? completedTutorialIds
    : new Set(completedTutorialIds)

const sortTutorialsByPriority = (
  tutorials: readonly TutorialDefinition[]
): TutorialDefinition[] =>
  [...tutorials].sort((a, b) => (a.priority ?? 100) - (b.priority ?? 100))

export function areTutorialPrerequisitesMet(
  tutorial: TutorialDefinition,
  completedTutorialIds: CompletedTutorialIds = []
): boolean {
  const prerequisites = tutorial.prerequisites ?? []
  if (prerequisites.length === 0) {
    return true
  }

  const completedTutorials = getCompletedTutorialSet(completedTutorialIds)
  return prerequisites.every((prerequisiteId) =>
    completedTutorials.has(prerequisiteId)
  )
}

export function getTutorialsForRoute(
  pathname: string,
  options: GetTutorialsForRouteOptions = {}
): TutorialDefinition[] {
  if (!options.ignoreRuntimeSuppression && isTutorialRuntimeSuppressed()) {
    return []
  }

  const matches = TUTORIAL_REGISTRY.filter((tutorial) =>
    matchRoute(tutorial.routePattern, pathname)
  )
  const completedSet = getCompletedTutorialSet(options.completedTutorialIds)
  const eligibleMatches = options.includeLocked
    ? matches
    : matches.filter((tutorial) =>
        areTutorialPrerequisitesMet(tutorial, completedSet)
      )

  // Sort by priority (lower priority number = higher in list)
  return sortTutorialsByPriority(eligibleMatches)
}

/**
 * Get a specific tutorial by its ID
 * @param tutorialId - The unique ID of the tutorial
 * @returns The tutorial definition or undefined if not found
 */
export function getTutorialById(
  tutorialId: string
): TutorialDefinition | undefined {
  return TUTORIAL_REGISTRY.find((tutorial) => tutorial.id === tutorialId)
}

/**
 * Get the primary/basics tutorial for a route (for first-visit prompts)
 * @param pathname - The current route pathname
 * @returns The primary tutorial (first one with "basics" in ID, or first tutorial)
 */
export function getPrimaryTutorialForRoute(
  pathname: string,
  options: GetTutorialsForRouteOptions = {}
): TutorialDefinition | undefined {
  const tutorials = getTutorialsForRoute(pathname, options)
  if (tutorials.length === 0) return undefined

  const topPriority = tutorials[0].priority ?? 100
  const topPriorityTutorials = tutorials.filter(
    (tutorial) => (tutorial.priority ?? 100) === topPriority
  )

  // Prefer basics/overview only within the highest-priority group.
  const basicsTutorial = topPriorityTutorials.find(
    (t) => t.id.includes("basics") || t.id.includes("overview")
  )

  return basicsTutorial || tutorials[0]
}

/**
 * Check if any tutorials are available for a route
 * @param pathname - The current route pathname
 * @returns True if at least one tutorial is available
 */
export function hasTutorialsForRoute(
  pathname: string,
  options: GetTutorialsForRouteOptions = {}
): boolean {
  return getTutorialsForRoute(pathname, options).length > 0
}

/**
 * Get count of tutorials for a route
 * @param pathname - The current route pathname
 * @returns Number of available tutorials
 */
export function getTutorialCountForRoute(
  pathname: string,
  options: GetTutorialsForRouteOptions = {}
): number {
  return getTutorialsForRoute(pathname, options).length
}

/**
 * Resolve the next eligible tutorial in a sequence after a tutorial completes.
 */
export function getNextTutorialInSequence(
  completedTutorialId: string,
  completedTutorialIds: CompletedTutorialIds = []
): TutorialDefinition | undefined {
  const completedTutorials = new Set(completedTutorialIds)
  completedTutorials.add(completedTutorialId)
  const currentTutorial = getTutorialById(completedTutorialId)
  const explicitNextId = currentTutorial?.sequence?.nextTutorialId

  if (explicitNextId) {
    const explicitNextTutorial = getTutorialById(explicitNextId)
    if (
      explicitNextTutorial &&
      !completedTutorials.has(explicitNextTutorial.id) &&
      areTutorialPrerequisitesMet(explicitNextTutorial, completedTutorials)
    ) {
      return explicitNextTutorial
    }
  }

  return sortTutorialsByPriority(
    TUTORIAL_REGISTRY.filter(
      (tutorial) =>
        !completedTutorials.has(tutorial.id) &&
        (tutorial.prerequisites ?? []).includes(completedTutorialId) &&
        areTutorialPrerequisitesMet(tutorial, completedTutorials)
    )
  )[0]
}
