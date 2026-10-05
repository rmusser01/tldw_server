/**
 * Shared axe-core helpers for Playwright specs.
 *
 * - STAGE4_A11Y_RULES and formatAxeViolations back the Stage 4 smoke gate
 *   (e2e/smoke/stage4-axe-high-risk-routes.spec.ts).
 * - scanA11y and summariseA11yViolations back the ux-regression accessibility
 *   ratchet (e2e/ux-regression/a11y.spec.ts), which compares serious and
 *   critical violations with a committed baseline through e2e/utils/ratchet.ts.
 */
import AxeBuilder from "@axe-core/playwright"
import type { Page } from "@playwright/test"
import { waitForAppShell, waitForVisualSettle } from "./helpers"
import type { RatchetObservation } from "./ratchet"

export type AxeResults = Awaited<ReturnType<AxeBuilder["analyze"]>>
export type AxeViolation = AxeResults["violations"][number]
export type AxeImpact = NonNullable<AxeViolation["impact"]>

/** Rules enforced by the Stage 4 high-risk route gate. */
export const STAGE4_A11Y_RULES = [
  "landmark-one-main",
  "region",
  "link-name",
  "image-alt",
  "input-image-alt",
  "select-name",
  "aria-command-name",
  "aria-toggle-field-name",
]

/**
 * WCAG 2.0-2.2 A/AA plus axe best practices. Naming the tags turns on rules
 * that axe disables by default, such as target-size (WCAG 2.5.8, AX-16).
 */
export const UX_RATCHET_AXE_TAGS = [
  "wcag2a",
  "wcag2aa",
  "wcag21a",
  "wcag21aa",
  "wcag22aa",
  "best-practice",
]

export const BLOCKING_IMPACTS: readonly AxeImpact[] = ["serious", "critical"]

export function formatAxeViolations(routePath: string, violations: AxeViolation[]): string {
  if (violations.length === 0) return `${routePath}: no violations`
  return [
    `${routePath}: ${violations.length} serious/critical Axe violations`,
    ...violations.map((violation) => {
      const nodes = violation.nodes
        .slice(0, 3)
        .map((node) => node.target.join(" "))
        .join(" | ")
      return `- ${violation.id} [${violation.impact ?? "unknown"}] -> ${nodes}`
    }),
  ].join("\n")
}

export type ScanA11yOptions = {
  /** CSS selectors to scan; the whole page when omitted. */
  include?: string[]
  /** CSS selectors to skip. */
  exclude?: string[]
  /** Restrict to these axe tags (e.g. UX_RATCHET_AXE_TAGS); axe defaults when omitted. */
  tags?: string[]
  disableRules?: string[]
  /** Upper bound for waiting on finite CSS animations and transitions. */
  animationSettleMs?: number
  settleTimeoutMs?: number
}

/**
 * Wait for finite animations and transitions (antd popover fade-ins, drawer
 * slides) to finish so colour-contrast reads final colours. Infinite
 * animations such as spinners are ignored; the wait is capped.
 */
export async function waitForFiniteAnimations(page: Page, capMs = 3_000): Promise<void> {
  await page
    .evaluate(async (cap) => {
      const finite = document.getAnimations().filter((animation) => {
        const endTime = animation.effect?.getComputedTiming().endTime
        return typeof endTime === "number" && Number.isFinite(endTime)
      })
      await Promise.race([
        Promise.allSettled(finite.map((animation) => animation.finished)),
        new Promise((resolve) => setTimeout(resolve, cap)),
      ])
    }, capMs)
    .catch(() => {})
}

/**
 * Run axe on the current page and return every violation (all impacts).
 * Retries once when the page navigates during the scan ("Execution context
 * was destroyed"), the same recovery as the Stage 4 gate.
 */
export async function scanA11y(page: Page, options: ScanA11yOptions = {}): Promise<AxeViolation[]> {
  const settleTimeoutMs = options.settleTimeoutMs ?? 30_000
  let lastError: unknown
  for (let attempt = 0; attempt < 2; attempt += 1) {
    try {
      await waitForVisualSettle(page, settleTimeoutMs)
      await waitForFiniteAnimations(page, options.animationSettleMs)
      let builder = new AxeBuilder({ page })
      if (options.tags?.length) builder = builder.withTags(options.tags)
      for (const selector of options.include ?? []) builder = builder.include(selector)
      for (const selector of options.exclude ?? []) builder = builder.exclude(selector)
      if (options.disableRules?.length) builder = builder.disableRules(options.disableRules)
      const results = await builder.analyze()
      return results.violations
    } catch (error) {
      lastError = error
      const message = error instanceof Error ? error.message : String(error)
      if (!message.includes("Execution context was destroyed") || attempt === 1) {
        throw error
      }
      await waitForAppShell(page, settleTimeoutMs)
    }
  }
  throw lastError instanceof Error ? lastError : new Error("Axe scan failed")
}

export function blockingViolations(
  violations: AxeViolation[],
  impacts: readonly AxeImpact[] = BLOCKING_IMPACTS
): AxeViolation[] {
  return violations.filter((violation) => violation.impact != null && impacts.includes(violation.impact))
}

/**
 * Generated ids (React useId, antd rc-* ids) change between renders, so they
 * are masked in printed selectors.
 */
export function normaliseAxeTarget(target: string): string {
  return target
    .replace(/#(?:rc[_-][\w-]+|_r_[\w-]*|:r[\w]*:|r\d+)/g, "#<generated>")
    .replace(/\\:r[\w]*\\:/g, "<generated>")
}

/**
 * Per-rule node counts with sample selectors, keyed by axe rule id.
 * `includeHtml` appends each node's (truncated) markup, for attachments.
 */
export function summariseA11yViolations(
  violations: AxeViolation[],
  { sampleSize = 5, includeHtml = false }: { sampleSize?: number; includeHtml?: boolean } = {}
): Map<string, RatchetObservation & { impact: string }> {
  const summary = new Map<string, RatchetObservation & { impact: string }>()
  for (const violation of violations) {
    const targets = violation.nodes.map((node) => {
      const target = normaliseAxeTarget(node.target.join(" "))
      return includeHtml ? `${target}  ${node.html.replace(/\s+/g, " ").slice(0, 160)}` : target
    })
    summary.set(violation.id, {
      count: violation.nodes.length,
      impact: violation.impact ?? "unknown",
      details: [
        `${violation.impact ?? "unknown"}: ${violation.help}`,
        ...targets.slice(0, sampleSize),
        ...(targets.length > sampleSize ? [`… ${targets.length - sampleSize} more`] : []),
      ],
    })
  }
  return summary
}
