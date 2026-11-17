import { readFileSync } from "node:fs"
import { createRequire } from "node:module"
import postcss, { type AtRule, type Declaration, type Rule } from "postcss"

const require = createRequire(import.meta.url)

/** A style rule from tailwind-shared.css with the @media conditions around it. */
export type SharedStyleRule = {
  selectors: string[]
  media: string[]
  declarations: Declaration[]
}

let cachedRules: SharedStyleRule[] | null = null

/** Parse every style rule in the shared stylesheet, keeping @media context. */
export function sharedStylesheetRules(): SharedStyleRule[] {
  if (cachedRules) return cachedRules
  const root = postcss.parse(
    readFileSync(require.resolve("../../assets/tailwind-shared.css"), "utf8")
  )
  const rules: SharedStyleRule[] = []
  root.walkRules((rule: Rule) => {
    const media: string[] = []
    let parent = rule.parent
    while (parent && parent.type !== "root") {
      if (parent.type === "atrule" && (parent as AtRule).name === "media") {
        media.push((parent as AtRule).params)
      }
      parent = parent.parent
    }
    const declarations: Declaration[] = []
    rule.each((node) => {
      if (node.type === "decl") declarations.push(node)
    })
    rules.push({ selectors: rule.selectors, media, declarations })
  })
  cachedRules = rules
  return rules
}

/**
 * Whether `selector` targets `element`, ignoring the given pseudo-classes
 * (for example ":focus-visible"), which jsdom cannot put an element into.
 */
export function selectorTargets(
  selector: string,
  element: Element,
  ignoredPseudoClasses: string[] = []
): boolean {
  let candidate = selector
  for (const pseudo of ignoredPseudoClasses) {
    candidate = candidate.split(pseudo).join("")
  }
  try {
    return element.matches(candidate)
  } catch {
    return false
  }
}

/**
 * Rules whose selector targets `element`, optionally restricted to rules
 * inside an @media query containing `mediaIncludes`.
 */
export function rulesTargeting(
  element: Element,
  options: { ignoredPseudoClasses?: string[]; mediaIncludes?: string } = {}
): Array<SharedStyleRule & { matchedSelector: string }> {
  const matches: Array<SharedStyleRule & { matchedSelector: string }> = []
  for (const rule of sharedStylesheetRules()) {
    if (
      options.mediaIncludes &&
      !rule.media.some((media) => media.includes(options.mediaIncludes!))
    ) {
      continue
    }
    const matchedSelector = rule.selectors.find((selector) =>
      selectorTargets(selector, element, options.ignoredPseudoClasses)
    )
    if (matchedSelector) matches.push({ ...rule, matchedSelector })
  }
  return matches
}

/** Build a detached element from an HTML snippet. */
export function elementFrom(html: string): Element {
  const host = document.createElement("div")
  host.innerHTML = html.trim()
  const element = host.firstElementChild
  if (!element) throw new Error(`No element in ${html}`)
  return element
}
