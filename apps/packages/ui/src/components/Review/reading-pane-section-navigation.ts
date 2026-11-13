import type { ContentSection } from "@/components/Review/SectionNavigator"
import { smoothScrollBehavior } from "@/utils/reduced-motion"

const SECTION_HEADING_SELECTOR = "h1, h2, h3, h4, h5, h6"

const escapeAttributeValue = (value: string): string =>
  value.replace(/\\/g, "\\\\").replace(/"/g, '\\"')

export function scrollSectionIntoView(
  contentRoot: ParentNode | null,
  section: ContentSection
): boolean {
  if (!contentRoot) return false

  const escapedId = escapeAttributeValue(section.id)
  const anchoredTarget = contentRoot.querySelector<HTMLElement>(
    `[data-section-anchor="${escapedId}"]`
  )
  if (anchoredTarget) {
    anchoredTarget.scrollIntoView({ behavior: smoothScrollBehavior(), block: "start" })
    return true
  }

  const trimmedLabel = section.label.trim()
  const headings = contentRoot.querySelectorAll<HTMLElement>(SECTION_HEADING_SELECTOR)
  for (const heading of headings) {
    if (heading.textContent?.trim() === trimmedLabel) {
      heading.scrollIntoView({ behavior: smoothScrollBehavior(), block: "start" })
      return true
    }
  }

  const walker = document.createTreeWalker(contentRoot, NodeFilter.SHOW_TEXT)
  let node: Text | null
  while ((node = walker.nextNode() as Text | null)) {
    if (node.textContent && node.textContent.includes(trimmedLabel.slice(0, 20))) {
      node.parentElement?.scrollIntoView({ behavior: smoothScrollBehavior(), block: "start" })
      return true
    }
  }

  return false
}
