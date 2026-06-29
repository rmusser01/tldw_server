import type { JSONContent } from "@tiptap/react"

const joinTipTapDocumentBlocks = (blocks: string[]): string => {
  return blocks.join("\n\n").replace(/\n+$/, "")
}

const TIPTAP_BLOCK_CONTAINER_TYPES = new Set([
  "doc",
  "bulletList",
  "orderedList",
  "listItem",
  "blockquote"
])

/** Extract plain text from TipTap JSON document */
export function tipTapJsonToPlainText(json: JSONContent | null | undefined): string {
  if (!json) return ""
  if (json.type === "text") return json.text || ""
  if (json.type === "hardBreak") return "\n"

  const childBlocks = json.content?.map(tipTapJsonToPlainText) || []
  const childText = childBlocks.join("")
  if (TIPTAP_BLOCK_CONTAINER_TYPES.has(json.type || "")) {
    return joinTipTapDocumentBlocks(childBlocks)
  }
  if (json.type === "paragraph") return childText
  if (json.type === "heading") return childText
  if (json.type === "sceneBreak") return "***"
  return childText
}

const plainTextBlockToTipTapInlineContent = (block: string): JSONContent[] => {
  const content: JSONContent[] = []

  block.split("\n").forEach((part, index) => {
    if (index > 0) content.push({ type: "hardBreak" })
    if (part) content.push({ type: "text", text: part })
  })

  return content
}

/** Convert plain text to a minimal TipTap JSON document */
export function plainTextToTipTapJson(text: string): JSONContent {
  if (!text) return { type: "doc", content: [{ type: "paragraph" }] }
  const trimmed = text.replace(/\n+$/, '')
  const blocks = trimmed ? trimmed.split("\n\n") : [""]
  const content: JSONContent[] = blocks.map((block) => {
    if (block.trim() === "***") {
      return { type: "sceneBreak" }
    }
    return {
      type: "paragraph",
      content: plainTextBlockToTipTapInlineContent(block),
    }
  })
  return { type: "doc", content }
}

/** Resolve the TipTap document for a session payload. */
export function resolveTipTapDocument(
  prompt: string,
  promptRich: JSONContent | null | undefined,
): JSONContent {
  if (promptRich && promptRich.type === "doc") {
    return promptRich
  }
  return plainTextToTipTapJson(prompt)
}
