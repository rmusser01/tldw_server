import type { JSONContent } from "@tiptap/react"

const joinTipTapDocumentBlocks = (blocks: string[]): string => {
  let output = ""
  blocks.forEach((block, index) => {
    if (index > 0) {
      const previousBlock = blocks[index - 1]
      output += previousBlock === "" || block === "" ? "\n" : "\n\n"
    }
    output += block
  })
  return output.replace(/\n+$/, "")
}

/** Extract plain text from TipTap JSON document */
export function tipTapJsonToPlainText(json: JSONContent | null | undefined): string {
  if (!json) return ""
  if (json.type === "text") return json.text || ""
  const childText = json.content?.map(tipTapJsonToPlainText).join("") || ""
  if (json.type === "doc") {
    return joinTipTapDocumentBlocks(json.content?.map(tipTapJsonToPlainText) || [])
  }
  if (json.type === "paragraph") return childText
  if (json.type === "heading") return childText
  if (json.type === "sceneBreak") return "***"
  if (json.type === "bulletList" || json.type === "orderedList") return childText
  if (json.type === "listItem") return childText
  if (json.type === "blockquote") return childText
  if (json.type === "hardBreak") return "\n"
  return childText
}

/** Convert plain text to a minimal TipTap JSON document */
export function plainTextToTipTapJson(text: string): JSONContent {
  if (!text) return { type: "doc", content: [{ type: "paragraph" }] }
  const trimmed = text.replace(/\n+$/, '')
  const lines = trimmed ? trimmed.split('\n') : ['']
  const content: JSONContent[] = lines.map((line) => {
    if (line.trim() === "***") {
      return { type: "sceneBreak" }
    }
    return {
      type: "paragraph",
      content: line ? [{ type: "text", text: line }] : [],
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
