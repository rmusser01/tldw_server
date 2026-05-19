import { Mark, mergeAttributes } from "@tiptap/core"

const ANNOTATION_COLORS: Record<string, string> = {
  generated: "rgba(147, 51, 234, 0.1)",
  suggestion: "rgba(59, 130, 246, 0.1)",
  feedback: "rgba(250, 204, 21, 0.15)",
}

export const AIAnnotationExtension = Mark.create({
  name: "aiAnnotation",

  addAttributes() {
    return {
      type: { default: "generated" },
      confidence: { default: null },
    }
  },

  parseHTML() {
    return [{ tag: "span[data-ai-annotation]" }]
  },

  renderHTML({ HTMLAttributes }) {
    const bg = ANNOTATION_COLORS[HTMLAttributes.type as string] || ANNOTATION_COLORS.generated
    return [
      "span",
      mergeAttributes(HTMLAttributes, {
        "data-ai-annotation": HTMLAttributes.type,
        style: `background: ${bg}; border-radius: 2px; padding: 0 1px;`,
        title: `AI ${HTMLAttributes.type}${HTMLAttributes.confidence != null ? ` (${Math.round(HTMLAttributes.confidence * 100)}%)` : ""}`,
      }),
      0,
    ]
  },
})
