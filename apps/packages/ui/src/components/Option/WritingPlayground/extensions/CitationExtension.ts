import { Mark, mergeAttributes } from "@tiptap/core"

export const CitationExtension = Mark.create({
  name: "citation",
  inclusive: false,

  addAttributes() {
    return {
      sourceId: { default: null },
      sourceTitle: { default: null },
      sourceType: { default: null },
    }
  },

  parseHTML() {
    return [{ tag: "span[data-citation]" }]
  },

  renderHTML({ HTMLAttributes }) {
    return [
      "span",
      mergeAttributes(HTMLAttributes, {
        "data-citation": "",
        class: "citation-mark",
        title: HTMLAttributes.sourceTitle || "Citation",
      }),
      0,
    ]
  },
})
