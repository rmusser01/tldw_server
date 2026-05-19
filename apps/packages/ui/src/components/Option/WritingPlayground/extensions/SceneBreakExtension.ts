import { Node, InputRule } from "@tiptap/core"

export const SceneBreakExtension = Node.create({
  name: "sceneBreak",
  group: "block",
  atom: true,

  parseHTML() {
    return [{ tag: "hr.scene-break" }]
  },

  renderHTML() {
    return ["hr", { class: "scene-break", style: "border: none; margin: 1.5em 0;" }]
  },

  addInputRules() {
    return [
      new InputRule({
        find: /^\*\*\*\s$/,
        handler: ({ state, range, chain }) => {
          chain().deleteRange(range).insertContentAt(range.from, { type: this.name }).run()
        },
      }),
    ]
  },
})
