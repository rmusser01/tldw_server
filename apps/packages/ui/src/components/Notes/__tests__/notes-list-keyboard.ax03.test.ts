import { afterEach, describe, expect, it } from "vitest"

import { focusNotesEditorBody } from "../notes-list-keyboard"

// AX-03: focusing the editor reports whether focus actually landed, so the
// caller can retry while the editor is still disabled.

const mount = (html: string) => {
  document.body.innerHTML = html
}

const ref = <T extends HTMLElement>(selector: string) => ({
  current: document.querySelector<T>(selector)
})

describe("focusNotesEditorBody", () => {
  afterEach(() => {
    document.body.innerHTML = ""
  })

  it("skips a rich editor that is not editable yet and uses the editor region", () => {
    mount(`
      <section id="editor-region" tabindex="-1">
        <div id="rich" contenteditable="false"></div>
      </section>
    `)

    expect(
      focusNotesEditorBody({
        richEditorRef: ref<HTMLElement>("#rich"),
        editorRegionId: "editor-region"
      })
    ).toBe(true)
    expect(document.activeElement?.id).toBe("editor-region")
  })

  it("falls through to the Markdown textarea when the rich editor cannot take focus", () => {
    mount(`
      <section id="editor-region">
        <div id="rich"></div>
        <textarea id="body"></textarea>
      </section>
    `)

    expect(
      focusNotesEditorBody({
        richEditorRef: ref<HTMLElement>("#rich"),
        contentTextareaRef: ref<HTMLTextAreaElement>("#body"),
        editorRegionId: "editor-region"
      })
    ).toBe(true)
    expect(document.activeElement?.id).toBe("body")
  })

  it("reports failure when nothing in the editor took focus", () => {
    mount(`
      <section id="editor-region">
        <div id="rich"></div>
      </section>
    `)

    expect(
      focusNotesEditorBody({
        richEditorRef: ref<HTMLElement>("#rich"),
        editorRegionId: "editor-region"
      })
    ).toBe(false)
  })
})
