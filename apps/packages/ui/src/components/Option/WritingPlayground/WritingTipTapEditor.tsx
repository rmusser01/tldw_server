import { useCallback, useEffect, useMemo } from "react"
import { EditorContent, useEditor, type Editor, type JSONContent } from "@tiptap/react"
import StarterKit from "@tiptap/starter-kit"
import Placeholder from "@tiptap/extension-placeholder"
import CharacterCount from "@tiptap/extension-character-count"
import { SceneBreakExtension } from "./extensions/SceneBreakExtension"
import { CitationExtension } from "./extensions/CitationExtension"
import { AIAnnotationExtension } from "./extensions/AIAnnotationExtension"
import { tipTapJsonToPlainText } from "./writing-tiptap-utils"
import {
  createTipTapEditorAdapter,
  type WritingEditorAdapter
} from "./writing-editor-adapter"

export type WritingTipTapEditorProps = {
  content: JSONContent | null
  onContentChange: (json: JSONContent, plainText: string) => void
  onAdapterReady?: (adapter: WritingEditorAdapter | null) => void
  editable?: boolean
  placeholder?: string
  className?: string
}

export function WritingTipTapEditor({
  content,
  onContentChange,
  onAdapterReady,
  editable = true,
  placeholder = "Start writing...",
  className,
}: WritingTipTapEditorProps) {
  const extensions = useMemo(
    () => [
      StarterKit.configure({
        heading: { levels: [1, 2, 3] },
      }),
      SceneBreakExtension,
      CitationExtension,
      AIAnnotationExtension,
      Placeholder.configure({ placeholder }),
      CharacterCount,
    ],
    [placeholder],
  )

  const handleUpdate = useCallback(
    ({ editor }: { editor: Editor }) => {
      const json = editor.getJSON() as JSONContent
      const plain = tipTapJsonToPlainText(json)
      onContentChange(json, plain)
    },
    [onContentChange],
  )

  const editor = useEditor({
    extensions,
    content: content || { type: "doc", content: [{ type: "paragraph" }] },
    editable,
    onUpdate: handleUpdate,
  })

  const adapter = useMemo(
    () => createTipTapEditorAdapter(editor),
    [editor]
  )

  useEffect(() => {
    onAdapterReady?.(adapter)
    return () => {
      onAdapterReady?.(null)
    }
  }, [adapter, onAdapterReady])

  useEffect(() => {
    if (!editor) return
    const nextContent = content || { type: "doc", content: [{ type: "paragraph" }] }
    const currentJson = JSON.stringify(editor.getJSON())
    const nextJson = JSON.stringify(nextContent)
    if (currentJson !== nextJson) {
      editor.commands.setContent(nextContent, { emitUpdate: false })
    }
  }, [editor, content])

  // Sync editable state
  useEffect(() => {
    if (editor) {
      editor.setEditable(editable)
    }
  }, [editor, editable])

  if (!editor) return null

  return (
    <div className={className}>
      <EditorContent
        editor={editor}
        className="prose prose-sm max-w-none min-h-[300px] focus:outline-none p-4"
      />
    </div>
  )
}

export default WritingTipTapEditor
