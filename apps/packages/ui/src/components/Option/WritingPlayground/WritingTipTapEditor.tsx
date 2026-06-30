import { useCallback, useEffect, useMemo, useRef } from "react"
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
  getTipTapSelection,
  type WritingEditorAdapter,
  type WritingEditorSelection
} from "./writing-editor-adapter"

const EMPTY_TIPTAP_DOCUMENT: JSONContent = {
  type: "doc",
  content: [{ type: "paragraph" }]
}

export type WritingTipTapEditorProps = {
  content: JSONContent | null
  onContentChange: (json: JSONContent, plainText: string) => void
  onContentApplied?: () => void
  onAdapterReady?: (adapter: WritingEditorAdapter | null) => void
  onSelectionChange?: (selection: WritingEditorSelection) => void
  editable?: boolean
  placeholder?: string
  className?: string
}

export function WritingTipTapEditor({
  content,
  onContentChange,
  onContentApplied,
  onAdapterReady,
  onSelectionChange,
  editable = true,
  placeholder = "Start writing...",
  className,
}: WritingTipTapEditorProps) {
  const editorOriginContentRef = useRef<JSONContent | null>(null)
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
      editorOriginContentRef.current = json
      const plain = tipTapJsonToPlainText(json)
      onContentChange(json, plain)
    },
    [onContentChange],
  )

  const handleSelectionUpdate = useCallback(
    ({ editor }: { editor: Editor }) => {
      onSelectionChange?.(getTipTapSelection(editor))
    },
    [onSelectionChange],
  )

  const editor = useEditor({
    extensions,
    content: content || EMPTY_TIPTAP_DOCUMENT,
    editable,
    immediatelyRender: false,
    onUpdate: handleUpdate,
    onSelectionUpdate: handleSelectionUpdate,
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
    let frame: number | null = null
    const nextContent = content || EMPTY_TIPTAP_DOCUMENT
    if (content && editorOriginContentRef.current === content) {
      editorOriginContentRef.current = null
      return
    }
    const currentJson = JSON.stringify(editor.getJSON())
    const nextJson = JSON.stringify(nextContent)
    if (currentJson !== nextJson) {
      editor.commands.setContent(nextContent, { emitUpdate: false })
      frame = window.requestAnimationFrame(() => {
        onContentApplied?.()
      })
    }
    return () => {
      if (frame !== null) {
        window.cancelAnimationFrame(frame)
      }
    }
  }, [editor, content, onContentApplied])

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
