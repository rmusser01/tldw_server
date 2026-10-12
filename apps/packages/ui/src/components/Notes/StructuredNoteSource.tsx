import React from "react";
import type { OnMount } from "@monaco-editor/react";
import { useTranslation } from "react-i18next";
import { loadLocalMonaco } from "@/services/monaco-loader";
import type {
  DocumentRequest,
  FormatResult,
} from "@/services/document-editing/types";
import {
  MonacoFormatBridge,
  sourceProfile,
  sameKey,
  nativePrefix,
  modelOffset,
} from "./monaco-format-bridge";

type Editor = Parameters<OnMount>[0];
export interface StructuredNoteSourceHandle {
  captureRequest: (request: DocumentRequest) => boolean;
  applyFormat: (
    result: FormatResult,
    request: DocumentRequest,
    editable: boolean,
  ) => boolean;
  navigate: (start: number, end: number) => void;
}
type Props = {
  value: string;
  session: string;
  editable: boolean;
  onChange: (value: string, cursor: number) => void;
  onReady: (ready: boolean) => void;
  onCompositionChange: (composing: boolean) => void;
};

/** Native model/history lives for the document session, never a typing revision. */
export const StructuredNoteSource = React.forwardRef<
  StructuredNoteSourceHandle,
  Props
>((props, ref) => {
  const { t } = useTranslation("option");
  const translate = React.useRef(t);
  translate.current = t;
  const container = React.useRef<HTMLDivElement>(null);
  const current = React.useRef(props);
  current.current = props;
  const editorRef = React.useRef<Editor | null>(null);
  const bridge = React.useRef<MonacoFormatBridge | null>(null);
  const externalChange = React.useRef(false);
  const [failed, setFailed] = React.useState(false);
  const unrepresented = sourceProfile(props.value) === null;
  React.useImperativeHandle(
    ref,
    () => ({
      captureRequest(request) {
        const editor = editorRef.current;
        if (
          !current.current.editable ||
          !editor ||
          editor.inComposition ||
          editor.getRawOptions().readOnly ||
          !editor.getDomNode()?.isConnected ||
          !editor.getModel() ||
          nativePrefix(editor.getModel()!) === null ||
          editor.getModel()?.getValue(undefined, true) !== request.text
        )
          return false;
        editor.focus();
        bridge.current?.captureRequest(request);
        return bridge.current !== null;
      },
      applyFormat(result, request, editable) {
        const editor = editorRef.current;
        if (
          !current.current.editable ||
          !editable ||
          !editor ||
          editor.inComposition ||
          editor.getRawOptions().readOnly ||
          !editor.getDomNode()?.isConnected ||
          !editor.getModel() ||
          nativePrefix(editor.getModel()!) === null ||
          !sameKey(result.key, request.key) ||
          editor.getModel()?.getValue(undefined, true) !== request.text
        )
          return false;
        return (
          result.state === "unchanged" ||
          bridge.current?.applyFormat(result, request.key, editable) === true
        );
      },
      navigate(start, end) {
        const editor = editorRef.current,
          model = editor?.getModel();
        if (!editor || !model || model.isDisposed()) return;
        const prefix = nativePrefix(model);
        if (prefix === null) return;
        const a = model.getPositionAt(
            modelOffset(start, prefix, model.getValueLength()),
          ),
          b = model.getPositionAt(
            modelOffset(end, prefix, model.getValueLength()),
          );
        editor.setSelection({
          startLineNumber: a.lineNumber,
          startColumn: a.column,
          endLineNumber: b.lineNumber,
          endColumn: b.column,
        });
        editor.revealPositionInCenterIfOutsideViewport(a);
        editor.focus();
      },
    }),
    [],
  );

  React.useEffect(() => {
    let cancelled = false;
    let cleanup = () => {};
    const initial = current.current;
    initial.onReady(false);
    setFailed(false);
    if (sourceProfile(initial.value) === null) return;
    void loadLocalMonaco()
      .then((monaco) => {
        if (cancelled || !container.current) return;
        const latest = current.current;
        if (sourceProfile(latest.value) === null) return;
        const model = monaco.editor.createModel(latest.value, "plaintext");
        model.setEOL(
          sourceProfile(latest.value) === "\r\n"
            ? monaco.editor.EndOfLineSequence.CRLF
            : monaco.editor.EndOfLineSequence.LF,
        );
        if (
          nativePrefix(model) === null ||
          model.getValue(undefined, true) !== latest.value
        ) {
          model.dispose();
          setFailed(true);
          return;
        }
        const nativeTheme = () =>
          document.documentElement.classList.contains("dark") ||
          document.documentElement.classList.contains("theme-dark")
            ? "vs-dark"
            : "vs";
        const editor = monaco.editor.create(container.current, {
          theme: nativeTheme(),
          model,
          ariaLabel: translate.current("notesSearch.editorAriaLabel", {
            defaultValue: "Note content",
          }),
          automaticLayout: true,
          readOnly: !latest.editable,
          minimap: { enabled: false },
          scrollBeyondLastLine: false,
          wordWrap: "off",
          links: false,
          contextmenu: false,
          quickSuggestions: false,
          suggestOnTriggerCharacters: false,
          formatOnType: false,
          formatOnPaste: false,
          autoIndent: "none",
          autoClosingBrackets: "never",
          autoClosingQuotes: "never",
          autoSurround: "never",
          trimAutoWhitespace: false,
          unusualLineTerminators: "off",
          editContext: false,
        });
        const themeChanged = new MutationObserver(() =>
          monaco.editor.setTheme(nativeTheme()),
        );
        themeChanged.observe(document.documentElement, {
          attributes: true,
          attributeFilter: ["class"],
        });
        editorRef.current = editor;
        const ownedBridge = new MonacoFormatBridge(editor, monaco);
        bridge.current = ownedBridge;
        const changed = editor.onDidChangeModelContent(() => {
          if (
            externalChange.current ||
            !current.current.editable ||
            model.isDisposed()
          )
            return;
          const position = editor.getPosition();
          const prefix = nativePrefix(model);
          if (prefix === null) {
            setFailed(true);
            current.current.onReady(false);
            return;
          }
          current.current.onChange(
            model.getValue(undefined, true),
            (position ? model.getOffsetAt(position) : model.getValueLength()) +
              prefix,
          );
        });
        const started = editor.onDidCompositionStart(() =>
          current.current.onCompositionChange(true),
        );
        const ended = editor.onDidCompositionEnd(() =>
          current.current.onCompositionChange(false),
        );
        latest.onReady(true);
        cleanup = () => {
          current.current.onReady(false);
          current.current.onCompositionChange(false);
          bridge.current = null;
          editorRef.current = null;
          themeChanged.disconnect();
          ownedBridge.dispose();
          changed.dispose();
          started.dispose();
          ended.dispose();
          editor.dispose();
          model.dispose();
        };
      })
      .catch(() => {
        if (!cancelled) setFailed(true);
      });
    return () => {
      cancelled = true;
      cleanup();
    };
    // Losing edit authority retires a pending loader and its native session as well.
  }, [props.session, props.editable]);

  React.useLayoutEffect(() => {
    const editor = editorRef.current,
      model = editor?.getModel();
    if (!editor || !model || model.isDisposed() || unrepresented) return;
    if (model.getValue(undefined, true) === props.value) return;
    const prefix = nativePrefix(model);
    if (
      prefix === null ||
      (prefix === 1 && !props.value.startsWith("\ufeff"))
    ) {
      setFailed(true);
      current.current.onReady(false);
      return;
    }
    // AI/external canonical changes are native edits; formatting never uses setValue.
    externalChange.current = true;
    try {
      editor.pushUndoStop();
      editor.executeEdits("notes.canonical", [
        { range: model.getFullModelRange(), text: props.value.slice(prefix) },
      ]);
      editor.pushUndoStop();
    } finally {
      externalChange.current = false;
    }
  }, [props.value, unrepresented]);
  const reason = unrepresented || failed;
  return (
    <div className="flex min-h-0 flex-1 flex-col gap-2">
      {reason ? (
        <>
          <p role="status" className="text-xs text-text-muted">
            {t("notesDocument.sourceOnly", {
              defaultValue:
                "Source retained exactly. Body editing is unavailable for this representation; title and Save remain available.",
            })}
          </p>
          <pre
            tabIndex={0}
            role="textbox"
            aria-label={t("notesSearch.editorAriaLabel", {
              defaultValue: "Note content",
            })}
            aria-readonly="true"
            className="overflow-auto whitespace-pre-wrap break-words rounded border border-border bg-surface2 p-4 text-sm font-mono"
          >
            {props.value}
          </pre>
        </>
      ) : null}
      <div
        ref={container}
        className={reason ? "hidden" : "h-80 min-h-0 border border-border"}
        data-testid="notes-native-source"
      />
    </div>
  );
});
StructuredNoteSource.displayName = "StructuredNoteSource";
