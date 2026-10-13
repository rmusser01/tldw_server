/** Removable I1 harness. All visible source changes pass through the real Notes handler. */
import React from "react";
import { loader, type OnMount } from "@monaco-editor/react";
import {
  createCheckResult,
  createDocumentRequest,
  createFormatResult,
  type RequestKey,
} from "@/services/document-editing/types";
import { MonacoFormatBridge, sourceProfile } from "./monaco-format-candidate";

type Props = {
  value: string;
  authority: string;
  documentId: string;
  language: string;
  editable: boolean;
  onChange: (event: React.ChangeEvent<HTMLTextAreaElement>) => void;
};
type Editor = Parameters<OnMount>[0];
let generation = 0;
const retired: Array<{ uri: string; disposed: boolean }> = [];
export type QualificationProbe = {
  snapshot: () => {
    source: string;
    selection: number[];
    top: number;
    left: number;
    focused: boolean;
    composing: boolean;
    uri: string;
    models: number;
    retired: typeof retired;
    key: RequestKey;
  };
  capture: (replacement?: string) => void;
  complete: (options?: {
    key?: Partial<RequestKey>;
    editable?: boolean;
  }) => boolean;
  select: (anchor: number, active: number) => void;
  scroll: (top: number, left?: number) => void;
  readOnly: (value: boolean) => void;
};
declare global {
  interface Window {
    __i1Monaco?: QualificationProbe;
  }
}

export function MonacoNotesQualification(props: Props) {
  const container = React.useRef<HTMLDivElement>(null);
  const current = React.useRef(props);
  current.current = props;
  const editorRef = React.useRef<Editor | null>(null);
  const [reason, setReason] = React.useState<string | null>(null);
  const [ready, setReady] = React.useState(false);
  React.useEffect(() => {
    let cancelled = false;
    let cleanup = () => {};
    const initial = current.current;
    if (sourceProfile(initial.value) === null) {
      setReason(
        "Representation refused; original source retained (BOM, mixed newlines or lone CR).",
      );
      return;
    }
    loader.config({
      paths: { vs: new URL("/i1-monaco/vs", window.location.href).href },
    });
    void loader
      .init()
      .then((monaco: Parameters<OnMount>[1]) => {
        if (cancelled || !container.current) return;
        // Same-origin static worker bootstraps also satisfy extension CSP (no blob/eval).
        (
          globalThis as typeof globalThis & {
            MonacoEnvironment: {
              getWorker: (moduleId: string, label: string) => Worker;
            };
          }
        ).MonacoEnvironment = {
          getWorker: (_moduleId, label) =>
            new Worker(
              new URL(
                `/i1-monaco/${["json", "css", "html", "typescript", "javascript"].includes(label) ? (label === "typescript" || label === "javascript" ? "ts" : label) : "editor"}-worker.js`,
                window.location.href,
              ),
            ),
        };
        const model = monaco.editor.createModel(initial.value, "plaintext");
        // Set the explicit source EOL at document load, including newline-free inputs.
        model.setEOL(
          sourceProfile(initial.value) === "\r\n"
            ? monaco.editor.EndOfLineSequence.CRLF
            : monaco.editor.EndOfLineSequence.LF,
        );
        if (model.getValue(undefined, true) !== initial.value) {
          model.dispose();
          setReason(
            "Model round-trip changed the representation; original source retained.",
          );
          return;
        }
        const editor = monaco.editor.create(container.current, {
          model,
          ariaLabel: "Note content",
          automaticLayout: true,
          readOnly: !current.current.editable,
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
        editorRef.current = editor;
        const bridge = new MonacoFormatBridge(editor, monaco);
        const session = ++generation;
        let revision = 0,
          requestId = 0;
        let pending: ReturnType<typeof createFormatResult> | null = null;
        const key = (): RequestKey => ({
          authority_id: current.current.authority,
          document_id: current.current.documentId,
          session_generation: session,
          draft_revision: revision,
          language_revision: 0,
          request_id: requestId,
        });
        const changed = editor.onDidChangeModelContent(() => {
          revision++;
          const value = model.getValue(undefined, true);
          const position = editor.getPosition();
          current.current.onChange({
            target: {
              value,
              selectionStart: position
                ? model.getOffsetAt(position)
                : value.length,
            },
          } as React.ChangeEvent<HTMLTextAreaElement>);
        });
        const probe: QualificationProbe = {
          snapshot: () => {
            const selection = editor.getSelection();
            return {
              source: model.getValue(undefined, true),
              selection: selection
                ? [
                    model.getOffsetAt(selection.getSelectionStart()),
                    model.getOffsetAt(selection.getPosition()),
                  ]
                : [],
              top: editor.getScrollTop(),
              left: editor.getScrollLeft(),
              focused: editor.hasTextFocus(),
              composing: editor.inComposition,
              uri: model.uri.toString(),
              models: monaco.editor.getModels().length,
              retired: [...retired],
              key: key(),
            };
          },
          capture: (replacement = '{\n  "a": 1\n}') => {
            requestId++;
            const text = model.getValue(undefined, true);
            const request = createDocumentRequest({
              key: key(),
              text,
              language: current.current.language,
              newline: sourceProfile(text) ?? "\n",
            });
            bridge.captureRequest(request);
            const check = createCheckResult(request, {
              state: "checked",
              complete: true,
              diagnostics: [],
              truncated: false,
              format_allowed: true,
            });
            pending = createFormatResult({
              key: request.key,
              state: "formatted",
              replacement,
              check,
            });
          },
          complete: (options = {}) =>
            pending !== null &&
            current.current.value === model.getValue(undefined, true) &&
            bridge.applyFormat(
              pending,
              { ...key(), ...options.key },
              current.current.editable && options.editable !== false,
            ),
          select: (anchor, active) => {
            const a = model.getPositionAt(anchor),
              b = model.getPositionAt(active);
            editor.setSelection(
              new monaco.Selection(
                a.lineNumber,
                a.column,
                b.lineNumber,
                b.column,
              ),
            );
          },
          scroll: (top, left = 0) =>
            editor.setScrollPosition({ scrollTop: top, scrollLeft: left }),
          readOnly: (readOnly) => editor.updateOptions({ readOnly }),
        };
        window.__i1Monaco = probe;
        setReady(true);
        cleanup = () => {
          const uri = model.uri.toString();
          pending = null;
          bridge.dispose();
          changed.dispose();
          editor.dispose();
          model.dispose();
          retired.push({ uri, disposed: model.isDisposed() });
          editorRef.current = null;
          if (window.__i1Monaco === probe) delete window.__i1Monaco;
        };
      })
      .catch((error: unknown) => {
        if (!cancelled) setReason(`Local editor unavailable: ${String(error)}`);
      });
    return () => {
      cancelled = true;
      cleanup();
    };
  }, []);
  React.useEffect(() => {
    editorRef.current?.updateOptions({ readOnly: !props.editable });
  }, [props.editable]);
  return (
    <div className="flex flex-col gap-2">
      {reason ? (
        <>
          <p role="status">{reason}</p>
          <pre
            tabIndex={0}
            role="textbox"
            aria-label="Note content"
            aria-readonly="true"
          >
            {props.value}
          </pre>
        </>
      ) : (
        <>
          <button
            type="button"
            disabled={!ready}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              editorRef.current?.focus();
              window.__i1Monaco?.capture();
              window.__i1Monaco?.complete();
            }}
          >
            Format document
          </button>
          <button
            type="button"
            disabled={!ready}
            onMouseDown={(e) => e.preventDefault()}
            onClick={() => {
              editorRef.current?.focus();
              window.__i1Monaco?.complete();
            }}
          >
            Apply captured format
          </button>
          <div ref={container} className="h-80 border border-border" />
        </>
      )}
      <output data-testid="monaco-source" className="sr-only">
        {props.value}
      </output>
    </div>
  );
}
