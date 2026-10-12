/** Captured-source admission around one native history transaction. */
import type { OnMount } from "@monaco-editor/react";
import {
  createDocumentRequest,
  type DocumentRequest,
  type FormatResult,
  type RequestKey,
} from "@/services/document-editing/types";

type Editor = Parameters<OnMount>[0];

/** Refuse model-normalized representations before editing; never normalize a draft. */
export function sourceProfile(text: string): "\n" | "\r\n" | null {
  if (/\r(?!\n)/.test(text)) return null;
  const crlf = text.includes("\r\n");
  return crlf && /(?<!\r)\n/.test(text) ? null : crlf ? "\r\n" : "\n";
}

/** Monaco's immutable BOM may sit outside its editable buffer. */
export function nativePrefix(model: {
  getValue: (eol?: undefined, preserveBOM?: boolean) => string;
}): 0 | 1 | null {
  const plain = model.getValue(),
    exact = model.getValue(undefined, true);
  return exact === plain ? 0 : exact === "\ufeff" + plain ? 1 : null;
}
export function modelOffset(
  offset: number,
  prefix: 0 | 1,
  length: number,
): number {
  return Math.max(0, Math.min(length, offset - prefix));
}

export function sameKey(a: RequestKey, b: RequestKey): boolean {
  return (
    a.authority_id === b.authority_id &&
    a.document_id === b.document_id &&
    a.session_generation === b.session_generation &&
    a.draft_revision === b.draft_revision &&
    a.language_revision === b.language_revision &&
    a.request_id === b.request_id
  );
}

// Linear UTF-16 anchor mapping, including literal whitespace and surrogate halves.
function mapOffset(
  source: string,
  replacement: string,
  offset: number,
): number {
  if (offset === source.length) return replacement.length;
  let rank = 0;
  for (let i = 0; i < offset; i++) if (!/\s/.test(source[i])) rank++;
  let target = replacement.length;
  for (let i = 0; i < replacement.length; i++) {
    if (!/\s/.test(replacement[i])) {
      if (rank === 0) {
        target = i;
        break;
      }
      rank--;
    }
  }
  let sourceEnd = offset,
    runStart = target;
  while (sourceEnd < source.length && /\s/.test(source[sourceEnd])) sourceEnd++;
  while (runStart > 0 && /\s/.test(replacement[runStart - 1])) runStart--;
  return target - Math.min(sourceEnd - offset, target - runStart);
}

export class MonacoFormatBridge {
  private request: DocumentRequest | null = null;
  private disposed = false;
  private prefix: 0 | 1 | null = null;
  private readonly model;

  constructor(
    private readonly editor: Editor,
    private readonly monaco: Parameters<OnMount>[1],
  ) {
    this.model = editor.getModel();
  }

  captureRequest(request: DocumentRequest): void {
    this.request = createDocumentRequest(request);
    this.prefix = this.model ? nativePrefix(this.model) : null;
  }
  dispose(): void {
    this.disposed = true;
    this.request = null;
  }

  applyFormat(
    result: FormatResult,
    currentKey: RequestKey,
    editable: boolean,
  ): boolean {
    const { editor, model, request } = this;
    if (
      this.disposed ||
      !request ||
      !model ||
      model.isDisposed() ||
      editor.getModel() !== model ||
      !editor.getDomNode()?.isConnected ||
      !editor.hasTextFocus() ||
      editor.inComposition ||
      !editable ||
      editor.getRawOptions().readOnly ||
      request.is_excerpt ||
      this.prefix === null ||
      nativePrefix(model) !== this.prefix ||
      request.has_bom !== request.text.startsWith("\ufeff") ||
      !sameKey(result.key, currentKey) ||
      !sameKey(request.key, currentKey) ||
      !sameKey(result.check.key, currentKey) ||
      result.state !== "formatted" ||
      result.replacement === null ||
      !result.check.format_allowed ||
      !result.check.complete ||
      result.check.state !== "checked" ||
      result.check.truncated ||
      model.getEOL() !== request.newline ||
      sourceProfile(request.text) === null ||
      sourceProfile(result.replacement) !== sourceProfile(request.text) ||
      model.getValue(undefined, true) !== request.text ||
      result.replacement.startsWith("\ufeff") !== request.has_bom ||
      result.replacement === request.text
    )
      return false;
    const selection = editor.getSelection();
    const top = editor.getScrollTop(),
      left = editor.getScrollLeft();
    const anchor = selection
      ? mapOffset(
          request.text,
          result.replacement,
          model.getOffsetAt(selection.getSelectionStart()) + this.prefix,
        )
      : 0;
    const active = selection
      ? mapOffset(
          request.text,
          result.replacement,
          model.getOffsetAt(selection.getPosition()) + this.prefix,
        )
      : 0;
    editor.pushUndoStop();
    const applied = editor.executeEdits(
      "notes.format",
      [
        {
          range: model.getFullModelRange(),
          text: result.replacement.slice(this.prefix),
        },
      ],
      () => {
        const start = model.getPositionAt(
            modelOffset(anchor, this.prefix!, model.getValueLength()),
          ),
          end = model.getPositionAt(
            modelOffset(active, this.prefix!, model.getValueLength()),
          );
        return selection
          ? [
              new this.monaco.Selection(
                start.lineNumber,
                start.column,
                end.lineNumber,
                end.column,
              ),
            ]
          : null;
      },
    );
    editor.pushUndoStop();
    editor.setScrollPosition({ scrollTop: top, scrollLeft: left });
    if (applied) this.request = null;
    return applied;
  }
}
