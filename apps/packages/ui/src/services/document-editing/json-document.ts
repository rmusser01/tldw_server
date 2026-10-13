import { applyEdits, createScanner, format, visit } from "jsonc-parser";
import type { Analysis, ParseContext } from "./engine";
import type { DocumentRequest } from "./types";

const STRICT = {
  disallowComments: true,
  allowTrailingComma: false,
  allowEmptyContent: false,
};

/** Strict visitor validation; no parsed numeric value is retained or serialized. */
export function analyzeJson(
  request: DocumentRequest,
  context: ParseContext,
): Analysis {
  const records: { source: string; terminator: string; tokens: string[] }[] =
    [];
  const lines = request.language === "jsonl";
  if (lines && (request.has_bom || request.text.startsWith("\ufeff"))) {
    context.report(
      "syntax",
      "JSONL does not permit a byte order mark.",
      0,
      request.text.startsWith("\ufeff") ? 1 : 0,
    );
  }
  let offset = 0;
  do {
    if (lines && offset === request.text.length) break;
    const newline = lines ? request.text.indexOf("\n", offset) : -1;
    const end = newline < 0 ? request.text.length : newline;
    const cr = lines && newline >= 0 && request.text[end - 1] === "\r";
    const source = request.text.slice(offset, cr ? end - 1 : end);
    const terminator = newline < 0 ? "" : cr ? "\r\n" : "\n";
    const tokens: string[] = [];
    if (lines && /^[ \t\r]*$/.test(source))
      context.report(
        "blank_record",
        "JSONL records cannot be blank.",
        offset,
        offset + source.length,
      );
    else {
      const stack: (Set<string> | null)[] = [];
      const begin = (object: boolean) => {
        stack.push(object ? new Set() : null);
        context.checkpoint(stack.length);
      };
      visit(
        source,
        {
          onObjectBegin: () => begin(true),
          onArrayBegin: () => begin(false),
          onObjectEnd: () => {
            stack.pop();
          },
          onArrayEnd: () => {
            stack.pop();
          },
          onObjectProperty: (name, start, length) => {
            context.checkpoint();
            const keys = stack[stack.length - 1];
            if (keys?.has(name))
              context.report(
                "duplicate_key",
                "Repeated object name blocks formatting.",
                offset + start,
                offset + start + length,
                "warning",
              );
            keys?.add(name);
          },
          onLiteralValue: () => context.checkpoint(),
          onError: (_error, start, length) => {
            context.checkpoint();
            context.report(
              "syntax",
              "Invalid strict JSON syntax.",
              offset + start,
              offset + start + length,
            );
          },
        },
        STRICT,
      );
      const scanner = createScanner(source, true);
      for (
        scanner.scan();
        scanner.getTokenOffset() < source.length;
        scanner.scan()
      ) {
        context.checkpoint();
        if (scanner.getTokenError())
          context.report(
            "syntax",
            "Invalid JSON token.",
            offset + scanner.getTokenOffset(),
            offset + scanner.getTokenOffset() + scanner.getTokenLength(),
          );
        tokens.push(
          source.slice(
            scanner.getTokenOffset(),
            scanner.getTokenOffset() + scanner.getTokenLength(),
          ),
        );
      }
    }
    records.push({ source, terminator, tokens });
    offset = end + terminator.length - (cr ? 1 : 0);
    if (newline < 0) break;
  } while (offset < request.text.length);
  return {
    signature: JSON.stringify(records.map((r) => [r.tokens, r.terminator])),
    format: () =>
      records
        .map((record) => {
          context.checkpoint();
          if (lines) return record.tokens.join("") + record.terminator;
          const formatted = applyEdits(
            record.source,
            format(record.source, undefined, {
              tabSize: 2,
              insertSpaces: true,
              eol: request.newline,
              keepLines: false,
            }),
          );
          const hadFinalNewline = /\r?\n$/.test(record.source);
          return (
            formatted.replace(/[ \t\r\n]+$/, "") +
            (hadFinalNewline ? request.newline : "")
          );
        })
        .join(""),
  };
}
