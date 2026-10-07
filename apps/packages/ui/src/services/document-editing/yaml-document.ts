import {
  Composer,
  CST,
  Lexer,
  Parser,
  isAlias,
  isMap,
  isScalar,
  isSeq,
  type Document,
  type ParsedNode as Node,
  type Scalar,
} from "yaml";
import type { Analysis, ParseContext } from "./engine";
import type { DocumentRequest } from "./types";

const TAG = "tag:yaml.org,2002:";
const SCALARS = new Set(["str", "null", "bool", "int", "float", "merge"]);

/** Exact decimal identity, without converting coefficients/exponents to Number. */
function decimalIdentity(source: string): string {
  const text = source.replaceAll("_", "").toLowerCase();
  if (text.includes(".nan")) return "nan";
  if (text.includes(".inf")) return text.startsWith("-") ? "-inf" : "inf";
  const [mantissa, exponent = "0"] = text.split("e");
  const [whole, fraction = ""] = mantissa.replace(/^[-+]/, "").split(".");
  let coefficient = (whole + fraction).replace(/^0+/, "");
  if (!coefficient) return "0";
  const zeros = coefficient.length - coefficient.replace(/0+$/, "").length;
  coefficient = coefficient.slice(0, coefficient.length - zeros);
  return `${text.startsWith("-") ? "-" : ""}${coefficient}e${BigInt(exponent) - BigInt(fraction.length) + BigInt(zeros)}`;
}

/** Symbolic CST signature: retain graph, scalar bytes, comments and line breaks. */
function sourceShape(value: unknown): unknown {
  if (Array.isArray(value))
    return value.filter((v) => v?.type !== "space").map(sourceShape);
  if (value && typeof value === "object") {
    return Object.fromEntries(
      Object.entries(value)
        .filter(([key]) => key !== "offset" && key !== "indent")
        .map(([key, child]) => [key, sourceShape(child)]),
    );
  }
  return value;
}

export function analyzeYaml(
  request: DocumentRequest,
  context: ParseContext,
): Analysis {
  const text = request.text;
  const parser = new Parser(),
    lexer = new Lexer();
  const tokens: CST.Token[] = [];
  // Parser.stack is a public CST API. Check after EACH lexical step, before composition.
  for (const lexeme of lexer.lex(text)) {
    context.checkpoint();
    tokens.push(...parser.next(lexeme));
    context.checkpoint(parser.stack.filter(CST.isCollection).length);
  }
  tokens.push(...parser.end());
  const composer = new Composer({
    version: "1.2",
    keepSourceTokens: true,
    intAsBigInt: true,
    uniqueKeys: false,
    merge: true,
    strict: true,
    prettyErrors: false,
    logLevel: "silent",
  });
  const documents: Document.Parsed[] = [];
  const documentVersions = new Map<number, string>();
  let nextVersion = "1.2";
  for (const token of tokens) {
    context.checkpoint();
    if (token.type === "directive") {
      const directive = /^%YAML[ \t]+([^ \t]+)/.exec(token.source);
      if (directive) nextVersion = directive[1];
    }
    if (token.type === "document") {
      documentVersions.set(token.offset, nextVersion);
      nextVersion = "1.2";
    }
    documents.push(...composer.next(token));
  }
  documents.push(...composer.end(false, text.length));
  const unsupported = (
    start: number,
    end: number,
    message = "This YAML construct is not qualified for source-preserving formatting.",
  ) => context.report("unsupported", message, start, end);
  const versions = new Set<string>();
  const info = composer.streamInfo();
  const reportErrors = (
    errors: typeof info.errors,
    warnings: typeof info.warnings,
  ) => {
    for (const error of errors) {
      context.report(
        error.code === "TAG_RESOLVE_FAILED" ? "unsupported" : "syntax",
        error.code === "TAG_RESOLVE_FAILED"
          ? "This YAML tag/value combination is not qualified."
          : "Invalid YAML syntax.",
        error.pos[0],
        error.pos[1],
      );
    }
    for (const warning of warnings) unsupported(warning.pos[0], warning.pos[1]);
  };
  reportErrors(info.errors, info.warnings);
  for (const doc of documents) {
    context.checkpoint();
    reportErrors(doc.errors, doc.warnings);
    const version = documentVersions.get(doc.range[0]) ?? "1.2";
    versions.add(version);
    if (versions.size > 1 || version !== doc.directives.yaml.version) {
      unsupported(
        doc.range[0],
        doc.range[0],
        "Mixed or inherited YAML versions are not qualified for formatting.",
      );
      continue;
    }
    const anchors = new Map<string, Node>();
    const tagOf = (node: Scalar, key: boolean) =>
      node.tag ??
      (node.type === "PLAIN"
        ? doc.schema.tags.find(
            (t) =>
              (t.default === true || (key && t.default === "key")) &&
              t.test?.test(node.source ?? ""),
          )?.tag
        : undefined) ??
      TAG + "str";
    const scalarIdentity = (node: Scalar, key: boolean): string | null => {
      const tag = tagOf(node, key),
        name = tag.slice(TAG.length),
        source = node.source ?? "";
      if (
        !tag.startsWith(TAG) ||
        !SCALARS.has(name) ||
        (node.tag &&
          !["str", "merge"].includes(name) &&
          !doc.schema.tags.some((t) => t.tag === tag && t.test?.test(source)))
      ) {
        unsupported(node.range?.[0] ?? 0, node.range?.[1] ?? 0);
        return null;
      }
      if (
        name === "float" &&
        !/^[-+]?\.(?:inf|nan)$/i.test(source) &&
        !/[0-9]/.test(source.split(/[eE]/)[0])
      ) {
        unsupported(
          node.range?.[0] ?? 0,
          node.range?.[1] ?? 0,
          "This candidate YAML 1.1 float spelling is not qualified.",
        );
        return null;
      }
      if (name === "float" && source.includes(":")) {
        unsupported(
          node.range?.[0] ?? 0,
          node.range?.[1] ?? 0,
          "Sexagesimal floating-point keys and values are not qualified.",
        );
        return null;
      }
      if (name === "int") return tag + ":" + String(node.value);
      if (name === "float") return tag + ":" + decimalIdentity(source);
      if (name === "null") return tag;
      if (name === "bool") return tag + ":" + String(node.value);
      return tag + ":" + source;
    };
    const walk = (node: Node | null, key = false, depth = 0): string | null => {
      if (!node) return TAG + "null";
      context.checkpoint(depth);
      if ("anchor" in node && node.anchor) anchors.set(node.anchor, node);
      if (isAlias(node)) {
        if (key)
          unsupported(
            node.range[0],
            node.range[1],
            "Alias mapping keys are not qualified.",
          );
        if (!anchors.has(node.source))
          context.report(
            "syntax",
            "Alias has no preceding anchor.",
            node.range[0],
            node.range[1],
          );
        return null;
      }
      if (isScalar(node)) return scalarIdentity(node, key);
      if (key)
        unsupported(
          node.range[0],
          node.range[1],
          "Complex mapping keys are not qualified.",
        );
      if (node.tag && node.tag !== TAG + (isMap(node) ? "map" : "seq"))
        unsupported(node.range[0], node.range[1]);
      if (isMap(node)) {
        const keys = new Set<string>();
        for (const pair of node.items) {
          const k = pair.key as Node | null,
            v = pair.value as Node | null;
          const identity = walk(k, true, depth + 1);
          if (isScalar(k) && tagOf(k, true) === TAG + "merge") {
            const qualified = (target: Node | null): boolean =>
              isMap(target) ||
              (isAlias(target) && isMap(anchors.get(target.source)));
            if (
              !(
                qualified(v) ||
                (isSeq(v) && v.items.every((item) => qualified(item as Node)))
              )
            )
              unsupported(
                k.range[0],
                k.range[1],
                "Merge target is not a qualified mapping.",
              );
          } else if (identity !== null) {
            if (keys.has(identity))
              context.report(
                "duplicate_key",
                "Repeated YAML mapping key blocks formatting.",
                k?.range?.[0] ?? node.range[0],
                k?.range?.[1] ?? node.range[0],
              );
            keys.add(identity);
          }
          walk(v, false, depth + 1);
        }
      } else if (isSeq(node))
        for (const child of node.items) walk(child as Node, false, depth + 1);
      return null;
    };
    // The lexer/parser already bounded collection depth before composition.
    walk(doc.contents);
  }

  const edits = new Map<number, { length: number; content: string }>();
  const multiline: CST.Token[] = [];
  const leading = (offset: number, indent: number) => {
    const start = text.lastIndexOf("\n", offset - 1) + 1;
    const prefix = text.slice(start, offset);
    if (/^[ ]*$/.test(prefix) && prefix.length !== indent)
      edits.set(start, { length: prefix.length, content: " ".repeat(indent) });
  };
  const separators = (parts: CST.SourceToken[]) => {
    for (let i = 1; i < parts.length; i++) {
      const prev = parts[i - 1],
        part = parts[i];
      if (
        part.type === "space" &&
        ["map-value-ind", "seq-item-ind"].includes(prev.type) &&
        parts[i + 1]?.type !== "newline"
      ) {
        edits.set(part.offset, { length: part.source.length, content: " " });
      }
    }
  };
  const layout = (token: CST.Token | undefined, indent: number) => {
    if (!token) return;
    context.checkpoint();
    if (token.type === "document") {
      layout(token.value, 0);
      return;
    }
    if (token.type === "block-map" || token.type === "block-seq") {
      leading(token.offset, indent);
      if (token.type === "block-seq" && token.indent < indent)
        unsupported(
          token.offset,
          token.offset,
          "Indentless sequences are not qualified for formatting.",
        );
      for (const item of token.items) {
        separators(item.start);
        if (item.sep) separators(item.sep);
        for (const start of item.start)
          if (start.type === "seq-item-ind") leading(start.offset, indent);
        if (item.key) {
          leading(item.key.offset, indent);
          layout(item.key, indent);
        }
        const valueIndent =
          item.value &&
          CST.isCollection(item.value) &&
          item.value.type !== "flow-collection"
            ? indent + 2
            : indent;
        layout(item.value, valueIndent);
      }
    } else if (token.type === "flow-collection") {
      // Keep flow whitespace; only containing block indentation is reformatted.
      for (const item of token.items) {
        layout(item.key ?? undefined, indent);
        layout(item.value, indent);
      }
    } else if (CST.isScalar(token)) {
      if (token.type === "block-scalar" || /[\r\n]/.test(token.source))
        multiline.push(token);
    }
  };
  for (const token of tokens) layout(token, 0);
  for (const token of multiline) {
    const start = text.lastIndexOf("\n", token.offset - 1) + 1;
    if (edits.has(start))
      unsupported(
        token.offset,
        token.offset,
        "Reindenting this multiline scalar requires unqualified token edits.",
      );
  }
  return {
    signature: JSON.stringify(sourceShape(tokens)),
    format: () => {
      const parts: string[] = [];
      let cursor = 0;
      for (const [offset, edit] of [...edits].sort((a, b) => a[0] - b[0])) {
        context.checkpoint();
        parts.push(text.slice(cursor, offset), edit.content);
        cursor = offset + edit.length;
      }
      parts.push(text.slice(cursor));
      return parts.join("");
    },
  };
}
