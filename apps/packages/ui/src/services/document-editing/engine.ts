import { analyzeJson } from "./json-document";
import { analyzeYaml } from "./yaml-document";
import { scalarOffset } from "./source-positions";
import {
  createCheckResult,
  createFormatResult,
  type CheckResult,
  type Diagnostic,
  type DocumentRequest,
  type FormatResult,
} from "./types";

const MAX_BYTES = 1024 * 1024;
const MAX_DEPTH = 100;
const MAX_DIAGNOSTICS = 100;
const DEADLINE_MS = 2000;
class ResourceLimit extends Error {}

/** Internal parser boundary; messages are fixed strings, never parser snippets. */
export interface ParseContext {
  checkpoint(depth?: number): void;
  report(
    code: string,
    message: string,
    start: number,
    end: number,
    severity?: Diagnostic["severity"],
  ): void;
}
export interface Analysis {
  signature: string;
  format(): string;
}

function inspect(
  request: DocumentRequest,
  deadline: number,
): { check: CheckResult; analysis?: Analysis } {
  const diagnostics: Diagnostic[] = [];
  let count = 0;
  const context: ParseContext = {
    checkpoint(depth = 0) {
      if (depth > MAX_DEPTH || performance.now() >= deadline)
        throw new ResourceLimit();
    },
    report(code, message, start, end, severity = "error") {
      count++;
      if (diagnostics.length >= MAX_DIAGNOSTICS) return;
      diagnostics.push({
        code,
        message,
        severity,
        start: scalarOffset(
          request.text,
          Math.max(0, Math.min(start, request.text.length)),
        ),
        end: scalarOffset(
          request.text,
          Math.max(start, Math.min(end, request.text.length)),
        ),
      });
    },
  };
  const result = (state: CheckResult["state"], analysis?: Analysis) => {
    const truncated = count > MAX_DIAGNOSTICS;
    const complete = state === "checked" && !truncated;
    return {
      analysis,
      check: createCheckResult(request, {
        state,
        complete,
        diagnostics,
        truncated,
        format_allowed: complete && diagnostics.length === 0,
      }),
    };
  };
  try {
    if (
      request.is_excerpt ||
      !["json", "jsonl", "yaml"].includes(request.language ?? "")
    ) {
      context.report(
        "unsupported",
        "Whole-document checking is unavailable for this mode or excerpt.",
        0,
        0,
      );
      return result("unsupported");
    }
    if (new TextEncoder().encode(request.text).byteLength > MAX_BYTES)
      throw new ResourceLimit();
    context.checkpoint();
    const analysis =
      request.language === "yaml"
        ? analyzeYaml(request, context)
        : analyzeJson(request, context);
    context.checkpoint();
    return result(
      diagnostics.some((d) => d.code === "unsupported")
        ? "unsupported"
        : "checked",
      analysis,
    );
  } catch (error) {
    const state =
      error instanceof ResourceLimit ? "resource_limited" : "failed";
    diagnostics.length = 0;
    count = 0;
    context.report(
      state,
      state === "resource_limited"
        ? "Document exceeds processing limits."
        : "Document check failed; the draft is unchanged.",
      0,
      0,
    );
    return result(state);
  }
}

export function checkDocument(request: DocumentRequest): CheckResult {
  return inspect(request, performance.now() + DEADLINE_MS).check;
}

/** Return a proposal only after an authoritative check and source invariant check. */
export function formatDocument(request: DocumentRequest): FormatResult {
  const deadline = performance.now() + DEADLINE_MS;
  const { check, analysis } = inspect(request, deadline);
  const result = (
    state: FormatResult["state"],
    replacement: string | null = null,
    authoritative = check,
  ) =>
    createFormatResult({
      key: request.key,
      state,
      replacement,
      check: authoritative,
    });
  if (!check.format_allowed || !analysis)
    return result(
      check.state === "resource_limited" || check.state === "failed"
        ? check.state
        : "blocked",
    );
  try {
    const replacement = analysis.format();
    if (
      new TextEncoder().encode(replacement).byteLength > MAX_BYTES ||
      performance.now() >= deadline
    )
      throw new ResourceLimit();
    const verified = inspect({ ...request, text: replacement }, deadline);
    if (verified.check.state === "resource_limited") throw new ResourceLimit();
    if (
      !verified.check.format_allowed ||
      verified.analysis?.signature !== analysis.signature
    )
      throw new Error("Preservation invariant failed");
    return replacement === request.text
      ? result("unchanged")
      : result("formatted", replacement);
  } catch (error) {
    const state =
      error instanceof ResourceLimit ? "resource_limited" : "failed";
    return result(
      state,
      null,
      createCheckResult(request, {
        state,
        complete: false,
        truncated: false,
        format_allowed: false,
        diagnostics: [
          {
            code: state,
            severity: "error",
            message:
              state === "failed"
                ? "Formatting could not preserve this source; the draft is unchanged."
                : "Formatting exceeds processing limits.",
            start: 0,
            end: 0,
          },
        ],
      }),
    );
  }
}
