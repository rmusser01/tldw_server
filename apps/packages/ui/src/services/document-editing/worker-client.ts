import { DocumentCheckScheduler } from "./scheduler";
import {
  createCheckResult,
  createDocumentRequest,
  createFormatResult,
  type CheckResult,
  type Diagnostic,
  type DocumentRequest,
  type FormatResult,
  type RequestKey,
} from "./types";

export const MAX_BYTES = 1024 * 1024;
const MAX_FRAME = 6 * MAX_BYTES + 65536;
const messages = {
  syntax: "Invalid document syntax.",
  blank_record: "Blank JSONL records are not allowed.",
  duplicate_key: "Duplicate mapping key; formatting is blocked.",
  unsupported:
    "This source representation is not qualified for checking or formatting.",
  resource_limited: "Document processing reached its resource limit.",
  failed: "Document engine failed; keep the original draft and retry.",
};
export type Operation = "check" | "format";

export function refusal(
  request: DocumentRequest,
  state: "unsupported" | "resource_limited" | "failed",
): CheckResult {
  return createCheckResult(request, {
    state,
    complete: false,
    diagnostics: [
      {
        code: state,
        severity: "error",
        message: messages[state],
        start: 0,
        end: 0,
      },
    ],
    truncated: false,
    format_allowed: false,
  });
}
function validScalars(text: string): boolean {
  for (let i = 0; i < text.length; i++) {
    const code = text.charCodeAt(i);
    if (code >= 0xd800 && code <= 0xdbff) {
      const low = text.charCodeAt(++i);
      if (!(low >= 0xdc00 && low <= 0xdfff)) return false;
    } else if (code >= 0xdc00 && code <= 0xdfff) return false;
  }
  return true;
}
export function admit(request: DocumentRequest): CheckResult | null {
  if (
    request.is_excerpt ||
    !["json", "jsonl", "yaml"].includes(request.language ?? "")
  )
    return refusal(request, "unsupported");
  if (request.text.length > MAX_BYTES)
    return refusal(request, "resource_limited");
  if (!validScalars(request.text)) return refusal(request, "unsupported");
  if (new TextEncoder().encode(request.text).byteLength > MAX_BYTES)
    return refusal(request, "resource_limited");
  return null;
}
export function encode(value: unknown): ArrayBuffer {
  const bytes = new TextEncoder().encode(JSON.stringify(value));
  if (bytes.byteLength > MAX_FRAME) throw new Error("Frame exceeds limit");
  return bytes.buffer as ArrayBuffer;
}
export function decode(frame: unknown): unknown {
  if (!(frame instanceof ArrayBuffer) || frame.byteLength > MAX_FRAME)
    throw new Error("Invalid frame");
  return JSON.parse(new TextDecoder("utf-8", { fatal: true }).decode(frame));
}
function fields(value: unknown, names: string[]): Record<string, unknown> {
  if (
    !value ||
    typeof value !== "object" ||
    Array.isArray(value) ||
    Object.keys(value).length !== names.length ||
    !names.every((name) => Object.hasOwn(value, name))
  )
    throw new Error("Invalid fields");
  return value as Record<string, unknown>;
}
function key(value: unknown): RequestKey {
  const result = fields(value, [
    "authority_id",
    "document_id",
    "session_generation",
    "draft_revision",
    "language_revision",
    "request_id",
  ]);
  for (const [name, item] of Object.entries(result)) {
    if (["authority_id", "document_id"].includes(name)) {
      if (typeof item !== "string" || item.length > 4096)
        throw new Error("Invalid identity");
    } else if (!Number.isSafeInteger(item) || (item as number) < 0)
      throw new Error("Invalid revision");
  }
  return result as unknown as RequestKey;
}
function sameKey(value: unknown, expected: RequestKey): boolean {
  const actual = key(value);
  return Object.keys(expected).every(
    (name) =>
      actual[name as keyof RequestKey] === expected[name as keyof RequestKey],
  );
}
export function decodeRequest(value: unknown): {
  operation: Operation;
  request: DocumentRequest;
} {
  const outer = fields(value, ["operation", "request"]);
  if (outer.operation !== "check" && outer.operation !== "format")
    throw new Error("Invalid operation");
  const request = fields(outer.request, [
    "key",
    "text",
    "language",
    "newline",
    "has_bom",
    "is_excerpt",
  ]);
  key(request.key);
  if (
    typeof request.text !== "string" ||
    (request.language !== null &&
      (typeof request.language !== "string" ||
        request.language.length > 100)) ||
    !["\n", "\r\n"].includes(request.newline as string) ||
    typeof request.has_bom !== "boolean" ||
    typeof request.is_excerpt !== "boolean"
  )
    throw new Error("Invalid request");
  return {
    operation: outer.operation,
    request: createDocumentRequest(request as unknown as DocumentRequest),
  };
}
function decodeCheck(request: DocumentRequest, value: unknown): CheckResult {
  const check = fields(value, [
    "key",
    "state",
    "complete",
    "diagnostics",
    "truncated",
    "format_allowed",
  ]);
  if (
    !sameKey(check.key, request.key) ||
    !["checked", "unsupported", "resource_limited", "failed"].includes(
      check.state as string,
    ) ||
    [check.complete, check.truncated, check.format_allowed].some(
      (value) => typeof value !== "boolean",
    ) ||
    (check.truncated && check.complete) ||
    !Array.isArray(check.diagnostics) ||
    check.diagnostics.length > 100
  )
    throw new Error("Invalid check");
  const diagnostics = check.diagnostics.map((value) => {
    const item = fields(value, ["code", "severity", "message", "start", "end"]);
    if (
      typeof item.code !== "string" ||
      !Object.hasOwn(messages, item.code) ||
      !["error", "warning"].includes(item.severity as string) ||
      typeof item.message !== "string" ||
      item.message.length > 512
    )
      throw new Error("Invalid diagnostic");
    // Never trust a broken worker's prose to be source-free.
    return {
      ...item,
      message: messages[item.code as keyof typeof messages],
    } as unknown as Diagnostic;
  });
  return createCheckResult(request, {
    ...(check as unknown as CheckResult),
    diagnostics,
  });
}
export function decodeResult(
  request: DocumentRequest,
  operation: Operation,
  value: unknown,
): CheckResult | FormatResult {
  if (operation === "check") return decodeCheck(request, value);
  const result = fields(value, ["key", "state", "replacement", "check"]);
  if (
    !sameKey(result.key, request.key) ||
    ![
      "formatted",
      "unchanged",
      "blocked",
      "resource_limited",
      "failed",
    ].includes(result.state as string)
  )
    throw new Error("Invalid format result");
  if (
    result.replacement !== null &&
    (typeof result.replacement !== "string" ||
      !validScalars(result.replacement) ||
      new TextEncoder().encode(result.replacement).byteLength > MAX_BYTES)
  )
    throw new Error("Invalid replacement");
  return createFormatResult({
    ...(result as unknown as FormatResult),
    check: decodeCheck(request, result.check),
  });
}
function failedFormat(
  request: DocumentRequest,
  check: CheckResult,
): FormatResult {
  return createFormatResult({
    key: request.key,
    state:
      check.state === "resource_limited" || check.state === "failed"
        ? check.state
        : "blocked",
    replacement: null,
    check,
  });
}

/** One app-owned, lazy worker lane. A lease never owns a second parser worker. */
export class DocumentWorkerClient {
  private active: {
    worker: Worker;
    finish: (state: "failed" | "resource_limited") => void;
  } | null = null;
  private lease: DocumentCheckScheduler | null = null;
  private closed = false;

  constructor(
    private readonly createWorker: () => Worker = () =>
      new Worker(new URL("./worker.ts", import.meta.url), { type: "module" }),
  ) {}

  acquire(onResult: (result: CheckResult) => void): DocumentCheckScheduler {
    if (this.closed) throw new Error("Document worker is closed");
    this.lease?.close();
    this.lease = new DocumentCheckScheduler(this, onResult);
    return this.lease;
  }
  check(request: DocumentRequest): Promise<CheckResult> {
    return this.run(request, "check") as Promise<CheckResult>;
  }
  format(request: DocumentRequest): Promise<FormatResult> {
    return this.run(request, "format") as Promise<FormatResult>;
  }

  private run(
    request: DocumentRequest,
    operation: Operation,
  ): Promise<CheckResult | FormatResult> {
    const denied =
      admit(request) ??
      (this.closed || this.active ? refusal(request, "failed") : null);
    if (denied)
      return Promise.resolve(
        operation === "check" ? denied : failedFormat(request, denied),
      );
    return new Promise((resolve) => {
      let worker: Worker | null = null;
      let timer: ReturnType<typeof setTimeout> | undefined;
      let finished = false;
      const finish = (
        state: "failed" | "resource_limited",
        result?: CheckResult | FormatResult,
      ) => {
        if (finished) return;
        finished = true;
        clearTimeout(timer);
        if (worker) {
          worker.onmessage = worker.onerror = worker.onmessageerror = null;
          worker.terminate();
        }
        this.active = null;
        const check = result ? null : refusal(request, state);
        resolve(
          result ??
            (operation === "check" ? check! : failedFormat(request, check!)),
        );
      };
      timer = setTimeout(() => finish("resource_limited"), 2000);
      try {
        decodeRequest({ operation, request });
        const frame = encode({ operation, request });
        worker = this.createWorker();
        this.active = { worker, finish };
        worker.onmessage = (event) => {
          try {
            finish(
              "failed",
              decodeResult(request, operation, decode(event.data)),
            );
          } catch {
            finish("failed");
          }
        };
        worker.onerror = (event) => {
          event.preventDefault();
          finish("failed");
        };
        worker.onmessageerror = () => finish("failed");
        worker.postMessage(frame, [frame]);
      } catch {
        finish("failed");
      }
    });
  }
  cancel(): void {
    this.active?.finish("failed");
  }
  close(): void {
    this.closed = true;
    this.lease?.close();
    this.lease = null;
    this.cancel();
  }
}
