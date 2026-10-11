/** Pure wire contracts for local structured-document workers. */
export interface RequestKey {
  readonly authority_id: string;
  readonly document_id: string;
  readonly session_generation: number;
  readonly draft_revision: number;
  readonly language_revision: number;
  readonly request_id: number;
}
export interface DocumentRequest {
  readonly key: RequestKey;
  readonly text: string;
  readonly language: string | null;
  readonly newline: "\n" | "\r\n";
  readonly has_bom: boolean;
  readonly is_excerpt: boolean;
}
export interface Diagnostic {
  readonly code: string;
  readonly severity: "error" | "warning";
  readonly message: string;
  readonly start: number;
  readonly end: number;
}
export interface CheckResult {
  readonly key: RequestKey;
  readonly state: "checked" | "unsupported" | "resource_limited" | "failed";
  readonly complete: boolean;
  readonly diagnostics: readonly Diagnostic[];
  readonly truncated: boolean;
  readonly format_allowed: boolean;
}
export interface FormatResult {
  readonly key: RequestKey;
  readonly state:
    | "formatted"
    | "unchanged"
    | "blocked"
    | "resource_limited"
    | "failed";
  readonly replacement: string | null;
  readonly check: CheckResult;
}

export function createDocumentRequest(
  fields: Pick<DocumentRequest, "key" | "text" | "language"> &
    Partial<Pick<DocumentRequest, "newline" | "has_bom" | "is_excerpt">>,
): DocumentRequest {
  return Object.freeze({
    newline: "\n",
    has_bom: false,
    is_excerpt: false,
    ...fields,
    key: Object.freeze({ ...fields.key }),
  });
}
export function createDiagnostic(fields: Diagnostic): Diagnostic {
  if (
    !Number.isSafeInteger(fields.start) ||
    !Number.isSafeInteger(fields.end) ||
    fields.start < 0 ||
    fields.end < fields.start
  ) {
    throw new Error(
      "Diagnostic ranges must be ordered nonnegative scalar offsets",
    );
  }
  return Object.freeze({ ...fields });
}
export function createCheckResult(
  request: DocumentRequest,
  fields: Omit<CheckResult, "key">,
): CheckResult {
  if (
    request.is_excerpt &&
    (fields.complete || fields.state === "checked" || fields.format_allowed)
  ) {
    throw new Error("Excerpts cannot report successful whole-document checks");
  }
  const diagnostics = Object.freeze(fields.diagnostics.map(createDiagnostic));
  const requiredLength = diagnostics.reduce(
    (end, d) => Math.max(end, d.end),
    0,
  );
  let sourceLength = 0;
  let offset = 0;
  // Prove only the furthest endpoint; zero-range refusals never scan source.
  while (sourceLength < requiredLength && offset < request.text.length) {
    offset += request.text.codePointAt(offset)! > 0xffff ? 2 : 1;
    sourceLength++;
  }
  if (sourceLength < requiredLength)
    throw new Error("Diagnostic range exceeds source snapshot");
  const result = Object.freeze({
    ...fields,
    key: Object.freeze({ ...request.key }),
    diagnostics,
  });
  requireQualifiedCheck(result);
  return result;
}
// Also checks reconstructed worker messages used to construct a format result.
function requireQualifiedCheck(check: CheckResult): void {
  if (check.complete && check.state !== "checked")
    throw new Error("Only checked results can be complete");
  if (
    check.format_allowed &&
    (check.state !== "checked" ||
      !check.complete ||
      check.truncated ||
      check.diagnostics.some(
        (d) => d.severity === "error" || d.code === "duplicate_key",
      ))
  ) {
    throw new Error("Formatting requires a complete qualified check");
  }
}

export function createFormatResult(fields: FormatResult): FormatResult {
  requireQualifiedCheck(fields.check);
  const key = fields.key,
    checkedKey = fields.check.key;
  if (
    key.authority_id !== checkedKey.authority_id ||
    key.document_id !== checkedKey.document_id ||
    key.session_generation !== checkedKey.session_generation ||
    key.draft_revision !== checkedKey.draft_revision ||
    key.language_revision !== checkedKey.language_revision ||
    key.request_id !== checkedKey.request_id
  ) {
    throw new Error("Format and check identities must match");
  }
  if ((fields.state === "formatted") !== (fields.replacement !== null)) {
    throw new Error("Only formatted results carry a replacement");
  }
  if (
    (fields.state === "formatted" || fields.state === "unchanged") &&
    !fields.check.format_allowed
  ) {
    throw new Error(
      "Successful formatting requires an authoritative qualified check",
    );
  }
  return Object.freeze({
    ...fields,
    key: Object.freeze({ ...key }),
    check: Object.freeze({
      ...fields.check,
      key: Object.freeze({ ...checkedKey }),
      diagnostics: Object.freeze(
        fields.check.diagnostics.map(createDiagnostic),
      ),
    }),
  });
}
