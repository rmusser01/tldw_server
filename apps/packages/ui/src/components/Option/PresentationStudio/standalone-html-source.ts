export const MAX_STANDALONE_HTML_SOURCE_BYTES = 1_048_576

export type StandaloneHtmlSourceFailureCode =
  | "source_required"
  | "source_contains_nul"
  | "invalid_unicode_scalar"
  | "source_too_large"
  | "digest_unavailable"

export type StandaloneHtmlSourceFailure = {
  ok: false
  code: StandaloneHtmlSourceFailureCode
  message: string
}

export type AcceptedStandaloneHtmlSource = {
  ok: true
  source: string
  scalarCount: number
  byteLength: number
  bytes: Uint8Array
  digest: string
}

export type StandaloneHtmlSourceResult =
  | AcceptedStandaloneHtmlSource
  | StandaloneHtmlSourceFailure

type SourcePreflightSuccess = {
  ok: true
  scalarCount: number
  byteLength: number
}

export type StandaloneHtmlSourcePreflight = SourcePreflightSuccess | StandaloneHtmlSourceFailure

const failure = (
  code: StandaloneHtmlSourceFailureCode,
  message: string
): StandaloneHtmlSourceFailure => ({ ok: false, code, message })

const scalarUtf8Bytes = (codePoint: number): number => {
  if (codePoint <= 0x7f) return 1
  if (codePoint <= 0x7ff) return 2
  if (codePoint <= 0xffff) return 3
  return 4
}

/**
 * Rejects forbidden UTF-16 before any encoder, component state, worker, or persistence boundary.
 */
export const preflightStandaloneHtmlSource = (
  source: unknown,
  options: { allowEmpty?: boolean } = {}
): StandaloneHtmlSourcePreflight => {
  if (typeof source !== "string") {
    return failure("source_required", "HTML source is required.")
  }
  if (options.allowEmpty === false && source.length === 0) {
    return failure("source_required", "HTML source is required.")
  }

  let scalarCount = 0
  let byteLength = 0
  for (let index = 0; index < source.length; index += 1) {
    const unit = source.charCodeAt(index)
    if (unit === 0) {
      return failure("source_contains_nul", "HTML source cannot contain U+0000.")
    }

    let codePoint = unit
    if (unit >= 0xd800 && unit <= 0xdbff) {
      const next = source.charCodeAt(index + 1)
      if (index + 1 >= source.length || next < 0xdc00 || next > 0xdfff) {
        return failure(
          "invalid_unicode_scalar",
          "HTML source must contain valid Unicode scalar values."
        )
      }
      codePoint = 0x10000 + ((unit - 0xd800) << 10) + (next - 0xdc00)
      index += 1
    } else if (unit >= 0xdc00 && unit <= 0xdfff) {
      return failure(
        "invalid_unicode_scalar",
        "HTML source must contain valid Unicode scalar values."
      )
    }

    scalarCount += 1
    byteLength += scalarUtf8Bytes(codePoint)
    if (byteLength > MAX_STANDALONE_HTML_SOURCE_BYTES) {
      return failure(
        "source_too_large",
        "HTML source cannot exceed 1 MiB of UTF-8 text."
      )
    }
  }

  return { ok: true, scalarCount, byteLength }
}

const bytesToHex = (bytes: Uint8Array): string =>
  Array.from(bytes, (byte) => byte.toString(16).padStart(2, "0")).join("")

export const validateStandaloneHtmlSource = async (
  source: unknown,
  options: { allowEmpty?: boolean } = {}
): Promise<StandaloneHtmlSourceResult> => {
  const preflight = preflightStandaloneHtmlSource(source, options)
  if (preflight.ok === false) return preflight

  const exactSource = source as string
  const bytes = new TextEncoder().encode(exactSource)
  if (bytes.byteLength !== preflight.byteLength) {
    return failure("invalid_unicode_scalar", "HTML source could not be encoded exactly.")
  }
  try {
    const digestBuffer = await globalThis.crypto.subtle.digest("SHA-256", bytes)
    return {
      ok: true,
      source: exactSource,
      scalarCount: preflight.scalarCount,
      byteLength: bytes.byteLength,
      bytes,
      digest: bytesToHex(new Uint8Array(digestBuffer))
    }
  } catch {
    return failure("digest_unavailable", "HTML source digest is unavailable.")
  }
}
