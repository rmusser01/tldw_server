export type RagTerminalEvent = {
  schema_version: 1
  type: "complete" | "error"
  code: string
  status_code?: number
  upstream_dispatched: boolean
  output_emitted: boolean
  allow_non_stream_fallback: boolean
  message: string
}

const TERMINAL_CODE_PATTERN = /^[a-z][a-z0-9_]{0,63}$/
const MAX_TERMINAL_MESSAGE_LENGTH = 240
const REPLAY_CERTIFICATION_CODE = "stream_transport_unavailable"

const isRecord = (value: unknown): value is Record<string, unknown> =>
  Boolean(value) && typeof value === "object" && !Array.isArray(value)

const looksLikeTerminalEvent = (value: Record<string, unknown>): boolean =>
  value.type === "complete" ||
  value.type === "error" ||
  "schema_version" in value ||
  "upstream_dispatched" in value ||
  "output_emitted" in value ||
  "allow_non_stream_fallback" in value

export class RagTerminalStreamError extends Error {
  readonly event: RagTerminalEvent | null

  constructor(message: string, event: RagTerminalEvent | null = null) {
    super(message)
    this.name = "RagTerminalStreamError"
    this.event = event
  }
}

export const parseRagTerminalEvent = (
  value: unknown
): RagTerminalEvent | null => {
  if (!isRecord(value) || !looksLikeTerminalEvent(value)) return null

  const validStrings =
    typeof value.code === "string" &&
    TERMINAL_CODE_PATTERN.test(value.code) &&
    typeof value.message === "string" &&
    value.message.length > 0 &&
    value.message.length <= MAX_TERMINAL_MESSAGE_LENGTH
  const validBooleans =
    typeof value.upstream_dispatched === "boolean" &&
    typeof value.output_emitted === "boolean" &&
    typeof value.allow_non_stream_fallback === "boolean"
  const validStatusCode =
    !("status_code" in value) ||
    (typeof value.status_code === "number" &&
      Number.isInteger(value.status_code) &&
      value.status_code >= 100 &&
      value.status_code <= 599)

  if (
    value.schema_version !== 1 ||
    (value.type !== "complete" && value.type !== "error") ||
    !validStrings ||
    !validBooleans ||
    !validStatusCode ||
    (value.output_emitted === true && value.upstream_dispatched !== true)
  ) {
    throw new RagTerminalStreamError("Invalid RAG terminal stream event.")
  }

  if (
    value.type === "complete" &&
    (value.code !== "complete" ||
      value.upstream_dispatched !== true ||
      value.allow_non_stream_fallback !== false)
  ) {
    throw new RagTerminalStreamError("Invalid RAG terminal stream event.")
  }
  if (
    value.type === "error" &&
    (value.code === "complete" ||
      (value.allow_non_stream_fallback === true &&
        (value.code !== REPLAY_CERTIFICATION_CODE ||
          value.upstream_dispatched !== false ||
          value.output_emitted !== false)))
  ) {
    throw new RagTerminalStreamError("Invalid RAG terminal stream event.")
  }

  const event: RagTerminalEvent = {
    schema_version: 1,
    type: value.type,
    code: value.code as string,
    upstream_dispatched: value.upstream_dispatched as boolean,
    output_emitted: value.output_emitted as boolean,
    allow_non_stream_fallback: value.allow_non_stream_fallback as boolean,
    message: value.message as string
  }
  if ("status_code" in value) {
    event.status_code = value.status_code as number
  }
  return event
}

export const parseRagStreamLine = (line: string): unknown => {
  let parsed: unknown
  try {
    parsed = JSON.parse(line)
  } catch {
    throw new RagTerminalStreamError("Invalid RAG stream event.")
  }
  return parseRagTerminalEvent(parsed) ?? parsed
}

export const mayReplayNonStream = (event: RagTerminalEvent): boolean =>
  event.schema_version === 1 &&
  event.type === "error" &&
  event.code === REPLAY_CERTIFICATION_CODE &&
  event.upstream_dispatched === false &&
  event.output_emitted === false &&
  event.allow_non_stream_fallback === true
