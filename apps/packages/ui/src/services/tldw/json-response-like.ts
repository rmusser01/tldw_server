type JsonPrimitive = string | number | boolean | null
type JsonValue = JsonPrimitive | JsonValue[] | { [key: string]: JsonValue }

const toJsonValue = (value: unknown): JsonValue => {
  if (
    value === null ||
    typeof value === "string" ||
    typeof value === "number" ||
    typeof value === "boolean"
  ) {
    return value as JsonPrimitive
  }
  if (Array.isArray(value)) {
    return value.map((item) => toJsonValue(item)) as JsonValue
  }
  if (value && typeof value === "object") {
    const record: Record<string, JsonValue> = {}
    for (const [key, item] of Object.entries(value)) {
      record[key] = toJsonValue(item)
    }
    return record
  }
  return String(value ?? "")
}

export const createJsonResponseLike = (
  payload: unknown,
  init: { status?: number } = {}
): Response => {
  const safePayload = toJsonValue(payload)
  const bodyText = JSON.stringify(safePayload)
  const headers = new Headers({ "content-type": "application/json" })
  const status = init.status ?? 200

  return {
    ok: status >= 200 && status < 300,
    redirected: false,
    status,
    statusText: status === 200 ? "OK" : String(status),
    type: "default",
    url: "",
    body: null,
    bodyUsed: false,
    headers,
    clone() {
      return createJsonResponseLike(safePayload, { status })
    },
    async arrayBuffer() {
      return new TextEncoder().encode(bodyText).buffer
    },
    async blob() {
      return new Blob([bodyText], { type: "application/json" })
    },
    async formData() {
      throw new TypeError("formData() is not implemented for JSON response-like objects")
    },
    async json() {
      return JSON.parse(bodyText)
    },
    async text() {
      return bodyText
    }
  } as unknown as Response
}
