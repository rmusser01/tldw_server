import { getStructuredApiErrorDetail } from "@/services/tldw/api-error"

const feedback = {
  failed: {
    key: "editMessage.failed",
    fallback:
      "We could not confirm whether your edit was saved. Your edited text is kept. Copy it, then cancel and reload this conversation to check the saved message before saving again."
  },
  changed: {
    key: "editMessage.changed",
    fallback:
      "The saved message has changed. Your edited text is kept. Copy it, then cancel and reload this conversation to review the latest version."
  },
  busy: {
    key: "editMessage.busy",
    fallback:
      "A reply is still in progress. Your edited text is kept. Wait for it to finish, then save your edit."
  },
  scope: {
    key: "editMessage.scope",
    fallback:
      "The server or signed-in account changed. Your edited text is kept. Copy it, then return to the intended server and account and reload this conversation."
  },
  unsupported: {
    key: "editMessage.unsupported",
    fallback:
      "This conversation does not support this edit action yet. Your edited text is kept. Copy it before cancelling."
  },
  payload: {
    key: "editMessage.payload",
    fallback:
      "Only non-empty plain-text messages can be saved here. Your edited text is kept. Copy it before cancelling."
  },
  sync: {
    key: "editMessage.sync",
    fallback:
      "Your sync setup does not support editing saved messages yet. Your edited text is kept. Copy it before cancelling."
  },
  retired: {
    key: "editMessage.retired",
    fallback:
      "The conversation or edited text changed during this save. Your edited text is kept. Review the current message before saving again."
  },
  setup: {
    key: "editMessage.setup",
    fallback:
      "Your edited text is kept. Review the conversation setup, then try Save & Send again."
  },
  unavailable: {
    key: "editMessage.unavailable",
    fallback:
      "The saved message is unavailable in this conversation. Your edited text is kept. Copy it, then cancel and reload this conversation to review its current messages."
  },
  access: {
    key: "editMessage.access",
    fallback:
      "This account or connection cannot access the saved message. Your edited text is kept. Copy it, then check your server connection and sign-in and reload this conversation."
  }
} as const
const codes: Readonly<Record<string, keyof typeof feedback>> = {
  message_changed: "changed",
  version_conflict: "changed",
  stale_snapshot: "changed",
  stale_projection: "changed",
  response_in_progress: "busy",
  request_config_scope_changed: "scope",
  owner_lease_expired: "scope",
  owner_conversation_mismatch: "scope",
  owner_namespace_changed: "scope",
  stale_selection: "retired",
  unsupported_history_capability: "unsupported",
  native_history_mutation_unavailable: "unsupported",
  unsupported_history_edit_and_send: "unsupported",
  unsupported_message_payload: "payload",
  sync_v2_message_edit_not_supported: "sync",
  validation_failed: "setup",
  message_edit_unconfirmed: "failed",
  submission_unacknowledged: "failed"
}
const category = (code: string) => {
  const key = code.startsWith("unsupported_history_edit_and_send:")
    ? "unsupported_history_edit_and_send"
    : code
  return Object.hasOwn(codes, key) ? codes[key] : null
}
const record = (value: unknown): Record<string, unknown> =>
  value !== null && typeof value === "object" && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {}

/** Classify declared conditions; raw server details are never presentation copy. */
export const getMessageEditFeedback = (issue: unknown) => {
  let current = issue
  for (let depth = 0; current != null && depth < 4; depth++) {
    const value = record(current),
      details = record(value.details)
    const detail = getStructuredApiErrorDetail({
      detail: value.detail ?? details.detail ?? value.details
    })
    const declared = [
      value.reason,
      value.code,
      detail?.code,
      detail?.error_code
    ].filter(
      (code): code is string => typeof code === "string" && code.length > 0
    )
    if (declared.length) {
      const kinds = declared.map(category)
      return feedback[
        kinds[0] && kinds.every((kind) => kind === kinds[0])
          ? kinds[0]
          : "failed"
      ]
    }
    const kind = category(
      typeof current === "string"
        ? current
        : typeof value.message === "string"
          ? value.message
          : typeof value.errorMessage === "string"
            ? value.errorMessage
            : ""
    )
    if (kind) return feedback[kind]
    if (typeof value.status === "number") {
      return feedback[
        value.status === 409
          ? "changed"
          : value.status === 404
            ? "unavailable"
            : value.status === 401 || value.status === 403
              ? "access"
              : "failed"
      ]
    }
    current = value.cause
  }
  return feedback.failed
}
