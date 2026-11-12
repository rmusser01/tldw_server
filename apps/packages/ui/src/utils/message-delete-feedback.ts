const feedback = {
  failed: {
    key: "deleteMessage.failed",
    fallback:
      "We could not confirm deletion. Reload this conversation to check the saved message before trying again."
  },
  descendants: {
    key: "deleteMessage.descendants",
    fallback:
      "This message has replies. Delete those replies first, including any other branches, then try again."
  },
  changed: {
    key: "deleteMessage.changed",
    fallback:
      "The saved message has changed. Reload this conversation and review the latest message before deleting it."
  },
  busy: {
    key: "deleteMessage.busy",
    fallback:
      "A reply is still in progress. Wait for it to finish before deleting a saved message."
  },
  scope: {
    key: "deleteMessage.scope",
    fallback:
      "The server or signed-in account changed. Return to the intended server and account, then reload this conversation."
  },
  unsupported: {
    key: "deleteMessage.unsupported",
    fallback:
      "This conversation does not support deleting saved messages from this view yet."
  },
  sync: {
    key: "deleteMessage.sync",
    fallback:
      "Your sync setup does not support deleting saved messages from this view yet."
  },
  unavailable: {
    key: "deleteMessage.unavailable",
    fallback:
      "This saved message is unavailable. Reload the conversation to review its current messages."
  },
  refreshFailed: {
    key: "deleteMessage.refreshFailed",
    fallback:
      "The message was deleted, but this view or its saved copy could not be refreshed. Reload the conversation before continuing."
  }
} as const
const categories: Readonly<Record<string, keyof typeof feedback>> = {
  message_has_descendants: "descendants",
  message_changed: "changed",
  version_conflict: "changed",
  response_in_progress: "busy",
  request_config_scope_changed: "scope",
  owner_conversation_mismatch: "scope",
  unsupported_native_sync_delete: "sync",
  native_history_mutation_unavailable: "unsupported",
  unsupported_history_capability: "unsupported",
  message_delete_refresh_failed: "refreshFailed"
}
const record = (value: unknown): Record<string, unknown> =>
  value && typeof value === "object" ? (value as Record<string, unknown>) : {}

/** Keep internal server tokens and arbitrary provider text out of Delete recovery copy. */
export const getMessageDeleteFeedback = (issue: unknown) => {
  const error = record(issue),
    details = record(error.details)
  const detail = error.detail ?? details.detail
  const declared =
    error.code ??
    record(detail).error_code ??
    record(detail).code ??
    (typeof detail === "string" ? detail : undefined) ??
    (typeof issue === "string" ? issue : error.message)
  const kind =
    typeof declared === "string" && Object.hasOwn(categories, declared)
      ? categories[declared]
      : undefined
  return feedback[
    kind ??
      (error.status === 409
        ? "changed"
        : error.status === 404
          ? "unavailable"
          : "failed")
  ]
}
