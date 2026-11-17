/** Declared deployed request contract; API integration checks qualify server enforcement. */
export const protectedEditRequestSchema = () => ({
  type: "object",
  properties: {
    tldw_history_branch: { anyOf: [{ type: "boolean" }, { type: "null" }] },
    tldw_history_edit_v1: {
      anyOf: [
        {
          type: "object",
          additionalProperties: false,
          required: ["version", "message_id", "message_revision"],
          properties: {
            version: { type: "integer", const: 1 },
            message_id: { type: "string", minLength: 1 },
            message_revision: { type: "string", minLength: 1 }
          }
        },
        { type: "null" }
      ]
    }
  }
})
