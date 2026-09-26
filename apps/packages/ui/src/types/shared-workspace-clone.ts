import { z } from "zod"

export const cloneIdSchema = z.string().uuid()
export const cloneKeySchema = z.string().regex(/^[A-Za-z0-9._~-]{16,200}$/)
// Python/Pydantic bounds names by Unicode code points, not UTF-16 code units.
export const cloneNameSchema = z
  .string()
  .min(1)
  .max(510)
  .refine((name) => Array.from(name).length <= 255)
export const cloneRequestSchema = z.strictObject({
  name: cloneNameSchema.refine((name) => Boolean(name.trim())).optional()
})
const count = z.number().int().min(0).max(1_000_000_000)
const code = z.string().regex(/^[a-z][a-z0-9_]{0,63}$/)
const timestamp = z.string().max(64).datetime({ offset: true })
const counts = z
  .strictObject({
    sources_attempted: count,
    sources_copied: count,
    sources_failed: count,
    notes_attempted: count,
    notes_copied: count,
    notes_failed: count,
    artifacts_attempted: count,
    artifacts_copied: count,
    artifacts_failed: count,
    media_attempted: count,
    media_copied: count,
    media_failed: count,
    operation_owned_media_count: count
  })
  .refine(
    (value) =>
      (["sources", "notes", "artifacts", "media"] as const).every(
        (kind) =>
          value[`${kind}_copied`] + value[`${kind}_failed`] <=
          value[`${kind}_attempted`]
      ) && value.operation_owned_media_count <= value.media_copied
  )
const result = z.strictObject({
  schema_version: z.literal(1),
  outcome: z.enum(["complete", "partial"]),
  workspace_id: cloneIdSchema,
  name: cloneNameSchema,
  publication_confirmed: z.literal(true),
  counts,
  readiness: z.strictObject({
    text_search: z.enum(["ready", "unavailable"]),
    citations: z.enum(["ready", "unavailable"]),
    vector_search: z.enum(["ready", "needs_indexing", "not_configured"])
  }),
  warnings: z.array(z.strictObject({ code, count })).max(8)
})
const base = {
  schema_version: z.literal(1),
  operation_id: cloneIdSchema,
  workspace_id: cloneIdSchema,
  command: z.literal("shared_workspace_clone"),
  share_id: z.number().int().positive().max(Number.MAX_SAFE_INTEGER),
  started_at: timestamp,
  updated_at: timestamp,
  retryable: z.boolean(),
  diagnostics: z
    .record(code, z.union([code, count, z.boolean(), z.null()]))
    .refine((value) => Object.keys(value).length <= 16),
  poll_href: z.string().min(1).max(512)
}
export const sharedCloneOperationSchema = z
  .discriminatedUnion("status", [
    z.strictObject({
      ...base,
      status: z.enum(["queued", "running"]),
      progress: z.strictObject({
        phase: z.enum([
          "queued",
          "authorizing",
          "preparing",
          "sources",
          "notes",
          "artifacts",
          "finalizing"
        ]),
        percent: z.number().int().min(0).max(100),
        message_code: code
      }),
      result: z.null(),
      error: z.null()
    }),
    z.strictObject({
      ...base,
      status: z.literal("succeeded"),
      progress: z.null(),
      result,
      error: z.null()
    }),
    z.strictObject({
      ...base,
      status: z.literal("failed"),
      progress: z.null(),
      result: z.null(),
      error: z.strictObject({
        code: z.string().min(1).max(128),
        message_key: z.string().min(1).max(160),
        message: z.string().min(1).max(320),
        cleanup_state: z.enum(["complete", "pending", "unknown"])
      })
    })
  ])
  .refine(
    (value) =>
      value.poll_href ===
      `/api/v1/sharing/shared-with-me/${value.share_id}/clone/${value.operation_id}`
  )
  .refine(
    (value) =>
      value.status !== "succeeded" ||
      value.result.workspace_id === value.workspace_id
  )

export type SharedCloneOperation = z.infer<typeof sharedCloneOperationSchema>
export type SharedCloneRequest = z.infer<typeof cloneRequestSchema>
