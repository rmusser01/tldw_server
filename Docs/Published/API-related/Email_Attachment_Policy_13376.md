# Email attachment extraction options

Both `/api/v1/media/add` with `media_type=email` and
`/api/v1/media/process-emails` accept the following multipart fields. These options
apply to EML messages and messages expanded from ZIP/MBOX uploads.

| Field | Default | Behavior |
| --- | --- | --- |
| `ingest_attachments` | `false` | Existing flag enabling attached EML children. |
| `extract_attachments` | omitted | Inherits `ingest_attachments`. Explicit `true` enables extraction; explicit `false` keeps descriptors only even if the legacy flag is true. |
| `attachment_mime_allowlist` | omitted | Selects `message/rfc822` plus the existing `.eml` filename fallback. An explicit list selects declared MIME types and disables filename inference; an empty list selects none. |
| `attachment_mime_denylist` | omitted | No exclusions. Denies override allows, including the declared MIME type and `.eml` inference. |
| `max_depth` | `2` | Existing recursion depth, including the root message; range `1..5`. |

MIME lists accept repeated fields or comma-separated values. Rules are
case-insensitive exact types (`message/rfc822`), type wildcards (`message/*`),
or `*/*`; malformed rules return HTTP 422. At most 64 rules per list are accepted.

Only nested EML extraction is supported. Allowing PDF, document, image, audio,
or arbitrary binary MIME types does not activate additional processors. Those
attachments retain their metadata. PST/OST attachments also remain metadata-only
because the optional pypff adapter exports descriptors, not attachment payloads.
Missing pypff produces the existing deterministic unsupported-format error.

Attachment descriptors retain available `name`, `content_type`, `size`,
`content_id`, and `disposition`. Size is `null` when it cannot be determined
without decoding or serializing the attachment, including quoted-printable
payloads and nested message objects. Additive `extraction_status` is `captured` when
EML bytes are passed into recursive processing, or `skipped` with an
`extraction_reason` of `disabled`, `depth_limit`, `mime_denied`,
`mime_not_allowed`, `unsupported_mime`, `size_limit`, `count_limit`,
`empty_payload`, `capture_failed`, or `pst_metadata_only`.
`captured` describes extraction; the child result reports its processing outcome.

Skipped attached EML messages are not serialized or processed. Their text never
enters the parent message body. Child EML analysis stays disabled; extraction
options do not enable models, embeddings or claims. Parent-message analysis
retains its independent `perform_analysis` option.

Existing archive size/count, per-member size, unsafe-path, encryption and depth
checks remain. Each message's selected nested EML children honor the configured
archive `max_internal_files` and `max_member_uncompressed_size_mb` limits.
No binary attachment content is stored.

For metadata-only import, add `-F 'extract_attachments=false'`. For selected EML
extraction, add:

```sh
-F 'extract_attachments=true' \
-F 'attachment_mime_allowlist=message/rfc822' \
-F 'attachment_mime_denylist=application/pdf' \
-F 'max_depth=2'
```

For entirely model-free ingestion, separately set `perform_analysis=false`,
`perform_claims_extraction=false`, `perform_chunking=false`,
`auto_chunking_use_llm=false`, and `generate_embeddings=false`.
