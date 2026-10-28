# Document Workspace: generated PDF drafts and saved-annotation export

Task: TASK-13260.278.18.83.45. Human authorizes these additions and all necessary fixes/validation; the historical Kanban transport incident stays open while Document proceeds. S03 must finish before S04.

ADR required: no new durable architecture rule. Reuse ADR008 workspace persistence, ADR023 Media-owned artifacts and ADR059 task editing. This is bounded behavior within existing Chat, Media storage, document annotations and foreground ownership contracts; no new store, dependency, worker service or provider integration.

## User flow

Generate opens a separate owned dialog, using the configured model selector and normal scoped Chat transport. Strict complete JSON describes a title and 1–10 plain-text pages. Draft headings/text are editable before Save. Stop/Cancel retires generation and preserves the current saved document and annotations. No preview receives a fake Media ID or enters openDocuments.

Save renders the reviewed draft to PDF, then uses existing authenticated Media ingestion with keep_original_file=true. Open only a positive saved Media ID with its preserved original. Cache one exact PDF/file name for that save intent. A known save ACK followed by an open failure offers Open saved document without another upload. An uncertain upload offers Library reconciliation and the retained PDF, rather than automatically replaying an unacknowledged write. Existing saved documents are never overwritten by generation.

PDF with saved annotations retrieves the authorized preserved PDF and canonical saved annotations. It retains original pages and adds PDF page comments containing the exact selected text and note, with named color/type/page. Highlight coordinates are not stored: do not invent highlight geometry. Explain that this export uses page comments. Existing Markdown/JSON/plain-text annotation export remains available, including EPUB; new combined PDF export is PDF-only and explicitly labelled. Unsaved changes block a false saved-annotation export.

## Annotation correctness before feature exposure

Establish causal controls for stale A/B reads, cached A→B→A, pending creates during GET/ACK, document-switch sync destination, account/server retirement, and batch retry. Keep pending arrays partitioned by document under one current account; merge canonical reads without clearing unsaved creates. ACK only the submitted snapshots and reconcile client/server IDs. Use captured authenticated request scopes and cancellation for every affected request.

Client IDs identify immutable creation intents within user/document. Matching replay returns the same row; changed/deleted identity conflicts without overwrite, and a failed batch rolls back. Existing calls without client IDs retain generated identities. Pending creations cannot be edited/deleted until their ACK; later saved-note edits/deletes await canonical server success. Replace unsafe delayed deletion with explicit confirmation. No unauthenticated beacon or queueing-is-delivery claim: unfinished changes produce a before-unload warning. Scope retirement aborts old work, clears private workspace state and revokes its Blob URLs; URL reload/open follows the same authority fence.

## PDF processing and boundaries

Use installed PyMuPDF in a short-lived child process, not concurrent application threads. Its documented thread restriction is at https://pymupdf.readthedocs.io/en/latest/recipes-multiprocessing.html . Reuse existing bounded owned-worker/drain helpers around one fixed Python module command and stdlib subprocess timeout; no generic process framework or new dependency. The child only reads bounded stdin and returns PDF bytes, with fixed safe errors. Two concurrent operations maximum, 10-second child deadline; cancellation drains owned work before releasing capacity.

Use Story/DocumentWriter APIs available at the declared PyMuPDF floor, escaped plain text and fixed readable typography. Reject overflow rather than truncate text or shrink it unreadably. Rendering accepts at most10 pages/6000 characters per page, with total bounded JSON; output and source PDF have explicit20MiB limits. Annotation export reads at most501 bounded rows to enforce500 annotations, per-field and total payload limits, and validates actual page numbers. Locked/corrupt/non-PDF/oversized input fails clearly without partial output. No source path, subprocess stderr, credential or user text is logged in errors.

Endpoints: authenticated POST /media/documents/render (PDF response; normal Media-create authority/rate limit) and GET /media/{id}/annotated-file (normal Media-read authority; canonical storage abstraction). No caller-provided paths, files, annotations, user IDs or model calls on export. Rendering is stateless; quotas/storage admission remain in normal Chat and Media-save paths.

## Scope and validation

Touched consumers include DocumentWorkspacePage/picker/generation dialog, annotation store/hooks/panel/export modal, typed Tldw client/domain and exact request-scope allowlists, Media annotation repository/endpoints/schemas/router, PDF service/child, and their real fixtures/tests. Shared canonical API export/types/fingerprint generation is required after the combined S03 source is qualified; never select a digest manually.

First prove annotation defects on unchanged code, then minimal fixes and affected UI/WebUI/backend tests. PDF tests use real child rendering and fresh independent readers; UI controls cover Stop, malformed/truncated completion, pending Save, uncertain ACK, original-file failure and scope retirement. Native controlled two-page Rowan document: edit exact text, Save, annotation, reload, one combined export, PDF.js independent full text/comment reader, failed generation/Stop/legitimate Retry retaining saved pages/annotation, four deployment modes and account boundaries. Capture a settled screenshot for human UX; remove completed private payloads after actual process/request release. Preserve the original session and all source/normal-hook/security/review gates.
