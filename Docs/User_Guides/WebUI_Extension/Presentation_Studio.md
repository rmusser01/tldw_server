# Presentation Studio

Presentation Studio supports two separate project kinds:

- **Structured slides** use the existing slide editor, preview, render, and export workflows.
- **Standalone HTML + JavaScript** stores one complete HTML document as untrusted, executable text.

These project kinds cannot be converted in place. Their editing and export contracts are intentionally different.

## Important Security Boundary

Presentation Studio never previews, renders, navigates to, or runs a standalone HTML document. The code editor is inert text. The Safe outline is an application-owned text-only summary produced without loading resources or executing the document.

Validation checks whether a document meets the storage contract. It does not make the JavaScript safe. A downloaded `presentation.html` can execute when you open it outside tldw, with the permissions of that external browser or viewer. Review the source before opening it.

## Availability

The WebUI reads `GET /api/v1/slides/capabilities` before offering standalone generation or source-bearing actions. Generation may be disabled while saved standalone projects remain readable and editable. The UI shows a bounded reason and Retry action when the current server, account, validator, worker, or generation configuration cannot be confirmed. On the creation form, Retry revalidates the current server and account while keeping all source fields unmounted until that authority check succeeds.

The first release provides a direct-material form in the WebUI. The backend also accepts owner-scoped chat, media, notes, and RAG sources, but this release does not expose those source selectors in the standalone WebUI form.

## Create A Standalone Presentation

1. Open **Presentation Studio** and choose **New presentation**.
2. Select **Standalone HTML + JavaScript**.
3. Paste the subject and direct material. Choose the presentation type, audience, slide count, visual direction, and delivery style.
4. Review the displayed provider, model, adapter, endpoint identity, and generation configuration revision.
5. Select **Generate standalone presentation**.

Submission is asynchronous. The form shows the immutable submitted request and the real job state. **Stop waiting** stops local polling; it does not cancel server work. A stopped or interrupted request can be resumed from the same authenticated server/account scope.

The form uses two capped, 24-hour records in scoped `sessionStorage`: one form
draft and one bounded resume/job-metadata record. It does not use
`localStorage` or extension storage. Recovery is cleared on logout,
account/server mismatch, successful handoff, explicit Forget, or expiry. If
browser storage is unavailable, the current tab remains usable and displays a
persistent recovery warning.

Provider calls follow at-least-once worker semantics. A process crash after a provider response but before commit may repeat the provider call and its cost. Replaying the same completed receipt does not enqueue a replacement job.

## Edit A Standalone Presentation

The standalone workspace contains:

- **Code**, which holds the complete inert source in the editor.
- **Outline**, which shows bounded trusted text only.
- **Speaker notes**, shown in a labelled disclosure when the document contains notes.

The Safe outline is not a browser preview and is not authoritative validation. It omits scripts, styles, active/resource elements, URLs, generated deck chrome, and other untrusted structures. It can become stale or unavailable without changing the editor source.

There is no autosave. **Save** validates and writes the complete document using the current strong ETag. If the server response is lost, the workspace reconciles by owner, project, digest, and a fresh ETag. Your local draft stays visible on every error.

If another writer changes the project, the workspace offers three explicit choices:

- **Discard my changes and load server version**
- **Overwrite server with my draft** after a fresh server-version check and confirmation
- **Download my draft**

Ordinary Save is disabled while the conflict requires one of those choices.

## Recovery And Navigation

Unsaved workspace source stays in component memory and, when available, one capped recovery record scoped to canonical server origin, authenticated principal, and presentation ID. The record expires after 24 hours and is never applied automatically.

When a matching record exists, choose one of:

- **Restore recovered draft**
- **Download recovered draft**
- **Discard recovered draft** after confirmation

Dirty Back navigation and browser unload are guarded. On pagehide, the latest preflight-valid editor candidate is synchronously written or cleared, then source-bearing memory is scrubbed before bfcache capture. A same-scope return revalidates authority before restoration. Logout, account change, or server-origin change scrubs the old scope instead.

Keep the tab open or download the draft when the workspace reports **Recovery unavailable**.

## Download

**Download current draft** returns the exact editor bytes as `application/octet-stream` with the fixed filename `presentation.html`. It does not save the draft. A saved HTML export uses the same fixed attachment boundary.

The WebUI creates a temporary download-only Blob URL only after validating the authenticated response status, media type, filename, and security headers. It never assigns that URL to a preview, frame, worker, popup, or navigation target, and revokes it within one second or earlier during cleanup.

Opening the downloaded file occurs outside tldw's security boundary and may execute its JavaScript.

## Browser Extension

The extension remains source-free:

- `/presentation-studio` provides the metadata index.
- `/presentation-studio/start` preserves the structured quick-start flow.
- `/presentation-studio/new` is WebUI-only and is not registered in the extension.
- A direct standalone or unknown-kind project link reads exact-ID metadata only, then offers **Open in WebUI**.

The handoff URL is built from the configured canonical WebUI base and the encoded trusted route ID. The extension does not request standalone detail, versions, source, save, export, draft attachment, render, or preview data. It does not store or send source-bearing extension messages.

## Troubleshooting

- **Standalone generation is disabled**: ask an administrator to check the capability reason. Saved projects can remain available.
- **Validation is unavailable**: inert reads and draft recovery may remain available, while save, restore, export, and generation fail closed.
- **Generation configuration changed**: refresh capabilities and reconfirm the displayed provider/model target before submitting again.
- **Outline unavailable**: keep editing or download the draft; the outline failure does not mutate source.
- **Conflict**: choose one of the three explicit conflict actions. Do not retry the stale Save action.
- **Recovery unavailable**: keep the tab open or download the draft. The application does not silently claim that recovery succeeded.

For operator setup and rollback, see [Standalone HTML Presentations](../../Deployment/Standalone_HTML_Presentations.md).
