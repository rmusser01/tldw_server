# Import downloaded avatar packs

Backlog: TASK-15648.

In WebUI and the browser extension, Avatar settings lets the user choose a
native `.tldw-persona-vpack` and review its name, contents and credits. Apply
saves the artwork as an independent Buddy and uses the existing attachment
flow for the conversation or workspace the user selected. If saving artwork
succeeds and attaching fails, the user can retry with the saved Buddy.

The server provides an authenticated multipart preview and an import request.
Preview validates the complete archive without creating a profile. The client
holds the selected file and sends it again on Apply with the reviewed SHA-256.
Import validates the bytes again and uses the same owned snapshot writer as
ready-made artwork. A digest mismatch requires a fresh review. Archive and
asset sizes, raster dimensions, activation eligibility and artwork credits use
the native validation rules. An archive cannot set behavior or attachment
permissions. Validation errors preserve existing profiles and attachments.

The independent snapshot path supports SQLite and PostgreSQL. The Persona
visual-pack editor retains its job-based preview and commit flow. The shared
Avatar settings component supplies the file picker and review for both clients.

ADR required: no new ADR. [Independent Buddy ownership](../../backlog/decisions/005-independent-buddy-bindings-and-work-ownership.md)
and [Artwork credit portability](../../backlog/decisions/006-buddy-artwork-credit-portability.md)
govern the new entry point. It adds no persistence schema or job owner.

Implementation and verification:

1. Drive the desired API and Avatar settings behavior in focused regressions.
2. Share native archive inspection and validate, preview and publish through
   the existing Buddy snapshot path on both databases.
3. Add the file review to the shared modal and reuse its Apply and retry flow.
4. Document the import and run the affected API, portability and UI checks,
   plus touched-file lint and Bandit.
