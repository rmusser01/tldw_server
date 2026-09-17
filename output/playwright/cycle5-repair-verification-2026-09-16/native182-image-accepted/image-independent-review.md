# UAT182 independent native image audit

## Verdict

Pass for the bounded PostgreSQL asset upload, byte retrieval, saved reference persistence and rendered preview after reload. This evidence does **not** pass pointer-based Create: that click was blocked by nextjs-portal, and the successful creation used keyboard Enter. UAT161 remains separately responsible for that obstruction.

## Evidence chain

1. `image-upload-result.txt` records the ordinary file chooser selecting repository `apps/tldw-frontend/public/icon.png`. Actual asset POST at00:55:07.514Z returns200 at00:55:07.568Z with asset UUID `c32ba7d6-2091-4d3d-9fb6-24ff206f25e0`, PNG,2962bytes,120×120, and its markdown reference.
2. The attempted pointer Create in `image-card-create-result.txt` times out after5000ms: the actual button is visible/enabled but nextjs-portal intercepts pointer events. The reviewer does not reinterpret this as a successful click.
3. `image-keyboard-create.txt` records focus on the actual Create button and `page.keyboard.press("Enter")`. At01:00:22.317Z the real POST contains the asset markdown reference, deck1 and the disposable-card text. Its200 response at01:00:22.363Z returns card UUID `f2a61712-8d3e-45df-bfb9-0527f86cda21`, owner client2, version1. This is the successful creation path.
4. `image-saved-card-reload.txt` records `page.reload()`. Subsequent real list responses at01:00:55.983Z and01:00:56.003Z retain the same card UUID and full asset URI in its front. The original Citrine card is also still present in the list evidence.
5. `image-edit-after-reload.txt` records an ordinary click on the saved card's Edit control; `image-persisted-preview.txt` records clicking Show preview. At01:01:44.352Z the content GET returns200 image/png. The reviewer independently decodes the retained base64 response and hashes2962bytes to `1792198785947731fc31e4c6184adeb00b988e703fc548893e3c1049a9b45453`, identical to the actual public icon file. All nine retained content responses for this asset have the same bytes/hash; the final response follows reload.
6. At01:01:45.500Z the actual image element reports complete=true, naturalWidth120 and naturalHeight120. `image-persisted-visible.png` was independently viewed: the saved Edit Card preview visibly shows the white speech-bubble icon with three black dots, alongside the persisted asset markdown and disposable text. The element's blob URL is presentation evidence; byte identity is established separately from the captured HTTP response.

All timestamps above are2026-09-17 UTC. `image-independent-event-summary.json` retains the selected semantic events without the repeated base64 bodies; the full original body remains in image-final-evidence.txt. `image-independent-source-and-bytes.json` records the independent binary/source checks. `image-independent-manifest.json` hashes the reviewed receipts and screenshot.

## Source attribution

The00:12 study source-manifest is only the earlier protocol-preparation snapshot and is excluded as acceptance source proof. The separate `.tmp/uat181-native-20260917/fixed-runtime-source.json` records backend PID14540 at00:38:21.252Z with frozen181/182 source, including ChaChaNotes_DB SHA `17f1a2db3214b6488fd9cb1e6c137dcdfb08594faca869b71115b9afa799a84d`. The parent stop receipt at01:04:57.775Z records the same PID after image capture, productionDiffEmpty=true, and commit `a647f5cd890e980dd79027ead3452486ad2b2a47`.

The reviewer independently read that commit's four recorded production paths and verified every byte hash equals the pre-capture frozen runtime receipt. This safely links the later committed181/182 source to the tested backend without claiming the whole earlier working tree was clean. The asset named-column fix is part of that exact ChaCha hash.

## Limits

This read-only audit performs no browser, runtime or database action. It validates the retained native operation/response/render sequence and source receipts. It does not claim a clean console across unrelated Study requests, a pointer Create success, or an end-to-end guarantee for every asset format. Later reset/delete of this deliberately disposable card belongs to UAT180 and does not erase the successful saved-image evidence.
