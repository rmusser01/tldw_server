# Packaged sidebar native evidence

Observed2026-10-04/05 using existing launchWithBuiltExtension with TLDW_E2E_EXTENSION_HEADLESS=false. Actual Chrome for Testing, final development-profile Chrome MV3 package, isolated backend127.0.0.1:18882 and synthetic account. User browser/backend8000 untouched. These are concise observed native accessibility excerpts, not a screenshot or synthetic DOM reconstruction.

1. Native address-bar paste navigated to https://example.com/. Actual context menu: tldw-assistant → Open Sidebar to chat. The sidebar accessibility root was sidepanel.html. Companion Open Chat → Expand sidebar → Pro → More tools → Quick Ingest.
2. Native Capture current tab produced Queue1 and https://example.com/. Capture current tab was enabled again. A second click produced Queue2 and “Web page (auto) Already queued — excluded”, with Configure1item.
3. Native address-bar navigation to chrome://settings/ followed by Capture current tab produced “This tab cannot be captured. Open an HTTP or HTTPS page, or paste its URL here.” Queue remained2 with1eligible. No internal-page source was added.
4. Quick preset → Review → Start processing produced “2 added · 1 excluded/skipped · 1 succeeded (1 saved) · 0 failed · 0 cancelled”, Saved and “Knowledge readiness unconfirmed”. This checks actual UI/background transport with simulated processor outcomes; it does not establish real ingestion/ML reliability.
5. This first full journey exposed sidebar-only404 when Review these1saveditems navigated within the limited router. Narrow repair reuses the local options-tab opening pattern. Fresh package journey repeated real HTTPS capture and processing, then clicked Review these1saveditems.
6. Actual new full-page URL: options.html#/media-multi. Native AX: “1 selected”, “Selected across pages:1”, “Selected reading: Imported source43”, “Reading1–1 of1selected (30at a time)”, “Item1of1”, actual content. No404. Read-only check in the launched browser decoded the stored snapshot to {version:1,ownerPresent:true,selectedIds:[43]}, with legacy mirror[43]. Full authority/credential values intentionally omitted.

Retained harness logs: /private/tmp/media-task5-packaged-final-harness.log, /private/tmp/media-task5-packaged-route-final-harness.log. Initial raw storage inspection treated serialized JSON as an object and misleadingly reported ownerPresent:false; decoding the existing JSON value gave version1/ownerPresent:true/[43], matching visible selected reading.

ScreenCaptureKit screenshot API intermittently failed with-3811; further native screenshot calls were paused. AX/real native gestures establish capture and routing, not a screen-reader audio session. The committed extension-mobile-review.png is separate earlier packaged Review layout evidence, not proof of active-tab capture.

Final post-settlement build recheck repeated the real HTTPS capture/process/review gesture. Full Options displayed Imported source44 / Reading1–1of1. Decoded snapshot {version:1,ownerPresent:true,selectedIds:[44]}, dialogs0. Log: /private/tmp/media-task5-packaged-settlement-final-harness.log. Earlier43 evidence remains a prior route-build checkpoint.
