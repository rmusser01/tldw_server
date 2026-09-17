# Reopened UAT161

TASK13260.98. Bounded repair of the actual bottom-right Create collision. Earlier Settings repair remains historical and was insufficient across tested routes.

## Design
Use installed Next16.1.4 supported devIndicators:false. This removes the optional floating badge rather than choosing another corner. The Pages dev overlay registration/render and error boundary are independent of that configuration: installed next-devtools/userspace/pages/pages-dev-overlay-setup.js registers error and rejection listeners without the indicator flag; build/define-env.js maps only the optional indicator flag. Runtime error visibility is separately checked in a disposable browser tab using a clearly labeled synthetic ErrorEvent.

## Evidence and limits
Native pointer RED: ../uat-study-native-20260917/image-card-create-result.txt (nextjs-portal intercept; no Create POST). Existing public image was saved later through keyboard Enter; the image acceptance is182, separate from161.

After repair, actual Create center is owned by the button, badge count0; normal pointer click invokes required Front/Back validation. This tests the original obstruction without adding a duplicate card. Empty form canceled, original Settings ordinary pointer checked. Error control results and independent review are pending.

Source: apps/tldw-frontend/next.config.mjs only (three-line comment/option change). node --check passes; installed schema valid; existing config suites12/2 pass; scoped ESLint clean. JavaScript is outside Bandit parser support; no Python source is touched by161. No portal removal, CSS pointer bypass, force click or persisted developer preference mutation.

GREEN: normal Create pointer invokes both required-field messages; original Settings pointer navigates /settings. In disposable tab12, labeled synthetic unexpected TypeError remains visible with devIndicators:false. The control does not represent an observed product bug and is isolated from normal acceptance traces; tab closed afterward. See error-visibility-result.txt/error-visible.png.
