# UAT137/138 follow-up repairs

Base3c30685611. Native evidence in retained native-single bundle:137 screenshot/snapshot032 and138 outage091/095. Current source paths/hashes in cycle5-uat137-138-owned.json. No runtime restarted for these edits.

137: actual ICU adapter + English resource test fails singular case before fix (red-icu:1failed/2passed). Count plus English ICU plural resources synchronize visible and accessible labels. English public-resource mirror updated. Existing fallback mocks retain simple English fallback; two stale singular expectations corrected. Green32/4 includes Cram re-rating and queue controls, with no scheduling code change.

138: actual useMediaSearch hook + QueryClient, mocked bgRequest transport/storage only. Confirmed red1failed/2passed: network Error emitted through console.error, while missing-endpoint/unknown-error controls pass. First red run had an invalid404fixture without request path; corrected before production edit and not counted as a product failure. Existing pure backend-unreachable classifier recognizes request failures; those now emit fixed console.warn text while existing toast, endpoint-missing gate, and unknown-error console.error remain. Same-hook successful refetch returns preserved source. Green7/2 with adjacent search/filter controls; final focused3/1 after neutral console wording. Existing mocked-input React warnings remain in adjacent suite.

Scoped lint0errors/42warnings; all42 in existing useMediaSearch. Baseline and current normalized rule/severity/message signatures are identical. TS-only changes: no Python Bandit scope. Independent review, integrated compiler comparison and native acceptance still pending.
