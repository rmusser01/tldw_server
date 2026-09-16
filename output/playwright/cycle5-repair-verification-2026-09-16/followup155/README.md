# UAT155: truthful Settings sidebar capability

One local capability condition disables the extension-only sidebar action in Next WebUI. Existing genuine Chrome and Firefox extension paths remain enabled.

Author and independent runs each pass 24 tests across 5 suites. Independent baseline replay reproduces the single WebUI failure while all three controls pass. ESLint has zero errors/warnings; the production formatting warning predates this change. TypeScript-only change; Bandit does not apply.

Exact source/test hashes were verified during retention. Native Settings acceptance remains pending; no missing browser artifacts have been reconstructed.
