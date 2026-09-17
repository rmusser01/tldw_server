# UAT167/174/175 native deck creation and retry acceptance

One real source-note generation produced one card. Real duplicate deckPOST returned409, preserving the editable draft and usable inline Retry without a blocking Next overlay. Editing the deck name and using ordinary Retry returned actual deckPOST200 (id4, serialized timestamps) and exactly one cardPOST200 (b59f0819-47a7-4284-b78e-7805015430b0). Normal Manage reload retains deck identity/settings, card question/answer/tags and source-note reference. Original cardv2, disposablev4 and prior generated cardv1 remain unchanged. Root machine assertions and duplicate/reloaded visual inspections pass. No synthetic network response or active fault was used.

UAT196 separately tracks internal Entity/POST text in the duplicate guidance; this does not invalidate174 conflict classification or175 usable recovery. Native API63046 runs3ce1b63d57 without hot reload; later backend edits are not loaded. Its role is superuser/BYPASSRLS, so no tenant-security or192 tag-link-integrity claim.

Initial reload snapshot briefly showed Not connected before authenticated bootstrap; settled realGET200 responses and visible saved data are retained. Initial audit used lexicographic timestamps with mismatched fractional precision; corrected Date.parse comparisons pass, and the initial script remains as a harness error. Response observer timestamps follow body consumption. Full fresh UAT remains gated.

Known runtime credentials and JWT/PEM patterns scanned, zero matches. Original evidence remains unchanged.
