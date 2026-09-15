# Targeted route, guide and saved Note source checks

These are targeted repair observations on the existing isolated multi-user runtime, not a new fresh UAT or release sign-off. The frozen full-run evidence is unchanged.

## Outcomes

- Flashcards, Provider Keys and Server Settings have their expected route titles at 18:09:36 UTC. Setup has its expected title and setup-choice headings at 18:22:09 UTC. The earlier Notes/Home bundle covers the other core route titles.
- The actual **View server setup guide** button opens the maintained GitHub self-hosting index. Its loaded heading and local single-user, Docker single-user/WebUI and multi-user/Postgres profile links are verified at 18:19:51 UTC. This checks navigation and available guidance, not deployment of each profile.
- The actual saved Biology card's Note source link opens Bob's original saved Note, independently of the last-note hint. The settled editor and a full reload display the exact saved title and five-fact content, including whitespace, with Saved status. The 18:42:52 UTC metadata contains three successful detail-resource entries; it does not claim a single request.

## Scope and limits

Route/guide implementation is `545a7a59d2`. Note source behavior was tested from the working tree later committed as `16485e90d4`; subsequent confirmation-boundary corrections have independent regression coverage but these captures do not certify live dirty-save cancellation. Private handoff implementation is separately committed as `aae6b72d05` and is not exercised here.

Media/Chat source clicks, unavailable/foreign sources, dirty-editor controls, both complete fresh workflow matrices and clean-machine dependency installation remain separate acceptance work. No generated response, seeded artifact or API-only navigation counts as a browser pass.

Seven original captures are retained. They were scanned against fourteen known isolated-runtime credentials plus JWT/private-key patterns with zero matches. No raw request headers or credential stores are included. There are no screenshots in this bundle. `SHA256SUMS` covers this README and every capture. An earlier inspection used the wrong textbox placeholder and timed out; that failed automation is excluded from acceptance evidence.
