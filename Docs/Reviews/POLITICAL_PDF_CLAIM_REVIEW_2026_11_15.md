# Political PDF claim review packet

Task: TASK-15505.6. This frozen set contains five preserved failure cases and eight held-out cases. Source expectations were checked by an agent against the specified physical PDF pages. Human labels remain pending; no human accuracy or calibrated judge score is claimed.

Follow-through qualification: these were the original set assignments. H05 later exposed the multi-page citation boundary defect and informed its repair, so it is now regression evidence and must not be counted as untouched held-out validation. One post-fix browser answer correctly states the CFPB proposal and court condition but cites printed 838–839 / PDF 871–872 rather than pinpointing 839 / 872. The original failure and single retest are preserved in the [follow-through report](POLITICAL_PDF_FOLLOWTHROUGH_2026_11_15.md). Other known attribution and disagreement failures below remain unresolved semantic risks; review controls preserve corrections without replacing the model draft.

Use the original PDF checksum and the fixture’s per-page extraction hashes to identify the source. Review both the retrieved evidence and generated answer: missing evidence is a retrieval failure; unsupported wording despite sufficient evidence is a generation failure.

For each case, a human reviewer should record Accept / Correct / Unclear, the unsupported or omitted claim, and a correction with page evidence. A single passing run does not establish general accuracy.

Source: [Mandate for Leadership](https://static.heritage.org/project2025/2025_MandateForLeadership_FULL.pdf). Fixture: `tldw_Server_API/tests/Research_Workspace/fixtures/political_pdf_claim_cases.json`.

## R01 — regression / voice_attribution

Does the FTC chapter describe Lina Khan as opposing its proposed response to ESG and DEI offers? Explain whose position is being quoted.

Evidence: PDF 906 / printed 873.

Agent source expectations:

- The passage reports Khan rejecting offers of social policies in exchange for merger approval; it uses that testimony as evidence for the chapter discussion.
- Do not invent a Candeub-versus-Khan disagreement from this passage.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly distinguishes Khan’s reported refusal of ESG-for-merger offers from the chapter task-force proposal, and does not infer a Candeub–Khan dispute.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-R01.json`.

## R02 — regression / agency_identity

Which agency chapter discusses restoring regional-office authority, and does it record an alternative conservative view?

Evidence: PDF 909 / printed 876.

Agent source expectations:

- FTC, the Federal Trade Commission; not the FEC.
- The text proposes considering restored regional authority and records conservative objections.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly identifies FTC and records the alternative conservative objection to regional-office expansion; no FEC substitution.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-R02.json`.

## R03 — regression / voice_attribution

Compare how de Rugy and Hazelton use EXIM competitiveness reports. Keep the report years and each author’s argument separate.

Evidence: PDF 753 / printed 720, PDF 759 / printed 726.

Agent source expectations:

- De Rugy critiques the 2020 report and its quorum/competitive-standing argument.
- Hazelton cites the 2018 report when arguing foreign export-credit agencies embed their exporters in global supply chains.
- Do not attribute the 2020 quorum critique to Hazelton.

Direct-context DeepSeek result: **fail** (agent source audit). Summary correctly separates de Rugy/2020 from Hazelton/2018, but evidence items 4 and 5 assign Hazelton claims to “Kent Lassman (as identified in the source)”. Neither supplied page names Lassman. The invented author field fails attribution even though the prose is correct.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-R03.json`.

## R04 — regression / unsupported_agreement

Do de Rugy and Hazelton agree that EXIM fills a private-financing gap for its major beneficiaries?

Evidence: PDF 752 / printed 719, PDF 759 / printed 726.

Agent source expectations:

- De Rugy argues major beneficiaries retained access to private financing during the quorum restriction.
- Hazelton argues EXIM provides financing when the private sector will not.
- Shared subject matter does not establish shared agreement.

Direct-context DeepSeek result: **partial_context_limitation** (agent source audit). Correctly rejects shared financing-gap agreement and explains both opposed positions. It declines named attribution because the two frozen pages contain neither author heading. This is an honest evidence limitation, but does not fully satisfy the case’s expected named-author comparison; the test context omits authoritative author boundaries.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-R04.json`.

## R05 — regression / numerical_direction

Describe the change in EXIM authorizations from FY2007 through FY2014 to FY2022, preserving the direction of each change.

Evidence: PDF 751 / printed 718.

Agent source expectations:

- 12.6 billion to 21.5 billion is an increase; 21.5 billion to 5.2 billion is a decrease.
- The final figure is below the initial one; the series is not a continuous decline.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly states authorizations rose from $12.6bn to $21.5bn, then fell to $5.2bn, separately from exposure values.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-R05.json`.

## H01 — held_out / agency_identity

What organizational and forecasting changes does the Commerce chapter propose for NOAA and the National Weather Service?

Evidence: PDF 707 / printed 674, PDF 708 / printed 675.

Agent source expectations:

- It proposes breaking up and downsizing NOAA.
- It proposes NWS emphasis on gathering data and commercialization of forecasting operations.
- Describe the chapter’s proposals and attributions; do not endorse its contested performance claims as independently established facts.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly attributes NOAA breakup/downsize and NWS data-gathering/commercialization proposals to the chapter and does not endorse forecast-performance assertions as established fact.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H01.json`.

## H02 — held_out / agency_identity

Where does the Commerce chapter propose transferring National Ocean Service survey functions?

Evidence: PDF 709 / printed 676.

Agent source expectations:

- U.S. Coast Guard and U.S. Geological Survey.
- The proposal concerns NOS survey functions, not transfer of all NOAA functions.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly identifies Coast Guard and USGS as destinations for NOS survey functions, not all NOAA functions.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H02.json`.

## H03 — held_out / proposed_vs_enacted

How does the Health and Human Services chapter propose reorganizing CDC data collection and public-health recommendations?

Evidence: PDF 485 / printed 452, PDF 486 / printed 453.

Agent source expectations:

- It proposes two separate agencies with a firewall: one publishing epidemiological data and one public-health agency with restricted policy-recommendation authority.
- This is a proposal in the document, not evidence that CDC has already been reorganized.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly describes two CDC entities separated by a firewall and constrained recommendation authority as proposals.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H03.json`.

## H04 — held_out / agency_identity

Which agency would receive VAERS, VSD and CISA vaccine-safety functions and their funding under this document’s proposal?

Evidence: PDF 487 / printed 454.

Agent source expectations:

- FDA; not NIH, a new CDC data agency, or another department.
- Attribute the proposed transfer to the document.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly identifies FDA as proposed recipient of VAERS, VSD, CISA functions and funding.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H04.json`.

## H05 — held_out / conditional_proposal

What does the financial-regulatory chapter propose for the CFPB, and what condition does it attach to presidential dissolution?

Evidence: PDF 872 / printed 839.

Agent source expectations:

- Congress should abolish CFPB and return consumer protection to banking regulators and the FTC.
- Presidential dissolution is conditional on the Supreme Court affirming the referenced Fifth Circuit holding.
- Do not present the book’s constitutional assertion or anticipated court outcome as a verified current legal fact.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly preserves congressional abolition proposal and conditional presidential dissolution dependent on the specified Supreme Court outcome. Constitutional claims remain attributed to the chapter, not current law.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H05.json`.

## H06 — held_out / unsupported_agreement

Does the Labor chapter present one unanimous view of worker voice and representation? Compare its main recommendation and Alternative View.

Evidence: PDF 632 / printed 599.

Agent source expectations:

- The main recommendation supports the TEAM Act and employee involvement organizations, including a non-voting supervisory board member for qualifying corporations.
- The Alternative View interprets workforce changes differently and favors the Worker’s Choice Act ending exclusive representation in right-to-work states.
- Do not erase the stated disagreement or invent named authors for the alternative paragraph.

Direct-context DeepSeek result: **pass** (agent source audit). Preserves TEAM Act/EIO main recommendation and Worker’s Choice Act alternative without inventing named authors or unanimity.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H06.json`.

## H07 — held_out / numerical_scope

What Commerce budget and staffing figures does the chapter use to describe NOAA, and what time qualification applies?

Evidence: PDF 707 / printed 674.

Agent source expectations:

- NOAA is described as receiving 6.5 billion of a 12 billion departmental annual operational budget.
- It accounts for more than half of personnel in non-decadal Census years; the cited figures are for 2021.
- Do not call these current 2026 budget facts or conflate budget share with a precise personnel percentage.

Direct-context DeepSeek result: **pass** (agent source audit). Preserves $6.5bn/$12bn, more-than-half staffing, non-decennial-Census qualification and 2021 date; does not claim current-2026 figures.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H07.json`.

## H08 — held_out / proposed_vs_enacted

Based only on this PDF, can you establish whether Congress has abolished the CFPB as of November 15, 2026?

Evidence: PDF 872 / printed 839.

Agent source expectations:

- No; the PDF contains policy recommendations and an anticipated court decision, not current enactment evidence.
- State the evidence limitation without asserting a current legal outcome.

Direct-context DeepSeek result: **pass** (agent source audit). Correctly declines to establish a November 2026 legal outcome from this proposal document; retains conditional/prospective status.

Human decision: pending. Retrieval is tested separately; this call supplied the frozen pages directly. Raw output: `/tmp/political-claim-replay-case-H08.json`.

## Fixed follow-up run

All 13 frozen source-page checks used `deepseek-flash` once, with expected answers withheld and thinking disabled. All returned structurally valid JSON and known passage IDs. Agent source audit: 11 passed, one failed attribution, one was incomplete because the frozen context omitted author headings. These are case dispositions, not a calibrated accuracy estimate. All human decisions remain pending.

The saved 30-chapter overview was replayed through the production reduction and synthesis helpers without rerunning its 117 extraction portions. It completed with 125 registered citation IDs, but introduced an unsupported Burton–Bowes disagreement in the final synthesis. Different SEC and CFPB recommendations do not establish opposing positions. The earlier Khan/Candeub false disagreement and FTC/FEC substitution were absent. Preserve the failed overview for review; do not label it verified.

The bounded run made 41 provider calls, without retries, using 145,863 prompt tokens and 26,350 completion tokens. Source SHA-256: `7f2e97507370bbadb109034ecda0085c5a2e5f3837cfb843da272c3a41523c4c`; frozen fixture SHA-256: `e13b0b5bc5ed9bfe4791f27c32cbc125643b0b3b599c70b146787ad2f837b7d0`. All frozen page hashes matched. Raw records are retained under `/tmp/political-claim-replay-*`, with the audit in `/tmp/political-claim-replay-audit.json`.

## Participant walkthrough

Use a fresh browser and the same PDF. Ask the participant to complete these tasks without coaching; record success, elapsed time, errors, confidence, and where assistance was needed. This script is prepared, not a completed user study.

1. Add the PDF, explain its processing state, and return after refresh.
2. Find the Education chapter summary, state its main proposal, and open its supporting physical PDF page.
3. Ask the FBI supervision question, identify both receiving offices, and check the cited passage.
4. Compare the two EXIM contributions while keeping each author’s argument separate.
5. Flag an unsupported overview claim, save a correction, reopen it, and distinguish the reviewer correction from the model draft.
6. Add an answer to Notes, explain whether it is only a draft, then save and reopen the Note.
7. Export the complete research pack, read it offline, and restore it on a second server. Check that source citations still open the correct PDF.

Human source review: mark each case Accept / Correct / Unclear, record the exact unsupported or missing claim and supporting page. Human product review: record observed friction under NN/g heuristics and severity 0–4; do not infer participant acceptance from an agent walkthrough.
