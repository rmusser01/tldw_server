# A05 configured OCR egress
Task: TASK-13260.278.18.83.45; parent A05 ingest plan. Frozen523ef323 plus preserved WIP, not latest dev.
ADR assessment: existing ADR026/ADR030 require checked egress and exact currently server-configured origin. No new rule. Scope comes from fresh OCR environment/runtime configuration, never request extras, final URL, or cached managed endpoint. Global denylist, metadata/DNS, redirects and public-capture restrictions remain.

## Stage1: Causal contract and security boundaries
**Goal:** Reproduce configured local OCR being rejected by checked transport.
**Success Criteria:** Real policy+HTTPX transport doubles show remote/managed origin failure; denylist/metadata/public-capture/unscoped calls and managed origin drift protected.
**Tests:** Normal backend tests with real egress validation, mock external wire only; no live provider draw beyond exhausted3 diagnostics.
**Status:** Complete

## Stage2: Minimal scoped calls and affected qualification
**Goal:** Pass existing ConfiguredEndpointScope at remote and managed trusted boundaries.
**Success Criteria:** Causal greens, complete affected backend/PDF/security tests; original mocks/assertions retained; scoped baseline/current Bandit/lint/syntax/source review.
**Tests:** Actual backend transport-policy tests and related configured endpoint/redirect contracts, normal pytest.
**Status:** Complete

## Stage3: Changed-source live acceptance
**Goal:** Qualify the actual corrected API/client route after real fix evidence.
**Success Criteria:** One guarded necessary API transition, one declared changed-source OCR validation with actual text/parser/owned identity. EarlierHTTP500/180s and all failed diagnostic/control outcomes remain. Root relationship to API failure must be verified, not assumed.
**Tests:** Corrected real source/normal chooser/provider/response/storage, no globalsafety weakening, source/timer/cache workaround or fake UAT. All broader A05 gates stay open.
**Status:** In Progress

Causal5failed6passed21deselected; complete backend/PDF41passed, followed only by two added unscoped-authority cases plus38existingconfigured-scope contracts40passed59deselected. Real central policy/HTTPX, external wire mocked; managed process lifecycle not live-qualified. ProductionBandit0->0/tests1->1 inherited0errors/no new; productionRuff one inheritedI001/tests0/syntax0. Independent complete changed source/all13case/fixture/cachedruntime/transport review CLEAR. Scope is derived before transport/runtime from fresh trusted configuration; global denylist, metadata/DNS, redirect and public-capture boundaries retained. Original UI500/180s cause remains unproven. Provider SSE11.653s/allfive facts and adapterEgressPolicyError0.501s are separate diagnostics only; three diagnostics exhausted, no further blind replay. New API39164/39161 stop/start operator identity/marker delta prepared, source review pending before ONE changed-source live validation. FrontendBgHH stays running; no install/build/model action or publication.

Changed-source API transition independentlyCLEAR, old39164/39161 stopped0.421s, new73132/73129 ready8.423s/exactparent/profile/listener. ExistingoneTab quiesced beforestop; no causal shutdown attribution. ONE corrected-egress scanned review ACK103ms stillactualbrowser500/plainbody/UI180s/0saved. Final result observed just after deadline, not accepted; postCount1. UTC-normalized current server log gives400/180034ms beforePDFhandler entry. Scope repair remains causaltransport-qualified but not a complete live500 fix. New source-attributed multipart caller unit is tracked separately in IMPLEMENTATION_PLAN_uat_a05_review_multipart_20261014.md. No further unchanged model diagnostic/Retry, no proxy timer/limit change.
