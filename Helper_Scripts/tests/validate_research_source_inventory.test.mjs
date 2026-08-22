import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import os from "node:os";
import path from "node:path";
import test from "node:test";
import { fileURLToPath } from "node:url";

import {
  canonicalJson,
  collectCertificationArtifacts,
  gateExitCode,
  parseAsOf,
  parseRepeatedOption,
  selectInventoryPython,
  sha256,
  validateInventoryDocuments,
} from "../validate_research_source_inventory.mjs";


const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..", "..");
const VALIDATOR_PATH = path.join(ROOT, "Helper_Scripts/validate_research_source_inventory.mjs");


function validDocuments() {
  const sourceRecord = {
    position: 1,
    label: "Example Source",
    url: "https://example.test/research",
    seed_categories: ["Example Category"],
  };
  const items = [{
    inventory_id: "sourclip-2026-07-13-0001",
    ...sourceRecord,
    row_sha256: sha256(canonicalJson(sourceRecord)),
  }];
  const manifest = {
    schema_version: "research-source-seed-manifest.v1",
    manifest_id: "sourclip-research-sources-2026-07-13",
    source: {
      page_url: "https://www.sourclip.com/resources/research-sources",
      captured_on: "2026-07-13",
      page_sha256: "a".repeat(64),
      extraction: "CollectionPage.mainEntity.ItemList",
      page_content_stored: false,
    },
    expected_item_count: 1,
    expected_category_placement_count: 1,
    items_sha256: sha256(canonicalJson(items)),
    items,
  };
  const ledger = {
    schema_version: "research-source-coverage-ledger.v2",
    manifest_id: manifest.manifest_id,
    manifest_items_sha256: manifest.items_sha256,
    closure_policy_version: "credentialless-public-v2",
    catalog_version: "research-discovery-v1",
    certification_policy_version: "discovery-egress-v1",
    target_definitions_sha256: sha256(canonicalJson([])),
    target_definitions: [],
    rows: [
      {
        inventory_id: items[0].inventory_id,
        source_snapshot_sha256: items[0].row_sha256,
        resolution: "unreviewed",
        resolution_code: "unreviewed_pending",
        resolution_reason: "Awaiting route and policy triage.",
        review_status: "unreviewed",
        route_kinds: [],
        route_candidates: [],
        capabilities: [],
        declared_surfaces: [],
        implementation_state: "planned",
        fixture_state: "not_run",
        live_state: "not_run",
        canonical_targets: [],
        duplicate_of_inventory_id: null,
        credentialless_route_review: null,
        closure_approval: null,
        certifications: [],
        evidence: [
          {
            kind: "seed_manifest",
            reference_type: "manifest_fragment",
            reference: `${manifest.manifest_id}#${items[0].inventory_id}`,
            claim: "Captured source identity and category placement.",
          },
        ],
        ownership: {
          reviewer: null,
          workstream: "TASK-12968.1",
          follow_up_task: "TASK-12968",
          review_date: null,
          revisit_trigger: "Route and policy triage required.",
        },
      },
    ],
  };
  ledger.rows_sha256 = sha256(canonicalJson(ledger.rows));
  return { manifest, ledger };
}


function freezeFor(manifest) {
  return {
    manifest_id: manifest.manifest_id,
    captured_on: manifest.source.captured_on,
    item_count: manifest.expected_item_count,
    category_placement_count: manifest.expected_category_placement_count,
    items_sha256: manifest.items_sha256,
    page_sha256: manifest.source.page_sha256,
  };
}


function refreshLedgerDigest(ledger) {
  ledger.target_definitions_sha256 = sha256(canonicalJson(ledger.target_definitions));
  ledger.rows_sha256 = sha256(canonicalJson(ledger.rows));
}


function mappedRoute({
  routeCandidateId = "example_direct",
  plannedBackendId = "example_api",
  evidenceReference = "https://example.test/api-docs",
  queryModes = ["general_free_text"],
} = {}) {
  return {
    route_candidate_id: routeCandidateId,
    route_kind: "direct",
    credential_requirement: "none",
    planned_backend_id: plannedBackendId,
    query_modes: queryModes,
    source_constraint: "native_corpus",
    source_constraint_predicate: null,
    attribution_basis: "native_response",
    coverage_notes: "Public source-constrained metadata discovery route.",
    evidence_reference: evidenceReference,
  };
}


function mapExampleRow(manifest, ledger, route = mappedRoute()) {
  const row = ledger.rows[0];
  row.resolution = "mapped";
  row.resolution_code = "credentialless_route_identified";
  row.resolution_reason = "Credentialless source-constrained route selected for both product surfaces.";
  row.review_status = "reviewed";
  row.route_kinds = [route.route_kind];
  row.route_candidates = [route];
  row.capabilities = ["search", "metadata", "snippet"];
  row.declared_surfaces = ["standalone_search", "deep_research"];
  row.canonical_targets = ["example_source"];
  row.ownership.reviewer = "research-maintainer";
  row.ownership.review_date = "2026-07-13";
  row.evidence.push({
    kind: "route_triage",
    reference_type: "https_url",
    reference: route.evidence_reference,
    claim: "Official route documentation supports the recorded discovery semantics.",
  });
  ledger.target_definitions = [{
    canonical_target_id: "example_source",
    display_name: "Example Source",
    inventory_ids: [manifest.items[0].inventory_id],
  }];
  refreshLedgerDigest(ledger);
  return row;
}


function exampleRoutePolicy() {
  return {
    allowed_methods: ["GET"],
    allowed_url_prefixes: ["https://example.test/api/"],
    allowed_transport_origins: ["https://example.test"],
    credential_mode: "none",
    gateway_required: true,
    result_link_dereference: false,
  };
}


function certificationArtifact({
  artifactType,
  row,
  surface,
  observedAtUtc = "2026-07-13T12:00:00Z",
}) {
  const routePolicy = exampleRoutePolicy();
  const common = {
    schema_version: "research-source-certification-artifact.v1",
    artifact_type: artifactType,
    route_candidate_id: row.route_candidates[0].route_candidate_id,
    canonical_target: row.canonical_targets[0],
    surface,
    route_candidate_sha256: sha256(canonicalJson(row.route_candidates[0])),
    route_policy_sha256: sha256(canonicalJson(routePolicy)),
    catalog_version: "research-discovery-v1",
    policy_version: "discovery-egress-v1",
    observed_at_utc: observedAtUtc,
    sanitized: true,
  };
  if (artifactType === "fixture") {
    return {
      ...common,
      outcome: "passed",
      details: {
        test_command: "python -m pytest tests/example_fixture.py -q",
        test_count: 8,
        fixture_cases: ["success", "valid_empty", "malformed", "partial_failure"],
      },
    };
  }
  if (artifactType === "live") {
    return {
      ...common,
      outcome: "passed",
      details: {
        checked_endpoint: "https://example.test/api/search",
        request_method: "GET",
        request_count: 1,
        result_count: 1,
        transport_origins: ["https://example.test"],
        gateway_attested: true,
        credential_mode: "none",
        result_link_dereference_count: 0,
      },
    };
  }
  return {
    ...common,
    outcome: "allowed",
    details: {
      terms_url: "https://example.test/terms",
      robots_url: "https://example.test/robots.txt",
      reviewer: "policy-maintainer",
      decision_notes: "Metadata-only discovery is allowed under the reviewed route policy.",
      route_policy: routePolicy,
    },
  };
}


test("validates a complete reconciled manifest and ledger", () => {
  const { manifest, ledger } = validDocuments();

  const report = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    schema: { $id: "test-schema" },
    schemaValidated: true,
    requiredSources: {},
    requiredImplementedSources: {},
  });

  assert.deepEqual(report.errors, []);
  assert.equal(report.counts.manifest, 1);
  assert.equal(report.counts.ledger, 1);
  assert.equal(report.counts.unreviewed, 1);
  assert.equal(report.as_of, "2026-07-13");
  assert.equal(report.counts.terminal, 0);
  assert.equal(report.contract_freeze_ready, false);
  assert.equal(report.inventory_delivery_ready, false);
  assert.match(report.digests.manifest, /^[a-f0-9]{64}$/);
  assert.match(report.digests.ledger, /^[a-f0-9]{64}$/);
  assert.match(report.digests.schema, /^[a-f0-9]{64}$/);
});


test("rejects digest drift and duplicate ledger rows", () => {
  const { manifest, ledger } = validDocuments();
  manifest.items[0].label = "Changed after digest";
  ledger.rows.push(structuredClone(ledger.rows[0]));
  refreshLedgerDigest(ledger);

  const report = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
  });

  assert.ok(report.errors.includes("manifest items_sha256 does not match canonical items"));
  assert.ok(
    report.errors.includes(
      "ledger contains duplicate inventory_id sourclip-2026-07-13-0001",
    ),
  );
});


test("reports a null manifest items container without throwing", () => {
  const { manifest, ledger } = validDocuments();
  manifest.items = null;
  let report;

  assert.doesNotThrow(() => {
    report = validateInventoryDocuments(manifest, ledger, {
      freeze: freezeFor(manifest),
      requiredSources: {},
      requiredImplementedSources: {},
      schemaValidated: true,
    });
  });
  assert.ok(report.errors.includes("manifest items must be an array"));
});


test("reports null manifest item entries without throwing", () => {
  const { manifest, ledger } = validDocuments();
  manifest.items = [null];
  let report;

  assert.doesNotThrow(() => {
    report = validateInventoryDocuments(manifest, ledger, {
      freeze: freezeFor(manifest),
      requiredSources: {},
      requiredImplementedSources: {},
      schemaValidated: true,
    });
  });
  assert.ok(report.errors.includes("manifest item 1 must be an object"));
});


test("does not retain target definitions with non-array inventory IDs", () => {
  const { manifest, ledger } = validDocuments();
  ledger.target_definitions = [{
    canonical_target_id: "example_source",
    display_name: "Example Source",
    inventory_ids: { invalid: true },
  }];
  refreshLedgerDigest(ledger);
  let report;

  assert.doesNotThrow(() => {
    report = validateInventoryDocuments(manifest, ledger, {
      freeze: freezeFor(manifest),
      requiredSources: {},
      requiredImplementedSources: {},
      schemaValidated: true,
    });
  });
  assert.ok(
    report.errors.includes(
      "target definition example_source requires unique inventory_ids",
    ),
  );
});


test("requires current source-route-surface certification before mapped rows close", () => {
  const { manifest, ledger } = validDocuments();
  const row = mapExampleRow(manifest, ledger);
  row.implementation_state = "implemented";
  refreshLedgerDigest(ledger);

  const options = {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
    artifactDigests: {},
    trustedReviewerIds: ["research-maintainer"],
  };
  const beforeCertification = validateInventoryDocuments(manifest, ledger, options);
  assert.deepEqual(beforeCertification.errors, []);
  assert.equal(beforeCertification.contract_freeze_ready, true);
  assert.equal(beforeCertification.counts.terminal, 0);
  assert.equal(beforeCertification.inventory_delivery_ready, false);

  row.implementation_state = "implemented";
  row.fixture_state = "passed";
  row.live_state = "current";
  const artifactDigests = {};
  const certificationArtifacts = {};
  row.certifications = row.declared_surfaces.map((surface) => {
    const fixtureEvidence = `Docs/Design/research_source_inventory/certifications/example-source-${surface}-fixture.json`;
    const liveEvidence = `Docs/Design/research_source_inventory/certifications/example-source-${surface}-live.json`;
    const policyEvidence = `Docs/Design/research_source_inventory/certifications/example-source-${surface}-policy.json`;
    certificationArtifacts[fixtureEvidence] = certificationArtifact({
      artifactType: "fixture",
      row,
      surface,
    });
    certificationArtifacts[liveEvidence] = certificationArtifact({
      artifactType: "live",
      row,
      surface,
    });
    certificationArtifacts[policyEvidence] = certificationArtifact({
      artifactType: "policy",
      row,
      surface,
    });
    artifactDigests[fixtureEvidence] = sha256(canonicalJson(
      certificationArtifacts[fixtureEvidence],
    ));
    artifactDigests[liveEvidence] = sha256(canonicalJson(
      certificationArtifacts[liveEvidence],
    ));
    artifactDigests[policyEvidence] = sha256(canonicalJson(
      certificationArtifacts[policyEvidence],
    ));
    return {
    route_candidate_id: "example_direct",
    canonical_target: "example_source",
    surface,
    route_candidate_sha256: sha256(canonicalJson(row.route_candidates[0])),
    route_policy_sha256: sha256(canonicalJson(exampleRoutePolicy())),
    catalog_version: "research-discovery-v1",
    policy_version: "discovery-egress-v1",
    fixture_evidence: fixtureEvidence,
    fixture_sha256: artifactDigests[fixtureEvidence],
    live_evidence: liveEvidence,
    live_sha256: artifactDigests[liveEvidence],
    policy_evidence: policyEvidence,
    certified_on: "2026-07-13",
    valid_until: "2026-08-12",
    policy_digest: artifactDigests[policyEvidence],
    };
  });
  refreshLedgerDigest(ledger);

  const missingArtifactContent = validateInventoryDocuments(manifest, ledger, {
    ...options,
    artifactDigests,
  });
  assert.ok(missingArtifactContent.errors.some((error) => error.includes("artifact content")));
  assert.equal(missingArtifactContent.counts.terminal, 0);
  assert.equal(missingArtifactContent.inventory_delivery_ready, false);

  const afterCertification = validateInventoryDocuments(manifest, ledger, {
    ...options,
    artifactDigests,
    certificationArtifacts,
  });
  assert.deepEqual(afterCertification.errors, []);
  assert.equal(afterCertification.counts.terminal, 1);
  assert.equal(afterCertification.contract_freeze_ready, true);
  assert.equal(afterCertification.inventory_delivery_ready, true);

  const firstCertification = row.certifications[0];
  const firstLiveArtifact = certificationArtifacts[firstCertification.live_evidence];
  const refreshFirstLiveEvidence = () => {
    artifactDigests[firstCertification.live_evidence] = sha256(canonicalJson(firstLiveArtifact));
    firstCertification.live_sha256 = artifactDigests[firstCertification.live_evidence];
    refreshLedgerDigest(ledger);
  };

  firstLiveArtifact.details.result_count = 0;
  refreshFirstLiveEvidence();
  const emptyLiveResult = validateInventoryDocuments(manifest, ledger, {
    ...options,
    artifactDigests,
    certificationArtifacts,
  });
  assert.ok(emptyLiveResult.errors.some((error) => error.includes("live artifact content is invalid")));
  assert.equal(emptyLiveResult.inventory_delivery_ready, false);

  firstLiveArtifact.details.result_count = 1;
  firstLiveArtifact.details.checked_endpoint = "https://unrelated.test/api/search";
  firstLiveArtifact.details.transport_origins = ["https://unrelated.test"];
  refreshFirstLiveEvidence();
  const unrelatedEndpoint = validateInventoryDocuments(manifest, ledger, {
    ...options,
    artifactDigests,
    certificationArtifacts,
  });
  assert.ok(unrelatedEndpoint.errors.some((error) => error.includes("live artifact content is invalid")));
  assert.equal(unrelatedEndpoint.inventory_delivery_ready, false);

  firstLiveArtifact.details.checked_endpoint = "https://example.test/api/search";
  firstLiveArtifact.details.transport_origins = ["https://example.test"];
  firstLiveArtifact.details.gateway_attested = false;
  firstLiveArtifact.details.result_link_dereference_count = 1;
  refreshFirstLiveEvidence();
  const bypassedGatewayPolicy = validateInventoryDocuments(manifest, ledger, {
    ...options,
    artifactDigests,
    certificationArtifacts,
  });
  assert.ok(bypassedGatewayPolicy.errors.some((error) => error.includes("live artifact content is invalid")));
  assert.equal(bypassedGatewayPolicy.inventory_delivery_ready, false);

  firstLiveArtifact.details.gateway_attested = true;
  firstLiveArtifact.details.result_link_dereference_count = 0;
  refreshFirstLiveEvidence();

  row.certifications[0].catalog_version = "unrelated-catalog";
  row.certifications[0].policy_version = "unrelated-policy";
  refreshLedgerDigest(ledger);
  const versionDrift = validateInventoryDocuments(manifest, ledger, {
    ...options,
    artifactDigests,
    certificationArtifacts,
  });
  assert.ok(versionDrift.errors.some((error) => error.includes("catalog_version")));
  assert.ok(versionDrift.errors.some((error) => error.includes("policy_version")));
  assert.equal(versionDrift.counts.terminal, 0);
  assert.equal(versionDrift.inventory_delivery_ready, false);

  for (const certification of row.certifications) certification.valid_until = "9999-12-31";
  refreshLedgerDigest(ledger);
  const unboundedValidity = validateInventoryDocuments(manifest, ledger, {
    ...options,
    artifactDigests,
    certificationArtifacts,
  });
  assert.ok(unboundedValidity.errors.some((error) => error.includes("validity horizon")));
  assert.equal(unboundedValidity.counts.terminal, 0);
});


test("rejects route kinds that misstate source constraint and attribution", () => {
  const { manifest, ledger } = validDocuments();
  const route = mappedRoute();
  route.route_kind = "aggregator";
  route.source_constraint = "native_corpus";
  route.attribution_basis = "native_response";
  mapExampleRow(manifest, ledger, route);

  const report = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });

  assert.ok(report.errors.some((error) => error.includes("aggregator route")));
  assert.equal(report.contract_freeze_ready, false);
});


test("aggregator routes require a machine-readable source predicate", () => {
  const { manifest, ledger } = validDocuments();
  const route = mappedRoute();
  route.route_kind = "aggregator";
  route.source_constraint = "provider_source_filter";
  route.attribution_basis = "provider_source_field";
  mapExampleRow(manifest, ledger, route);

  const options = {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  };
  const missingPredicate = validateInventoryDocuments(manifest, ledger, options);
  assert.ok(missingPredicate.errors.some((error) => error.includes("source predicate")));
  assert.equal(missingPredicate.contract_freeze_ready, false);

  route.source_constraint_predicate = {
    provider_field: "prefix",
    operator: "equals",
    values: ["10.1234"],
  };
  refreshLedgerDigest(ledger);
  const constrained = validateInventoryDocuments(manifest, ledger, options);
  assert.deepEqual(constrained.errors, []);
  assert.equal(constrained.contract_freeze_ready, true);
});


test("credentialed exclusions do not require invented route candidates", () => {
  const { manifest, ledger } = validDocuments();
  const row = ledger.rows[0];
  row.resolution = "credentialed_out_of_scope";
  row.resolution_code = "credential_required_no_public_route";
  row.resolution_reason = "The licensed corpus requires a subscription and no concrete authenticated route contract was identified.";
  row.review_status = "reviewed";
  row.ownership.reviewer = "research-maintainer";
  row.ownership.review_date = "2026-07-13";
  row.ownership.follow_up_task = "TASK-12969";
  row.credentialless_route_review = [
    {
      route_kind: "direct",
      finding: "not_identified",
      evidence_reference: "https://example.test/access",
      notes: "No credentialless direct API route was identified during review.",
    },
    {
      route_kind: "aggregator",
      finding: "not_source_faithful",
      evidence_reference: "https://example.test/access",
      notes: "Substitute aggregators would not faithfully represent the licensed corpus.",
    },
    {
      route_kind: "site_search",
      finding: "credential_required",
      evidence_reference: "https://example.test/access",
      notes: "The native search experience requires an authenticated subscription.",
    },
  ];
  row.evidence.push({
    kind: "resolution_review",
    reference_type: "https_url",
    reference: "https://example.test/access",
    claim: "The reviewed access documentation identifies a subscription-only corpus.",
  });
  refreshLedgerDigest(ledger);

  const report = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });

  assert.deepEqual(report.errors, []);
  assert.equal(report.counts.triaged, 1);
  assert.equal(report.contract_freeze_ready, true);
  assert.equal(report.inventory_delivery_ready, false);

  row.canonical_targets = ["licensed_source", "second_licensed_source"];
  ledger.target_definitions = row.canonical_targets.map((canonicalTargetId) => ({
    canonical_target_id: canonicalTargetId,
    display_name: canonicalTargetId,
    inventory_ids: [row.inventory_id],
  }));
  refreshLedgerDigest(ledger);
  const ambiguousTargets = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.ok(ambiguousTargets.errors.some((error) => error.includes("at most one canonical target")));
  assert.equal(ambiguousTargets.contract_freeze_ready, false);
});


test("contract freeze records an explicit trusted reviewer boundary", () => {
  const { manifest, ledger } = validDocuments();
  mapExampleRow(manifest, ledger);
  const options = {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
  };

  const untrusted = validateInventoryDocuments(manifest, ledger, options);
  assert.deepEqual(untrusted.errors, []);
  assert.equal(untrusted.counts.triaged, 0);
  assert.equal(untrusted.contract_freeze_ready, false);

  const trusted = validateInventoryDocuments(manifest, ledger, {
    ...options,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(trusted.errors, []);
  assert.deepEqual(trusted.trusted_reviewer_ids, ["research-maintainer"]);
  assert.equal(trusted.counts.triaged, 1);
  assert.equal(trusted.contract_freeze_ready, true);
});


test("the canonical closure gate rejects a structurally valid one-row substitute", () => {
  const { manifest, ledger } = validDocuments();

  const report = validateInventoryDocuments(manifest, ledger);

  assert.ok(report.errors.includes("frozen manifest item count must be 235"));
  assert.equal(report.contract_freeze_ready, false);
  assert.equal(report.inventory_delivery_ready, false);
});


test("required sources require exact core routes without excluding reviewed additions", () => {
  const { manifest, ledger } = validDocuments();
  const publisherPredicate = {
    provider_field: "bookOrReportDetails.publisher",
    operator: "equals",
    values: ["bioRxiv"],
  };
  const requiredSources = {
    "sourclip-2026-07-13-0001": {
      canonicalTarget: "biorxiv",
      generalRoute: {
        id: "biorxiv_europe_pmc_search_aggregator",
        routeKind: "aggregator",
        backendId: "europe_pmc_rest_api",
        queryModes: ["general_free_text"],
        sourceConstraint: "provider_source_filter",
        sourcePredicate: publisherPredicate,
        attributionBasis: "provider_source_field",
        evidenceHosts: ["example.test"],
      },
      lookupRoute: {
        id: "biorxiv_details_lookup_direct",
        routeKind: "direct",
        backendId: "biorxiv_details_api",
        queryModes: ["identifier_lookup"],
        sourceConstraint: "native_corpus",
        sourcePredicate: null,
        attributionBasis: "native_response",
        evidenceHosts: ["api.example.test"],
      },
      intervalRoute: {
        id: "biorxiv_details_interval_direct",
        routeKind: "direct",
        backendId: "biorxiv_details_api",
        queryModes: ["date_interval", "category_browse"],
        sourceConstraint: "native_corpus",
        sourcePredicate: null,
        attributionBasis: "native_response",
        evidenceHosts: ["api.example.test"],
      },
    },
  };

  const unreviewed = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(unreviewed.errors, []);
  assert.equal(unreviewed.structurally_valid, true);
  assert.equal(unreviewed.contract_freeze_ready, false);
  assert.equal(
    unreviewed.required_sources["sourclip-2026-07-13-0001"].mapping_satisfied,
    false,
  );

  const row = mapExampleRow(manifest, ledger);
  row.canonical_targets = ["wrong_target"];
  ledger.target_definitions[0].canonical_target_id = "wrong_target";
  refreshLedgerDigest(ledger);
  const wrongTarget = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(wrongTarget.errors, []);
  assert.equal(wrongTarget.counts.triaged, 1);
  assert.equal(wrongTarget.contract_freeze_ready, false);

  row.canonical_targets = ["biorxiv"];
  ledger.target_definitions[0].canonical_target_id = "biorxiv";
  refreshLedgerDigest(ledger);
  const correctTargetWrongRoute = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(correctTargetWrongRoute.errors, []);
  assert.equal(correctTargetWrongRoute.contract_freeze_ready, false);

  row.route_candidates = [
    {
      ...mappedRoute({
        routeCandidateId: "biorxiv_europe_pmc_search_aggregator",
        plannedBackendId: "europe_pmc_rest_api",
        evidenceReference: "https://example.test/api",
      }),
      route_kind: "aggregator",
      source_constraint: "provider_source_filter",
      source_constraint_predicate: structuredClone(publisherPredicate),
      attribution_basis: "provider_source_field",
    },
    {
      ...mappedRoute({
        routeCandidateId: "biorxiv_details_lookup_direct",
        plannedBackendId: "biorxiv_details_api",
        evidenceReference: "https://api.example.test/details/biorxiv/help",
        queryModes: ["identifier_lookup"],
      }),
      route_kind: "direct",
    },
    {
      ...mappedRoute({
        routeCandidateId: "biorxiv_details_interval_direct",
        plannedBackendId: "biorxiv_details_api",
        evidenceReference: "https://api.example.test/details/biorxiv/help",
        queryModes: ["date_interval", "category_browse"],
      }),
      route_kind: "direct",
    },
  ];
  row.route_kinds = ["aggregator", "direct"];
  refreshLedgerDigest(ledger);
  const completeMapping = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(completeMapping.errors, []);
  assert.equal(completeMapping.contract_freeze_ready, true);
  const requiredState = completeMapping.required_sources["sourclip-2026-07-13-0001"];
  assert.deepEqual(requiredState, {
    canonical_target: "biorxiv",
    required_general_route_id: "biorxiv_europe_pmc_search_aggregator",
    required_lookup_route_id: "biorxiv_details_lookup_direct",
    required_interval_route_id: "biorxiv_details_interval_direct",
    captured_label: "Example Source",
    resolution: "mapped",
    canonical_targets: ["biorxiv"],
    declared_surfaces: ["standalone_search", "deep_research"],
    mapping_satisfied: true,
  });

  row.route_candidates.push({
    ...mappedRoute({
      routeCandidateId: "biorxiv_authenticated_archive_search",
      plannedBackendId: "biorxiv_authenticated_browser",
      evidenceReference: "https://example.test/authenticated-search",
    }),
    route_kind: "site_search",
    credential_requirement: "browser_session",
  });
  row.route_kinds = ["aggregator", "direct", "site_search"];
  refreshLedgerDigest(ledger);
  const futureAuthenticatedCandidate = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.equal(
    futureAuthenticatedCandidate.required_sources["sourclip-2026-07-13-0001"]
      .mapping_satisfied,
    true,
  );
  row.route_candidates.pop();
  row.route_kinds = ["aggregator", "direct"];

  row.route_candidates.pop();
  refreshLedgerDigest(ledger);
  const missingIntervalRoute = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.equal(missingIntervalRoute.contract_freeze_ready, false);

  row.route_candidates.push({
    ...mappedRoute({
      routeCandidateId: "biorxiv_details_interval_direct",
      plannedBackendId: "biorxiv_details_api",
      evidenceReference: "https://api.example.test/details/biorxiv/help",
      queryModes: ["recent_feed", "date_interval", "category_browse"],
    }),
    route_kind: "direct",
  });
  refreshLedgerDigest(ledger);
  const intervalRouteAdvertisesRecent = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.equal(intervalRouteAdvertisesRecent.contract_freeze_ready, false);

  row.route_candidates[2].query_modes = ["date_interval", "category_browse"];
  row.route_candidates[0].source_constraint_predicate.values = ["medRxiv"];
  refreshLedgerDigest(ledger);
  const wrongPublisher = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(wrongPublisher.errors, []);
  assert.equal(wrongPublisher.contract_freeze_ready, false);

  for (const predicate of [
    {
      provider_field: "source",
      operator: "equals",
      values: ["bioRxiv"],
    },
    {
      provider_field: "bookOrReportDetails.publisher",
      operator: "one_of",
      values: ["bioRxiv"],
    },
    {
      operator: "equals",
      values: ["bioRxiv"],
    },
  ]) {
    row.route_candidates[0].source_constraint_predicate = predicate;
    refreshLedgerDigest(ledger);
    const predicateMismatch = validateInventoryDocuments(manifest, ledger, {
      freeze: freezeFor(manifest),
      requiredSources,
      requiredImplementedSources: {},
      schemaValidated: true,
      trustedReviewerIds: ["research-maintainer"],
    });
    assert.equal(
      predicateMismatch.required_sources["sourclip-2026-07-13-0001"]
        .mapping_satisfied,
      false,
    );
    assert.deepEqual(predicateMismatch.required_implemented_source_blockers, []);
  }
});


test("implemented sources require exact shadow evidence and report blockers for every drift", () => {
  const inventoryId = "sourclip-2026-07-13-0001";
  const implementationRevision = "8bbff7820f0d05a2c25d0f2561b0241d0024d5d9";
  const implementationPath = "tldw_Server_API/tests/Research/test_research_discovery_clinicaltrials_pubmed_central.py";
  const implementationReference = `https://github.com/rmusser01/tldw_server/blob/${implementationRevision}/${implementationPath}`;
  const implementationClaim = "The fixture-only adapter implements the reviewed bounded metadata projection.";

  const documents = () => {
    const { manifest, ledger } = validDocuments();
    const route = mappedRoute({
      routeCandidateId: "clinicaltrials_gov_studies_search_direct",
      plannedBackendId: "clinicaltrials_gov_api_v2",
      evidenceReference: "https://clinicaltrials.gov/data-api/api",
    });
    const row = mapExampleRow(manifest, ledger, route);
    row.capabilities = ["search", "detail", "metadata", "snippet"];
    row.implementation_state = "implemented";
    row.fixture_state = "passed";
    row.live_state = "not_run";
    row.evidence.push({
      kind: "implementation",
      reference_type: "https_url",
      reference: implementationReference,
      claim: implementationClaim,
    });
    refreshLedgerDigest(ledger);
    const requirement = {
      sourceSnapshotSha256: manifest.items[0].row_sha256,
      canonicalTarget: "example_source",
      declaredSurfaces: ["standalone_search", "deep_research"],
      capabilities: ["search", "detail", "metadata", "snippet"],
      route: {
        id: "clinicaltrials_gov_studies_search_direct",
        routeKind: "direct",
        backendId: "clinicaltrials_gov_api_v2",
        queryModes: ["general_free_text"],
        sourceConstraint: "native_corpus",
        sourcePredicate: null,
        attributionBasis: "native_response",
        evidenceHosts: ["clinicaltrials.gov"],
      },
      implementationState: "implemented",
      fixtureState: "passed",
      liveState: "not_run",
      certifications: [],
      implementationEvidence: {
        referenceType: "https_url",
        host: "github.com",
        path: implementationPath,
        revision: implementationRevision,
        reference: implementationReference,
        claim: implementationClaim,
      },
    };
    return { manifest, ledger, row, requirement };
  };

  const validate = ({ manifest, ledger, requirement }) => validateInventoryDocuments(
    manifest,
    ledger,
    {
      freeze: freezeFor(manifest),
      asOf: "2026-07-13",
      requiredSources: {},
      requiredImplementedSources: { [inventoryId]: requirement },
      schemaValidated: true,
      trustedReviewerIds: ["research-maintainer"],
    },
  );

  const valid = documents();
  const report = validate(valid);
  assert.deepEqual(report.errors, []);
  assert.deepEqual(report.required_implemented_source_blockers, []);
  assert.equal(report.contract_freeze_ready, true);
  assert.deepEqual(report.required_implemented_sources[inventoryId], {
    source_snapshot_sha256: valid.requirement.sourceSnapshotSha256,
    canonical_target: "example_source",
    required_route_id: "clinicaltrials_gov_studies_search_direct",
    captured_label: "Example Source",
    resolution: "mapped",
    canonical_targets: ["example_source"],
    declared_surfaces: ["standalone_search", "deep_research"],
    capabilities: ["search", "detail", "metadata", "snippet"],
    implementation_state: "implemented",
    fixture_state: "passed",
    live_state: "not_run",
    certifications: [],
    implementation_evidence: true,
    implementation_evidence_identity: {
      required: {
        reference_type: "https_url",
        host: "github.com",
        path: implementationPath,
        revision: implementationRevision,
        reference: implementationReference,
        claim: implementationClaim,
      },
      entries: [{
        kind: "implementation",
        reference_type: "https_url",
        host: "github.com",
        path: implementationPath,
        revision: implementationRevision,
        reference: implementationReference,
        claim: implementationClaim,
      }],
      implementation_entry_count: 1,
      exact_match_count: 1,
    },
    substantively_triaged: true,
    implementation_satisfied: true,
  });

  const mutations = [
    ["snapshot", ({ row }) => { row.source_snapshot_sha256 = "f".repeat(64); }, false],
    ["target", ({ ledger, row }) => {
      row.canonical_targets = ["wrong_target"];
      ledger.target_definitions[0].canonical_target_id = "wrong_target";
    }, true],
    ["surfaces", ({ row }) => { row.declared_surfaces.reverse(); }, true],
    ["capabilities", ({ row }) => { row.capabilities.pop(); }, true],
    ["route.id", ({ row }) => {
      row.route_candidates[0].route_candidate_id = "wrong_direct";
    }, true],
    ["route.routeKind", ({ row }) => {
      row.route_candidates[0].route_kind = "site_search";
      row.route_kinds = ["site_search"];
    }, true],
    ["route.backendId", ({ row }) => {
      row.route_candidates[0].planned_backend_id = "wrong_api";
    }, true],
    ["route.queryModes", ({ row }) => {
      row.route_candidates[0].query_modes.push("identifier_lookup");
    }, true],
    ["route.sourceConstraint", ({ row }) => {
      row.route_candidates[0].route_kind = "aggregator";
      row.route_kinds = ["aggregator"];
      row.route_candidates[0].source_constraint = "provider_source_filter";
      row.route_candidates[0].source_constraint_predicate = {
        provider_field: "source",
        operator: "equals",
        values: ["ClinicalTrials.gov"],
      };
      row.route_candidates[0].attribution_basis = "provider_source_field";
    }, true],
    ["route.sourcePredicate", ({ row }) => {
      row.route_candidates[0].route_kind = "aggregator";
      row.route_kinds = ["aggregator"];
      row.route_candidates[0].source_constraint = "provider_source_filter";
      row.route_candidates[0].source_constraint_predicate = {
        provider_field: "source",
        operator: "one_of",
        values: ["ClinicalTrials.gov"],
      };
      row.route_candidates[0].attribution_basis = "provider_source_field";
    }, true],
    ["route.attributionBasis", ({ row }) => {
      row.route_candidates[0].route_kind = "aggregator";
      row.route_kinds = ["aggregator"];
      row.route_candidates[0].source_constraint = "provider_domain_filter";
      row.route_candidates[0].source_constraint_predicate = {
        provider_field: "url",
        operator: "domain_suffix",
        values: ["clinicaltrials.gov"],
      };
      row.route_candidates[0].attribution_basis = "verified_reported_origin";
    }, true],
    ["route.evidenceHosts", ({ row }) => {
      row.route_candidates[0].evidence_reference = "https://example.test/api";
    }, true],
    ["implementation_state", ({ row }) => { row.implementation_state = "planned"; }, true],
    ["fixture_state", ({ row }) => { row.fixture_state = "not_run"; }, true],
    ["live_state", ({ row }) => { row.live_state = "current"; }, true],
    ["certifications", ({ row }) => { row.certifications = [{}]; }, false],
    ["implementation evidence", ({ row }) => {
      row.evidence = row.evidence.filter((entry) => entry.kind !== "implementation");
    }, true],
    ["substantive triage", ({ row }) => { row.ownership.reviewer = "untrusted-reviewer"; }, true],
  ];

  for (const [label, mutate, structurallyValid] of mutations) {
    const mutated = documents();
    mutate(mutated);
    refreshLedgerDigest(mutated.ledger);
    const drift = validate(mutated);
    if (structurallyValid) assert.deepEqual(drift.errors, [], label);
    else assert.ok(drift.errors.length > 0, label);
    assert.deepEqual(drift.required_implemented_source_blockers, [inventoryId], label);
    assert.equal(drift.required_implemented_sources[inventoryId].implementation_satisfied, false, label);
    assert.equal(drift.contract_freeze_ready, false, label);
  }
});


test("authoritative implemented rows require one exact immutable implementation evidence entry", () => {
  const inventoryDirectory = path.join(
    ROOT,
    "Docs/Design/research_source_inventory",
  );
  const manifest = JSON.parse(fs.readFileSync(path.join(
    inventoryDirectory,
    "sourclip-research-sources-2026-07-13.json",
  ), "utf8"));
  const authoritativeLedger = JSON.parse(fs.readFileSync(path.join(
    inventoryDirectory,
    "research-source-coverage-ledger-2026-07-13.json",
  ), "utf8"));
  const inventoryIds = [
    "sourclip-2026-07-13-0026",
    "sourclip-2026-07-13-0027",
  ];
  const mutations = [
    ["reference type", (row, entry) => {
      entry.reference_type = "repo_path";
    }, 1, 0],
    ["host", (row, entry) => {
      entry.reference = entry.reference.replace("github.com", "gitlab.com");
    }, 1, 0],
    ["path", (row, entry) => {
      entry.reference = entry.reference.replace(
        "test_research_discovery_clinicaltrials_pubmed_central.py",
        "test_research_discovery_contracts.py",
      );
    }, 1, 0],
    ["revision", (row, entry) => {
      entry.reference = entry.reference.replace(
        "8bbff7820f0d05a2c25d0f2561b0241d0024d5d9",
        "0000000000000000000000000000000000000000",
      );
    }, 1, 0],
    ["claim", (row, entry) => {
      entry.claim = `${entry.claim} This is not the frozen row-specific claim.`;
    }, 1, 0],
    ["duplicate", (row, entry) => {
      row.evidence.push(structuredClone(entry));
    }, 2, 2],
    ["unrelated implementation evidence", (row) => {
      row.evidence.push({
        kind: "implementation",
        reference_type: "https_url",
        reference: "https://github.com/rmusser01/tldw_server/blob/8bbff7820f0d05a2c25d0f2561b0241d0024d5d9/README.md",
        claim: "An unrelated checked-in artifact cannot certify this implemented inventory row.",
      });
    }, 2, 1],
  ];

  for (const inventoryId of inventoryIds) {
    for (const [label, mutate, expectedEntries, expectedMatches] of mutations) {
      const ledger = structuredClone(authoritativeLedger);
      const row = ledger.rows.find((candidate) => candidate.inventory_id === inventoryId);
      const entry = row.evidence.find((candidate) => candidate.kind === "implementation");
      mutate(row, entry);
      refreshLedgerDigest(ledger);

      const report = validateInventoryDocuments(manifest, ledger, {
        asOf: "2026-07-15",
        schemaValidated: true,
        trustedReviewerIds: [
          "codex-task-12968.1-source-triage",
          "codex-task-12968.5-inventory-review",
        ],
      });
      const state = report.required_implemented_sources[inventoryId];

      assert.deepEqual(
        report.required_implemented_source_blockers,
        [inventoryId],
        `${inventoryId}: ${label}`,
      );
      assert.equal(state.implementation_evidence, false, `${inventoryId}: ${label}`);
      assert.equal(
        state.implementation_evidence_identity.implementation_entry_count,
        expectedEntries,
        `${inventoryId}: ${label}`,
      );
      assert.equal(
        state.implementation_evidence_identity.exact_match_count,
        expectedMatches,
        `${inventoryId}: ${label}`,
      );
      assert.equal(state.implementation_satisfied, false, `${inventoryId}: ${label}`);
      assert.equal(report.contract_freeze_ready, false, `${inventoryId}: ${label}`);
    }
  }
});


test("exclusions close only with an externally trusted matching approval", () => {
  const { manifest, ledger } = validDocuments();
  const row = ledger.rows[0];
  row.resolution = "policy_blocked";
  row.resolution_code = "automation_prohibited";
  row.resolution_reason = "The reviewed provider policy prohibits automated discovery through this route.";
  row.review_status = "reviewed";
  row.ownership.reviewer = "research-maintainer";
  row.ownership.review_date = "2026-07-12";
  row.ownership.follow_up_task = "TASK-12968";
  row.evidence.push({
    kind: "resolution_review",
    reference_type: "https_url",
    reference: "https://example.test/terms",
    claim: "The reviewed terms prohibit automated access to the searchable corpus.",
  });
  const decision = {
    inventory_id: row.inventory_id,
    resolution: row.resolution,
    resolution_code: row.resolution_code,
    resolution_reason: row.resolution_reason,
    canonical_targets: row.canonical_targets,
    duplicate_of_inventory_id: row.duplicate_of_inventory_id,
    route_candidates: row.route_candidates,
    credentialless_route_review: row.credentialless_route_review,
    evidence: row.evidence,
  };
  row.closure_approval = {
    approved_by: "research-owner",
    approved_on: "2026-07-13",
    approval_reference_type: "backlog_task",
    approval_reference: "TASK-12968",
    decision_sha256: sha256(canonicalJson(decision)),
  };
  refreshLedgerDigest(ledger);

  const options = {
    freeze: freezeFor(manifest),
    asOf: "2026-07-13",
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  };
  const untrusted = validateInventoryDocuments(manifest, ledger, options);
  assert.deepEqual(untrusted.errors, []);
  assert.equal(untrusted.counts.terminal, 0);
  assert.equal(untrusted.inventory_delivery_ready, false);

  const trusted = validateInventoryDocuments(manifest, ledger, {
    ...options,
    trustedApprovalReferences: ["TASK-12968"],
  });
  assert.deepEqual(trusted.errors, []);
  assert.equal(trusted.counts.terminal, 1);
  assert.equal(trusted.inventory_delivery_ready, true);

  row.closure_approval.approval_reference = "TASK-1";
  refreshLedgerDigest(ledger);
  const wrongTask = validateInventoryDocuments(manifest, ledger, {
    ...options,
    trustedApprovalReferences: ["TASK-1"],
  });
  assert.ok(wrongTask.errors.some((error) => error.includes("follow_up_task")));
  assert.equal(wrongTask.counts.terminal, 0);
});


test("rejects superficial terminal exclusions instead of freezing them", () => {
  const { manifest, ledger } = validDocuments();
  const row = ledger.rows[0];
  row.resolution = "technically_infeasible";
  row.resolution_code = "no_stable_live_route";
  row.resolution_reason = "x";
  row.review_status = "reviewed";
  row.ownership.reviewer = "x";
  row.ownership.review_date = "2099-99-99";
  row.evidence.push({
    kind: "resolution_review",
    reference_type: "https_url",
    reference: "x",
    claim: "x",
  });
  refreshLedgerDigest(ledger);

  const report = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
    asOf: "2026-07-13",
  });

  assert.ok(report.errors.some((error) => error.includes("resolution_reason")));
  assert.ok(report.errors.some((error) => error.includes("HTTPS")));
  assert.ok(report.errors.some((error) => error.includes("reviewer")));
  assert.ok(report.errors.some((error) => error.includes("review_date")));
  assert.equal(report.contract_freeze_ready, false);
  assert.equal(report.inventory_delivery_ready, false);
});


test("canonical JSON sorts object keys recursively", () => {
  assert.equal(
    canonicalJson({ z: 1, a: { y: 2, b: 3 }, list: [{ d: 4, c: 5 }] }),
    '{"a":{"b":3,"y":2},"list":[{"c":5,"d":4}],"z":1}',
  );
});


test("validation does not mutate malformed input", () => {
  const { manifest, ledger } = validDocuments();
  ledger.rows[0].evidence = null;
  const before = structuredClone(ledger);

  validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources: {},
    requiredImplementedSources: {},
    schemaValidated: true,
  });

  assert.deepEqual(ledger, before);
});


test("the CLI gate distinguishes structural, contract, and inventory delivery readiness", () => {
  const incomplete = {
    errors: [],
    structurally_valid: true,
    contract_freeze_ready: false,
    inventory_delivery_ready: false,
  };
  assert.equal(gateExitCode(incomplete, "structure"), 0);
  assert.equal(gateExitCode(incomplete, "contract"), 2);
  assert.equal(gateExitCode(incomplete, "delivery"), 2);
  assert.equal(gateExitCode({ ...incomplete, errors: ["bad"] }, "structure"), 1);
  assert.equal(
    gateExitCode({ ...incomplete, contract_freeze_ready: true }, "contract"),
    0,
  );
  assert.equal(
    gateExitCode({ ...incomplete, inventory_delivery_ready: true }, "delivery"),
    0,
  );
});


test("the CLI accepts an explicit report date and rejects malformed values", () => {
  assert.equal(parseAsOf([]), new Date().toISOString().slice(0, 10));
  assert.equal(parseAsOf(["--as-of", "2026-07-13"]), "2026-07-13");
  assert.throws(
    () => parseAsOf(["--as-of"]),
    /--as-of requires YYYY-MM-DD/,
  );
  assert.throws(
    () => parseAsOf(["--as-of", "July 13"]),
    /--as-of requires YYYY-MM-DD/,
  );
});


test("the CLI parses explicit reviewer and approval trust inputs", () => {
  assert.deepEqual(
    parseRepeatedOption([
      "--trusted-reviewer", "reviewer-a",
      "--trusted-reviewer", "reviewer-a",
      "--trusted-reviewer", "reviewer-b",
    ], "--trusted-reviewer"),
    ["reviewer-a", "reviewer-b"],
  );
  assert.throws(
    () => parseRepeatedOption(["--trusted-reviewer", "--json"], "--trusted-reviewer"),
    /requires a value/,
  );
});


test("selects the inventory Python interpreter portably", () => {
  assert.equal(
    selectInventoryPython({
      RESEARCH_INVENTORY_PYTHON: "custom-python",
      VIRTUAL_ENV: "C:\\ignored",
    }, "win32"),
    "custom-python",
  );
  assert.equal(
    selectInventoryPython({ VIRTUAL_ENV: "/tmp/research-venv" }, "linux"),
    "/tmp/research-venv/bin/python",
  );
  assert.equal(
    selectInventoryPython({ VIRTUAL_ENV: "C:\\research-venv" }, "win32"),
    "C:\\research-venv\\Scripts\\python.exe",
  );
  assert.equal(selectInventoryPython({}, "darwin"), "python3");
  assert.equal(selectInventoryPython({}, "win32"), "python");
});


test("certification collection rejects symlinks outside the repository", (t) => {
  const root = fs.mkdtempSync(path.join(os.tmpdir(), "research-inventory-artifacts-"));
  t.after(() => fs.rmSync(root, { recursive: true, force: true }));
  const certificationDirectory = path.join(
    root,
    "Docs/Design/research_source_inventory/certifications",
  );
  fs.mkdirSync(certificationDirectory, { recursive: true });
  const outside = path.join(root, "outside.json");
  fs.writeFileSync(outside, "{}\n", "utf8");
  const relativePath = "Docs/Design/research_source_inventory/certifications/link.json";
  fs.symlinkSync(outside, path.join(root, relativePath));
  const ledger = {
    rows: [{
      certifications: [{ fixture_evidence: relativePath }],
    }],
  };

  const collected = collectCertificationArtifacts(root, ledger);

  assert.deepEqual(collected.digests, {});
  assert.deepEqual(collected.documents, {});
  assert.ok(collected.errors.some((error) => error.includes("symbolic link")));
});


test("the authoritative CLI composes schema and semantic validation", (t) => {
  const python = selectInventoryPython();
  const args = [
    VALIDATOR_PATH,
    "--root", ROOT,
    "--gate", "contract",
    "--as-of", "2026-07-15",
    "--trusted-reviewer", "codex-task-12968.1-source-triage",
    "--trusted-reviewer", "codex-task-12968.5-inventory-review",
    "--json",
  ];
  const valid = spawnSync(process.execPath, args, {
    encoding: "utf8",
    env: { ...process.env, RESEARCH_INVENTORY_PYTHON: python },
    timeout: 30_000,
  });
  assert.equal(valid.status, 0, valid.stderr);
  const validReport = JSON.parse(valid.stdout);
  assert.equal(validReport.schema_validated, true);
  assert.deepEqual(validReport.errors, []);
  assert.equal(validReport.as_of, "2026-07-15");
  assert.deepEqual(validReport.trusted_reviewer_ids, [
    "codex-task-12968.1-source-triage",
    "codex-task-12968.5-inventory-review",
  ]);
  assert.equal(validReport.contract_freeze_ready, true);
  assert.equal(validReport.inventory_delivery_ready, false);
  assert.match(validReport.digests.schema_validator, /^[a-f0-9]{64}$/);
  assert.deepEqual(
    validReport,
    JSON.parse(fs.readFileSync(path.join(
      ROOT,
      "Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json",
    ), "utf8")),
  );
  assert.equal(
    valid.stdout,
    fs.readFileSync(path.join(
      ROOT,
      "Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json",
    ), "utf8"),
  );
  assert.equal(validReport.counts.resolution.mapped, 191);
  assert.equal(validReport.counts.resolution.credentialed_out_of_scope, 35);
  assert.deepEqual(validReport.counts.implementation, {
    planned: 231,
    implemented: 4,
  });
  assert.deepEqual(validReport.counts.fixture, {
    not_run: 231,
    passed: 4,
    failed: 0,
  });
  assert.deepEqual(validReport.counts.live, {
    not_run: 235,
    current: 0,
    expired: 0,
    failed: 0,
  });

  const authoritativeLedger = JSON.parse(fs.readFileSync(path.join(
    ROOT,
    "Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json",
  ), "utf8"));
  const rowsById = new Map(
    authoritativeLedger.rows.map((row) => [row.inventory_id, row]),
  );
  assert.equal(authoritativeLedger.generated_at_utc, "2026-07-15T20:46:48Z");
  const expectedRequiredSources = {
    "sourclip-2026-07-13-0021": {
      target: "biorxiv",
      publisher: "bioRxiv",
    },
    "sourclip-2026-07-13-0022": {
      target: "medrxiv",
      publisher: "medRxiv",
    },
  };
  for (const [inventoryId, expected] of Object.entries(expectedRequiredSources)) {
    const required = validReport.required_sources[inventoryId];
    assert.equal(
      required.required_general_route_id,
      `${expected.target}_europe_pmc_search_aggregator`,
    );
    assert.equal(
      required.required_lookup_route_id,
      `${expected.target}_details_lookup_direct`,
    );
    assert.equal(
      required.required_interval_route_id,
      `${expected.target}_details_interval_direct`,
    );
    assert.equal(required.mapping_satisfied, true);

    const row = rowsById.get(inventoryId);
    assert.deepEqual(row.canonical_targets, [expected.target]);
    assert.deepEqual(row.declared_surfaces, ["standalone_search", "deep_research"]);
    assert.equal(row.implementation_state, "implemented");
    assert.equal(row.fixture_state, "passed");
    assert.equal(row.live_state, "not_run");
    assert.deepEqual(row.certifications, []);
    assert.deepEqual(row.ownership, {
      reviewer: "codex-task-12968.5-inventory-review",
      review_date: "2026-07-15",
      workstream: "TASK-12968.5",
      follow_up_task: "TASK-12968.5",
      revisit_trigger: "Run live and product-surface certification before presenting these implemented shadow routes as user-ready.",
    });
    assert.deepEqual(row.route_kinds, ["aggregator", "direct"]);
    assert.deepEqual(
      row.route_candidates.map((route) => route.route_candidate_id),
      [
        `${expected.target}_europe_pmc_search_aggregator`,
        `${expected.target}_details_lookup_direct`,
        `${expected.target}_details_interval_direct`,
      ],
    );
    const [general, lookup, interval] = row.route_candidates;
    assert.deepEqual(general, {
      route_candidate_id: `${expected.target}_europe_pmc_search_aggregator`,
      route_kind: "aggregator",
      credential_requirement: "none",
      planned_backend_id: "europe_pmc_rest_api",
      query_modes: ["general_free_text"],
      source_constraint: "provider_source_filter",
      source_constraint_predicate: {
        provider_field: "bookOrReportDetails.publisher",
        operator: "equals",
        values: [expected.publisher],
      },
      attribution_basis: "provider_source_field",
      coverage_notes: general.coverage_notes,
      evidence_reference: "https://europepmc.org/RestfulWebService",
    });
    assert.deepEqual(lookup.query_modes, ["identifier_lookup"]);
    assert.equal(lookup.route_kind, "direct");
    assert.equal(lookup.planned_backend_id, "biorxiv_details_api");
    assert.deepEqual(lookup.source_constraint_predicate, null);
    assert.deepEqual(interval.query_modes, ["date_interval", "category_browse"]);
    assert.equal(interval.route_kind, "direct");
    assert.equal(interval.planned_backend_id, "biorxiv_details_api");
    assert.deepEqual(interval.source_constraint_predicate, null);
    assert.ok(row.route_candidates.every(
      (route) => !route.query_modes.includes("recent_feed"),
    ));
    assert.ok(row.evidence.some(
      (entry) => /2026-07-15/.test(entry.claim) && /recent/i.test(entry.claim),
    ));
  }
  assert.deepEqual(validReport.required_implemented_source_blockers, []);
  const expectedImplementedSources = {
    "sourclip-2026-07-13-0026": {
      sourceSnapshotSha256: "cbc4a8445252460ef4502924edf409c7fc8098eb6987745b83cc426bd2fc8e73",
      canonicalTarget: "clinicaltrials_gov",
      capabilities: ["search", "detail", "metadata", "snippet"],
      routeId: "clinicaltrials_gov_studies_search_direct",
      backendId: "clinicaltrials_gov_api_v2",
      evidenceHost: "clinicaltrials.gov",
    },
    "sourclip-2026-07-13-0027": {
      sourceSnapshotSha256: "34d7fc36d4b64b2dca99c0472ad3d804c7ed9ff5a96574a8146947133913b32b",
      canonicalTarget: "pubmed_central",
      capabilities: ["search", "detail", "metadata"],
      routeId: "pubmed_central_esearch_summary_direct",
      backendId: "ncbi_eutils_pmc",
      evidenceHost: "www.ncbi.nlm.nih.gov",
    },
  };
  for (const [inventoryId, expected] of Object.entries(expectedImplementedSources)) {
    const row = rowsById.get(inventoryId);
    const state = validReport.required_implemented_sources[inventoryId];
    assert.equal(state.implementation_satisfied, true);
    assert.equal(row.source_snapshot_sha256, expected.sourceSnapshotSha256);
    assert.deepEqual(row.canonical_targets, [expected.canonicalTarget]);
    assert.deepEqual(row.declared_surfaces, ["standalone_search", "deep_research"]);
    assert.deepEqual(row.capabilities, expected.capabilities);
    assert.equal(row.implementation_state, "implemented");
    assert.equal(row.fixture_state, "passed");
    assert.equal(row.live_state, "not_run");
    assert.deepEqual(row.certifications, []);
    assert.ok(row.evidence.some((entry) => entry.kind === "implementation"));
    assert.equal(row.route_candidates.filter(
      (candidate) => candidate.route_candidate_id === expected.routeId,
    ).length, 1);
    const [route] = row.route_candidates;
    assert.deepEqual(route, {
      route_candidate_id: expected.routeId,
      route_kind: "direct",
      credential_requirement: "none",
      planned_backend_id: expected.backendId,
      query_modes: ["general_free_text"],
      source_constraint: "native_corpus",
      source_constraint_predicate: null,
      attribution_basis: "native_response",
      coverage_notes: route.coverage_notes,
      evidence_reference: `https://${expected.evidenceHost}${new URL(route.evidence_reference).pathname}`,
    });
  }
  const openAlex = rowsById.get("sourclip-2026-07-13-0088");
  assert.equal(openAlex.resolution, "credentialed_out_of_scope");
  assert.equal(openAlex.resolution_code, "credential_required_no_public_route");
  assert.ok(openAlex.route_candidates.every(
    (route) => route.credential_requirement !== "none",
  ));
  assert.equal(openAlex.route_candidates[0].credential_requirement, "api_key");
  assert.match(openAlex.resolution_reason, /free API key is still a credential/i);
  const officialEvidence = new Map(
    openAlex.evidence
      .filter((entry) => entry.kind === "resolution_review")
      .map((entry) => [entry.reference, entry.claim]),
  );
  const authenticationClaim = officialEvidence.get(
    "https://developers.openalex.org/api-reference/authentication",
  ) ?? "";
  assert.match(authenticationClaim, /anonymous (?:trial|demo) budget/i);
  const pricingAnnouncementClaim = officialEvidence.get(
    "https://blog.openalex.org/openalex-api-new-features-and-usage-based-pricing/",
  ) ?? "";
  assert.match(pricingAnnouncementClaim, /2026-02-24/);
  assert.match(
    pricingAnnouncementClaim,
    /no-key calls.*demo.*unsuitable for production/i,
  );
  const overviewClaim = officialEvidence.get(
    "https://developers.openalex.org/api-reference/introduction",
  ) ?? "";
  assert.match(overviewClaim, /api_key.*required/i);
  assert.deepEqual(
    rowsById.get("sourclip-2026-07-13-0202").canonical_targets,
    ["crossref"],
  );
  for (const position of [
    173, 174, 178, 179, 180, 181, 182, 186, 190, 191,
    203, 204, 209, 228, 229, 230, 231, 232, 233, 235,
  ]) {
    const row = rowsById.get(`sourclip-2026-07-13-${String(position).padStart(4, "0")}`);
    assert.deepEqual(row.route_candidates, []);
    assert.deepEqual(row.canonical_targets, []);
  }
  for (const row of authoritativeLedger.rows) {
    for (const route of row.route_candidates) {
      if (route.route_kind !== "aggregator") continue;
      assert.equal(typeof route.source_constraint_predicate.provider_field, "string");
      assert.ok(route.source_constraint_predicate.values.length > 0);
    }
  }

  const temporaryRoot = fs.mkdtempSync(path.join(os.tmpdir(), "research-inventory-cli-"));
  t.after(() => fs.rmSync(temporaryRoot, { recursive: true, force: true }));
  const inventoryDirectory = path.join(
    temporaryRoot,
    "Docs/Design/research_source_inventory",
  );
  const helperDirectory = path.join(temporaryRoot, "Helper_Scripts");
  fs.mkdirSync(inventoryDirectory, { recursive: true });
  fs.mkdirSync(helperDirectory, { recursive: true });
  fs.copyFileSync(
    path.join(ROOT, "Helper_Scripts/validate_research_source_inventory_schema.py"),
    path.join(helperDirectory, "validate_research_source_inventory_schema.py"),
  );
  for (const filename of [
    "research-source-inventory.schema.json",
    "sourclip-research-sources-2026-07-13.json",
    "research-source-coverage-ledger-2026-07-13.json",
  ]) {
    fs.copyFileSync(
      path.join(ROOT, "Docs/Design/research_source_inventory", filename),
      path.join(inventoryDirectory, filename),
    );
  }
  const ledgerPath = path.join(
    inventoryDirectory,
    "research-source-coverage-ledger-2026-07-13.json",
  );
  const invalidLedger = JSON.parse(fs.readFileSync(ledgerPath, "utf8"));
  invalidLedger.unexpected = true;
  fs.writeFileSync(ledgerPath, `${JSON.stringify(invalidLedger, null, 2)}\n`, "utf8");

  const invalid = spawnSync(process.execPath, [
    VALIDATOR_PATH,
    "--root", temporaryRoot,
    "--gate", "contract",
    "--as-of", "2026-07-15",
    "--trusted-reviewer", "codex-task-12968.1-source-triage",
    "--trusted-reviewer", "codex-task-12968.5-inventory-review",
    "--json",
  ], {
    encoding: "utf8",
    env: { ...process.env, RESEARCH_INVENTORY_PYTHON: python },
    timeout: 30_000,
  });
  assert.equal(invalid.status, 1, invalid.stderr);
  const invalidReport = JSON.parse(invalid.stdout);
  assert.equal(invalidReport.schema_validated, true);
  assert.ok(invalidReport.errors.some((error) => error.includes("unexpected")));
  assert.equal(invalidReport.structurally_valid, false);
});
