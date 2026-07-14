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
    schemaValidated: true,
  });

  assert.ok(report.errors.includes("manifest items_sha256 does not match canonical items"));
  assert.ok(
    report.errors.includes(
      "ledger contains duplicate inventory_id sourclip-2026-07-13-0001",
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


test("required sources block contract freeze until mapped to their exact targets", () => {
  const { manifest, ledger } = validDocuments();
  const requiredSources = {
    "sourclip-2026-07-13-0001": {
      canonicalTarget: "biorxiv",
      generalRoute: {
        id: "biorxiv_site_search",
        routeKind: "site_search",
        backendId: "biorxiv_site_search",
        queryModes: ["general_free_text"],
        sourceConstraint: "native_corpus",
        attributionBasis: "native_response",
        evidenceHosts: ["example.test"],
      },
      boundedRoute: {
        id: "biorxiv_details_api",
        routeKind: "direct",
        backendId: "biorxiv_details_api",
        queryModes: ["identifier_lookup", "recent_feed", "date_interval", "category_browse"],
        sourceConstraint: "native_corpus",
        attributionBasis: "native_response",
        evidenceHosts: ["api.example.test"],
      },
    },
  };

  const unreviewed = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
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

  const row = mapExampleRow(manifest, ledger, mappedRoute({
    routeCandidateId: "fake_route",
    plannedBackendId: "fake_backend",
  }));
  row.canonical_targets = ["wrong_target"];
  ledger.target_definitions[0].canonical_target_id = "wrong_target";
  refreshLedgerDigest(ledger);

  const wrongTarget = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(wrongTarget.errors, []);
  assert.equal(wrongTarget.counts.triaged, 1);
  assert.equal(wrongTarget.contract_freeze_ready, false);
  assert.equal(
    wrongTarget.required_sources["sourclip-2026-07-13-0001"].mapping_satisfied,
    false,
  );

  row.canonical_targets = ["biorxiv"];
  ledger.target_definitions[0].canonical_target_id = "biorxiv";
  refreshLedgerDigest(ledger);
  const correctTargetWrongRoute = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(correctTargetWrongRoute.errors, []);
  assert.equal(correctTargetWrongRoute.contract_freeze_ready, false);
  assert.equal(
    correctTargetWrongRoute.required_sources["sourclip-2026-07-13-0001"].mapping_satisfied,
    false,
  );

  row.route_candidates = [
    {
      ...mappedRoute({
        routeCandidateId: "biorxiv_site_search",
        plannedBackendId: "biorxiv_site_search",
        evidenceReference: "https://example.test/search",
      }),
      route_kind: "site_search",
    },
    {
      ...mappedRoute({
        routeCandidateId: "biorxiv_details_api",
        plannedBackendId: "biorxiv_details_api",
        evidenceReference: "https://api.example.test/details/biorxiv/help",
        queryModes: ["identifier_lookup", "recent_feed", "date_interval", "category_browse"],
      }),
      route_kind: "direct",
    },
  ];
  row.route_kinds = ["site_search", "direct"];
  refreshLedgerDigest(ledger);
  const correctTargetAndRoute = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.deepEqual(correctTargetAndRoute.errors, []);
  assert.equal(correctTargetAndRoute.contract_freeze_ready, true);
  assert.equal(
    correctTargetAndRoute.required_sources["sourclip-2026-07-13-0001"].mapping_satisfied,
    true,
  );

  row.route_candidates.pop();
  row.route_kinds = ["site_search"];
  refreshLedgerDigest(ledger);
  const missingBoundedRoute = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.equal(missingBoundedRoute.contract_freeze_ready, false);

  row.route_candidates.push({
    ...mappedRoute({
      routeCandidateId: "biorxiv_details_api",
      plannedBackendId: "biorxiv_details_api",
      evidenceReference: "https://api.example.test/details/biorxiv/help",
      queryModes: ["general_free_text"],
    }),
    route_kind: "direct",
  });
  row.route_kinds = ["site_search", "direct"];
  refreshLedgerDigest(ledger);
  const boundedRouteMislabeledGeneral = validateInventoryDocuments(manifest, ledger, {
    freeze: freezeFor(manifest),
    requiredSources,
    schemaValidated: true,
    trustedReviewerIds: ["research-maintainer"],
  });
  assert.equal(boundedRouteMislabeledGeneral.contract_freeze_ready, false);
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
  const python = process.env.RESEARCH_INVENTORY_PYTHON
    ?? (process.env.VIRTUAL_ENV
      ? path.join(process.env.VIRTUAL_ENV, "bin", "python")
      : "python3");
  const args = [
    VALIDATOR_PATH,
    "--root", ROOT,
    "--gate", "contract",
    "--as-of", "2026-07-13",
    "--trusted-reviewer", "codex-task-12968.1-source-triage",
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
  assert.equal(validReport.contract_freeze_ready, true);
  assert.match(validReport.digests.schema_validator, /^[a-f0-9]{64}$/);
  assert.deepEqual(
    validReport,
    JSON.parse(fs.readFileSync(path.join(
      ROOT,
      "Docs/Design/research_source_inventory/research-source-inventory-freeze-report-2026-07-13.json",
    ), "utf8")),
  );
  assert.equal(validReport.counts.resolution.mapped, 191);
  assert.equal(validReport.counts.resolution.credentialed_out_of_scope, 35);

  const authoritativeLedger = JSON.parse(fs.readFileSync(path.join(
    ROOT,
    "Docs/Design/research_source_inventory/research-source-coverage-ledger-2026-07-13.json",
  ), "utf8"));
  const rowsById = new Map(
    authoritativeLedger.rows.map((row) => [row.inventory_id, row]),
  );
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
    "--as-of", "2026-07-13",
    "--trusted-reviewer", "codex-task-12968.1-source-triage",
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
