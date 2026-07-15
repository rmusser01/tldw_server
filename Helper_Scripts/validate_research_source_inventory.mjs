#!/usr/bin/env node

import crypto from "node:crypto";
import { spawnSync } from "node:child_process";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";


const MANIFEST_SCHEMA = "research-source-seed-manifest.v1";
const LEDGER_SCHEMA = "research-source-coverage-ledger.v2";
const CLOSURE_POLICY = "credentialless-public-v2";
const MAX_CERTIFICATION_VALIDITY_DAYS = 90;
const FROZEN_SEED = Object.freeze({
  manifest_id: "sourclip-research-sources-2026-07-13",
  captured_on: "2026-07-13",
  item_count: 235,
  category_placement_count: 418,
  items_sha256: "cef8c83a2f6cf0640d88e6300f54205363654d800927263c2d918060e6a28339",
  page_sha256: "170f16c7bbb34a41d3a1f5ed33e3e411d38288dbc9b9cd636b31d005c1fb0221",
});
const REQUIRED_SOURCES = Object.freeze({
  "sourclip-2026-07-13-0021": Object.freeze({
    canonicalTarget: "biorxiv",
    generalRoute: Object.freeze({
      id: "biorxiv_europe_pmc_search_aggregator",
      routeKind: "aggregator",
      backendId: "europe_pmc_rest_api",
      queryModes: Object.freeze(["general_free_text"]),
      sourceConstraint: "provider_source_filter",
      sourcePredicate: Object.freeze({
        provider_field: "bookOrReportDetails.publisher",
        operator: "equals",
        values: Object.freeze(["bioRxiv"]),
      }),
      attributionBasis: "provider_source_field",
      evidenceHosts: Object.freeze(["europepmc.org"]),
    }),
    lookupRoute: Object.freeze({
      id: "biorxiv_details_lookup_direct",
      routeKind: "direct",
      backendId: "biorxiv_details_api",
      queryModes: Object.freeze(["identifier_lookup"]),
      sourceConstraint: "native_corpus",
      sourcePredicate: null,
      attributionBasis: "native_response",
      evidenceHosts: Object.freeze(["api.biorxiv.org"]),
    }),
    intervalRoute: Object.freeze({
      id: "biorxiv_details_interval_direct",
      routeKind: "direct",
      backendId: "biorxiv_details_api",
      queryModes: Object.freeze(["date_interval", "category_browse"]),
      sourceConstraint: "native_corpus",
      sourcePredicate: null,
      attributionBasis: "native_response",
      evidenceHosts: Object.freeze(["api.biorxiv.org"]),
    }),
  }),
  "sourclip-2026-07-13-0022": Object.freeze({
    canonicalTarget: "medrxiv",
    generalRoute: Object.freeze({
      id: "medrxiv_europe_pmc_search_aggregator",
      routeKind: "aggregator",
      backendId: "europe_pmc_rest_api",
      queryModes: Object.freeze(["general_free_text"]),
      sourceConstraint: "provider_source_filter",
      sourcePredicate: Object.freeze({
        provider_field: "bookOrReportDetails.publisher",
        operator: "equals",
        values: Object.freeze(["medRxiv"]),
      }),
      attributionBasis: "provider_source_field",
      evidenceHosts: Object.freeze(["europepmc.org"]),
    }),
    lookupRoute: Object.freeze({
      id: "medrxiv_details_lookup_direct",
      routeKind: "direct",
      backendId: "biorxiv_details_api",
      queryModes: Object.freeze(["identifier_lookup"]),
      sourceConstraint: "native_corpus",
      sourcePredicate: null,
      attributionBasis: "native_response",
      evidenceHosts: Object.freeze(["api.biorxiv.org"]),
    }),
    intervalRoute: Object.freeze({
      id: "medrxiv_details_interval_direct",
      routeKind: "direct",
      backendId: "biorxiv_details_api",
      queryModes: Object.freeze(["date_interval", "category_browse"]),
      sourceConstraint: "native_corpus",
      sourcePredicate: null,
      attributionBasis: "native_response",
      evidenceHosts: Object.freeze(["api.biorxiv.org"]),
    }),
  }),
});
const RESOLUTIONS = new Set([
  "unreviewed",
  "mapped",
  "duplicate",
  "not_applicable",
  "credentialed_out_of_scope",
  "policy_blocked",
  "technically_infeasible",
]);
const REVIEW_STATUSES = new Set(["unreviewed", "provisional", "reviewed"]);
const ROUTE_KINDS = new Set(["direct", "aggregator", "site_search"]);
const CAPABILITIES = new Set([
  "search",
  "detail",
  "metadata",
  "snippet",
  "future_retrieval",
  "future_ingestion",
]);
const SURFACES = new Set(["standalone_search", "deep_research"]);
const IMPLEMENTATION_STATES = new Set(["planned", "implemented"]);
const FIXTURE_STATES = new Set(["not_run", "passed", "failed"]);
const LIVE_STATES = new Set(["not_run", "current", "expired", "failed"]);
const CREDENTIAL_REQUIREMENTS = new Set([
  "none",
  "api_key",
  "subscription",
  "browser_session",
]);
const EVIDENCED_TERMINAL_RESOLUTIONS = new Set([
  "duplicate",
  "not_applicable",
  "credentialed_out_of_scope",
  "policy_blocked",
  "technically_infeasible",
]);
const RESOLUTION_CODES = Object.freeze({
  unreviewed: "unreviewed_pending",
  mapped: "credentialless_route_identified",
  duplicate: "duplicate_seed_entry",
  not_applicable: "no_searchable_corpus",
  credentialed_out_of_scope: "credential_required_no_public_route",
  policy_blocked: "automation_prohibited",
  technically_infeasible: "no_stable_live_route",
});
const QUERY_MODES = new Set([
  "general_free_text",
  "structured_query",
  "identifier_lookup",
  "recent_feed",
  "date_interval",
  "category_browse",
]);
const SOURCE_CONSTRAINTS = new Set([
  "native_corpus",
  "provider_source_filter",
  "provider_domain_filter",
]);
const SOURCE_FILTER_OPERATORS = new Set(["equals", "one_of", "prefix"]);
const DOMAIN_FILTER_OPERATORS = new Set(["domain_suffix", "url_prefix"]);
const ATTRIBUTION_BASES = new Set([
  "native_response",
  "provider_source_field",
  "verified_reported_origin",
]);
const EVIDENCE_KINDS = new Set([
  "seed_manifest",
  "route_triage",
  "implementation",
  "resolution_review",
]);
const EVIDENCE_REFERENCE_TYPES = new Set([
  "manifest_fragment",
  "https_url",
  "repo_path",
]);
const CREDENTIALLESS_REVIEW_FINDINGS = new Set([
  "credential_required",
  "policy_blocked",
  "not_source_faithful",
  "not_identified",
]);


export function canonicalJson(value) {
  if (Array.isArray(value)) return `[${value.map(canonicalJson).join(",")}]`;
  if (isObject(value)) {
    return `{${Object.keys(value)
      .sort()
      .map((key) => `${JSON.stringify(key)}:${canonicalJson(value[key])}`)
      .join(",")}}`;
  }
  return JSON.stringify(value);
}


export function sha256(value) {
  return crypto.createHash("sha256").update(value, "utf8").digest("hex");
}


function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}


function isNonEmptyString(value) {
  return typeof value === "string" && value.trim().length > 0;
}


function isActualDate(value) {
  if (!/^\d{4}-\d{2}-\d{2}$/.test(value ?? "")) return false;
  const parsed = new Date(`${value}T00:00:00Z`);
  return !Number.isNaN(parsed.valueOf())
    && parsed.toISOString().slice(0, 10) === value;
}


function isActualTimestamp(value) {
  if (typeof value !== "string" || !value.includes("T")) return false;
  const parsed = new Date(value);
  return !Number.isNaN(parsed.valueOf()) && /(?:Z|[+-]\d{2}:\d{2})$/.test(value);
}


function daysBetween(startDate, endDate) {
  return (
    new Date(`${endDate}T00:00:00Z`).valueOf()
    - new Date(`${startDate}T00:00:00Z`).valueOf()
  ) / (24 * 60 * 60 * 1000);
}


function isHttpsUrl(value) {
  if (!isNonEmptyString(value)) return false;
  try {
    const parsed = new URL(value);
    return parsed.protocol === "https:" && !parsed.username && !parsed.password;
  } catch {
    return false;
  }
}


function isHttpsOrigin(value) {
  if (!isHttpsUrl(value)) return false;
  const parsed = new URL(value);
  return value === parsed.origin
    && parsed.pathname === "/"
    && parsed.search === ""
    && parsed.hash === "";
}


function endpointMatchesPrefix(endpoint, prefix) {
  if (!isHttpsUrl(endpoint) || !isHttpsUrl(prefix)) return false;
  const checked = new URL(endpoint);
  const allowed = new URL(prefix);
  if (checked.origin !== allowed.origin || allowed.search || allowed.hash) return false;
  if (checked.pathname === allowed.pathname) return true;
  const allowedDirectory = allowed.pathname.endsWith("/")
    ? allowed.pathname
    : `${allowed.pathname}/`;
  return checked.pathname.startsWith(allowedDirectory);
}


function routePolicyIsWellFormed(policy) {
  if (!isObject(policy)
    || !Array.isArray(policy.allowed_methods)
    || policy.allowed_methods.length === 0
    || new Set(policy.allowed_methods).size !== policy.allowed_methods.length
    || policy.allowed_methods.some((method) => !new Set(["GET", "POST"]).has(method))
    || !Array.isArray(policy.allowed_url_prefixes)
    || policy.allowed_url_prefixes.length === 0
    || new Set(policy.allowed_url_prefixes).size !== policy.allowed_url_prefixes.length
    || !Array.isArray(policy.allowed_transport_origins)
    || policy.allowed_transport_origins.length === 0
    || new Set(policy.allowed_transport_origins).size !== policy.allowed_transport_origins.length
    || !policy.allowed_transport_origins.every(isHttpsOrigin)
    || policy.credential_mode !== "none"
    || policy.gateway_required !== true
    || policy.result_link_dereference !== false) return false;
  return policy.allowed_url_prefixes.every((prefix) => (
    isHttpsUrl(prefix)
      && !new URL(prefix).search
      && !new URL(prefix).hash
      && policy.allowed_transport_origins.includes(new URL(prefix).origin)
  ));
}


function liveArtifactMatchesRoutePolicy(details, policy) {
  if (!routePolicyIsWellFormed(policy)
    || !isObject(details)
    || !policy.allowed_methods.includes(details.request_method)
    || !policy.allowed_url_prefixes.some((prefix) => (
      endpointMatchesPrefix(details.checked_endpoint, prefix)
    ))
    || !Array.isArray(details.transport_origins)
    || details.transport_origins.length === 0
    || !details.transport_origins.every((origin) => (
      isHttpsOrigin(origin) && policy.allowed_transport_origins.includes(origin)
    ))
    || !details.transport_origins.includes(new URL(details.checked_endpoint).origin)
    || details.gateway_attested !== true
    || details.credential_mode !== "none"
    || details.result_link_dereference_count !== 0) return false;
  return true;
}


function isSafeRepoEvidencePath(value) {
  return typeof value === "string"
    && value.startsWith("Docs/Design/research_source_inventory/certifications/")
    && !value.includes("..")
    && !value.startsWith("/")
    && value.endsWith(".json");
}


function urlHostMatches(value, allowedHosts) {
  if (!isHttpsUrl(value)) return false;
  const host = new URL(value).hostname.toLowerCase();
  return allowedHosts.some((allowed) => (
    host === allowed || host.endsWith(`.${allowed}`)
  ));
}


function evidenceReferenceIsValid(entry) {
  if (entry?.reference_type === "https_url") return isHttpsUrl(entry.reference);
  if (entry?.reference_type === "repo_path") return isSafeRepoEvidencePath(entry.reference);
  if (entry?.reference_type === "manifest_fragment") {
    return /^sourclip-research-sources-2026-07-13#sourclip-2026-07-13-\d{4}$/.test(
      entry.reference ?? "",
    );
  }
  return false;
}


function validateEnumArray(errors, rowId, field, values, allowed) {
  if (!Array.isArray(values)) {
    errors.push(`${rowId} ${field} must be an array`);
    return;
  }
  if (new Set(values).size !== values.length) {
    errors.push(`${rowId} ${field} contains duplicates`);
  }
  for (const value of values) {
    if (!allowed.has(value)) errors.push(`${rowId} has unknown ${field} value ${value}`);
  }
}


function sameStringSet(left, right) {
  return Array.isArray(left)
    && Array.isArray(right)
    && left.length === right.length
    && new Set(left).size === left.length
    && left.every((value) => right.includes(value));
}


function routeMatchesRequirement(candidate, requirement) {
  return candidate?.route_candidate_id === requirement.id
    && candidate?.route_kind === requirement.routeKind
    && candidate?.credential_requirement === "none"
    && candidate?.planned_backend_id === requirement.backendId
    && sameStringSet(candidate?.query_modes, requirement.queryModes)
    && candidate?.source_constraint === requirement.sourceConstraint
    && canonicalJson(candidate?.source_constraint_predicate)
      === canonicalJson(requirement.sourcePredicate)
    && candidate?.attribution_basis === requirement.attributionBasis
    && urlHostMatches(candidate?.evidence_reference, requirement.evidenceHosts);
}


function hasEvidence(row, kind) {
  return Array.isArray(row?.evidence)
    && row.evidence.some((entry) => isObject(entry) && entry.kind === kind);
}


function hasReviewedOwnership(row, asOf, trustedReviewerIds = new Set()) {
  return row.review_status === "reviewed"
    && typeof row.ownership?.reviewer === "string"
    && row.ownership.reviewer.trim().length >= 3
    && isActualDate(row.ownership?.review_date)
    && row.ownership.review_date <= asOf
    && trustedReviewerIds.has(row.ownership.reviewer);
}


function hasWellFormedReviewedOwnership(row, asOf) {
  return row.review_status === "reviewed"
    && typeof row.ownership?.reviewer === "string"
    && row.ownership.reviewer.trim().length >= 3
    && isActualDate(row.ownership?.review_date)
    && row.ownership.review_date <= asOf;
}


function credentiallessRouteCandidates(row) {
  return Array.isArray(row?.route_candidates)
    ? row.route_candidates.filter((candidate) => candidate?.credential_requirement === "none")
    : [];
}


function routeSemanticsAreConsistent(candidate) {
  const expectedAttribution = {
    native_corpus: "native_response",
    provider_source_filter: "provider_source_field",
    provider_domain_filter: "verified_reported_origin",
  }[candidate?.source_constraint];
  if (!expectedAttribution || candidate?.attribution_basis !== expectedAttribution) return false;
  return candidate?.route_kind === "aggregator"
    ? candidate.source_constraint !== "native_corpus"
    : candidate.source_constraint === "native_corpus";
}


function sourcePredicateIsConsistent(candidate) {
  const predicate = candidate?.source_constraint_predicate;
  if (candidate?.source_constraint === "native_corpus") return predicate === null;
  if (!isObject(predicate)
    || !isNonEmptyString(predicate.provider_field)
    || !Array.isArray(predicate.values)
    || predicate.values.length === 0
    || new Set(predicate.values).size !== predicate.values.length
    || predicate.values.some((value) => !isNonEmptyString(value))) return false;
  if (candidate.source_constraint === "provider_source_filter") {
    return SOURCE_FILTER_OPERATORS.has(predicate.operator);
  }
  if (candidate.source_constraint === "provider_domain_filter") {
    return DOMAIN_FILTER_OPERATORS.has(predicate.operator);
  }
  return false;
}


function isDiscoverableRoute(candidate) {
  return candidate?.credential_requirement === "none"
    && Array.isArray(candidate.query_modes)
    && candidate.query_modes.some((mode) => (
      mode === "general_free_text" || mode === "structured_query"
    ))
    && SOURCE_CONSTRAINTS.has(candidate.source_constraint)
    && ATTRIBUTION_BASES.has(candidate.attribution_basis)
    && routeSemanticsAreConsistent(candidate)
    && sourcePredicateIsConsistent(candidate)
    && isHttpsUrl(candidate.evidence_reference);
}


function hasCompleteCredentiallessReview(row) {
  if (!Array.isArray(row?.credentialless_route_review)
    || row.credentialless_route_review.length !== ROUTE_KINDS.size) return false;
  const kinds = new Set();
  let credentialRequired = false;
  for (const review of row.credentialless_route_review) {
    if (!ROUTE_KINDS.has(review?.route_kind)
      || !CREDENTIALLESS_REVIEW_FINDINGS.has(review?.finding)
      || !isHttpsUrl(review?.evidence_reference)
      || typeof review?.notes !== "string"
      || review.notes.trim().length < 20) return false;
    kinds.add(review.route_kind);
    if (review.finding === "credential_required") credentialRequired = true;
  }
  return kinds.size === ROUTE_KINDS.size && credentialRequired;
}


function isSubstantivelyTriaged(row, asOf, trustedReviewerIds = new Set()) {
  if (!hasReviewedOwnership(row, asOf, trustedReviewerIds)
    || typeof row.resolution_reason !== "string"
    || row.resolution_reason.trim().length < 40
    || row.resolution_code !== RESOLUTION_CODES[row.resolution]
    || !Array.isArray(row.canonical_targets)
    || row.canonical_targets.length > 1) return false;
  if (row.resolution === "mapped") {
    return Array.isArray(row.canonical_targets)
      && row.canonical_targets.length === 1
      && credentiallessRouteCandidates(row).some(isDiscoverableRoute)
      && Array.isArray(row.capabilities)
      && row.capabilities.includes("search")
      && Array.isArray(row.declared_surfaces)
      && row.declared_surfaces.length === SURFACES.size
      && SURFACES.size === new Set(row.declared_surfaces).size
      && [...SURFACES].every((surface) => row.declared_surfaces.includes(surface))
      && hasEvidence(row, "route_triage");
  }
  if (row.resolution === "duplicate") {
    return isNonEmptyString(row.duplicate_of_inventory_id)
      && hasEvidence(row, "resolution_review");
  }
  if (row.resolution === "credentialed_out_of_scope") {
    return Array.isArray(row.route_candidates)
      && credentiallessRouteCandidates(row).length === 0
      && hasCompleteCredentiallessReview(row)
      && hasEvidence(row, "resolution_review");
  }
  return EVIDENCED_TERMINAL_RESOLUTIONS.has(row.resolution)
    && hasEvidence(row, "resolution_review");
}


function certificationArtifactMatches(
  artifact,
  artifactType,
  certification,
  candidate,
  routePolicy,
) {
  if (!isObject(artifact)
    || artifact.schema_version !== "research-source-certification-artifact.v1"
    || artifact.artifact_type !== artifactType
    || artifact.route_candidate_id !== certification.route_candidate_id
    || artifact.canonical_target !== certification.canonical_target
    || artifact.surface !== certification.surface
    || artifact.route_candidate_sha256 !== certification.route_candidate_sha256
    || artifact.route_candidate_sha256 !== sha256(canonicalJson(candidate))
    || artifact.route_policy_sha256 !== certification.route_policy_sha256
    || artifact.catalog_version !== certification.catalog_version
    || artifact.policy_version !== certification.policy_version
    || !isActualTimestamp(artifact.observed_at_utc)
    || artifact.observed_at_utc.slice(0, 10) !== certification.certified_on
    || artifact.sanitized !== true
    || !isObject(artifact.details)) return false;
  if (artifactType === "fixture") {
    const requiredCases = new Set(["success", "valid_empty", "malformed", "partial_failure"]);
    return artifact.outcome === "passed"
      && isNonEmptyString(artifact.details.test_command)
      && Number.isInteger(artifact.details.test_count)
      && artifact.details.test_count > 0
      && Array.isArray(artifact.details.fixture_cases)
      && [...requiredCases].every((value) => artifact.details.fixture_cases.includes(value));
  }
  if (artifactType === "live") {
    return artifact.outcome === "passed"
      && isHttpsUrl(artifact.details.checked_endpoint)
      && new Set(["GET", "POST"]).has(artifact.details.request_method)
      && Number.isInteger(artifact.details.request_count)
      && artifact.details.request_count > 0
      && Number.isInteger(artifact.details.result_count)
      && artifact.details.result_count > 0
      && Array.isArray(artifact.details.transport_origins)
      && artifact.details.transport_origins.length > 0
      && artifact.details.transport_origins.every(isHttpsOrigin)
      && liveArtifactMatchesRoutePolicy(artifact.details, routePolicy);
  }
  if (artifactType === "policy") {
    return artifact.outcome === "allowed"
      && isHttpsUrl(artifact.details.terms_url)
      && (artifact.details.robots_url === null || isHttpsUrl(artifact.details.robots_url))
      && typeof artifact.details.reviewer === "string"
      && artifact.details.reviewer.trim().length >= 3
      && typeof artifact.details.decision_notes === "string"
      && artifact.details.decision_notes.trim().length >= 40
      && routePolicyIsWellFormed(artifact.details.route_policy)
      && artifact.route_policy_sha256
        === sha256(canonicalJson(artifact.details.route_policy));
  }
  return false;
}


function hasCurrentCertification(
  row,
  candidate,
  surface,
  asOf,
  artifactDigests,
  certificationArtifacts,
  catalogVersion,
  policyVersion,
) {
  if (!Array.isArray(row?.certifications)) return false;
  return row.certifications.some((certification) => (
    certification?.route_candidate_id === candidate.route_candidate_id
      && row.canonical_targets.includes(certification?.canonical_target)
      && certification?.surface === surface
      && certification?.route_candidate_sha256 === sha256(canonicalJson(candidate))
      && routePolicyIsWellFormed(
        certificationArtifacts[certification?.policy_evidence]?.details?.route_policy,
      )
      && certification?.route_policy_sha256 === sha256(canonicalJson(
        certificationArtifacts[certification?.policy_evidence]?.details?.route_policy,
      ))
      && certification?.catalog_version === catalogVersion
      && certification?.policy_version === policyVersion
      && new Set([
        certification?.fixture_evidence,
        certification?.live_evidence,
        certification?.policy_evidence,
      ]).size === 3
      && isSafeRepoEvidencePath(certification?.fixture_evidence)
      && isSafeRepoEvidencePath(certification?.live_evidence)
      && isSafeRepoEvidencePath(certification?.policy_evidence)
      && artifactDigests[certification.fixture_evidence] === certification?.fixture_sha256
      && artifactDigests[certification.live_evidence] === certification?.live_sha256
      && artifactDigests[certification.policy_evidence] === certification?.policy_digest
      && certificationArtifactMatches(
        certificationArtifacts[certification.fixture_evidence],
        "fixture",
        certification,
        candidate,
        certificationArtifacts[certification?.policy_evidence]?.details?.route_policy,
      )
      && certificationArtifactMatches(
        certificationArtifacts[certification.live_evidence],
        "live",
        certification,
        candidate,
        certificationArtifacts[certification?.policy_evidence]?.details?.route_policy,
      )
      && certificationArtifactMatches(
        certificationArtifacts[certification.policy_evidence],
        "policy",
        certification,
        candidate,
        certificationArtifacts[certification?.policy_evidence]?.details?.route_policy,
      )
      && isActualDate(certification?.certified_on)
      && isActualDate(certification?.valid_until)
      && certification.certified_on <= asOf
      && certification.valid_until >= asOf
      && certification.certified_on <= certification.valid_until
      && daysBetween(certification.certified_on, certification.valid_until)
        <= MAX_CERTIFICATION_VALIDITY_DAYS
      && /^[a-f0-9]{64}$/.test(certification?.policy_digest ?? "")
  ));
}


function closureDecisionSnapshot(row) {
  return {
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
}


function closureApprovalIsWellFormed(row, asOf) {
  const approval = row?.closure_approval;
  return isObject(approval)
    && typeof approval.approved_by === "string"
    && approval.approved_by.trim().length >= 3
    && isActualDate(approval.approved_on)
    && approval.approved_on <= asOf
    && isActualDate(row.ownership?.review_date)
    && approval.approved_on >= row.ownership.review_date
    && (
      (approval.approval_reference_type === "https_url"
        && isHttpsUrl(approval.approval_reference))
      || (approval.approval_reference_type === "backlog_task"
        && /^TASK-\d+(?:\.\d+)?$/.test(approval.approval_reference ?? ""))
    )
    && approval.approval_reference === row.ownership?.follow_up_task
    && approval.decision_sha256 === sha256(canonicalJson(closureDecisionSnapshot(row)));
}


function hasTrustedClosureApproval(row, asOf, trustedApprovalReferences) {
  return closureApprovalIsWellFormed(row, asOf)
    && trustedApprovalReferences.has(row.closure_approval.approval_reference);
}


function isTerminal(
  row,
  asOf,
  artifactDigests,
  certificationArtifacts,
  catalogVersion,
  policyVersion,
  trustedReviewerIds,
  trustedApprovalReferences,
) {
  if (!isSubstantivelyTriaged(row, asOf, trustedReviewerIds)) return false;
  if (row.resolution === "mapped") {
    if (row.implementation_state !== "implemented"
      || row.fixture_state !== "passed"
      || row.live_state !== "current") return false;
    return credentiallessRouteCandidates(row).every((candidate) => (
      row.declared_surfaces.every((surface) => (
        hasCurrentCertification(
          row,
          candidate,
          surface,
          asOf,
          artifactDigests,
          certificationArtifacts,
          catalogVersion,
          policyVersion,
        )
      ))
    ));
  }
  return EVIDENCED_TERMINAL_RESOLUTIONS.has(row.resolution)
    && hasTrustedClosureApproval(row, asOf, trustedApprovalReferences);
}


export function validateInventoryDocuments(
  manifest,
  ledger,
  {
    freeze = FROZEN_SEED,
    asOf = new Date().toISOString().slice(0, 10),
    schema = null,
    requiredSources = REQUIRED_SOURCES,
    schemaValidated = false,
    schemaErrors = [],
    artifactDigests = {},
    certificationArtifacts = {},
    trustedReviewerIds = [],
    trustedApprovalReferences = [],
    validatorDigest = null,
    schemaValidatorDigest = null,
  } = {},
) {
  const errors = [...schemaErrors];
  const manifestItems = Array.isArray(manifest?.items) ? manifest.items : [];
  const ledgerRows = Array.isArray(ledger?.rows) ? ledger.rows : [];
  const trustedReviewers = new Set(trustedReviewerIds);
  const trustedApprovals = new Set(trustedApprovalReferences);

  if (manifest?.manifest_id !== freeze.manifest_id) {
    errors.push(`frozen manifest_id must be ${freeze.manifest_id}`);
  }
  if (manifest?.source?.captured_on !== freeze.captured_on) {
    errors.push(`frozen manifest capture date must be ${freeze.captured_on}`);
  }
  if (manifestItems.length !== freeze.item_count) {
    errors.push(`frozen manifest item count must be ${freeze.item_count}`);
  }
  if (manifest?.expected_category_placement_count !== freeze.category_placement_count) {
    errors.push(
      `frozen category placement count must be ${freeze.category_placement_count}`,
    );
  }
  if (manifest?.items_sha256 !== freeze.items_sha256) {
    errors.push(`frozen items_sha256 must be ${freeze.items_sha256}`);
  }
  if (freeze.page_sha256 && manifest?.source?.page_sha256 !== freeze.page_sha256) {
    errors.push(`frozen page_sha256 must be ${freeze.page_sha256}`);
  }

  if (manifest?.schema_version !== MANIFEST_SCHEMA) {
    errors.push(`manifest schema_version must be ${MANIFEST_SCHEMA}`);
  }
  if (!isNonEmptyString(manifest?.manifest_id)) errors.push("manifest_id is required");
  if (manifest?.source?.page_url !== "https://www.sourclip.com/resources/research-sources") {
    errors.push("manifest source page_url is not the approved coverage seed");
  }
  if (!/^\d{4}-\d{2}-\d{2}$/.test(manifest?.source?.captured_on ?? "")) {
    errors.push("manifest captured_on must be YYYY-MM-DD");
  }
  if (!/^[a-f0-9]{64}$/.test(manifest?.source?.page_sha256 ?? "")) {
    errors.push("manifest source page_sha256 must be a lowercase SHA-256 digest");
  }
  if (manifest?.source?.page_content_stored !== false) {
    errors.push("manifest must declare that external page content is not stored");
  }
  if (manifest?.expected_item_count !== manifestItems.length) {
    errors.push(
      `manifest expected_item_count ${manifest?.expected_item_count} does not match ${manifestItems.length} items`,
    );
  }
  const categoryPlacements = manifestItems.reduce(
    (total, item) => total + (Array.isArray(item.seed_categories) ? item.seed_categories.length : 0),
    0,
  );
  if (manifest?.expected_category_placement_count !== categoryPlacements) {
    errors.push(
      "manifest expected_category_placement_count does not match item category placements",
    );
  }
  if (manifest?.items_sha256 !== sha256(canonicalJson(manifestItems))) {
    errors.push("manifest items_sha256 does not match canonical items");
  }

  const manifestById = new Map();
  const manifestTuples = new Set();
  for (const [index, item] of manifestItems.entries()) {
    const rowNumber = index + 1;
    const expectedId = `sourclip-${manifest?.source?.captured_on}-${String(rowNumber).padStart(4, "0")}`;
    if (item.inventory_id !== expectedId) {
      errors.push(`manifest position ${rowNumber} inventory_id must be ${expectedId}`);
    }
    if (item.position !== rowNumber) {
      errors.push(`manifest ${item.inventory_id ?? rowNumber} position must be ${rowNumber}`);
    }
    if (manifestById.has(item.inventory_id)) {
      errors.push(`manifest contains duplicate inventory_id ${item.inventory_id}`);
    } else {
      manifestById.set(item.inventory_id, item);
    }
    if (!isNonEmptyString(item.label)) errors.push(`${item.inventory_id} label is required`);
    try {
      const parsed = new URL(item.url);
      if (!(parsed.protocol === "https:" || parsed.protocol === "http:")) {
        errors.push(`${item.inventory_id} URL must use HTTP or HTTPS`);
      }
      if (parsed.username || parsed.password) errors.push(`${item.inventory_id} URL contains userinfo`);
    } catch {
      errors.push(`${item.inventory_id} URL is invalid`);
    }
    const tuple = canonicalJson({ label: item.label, url: item.url });
    if (manifestTuples.has(tuple)) {
      errors.push(`manifest contains duplicate captured row ${item.label} ${item.url}`);
    } else {
      manifestTuples.add(tuple);
    }
    if (!Array.isArray(item.seed_categories) || item.seed_categories.length === 0) {
      errors.push(`${item.inventory_id} must retain at least one seed category placement`);
    } else if (new Set(item.seed_categories).size !== item.seed_categories.length) {
      errors.push(`${item.inventory_id} seed_categories contains duplicates`);
    }
    const rowSnapshot = {
      position: item.position,
      label: item.label,
      url: item.url,
      seed_categories: item.seed_categories,
    };
    if (item.row_sha256 !== sha256(canonicalJson(rowSnapshot))) {
      errors.push(`${item.inventory_id} row_sha256 does not match its source snapshot`);
    }
  }

  if (ledger?.schema_version !== LEDGER_SCHEMA) {
    errors.push(`ledger schema_version must be ${LEDGER_SCHEMA}`);
  }
  if (ledger?.manifest_id !== manifest?.manifest_id) {
    errors.push("ledger manifest_id does not match the manifest");
  }
  if (ledger?.manifest_items_sha256 !== manifest?.items_sha256) {
    errors.push("ledger manifest_items_sha256 does not match the manifest");
  }
  if (ledger?.closure_policy_version !== CLOSURE_POLICY) {
    errors.push(`ledger closure_policy_version must be ${CLOSURE_POLICY}`);
  }
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(ledger?.catalog_version ?? "")) {
    errors.push("ledger catalog_version must be a stable version identifier");
  }
  if (!/^[a-z0-9][a-z0-9._-]*$/.test(ledger?.certification_policy_version ?? "")) {
    errors.push("ledger certification_policy_version must be a stable version identifier");
  }
  const targetDefinitions = Array.isArray(ledger?.target_definitions)
    ? ledger.target_definitions
    : [];
  if (!Array.isArray(ledger?.target_definitions)) {
    errors.push("ledger target_definitions must be an array");
  }
  if (ledger?.target_definitions_sha256 !== sha256(canonicalJson(targetDefinitions))) {
    errors.push("ledger target_definitions_sha256 does not match canonical targets");
  }
  if (ledger?.rows_sha256 !== sha256(canonicalJson(ledgerRows))) {
    errors.push("ledger rows_sha256 does not match canonical rows");
  }

  const targetDefinitionsById = new Map();
  for (const definition of targetDefinitions) {
    const targetId = definition?.canonical_target_id;
    if (!/^[a-z0-9]+(?:_[a-z0-9]+)*$/.test(targetId ?? "")) {
      errors.push("target definition contains an invalid canonical_target_id");
      continue;
    }
    if (targetDefinitionsById.has(targetId)) {
      errors.push(`target definitions contain duplicate canonical_target_id ${targetId}`);
      continue;
    }
    if (!isNonEmptyString(definition?.display_name)) {
      errors.push(`target definition ${targetId} requires display_name`);
    }
    if (!Array.isArray(definition?.inventory_ids)
      || definition.inventory_ids.length === 0
      || new Set(definition.inventory_ids).size !== definition.inventory_ids.length) {
      errors.push(`target definition ${targetId} requires unique inventory_ids`);
    }
    targetDefinitionsById.set(targetId, definition);
  }

  const ledgerById = new Map();
  let terminal = 0;
  let unreviewed = 0;
  let triaged = 0;
  const resolutionCounts = Object.fromEntries([...RESOLUTIONS].map((value) => [value, 0]));
  const implementationCounts = Object.fromEntries(
    [...IMPLEMENTATION_STATES].map((value) => [value, 0]),
  );
  const fixtureCounts = Object.fromEntries([...FIXTURE_STATES].map((value) => [value, 0]));
  const liveCounts = Object.fromEntries([...LIVE_STATES].map((value) => [value, 0]));
  const blockers = [];
  const globalRouteCandidateIds = new Set();
  for (const row of ledgerRows) {
    const rowId = row?.inventory_id ?? "<missing inventory_id>";
    if (ledgerById.has(rowId)) {
      errors.push(`ledger contains duplicate inventory_id ${rowId}`);
      continue;
    }
    ledgerById.set(rowId, row);
    const item = manifestById.get(rowId);
    if (!item) {
      errors.push(`ledger contains unknown inventory_id ${rowId}`);
      continue;
    }
    if (row.source_snapshot_sha256 !== item.row_sha256) {
      errors.push(`${rowId} source_snapshot_sha256 does not match the manifest row`);
    }
    if (!RESOLUTIONS.has(row.resolution)) {
      errors.push(`${rowId} has unknown resolution ${row.resolution}`);
    } else {
      resolutionCounts[row.resolution] += 1;
      if (row.resolution_code !== RESOLUTION_CODES[row.resolution]) {
        errors.push(`${rowId} resolution_code does not match ${row.resolution}`);
      }
    }
    if (!REVIEW_STATUSES.has(row.review_status)) {
      errors.push(`${rowId} has unknown review_status ${row.review_status}`);
    }
    validateEnumArray(errors, rowId, "route_kinds", row.route_kinds, ROUTE_KINDS);
    validateEnumArray(errors, rowId, "capabilities", row.capabilities, CAPABILITIES);
    validateEnumArray(errors, rowId, "declared_surfaces", row.declared_surfaces, SURFACES);
    if (!IMPLEMENTATION_STATES.has(row.implementation_state)) {
      errors.push(`${rowId} has unknown implementation_state ${row.implementation_state}`);
    } else {
      implementationCounts[row.implementation_state] += 1;
    }
    if (!FIXTURE_STATES.has(row.fixture_state)) {
      errors.push(`${rowId} has unknown fixture_state ${row.fixture_state}`);
    } else {
      fixtureCounts[row.fixture_state] += 1;
    }
    if (!LIVE_STATES.has(row.live_state)) {
      errors.push(`${rowId} has unknown live_state ${row.live_state}`);
    } else {
      liveCounts[row.live_state] += 1;
    }
    if (!Array.isArray(row.canonical_targets)) {
      errors.push(`${rowId} canonical_targets must be an array`);
    } else if (row.canonical_targets.length > 1) {
      errors.push(`${rowId} may declare at most one canonical target`);
    } else if (new Set(row.canonical_targets).size !== row.canonical_targets.length) {
      errors.push(`${rowId} canonical_targets contains duplicates`);
    } else if (row.canonical_targets.some((value) => !/^[a-z0-9]+(?:_[a-z0-9]+)*$/.test(value))) {
      errors.push(`${rowId} canonical_targets contains an invalid ID`);
    } else {
      for (const targetId of row.canonical_targets) {
        const definition = targetDefinitionsById.get(targetId);
        if (!definition) {
          errors.push(`${rowId} references undeclared canonical target ${targetId}`);
        } else if (!definition.inventory_ids.includes(rowId)) {
          errors.push(`${rowId} is absent from target definition ${targetId}`);
        }
      }
    }
    const routeCandidateIds = new Set();
    const candidateKinds = new Set();
    if (!Array.isArray(row.route_candidates)) {
      errors.push(`${rowId} route_candidates must be an array`);
    } else {
      for (const candidate of row.route_candidates) {
        const candidateId = candidate?.route_candidate_id;
        if (!isNonEmptyString(candidateId)
          || !/^[a-z0-9]+(?:_[a-z0-9]+)*$/.test(candidateId)) {
          errors.push(`${rowId} contains an invalid route_candidate_id`);
        } else if (routeCandidateIds.has(candidateId)) {
          errors.push(`${rowId} contains duplicate route_candidate_id ${candidateId}`);
        } else if (globalRouteCandidateIds.has(candidateId)) {
          errors.push(`${rowId} reuses global route_candidate_id ${candidateId}`);
        } else {
          routeCandidateIds.add(candidateId);
          globalRouteCandidateIds.add(candidateId);
        }
        if (!ROUTE_KINDS.has(candidate?.route_kind)) {
          errors.push(`${rowId} route candidate ${candidateId} has unknown route_kind`);
        } else {
          candidateKinds.add(candidate.route_kind);
        }
        if (!CREDENTIAL_REQUIREMENTS.has(candidate?.credential_requirement)) {
          errors.push(`${rowId} route candidate ${candidateId} has unknown credential_requirement`);
        }
        if (!isNonEmptyString(candidate?.planned_backend_id)
          || !/^[a-z0-9]+(?:_[a-z0-9]+)*$/.test(candidate.planned_backend_id)) {
          errors.push(`${rowId} route candidate ${candidateId} has invalid planned_backend_id`);
        }
        if (!isNonEmptyString(candidate?.evidence_reference)) {
          errors.push(`${rowId} route candidate ${candidateId} lacks evidence_reference`);
        } else if (!isHttpsUrl(candidate.evidence_reference)) {
          errors.push(`${rowId} route candidate ${candidateId} evidence_reference must be HTTPS`);
        }
        validateEnumArray(errors, rowId, "query_modes", candidate?.query_modes, QUERY_MODES);
        if (!SOURCE_CONSTRAINTS.has(candidate?.source_constraint)) {
          errors.push(`${rowId} route candidate ${candidateId} has unknown source_constraint`);
        }
        if (!ATTRIBUTION_BASES.has(candidate?.attribution_basis)) {
          errors.push(`${rowId} route candidate ${candidateId} has unknown attribution_basis`);
        }
        if (ROUTE_KINDS.has(candidate?.route_kind)
          && SOURCE_CONSTRAINTS.has(candidate?.source_constraint)
          && ATTRIBUTION_BASES.has(candidate?.attribution_basis)
          && !routeSemanticsAreConsistent(candidate)) {
          const label = candidate.route_kind === "aggregator"
            ? "aggregator route"
            : `${candidate.route_kind} route`;
          errors.push(
            `${rowId} ${label} ${candidateId} misstates source constraint or attribution`,
          );
        }
        if (SOURCE_CONSTRAINTS.has(candidate?.source_constraint)
          && !sourcePredicateIsConsistent(candidate)) {
          errors.push(
            `${rowId} route candidate ${candidateId} source predicate does not match its constraint`,
          );
        }
        if (typeof candidate?.coverage_notes !== "string"
          || candidate.coverage_notes.trim().length < 20) {
          errors.push(`${rowId} route candidate ${candidateId} requires coverage_notes`);
        }
      }
      const declaredKinds = Array.isArray(row.route_kinds) ? [...new Set(row.route_kinds)].sort() : [];
      if (canonicalJson([...candidateKinds].sort()) !== canonicalJson(declaredKinds)) {
        errors.push(`${rowId} route_kinds do not match route_candidates`);
      }
    }

    if (!Array.isArray(row.certifications)) {
      errors.push(`${rowId} certifications must be an array`);
    } else {
      const certificationKeys = new Set();
      for (const certification of row.certifications) {
        const key = `${certification?.canonical_target}:${certification?.route_candidate_id}:${certification?.surface}`;
        if (certificationKeys.has(key)) {
          errors.push(`${rowId} contains duplicate certification ${key}`);
        } else {
          certificationKeys.add(key);
        }
        if (!routeCandidateIds.has(certification?.route_candidate_id)) {
          errors.push(`${rowId} certification references unknown route candidate`);
        }
        if (!SURFACES.has(certification?.surface)) {
          errors.push(`${rowId} certification has unknown surface`);
        }
        const candidate = Array.isArray(row.route_candidates)
          ? row.route_candidates.find((entry) => (
            entry?.route_candidate_id === certification?.route_candidate_id
          ))
          : null;
        if (!row.canonical_targets?.includes(certification?.canonical_target)) {
          errors.push(`${rowId} certification ${key} has unknown canonical_target`);
        }
        if (!candidate
          || certification?.route_candidate_sha256 !== sha256(canonicalJson(candidate))) {
          errors.push(`${rowId} certification ${key} route snapshot does not match`);
        }
        if (!/^[a-f0-9]{64}$/.test(certification?.route_policy_sha256 ?? "")) {
          errors.push(`${rowId} certification ${key} has invalid route_policy_sha256`);
        }
        if (!isNonEmptyString(certification?.catalog_version)
          || !isNonEmptyString(certification?.policy_version)) {
          errors.push(`${rowId} certification ${key} lacks catalog or policy version`);
        }
        if (certification?.catalog_version !== ledger?.catalog_version) {
          errors.push(`${rowId} certification ${key} catalog_version does not match ledger`);
        }
        if (certification?.policy_version !== ledger?.certification_policy_version) {
          errors.push(`${rowId} certification ${key} policy_version does not match ledger`);
        }
        for (const [evidenceField, digestField] of [
          ["fixture_evidence", "fixture_sha256"],
          ["live_evidence", "live_sha256"],
          ["policy_evidence", "policy_digest"],
        ]) {
          const evidencePath = certification?.[evidenceField];
          const declaredDigest = certification?.[digestField];
          if (!isSafeRepoEvidencePath(evidencePath)) {
            errors.push(`${rowId} certification ${key} ${evidenceField} must be a safe repo path`);
          }
          if (!/^[a-f0-9]{64}$/.test(declaredDigest ?? "")) {
            errors.push(`${rowId} certification ${key} has invalid ${digestField}`);
          } else if (artifactDigests[evidencePath] !== declaredDigest) {
            errors.push(`${rowId} certification ${key} ${evidenceField} digest does not match`);
          }
        }
        const evidencePaths = [
          certification?.fixture_evidence,
          certification?.live_evidence,
          certification?.policy_evidence,
        ];
        if (new Set(evidencePaths).size !== evidencePaths.length) {
          errors.push(`${rowId} certification ${key} must use distinct evidence artifacts`);
        }
        const routePolicy = certificationArtifacts[
          certification?.policy_evidence
        ]?.details?.route_policy;
        for (const [artifactType, evidenceField] of [
          ["fixture", "fixture_evidence"],
          ["live", "live_evidence"],
          ["policy", "policy_evidence"],
        ]) {
          if (!candidate || !certificationArtifactMatches(
            certificationArtifacts[certification?.[evidenceField]],
            artifactType,
            certification,
            candidate,
            routePolicy,
          )) {
            errors.push(`${rowId} certification ${key} ${artifactType} artifact content is invalid`);
          }
        }
        if (!isActualDate(certification?.certified_on)
          || !isActualDate(certification?.valid_until)
          || certification.certified_on > asOf
          || certification.certified_on > certification.valid_until) {
          errors.push(`${rowId} certification ${key} has invalid dates`);
        } else if (daysBetween(certification.certified_on, certification.valid_until)
          > MAX_CERTIFICATION_VALIDITY_DAYS) {
          errors.push(`${rowId} certification ${key} exceeds the validity horizon`);
        }
      }
    }

    if (!Array.isArray(row.evidence) || row.evidence.length === 0) {
      errors.push(`${rowId} must contain evidence`);
    } else {
      for (const entry of row.evidence) {
        if (!isObject(entry)
          || !EVIDENCE_KINDS.has(entry.kind)
          || !EVIDENCE_REFERENCE_TYPES.has(entry.reference_type)
          || !evidenceReferenceIsValid(entry)
          || typeof entry.claim !== "string"
          || entry.claim.trim().length < 20) {
          errors.push(`${rowId} contains malformed evidence; HTTPS or typed repo evidence is required`);
        }
      }
    }
    if (!isObject(row.ownership)
      || !isNonEmptyString(row.ownership.follow_up_task)
      || !isNonEmptyString(row.ownership.revisit_trigger)) {
      errors.push(`${rowId} ownership requires follow_up_task and revisit_trigger`);
    }
    if (!isNonEmptyString(row.resolution_reason)) {
      errors.push(`${rowId} resolution_reason is required`);
    } else if (row.resolution !== "unreviewed" && row.resolution_reason.trim().length < 40) {
      errors.push(`${rowId} resolution_reason must be at least 40 characters`);
    }
    if (row.resolution === "mapped") {
      if (!Array.isArray(row.canonical_targets) || row.canonical_targets.length !== 1) {
        errors.push(`${rowId} mapped resolution requires exactly one canonical target`);
      }
      if (!credentiallessRouteCandidates(row).some(isDiscoverableRoute)) {
        errors.push(`${rowId} mapped resolution requires a discoverable credentialless route candidate`);
      }
      if (!Array.isArray(row.capabilities) || !row.capabilities.includes("search")) {
        errors.push(`${rowId} mapped resolution requires search capability`);
      }
      if (!Array.isArray(row.declared_surfaces)
        || row.declared_surfaces.length !== SURFACES.size
        || ![...SURFACES].every((surface) => row.declared_surfaces.includes(surface))) {
        errors.push(`${rowId} mapped resolution requires both declared surfaces`);
      }
      if (!hasEvidence(row, "route_triage")) {
        errors.push(`${rowId} mapped resolution requires route_triage evidence`);
      }
    }
    if (row.resolution === "duplicate" && !isNonEmptyString(row.duplicate_of_inventory_id)) {
      errors.push(`${rowId} duplicate resolution requires duplicate_of_inventory_id`);
    }
    if (row.resolution === "credentialed_out_of_scope") {
      if (Array.isArray(row.route_candidates)
        && credentiallessRouteCandidates(row).length > 0) {
        errors.push(`${rowId} credentialed_out_of_scope cannot contain a credentialless route`);
      }
      if (!hasCompleteCredentiallessReview(row)) {
        errors.push(`${rowId} credentialed_out_of_scope requires all credentialless route kinds reviewed`);
      }
    } else if (row.credentialless_route_review !== null) {
      errors.push(`${rowId} credentialless_route_review is only valid for credentialed exclusions`);
    }
    if (EVIDENCED_TERMINAL_RESOLUTIONS.has(row.resolution)
      && !hasEvidence(row, "resolution_review")) {
      errors.push(`${rowId} ${row.resolution} requires resolution_review evidence`);
    }
    if (row.review_status === "reviewed" && !hasWellFormedReviewedOwnership(row, asOf)) {
      if (typeof row.ownership?.reviewer !== "string"
        || row.ownership.reviewer.trim().length < 3) {
        errors.push(`${rowId} reviewed rows require a stable reviewer identity`);
      }
      if (!isActualDate(row.ownership?.review_date)
        || row.ownership.review_date > asOf) {
        errors.push(`${rowId} reviewed rows require a valid non-future review_date`);
      }
    }
    if (row.closure_approval !== null && !closureApprovalIsWellFormed(row, asOf)) {
      errors.push(`${rowId} closure_approval is malformed or does not bind the decision`);
      if (row.closure_approval?.approval_reference !== row.ownership?.follow_up_task) {
        errors.push(`${rowId} closure_approval must reference ownership follow_up_task`);
      }
    }
    if (row.review_status === "unreviewed") unreviewed += 1;
    if (isSubstantivelyTriaged(row, asOf, trustedReviewers)) triaged += 1;
    else blockers.push(rowId);
    if (isTerminal(
      row,
      asOf,
      artifactDigests,
      certificationArtifacts,
      ledger?.catalog_version,
      ledger?.certification_policy_version,
      trustedReviewers,
      trustedApprovals,
    )) terminal += 1;
  }

  const requiredSourceStates = {};
  for (const [inventoryId, requirement] of Object.entries(requiredSources)) {
    const row = ledgerById.get(inventoryId);
    const generalRoute = Array.isArray(row?.route_candidates)
      ? row.route_candidates.find((candidate) => (
        routeMatchesRequirement(candidate, requirement.generalRoute)
      ))
      : null;
    const lookupRoute = Array.isArray(row?.route_candidates)
      ? row.route_candidates.find((candidate) => (
        routeMatchesRequirement(candidate, requirement.lookupRoute)
      ))
      : null;
    const intervalRoute = Array.isArray(row?.route_candidates)
      ? row.route_candidates.find((candidate) => (
        routeMatchesRequirement(candidate, requirement.intervalRoute)
      ))
      : null;
    const mappingSatisfied = row?.resolution === "mapped"
      && Array.isArray(row.canonical_targets)
      && row.canonical_targets.length === 1
      && row.canonical_targets[0] === requirement.canonicalTarget
      && Array.isArray(row.declared_surfaces)
      && [...SURFACES].every((surface) => row.declared_surfaces.includes(surface))
      && Boolean(generalRoute)
      && Boolean(lookupRoute)
      && Boolean(intervalRoute)
      && isSubstantivelyTriaged(row, asOf, trustedReviewers);
    requiredSourceStates[inventoryId] = {
      canonical_target: requirement.canonicalTarget,
      required_general_route_id: requirement.generalRoute.id,
      required_lookup_route_id: requirement.lookupRoute.id,
      required_interval_route_id: requirement.intervalRoute.id,
      captured_label: manifestById.get(inventoryId)?.label ?? null,
      resolution: row?.resolution ?? null,
      canonical_targets: Array.isArray(row?.canonical_targets) ? [...row.canonical_targets] : [],
      declared_surfaces: Array.isArray(row?.declared_surfaces) ? [...row.declared_surfaces] : [],
      mapping_satisfied: mappingSatisfied,
    };
  }
  const requiredSourceBlockers = Object.entries(requiredSourceStates)
    .filter(([, state]) => !state.mapping_satisfied)
    .map(([inventoryId]) => inventoryId);

  for (const item of manifestItems) {
    if (!ledgerById.has(item.inventory_id)) {
      errors.push(`ledger is missing inventory_id ${item.inventory_id}`);
    }
  }

  for (const [targetId, definition] of targetDefinitionsById) {
    for (const inventoryId of definition.inventory_ids ?? []) {
      const row = ledgerById.get(inventoryId);
      if (!row) {
        errors.push(`target definition ${targetId} references unknown ${inventoryId}`);
      } else if (!Array.isArray(row.canonical_targets)
        || !row.canonical_targets.includes(targetId)) {
        errors.push(`target definition ${targetId} is not referenced by ${inventoryId}`);
      }
    }
  }

  for (const row of ledgerRows) {
    if (row?.resolution !== "duplicate") continue;
    if (!ledgerById.has(row.duplicate_of_inventory_id)) {
      errors.push(`${row.inventory_id} duplicate_of_inventory_id is unknown`);
    } else if (row.duplicate_of_inventory_id === row.inventory_id) {
      errors.push(`${row.inventory_id} cannot duplicate itself`);
    }
  }

  for (const row of ledgerRows) {
    if (row?.resolution !== "duplicate") continue;
    const visited = new Set([row.inventory_id]);
    let current = row;
    while (current?.resolution === "duplicate") {
      const nextId = current.duplicate_of_inventory_id;
      if (visited.has(nextId)) {
        errors.push(`${row.inventory_id} duplicate chain contains a cycle`);
        current = null;
        break;
      }
      visited.add(nextId);
      current = ledgerById.get(nextId);
      if (!current) break;
    }
    if (current && current.resolution !== "mapped") {
      errors.push(`${row.inventory_id} duplicate chain must terminate at a mapped source`);
    }
  }

  const counts = {
    manifest: manifestItems.length,
    ledger: ledgerRows.length,
    unreviewed,
    triaged,
    terminal,
    non_terminal: manifestItems.length - terminal,
    resolution: resolutionCounts,
    implementation: implementationCounts,
    fixture: fixtureCounts,
    live: liveCounts,
  };
  const structurallyValid = schemaValidated && errors.length === 0;
  const contractFreezeReady = structurallyValid
    && manifestItems.length > 0
    && triaged === manifestItems.length
    && requiredSourceBlockers.length === 0;
  return {
    errors,
    schema_validated: schemaValidated,
    as_of: asOf,
    trusted_reviewer_ids: [...trustedReviewers].sort(),
    trusted_approval_references: [...trustedApprovals].sort(),
    counts,
    blockers,
    required_sources: requiredSourceStates,
    required_source_blockers: requiredSourceBlockers,
    digests: {
      manifest: sha256(canonicalJson(manifest)),
      ledger: sha256(canonicalJson(ledger)),
      ledger_rows: sha256(canonicalJson(ledgerRows)),
      schema: schema ? sha256(canonicalJson(schema)) : null,
      validator: validatorDigest,
      schema_validator: schemaValidatorDigest,
    },
    structurally_valid: structurallyValid,
    contract_freeze_ready: contractFreezeReady,
    inventory_delivery_ready: contractFreezeReady && terminal === manifestItems.length,
  };
}


function parseRoot(argv) {
  const rootIndex = argv.indexOf("--root");
  if (rootIndex === -1) return process.cwd();
  if (!argv[rootIndex + 1]) throw new Error("--root requires a path");
  return path.resolve(argv[rootIndex + 1]);
}


function parseGate(argv) {
  const gateIndex = argv.indexOf("--gate");
  if (gateIndex === -1) return "contract";
  const gate = argv[gateIndex + 1];
  if (!new Set(["structure", "contract", "delivery"]).has(gate)) {
    throw new Error("--gate must be structure, contract, or delivery");
  }
  return gate;
}


function runSchemaGate(root, schemaPath, manifestPath, ledgerPath, certificationPaths) {
  const python = process.env.RESEARCH_INVENTORY_PYTHON
    ?? (process.env.VIRTUAL_ENV
      ? path.join(process.env.VIRTUAL_ENV, "bin", "python")
      : "python3");
  const checkerPath = path.join(
    root,
    "Helper_Scripts",
    "validate_research_source_inventory_schema.py",
  );
  const certificationArguments = certificationPaths.flatMap((relativePath, index) => [
    "--document",
    `certification-${index + 1}`,
    path.join(root, relativePath),
  ]);
  const result = spawnSync(python, [
    checkerPath,
    "--schema", schemaPath,
    "--document", "manifest", manifestPath,
    "--document", "ledger", ledgerPath,
    ...certificationArguments,
  ], {
    encoding: "utf8",
    maxBuffer: 10 * 1024 * 1024,
    timeout: 30_000,
  });
  if (result.error) {
    return {
      validated: false,
      errors: [`schema validation could not run: ${result.error.message}`],
    };
  }
  try {
    const payload = JSON.parse(result.stdout);
    if (!Array.isArray(payload.errors)) throw new Error("missing errors array");
    const expectedStatus = payload.errors.length === 0 ? 0 : 1;
    if (result.status !== expectedStatus) {
      throw new Error(
        `checker exit ${result.status ?? "signal"} did not match ${payload.errors.length} errors`,
      );
    }
    return { validated: true, errors: payload.errors };
  } catch (error) {
    const detail = result.stderr?.trim() || error.message;
    return {
      validated: false,
      errors: [`schema validation returned an invalid result: ${detail}`],
    };
  }
}


export function collectCertificationArtifacts(root, ledger) {
  const digests = {};
  const documents = {};
  const paths = [];
  const errors = [];
  const seen = new Set();
  const realRoot = fs.realpathSync(root);
  const certificationRoot = path.join(
    realRoot,
    "Docs",
    "Design",
    "research_source_inventory",
    "certifications",
  );
  for (const row of Array.isArray(ledger?.rows) ? ledger.rows : []) {
    for (const certification of Array.isArray(row?.certifications) ? row.certifications : []) {
      for (const field of ["fixture_evidence", "live_evidence", "policy_evidence"]) {
        const relativePath = certification?.[field];
        if (!isSafeRepoEvidencePath(relativePath) || seen.has(relativePath)) continue;
        seen.add(relativePath);
        const absolutePath = path.resolve(realRoot, relativePath);
        try {
          const stat = fs.lstatSync(absolutePath);
          if (stat.isSymbolicLink()) {
            errors.push(`certification artifact ${relativePath} cannot be a symbolic link`);
            continue;
          }
          if (!stat.isFile()) {
            errors.push(`certification artifact ${relativePath} must be a regular file`);
            continue;
          }
          const realCertificationRoot = fs.realpathSync(certificationRoot);
          const realArtifactPath = fs.realpathSync(absolutePath);
          if (!realCertificationRoot.startsWith(`${realRoot}${path.sep}`)
            || !realArtifactPath.startsWith(`${realCertificationRoot}${path.sep}`)) {
            errors.push(`certification artifact ${relativePath} resolves outside its repository directory`);
            continue;
          }
          const raw = fs.readFileSync(realArtifactPath);
          digests[relativePath] = crypto.createHash("sha256").update(raw).digest("hex");
          documents[relativePath] = JSON.parse(raw.toString("utf8"));
          paths.push(relativePath);
        } catch (error) {
          errors.push(`certification artifact ${relativePath} could not be read: ${error.message}`);
        }
      }
    }
  }
  return { digests, documents, paths, errors };
}


export function parseAsOf(argv) {
  const asOfIndex = argv.indexOf("--as-of");
  if (asOfIndex === -1) return new Date().toISOString().slice(0, 10);
  const asOf = argv[asOfIndex + 1];
  const parsed = /^\d{4}-\d{2}-\d{2}$/.test(asOf ?? "")
    ? new Date(`${asOf}T00:00:00Z`)
    : null;
  if (!parsed || Number.isNaN(parsed.valueOf())
    || parsed.toISOString().slice(0, 10) !== asOf) {
    throw new Error("--as-of requires YYYY-MM-DD");
  }
  return asOf;
}


export function parseRepeatedOption(argv, option) {
  const values = [];
  for (let index = 0; index < argv.length; index += 1) {
    if (argv[index] !== option) continue;
    const value = argv[index + 1];
    if (!isNonEmptyString(value) || value.startsWith("--")) {
      throw new Error(`${option} requires a value`);
    }
    values.push(value);
    index += 1;
  }
  return [...new Set(values)];
}


export function gateExitCode(report, gate) {
  if (report.errors.length > 0 || !report.structurally_valid) return 1;
  if (gate === "contract" && !report.contract_freeze_ready) return 2;
  if (gate === "delivery" && !report.inventory_delivery_ready) return 2;
  return 0;
}


function runCli() {
  const argv = process.argv.slice(2);
  const root = parseRoot(argv);
  const gate = parseGate(argv);
  const asOf = parseAsOf(argv);
  const trustedReviewerIds = parseRepeatedOption(argv, "--trusted-reviewer");
  const trustedApprovalReferences = parseRepeatedOption(argv, "--trusted-approval");
  const directory = path.join(root, "Docs", "Design", "research_source_inventory");
  const manifestPath = path.join(directory, "sourclip-research-sources-2026-07-13.json");
  const ledgerPath = path.join(directory, "research-source-coverage-ledger-2026-07-13.json");
  const schemaPath = path.join(directory, "research-source-inventory.schema.json");
  const manifest = JSON.parse(fs.readFileSync(manifestPath, "utf8"));
  const ledger = JSON.parse(fs.readFileSync(ledgerPath, "utf8"));
  const schema = JSON.parse(fs.readFileSync(schemaPath, "utf8"));
  const certificationCollection = collectCertificationArtifacts(root, ledger);
  const schemaGate = runSchemaGate(
    root,
    schemaPath,
    manifestPath,
    ledgerPath,
    certificationCollection.paths,
  );
  const schemaValidatorPath = path.join(
    root,
    "Helper_Scripts",
    "validate_research_source_inventory_schema.py",
  );
  const report = validateInventoryDocuments(manifest, ledger, {
    schema,
    asOf,
    schemaValidated: schemaGate.validated,
    schemaErrors: [...schemaGate.errors, ...certificationCollection.errors],
    artifactDigests: certificationCollection.digests,
    certificationArtifacts: certificationCollection.documents,
    trustedReviewerIds,
    trustedApprovalReferences,
    validatorDigest: sha256(fs.readFileSync(fileURLToPath(import.meta.url), "utf8")),
    schemaValidatorDigest: sha256(fs.readFileSync(schemaValidatorPath, "utf8")),
  });

  if (argv.includes("--json")) {
    process.stdout.write(`${JSON.stringify(report, null, 2)}\n`);
  } else if (report.errors.length) {
    for (const error of report.errors) process.stderr.write(`ERROR: ${error}\n`);
  } else {
    process.stdout.write(
      `Inventory structurally valid: ${report.counts.manifest} manifest rows; `
      + `contract_freeze_ready=${report.contract_freeze_ready}; `
      + `inventory_delivery_ready=${report.inventory_delivery_ready}; `
      + `${report.counts.terminal} terminal and ${report.counts.non_terminal} non-terminal.\n`,
    );
  }
  process.exitCode = gateExitCode(report, gate);
}


const isMain = process.argv[1]
  && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);
if (isMain) runCli();
