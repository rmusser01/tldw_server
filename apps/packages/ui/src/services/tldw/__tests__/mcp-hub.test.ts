import { beforeEach, describe, expect, it, vi } from "vitest"

const mocks = vi.hoisted(() => ({
  bgRequestClient: vi.fn()
}))

vi.mock("@/services/background-proxy", () => ({
  bgRequestClient: (...args: unknown[]) => mocks.bgRequestClient(...args)
}))

import {
  checkGovernancePackUpdates,
  dryRunGovernancePack,
  getEffectivePolicy,
  getMcpHubReadiness,
  getGovernancePackTrustPolicy,
  listGovernancePacks,
  listCapabilityAdapterMappings,
  previewCapabilityAdapterMapping,
  refreshExternalServerDiscovery,
  refreshExternalServerReadinessDiscovery,
  setExternalServerSecret,
  validateExternalServer,
  updateGovernancePackTrustPolicy
} from "../mcp-hub"
import type { McpHubServerReadiness } from "../mcp-hub"

describe("mcp hub service client", () => {
  beforeEach(() => {
    vi.clearAllMocks()
  })

  it("maps external secret set response without exposing plaintext", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      server_id: "docs",
      secret_configured: true,
      key_hint: "cdef"
    })

    const out = await setExternalServerSecret("docs", "my-secret")

    expect(out.secret_configured).toBe(true)
    expect(JSON.stringify(out)).not.toContain("my-secret")
    expect(mocks.bgRequestClient).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/mcp/hub/external-servers/docs/secret",
        method: "POST",
        body: { secret: "my-secret" }
      })
    )
  })

  it("returns structured external discovery refresh errors for completed refresh responses", async () => {
    const partialRefresh = {
      ok: false,
      message: "Discovery refreshed with errors",
      errors: { docs: "external_server_discovery_failed" },
      refreshed_servers: 1,
      total_servers: 2
    }
    mocks.bgRequestClient.mockResolvedValueOnce(partialRefresh)

    const out = await refreshExternalServerDiscovery("docs")

    expect(out).toEqual(partialRefresh)
    expect(mocks.bgRequestClient).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/mcp/hub/external-servers/refresh-discovery",
        method: "POST",
        body: { server_id: "docs" }
      })
    )
  })

  it("requests MCP Hub readiness through the dedicated readiness endpoint", async () => {
    const unknownCredentialServer: McpHubServerReadiness = {
      server_id: "remote-docs",
      server_name: "Remote Docs",
      display_state: "needs_attention",
      credential_state: "unknown",
      tool_count: 0,
      reason_codes: ["discovery_not_run"],
      primary_reason_code: "discovery_not_run",
      allowed_actions: ["refresh_discovery", "edit_config"],
      message: "Server is saved, but tool discovery has not run."
    }
    mocks.bgRequestClient.mockResolvedValueOnce({
      display_state: "needs_attention",
      reason_codes: ["discovery_not_run"],
      primary_reason_code: "discovery_not_run",
      allowed_actions: ["refresh_discovery"],
      message: "Server is saved, but tool discovery has not run.",
      servers: [unknownCredentialServer],
      total_servers: 1,
      ready_server_count: 0,
      checking_server_count: 0,
      attention_server_count: 1,
      no_tool_server_count: 0,
      stale_server_count: 0
    })

    const out = await getMcpHubReadiness()

    expect(out.display_state).toBe("needs_attention")
    expect(out.servers[0]?.credential_state).toBe("unknown")
    expect(mocks.bgRequestClient).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/mcp/hub/readiness",
        method: "GET"
      })
    )
  })

  it("validates external server readiness through the dedicated validate endpoint", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      server_id: "docs",
      server_name: "Docs",
      display_state: "ready",
      credential_state: "not_required",
      tool_count: 1,
      reason_codes: [],
      primary_reason_code: null,
      allowed_actions: ["open_tool_catalog", "view_details"],
      message: "Ready."
    })

    const out = await validateExternalServer("docs")

    expect(out.server_id).toBe("docs")
    expect(mocks.bgRequestClient).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/mcp/hub/external-servers/docs/validate",
        method: "POST"
      })
    )
  })

  it("refreshes external server discovery through the dedicated refresh endpoint", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      server_id: "docs",
      server_name: "Docs",
      display_state: "ready",
      credential_state: "not_required",
      tool_count: 1,
      reason_codes: [],
      primary_reason_code: null,
      allowed_actions: ["open_tool_catalog", "view_details"],
      message: "Ready.",
      refresh_result: {
        refreshed_servers: 1,
        total_servers: 1,
        virtual_tools: 1,
        errors: {}
      }
    })

    const out = await refreshExternalServerReadinessDiscovery("docs")

    expect(out.refresh_result?.virtual_tools).toBe(1)
    expect(mocks.bgRequestClient).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/mcp/hub/external-servers/docs/refresh-discovery",
        method: "POST"
      })
    )
  })

  it("encodes server ids for validate and refresh-discovery requests", async () => {
    mocks.bgRequestClient
      .mockResolvedValueOnce({
        server_id: "docs team:one",
        server_name: "Docs",
        display_state: "ready",
        credential_state: "not_required",
        tool_count: 1,
        reason_codes: [],
        primary_reason_code: null,
        allowed_actions: ["open_tool_catalog", "view_details"],
        message: "Ready."
      })
      .mockResolvedValueOnce({
        server_id: "docs team:one",
        server_name: "Docs",
        display_state: "ready",
        credential_state: "not_required",
        tool_count: 1,
        reason_codes: [],
        primary_reason_code: null,
        allowed_actions: ["open_tool_catalog", "view_details"],
        message: "Ready."
      })

    await validateExternalServer("docs team:one")
    await refreshExternalServerReadinessDiscovery("docs team:one")

    expect(mocks.bgRequestClient).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/mcp/hub/external-servers/docs%20team%3Aone/validate",
        method: "POST"
      })
    )
    expect(mocks.bgRequestClient).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/mcp/hub/external-servers/docs%20team%3Aone/refresh-discovery",
        method: "POST"
      })
    )
  })

  it("requests governance pack dry-run reports through the MCP Hub preview endpoint", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      report: {
        manifest: {
          pack_id: "researcher-pack",
          pack_version: "1.0.0",
          title: "Researcher Pack"
        },
        digest: "a".repeat(64),
        resolved_capabilities: ["tool.invoke.research"],
        unresolved_capabilities: [],
        capability_mapping_summary: [
          {
            capability_name: "tool.invoke.research",
            mapping_id: "research.global",
            mapping_scope_type: "global",
            mapping_scope_id: null,
            resolved_effects: { allowed_tools: ["web.search"] },
            supported_environment_requirements: ["workspace_bounded_read"],
            unsupported_environment_requirements: []
          }
        ],
        supported_environment_requirements: ["workspace_bounded_read"],
        unsupported_environment_requirements: [],
        warnings: [],
        blocked_objects: [],
        verdict: "importable"
      }
    })

    const payload = {
      owner_scope_type: "user" as const,
      pack: {
        manifest: { pack_id: "researcher-pack" },
        profiles: [],
        approvals: [],
        personas: [],
        assignments: []
      }
    }
    const out = await dryRunGovernancePack(payload)

    expect(out.report.verdict).toBe("importable")
    expect(out.report.capability_mapping_summary[0]?.mapping_id).toBe("research.global")
    expect(mocks.bgRequestClient).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/mcp/hub/governance-packs/dry-run",
        method: "POST",
        body: payload
      })
    )
  })

  it("lists governance packs with scope filters", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce([
      {
        id: 81,
        pack_id: "researcher-pack",
        pack_version: "1.0.0",
        title: "Researcher Pack",
        owner_scope_type: "user",
        owner_scope_id: 7,
        bundle_digest: "a".repeat(64),
        signer_fingerprint: "ABCD1234",
        signer_identity: "Release Bot <bot@example.com>",
        verified_object_type: "commit",
        verification_result_code: "verified_and_trusted",
        verification_warning_code: "signer_rotated_trusted",
        manifest: {}
      }
    ])

    const out = await listGovernancePacks({ owner_scope_type: "user", owner_scope_id: 7 })

    expect(out).toHaveLength(1)
    expect(out[0]?.signer_fingerprint).toBe("ABCD1234")
    expect(out[0]?.verification_warning_code).toBe("signer_rotated_trusted")
    expect(mocks.bgRequestClient).toHaveBeenCalledWith(
      expect.objectContaining({
        path: "/api/v1/mcp/hub/governance-packs?owner_scope_type=user&owner_scope_id=7",
        method: "GET"
      })
    )
  })

  it("maps effective policy responses with authored and resolved documents", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      enabled: true,
      allowed_tools: ["web.search"],
      denied_tools: [],
      capabilities: ["tool.invoke.research", "network.external.search"],
      authored_policy_document: { capabilities: ["tool.invoke.research", "network.external.search"] },
      resolved_policy_document: {
        capabilities: ["tool.invoke.research", "network.external.search"],
        allowed_tools: ["web.search"]
      },
      resolved_capabilities: ["tool.invoke.research"],
      unresolved_capabilities: ["network.external.search"],
      capability_mapping_summary: [
        {
          capability_name: "tool.invoke.research",
          mapping_id: "research.global",
          mapping_scope_type: "global",
          mapping_scope_id: null,
          resolved_effects: { allowed_tools: ["web.search"] },
          supported_environment_requirements: ["workspace_bounded_read"],
          unsupported_environment_requirements: []
        }
      ],
      capability_warnings: [
        "profile:researcher: No active capability adapter mapping found for 'network.external.search'"
      ],
      policy_document: { allowed_tools: ["web.search"] },
      sources: [],
      provenance: []
    })

    const out = await getEffectivePolicy({ persona_id: "researcher" })

    expect(out.authored_policy_document.capabilities).toEqual([
      "tool.invoke.research",
      "network.external.search"
    ])
    expect(out.capability_mapping_summary[0]?.mapping_id).toBe("research.global")
    expect(out.unresolved_capabilities).toEqual(["network.external.search"])
  })

  it("requests capability mapping previews and listing through MCP Hub endpoints", async () => {
    mocks.bgRequestClient
      .mockResolvedValueOnce([
        {
          id: 9,
          mapping_id: "research.global",
          title: "Research Mapping",
          owner_scope_type: "global",
          owner_scope_id: null,
          capability_name: "tool.invoke.research",
          adapter_contract_version: 1,
          resolved_policy_document: { allowed_tools: ["web.search"] },
          supported_environment_requirements: ["workspace_bounded_read"],
          is_active: true
        }
      ])
      .mockResolvedValueOnce({
        normalized_mapping: {
          mapping_id: "research.global",
          title: "Research Mapping",
          owner_scope_type: "global",
          owner_scope_id: null,
          capability_name: "tool.invoke.research",
          adapter_contract_version: 1,
          resolved_policy_document: { allowed_tools: ["web.search"] },
          supported_environment_requirements: ["workspace_bounded_read"],
          is_active: true
        },
        warnings: [],
        affected_scope_summary: {
          owner_scope_type: "global",
          owner_scope_id: null,
          display_scope: "Global"
        }
      })

    const listOut = await listCapabilityAdapterMappings({ owner_scope_type: "global" })
    const previewOut = await previewCapabilityAdapterMapping({
      mapping_id: "research.global",
      title: "Research Mapping",
      owner_scope_type: "global",
      owner_scope_id: null,
      capability_name: "tool.invoke.research",
      adapter_contract_version: 1,
      resolved_policy_document: { allowed_tools: ["web.search"] },
      supported_environment_requirements: ["workspace_bounded_read"],
      is_active: true
    })

    expect(listOut).toHaveLength(1)
    expect(previewOut.affected_scope_summary.display_scope).toBe("Global")
    expect(mocks.bgRequestClient).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/mcp/hub/capability-mappings?owner_scope_type=global",
        method: "GET"
      })
    )
    expect(mocks.bgRequestClient).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/mcp/hub/capability-mappings/preview",
        method: "POST"
      })
    )
  })

  it("maps governance pack trust policies with trusted signer bindings", async () => {
    mocks.bgRequestClient
      .mockResolvedValueOnce({
        allow_local_path_sources: false,
        allowed_local_roots: [],
        allow_git_sources: true,
        allowed_git_hosts: ["github.com"],
        allowed_git_repositories: ["github.com/example/researcher-pack"],
        allowed_git_ref_kinds: ["tag"],
        require_git_signature_verification: true,
        trusted_signers: [
          {
            fingerprint: "ABCD1234",
            display_name: "Release Bot",
            repo_bindings: ["github.com/example/researcher-pack"],
            status: "active"
          }
        ],
        policy_fingerprint: "policy-1"
      })
      .mockResolvedValueOnce({
        allow_local_path_sources: false,
        allowed_local_roots: [],
        allow_git_sources: true,
        allowed_git_hosts: ["github.com"],
        allowed_git_repositories: ["github.com/example/researcher-pack"],
        allowed_git_ref_kinds: ["tag"],
        require_git_signature_verification: true,
        trusted_signers: [
          {
            fingerprint: "ABCD1234",
            display_name: "Release Bot",
            repo_bindings: ["github.com/example/researcher-pack"],
            status: "active"
          }
        ],
        policy_fingerprint: "policy-2"
      })

    const current = await getGovernancePackTrustPolicy()
    const updated = await updateGovernancePackTrustPolicy({
      ...current,
      policy_fingerprint: "policy-1"
    })

    expect(current.trusted_signers[0]?.fingerprint).toBe("ABCD1234")
    expect(updated.policy_fingerprint).toBe("policy-2")
    expect(mocks.bgRequestClient).toHaveBeenNthCalledWith(
      1,
      expect.objectContaining({
        path: "/api/v1/mcp/hub/governance-packs/trust-policy",
        method: "GET"
      })
    )
    expect(mocks.bgRequestClient).toHaveBeenNthCalledWith(
      2,
      expect.objectContaining({
        path: "/api/v1/mcp/hub/governance-packs/trust-policy",
        method: "PUT",
        body: expect.objectContaining({
          trusted_signers: [
            expect.objectContaining({
              fingerprint: "ABCD1234",
              repo_bindings: ["github.com/example/researcher-pack"]
            })
          ],
          policy_fingerprint: "policy-1"
        })
      })
    )
  })

  it("maps signer diagnostics on governance pack update checks", async () => {
    mocks.bgRequestClient.mockResolvedValueOnce({
      governance_pack_id: 81,
      status: "newer_version_available",
      installed_manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.0.0",
        title: "Researcher Pack"
      },
      candidate_manifest: {
        pack_id: "researcher-pack",
        pack_version: "1.1.0",
        title: "Researcher Pack"
      },
      source_commit_resolved: "def456",
      pack_content_digest: "d".repeat(64),
      signer_fingerprint: "BBBB2222",
      signer_identity: "Backup Bot <backup@example.com>",
      verified_object_type: "commit",
      verification_result_code: "verified_and_trusted",
      verification_warning_code: "signer_rotated_trusted"
    })

    const out = await checkGovernancePackUpdates(81)

    expect(out.signer_fingerprint).toBe("BBBB2222")
    expect(out.verification_warning_code).toBe("signer_rotated_trusted")
  })
})
