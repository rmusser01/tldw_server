import { describe, expect, it } from "vitest"

import {
  buildACPAgentSetupSummary,
  buildACPSetupIssues,
  isACPAgentReadyToStart,
  normalizeACPHealthStatus
} from "@/services/acp/readiness"
import type { ACPAgentInfo } from "@/services/acp/types"

describe("ACP readiness normalization", () => {
  it("treats an empty agent inventory as unavailable even when overall is degraded", () => {
    const health = normalizeACPHealthStatus({
      runner: { status: "ok" },
      agents: [],
      overall: "degraded",
      message: "Runner is present but no agents are configured"
    })

    expect(health?.agent).toBe("unavailable")
    expect(buildACPSetupIssues(health).map((issue) => issue.code)).toContain("agent_unavailable")
  })

  it("treats external ACP adapters as startable when the entrypoint is ready to probe", () => {
    const agent: ACPAgentInfo = {
      type: "codex",
      name: "Codex",
      description: "Codex CLI via codex-acp adapter.",
      is_configured: false,
      entrypoint: {
        profile_key: "codex",
        entrypoint_strategy: "external_acp_adapter",
        probe_state: "ready_to_probe",
        acp_command: "codex-acp",
        acp_args: [],
        primary_blocker: null,
        blockers: [],
        status_message: "codex-acp is installed and ready to start.",
        docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
        display_command: "codex-acp",
        display_binary_found: true,
        adapter_found: true,
        credential_state: "delegated",
        adapter_source: "npm",
        adapter_package: "@zed-industries/codex-acp",
        adapter_version: "0.15.0",
        runtime_backend: "node"
      }
    }

    expect(isACPAgentReadyToStart(agent)).toBe(true)
    expect(buildACPAgentSetupSummary(agent)).toEqual({
      disabled: false,
      title: "Ready to start",
      description: "codex-acp is installed and ready to start."
    })
  })

  it("explains missing external adapters without asking for an API key", () => {
    const agent: ACPAgentInfo = {
      type: "codex",
      name: "Codex",
      description: "Codex CLI via codex-acp adapter.",
      is_configured: false,
      requires_api_key: "OPENAI_API_KEY",
      entrypoint: {
        profile_key: "codex",
        entrypoint_strategy: "external_acp_adapter",
        probe_state: "blocked",
        acp_command: "codex-acp",
        acp_args: [],
        primary_blocker: "adapter_missing",
        blockers: ["adapter_missing"],
        status_message: "Install codex-acp before starting Codex through ACP.",
        docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
        display_command: "codex-acp",
        display_binary_found: false,
        adapter_found: false,
        credential_state: "delegated",
        adapter_source: "npm",
        adapter_package: "@zed-industries/codex-acp",
        adapter_version: "0.15.0",
        runtime_backend: "node"
      }
    }

    const summary = buildACPAgentSetupSummary(agent)

    expect(isACPAgentReadyToStart(agent)).toBe(false)
    expect(summary.disabled).toBe(true)
    expect(summary.title).toBe("Install adapter")
    expect(summary.description).toContain("codex-acp")
    expect(summary.description).toContain("@zed-industries/codex-acp")
    expect(summary.description).not.toContain("OPENAI_API_KEY")
  })

  it("does not let stale is_configured override a blocked mutable adapter entrypoint", () => {
    const agent: ACPAgentInfo = {
      type: "codex",
      name: "Codex",
      description: "Codex CLI via codex-acp adapter.",
      is_configured: true,
      entrypoint: {
        profile_key: "codex",
        entrypoint_strategy: "external_acp_adapter",
        probe_state: "blocked",
        acp_command: "npx",
        acp_args: ["@zed-industries/codex-acp@latest"],
        primary_blocker: "mutable_adapter_invocation",
        blockers: ["mutable_adapter_invocation"],
        status_message: "Pin codex-acp to an explicit version before enabling this profile.",
        docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
        display_command: "npx @zed-industries/codex-acp@latest",
        display_binary_found: true,
        adapter_found: true,
        credential_state: "delegated",
        adapter_source: "npm",
        adapter_package: "@zed-industries/codex-acp",
        runtime_backend: "node"
      }
    }

    const summary = buildACPAgentSetupSummary(agent)

    expect(isACPAgentReadyToStart(agent)).toBe(false)
    expect(summary.disabled).toBe(true)
    expect(summary.title).toBe("Pin adapter version")
    expect(summary.description).toContain("Pin codex-acp")
  })

  it("uses primary_blocker for setup copy when secondary blockers are also present", () => {
    const agent: ACPAgentInfo = {
      type: "codex",
      name: "Codex",
      description: "Codex CLI via codex-acp adapter.",
      is_configured: false,
      entrypoint: {
        profile_key: "codex",
        entrypoint_strategy: "external_acp_adapter",
        probe_state: "blocked",
        acp_command: "npx",
        acp_args: ["@zed-industries/codex-acp@latest"],
        primary_blocker: "mutable_adapter_invocation",
        blockers: ["adapter_missing", "mutable_adapter_invocation"],
        status_message: "Pin codex-acp before resolving remaining setup issues.",
        docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
        display_command: "npx @zed-industries/codex-acp@latest",
        display_binary_found: false,
        adapter_found: false,
        credential_state: "delegated",
        adapter_source: "npm",
        adapter_package: "@zed-industries/codex-acp",
        runtime_backend: "node"
      }
    }

    const summary = buildACPAgentSetupSummary(agent)

    expect(summary.disabled).toBe(true)
    expect(summary.title).toBe("Pin adapter version")
    expect(summary.description).toContain("Pin codex-acp")
  })

  it("maps agent binary blocker aliases to agent binary setup copy", () => {
    const agent: ACPAgentInfo = {
      type: "goose",
      name: "Goose",
      description: "Native ACP agent.",
      is_configured: false,
      entrypoint: {
        profile_key: "goose",
        entrypoint_strategy: "native_acp",
        probe_state: "blocked",
        acp_command: "goose",
        acp_args: [],
        primary_blocker: "binary_missing",
        blockers: ["binary_missing"],
        status_message: "Install Goose before starting ACP sessions.",
        docs_url: "/docs-static/Development/ACP_Compatibility_Matrix.md",
        display_command: "goose",
        display_binary_found: false,
        adapter_found: null,
        credential_state: "unknown",
        runtime_backend: "acp_downstream"
      }
    }

    const summary = buildACPAgentSetupSummary(agent)

    expect(summary.disabled).toBe(true)
    expect(summary.title).toBe("Install agent binary")
    expect(summary.description).toContain("Install Goose")
  })
})
