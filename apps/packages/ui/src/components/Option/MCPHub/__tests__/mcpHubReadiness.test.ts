import { describe, expect, it } from "vitest"

import {
  getMcpCredentialState,
  getMcpHubReadiness,
  getMcpServerReadiness,
  type McpReadinessAction,
  type McpReasonCode
} from "../mcpHubReadiness"
import type {
  McpHubExternalServer,
  McpHubExternalServerCredentialSlot,
  McpHubToolRegistryEntry
} from "@/services/tldw/mcp-hub"

const server = (overrides: Partial<McpHubExternalServer> = {}): McpHubExternalServer => ({
  id: "toy-server",
  name: "Toy Server",
  enabled: true,
  owner_scope_type: "user",
  transport: "stdio",
  config: {},
  secret_configured: false,
  ...overrides
})

const credentialSlot = (
  overrides: Partial<McpHubExternalServerCredentialSlot> = {}
): McpHubExternalServerCredentialSlot => ({
  server_id: "toy-server",
  slot_name: "api_key",
  display_name: "API key",
  secret_kind: "api_key",
  privilege_class: "read",
  is_required: true,
  secret_configured: false,
  ...overrides
})

const registryEntry = (
  overrides: Partial<McpHubToolRegistryEntry> = {}
): McpHubToolRegistryEntry => ({
  tool_name: "ext.toy-server.search",
  display_name: "Search",
  description: "Search the toy service",
  module: "external.toy-server",
  category: "search",
  risk_class: "low",
  capabilities: ["search.query"],
  mutates_state: false,
  uses_filesystem: false,
  uses_processes: false,
  uses_network: true,
  uses_credentials: false,
  supports_arguments_preview: true,
  path_boundable: false,
  path_argument_hints: [],
  metadata_source: "explicit",
  metadata_warnings: [],
  ...overrides
})

const expectSingleReasonActions = (
  reasonCode: McpReasonCode,
  allowedActions: McpReadinessAction[]
) => {
  const readiness = getMcpServerReadiness({
    server: server(),
    registryEntries: reasonCode === "discovery_not_run" ? [] : [registryEntry()],
    readinessHint: {
      preflightFailed: reasonCode === "preflight_failed",
      unreachable: reasonCode === "unreachable",
      discoveryFailed: reasonCode === "discovery_failed",
      configChanged: reasonCode === "config_changed",
      catalogExpired: reasonCode === "catalog_expired",
      partialCapability: reasonCode === "partial_capability"
    }
  })

  expect(readiness.primaryReasonCode).toBe(reasonCode)
  expect(readiness.reasonCodes).toEqual([reasonCode])
  expect(readiness.allowedActions).toEqual(allowedActions)
}

describe("mcpHubReadiness", () => {
  it("maps an empty hub to setup with an add server action", () => {
    expect(
      getMcpHubReadiness({
        servers: [],
        registryEntries: []
      })
    ).toMatchObject({
      displayState: "needs_setup",
      primaryReasonCode: "not_configured",
      reasonCodes: ["not_configured"],
      allowedActions: ["add_server"]
    })
  })

  it("marks no-auth stdio servers as not requiring credentials", () => {
    const readiness = getMcpServerReadiness({
      server: server({ auth_template_blocked_reason: "no_auth_template" }),
      registryEntries: [registryEntry()]
    })

    expect(readiness.credentialState).toBe("not_required")
    expect(readiness.message).toContain("No credentials required")
  })

  it("uses current operation hints as an in-progress checking override", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: [registryEntry()],
        readinessHint: {
          currentOperation: { operation: "discovery", label: "Refreshing discovery" }
        }
      })
    ).toMatchObject({
      displayState: "checking",
      currentOperation: { operation: "discovery", label: "Refreshing discovery" }
    })
  })

  it("maps preflight failure hints to preflight_failed", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: [registryEntry()],
        readinessHint: { preflightFailed: true }
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "preflight_failed",
      reasonCodes: ["preflight_failed"]
    })
  })

  it("uses the planned action contract for preflight_failed", () => {
    expectSingleReasonActions("preflight_failed", [
      "edit_config",
      "validate",
      "view_details"
    ])
  })

  it("maps missing required credential slots to auth_missing", () => {
    const readiness = getMcpServerReadiness({
      server: server({
        credential_slots: [credentialSlot()]
      }),
      registryEntries: [registryEntry()]
    })

    expect(readiness.credentialState).toBe("required_missing")
    expect(readiness).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "auth_missing",
      reasonCodes: ["auth_missing"]
    })
    expect(readiness.allowedActions).toContain("open_credentials")
  })

  it("uses the planned action contract for auth_missing", () => {
    const readiness = getMcpServerReadiness({
      server: server({
        credential_slots: [credentialSlot()]
      }),
      registryEntries: [registryEntry()]
    })

    expect(readiness.allowedActions).toEqual(["open_credentials", "view_details"])
  })

  it("maps configured credential slots to configured", () => {
    expect(
      getMcpCredentialState(
        server({
          credential_slots: [credentialSlot({ secret_configured: true })]
        })
      )
    ).toBe("configured")
  })

  it("maps valid auth templates to configured", () => {
    expect(
      getMcpCredentialState(
        server({
          auth_template_present: true,
          auth_template_valid: true,
          credential_slots: []
        })
      )
    ).toBe("configured")
  })

  it("maps non-stdio servers without credential signals to unknown", () => {
    expect(
      getMcpCredentialState(
        server({
          transport: "http",
          auth_template_present: false,
          auth_template_valid: false,
          credential_slots: []
        })
      )
    ).toBe("unknown")
  })

  it("maps invalid auth templates to preflight_failed even with matching tools", () => {
    expect(
      getMcpServerReadiness({
        server: server({
          auth_template_present: true,
          auth_template_valid: false,
          credential_slots: []
        }),
        registryEntries: [registryEntry()]
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "preflight_failed",
      reasonCodes: ["preflight_failed"]
    })
  })

  it("maps blocking auth template reasons to preflight_failed", () => {
    expect(
      getMcpServerReadiness({
        server: server({
          auth_template_blocked_reason: "missing_slot_mapping",
          credential_slots: []
        }),
        registryEntries: [registryEntry()]
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "preflight_failed",
      reasonCodes: ["preflight_failed"]
    })
  })

  it("maps unavailable runtimes to runtime_unavailable", () => {
    expect(
      getMcpServerReadiness({
        server: server({ runtime_executable: false }),
        registryEntries: [registryEntry()]
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "runtime_unavailable",
      reasonCodes: ["runtime_unavailable"]
    })
  })

  it("maps unreachable hints to unreachable", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: [registryEntry()],
        readinessHint: { unreachable: true }
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "unreachable",
      reasonCodes: ["unreachable"]
    })
  })

  it("uses the planned action contract for unreachable", () => {
    expectSingleReasonActions("unreachable", [
      "edit_config",
      "refresh_discovery",
      "view_details"
    ])
  })

  it("maps discovery failure hints to discovery_failed", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: [registryEntry()],
        readinessHint: { discoveryFailed: true }
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "discovery_failed",
      reasonCodes: ["discovery_failed"]
    })
  })

  it("treats zero tools without an explicit successful-zero-tools hint as discovery_not_run", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: []
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "discovery_not_run",
      reasonCodes: ["discovery_not_run"]
    })
  })

  it("uses the planned action contract for discovery_not_run", () => {
    expectSingleReasonActions("discovery_not_run", [
      "refresh_discovery",
      "edit_config"
    ])
  })

  it("maps configChanged hints to stale with refresh discovery", () => {
    const readiness = getMcpServerReadiness({
      server: server(),
      registryEntries: [registryEntry()],
      readinessHint: { configChanged: true }
    })

    expect(readiness).toMatchObject({
      displayState: "stale",
      primaryReasonCode: "config_changed",
      reasonCodes: ["config_changed"]
    })
    expect(readiness.allowedActions).toContain("refresh_discovery")
  })

  it("uses the planned action contract for config_changed", () => {
    expectSingleReasonActions("config_changed", [
      "refresh_discovery",
      "edit_config"
    ])
  })

  it("uses the planned action contract for catalog_expired", () => {
    expectSingleReasonActions("catalog_expired", [
      "refresh_discovery",
      "view_details"
    ])
  })

  it("preserves multiple reasons and unions actions without duplicates", () => {
    const readiness = getMcpServerReadiness({
      server: server({
        credential_slots: [credentialSlot()]
      }),
      registryEntries: [registryEntry()],
      readinessHint: { preflightFailed: true, configChanged: true }
    })

    expect(readiness.reasonCodes).toEqual([
      "auth_missing",
      "preflight_failed",
      "config_changed"
    ])
    expect(readiness.primaryReasonCode).toBe("auth_missing")
    expect(readiness.allowedActions).toEqual([
      ...new Set(readiness.allowedActions)
    ])
    expect(readiness.allowedActions).toEqual(
      expect.arrayContaining([
        "open_credentials",
        "view_details",
        "edit_config",
        "validate",
        "refresh_discovery"
      ])
    )
  })

  it("matches registry tools by server module, external module, or external tool prefix", () => {
    for (const entry of [
      registryEntry({ module: "toy-server", tool_name: "search" }),
      registryEntry({ module: "external.toy-server", tool_name: "search" }),
      registryEntry({ module: "other", tool_name: "ext.toy-server.search" })
    ]) {
      expect(
        getMcpServerReadiness({
          server: server(),
          registryEntries: [entry]
        })
      ).toMatchObject({
        displayState: "ready",
        primaryReasonCode: undefined,
        reasonCodes: []
      })
    }
  })

  it("keeps servers ready when the only reason is partial capability", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: [registryEntry()],
        readinessHint: { partialCapability: true }
      })
    ).toMatchObject({
      displayState: "ready",
      primaryReasonCode: "partial_capability",
      reasonCodes: ["partial_capability"]
    })
  })

  it("uses the planned action contract for partial_capability", () => {
    expectSingleReasonActions("partial_capability", [
      "open_tool_catalog",
      "view_details"
    ])
  })

  it("does not add partial_capability when no tools are registered", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: [],
        readinessHint: { partialCapability: true }
      })
    ).toMatchObject({
      displayState: "needs_attention",
      primaryReasonCode: "discovery_not_run",
      reasonCodes: ["discovery_not_run"]
    })
  })

  it("maps legacy server-level secrets without templates or slots to legacy_fallback", () => {
    expect(
      getMcpCredentialState(
        server({
          secret_configured: true,
          auth_template_present: false,
          auth_template_valid: false,
          credential_slots: []
        })
      )
    ).toBe("legacy_fallback")
  })

  it("maps explicit successful zero-tool discovery to no_tools_returned and no_tools", () => {
    expect(
      getMcpServerReadiness({
        server: server(),
        registryEntries: [],
        readinessHint: { discoverySucceededWithNoTools: true }
      })
    ).toMatchObject({
      displayState: "no_tools",
      primaryReasonCode: "no_tools_returned",
      reasonCodes: ["no_tools_returned"]
    })
  })

  it("aggregates hub reasons by priority instead of first row order", () => {
    const staleFirst = server({ id: "stale-first", name: "Stale First" })
    const authSecond = server({
      id: "auth-second",
      name: "Auth Second",
      credential_slots: [credentialSlot({ server_id: "auth-second" })]
    })

    const readiness = getMcpHubReadiness({
      servers: [staleFirst, authSecond],
      registryEntries: [
        registryEntry({
          module: "external.stale-first",
          tool_name: "ext.stale-first.search"
        }),
        registryEntry({
          module: "external.auth-second",
          tool_name: "ext.auth-second.search"
        })
      ],
      readinessHintsByServerId: {
        "stale-first": { configChanged: true }
      }
    })

    expect(readiness.reasonCodes).toEqual(["auth_missing", "config_changed"])
    expect(readiness.primaryReasonCode).toBe("auth_missing")
    expect(readiness.allowedActions).toEqual(
      expect.arrayContaining(["open_credentials", "refresh_discovery"])
    )
  })
})
