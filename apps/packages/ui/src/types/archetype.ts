export type MCPAuthType = "none" | "bearer" | "api_key"

export type ArchetypeSummary = {
  key: string
  label: string
  tagline: string
  icon: string
}

export type ArchetypePersonaDefaults = {
  name: string
  system_prompt: string | null
  personality_traits: string[]
}

export type ArchetypeMCPConfig = {
  enabled: string[]
  disabled: string[]
}

export type ArchetypeToolOverride = {
  tool: string
  requires_confirmation: boolean
}

export type ArchetypePolicyDefaults = {
  confirmation_mode: "always" | "destructive_only" | "never"
  tool_overrides: ArchetypeToolOverride[]
}

export type ArchetypeBuddyDefaults = {
  species: string | null
  palette: string | null
  silhouette: string | null
}

export type ArchetypeStarterCommand = {
  template_key?: string
  custom?: {
    name: string
    phrases: string[]
    tool_name: string
    slot_map: Record<string, string>
    requires_confirmation: boolean
  }
}

export type ArchetypeTemplate = ArchetypeSummary & {
  persona: ArchetypePersonaDefaults
  mcp_modules: ArchetypeMCPConfig
  suggested_external_servers: string[]
  policy: ArchetypePolicyDefaults
  voice_defaults: Record<string, unknown>
  scope_rules: Record<string, unknown>[]
  buddy: ArchetypeBuddyDefaults
  starter_commands: ArchetypeStarterCommand[]
}

export type ArchetypePreview = {
  name: string
  system_prompt: string | null
  archetype_key: string
  voice_defaults: Record<string, unknown>
  setup: {
    status: "not_started"
    current_step: "archetype"
  }
}

export type MCPCatalogEntry = {
  key: string
  name: string
  description: string
  url_template: string
  auth_type: MCPAuthType
  category: string
  logo_key: string | null
  suggested_for: string[]
}

export type MCPConnectionTestResult = {
  reachable: boolean
  tools_discovered: string[]
  error: string | null
}

export type MCPConnectionDraft = {
  serverKey?: string | null
  name: string
  baseUrl: string
  authType: MCPAuthType
  secret: string
}
