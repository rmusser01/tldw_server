package config

import (
	"os"
	"path/filepath"
	"reflect"
	"testing"
)

func TestRegisteredAgentParsesACPEntrypointFields(t *testing.T) {
	path := filepath.Join(t.TempDir(), "config.yaml")
	err := os.WriteFile(path, []byte(`
agents:
  default: codex
  agents:
    - type: codex
      name: Codex
      command: codex
      args: ["--display"]
      entrypoint_strategy: external_acp_adapter
      acp_command: codex-acp
      acp_args: ["--stdio"]
      adapter_source: zed-industries/codex-acp
      adapter_docs_url: https://github.com/zed-industries/codex-acp
      adapter_package: "@zed-industries/codex-acp"
      adapter_version: 0.15.0
      credential_policy: delegated_to_adapter
      runtime_backend: acp_downstream
`), 0644)
	if err != nil {
		t.Fatal(err)
	}

	cfg, err := LoadFrom(path)
	if err != nil {
		t.Fatal(err)
	}
	agent := cfg.Agents.Agents[0]

	if agent.EntrypointStrategy != "external_acp_adapter" {
		t.Fatalf("strategy = %q", agent.EntrypointStrategy)
	}
	if agent.ACPCommand != "codex-acp" {
		t.Fatalf("acp command = %q", agent.ACPCommand)
	}
	if !reflect.DeepEqual(agent.ACPArgs, []string{"--stdio"}) {
		t.Fatalf("acp args = %#v", agent.ACPArgs)
	}
	if agent.AdapterSource != "zed-industries/codex-acp" {
		t.Fatalf("adapter source = %q", agent.AdapterSource)
	}
	if agent.AdapterVersion != "0.15.0" {
		t.Fatalf("adapter version = %q", agent.AdapterVersion)
	}
	if agent.AdapterDocsURL != "https://github.com/zed-industries/codex-acp" {
		t.Fatalf("adapter docs url = %q", agent.AdapterDocsURL)
	}
	if agent.AdapterPackage != "@zed-industries/codex-acp" {
		t.Fatalf("adapter package = %q", agent.AdapterPackage)
	}
	if agent.CredentialPolicy != "delegated_to_adapter" {
		t.Fatalf("credential policy = %q", agent.CredentialPolicy)
	}
	if agent.RuntimeBackend != "acp_downstream" {
		t.Fatalf("runtime backend = %q", agent.RuntimeBackend)
	}
}
