// Package config handles loading and managing tldw-agent configuration.
package config

import (
	"os"
	"path/filepath"
	"runtime"

	"gopkg.in/yaml.v3"
)

// Config holds all configuration for the tldw-agent.
type Config struct {
	Server    ServerConfig    `yaml:"server"`
	Workspace WorkspaceConfig `yaml:"workspace"`
	Execution ExecutionConfig `yaml:"execution"`
	Security  SecurityConfig  `yaml:"security"`
	Agent     AgentConfig     `yaml:"agent"`
	Agents    AgentRegistry   `yaml:"agents"`
}

// ServerConfig holds LLM server connection settings.
type ServerConfig struct {
	LLMEndpoint string `yaml:"llm_endpoint"`
	APIKey      string `yaml:"api_key"`
}

// AgentConfig holds downstream ACP agent launch settings.
type AgentConfig struct {
	Command string   `yaml:"command"`
	Args    []string `yaml:"args"`
	Env     []string `yaml:"env"`
}

// AgentRegistry holds globally configured downstream agents.
type AgentRegistry struct {
	Default string            `yaml:"default"`
	Agents  []RegisteredAgent `yaml:"agents"`
}

// RegisteredAgent defines a configured ACP downstream agent.
type RegisteredAgent struct {
	Type               string   `yaml:"type"`
	Name               string   `yaml:"name"`
	Description        string   `yaml:"description"`
	Command            string   `yaml:"command"`
	Args               []string `yaml:"args"`
	Env                []string `yaml:"env"`
	RequiresAPIKey     string   `yaml:"requires_api_key"`
	EntrypointStrategy string   `yaml:"entrypoint_strategy"`
	ACPCommand         string   `yaml:"acp_command"`
	ACPArgs            []string `yaml:"acp_args"`
	AdapterSource      string   `yaml:"adapter_source"`
	AdapterDocsURL     string   `yaml:"adapter_docs_url"`
	AdapterPackage     string   `yaml:"adapter_package"`
	AdapterVersion     string   `yaml:"adapter_version"`
	CredentialPolicy   string   `yaml:"credential_policy"`
	RuntimeBackend     string   `yaml:"runtime_backend"`
}

// WorkspaceConfig holds workspace-related settings.
type WorkspaceConfig struct {
	DefaultRoot      string   `yaml:"default_root"`
	BlockedPaths     []string `yaml:"blocked_paths"`
	MaxFileSizeBytes int64    `yaml:"max_file_size_bytes"`
}

// CustomCommand represents a user-defined allowlisted command.
type CustomCommand struct {
	ID          string   `yaml:"id"`
	Template    string   `yaml:"template"`
	Description string   `yaml:"description"`
	Category    string   `yaml:"category"`
	AllowArgs   bool     `yaml:"allow_args"`
	MaxArgs     int      `yaml:"max_args"`
	Env         []string `yaml:"env,omitempty"`
}

// ExecutionConfig holds command execution settings.
type ExecutionConfig struct {
	Enabled        bool            `yaml:"enabled"`
	TimeoutMs      int             `yaml:"timeout_ms"`
	Shell          string          `yaml:"shell"`
	NetworkAllowed bool            `yaml:"network_allowed"`
	MaxOutputBytes int             `yaml:"max_output_bytes"`
	CustomCommands []CustomCommand `yaml:"custom_commands"`
}

// SecurityConfig holds security-related settings.
type SecurityConfig struct {
	RequireApprovalForWrites bool `yaml:"require_approval_for_writes"`
	RequireApprovalForExec   bool `yaml:"require_approval_for_exec"`
	RedactSecrets            bool `yaml:"redact_secrets"`
}

// Default returns a Config with sensible defaults.
func Default() *Config {
	return &Config{
		Server: ServerConfig{
			LLMEndpoint: "http://localhost:8000",
			APIKey:      "",
		},
		Workspace: WorkspaceConfig{
			DefaultRoot: "",
			BlockedPaths: []string{
				".env",
				"*.pem",
				"*.key",
				"**/node_modules/**",
				"**/.git/objects/**",
			},
			MaxFileSizeBytes: 10 * 1024 * 1024, // 10MB
		},
		Execution: ExecutionConfig{
			Enabled:        true,
			TimeoutMs:      30000,
			Shell:          "auto",
			NetworkAllowed: false,
			MaxOutputBytes: 1024 * 1024, // 1MB
			CustomCommands: []CustomCommand{},
		},
		Security: SecurityConfig{
			RequireApprovalForWrites: true,
			RequireApprovalForExec:   true,
			RedactSecrets:            true,
		},
		Agent: AgentConfig{
			Command: "",
			Args:    []string{},
			Env:     []string{},
		},
		Agents: AgentRegistry{
			Default: "",
			Agents:  []RegisteredAgent{},
		},
	}
}

// ConfigPath returns the path to the config file.
func ConfigPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return ""
	}
	return filepath.Join(home, ".tldw-agent", "config.yaml")
}

// Load reads configuration from the default config file.
func Load() (*Config, error) {
	path := ConfigPath()
	return LoadFrom(path)
}

// LoadFrom reads configuration from a specific file path.
func LoadFrom(path string) (*Config, error) {
	cfg := Default()

	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return cfg, nil // Return defaults if file doesn't exist
		}
		return nil, err
	}

	if err := yaml.Unmarshal(data, cfg); err != nil {
		return nil, err
	}

	return cfg, nil
}

// Save writes the configuration to the default config file.
func (c *Config) Save() error {
	path := ConfigPath()
	return c.SaveTo(path)
}

// SaveTo writes the configuration to a specific file path.
func (c *Config) SaveTo(path string) error {
	// Ensure directory exists
	dir := filepath.Dir(path)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return err
	}

	data, err := yaml.Marshal(c)
	if err != nil {
		return err
	}

	return os.WriteFile(path, data, 0644)
}

// GetShell returns the shell to use for command execution.
func (c *Config) GetShell() string {
	if c.Execution.Shell != "auto" {
		return c.Execution.Shell
	}

	if runtime.GOOS == "windows" {
		return "powershell"
	}
	return "bash"
}

// IsPathBlocked checks if a path matches any of the blocked patterns.
func (c *Config) IsPathBlocked(path string) bool {
	for _, pattern := range c.Workspace.BlockedPaths {
		matched, err := filepath.Match(pattern, filepath.Base(path))
		if err == nil && matched {
			return true
		}
		// Also check the full path for glob patterns
		matched, err = filepath.Match(pattern, path)
		if err == nil && matched {
			return true
		}
	}
	return false
}
