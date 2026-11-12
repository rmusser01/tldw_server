package main

import (
	"os"
	"path/filepath"
	"testing"
)

func TestExplicitRunnerConfig(t *testing.T) {
	home := t.TempDir()
	t.Setenv("HOME", home) // Unit fixture only; no operator/runtime HOME change.
	defaultDir := filepath.Join(home, ".tldw-agent")
	if err := os.MkdirAll(defaultDir, 0700); err != nil {
		t.Fatal(err)
	}
	defaultPath := filepath.Join(defaultDir, "config.yaml")
	original := []byte("agent:\n  command: legacy-host-agent\n")
	if err := os.WriteFile(defaultPath, original, 0600); err != nil {
		t.Fatal(err)
	}
	selected := filepath.Join(t.TempDir(), "isolated.yaml")
	if err := os.WriteFile(selected, []byte("agent:\n  command: isolated-agent\n"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg, err := loadRunnerConfig(selected, true)
	if err != nil || cfg.Agent.Command != "isolated-agent" {
		t.Fatalf("explicit config selection failed: config=%v error=%v", cfg, err)
	}
	cfg, err = loadRunnerConfig("", false)
	if err != nil || cfg.Agent.Command != "legacy-host-agent" {
		t.Fatalf("implicit default changed: config=%v error=%v", cfg, err)
	}
	after, err := os.ReadFile(defaultPath)
	if err != nil || string(after) != string(original) || os.Getenv("HOME") != home {
		t.Fatal("Selecting an explicit configuration modified default configuration or HOME")
	}
}

func TestExplicitRunnerConfigErrorsDoNotUseDefaults(t *testing.T) {
	root := t.TempDir()
	invalid := filepath.Join(root, "invalid.yaml")
	if err := os.WriteFile(invalid, []byte("agent: [invalid"), 0600); err != nil {
		t.Fatal(err)
	}
	for _, path := range []string{"", filepath.Join(root, "missing.yaml"), invalid, root} {
		if _, err := loadRunnerConfig(path, true); err == nil {
			t.Errorf("Explicit unreadable/invalid config silently used defaults: %s", path)
		}
	}
}
