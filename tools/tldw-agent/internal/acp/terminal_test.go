package acp

import (
	"testing"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/workspace"
)

func TestMatchAllowlist(t *testing.T) {
	cfg := config.Default()
	session := workspace.NewSession(cfg)
	manager := NewTerminalManager(cfg, session)

	cmd, extra, err := manager.matchAllowlist("python", []string{"-m", "pytest", "-k", "smoke"})
	if err != nil {
		t.Fatalf("expected allowlist match, got error: %v", err)
	}
	if cmd.Template != "python -m pytest" {
		t.Fatalf("unexpected template: %q", cmd.Template)
	}
	if len(extra) != 2 || extra[0] != "-k" || extra[1] != "smoke" {
		t.Fatalf("unexpected extra args: %#v", extra)
	}

	if _, _, err := manager.matchAllowlist("rm", []string{"-rf", "/"}); err == nil {
		t.Fatalf("expected allowlist rejection for rm")
	}

	if _, _, err := manager.matchAllowlist("npm", []string{"install", "leftover"}); err == nil {
		t.Fatalf("expected allowlist rejection for disallowed args")
	}

	excess := []string{"-m", "pytest", "a"}
	for i := 0; i < 21; i++ {
		excess = append(excess, "x")
	}
	if _, _, err := manager.matchAllowlist("python", excess); err == nil {
		t.Fatalf("expected allowlist rejection for too many args")
	}
}

func TestContainsShellMeta(t *testing.T) {
	cases := []struct {
		value string
		want  bool
	}{
		{value: "plain", want: false},
		{value: "arg-with-dash", want: false},
		{value: "semi;colon", want: true},
		{value: "pipe|cmd", want: true},
		{value: "backtick`cmd", want: true},
		{value: "$(sub)", want: true},
		{value: "newline\n", want: true},
	}

	for _, tc := range cases {
		if got := containsShellMeta(tc.value); got != tc.want {
			t.Fatalf("containsShellMeta(%q) = %v, want %v", tc.value, got, tc.want)
		}
	}
}
