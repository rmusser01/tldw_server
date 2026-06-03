package acp

import (
	"context"
	"encoding/json"
	"net"
	"os/exec"
	"reflect"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/tldw/tldw-agent/internal/config"
)

type stubAgent struct {
	conn         *Conn
	sessionID    string
	caps         map[string]interface{}
	sessionNewCh chan map[string]interface{}
	promptCh     chan promptParams
}

type promptParams struct {
	SessionID string                   `json:"sessionId"`
	Prompt    []map[string]interface{} `json:"prompt"`
}

func TestExpandAgentEnvResolvesHostPlaceholders(t *testing.T) {
	t.Setenv("TLDW_ACP_HOST_HOME", "/Users/operator")
	t.Setenv("TOKEN", "secret")

	got := expandAgentEnv([]string{
		"HOME=${TLDW_ACP_HOST_HOME}",
		"LITERAL=$TLDW_ACP_HOST_HOME",
		"TOKEN_${TOKEN}=value_${TOKEN}",
		"NO_EQUALS_${TOKEN}",
		"MISSING=${UNSET_PLACEHOLDER}",
		"TERM=xterm-256color",
	})
	want := []string{
		"HOME=/Users/operator",
		"LITERAL=$TLDW_ACP_HOST_HOME",
		"TOKEN_${TOKEN}=value_secret",
		"NO_EQUALS_${TOKEN}",
		"MISSING=",
		"TERM=xterm-256color",
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expanded env mismatch:\n got: %#v\nwant: %#v", got, want)
	}
}

func TestRunnerDefaultAgentSkipsEmptyConfiguredDefault(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "custom"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type:    "custom",
			Name:    "Custom",
			Command: "",
		},
		{
			Type:    "goose",
			Name:    "Goose",
			Command: "goose",
			Args:    []string{"acp"},
		},
	}
	runner := NewRunner(cfg)

	entry, err := runner.resolveAgentEntry("")
	if err != nil {
		t.Fatalf("default agent resolution failed: %v", err)
	}
	if entry.Type != "goose" {
		t.Fatalf("default agent type = %q, want goose", entry.Type)
	}
	if entry.Command != "goose" {
		t.Fatalf("default agent command = %q, want goose", entry.Command)
	}

	explicit, err := runner.resolveAgentEntry("custom")
	if err != nil {
		t.Fatalf("explicit custom agent resolution failed: %v", err)
	}
	if explicit.Type != "custom" || explicit.Command != "" {
		t.Fatalf("explicit custom should not fall back to goose: %#v", explicit)
	}
}

func TestRunnerLaunchesACPCommandForExternalAdapter(t *testing.T) {
	entry := config.RegisteredAgent{
		Type:               "codex",
		Command:            "codex",
		Args:               []string{"--display"},
		EntrypointStrategy: "external_acp_adapter",
		ACPCommand:         "codex-acp",
		ACPArgs:            []string{"--stdio"},
	}

	agentCfg, err := resolveLaunchAgentConfig(entry)
	if err != nil {
		t.Fatalf("resolve failed: %v", err)
	}
	if agentCfg.Command != "codex-acp" {
		t.Fatalf("command = %q, want codex-acp", agentCfg.Command)
	}
	if !reflect.DeepEqual(agentCfg.Args, []string{"--stdio"}) {
		t.Fatalf("args = %#v", agentCfg.Args)
	}
}

func TestRunnerDoesNotFallbackExternalAdapterToDisplayCommand(t *testing.T) {
	entry := config.RegisteredAgent{
		Type:               "codex",
		Command:            "codex",
		EntrypointStrategy: "external_acp_adapter",
		ACPCommand:         "",
	}

	_, err := resolveLaunchAgentConfig(entry)
	if err == nil || !strings.Contains(err.Error(), "acp_command is required") {
		t.Fatalf("expected missing acp command error, got %v", err)
	}
}

func TestRunnerLegacyNativeACPFallsBackToCommand(t *testing.T) {
	entry := config.RegisteredAgent{
		Type:               "goose",
		Command:            "goose",
		Args:               []string{"acp"},
		EntrypointStrategy: "native_acp",
		ACPCommand:         "",
	}

	agentCfg, err := resolveLaunchAgentConfig(entry)
	if err != nil {
		t.Fatalf("resolve failed: %v", err)
	}
	if agentCfg.Command != "goose" {
		t.Fatalf("command = %q", agentCfg.Command)
	}
	if !reflect.DeepEqual(agentCfg.Args, []string{"acp"}) {
		t.Fatalf("args = %#v", agentCfg.Args)
	}
}

func TestRunnerInitializeDoesNotSpawnDownstreamForPassiveCapabilities(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "codex"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type:               "codex",
			Name:               "Codex",
			Command:            "codex",
			EntrypointStrategy: "external_acp_adapter",
			ACPCommand:         "codex-acp",
		},
	}
	runner := NewRunner(cfg)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		t.Fatalf("initialize must not spawn downstream agents for passive capabilities")
		return nil, nil, nil
	})

	resp := callRunnerInitialize(t, runner)
	if resp.AgentCapabilities == nil {
		t.Fatalf("initialize should return default capability envelope")
	}
}

func TestRunnerAgentListUsesPassiveReadinessWithoutSpawning(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "codex"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type:               "codex",
			Name:               "Codex",
			Command:            "codex",
			EntrypointStrategy: "external_acp_adapter",
			ACPCommand:         "codex-acp",
			AdapterDocsURL:     "https://github.com/zed-industries/codex-acp",
			CredentialPolicy:   "delegated_to_adapter",
		},
	}
	runner := NewRunner(cfg)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		t.Fatalf("agent/list must not spawn or initialize downstream agents")
		return nil, nil, nil
	})
	runner.SetLookPathFunc(func(command string) (string, error) {
		switch command {
		case "codex", "codex-acp":
			return "/usr/bin/" + command, nil
		default:
			return "", exec.ErrNotFound
		}
	})

	resp := callRunnerAgentList(t, runner)
	agent := findAgentListItem(t, resp, "codex")

	if !agent.IsConfigured {
		t.Fatalf("codex should be passively configured when display and adapter commands resolve")
	}
	if agent.ProbeState != "ready_to_probe" {
		t.Fatalf("probe state = %q", agent.ProbeState)
	}
	if agent.DisplayCommand != "codex" || !agent.DisplayBinaryFound || !agent.AdapterFound {
		t.Fatalf("unexpected readiness metadata: %#v", agent)
	}
	if agent.CredentialState != "delegated" {
		t.Fatalf("credential state = %q", agent.CredentialState)
	}
	if agent.AdapterDocsURL != "https://github.com/zed-industries/codex-acp" {
		t.Fatalf("adapter docs url = %q", agent.AdapterDocsURL)
	}
}

func TestRunnerAgentListBlocksMutableNpxLatestWithoutSpawning(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "codex"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type:               "codex",
			Name:               "Codex",
			Command:            "codex",
			EntrypointStrategy: "external_acp_adapter",
			ACPCommand:         "npx",
			ACPArgs:            []string{"@zed-industries/codex-acp@latest"},
		},
	}
	runner := NewRunner(cfg)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		t.Fatalf("agent/list must not execute npx")
		return nil, nil, nil
	})
	runner.SetLookPathFunc(func(command string) (string, error) {
		switch command {
		case "codex", "npx":
			return "/usr/bin/" + command, nil
		default:
			return "", exec.ErrNotFound
		}
	})

	resp := callRunnerAgentList(t, runner)
	agent := findAgentListItem(t, resp, "codex")

	if agent.IsConfigured {
		t.Fatalf("mutable npx @latest adapter invocation must not be passively configured")
	}
	if agent.PrimaryBlocker != "mutable_adapter_invocation" {
		t.Fatalf("primary blocker = %q", agent.PrimaryBlocker)
	}
}

func TestPassiveBlockedStatusUsesActionableMessages(t *testing.T) {
	cases := map[string]string{
		"live_certification_required": "Run live ACP certification before claiming this agent is supported.",
		"entrypoint_strategy_missing": "Identify and configure a concrete ACP stdio entrypoint before live certification.",
	}

	for blocker, want := range cases {
		t.Run(blocker, func(t *testing.T) {
			if got := passiveBlockedStatus(blocker); got != want {
				t.Fatalf("status = %q, want %q", got, want)
			}
		})
	}
}

func newStubAgent(conn *Conn, sessionID string, caps map[string]interface{}) *stubAgent {
	agent := &stubAgent{
		conn:         conn,
		sessionID:    sessionID,
		caps:         caps,
		sessionNewCh: make(chan map[string]interface{}, 1),
		promptCh:     make(chan promptParams, 1),
	}

	conn.SetHandler(func(msg *RPCMessage) (*RPCResponse, error) {
		switch msg.Method {
		case "initialize":
			result := map[string]interface{}{
				"protocolVersion":   defaultProtocolVersion,
				"agentCapabilities": caps,
			}
			return NewResultResponse(msg.ID, result), nil
		case "session/new":
			var params map[string]interface{}
			if err := json.Unmarshal(msg.Params, &params); err == nil {
				select {
				case agent.sessionNewCh <- params:
				default:
				}
			}
			return NewResultResponse(msg.ID, map[string]interface{}{
				"sessionId": agent.sessionID,
			}), nil
		case "session/prompt":
			var params promptParams
			if err := json.Unmarshal(msg.Params, &params); err == nil {
				agent.promptCh <- params
			}
			_ = conn.Notify("session/update", map[string]interface{}{
				"sessionId": agent.sessionID,
				"event":     "message",
				"content":   "ok",
			})
			return NewResultResponse(msg.ID, map[string]interface{}{
				"stopReason": "end",
			}), nil
		default:
			return NewErrorResponse(msg.ID, ErrMethodNotFound, "method not found"), nil
		}
	})

	return agent
}

func TestRunnerSessionRoutingAndUpdates(t *testing.T) {
	cfg := config.Default()
	cfg.Agent.Command = "stub-agent"
	runner := NewRunner(cfg)

	caps := map[string]interface{}{
		"promptCapabilities": map[string]bool{
			"image":           true,
			"audio":           false,
			"embeddedContext": false,
		},
		"mcpCapabilities": map[string]bool{
			"http": true,
			"sse":  false,
		},
		"sessionCapabilities": map[string]interface{}{
			"cancel": true,
		},
	}

	var (
		mu                sync.Mutex
		stubAgentInstance *stubAgent
		spawnedConns      []net.Conn
	)

	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		clientConn, serverConn := net.Pipe()

		stubConn := NewConn(serverConn, serverConn)
		mu.Lock()
		spawnedConns = append(spawnedConns, clientConn, serverConn)
		stubAgentInstance = newStubAgent(stubConn, "session_stub", caps)
		mu.Unlock()
		go func() {
			_ = stubConn.Run()
		}()

		return NewConn(clientConn, clientConn), nil, nil
	})

	upstreamConn, runnerConn := net.Pipe()
	upstream := NewConn(upstreamConn, upstreamConn)
	updateCh := make(chan *RPCMessage, 1)
	upstream.SetNotificationHandler(func(msg *RPCMessage) {
		if msg.Method == "session/update" {
			updateCh <- msg
		}
	})

	go func() {
		_ = upstream.Run()
	}()

	runErr := make(chan error, 1)
	go func() {
		runErr <- runner.Run(runnerConn, runnerConn)
	}()

	t.Cleanup(func() {
		_ = upstreamConn.Close()
		_ = runnerConn.Close()
		mu.Lock()
		conns := append([]net.Conn(nil), spawnedConns...)
		mu.Unlock()
		for _, conn := range conns {
			_ = conn.Close()
		}
		select {
		case <-runErr:
		case <-time.After(time.Second):
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	newResp, err := upstream.Call(ctx, "session/new", map[string]interface{}{
		"cwd": t.TempDir(),
	})
	if err != nil {
		t.Fatalf("session/new failed: %v", err)
	}

	sessionID := extractSessionID(t, newResp.Result)
	if sessionID != "session_stub" {
		t.Fatalf("unexpected session id: %q", sessionID)
	}

	_, err = upstream.Call(ctx, "session/prompt", map[string]interface{}{
		"sessionId": sessionID,
		"prompt": []map[string]interface{}{
			{"role": "user", "content": "hello"},
		},
	})
	if err != nil {
		t.Fatalf("session/prompt failed: %v", err)
	}

	mu.Lock()
	instance := stubAgentInstance
	mu.Unlock()
	if instance == nil {
		t.Fatalf("stub agent was not spawned")
	}

	select {
	case params := <-instance.promptCh:
		if params.SessionID != sessionID {
			t.Fatalf("prompt forwarded with session %q, want %q", params.SessionID, sessionID)
		}
		if len(params.Prompt) != 1 {
			t.Fatalf("prompt forwarded with %d entries", len(params.Prompt))
		}
	case <-time.After(time.Second):
		t.Fatalf("prompt was not forwarded to downstream")
	}

	select {
	case msg := <-updateCh:
		var update map[string]interface{}
		if err := json.Unmarshal(msg.Params, &update); err != nil {
			t.Fatalf("failed to unmarshal update: %v", err)
		}
		if update["sessionId"] != sessionID {
			t.Fatalf("update session mismatch: %#v", update)
		}
	case <-time.After(time.Second):
		t.Fatalf("session/update not forwarded upstream")
	}
}

func TestRunnerStripsAgentTypeBeforeForwardingSessionNew(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "hermes"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type:    "hermes",
			Name:    "Hermes",
			Command: "hermes",
		},
	}
	runner := NewRunner(cfg)

	var (
		mu                sync.Mutex
		stubAgentInstance *stubAgent
		spawnedConns      []net.Conn
	)

	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		clientConn, serverConn := net.Pipe()

		stubConn := NewConn(serverConn, serverConn)
		mu.Lock()
		spawnedConns = append(spawnedConns, clientConn, serverConn)
		stubAgentInstance = newStubAgent(stubConn, "session_hermes", map[string]interface{}{})
		mu.Unlock()
		go func() {
			_ = stubConn.Run()
		}()

		return NewConn(clientConn, clientConn), nil, nil
	})

	upstreamConn, runnerConn := net.Pipe()
	upstream := NewConn(upstreamConn, upstreamConn)
	go func() {
		_ = upstream.Run()
	}()

	runErr := make(chan error, 1)
	go func() {
		runErr <- runner.Run(runnerConn, runnerConn)
	}()

	t.Cleanup(func() {
		_ = upstreamConn.Close()
		_ = runnerConn.Close()
		mu.Lock()
		conns := append([]net.Conn(nil), spawnedConns...)
		mu.Unlock()
		for _, conn := range conns {
			_ = conn.Close()
		}
		select {
		case <-runErr:
		case <-time.After(time.Second):
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := upstream.Call(ctx, "session/new", map[string]interface{}{
		"agentType":  "hermes",
		"cwd":        t.TempDir(),
		"mcpServers": []interface{}{},
	})
	if err != nil {
		t.Fatalf("session/new failed: %v", err)
	}

	mu.Lock()
	instance := stubAgentInstance
	mu.Unlock()
	if instance == nil {
		t.Fatalf("stub agent was not spawned")
	}

	select {
	case params := <-instance.sessionNewCh:
		if _, ok := params["agentType"]; ok {
			t.Fatalf("agentType should not be forwarded to downstream session/new: %#v", params)
		}
		if params["cwd"] == "" {
			t.Fatalf("cwd should be preserved for downstream session/new: %#v", params)
		}
		if _, ok := params["mcpServers"]; !ok {
			t.Fatalf("mcpServers should be preserved for downstream session/new: %#v", params)
		}
	case <-time.After(time.Second):
		t.Fatalf("session/new was not forwarded to downstream")
	}
}

func TestRunnerInitializeReflectsDownstreamCapabilities(t *testing.T) {
	cfg := config.Default()
	cfg.Agent.Command = "stub-agent"
	runner := NewRunner(cfg)

	caps := map[string]interface{}{
		"promptCapabilities": map[string]bool{
			"image":           true,
			"audio":           true,
			"embeddedContext": true,
		},
		"mcpCapabilities": map[string]bool{
			"http": true,
			"sse":  true,
		},
		"sessionCapabilities": map[string]interface{}{
			"cancel": true,
		},
	}

	var (
		mu           sync.Mutex
		spawnedConns []net.Conn
	)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		clientConn, serverConn := net.Pipe()

		stubConn := NewConn(serverConn, serverConn)
		stubAgent := newStubAgent(stubConn, "session_caps", caps)
		_ = stubAgent
		mu.Lock()
		spawnedConns = append(spawnedConns, clientConn, serverConn)
		mu.Unlock()
		go func() {
			_ = stubConn.Run()
		}()

		return NewConn(clientConn, clientConn), nil, nil
	})

	upstreamConn, runnerConn := net.Pipe()
	upstream := NewConn(upstreamConn, upstreamConn)
	go func() {
		_ = upstream.Run()
	}()

	runErr := make(chan error, 1)
	go func() {
		runErr <- runner.Run(runnerConn, runnerConn)
	}()

	t.Cleanup(func() {
		_ = upstreamConn.Close()
		_ = runnerConn.Close()
		mu.Lock()
		conns := append([]net.Conn(nil), spawnedConns...)
		mu.Unlock()
		for _, conn := range conns {
			_ = conn.Close()
		}
		select {
		case <-runErr:
		case <-time.After(time.Second):
		}
	})

	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()

	_, err := upstream.Call(ctx, "session/new", map[string]interface{}{
		"cwd": t.TempDir(),
	})
	if err != nil {
		t.Fatalf("session/new failed: %v", err)
	}

	initResp, err := upstream.Call(ctx, "initialize", map[string]interface{}{
		"protocolVersion": defaultProtocolVersion,
	})
	if err != nil {
		t.Fatalf("initialize failed: %v", err)
	}

	var initResult map[string]interface{}
	if err := json.Unmarshal(initResp.Result, &initResult); err != nil {
		t.Fatalf("invalid initialize result: %v", err)
	}

	rawCaps, ok := initResult["agentCapabilities"].(map[string]interface{})
	if !ok {
		t.Fatalf("missing agentCapabilities in initialize result")
	}

	promptCaps, ok := rawCaps["promptCapabilities"].(map[string]interface{})
	if !ok || promptCaps["image"] != true || promptCaps["audio"] != true {
		t.Fatalf("prompt capabilities not reflected: %#v", promptCaps)
	}

	mcpCaps, ok := rawCaps["mcpCapabilities"].(map[string]interface{})
	if !ok || mcpCaps["http"] != true || mcpCaps["sse"] != true {
		t.Fatalf("mcp capabilities not reflected: %#v", mcpCaps)
	}
}

func extractSessionID(t *testing.T, raw json.RawMessage) string {
	t.Helper()
	var payload struct {
		SessionID string `json:"sessionId"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("invalid session/new result: %v", err)
	}
	if payload.SessionID == "" {
		t.Fatalf("missing sessionId in result")
	}
	return payload.SessionID
}

type runnerInitializeResult struct {
	AgentCapabilities map[string]interface{} `json:"agentCapabilities"`
}

func callRunnerInitialize(t *testing.T, runner *Runner) runnerInitializeResult {
	t.Helper()
	resp, err := runner.handleInitialize(&RPCMessage{
		JSONRPC: JSONRPCVersion,
		ID:      json.RawMessage(`1`),
		Method:  "initialize",
	})
	if err != nil {
		t.Fatalf("initialize returned error: %v", err)
	}
	if resp == nil || resp.Error != nil {
		t.Fatalf("initialize returned RPC error: %#v", resp)
	}
	var result runnerInitializeResult
	if err := unmarshalRPCResult(resp.Result, &result); err != nil {
		t.Fatalf("invalid initialize result: %v", err)
	}
	return result
}

type runnerAgentListItem struct {
	Type               string   `json:"type"`
	IsConfigured       bool     `json:"isConfigured"`
	EntrypointStrategy string   `json:"entrypoint_strategy"`
	ACPCommand         string   `json:"acp_command"`
	ACPArgs            []string `json:"acp_args"`
	AdapterFound       bool     `json:"adapter_found"`
	AdapterDocsURL     string   `json:"adapter_docs_url"`
	DisplayCommand     string   `json:"display_command"`
	DisplayBinaryFound bool     `json:"display_binary_found"`
	CredentialState    string   `json:"credential_state"`
	ProbeState         string   `json:"probe_state"`
	PrimaryBlocker     string   `json:"primary_blocker"`
	Blockers           []string `json:"blockers"`
}

type runnerAgentListResult struct {
	Agents []runnerAgentListItem `json:"agents"`
}

func callRunnerAgentList(t *testing.T, runner *Runner) runnerAgentListResult {
	t.Helper()
	resp, err := runner.handleAgentList(&RPCMessage{
		JSONRPC: JSONRPCVersion,
		ID:      json.RawMessage(`1`),
		Method:  "agent/list",
	})
	if err != nil {
		t.Fatalf("agent/list returned error: %v", err)
	}
	if resp == nil || resp.Error != nil {
		t.Fatalf("agent/list returned RPC error: %#v", resp)
	}
	var result runnerAgentListResult
	if err := unmarshalRPCResult(resp.Result, &result); err != nil {
		t.Fatalf("invalid agent/list result: %v", err)
	}
	return result
}

func unmarshalRPCResult(raw interface{}, target interface{}) error {
	data, err := json.Marshal(raw)
	if err != nil {
		return err
	}
	return json.Unmarshal(data, target)
}

func findAgentListItem(t *testing.T, result runnerAgentListResult, agentType string) runnerAgentListItem {
	t.Helper()
	for _, agent := range result.Agents {
		if agent.Type == agentType {
			return agent
		}
	}
	t.Fatalf("agent %q not found in %#v", agentType, result.Agents)
	return runnerAgentListItem{}
}

func TestRunnerAgentListReportsOnlyReadyAgentsAsConfigured(t *testing.T) {
	cfg := config.Default()
	cfg.Agents.Default = "healthy"
	cfg.Agents.Agents = []config.RegisteredAgent{
		{
			Type:        "healthy",
			Name:        "Healthy",
			Description: "healthy downstream agent",
			Command:     "healthy-cmd",
		},
		{
			Type:        "broken",
			Name:        "Broken",
			Description: "non-ready downstream agent",
			Command:     "broken-cmd",
		},
		{
			Type:        "empty",
			Name:        "Empty",
			Description: "missing command",
			Command:     "",
		},
	}
	runner := NewRunner(cfg)
	runner.SetSpawnFunc(func(_ config.AgentConfig) (*Conn, *exec.Cmd, error) {
		t.Fatalf("agent/list must use passive readiness and not spawn downstream agents")
		return nil, nil, nil
	})
	runner.SetLookPathFunc(func(command string) (string, error) {
		switch command {
		case "healthy-cmd":
			return "/usr/bin/" + command, nil
		default:
			return "", exec.ErrNotFound
		}
	})

	result := callRunnerAgentList(t, runner)
	byType := map[string]bool{}
	for _, agent := range result.Agents {
		byType[agent.Type] = agent.IsConfigured
	}

	if !byType["healthy"] {
		t.Fatalf("healthy agent should be marked configured")
	}
	if byType["broken"] {
		t.Fatalf("broken agent should be marked not configured")
	}
	if byType["empty"] {
		t.Fatalf("empty-command agent should be marked not configured")
	}
}
