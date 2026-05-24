package acp

import (
	"context"
	"encoding/json"
	"net"
	"os/exec"
	"reflect"
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

	got := expandAgentEnv([]string{
		"HOME=${TLDW_ACP_HOST_HOME}",
		"TERM=xterm-256color",
	})
	want := []string{
		"HOME=/Users/operator",
		"TERM=xterm-256color",
	}

	if !reflect.DeepEqual(got, want) {
		t.Fatalf("expanded env mismatch:\n got: %#v\nwant: %#v", got, want)
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

	var (
		mu           sync.Mutex
		spawnedConns []net.Conn
	)

	runner.SetSpawnFunc(func(agent config.AgentConfig) (*Conn, *exec.Cmd, error) {
		clientConn, serverConn := net.Pipe()

		mu.Lock()
		spawnedConns = append(spawnedConns, clientConn, serverConn)
		mu.Unlock()

		switch agent.Command {
		case "healthy-cmd":
			stubConn := NewConn(serverConn, serverConn)
			_ = newStubAgent(stubConn, "session_health", map[string]interface{}{})
			go func() {
				_ = stubConn.Run()
			}()
		default:
			_ = serverConn.Close()
		}

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

	resp, err := upstream.Call(ctx, "agent/list", map[string]interface{}{})
	if err != nil {
		t.Fatalf("agent/list failed: %v", err)
	}

	var payload struct {
		Agents []struct {
			Type         string `json:"type"`
			IsConfigured bool   `json:"isConfigured"`
		} `json:"agents"`
	}
	if err := json.Unmarshal(resp.Result, &payload); err != nil {
		t.Fatalf("invalid agent/list result: %v", err)
	}

	byType := map[string]bool{}
	for _, agent := range payload.Agents {
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
