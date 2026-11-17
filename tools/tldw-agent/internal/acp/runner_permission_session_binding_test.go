package acp

import (
	"context"
	"encoding/json"
	"fmt"
	"net"
	"os/exec"
	"sync"
	"testing"
	"time"

	"github.com/tldw/tldw-agent/internal/config"
)

type permissionBindingWire struct {
	runner      *Runner
	client      *Conn
	children    []*Conn
	permissions chan *RPCMessage
	updates     chan *RPCMessage
}

func newPermissionBindingWire(t *testing.T, earlyUpdate bool) *permissionBindingWire {
	t.Helper()
	cfg := config.Default()
	cfg.Agent.Command = "controlled-binding-agent"
	runner := NewRunner(cfg)
	wire := &permissionBindingWire{runner: runner, permissions: make(chan *RPCMessage, 16), updates: make(chan *RPCMessage, 16)}
	a, b := net.Pipe()
	wire.client = NewConn(a, a)
	wire.client.SetHandler(func(msg *RPCMessage) (*RPCResponse, error) {
		wire.permissions <- msg
		return NewResultResponse(msg.ID, map[string]interface{}{"outcome": map[string]interface{}{"outcome": "approved"}}), nil
	})
	wire.client.SetNotificationHandler(func(msg *RPCMessage) { wire.updates <- msg })
	var mu sync.Mutex
	connections := []net.Conn{a, b}
	done := make(chan error, 4)
	runner.SetSpawnFunc(func(config.AgentConfig) (*Conn, *exec.Cmd, error) {
		c, d := net.Pipe()
		child := NewConn(d, d)
		mu.Lock()
		index := len(wire.children)
		wire.children = append(wire.children, child)
		connections = append(connections, c, d)
		mu.Unlock()
		id := fmt.Sprintf("binding-session-%d", index)
		child.SetHandler(func(msg *RPCMessage) (*RPCResponse, error) {
			switch msg.Method {
			case "initialize":
				if earlyUpdate && index == 0 {
					if err := child.Notify("session/update", map[string]interface{}{"update": map[string]interface{}{"sessionUpdate": "agent_message_chunk", "content": map[string]string{"type": "text", "text": "UNREGISTERED"}}}); err != nil {
						return nil, err
					}
				}
				return NewResultResponse(msg.ID, map[string]interface{}{"protocolVersion": 1}), nil
			case "session/new":
				return NewResultResponse(msg.ID, map[string]string{"sessionId": id}), nil
			}
			return NewErrorResponse(msg.ID, ErrMethodNotFound, "method not found"), nil
		})
		go func() { done <- child.Run() }()
		return NewConn(c, c), nil, nil
	})
	go func() { done <- wire.client.Run() }()
	go func() { done <- runner.Run(b, b) }()
	t.Cleanup(func() {
		mu.Lock()
		conns := append([]net.Conn(nil), connections...)
		mu.Unlock()
		for _, conn := range conns {
			_ = conn.Close()
		}
		for i := 0; i < 4; i++ {
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("registered child wire did not stop")
			}
		}
	})
	for index := 0; index < 2; index++ {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		response, err := wire.client.Call(ctx, "session/new", map[string]interface{}{"cwd": t.TempDir()})
		cancel()
		if err != nil || response == nil || response.Error != nil {
			t.Fatalf("normal registered child creation failed: %+v %v", response, err)
		}
		if got := extractSessionID(t, response.Result); got != fmt.Sprintf("binding-session-%d", index) {
			t.Fatalf("unexpected registered child identity %q", got)
		}
	}
	return wire
}

func TestRunnerPermissionRejectsForeignOrInvalidClaimOnTwoChildWire(t *testing.T) {
	cases := []struct {
		name   string
		params json.RawMessage
	}{
		{"foreign_registered_child", json.RawMessage(`{"sessionId":"binding-session-1","tool":{"name":"controlled/read"}}`)},
		{"missing", json.RawMessage(`{"tool":{"name":"controlled/read"}}`)},
		{"empty", json.RawMessage(`{"sessionId":"","tool":{"name":"controlled/read"}}`)},
		{"null", json.RawMessage(`{"sessionId":null,"tool":{"name":"controlled/read"}}`)},
		{"number", json.RawMessage(`{"sessionId":7,"tool":{"name":"controlled/read"}}`)},
		{"array_params", json.RawMessage(`[]`)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			wire := newPermissionBindingWire(t, false)
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			response, err := wire.children[0].CallRaw(ctx, "session/request_permission", tc.params)
			if err != nil || response == nil || response.Error == nil || response.Error.Code != ErrInvalidParams {
				t.Fatalf("invalid child claim was not refused: %+v %v", response, err)
			}
			select {
			case request := <-wire.permissions:
				t.Fatalf("invalid child claim reached policy/UI authority: %s", request.Params)
			default:
			}
		})
	}
}

func TestRunnerPermissionPreservesSameSessionRawClaimOnTwoChildWire(t *testing.T) {
	wire := newPermissionBindingWire(t, false)
	for index, child := range wire.children {
		params := json.RawMessage(fmt.Sprintf(`{"sessionId":"binding-session-%d","tool":{"name":"controlled/read","input":{"literal":"EXACT_AUTHORED_VALUE"}},"opaque":{"keep":[0,false]}}`, index))
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		response, err := child.CallRaw(ctx, "session/request_permission", params)
		cancel()
		if err != nil || response == nil || response.Error != nil {
			t.Fatalf("legitimate child claim failed: %+v %v", response, err)
		}
		var result struct {
			Outcome struct {
				Outcome string `json:"outcome"`
			} `json:"outcome"`
		}
		if err := json.Unmarshal(response.Result, &result); err != nil || result.Outcome.Outcome != "approved" {
			t.Fatalf("legitimate approval changed: %s %v", response.Result, err)
		}
		select {
		case request := <-wire.permissions:
			if request.Method != "session/request_permission" || string(request.Params) != string(params) {
				t.Fatalf("legitimate raw permission changed: method=%s params=%s", request.Method, request.Params)
			}
		default:
			t.Fatal("legitimate registered child did not reach policy authority")
		}
	}
}

func TestRunnerUpdateRequiresRegisteredSourceOnTwoChildWire(t *testing.T) {
	wire := newPermissionBindingWire(t, true)
	select {
	case update := <-wire.updates:
		t.Fatalf("unregistered empty identity reached session update consumer: %s", update.Params)
	default:
	}
	for index, child := range wire.children {
		params := json.RawMessage(fmt.Sprintf(`{"sessionId":"binding-session-%d","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"REGISTERED_CONTENT"}},"opaque":{"keep":true}}`, index))
		if err := child.NotifyRaw("session/update", params); err != nil {
			t.Fatal(err)
		}
		select {
		case update := <-wire.updates:
			if update.Method != "session/update" || string(update.Params) != string(params) {
				t.Fatalf("registered native update changed: %+v", update)
			}
		case <-time.After(time.Second):
			t.Fatal("registered native update did not reach consumer")
		}
	}
}

func TestRunnerPermissionRejectsUnregisteredOrigin(t *testing.T) {
	wire := newPermissionBindingWire(t, false)
	for _, source := range []*Session{nil, {id: "binding-session-0"}} {
		response, err := wire.runner.handlePermissionRequest(source, &RPCMessage{ID: json.RawMessage("21"), Params: json.RawMessage(`{"sessionId":"binding-session-0","tool":{"name":"controlled/read"}}`)})
		if err != nil || response == nil || response.Error == nil || response.Error.Code != ErrInvalidParams {
			t.Fatalf("unregistered origin accepted: %+v %v", response, err)
		}
		select {
		case request := <-wire.permissions:
			t.Fatalf("unregistered origin reached policy/UI authority: %s", request.Params)
		default:
		}
	}
}

func TestRunnerPermissionUnavailableUpstreamKeepsCancelledFallback(t *testing.T) {
	runner := NewRunner(config.Default())
	response, err := runner.handlePermissionRequest(nil, &RPCMessage{ID: json.RawMessage("21"), Params: json.RawMessage(`{}`)})
	if err != nil || response == nil || response.Error != nil {
		t.Fatalf("unavailable upstream fallback changed: %+v %v", response, err)
	}
	var result struct {
		Outcome struct {
			Outcome string `json:"outcome"`
		} `json:"outcome"`
	}
	if err := unmarshalRPCResult(response.Result, &result); err != nil || result.Outcome.Outcome != "cancelled" {
		t.Fatalf("unavailable upstream fallback is not cancelled: %s %v", response.Result, err)
	}
}

func TestRunnerPermissionCaseAliasCanonicalClaimOnTwoChildWire(t *testing.T) {
	for _, tc := range []struct {
		name, canonical, alias string
		allowed                bool
	}{
		{"foreign_canonical_then_source_alias", "binding-session-1", "binding-session-0", false},
		{"source_canonical_then_foreign_alias", "binding-session-0", "binding-session-1", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wire := newPermissionBindingWire(t, false)
			// Preserve this literal key order: struct decoding lets the later case alias overwrite the canonical key.
			params := json.RawMessage(fmt.Sprintf(`{"sessionId":"%s","SessionID":"%s","tool":{"name":"controlled/read"},"opaque":{"keep":true}}`, tc.canonical, tc.alias))
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			response, err := wire.children[0].CallRaw(ctx, "session/request_permission", params)
			if err != nil || response == nil {
				t.Fatalf("canonical claim transport failed: %+v %v", response, err)
			}
			if tc.allowed {
				if response.Error != nil {
					t.Fatalf("canonical source owner refused because of alias: %+v", response)
				}
				select {
				case request := <-wire.permissions:
					if string(request.Params) != string(params) {
						t.Fatalf("canonical owner raw fields changed: %s", request.Params)
					}
				default:
					t.Fatal("canonical owner did not reach policy authority")
				}
			} else {
				if response.Error == nil || response.Error.Code != ErrInvalidParams {
					t.Fatalf("foreign canonical identity accepted through source alias: %+v", response)
				}
				select {
				case request := <-wire.permissions:
					t.Fatalf("foreign canonical claim reached policy/UI: %s", request.Params)
				default:
				}
			}
		})
	}
}

func TestRunnerUpdateCaseAliasCanonicalClaimOnTwoChildWire(t *testing.T) {
	for _, tc := range []struct {
		name, canonical, alias string
		allowed                bool
	}{
		{"foreign_canonical_then_source_alias", "binding-session-1", "binding-session-0", false},
		{"source_canonical_then_foreign_alias", "binding-session-0", "binding-session-1", true},
	} {
		t.Run(tc.name, func(t *testing.T) {
			wire := newPermissionBindingWire(t, false)
			params := json.RawMessage(fmt.Sprintf(`{"sessionId":"%s","SessionID":"%s","update":{"sessionUpdate":"agent_message_chunk","content":{"type":"text","text":"CANONICAL_NATIVE_CONTENT"}},"opaque":{"keep":true}}`, tc.canonical, tc.alias))
			if err := wire.children[0].NotifyRaw("session/update", params); err != nil {
				t.Fatal(err)
			}
			// The normal same-child permission roundtrip is an ordering barrier: both real readers consume the preceding update first.
			ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
			defer cancel()
			response, err := wire.children[0].CallRaw(ctx, "session/request_permission", json.RawMessage(`{"sessionId":"binding-session-0","tool":{"name":"controlled/read"}}`))
			if err != nil || response == nil || response.Error != nil {
				t.Fatalf("normal ordering barrier failed: %+v %v", response, err)
			}
			select {
			case request := <-wire.permissions:
				if string(request.Params) != `{"sessionId":"binding-session-0","tool":{"name":"controlled/read"}}` {
					t.Fatalf("ordering barrier changed authority: %s", request.Params)
				}
			default:
				t.Fatal("normal ordering barrier did not reach policy authority")
			}
			select {
			case update := <-wire.updates:
				if !tc.allowed {
					t.Fatalf("foreign canonical update reached consumer through source alias: %s", update.Params)
				}
				if string(update.Params) != string(params) {
					t.Fatalf("canonical native raw fields changed: %s", update.Params)
				}
			default:
				if tc.allowed {
					t.Fatal("canonical source update refused because of alias")
				}
			}
		})
	}
}
