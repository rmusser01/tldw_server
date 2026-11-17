package acp

import (
	"encoding/json"
	"net"
	"os"
	"path/filepath"
	"reflect"
	"testing"
	"time"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/mcp/tools"
	"github.com/tldw/tldw-agent/internal/workspace"
)

func directPermissionSession(t *testing.T) (*Runner, *Session) {
	t.Helper()
	cfg := config.Default()
	cfg.Workspace.DefaultRoot = t.TempDir()
	if err := os.WriteFile(filepath.Join(cfg.Workspace.DefaultRoot, "answer.txt"), []byte("ORIGINAL_DIRECT_INPUT\n"), 0600); err != nil {
		t.Fatal(err)
	}
	cfg.Execution.CustomCommands = []config.CustomCommand{{ID: "harmless_echo", Template: "echo UAT_DIRECT_PERMISSION"}}
	ws := workspace.NewSession(cfg)
	session := &Session{id: "owned-direct-session", workspace: ws, fsTools: tools.NewFSTools(cfg, ws), terminal: NewTerminalManager(cfg, ws)}
	t.Cleanup(func() {
		for id := range session.terminal.terminals {
			_ = session.terminal.Release(id)
		}
	})
	runner := NewRunner(cfg)
	runner.sessionsMu.Lock()
	runner.sessions[session.id] = session
	runner.sessionsMu.Unlock()
	return runner, session
}

// Uses the real JSON-RPC readers/writers, separately from the normal prompt
// reader contract. The upstream peer supplies only external policy decisions.
func directPermissionPeer(t *testing.T, runner *Runner, result interface{}, rpcError bool, disconnect bool) <-chan *RPCMessage {
	t.Helper()
	a, b := net.Pipe()
	runner.upstream = NewConn(a, a)
	peer := NewConn(b, b)
	requests := make(chan *RPCMessage, 1)
	peer.SetHandler(func(msg *RPCMessage) (*RPCResponse, error) {
		requests <- msg
		if disconnect {
			_ = b.Close()
			return nil, nil
		}
		if rpcError {
			return NewErrorResponse(msg.ID, ErrInternal, "policy unavailable"), nil
		}
		return NewResultResponse(msg.ID, result), nil
	})
	done := make(chan error, 2)
	go func() { done <- runner.upstream.Run() }()
	go func() { done <- peer.Run() }()
	t.Cleanup(func() {
		_ = a.Close()
		_ = b.Close()
		for i := 0; i < 2; i++ {
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("permission wire did not stop")
			}
		}
	})
	return requests
}

func directPermissionRequest(t *testing.T, session *Session, method string) *RPCMessage {
	t.Helper()
	input := map[string]interface{}{"sessionId": session.id}
	if method == "fs/write_text_file" {
		input["path"] = filepath.Join(session.workspace.Root(), "answer.txt")
		input["content"] = "UAT_DIRECT_WRITE\n"
	} else {
		input["command"] = "echo"
		input["args"] = []string{"UAT_DIRECT_PERMISSION"}
		input["cwd"] = session.workspace.Root()
		input["outputByteLimit"] = 128
	}
	raw, err := json.Marshal(input)
	if err != nil {
		t.Fatal(err)
	}
	return &RPCMessage{JSONRPC: JSONRPCVersion, ID: json.RawMessage("41"), Method: method, Params: raw}
}

func assertDirectEffect(t *testing.T, session *Session, method string, approved bool) {
	t.Helper()
	if method == "fs/write_text_file" {
		path := filepath.Join(session.workspace.Root(), "answer.txt")
		content, err := os.ReadFile(path)
		if approved {
			if err != nil || string(content) != "UAT_DIRECT_WRITE\n" {
				t.Fatalf("approved write result %q, %v", content, err)
			}
		} else if err != nil || string(content) != "ORIGINAL_DIRECT_INPUT\n" {
			t.Fatalf("refused write changed original bytes: %q, %v", content, err)
		}
		return
	}
	if !approved {
		if len(session.terminal.terminals) != 0 {
			t.Fatal("refused execution created a terminal")
		}
		return
	}
	if len(session.terminal.terminals) != 1 {
		t.Fatal("approved execution did not create exactly one terminal")
	}
	for id := range session.terminal.terminals {
		status, err := session.terminal.WaitForExit(id)
		if err != nil || status.ExitCode == nil || *status.ExitCode != 0 {
			t.Fatalf("harmless command exit: %+v, %v", status, err)
		}
		// Output pipe copying can finish just after process settlement.
		deadline := time.Now().Add(time.Second)
		for {
			output, _, _, err := session.terminal.Output(id)
			if err == nil && output == "UAT_DIRECT_PERMISSION\n" {
				break
			}
			if time.Now().After(deadline) {
				t.Fatalf("harmless command output %q, %v", output, err)
			}
			time.Sleep(time.Millisecond)
		}
	}
}

func TestRunnerDirectPermissionDecisions(t *testing.T) {
	cases := []struct {
		name       string
		result     interface{}
		rpcError   bool
		disconnect bool
		upstream   bool
		approved   bool
	}{
		{name: "approved", result: map[string]interface{}{"outcome": map[string]interface{}{"outcome": "approved"}}, upstream: true, approved: true},
		{name: "denied", result: map[string]interface{}{"outcome": map[string]interface{}{"outcome": "denied"}}, upstream: true},
		{name: "cancelled", result: map[string]interface{}{"outcome": map[string]interface{}{"outcome": "cancelled"}}, upstream: true},
		{name: "nil_result", upstream: true},
		{name: "missing_outcome", result: map[string]interface{}{}, upstream: true},
		{name: "wrong_nesting", result: map[string]interface{}{"outcome": "approved"}, upstream: true},
		{name: "boolean_outcome", result: map[string]interface{}{"outcome": map[string]interface{}{"outcome": true}}, upstream: true},
		{name: "case_alias", result: map[string]interface{}{"outcome": map[string]interface{}{"outcome": "Approved"}}, upstream: true},
		{name: "rpc_error", rpcError: true, upstream: true},
		{name: "disconnected", disconnect: true, upstream: true},
		{name: "no_upstream"},
	}
	for _, method := range []string{"fs/write_text_file", "terminal/create"} {
		for _, tc := range cases {
			t.Run(method+"/"+tc.name, func(t *testing.T) {
				runner, session := directPermissionSession(t)
				var requests <-chan *RPCMessage
				if tc.upstream {
					requests = directPermissionPeer(t, runner, tc.result, tc.rpcError, tc.disconnect)
				}
				msg := directPermissionRequest(t, session, method)
				response, err := runner.handleDownstreamRequest(session, msg)
				if err != nil || response == nil || (response.Error == nil) != tc.approved {
					t.Errorf("permission decision approved=%v: response=%+v error=%v", tc.approved, response, err)
				}
				assertDirectEffect(t, session, method, tc.approved)
				if tc.upstream {
					select {
					case request := <-requests:
						if request.Method != "session/request_permission" {
							t.Fatalf("unexpected permission method %q", request.Method)
						}
						var params map[string]interface{}
						if err := json.Unmarshal(request.Params, &params); err != nil {
							t.Fatal(err)
						}
						var original map[string]interface{}
						_ = json.Unmarshal(msg.Params, &original)
						delete(original, "sessionId")
						want := map[string]interface{}{"sessionId": "owned-direct-session", "tool": map[string]interface{}{"name": method, "input": original}}
						if !reflect.DeepEqual(params, want) {
							t.Fatalf("permission not bound to exact operation: got %#v want %#v", params, want)
						}
					default:
						t.Fatal("direct operation bypassed permission wire")
					}
				}
			})
		}
	}
}

func TestRunnerDirectPermissionDisabled(t *testing.T) {
	for _, method := range []string{"fs/write_text_file", "terminal/create"} {
		t.Run(method, func(t *testing.T) {
			runner, session := directPermissionSession(t)
			runner.cfg.Security.RequireApprovalForWrites = false
			runner.cfg.Security.RequireApprovalForExec = false
			requests := directPermissionPeer(t, runner, nil, false, false)
			response, err := runner.handleDownstreamRequest(session, directPermissionRequest(t, session, method))
			if err != nil || response.Error != nil {
				t.Fatalf("approval-disabled operation failed: %+v %v", response, err)
			}
			assertDirectEffect(t, session, method, true)
			select {
			case <-requests:
				t.Fatal("approval-disabled operation requested permission")
			default:
			}
		})
	}
}

func TestRunnerDirectPermissionInputFences(t *testing.T) {
	for _, method := range []string{"fs/write_text_file", "terminal/create"} {
		t.Run(method+"/foreign_session", func(t *testing.T) {
			runner, session := directPermissionSession(t)
			requests := directPermissionPeer(t, runner, map[string]interface{}{"outcome": map[string]interface{}{"outcome": "approved"}}, false, false)
			msg := directPermissionRequest(t, session, method)
			var params map[string]interface{}
			_ = json.Unmarshal(msg.Params, &params)
			params["sessionId"] = "foreign-session"
			msg.Params, _ = json.Marshal(params)
			response, err := runner.handleDownstreamRequest(session, msg)
			if err != nil || response.Error == nil {
				t.Fatalf("foreign identity accepted: %+v %v", response, err)
			}
			assertDirectEffect(t, session, method, false)
			select {
			case <-requests:
				t.Fatal("invalid identity reached permission authority")
			default:
			}
		})
	}
	for _, pathKind := range []string{"relative", "outside", "blocked"} {
		t.Run("write/"+pathKind, func(t *testing.T) {
			runner, session := directPermissionSession(t)
			requests := directPermissionPeer(t, runner, map[string]interface{}{"outcome": map[string]interface{}{"outcome": "approved"}}, false, false)
			path := "relative.txt"
			if pathKind == "outside" {
				path = filepath.Join(t.TempDir(), "outside.txt")
			} else if pathKind == "blocked" {
				path = filepath.Join(session.workspace.Root(), ".env")
			}
			raw, _ := json.Marshal(map[string]interface{}{"path": path, "content": "NO_WRITE"})
			response, err := runner.handleDownstreamRequest(session, &RPCMessage{ID: json.RawMessage("41"), Method: "fs/write_text_file", Params: raw})
			if err != nil || response.Error == nil {
				t.Fatalf("invalid path accepted: %+v %v", response, err)
			}
			select {
			case <-requests:
				t.Fatal("invalid path reached permission authority")
			default:
			}
		})
	}
}
