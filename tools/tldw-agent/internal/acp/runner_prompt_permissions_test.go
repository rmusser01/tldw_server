package acp

import (
	"context"
	"encoding/json"
	"net"
	"os"
	"os/exec"
	"path/filepath"
	"sync"
	"testing"
	"time"

	"github.com/tldw/tldw-agent/internal/config"
)

type promptPermissionWire struct {
	client      *Conn
	root        string
	permissions chan *RPCMessage
	decisions   chan string
	answered    chan struct{}
	cancelled   chan struct{}
	settle      chan struct{}
	settleOnce  sync.Once
}

func newPromptPermissionWire(t *testing.T, holdSettlement bool) *promptPermissionWire {
	t.Helper()
	wire := &promptPermissionWire{root: t.TempDir(), permissions: make(chan *RPCMessage, 2), decisions: make(chan string, 2), answered: make(chan struct{}, 2), cancelled: make(chan struct{}, 2), settle: make(chan struct{})}
	if !holdSettlement {
		wire.settleOnce.Do(func() { close(wire.settle) })
	}
	cfg := config.Default()
	cfg.Agent.Command = "controlled-wire-agent"
	runner := NewRunner(cfg)
	a, b := net.Pipe()
	c, d := net.Pipe()
	wire.client = NewConn(a, a)
	peer := NewConn(d, d)
	peer.SetNotificationHandler(func(msg *RPCMessage) {
		if msg.Method == "session/cancel" {
			wire.cancelled <- struct{}{}
		}
	})
	peer.SetHandler(func(msg *RPCMessage) (*RPCResponse, error) {
		switch msg.Method {
		case "initialize":
			return NewResultResponse(msg.ID, map[string]interface{}{"protocolVersion": 1}), nil
		case "session/new":
			return NewResultResponse(msg.ID, map[string]interface{}{"sessionId": "wire-owned-session"}), nil
		case "session/prompt":
			// A real agent reads tool responses while its prompt runs. Keep its
			// protocol reader responsive rather than creating a fixture deadlock.
			go func() {
				ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
				defer cancel()
				tool, err := peer.Call(ctx, "fs/write_text_file", map[string]interface{}{"sessionId": "wire-owned-session", "path": filepath.Join(wire.root, "answer.txt"), "content": "APPROVED_WIRE_WRITE\n"})
				<-wire.settle
				result := map[string]interface{}{"stopReason": "end_turn", "toolRefused": err != nil || tool == nil || tool.Error != nil}
				_ = peer.SendResponse(NewResultResponse(msg.ID, result))
			}()
			return nil, nil
		}
		return NewErrorResponse(msg.ID, ErrMethodNotFound, "method not found"), nil
	})
	runner.SetSpawnFunc(func(config.AgentConfig) (*Conn, *exec.Cmd, error) { return NewConn(c, c), nil, nil })
	wire.client.SetHandler(func(msg *RPCMessage) (*RPCResponse, error) {
		wire.permissions <- msg
		decision := <-wire.decisions
		if err := wire.client.SendResponse(NewResultResponse(msg.ID, map[string]interface{}{"outcome": map[string]interface{}{"outcome": decision}})); err != nil {
			return nil, err
		}
		wire.answered <- struct{}{}
		return nil, nil
	})
	done := make(chan error, 3)
	go func() { done <- wire.client.Run() }()
	go func() { done <- peer.Run() }()
	go func() { done <- runner.Run(b, b) }()
	t.Cleanup(func() {
		wire.settleOnce.Do(func() { close(wire.settle) })
		close(wire.decisions)
		for _, conn := range []net.Conn{a, b, c, d} {
			_ = conn.Close()
		}
		for i := 0; i < 3; i++ {
			select {
			case <-done:
			case <-time.After(time.Second):
				t.Error("prompt wire did not stop")
			}
		}
	})
	ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
	defer cancel()
	response, err := wire.client.Call(ctx, "session/new", map[string]interface{}{"cwd": wire.root})
	if err != nil || response.Error != nil {
		t.Fatalf("real wire session creation failed: %+v %v", response, err)
	}
	return wire
}

func startWirePrompt(wire *promptPermissionWire) <-chan *RPCMessage {
	result := make(chan *RPCMessage, 1)
	go func() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Second)
		defer cancel()
		response, _ := wire.client.Call(ctx, "session/prompt", map[string]interface{}{"sessionId": "wire-owned-session", "prompt": []map[string]string{{"role": "user", "content": "Controlled source test"}}})
		result <- response
	}()
	return result
}

func TestRunnerPromptPermissionCanCompleteOnNormalWire(t *testing.T) {
	wire := newPromptPermissionWire(t, false)
	result := startWirePrompt(wire)
	select {
	case request := <-wire.permissions:
		if request.Method != "session/request_permission" {
			t.Fatalf("unexpected request: %s", request.Method)
		}
	case <-time.After(time.Second):
		t.Fatal("normal prompt did not request permission")
	}
	wire.decisions <- "approved"
	response := <-result
	if response == nil || response.Error != nil {
		t.Fatalf("approved normal prompt failed to settle: %+v", response)
	}
	content, err := os.ReadFile(filepath.Join(wire.root, "answer.txt"))
	if err != nil || string(content) != "APPROVED_WIRE_WRITE\n" {
		t.Fatalf("approved normal prompt did not write: %q %v", content, err)
	}
}

func TestRunnerPromptStopRetiresPermissionUntilSettlement(t *testing.T) {
	wire := newPromptPermissionWire(t, true)
	result := startWirePrompt(wire)
	select {
	case <-wire.permissions:
	case <-time.After(time.Second):
		t.Fatal("normal prompt did not request permission")
	}
	notified := make(chan error, 1)
	go func() {
		notified <- wire.client.Notify("session/cancel", map[string]string{"sessionId": "wire-owned-session"})
	}()
	select {
	case <-wire.cancelled:
	case <-time.After(time.Second):
		t.Fatal("Stop notification was blocked by pending permission")
	}
	if err := <-notified; err != nil {
		t.Fatal(err)
	}
	// The already stopped permission cannot be revived by a late approval.
	wire.decisions <- "approved"
	select {
	case <-wire.answered:
	case <-time.After(time.Second):
		t.Fatal("late permission reply did not finish sending")
	}
	ctx, cancel := context.WithTimeout(context.Background(), time.Second)
	defer cancel()
	busy, err := wire.client.Call(ctx, "session/prompt", map[string]string{"sessionId": "wire-owned-session"})
	if err != nil || busy == nil || busy.Error == nil {
		t.Fatalf("unsettled prompt admitted a successor: %+v %v", busy, err)
	}
	wire.settleOnce.Do(func() { close(wire.settle) })
	response := <-result
	if response == nil || response.Error != nil {
		t.Fatalf("stopped prompt failed to settle truthfully: %+v", response)
	}
	var outcome struct {
		ToolRefused bool `json:"toolRefused"`
	}
	if err := json.Unmarshal(response.Result, &outcome); err != nil || !outcome.ToolRefused {
		t.Fatalf("late approval permitted stopped tool: %s %v", response.Result, err)
	}
	if _, err := os.Stat(filepath.Join(wire.root, "answer.txt")); !os.IsNotExist(err) {
		t.Fatalf("stopped prompt wrote a file: %v", err)
	}
}
