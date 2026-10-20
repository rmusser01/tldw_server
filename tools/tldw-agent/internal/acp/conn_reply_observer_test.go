package acp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net"
	"strings"
	"sync/atomic"
	"testing"
	"time"
)

func TestConnReplyObserverRunsBeforeNextNotification(t *testing.T) {
	for _, kind := range []string{"result", "error", "default"} {
		t.Run(kind, func(t *testing.T) {
			a, b := net.Pipe()
			conn := NewConn(a, a)
			defer a.Close()
			defer b.Close()
			var observed atomic.Int32
			notifications := make(chan int32, 1)
			conn.SetNotificationHandler(func(*RPCMessage) { notifications <- observed.Load() })
			run := make(chan error, 1)
			go func() { run <- conn.Run() }()
			defer func() {
				_ = b.Close()
				select {
				case <-run:
				case <-time.After(time.Second):
					t.Error("actual reader did not retire")
				}
			}()
			peer := make(chan error, 1)
			const raw = `{ "stopReason" : "end_turn", "_meta" : {"original":null} }`
			go func() {
				line, err := ReadLineMessage(bufio.NewReader(b))
				if err != nil {
					peer <- err
					return
				}
				var request RPCMessage
				if err = json.Unmarshal(line, &request); err != nil {
					peer <- err
					return
				}
				response := fmt.Sprintf(`{"jsonrpc":"2.0","id":%s,"result":%s}`, request.ID, raw)
				if kind == "error" {
					response = fmt.Sprintf(`{"jsonrpc":"2.0","id":%s,"error":{"code":-32603,"message":"controlled"}}`, request.ID)
				}
				// The foreign response cannot close the observer; the next frame
				// after the matching response must see synchronous closure.
				_, err = io.WriteString(b, "{\"jsonrpc\":\"2.0\",\"id\":9999,\"result\":{}}\n"+response+"\n{\"jsonrpc\":\"2.0\",\"method\":\"session/update\",\"params\":{}}\n")
				peer <- err
			}()
			ctx, cancel := context.WithTimeout(context.Background(), time.Second)
			defer cancel()
			var response *RPCMessage
			var err error
			if kind == "default" {
				response, err = conn.CallRaw(ctx, "controlled", nil)
			} else {
				response, err = conn.callRaw(ctx, "controlled", nil, func() { observed.Add(1) })
			}
			if err != nil || response == nil {
				t.Fatalf("matched response failed: %+v %v", response, err)
			}
			if kind == "error" {
				if response.Error == nil || response.Error.Code != -32603 || response.Error.Message != "controlled" {
					t.Fatalf("raw RPC error changed: %+v", response)
				}
			} else if string(response.Result) != raw {
				t.Fatalf("raw result bytes changed: %s", response.Result)
			}
			want := int32(1)
			if kind == "default" {
				want = 0
			}
			select {
			case value := <-notifications:
				if value != want || observed.Load() != want {
					t.Fatalf("observer ordering/count: notification=%d total=%d want=%d", value, observed.Load(), want)
				}
			case <-ctx.Done():
				t.Fatal("following notification was not dispatched")
			}
			if err := <-peer; err != nil {
				t.Fatal(err)
			}
			conn.pendingMu.Lock()
			defer conn.pendingMu.Unlock()
			if len(conn.pending) != 0 || len(conn.responseObservers) != 0 {
				t.Fatal("matched call left pending state")
			}
		})
	}
}

func TestConnReplyObserverCleansDeadlineAndEOF(t *testing.T) {
	for _, kind := range []string{"deadline", "eof"} {
		t.Run(kind, func(t *testing.T) {
			a, b := net.Pipe()
			conn := NewConn(a, a)
			defer a.Close()
			defer b.Close()
			run := make(chan error, 1)
			go func() { run <- conn.Run() }()
			defer func() {
				_ = b.Close()
				select {
				case <-run:
				case <-time.After(time.Second):
					t.Error("actual reader did not retire")
				}
			}()
			read := make(chan error, 1)
			go func() {
				_, err := ReadLineMessage(bufio.NewReader(b))
				read <- err
				if kind == "eof" {
					_ = b.Close()
				}
			}()
			var observed atomic.Int32
			ctx, cancel := context.WithTimeout(context.Background(), 30*time.Millisecond)
			defer cancel()
			_, err := conn.callRaw(ctx, "held", nil, func() { observed.Add(1) })
			want := context.DeadlineExceeded
			if kind == "eof" {
				want = ErrConnectionClosed
			}
			if !errors.Is(err, want) || observed.Load() != 0 {
				t.Fatalf("unsettled call inferred reply: err=%v count=%d", err, observed.Load())
			}
			if err := <-read; err != nil {
				t.Fatal(err)
			}
			conn.pendingMu.Lock()
			defer conn.pendingMu.Unlock()
			if len(conn.pending) != 0 || len(conn.responseObservers) != 0 {
				t.Fatal("failed call left pending state")
			}
		})
	}
}

type failedReplyWriter struct{}

func (failedReplyWriter) Write([]byte) (int, error) { return 0, io.ErrClosedPipe }

func TestConnReplyObserverCleansSendFailure(t *testing.T) {
	conn := NewConn(strings.NewReader(""), failedReplyWriter{})
	var observed atomic.Int32
	_, err := conn.callRaw(context.Background(), "controlled", nil, func() { observed.Add(1) })
	if !errors.Is(err, io.ErrClosedPipe) || observed.Load() != 0 {
		t.Fatalf("send failure inferred reply: %v count=%d", err, observed.Load())
	}
	if len(conn.pending) != 0 || len(conn.responseObservers) != 0 {
		t.Fatal("send failure left pending state")
	}
}
