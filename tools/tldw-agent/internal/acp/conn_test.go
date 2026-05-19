package acp

import (
	"bufio"
	"context"
	"errors"
	"net"
	"testing"
	"time"
)

func TestConnCallRawReturnsConnectionClosedErrorOnEOF(t *testing.T) {
	clientConn, serverConn := net.Pipe()
	conn := NewConn(clientConn, clientConn)

	runErr := make(chan error, 1)
	go func() {
		runErr <- conn.Run()
	}()

	t.Cleanup(func() {
		_ = clientConn.Close()
		_ = serverConn.Close()
		select {
		case <-runErr:
		case <-time.After(time.Second):
		}
	})

	serverRead := make(chan struct{}, 1)
	go func() {
		reader := bufio.NewReader(serverConn)
		_, _ = ReadLineMessage(reader)
		serverRead <- struct{}{}
		_ = serverConn.Close()
	}()

	ctx, cancel := context.WithTimeout(context.Background(), 400*time.Millisecond)
	defer cancel()

	_, err := conn.CallRaw(ctx, "session/new", nil)
	if err == nil {
		t.Fatalf("expected call to fail when peer closes connection")
	}
	if errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("call should fail due closed connection, got timeout: %v", err)
	}

	select {
	case <-serverRead:
	case <-time.After(time.Second):
		t.Fatalf("test did not observe request write before close")
	}
}
