package guest

import (
	"bufio"
	"context"
	"io"
	"net"
	"reflect"
	"testing"
)

type recordingDialer struct {
	conns []io.ReadWriteCloser
	ports []uint32
}

func (d *recordingDialer) Dial(_ context.Context, port uint32) (io.ReadWriteCloser, error) {
	d.ports = append(d.ports, port)
	if len(d.conns) == 0 {
		return nil, io.EOF
	}
	conn := d.conns[0]
	d.conns = d.conns[1:]
	return conn, nil
}

func TestGuestVSockClientSendsHandshakeAndReady(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	guestConn, helperConn := net.Pipe()
	defer helperConn.Close()

	client := &VSockClient{
		cfg: VSockClientConfig{
			VMID:            "vm-handshake",
			ConnectionToken: "token-handshake",
			HostPort:        4242,
			WorkspaceRoot:   root,
			GuestVersion:    "1.0.0",
		},
		dialer: &recordingDialer{conns: []io.ReadWriteCloser{guestConn}},
	}

	done := make(chan error, 1)
	go func() {
		defer helperConn.Close()
		reader := bufio.NewReader(helperConn)

		var handshake HandshakeRequest
		if err := decodeLine(reader, &handshake); err != nil {
			done <- err
			return
		}
		if handshake.Type != "handshake" {
			done <- errUnexpectedValue("handshake type", handshake.Type)
			return
		}
		if handshake.VMID != "vm-handshake" {
			done <- errUnexpectedValue("vm_id", handshake.VMID)
			return
		}
		if handshake.ConnectionToken != "token-handshake" {
			done <- errUnexpectedValue("connection_token", handshake.ConnectionToken)
			return
		}
		expectedCapabilities := []string{"exec", "output_cap_v1"}
		if !reflect.DeepEqual(handshake.Capabilities, expectedCapabilities) {
			done <- errUnexpectedValue("capabilities", handshake.Capabilities)
			return
		}
		if err := writeJSONLine(helperConn, HandshakeAck{
			ProtocolVersion: ProtocolVersion,
			RequestID:       handshake.RequestID,
			Type:            "handshake_ack",
			Status:          "accepted",
			VMID:            handshake.VMID,
		}); err != nil {
			done <- err
			return
		}

		var ready ReadyRequest
		if err := decodeLine(reader, &ready); err != nil {
			done <- err
			return
		}
		if ready.Type != "ready" {
			done <- errUnexpectedValue("ready type", ready.Type)
			return
		}
		if err := writeJSONLine(helperConn, ReadyResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       ready.RequestID,
			Status:          "ready",
			WorkspaceRoot:   root,
		}); err != nil {
			done <- err
			return
		}
		done <- nil
	}()

	if err := client.primeConnection(context.Background(), guestConn, server, false); err != nil {
		t.Fatalf("primeConnection() error = %v", err)
	}
	if err := <-done; err != nil {
		t.Fatalf("helper side error = %v", err)
	}
}

func TestGuestVSockClientReconnectsWithSameVMIDAndToken(t *testing.T) {
	root := t.TempDir()
	server, err := NewServer(root)
	if err != nil {
		t.Fatalf("NewServer() error = %v", err)
	}

	firstGuestConn, firstHelperConn := net.Pipe()
	defer firstHelperConn.Close()
	secondGuestConn, secondHelperConn := net.Pipe()
	defer secondHelperConn.Close()

	client := &VSockClient{
		cfg: VSockClientConfig{
			VMID:            "vm-reconnect",
			ConnectionToken: "token-reconnect",
			HostPort:        4242,
			WorkspaceRoot:   root,
			GuestVersion:    "1.0.0",
		},
		dialer: &recordingDialer{conns: []io.ReadWriteCloser{firstGuestConn, secondGuestConn}},
	}

	assertHello := func(helperConn net.Conn, expectedType string, done chan error) {
		defer helperConn.Close()
		reader := bufio.NewReader(helperConn)

		var envelope map[string]any
		if err := decodeLine(reader, &envelope); err != nil {
			done <- err
			return
		}
		if envelope["type"] != expectedType {
			done <- errUnexpectedValue("message type", envelope["type"])
			return
		}
		if envelope["vm_id"] != "vm-reconnect" {
			done <- errUnexpectedValue("vm_id", envelope["vm_id"])
			return
		}
		if envelope["connection_token"] != "token-reconnect" {
			done <- errUnexpectedValue("connection_token", envelope["connection_token"])
			return
		}

		requestID, _ := envelope["request_id"].(string)
		if expectedType == "handshake" {
			if err := writeJSONLine(helperConn, HandshakeAck{
				ProtocolVersion: ProtocolVersion,
				RequestID:       requestID,
				Type:            "handshake_ack",
				Status:          "accepted",
				VMID:            "vm-reconnect",
			}); err != nil {
				done <- err
				return
			}
		} else {
			if err := writeJSONLine(helperConn, ReconnectAck{
				ProtocolVersion: ProtocolVersion,
				RequestID:       requestID,
				Type:            "reconnect_ack",
				Status:          "accepted",
				VMID:            "vm-reconnect",
			}); err != nil {
				done <- err
				return
			}
		}

		var ready ReadyRequest
		if err := decodeLine(reader, &ready); err != nil {
			done <- err
			return
		}
		if err := writeJSONLine(helperConn, ReadyResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       ready.RequestID,
			Status:          "ready",
			WorkspaceRoot:   root,
		}); err != nil {
			done <- err
			return
		}
		done <- nil
	}

	firstDone := make(chan error, 1)
	secondDone := make(chan error, 1)
	go assertHello(firstHelperConn, "handshake", firstDone)
	go assertHello(secondHelperConn, "reconnect", secondDone)

	if err := client.primeConnection(context.Background(), firstGuestConn, server, false); err != nil {
		t.Fatalf("primeConnection(first) error = %v", err)
	}
	if err := <-firstDone; err != nil {
		t.Fatalf("first helper side error = %v", err)
	}

	if err := client.primeConnection(context.Background(), secondGuestConn, server, true); err != nil {
		t.Fatalf("primeConnection(second) error = %v", err)
	}
	if err := <-secondDone; err != nil {
		t.Fatalf("second helper side error = %v", err)
	}
}

type unexpectedValueError struct {
	name  string
	value any
}

func (e unexpectedValueError) Error() string {
	return "unexpected " + e.name
}

func errUnexpectedValue(name string, value any) error {
	return unexpectedValueError{name: name, value: value}
}
