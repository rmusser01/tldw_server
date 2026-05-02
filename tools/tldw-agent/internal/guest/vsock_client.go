package guest

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"time"
)

const defaultGuestVersion = "dev"

var errGuestConnectionClosed = errors.New("guest transport connection closed")

type VSockClientConfig struct {
	VMID             string
	ConnectionToken  string
	HostPort         uint32
	WorkspaceRoot    string
	GuestVersion     string
	ReconnectDelay   time.Duration
	HandshakeTimeout time.Duration
}

type vsockDialer interface {
	Dial(ctx context.Context, port uint32) (io.ReadWriteCloser, error)
}

type systemVSockDialer struct{}

func (d systemVSockDialer) Dial(ctx context.Context, port uint32) (io.ReadWriteCloser, error) {
	return dialVSockConnection(ctx, port)
}

type VSockClient struct {
	cfg    VSockClientConfig
	dialer vsockDialer
}

func NewVSockClientFromEnv() (*VSockClient, error) {
	cfg, err := loadVSockClientConfigFromEnv()
	if err != nil {
		return nil, err
	}
	return &VSockClient{
		cfg:    cfg,
		dialer: systemVSockDialer{},
	}, nil
}

func ShouldUseVSockMode() bool {
	return os.Getenv("TLDW_AGENT_GUEST_HOST_VSOCK_PORT") != ""
}

func (c *VSockClient) Run(ctx context.Context, server *Server) error {
	reconnect := false
	for {
		if ctx.Err() != nil {
			return ctx.Err()
		}

		conn, err := c.dialer.Dial(ctx, c.cfg.HostPort)
		if err != nil {
			return err
		}

		err = c.primeConnection(ctx, conn, server, reconnect)
		if err != nil {
			_ = conn.Close()
			return err
		}

		err = server.ServeStream(conn, conn)
		_ = conn.Close()
		if err != nil {
			return err
		}

		reconnect = true
		if ctx.Err() != nil {
			return nil
		}
		select {
		case <-ctx.Done():
			return nil
		case <-time.After(c.cfg.ReconnectDelay):
		}
	}
}

func (c *VSockClient) primeConnection(
	ctx context.Context,
	conn io.ReadWriteCloser,
	server *Server,
	reconnect bool,
) error {
	reader := bufio.NewReader(conn)
	if reconnect {
		if err := c.sendReconnect(conn, reader); err != nil {
			return err
		}
	} else {
		if err := c.sendHandshake(conn, reader); err != nil {
			return err
		}
	}
	if err := c.sendReady(conn, reader, server); err != nil {
		return err
	}
	if ctx.Err() != nil {
		return ctx.Err()
	}
	return nil
}

func (c *VSockClient) sendHandshake(conn io.Writer, reader *bufio.Reader) error {
	requestID := newRequestID()
	if err := writeJSONLine(conn, HandshakeRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       requestID,
		Type:            "handshake",
		VMID:            c.cfg.VMID,
		ConnectionToken: c.cfg.ConnectionToken,
		GuestVersion:    c.cfg.GuestVersion,
		WorkspaceRoot:   c.cfg.WorkspaceRoot,
		Capabilities:    guestCapabilities(),
	}); err != nil {
		return err
	}

	var response HandshakeAck
	if err := decodeLine(reader, &response); err != nil {
		return err
	}
	if response.ProtocolVersion != ProtocolVersion {
		return fmt.Errorf("handshake protocol mismatch: %s", response.ProtocolVersion)
	}
	if response.RequestID != requestID {
		return fmt.Errorf("handshake request id mismatch: %s", response.RequestID)
	}
	if response.Status != "accepted" {
		return fmt.Errorf("handshake not accepted: %s", response.Status)
	}
	if response.VMID != c.cfg.VMID {
		return fmt.Errorf("handshake vm mismatch: %s", response.VMID)
	}
	return nil
}

func (c *VSockClient) sendReconnect(conn io.Writer, reader *bufio.Reader) error {
	requestID := newRequestID()
	if err := writeJSONLine(conn, ReconnectRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       requestID,
		Type:            "reconnect",
		VMID:            c.cfg.VMID,
		ConnectionToken: c.cfg.ConnectionToken,
	}); err != nil {
		return err
	}

	var response ReconnectAck
	if err := decodeLine(reader, &response); err != nil {
		return err
	}
	if response.ProtocolVersion != ProtocolVersion {
		return fmt.Errorf("reconnect protocol mismatch: %s", response.ProtocolVersion)
	}
	if response.RequestID != requestID {
		return fmt.Errorf("reconnect request id mismatch: %s", response.RequestID)
	}
	if response.Status != "accepted" {
		return fmt.Errorf("reconnect not accepted: %s", response.Status)
	}
	if response.VMID != c.cfg.VMID {
		return fmt.Errorf("reconnect vm mismatch: %s", response.VMID)
	}
	return nil
}

func (c *VSockClient) sendReady(conn io.Writer, reader *bufio.Reader, server *Server) error {
	requestID := newRequestID()
	if err := writeJSONLine(conn, ReadyRequest{
		ProtocolVersion: ProtocolVersion,
		RequestID:       requestID,
		Type:            "ready",
	}); err != nil {
		return err
	}

	var response ReadyResponse
	if err := decodeLine(reader, &response); err != nil {
		return err
	}
	if response.ProtocolVersion != ProtocolVersion {
		return fmt.Errorf("ready protocol mismatch: %s", response.ProtocolVersion)
	}
	if response.RequestID != requestID {
		return fmt.Errorf("ready request id mismatch: %s", response.RequestID)
	}
	if response.Status != "ready" {
		return fmt.Errorf("ready not accepted: %s", response.Status)
	}
	if response.WorkspaceRoot != server.session.Root() {
		return fmt.Errorf("workspace root mismatch: %s", response.WorkspaceRoot)
	}
	return nil
}

func guestCapabilities() []string {
	return []string{"exec", "output_cap_v1"}
}

func loadVSockClientConfigFromEnv() (VSockClientConfig, error) {
	portValue := os.Getenv("TLDW_AGENT_GUEST_HOST_VSOCK_PORT")
	if portValue == "" {
		return VSockClientConfig{}, errors.New("TLDW_AGENT_GUEST_HOST_VSOCK_PORT is required")
	}
	port, err := strconv.ParseUint(portValue, 10, 32)
	if err != nil {
		return VSockClientConfig{}, fmt.Errorf("parse TLDW_AGENT_GUEST_HOST_VSOCK_PORT: %w", err)
	}

	vmID := os.Getenv("TLDW_AGENT_GUEST_VM_ID")
	if vmID == "" {
		return VSockClientConfig{}, errors.New("TLDW_AGENT_GUEST_VM_ID is required")
	}
	connectionToken := os.Getenv("TLDW_AGENT_GUEST_CONNECTION_TOKEN")
	if connectionToken == "" {
		return VSockClientConfig{}, errors.New("TLDW_AGENT_GUEST_CONNECTION_TOKEN is required")
	}
	workspaceRoot := os.Getenv("TLDW_AGENT_GUEST_WORKSPACE_ROOT")
	if workspaceRoot == "" {
		return VSockClientConfig{}, errors.New("TLDW_AGENT_GUEST_WORKSPACE_ROOT is required")
	}
	guestVersion := os.Getenv("TLDW_AGENT_GUEST_VERSION")
	if guestVersion == "" {
		guestVersion = defaultGuestVersion
	}

	return VSockClientConfig{
		VMID:             vmID,
		ConnectionToken:  connectionToken,
		HostPort:         uint32(port),
		WorkspaceRoot:    workspaceRoot,
		GuestVersion:     guestVersion,
		ReconnectDelay:   250 * time.Millisecond,
		HandshakeTimeout: 5 * time.Second,
	}, nil
}

func newRequestID() string {
	return strconv.FormatInt(time.Now().UnixNano(), 10)
}

func decodeLine(reader *bufio.Reader, target any) error {
	line, err := reader.ReadBytes('\n')
	if err != nil {
		return err
	}
	return json.Unmarshal(line, target)
}

func writeJSONLine(writer io.Writer, value any) error {
	payload, err := json.Marshal(value)
	if err != nil {
		return err
	}
	payload = append(payload, '\n')
	_, err = writer.Write(payload)
	return err
}
