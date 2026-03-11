package guest

import (
	"fmt"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/workspace"
)

type Server struct {
	cfg     *config.Config
	session *workspace.Session
}

func NewServer(root string) (*Server, error) {
	cfg := config.Default()
	session := workspace.NewSession(cfg)
	if err := session.SetRoot(root); err != nil {
		return nil, fmt.Errorf("set workspace root: %w", err)
	}
	return &Server{
		cfg:     cfg,
		session: session,
	}, nil
}

func (s *Server) Ready(req ReadyRequest) ReadyResponse {
	return ReadyResponse{
		ProtocolVersion: ProtocolVersion,
		RequestID:       req.RequestID,
		Status:          "ready",
		WorkspaceRoot:   s.session.Root(),
	}
}

func (s *Server) Handshake(req HandshakeRequest) HandshakeAck {
	return HandshakeAck{
		ProtocolVersion: ProtocolVersion,
		RequestID:       req.RequestID,
		Type:            "handshake_ack",
		Status:          "accepted",
		VMID:            req.VMID,
	}
}

func (s *Server) Heartbeat(req HeartbeatRequest) HeartbeatResponse {
	return HeartbeatResponse{
		ProtocolVersion: ProtocolVersion,
		RequestID:       req.RequestID,
		Type:            "heartbeat",
		Status:          "alive",
		VMID:            req.VMID,
	}
}

func (s *Server) Reconnect(req ReconnectRequest) ReconnectAck {
	return ReconnectAck{
		ProtocolVersion: ProtocolVersion,
		RequestID:       req.RequestID,
		Type:            "reconnect_ack",
		Status:          "accepted",
		VMID:            req.VMID,
	}
}
