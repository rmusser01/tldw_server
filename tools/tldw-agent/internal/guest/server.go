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
