package guest

const ProtocolVersion = "1"

type ReadyRequest struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Type            string `json:"type"`
}

type ReadyResponse struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Status          string `json:"status"`
	WorkspaceRoot   string `json:"workspace_root,omitempty"`
}

type ExecRequest struct {
	ProtocolVersion string            `json:"protocol_version"`
	RequestID       string            `json:"request_id"`
	Type            string            `json:"type,omitempty"`
	Argv            []string          `json:"argv"`
	Cwd             string            `json:"cwd,omitempty"`
	Env             map[string]string `json:"env,omitempty"`
	TimeoutSec      int               `json:"timeout_sec,omitempty"`
}

type ExecResponse struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	ExitCode        int    `json:"exit_code"`
	Stdout          string `json:"stdout,omitempty"`
	Stderr          string `json:"stderr,omitempty"`
}

type ErrorResponse struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	ErrorCode       string `json:"error_code"`
	Message         string `json:"message"`
}
