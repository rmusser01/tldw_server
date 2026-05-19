package guest

const ProtocolVersion = "1"

type HandshakeRequest struct {
	ProtocolVersion string   `json:"protocol_version"`
	RequestID       string   `json:"request_id"`
	Type            string   `json:"type"`
	VMID            string   `json:"vm_id"`
	ConnectionToken string   `json:"connection_token"`
	GuestVersion    string   `json:"guest_version,omitempty"`
	WorkspaceRoot   string   `json:"workspace_root,omitempty"`
	Capabilities    []string `json:"capabilities,omitempty"`
}

type HandshakeAck struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Type            string `json:"type,omitempty"`
	Status          string `json:"status"`
	VMID            string `json:"vm_id"`
}

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

type HeartbeatRequest struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Type            string `json:"type"`
	VMID            string `json:"vm_id"`
}

type HeartbeatResponse struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Type            string `json:"type,omitempty"`
	Status          string `json:"status"`
	VMID            string `json:"vm_id"`
}

type ReconnectRequest struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Type            string `json:"type"`
	VMID            string `json:"vm_id"`
	ConnectionToken string `json:"connection_token"`
}

type ReconnectAck struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Type            string `json:"type,omitempty"`
	Status          string `json:"status"`
	VMID            string `json:"vm_id"`
}

type ExecRequest struct {
	ProtocolVersion string            `json:"protocol_version"`
	RequestID       string            `json:"request_id"`
	Type            string            `json:"type,omitempty"`
	Argv            []string          `json:"argv"`
	Cwd             string            `json:"cwd,omitempty"`
	Env             map[string]string `json:"env,omitempty"`
	TimeoutSec      int               `json:"timeout_sec,omitempty"`
	MaxOutputBytes  *int              `json:"max_output_bytes,omitempty"`
}

type ExecResponse struct {
	ProtocolVersion string            `json:"protocol_version"`
	RequestID       string            `json:"request_id"`
	ExitCode        int               `json:"exit_code"`
	Stdout          string            `json:"stdout,omitempty"`
	Stderr          string            `json:"stderr,omitempty"`
	Details         map[string]string `json:"details,omitempty"`
}

type ErrorResponse struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	ErrorCode       string `json:"error_code"`
	Message         string `json:"message"`
}
