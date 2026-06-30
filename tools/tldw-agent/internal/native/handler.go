package native

import (
	"encoding/json"
	"fmt"
	"io"
	"log"
	"os"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/mcp"
)

// Request represents an incoming request from the browser extension.
type Request struct {
	ID      string          `json:"id"`
	Type    string          `json:"type"`
	Payload json.RawMessage `json:"payload"`
}

// Response represents an outgoing response to the browser extension.
type Response struct {
	ID        string      `json:"id"`
	OK        bool        `json:"ok"`
	Data      interface{} `json:"data,omitempty"`
	Error     *ErrorInfo  `json:"error,omitempty"`
	Streaming bool        `json:"streaming,omitempty"`
}

// ErrorInfo contains error details.
type ErrorInfo struct {
	Code    string `json:"code"`
	Message string `json:"message"`
}

// Handler manages native messaging communication with the browser extension.
type Handler struct {
	mcpServer *mcp.Server
	config    *config.Config
	stdin     io.Reader
	stdout    io.Writer
}

// NewHandler creates a new native messaging handler.
func NewHandler(mcpServer *mcp.Server, cfg *config.Config) *Handler {
	return &Handler{
		mcpServer: mcpServer,
		config:    cfg,
		stdin:     os.Stdin,
		stdout:    os.Stdout,
	}
}

// Run starts the native messaging loop.
func (h *Handler) Run() error {
	log.Println("Native messaging handler started")

	for {
		// Read incoming request
		var req Request
		if err := ReadJSON(h.stdin, &req); err != nil {
			if err == io.EOF {
				log.Println("EOF received, shutting down")
				return nil
			}
			log.Printf("Error reading request: %v", err)
			continue
		}

		log.Printf("Received request: id=%s type=%s", req.ID, req.Type)

		// Process request and send response
		resp := h.handleRequest(&req)
		if err := WriteJSON(h.stdout, resp); err != nil {
			log.Printf("Error writing response: %v", err)
		}
	}
}

// handleRequest dispatches the request to the appropriate handler.
func (h *Handler) handleRequest(req *Request) *Response {
	switch req.Type {
	case "ping":
		return h.handlePing(req)
	case "config":
		return h.handleConfig(req)
	case "mcp_request":
		return h.handleMCPRequest(req)
	case "mcp_list_tools":
		return h.handleListTools(req)
	default:
		return &Response{
			ID: req.ID,
			OK: false,
			Error: &ErrorInfo{
				Code:    "unknown_type",
				Message: fmt.Sprintf("Unknown request type: %s", req.Type),
			},
		}
	}
}

// handlePing responds to ping requests (used to check if host is running).
func (h *Handler) handlePing(req *Request) *Response {
	return &Response{
		ID: req.ID,
		OK: true,
		Data: map[string]interface{}{
			"version": "0.1.0",
			"status":  "ready",
		},
	}
}

// handleConfig returns or updates configuration.
func (h *Handler) handleConfig(req *Request) *Response {
	// For now, just return current config (read-only)
	return &Response{
		ID: req.ID,
		OK: true,
		Data: map[string]interface{}{
			"llm_endpoint":    h.config.Server.LLMEndpoint,
			"execution_enabled": h.config.Execution.Enabled,
			"shell":           h.config.GetShell(),
		},
	}
}

// handleListTools returns the list of available MCP tools.
func (h *Handler) handleListTools(req *Request) *Response {
	tools := h.mcpServer.ListTools()
	return &Response{
		ID:   req.ID,
		OK:   true,
		Data: tools,
	}
}

// MCPRequest represents an MCP tool call request.
type MCPRequest struct {
	Method    string          `json:"method"`
	ToolName  string          `json:"tool_name"`
	Arguments json.RawMessage `json:"arguments"`
}

// handleMCPRequest processes an MCP tool call.
func (h *Handler) handleMCPRequest(req *Request) *Response {
	var mcpReq MCPRequest
	if err := json.Unmarshal(req.Payload, &mcpReq); err != nil {
		return &Response{
			ID: req.ID,
			OK: false,
			Error: &ErrorInfo{
				Code:    "invalid_payload",
				Message: fmt.Sprintf("Failed to parse MCP request: %v", err),
			},
		}
	}

	// Handle different MCP methods
	switch mcpReq.Method {
	case "tools/call":
		result, err := h.mcpServer.ExecuteTool(mcpReq.ToolName, mcpReq.Arguments)
		if err != nil {
			return &Response{
				ID: req.ID,
				OK: false,
				Error: &ErrorInfo{
					Code:    "tool_error",
					Message: err.Error(),
				},
			}
		}
		return &Response{
			ID:   req.ID,
			OK:   true,
			Data: result,
		}

	case "tools/list":
		tools := h.mcpServer.ListTools()
		return &Response{
			ID:   req.ID,
			OK:   true,
			Data: tools,
		}

	default:
		return &Response{
			ID: req.ID,
			OK: false,
			Error: &ErrorInfo{
				Code:    "unknown_method",
				Message: fmt.Sprintf("Unknown MCP method: %s", mcpReq.Method),
			},
		}
	}
}
