package acp

import "encoding/json"

const (
	// JSONRPCVersion is the only supported ACP JSON-RPC version.
	JSONRPCVersion = "2.0"

	// Standard JSON-RPC error codes.
	ErrParse          = -32700
	ErrInvalidReq     = -32600
	ErrMethodNotFound = -32601
	ErrInvalidParams  = -32602
	ErrInternal       = -32603
)

// RPCError represents a JSON-RPC error object.
type RPCError struct {
	Code    int         `json:"code"`
	Message string      `json:"message"`
	Data    interface{} `json:"data,omitempty"`
}

// RPCMessage is a generic JSON-RPC envelope used for requests, responses, and notifications.
type RPCMessage struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Method  string          `json:"method,omitempty"`
	Params  json.RawMessage `json:"params,omitempty"`
	Result  json.RawMessage `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// RPCResponse is a JSON-RPC response payload.
type RPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id,omitempty"`
	Result  interface{}     `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

// NewErrorResponse creates an error response for the given request id.
func NewErrorResponse(id json.RawMessage, code int, message string) *RPCResponse {
	return &RPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Error: &RPCError{
			Code:    code,
			Message: message,
		},
	}
}

// NewResultResponse creates a result response for the given request id.
func NewResultResponse(id json.RawMessage, result interface{}) *RPCResponse {
	return &RPCResponse{
		JSONRPC: JSONRPCVersion,
		ID:      id,
		Result:  result,
	}
}
