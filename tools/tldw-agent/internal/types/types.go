// Package types provides shared types for the tldw-agent.
package types

// ToolResult represents the result of a tool execution.
type ToolResult struct {
	OK    bool        `json:"ok"`
	Data  interface{} `json:"data,omitempty"`
	Error string      `json:"error,omitempty"`
}
