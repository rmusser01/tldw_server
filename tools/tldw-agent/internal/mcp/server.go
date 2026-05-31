// Package mcp implements the MCP (Model Context Protocol) server for workspace tools.
package mcp

import (
	"encoding/json"
	"fmt"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/mcp/tools"
	"github.com/tldw/tldw-agent/internal/types"
	"github.com/tldw/tldw-agent/internal/workspace"
)

// ToolDefinition describes an available tool.
type ToolDefinition struct {
	Name        string                 `json:"name"`
	Description string                 `json:"description"`
	Parameters  map[string]interface{} `json:"parameters"`
	Tier        string                 `json:"tier"` // "read", "write", "exec"
}

// ToolResult is an alias for types.ToolResult for convenience.
type ToolResult = types.ToolResult

// Server manages MCP tools and executes tool calls.
type Server struct {
	config      *config.Config
	session     *workspace.Session
	fsTools     *tools.FSTools
	gitTools    *tools.GitTools
	searchTools *tools.SearchTools
	execTools   *tools.ExecTools
}

// NewServer creates a new MCP server.
func NewServer(cfg *config.Config) *Server {
	session := workspace.NewSession(cfg)

	return &Server{
		config:      cfg,
		session:     session,
		fsTools:     tools.NewFSTools(cfg, session),
		gitTools:    tools.NewGitTools(cfg, session),
		searchTools: tools.NewSearchTools(cfg, session),
		execTools:   tools.NewExecTools(cfg, session),
	}
}

// ListTools returns all available tool definitions.
func (s *Server) ListTools() []ToolDefinition {
	return []ToolDefinition{
		// Tier 0: Navigation & Read (auto-approve)
		{
			Name:        "workspace.list",
			Description: "List all registered workspaces",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		},
		{
			Name:        "workspace.pwd",
			Description: "Get current working directory",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		},
		{
			Name:        "workspace.chdir",
			Description: "Change current working directory",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to change to (relative to workspace root)",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "fs.list",
			Description: "List directory contents",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Directory path to list",
					},
					"depth": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum depth to recurse (default 1)",
						"default":     1,
					},
					"include_hidden": map[string]interface{}{
						"type":        "boolean",
						"description": "Include hidden files",
						"default":     false,
					},
					"max_entries": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum entries to return",
						"default":     1000,
					},
				},
			},
		},
		{
			Name:        "fs.read",
			Description: "Read file contents",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path to read",
					},
					"start_line": map[string]interface{}{
						"type":        "integer",
						"description": "Starting line number (1-indexed)",
					},
					"end_line": map[string]interface{}{
						"type":        "integer",
						"description": "Ending line number (inclusive)",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "search.grep",
			Description: "Search file contents using regex pattern",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "Search pattern (regex)",
					},
					"paths": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Paths to search in",
					},
					"glob": map[string]interface{}{
						"type":        "string",
						"description": "File glob pattern (e.g., *.go)",
					},
					"case_sensitive": map[string]interface{}{
						"type":        "boolean",
						"description": "Case sensitive search",
						"default":     true,
					},
					"max_results": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum results to return",
						"default":     100,
					},
				},
				"required": []string{"pattern"},
			},
		},
		{
			Name:        "search.glob",
			Description: "Find files matching a glob pattern",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"pattern": map[string]interface{}{
						"type":        "string",
						"description": "Glob pattern to match",
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Base path to search from",
					},
					"max_results": map[string]interface{}{
						"type":        "integer",
						"description": "Maximum results to return",
						"default":     100,
					},
				},
				"required": []string{"pattern"},
			},
		},
		{
			Name:        "git.status",
			Description: "Get git repository status",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		},
		{
			Name:        "git.diff",
			Description: "Show git diff",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"paths": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Paths to diff",
					},
					"staged": map[string]interface{}{
						"type":        "boolean",
						"description": "Show staged changes",
						"default":     false,
					},
				},
			},
		},
		{
			Name:        "git.log",
			Description: "Show recent commits",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"count": map[string]interface{}{
						"type":        "integer",
						"description": "Number of commits to show",
						"default":     10,
					},
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Filter by path",
					},
				},
			},
		},
		{
			Name:        "git.branch",
			Description: "Show branch information",
			Tier:        "read",
			Parameters: map[string]interface{}{
				"type":       "object",
				"properties": map[string]interface{}{},
			},
		},
		// Tier 1: Editing (requires approval)
		{
			Name:        "fs.write",
			Description: "Write content to a file",
			Tier:        "write",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "File path to write",
					},
					"content": map[string]interface{}{
						"type":        "string",
						"description": "Content to write",
					},
				},
				"required": []string{"path", "content"},
			},
		},
		{
			Name:        "fs.apply_patch",
			Description: "Apply a unified diff patch",
			Tier:        "write",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"patch": map[string]interface{}{
						"type":        "string",
						"description": "Unified diff to apply",
					},
				},
				"required": []string{"patch"},
			},
		},
		{
			Name:        "fs.mkdir",
			Description: "Create a directory",
			Tier:        "write",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Directory path to create",
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "fs.delete",
			Description: "Delete a file or directory",
			Tier:        "write",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"path": map[string]interface{}{
						"type":        "string",
						"description": "Path to delete",
					},
					"recursive": map[string]interface{}{
						"type":        "boolean",
						"description": "Recursively delete directories",
						"default":     false,
					},
				},
				"required": []string{"path"},
			},
		},
		{
			Name:        "git.add",
			Description: "Stage files for commit",
			Tier:        "write",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"paths": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Paths to stage",
					},
				},
				"required": []string{"paths"},
			},
		},
		{
			Name:        "git.commit",
			Description: "Create a git commit",
			Tier:        "write",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"message": map[string]interface{}{
						"type":        "string",
						"description": "Commit message",
					},
				},
				"required": []string{"message"},
			},
		},
		// Tier 2: Execution (requires explicit approval)
		{
			Name:        "exec.run",
			Description: "Run an allowlisted command",
			Tier:        "exec",
			Parameters: map[string]interface{}{
				"type": "object",
				"properties": map[string]interface{}{
					"command_id": map[string]interface{}{
						"type":        "string",
						"description": "ID of the allowlisted command (e.g., pytest, npm_test)",
					},
					"args": map[string]interface{}{
						"type":        "array",
						"items":       map[string]interface{}{"type": "string"},
						"description": "Additional arguments",
					},
					"cwd": map[string]interface{}{
						"type":        "string",
						"description": "Working directory (relative to workspace)",
					},
					"timeout_ms": map[string]interface{}{
						"type":        "integer",
						"description": "Timeout in milliseconds",
					},
				},
				"required": []string{"command_id"},
			},
		},
	}
}

// ExecuteTool executes a tool with the given arguments.
func (s *Server) ExecuteTool(toolName string, arguments json.RawMessage) (*ToolResult, error) {
	// Parse arguments into a map
	var args map[string]interface{}
	if len(arguments) > 0 {
		if err := json.Unmarshal(arguments, &args); err != nil {
			return nil, fmt.Errorf("failed to parse arguments: %w", err)
		}
	}

	// Route to appropriate tool handler
	switch toolName {
	// Workspace tools
	case "workspace.list":
		return s.session.List()
	case "workspace.pwd":
		return s.session.Pwd()
	case "workspace.chdir":
		return s.session.Chdir(args)

	// Filesystem tools
	case "fs.list":
		return s.fsTools.List(args)
	case "fs.read":
		return s.fsTools.Read(args)
	case "fs.write":
		return s.fsTools.Write(args)
	case "fs.apply_patch":
		return s.fsTools.ApplyPatch(args)
	case "fs.mkdir":
		return s.fsTools.Mkdir(args)
	case "fs.delete":
		return s.fsTools.Delete(args)

	// Search tools
	case "search.grep":
		return s.searchTools.Grep(args)
	case "search.glob":
		return s.searchTools.Glob(args)

	// Git tools
	case "git.status":
		return s.gitTools.Status(args)
	case "git.diff":
		return s.gitTools.Diff(args)
	case "git.log":
		return s.gitTools.Log(args)
	case "git.branch":
		return s.gitTools.Branch(args)
	case "git.add":
		return s.gitTools.Add(args)
	case "git.commit":
		return s.gitTools.Commit(args)

	// Exec tools
	case "exec.run":
		return s.execTools.Run(args)

	default:
		return nil, fmt.Errorf("unknown tool: %s", toolName)
	}
}

// SetWorkspace sets the current workspace root.
func (s *Server) SetWorkspace(root string) error {
	return s.session.SetRoot(root)
}
