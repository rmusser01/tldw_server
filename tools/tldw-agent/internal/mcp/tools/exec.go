// Package tools provides MCP tool implementations.
package tools

import (
	"bytes"
	"context"
	"fmt"
	"os/exec"
	"runtime"
	"strings"
	"time"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/types"
	"github.com/tldw/tldw-agent/internal/workspace"
)

// Command represents an allowlisted command (alias for config.CustomCommand).
type Command = config.CustomCommand

// DefaultCommands returns the built-in command allowlist.
func DefaultCommands() []Command {
	if runtime.GOOS == "windows" {
		return defaultWindowsCommands()
	}
	return defaultUnixCommands()
}

func defaultUnixCommands() []Command {
	return []Command{
		// Test commands
		{ID: "pytest", Template: "python -m pytest", Description: "Run Python tests with pytest", Category: "test", AllowArgs: true, MaxArgs: 20},
		{ID: "npm_test", Template: "npm test", Description: "Run npm tests", Category: "test", AllowArgs: true, MaxArgs: 10},
		{ID: "go_test", Template: "go test ./...", Description: "Run Go tests", Category: "test", AllowArgs: true, MaxArgs: 10},
		{ID: "cargo_test", Template: "cargo test", Description: "Run Rust tests", Category: "test", AllowArgs: true, MaxArgs: 10},

		// Lint commands
		{ID: "ruff", Template: "ruff check", Description: "Run Ruff Python linter", Category: "lint", AllowArgs: true, MaxArgs: 10},
		{ID: "eslint", Template: "npx eslint", Description: "Run ESLint", Category: "lint", AllowArgs: true, MaxArgs: 10},
		{ID: "golint", Template: "golangci-lint run", Description: "Run Go linter", Category: "lint", AllowArgs: true, MaxArgs: 10},

		// Format commands
		{ID: "prettier", Template: "npx prettier --write", Description: "Format with Prettier", Category: "format", AllowArgs: true, MaxArgs: 20},
		{ID: "black", Template: "black", Description: "Format Python with Black", Category: "format", AllowArgs: true, MaxArgs: 20},
		{ID: "gofmt", Template: "go fmt ./...", Description: "Format Go code", Category: "format", AllowArgs: true, MaxArgs: 5},

		// Package commands
		{ID: "npm_install", Template: "npm install", Description: "Install npm dependencies", Category: "package", AllowArgs: false},
		{ID: "pip_install", Template: "pip install -r requirements.txt", Description: "Install Python dependencies", Category: "package", AllowArgs: false},
		{ID: "go_mod_tidy", Template: "go mod tidy", Description: "Tidy Go modules", Category: "package", AllowArgs: false},
	}
}

func defaultWindowsCommands() []Command {
	return []Command{
		// Test commands
		{ID: "pytest", Template: "python -m pytest", Description: "Run Python tests with pytest", Category: "test", AllowArgs: true, MaxArgs: 20},
		{ID: "npm_test", Template: "npm test", Description: "Run npm tests", Category: "test", AllowArgs: true, MaxArgs: 10},
		{ID: "go_test", Template: "go test ./...", Description: "Run Go tests", Category: "test", AllowArgs: true, MaxArgs: 10},
		{ID: "cargo_test", Template: "cargo test", Description: "Run Rust tests", Category: "test", AllowArgs: true, MaxArgs: 10},
		{ID: "dotnet_test", Template: "dotnet test", Description: "Run .NET tests", Category: "test", AllowArgs: true, MaxArgs: 10},

		// Lint commands
		{ID: "eslint", Template: "npx eslint", Description: "Run ESLint", Category: "lint", AllowArgs: true, MaxArgs: 10},

		// Format commands
		{ID: "prettier", Template: "npx prettier --write", Description: "Format with Prettier", Category: "format", AllowArgs: true, MaxArgs: 20},

		// Package commands
		{ID: "npm_install", Template: "npm install", Description: "Install npm dependencies", Category: "package", AllowArgs: false},
		{ID: "pip_install", Template: "pip install -r requirements.txt", Description: "Install Python dependencies", Category: "package", AllowArgs: false},
		{ID: "nuget_restore", Template: "nuget restore", Description: "Restore NuGet packages", Category: "package", AllowArgs: true, MaxArgs: 5},
	}
}

// ExecTools provides command execution tools.
type ExecTools struct {
	config   *config.Config
	session  *workspace.Session
	commands map[string]Command
}

// NewExecTools creates a new ExecTools instance.
func NewExecTools(cfg *config.Config, session *workspace.Session) *ExecTools {
	commands := make(map[string]Command)
	for _, cmd := range DefaultCommands() {
		commands[cmd.ID] = cmd
	}

	// Add custom commands from config
	for _, cmd := range cfg.Execution.CustomCommands {
		commands[cmd.ID] = cmd
	}

	return &ExecTools{
		config:   cfg,
		session:  session,
		commands: commands,
	}
}

// ExecResult represents the result of a command execution.
type ExecResult struct {
	ExitCode   int    `json:"exit_code"`
	Stdout     string `json:"stdout"`
	Stderr     string `json:"stderr"`
	DurationMs int64  `json:"duration_ms"`
	Truncated  bool   `json:"truncated"`
}

// Run executes an allowlisted command.
func (e *ExecTools) Run(args map[string]interface{}) (*types.ToolResult, error) {
	// Check if execution is enabled
	if !e.config.Execution.Enabled {
		return &types.ToolResult{
			OK:    false,
			Error: "command execution is disabled",
		}, nil
	}

	// Get command ID
	commandID, _ := args["command_id"].(string)
	if commandID == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "command_id is required",
		}, nil
	}

	// Look up command in allowlist
	cmd, ok := e.commands[commandID]
	if !ok {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("command %q not in allowlist", commandID),
		}, nil
	}

	// Get optional arguments
	var cmdArgs []string
	if argsRaw, ok := args["args"].([]interface{}); ok && cmd.AllowArgs {
		for _, a := range argsRaw {
			if s, ok := a.(string); ok {
				// Sanitize argument - reject shell metacharacters
				if containsShellMeta(s) {
					return &types.ToolResult{
						OK:    false,
						Error: fmt.Sprintf("argument %q contains disallowed characters", s),
					}, nil
				}
				cmdArgs = append(cmdArgs, s)
			}
		}

		// Check max args
		if cmd.MaxArgs > 0 && len(cmdArgs) > cmd.MaxArgs {
			return &types.ToolResult{
				OK:    false,
				Error: fmt.Sprintf("too many arguments (max %d)", cmd.MaxArgs),
			}, nil
		}
	}

	// Get working directory
	cwd := e.session.Root()
	if cwdArg, ok := args["cwd"].(string); ok && cwdArg != "" {
		// Validate and resolve path within workspace
		absPath, err := e.session.ResolvePath(cwdArg)
		if err != nil {
			return &types.ToolResult{
				OK:    false,
				Error: fmt.Sprintf("invalid cwd: %v", err),
			}, nil
		}
		cwd = absPath
	}

	// Get timeout
	timeout := time.Duration(e.config.Execution.TimeoutMs) * time.Millisecond
	if timeoutArg, ok := args["timeout_ms"].(float64); ok && timeoutArg > 0 {
		timeout = time.Duration(timeoutArg) * time.Millisecond
		// Cap at configured max
		maxTimeout := time.Duration(e.config.Execution.TimeoutMs) * time.Millisecond
		if timeout > maxTimeout {
			timeout = maxTimeout
		}
	}

	// Build the command
	fullCmd := cmd.Template
	if len(cmdArgs) > 0 {
		fullCmd = fullCmd + " " + strings.Join(cmdArgs, " ")
	}

	// Execute
	result, err := e.executeCommand(fullCmd, cwd, timeout, cmd.Env)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: err.Error(),
		}, nil
	}

	return &types.ToolResult{
		OK:   true,
		Data: result,
	}, nil
}

// ListCommands returns all available commands.
func (e *ExecTools) ListCommands() []Command {
	result := make([]Command, 0, len(e.commands))
	for _, cmd := range e.commands {
		result = append(result, cmd)
	}
	return result
}

func (e *ExecTools) executeCommand(cmdStr, cwd string, timeout time.Duration, env []string) (*ExecResult, error) {
	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	var cmd *exec.Cmd

	// Use appropriate shell based on OS
	if runtime.GOOS == "windows" {
		shell := e.config.Execution.Shell
		if shell == "auto" || shell == "" {
			shell = "powershell"
		}

		switch shell {
		case "powershell":
			cmd = exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", cmdStr)
		case "cmd":
			cmd = exec.CommandContext(ctx, "cmd", "/c", cmdStr)
		default:
			cmd = exec.CommandContext(ctx, shell, "-c", cmdStr)
		}
	} else {
		shell := e.config.Execution.Shell
		if shell == "auto" || shell == "" {
			shell = "sh"
		}
		cmd = exec.CommandContext(ctx, shell, "-c", cmdStr)
	}

	cmd.Dir = cwd

	// Set environment
	if len(env) > 0 {
		cmd.Env = append(cmd.Env, env...)
	}

	// Capture output
	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	start := time.Now()
	err := cmd.Run()
	duration := time.Since(start)

	result := &ExecResult{
		DurationMs: duration.Milliseconds(),
		Truncated:  false,
	}

	// Get exit code
	if err != nil {
		if exitErr, ok := err.(*exec.ExitError); ok {
			result.ExitCode = exitErr.ExitCode()
		} else if ctx.Err() == context.DeadlineExceeded {
			result.ExitCode = -1
			result.Stderr = "command timed out"
			return result, nil
		} else {
			return nil, fmt.Errorf("failed to execute command: %w", err)
		}
	}

	// Get output, truncating if too large
	maxOutput := e.config.Execution.MaxOutputBytes
	if maxOutput <= 0 {
		maxOutput = 1024 * 1024 // 1MB default
	}
	stdoutBytes := stdout.Bytes()
	stderrBytes := stderr.Bytes()

	if len(stdoutBytes) > maxOutput {
		stdoutBytes = stdoutBytes[:maxOutput]
		result.Truncated = true
	}
	if len(stderrBytes) > maxOutput {
		stderrBytes = stderrBytes[:maxOutput]
		result.Truncated = true
	}

	result.Stdout = string(stdoutBytes)
	result.Stderr = string(stderrBytes)

	return result, nil
}

// containsShellMeta checks if a string contains shell metacharacters.
func containsShellMeta(s string) bool {
	// List of dangerous shell metacharacters
	metaChars := []string{
		";", "&", "|", "`", "$", "(", ")", "{", "}", "<", ">",
		"'", "\"", "\\", "\n", "\r",
	}

	for _, meta := range metaChars {
		if strings.Contains(s, meta) {
			return true
		}
	}

	return false
}
