# Contributing to tldw-agent

Thank you for your interest in contributing to tldw-agent! This document provides guidelines and information for contributors.

## Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Project Structure](#project-structure)
- [Code Guidelines](#code-guidelines)
- [Testing](#testing)
- [Adding New Tools](#adding-new-tools)
- [Pull Request Process](#pull-request-process)

## Getting Started

### Prerequisites

- Go 1.21 or later
- Git
- A modern IDE (VSCode with Go extension, GoLand, etc.)
- Chrome or Firefox for testing

### Quick Start

```bash
# Clone the repository
git clone https://github.com/tldw/tldw-agent.git
cd tldw-agent

# Install dependencies
go mod download

# Run tests
go test ./...

# Build
go build -o bin/tldw-agent-host ./cmd/tldw-agent-host
```

## Development Setup

### IDE Configuration

**VSCode:**
1. Install the Go extension
2. Open the project folder
3. VSCode will prompt to install Go tools - accept all

**GoLand:**
1. Open the project
2. Go to Settings > Go > GOROOT and verify Go is detected
3. Go to Settings > Go > Go Modules and ensure it's enabled

### Environment Setup

Create a test configuration:

```bash
mkdir -p ~/.tldw-agent
cat > ~/.tldw-agent/config.yaml << EOF
server:
  llm_endpoint: "http://localhost:8000"

workspace:
  blocked_paths:
    - ".env"
    - "*.pem"

execution:
  enabled: true
  timeout_ms: 30000
  shell: "auto"

logging:
  level: "debug"
EOF
```

### Running Locally

For development, you can run the agent directly:

```bash
# Build and run
go run ./cmd/tldw-agent-host

# Or with debug logging
TLDW_AGENT_LOG_LEVEL=debug go run ./cmd/tldw-agent-host
```

Note: Native messaging requires running through the browser. For standalone testing, use unit tests.

## Project Structure

```
tldw-agent/
├── cmd/
│   └── tldw-agent-host/
│       └── main.go              # Entry point
│
├── internal/
│   ├── config/
│   │   └── config.go            # Configuration loading
│   │
│   ├── mcp/
│   │   ├── server.go            # MCP JSON-RPC server
│   │   ├── protocol.go          # Message types
│   │   └── tools/
│   │       ├── fs.go            # Filesystem tools
│   │       ├── search.go        # Search tools (grep, glob)
│   │       ├── git.go           # Git operations
│   │       └── exec.go          # Command execution
│   │
│   ├── native/
│   │   ├── framing.go           # Native messaging protocol
│   │   └── handler.go           # Message dispatch
│   │
│   └── workspace/
│       ├── session.go           # Workspace session state
│       └── validator.go         # Path validation
│
├── scripts/
│   ├── install-chrome.sh        # Chrome installer
│   ├── install-firefox.sh       # Firefox installer
│   └── install-windows.ps1      # Windows installer
│
├── docs/
│   └── CONTRIBUTING.md          # This file
│
├── go.mod
├── go.sum
└── README.md
```

### Package Responsibilities

| Package | Purpose |
|---------|---------|
| `cmd/tldw-agent-host` | Main entry point, CLI flags |
| `internal/config` | YAML config loading, defaults |
| `internal/mcp` | MCP protocol server |
| `internal/mcp/tools` | Tool implementations |
| `internal/native` | Native messaging framing |
| `internal/workspace` | Session and path validation |

## Code Guidelines

### Go Style

- Follow standard Go conventions
- Use `gofmt` for formatting
- Run `go vet` before committing
- Use meaningful variable names

### Error Handling

Always handle errors explicitly:

```go
// Good
result, err := doSomething()
if err != nil {
    return nil, fmt.Errorf("failed to do something: %w", err)
}

// Bad
result, _ := doSomething()  // Don't ignore errors
```

### Documentation

Write godoc comments for all exported functions:

```go
// Read reads a file from the workspace.
// It validates the path is within the workspace root and not blocked.
// Returns the file content as a string or an error.
func (f *FSTools) Read(args map[string]interface{}) (*mcp.ToolResult, error) {
    // ...
}
```

### Security

When writing code that handles user input:

1. **Validate all paths** against workspace root
2. **Sanitize arguments** before passing to external commands
3. **Limit resource usage** (file sizes, timeouts)
4. **Never execute arbitrary code**

```go
// Always validate paths
absPath, err := t.session.ValidatePath(path)
if err != nil {
    return &mcp.ToolResult{OK: false, Error: err.Error()}, nil
}

// Always check command allowlist
cmd, ok := e.commands[commandID]
if !ok {
    return &mcp.ToolResult{OK: false, Error: "command not allowed"}, nil
}
```

## Testing

### Running Tests

```bash
# All tests
go test ./...

# With verbose output
go test -v ./...

# With coverage
go test -cover ./...

# Specific package
go test ./internal/mcp/tools/...

# Specific test
go test -run TestFSRead ./internal/mcp/tools/
```

### Writing Tests

Use table-driven tests for comprehensive coverage:

```go
func TestValidatePath(t *testing.T) {
    tests := []struct {
        name      string
        root      string
        path      string
        wantErr   bool
        errContains string
    }{
        {
            name:    "valid relative path",
            root:    "/workspace",
            path:    "src/main.go",
            wantErr: false,
        },
        {
            name:      "path traversal attempt",
            root:      "/workspace",
            path:      "../etc/passwd",
            wantErr:   true,
            errContains: "escapes workspace",
        },
        {
            name:      "blocked path",
            root:      "/workspace",
            path:      ".env",
            wantErr:   true,
            errContains: "blocked",
        },
    }

    for _, tt := range tests {
        t.Run(tt.name, func(t *testing.T) {
            session := NewSession(tt.root)
            _, err := session.ValidatePath(tt.path)

            if tt.wantErr {
                require.Error(t, err)
                if tt.errContains != "" {
                    require.Contains(t, err.Error(), tt.errContains)
                }
            } else {
                require.NoError(t, err)
            }
        })
    }
}
```

### Test Utilities

Use `t.TempDir()` for filesystem tests:

```go
func TestFSWrite(t *testing.T) {
    // Create temporary workspace
    tmpDir := t.TempDir()
    session := workspace.NewSession(tmpDir)
    fs := tools.NewFSTools(session)

    // Test writing
    result, err := fs.Write(map[string]interface{}{
        "path":    "test.txt",
        "content": "hello world",
    })

    require.NoError(t, err)
    require.True(t, result.OK)

    // Verify file was created
    content, err := os.ReadFile(filepath.Join(tmpDir, "test.txt"))
    require.NoError(t, err)
    require.Equal(t, "hello world", string(content))
}
```

## Adding New Tools

### Step 1: Define the Tool

Create a new file in `internal/mcp/tools/` or add to an existing one:

```go
// internal/mcp/tools/my_tool.go
package tools

import (
    "github.com/tldw/tldw-agent/internal/mcp"
    "github.com/tldw/tldw-agent/internal/workspace"
)

// MyTool provides custom functionality.
type MyTool struct {
    session *workspace.Session
}

// NewMyTool creates a new MyTool instance.
func NewMyTool(session *workspace.Session) *MyTool {
    return &MyTool{session: session}
}

// DoSomething performs the tool's main function.
func (t *MyTool) DoSomething(args map[string]interface{}) (*mcp.ToolResult, error) {
    // Extract arguments
    param1, _ := args["param1"].(string)
    if param1 == "" {
        return &mcp.ToolResult{
            OK:    false,
            Error: "param1 is required",
        }, nil
    }

    // Do the work
    result := processParam(param1)

    return &mcp.ToolResult{
        OK: true,
        Data: map[string]interface{}{
            "result": result,
        },
    }, nil
}
```

### Step 2: Register the Tool

In `internal/mcp/server.go`:

```go
func (s *Server) registerTools() {
    // Existing tools...

    // Register new tool
    myTool := tools.NewMyTool(s.session)
    s.tools["my.do_something"] = myTool.DoSomething
}
```

### Step 3: Write Tests

```go
// internal/mcp/tools/my_tool_test.go
package tools

import (
    "testing"
    "github.com/stretchr/testify/require"
)

func TestMyToolDoSomething(t *testing.T) {
    tmpDir := t.TempDir()
    session := workspace.NewSession(tmpDir)
    tool := NewMyTool(session)

    t.Run("success case", func(t *testing.T) {
        result, err := tool.DoSomething(map[string]interface{}{
            "param1": "test",
        })

        require.NoError(t, err)
        require.True(t, result.OK)
        require.Equal(t, "expected", result.Data.(map[string]interface{})["result"])
    })

    t.Run("missing param", func(t *testing.T) {
        result, err := tool.DoSomething(map[string]interface{}{})

        require.NoError(t, err)
        require.False(t, result.OK)
        require.Contains(t, result.Error, "param1 is required")
    })
}
```

### Step 4: Update Documentation

Add the tool to README.md:

```markdown
### New Tool Category

| Tool | Description |
|------|-------------|
| `my.do_something` | Does something useful |
```

## Pull Request Process

### Before Submitting

1. **Run all tests**: `go test ./...`
2. **Run linter**: `go vet ./...`
3. **Format code**: `gofmt -w .`
4. **Update documentation** if needed
5. **Write meaningful commit messages**

### Commit Messages

Follow conventional commits:

```
feat: add new grep tool option for case-insensitive search
fix: handle symlinks correctly in path validation
docs: update README with new configuration options
test: add tests for edge cases in fs.write
refactor: extract path validation to separate function
```

### PR Description

Include:
- **What**: Brief description of changes
- **Why**: Motivation for the change
- **How**: Technical approach taken
- **Testing**: How the change was tested

Example:

```markdown
## What
Add support for custom command timeout per command in config

## Why
Some commands (like builds) need longer timeouts than the default 30s

## How
- Added `timeout_ms` field to CustomCommand struct
- Modified exec.Run to use per-command timeout if specified
- Falls back to global timeout if not set

## Testing
- Added unit tests for timeout override
- Manually tested with 60s timeout for `make build`
```

### Review Process

1. Open PR against `main` branch
2. Automated tests must pass
3. At least one maintainer review required
4. Address review feedback
5. Squash and merge

## Questions?

- Open an issue for bugs or feature requests
- Join our Discord for discussions
- Tag maintainers for urgent issues

Thank you for contributing!
