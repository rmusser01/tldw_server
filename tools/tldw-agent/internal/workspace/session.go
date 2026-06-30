// Package workspace manages workspace state and path validation.
package workspace

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/types"
)

// Session manages the current workspace state.
type Session struct {
	config *config.Config
	mu     sync.RWMutex
	root   string // Workspace root directory
	cwd    string // Current working directory (relative to root)
}

// NewSession creates a new workspace session.
func NewSession(cfg *config.Config) *Session {
	return &Session{
		config: cfg,
		root:   cfg.Workspace.DefaultRoot,
		cwd:    ".",
	}
}

// SetRoot sets the workspace root directory.
func (s *Session) SetRoot(root string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	// Resolve to absolute path
	absRoot, err := filepath.Abs(root)
	if err != nil {
		return fmt.Errorf("failed to resolve path: %w", err)
	}

	// Verify directory exists
	info, err := os.Stat(absRoot)
	if err != nil {
		return fmt.Errorf("failed to access directory: %w", err)
	}
	if !info.IsDir() {
		return fmt.Errorf("path is not a directory: %s", absRoot)
	}

	s.root = absRoot
	s.cwd = "."
	return nil
}

// Root returns the current workspace root.
func (s *Session) Root() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.root
}

// Cwd returns the current working directory (relative to root).
func (s *Session) Cwd() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.cwd
}

// AbsCwd returns the absolute current working directory.
func (s *Session) AbsCwd() string {
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.root == "" {
		return ""
	}
	return filepath.Join(s.root, s.cwd)
}

// List returns information about registered workspaces.
func (s *Session) List() (*types.ToolResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	workspaces := []map[string]interface{}{}
	if s.root != "" {
		workspaces = append(workspaces, map[string]interface{}{
			"id":   "current",
			"path": s.root,
			"cwd":  s.cwd,
		})
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"workspaces": workspaces,
		},
	}, nil
}

// Pwd returns the current working directory.
func (s *Session) Pwd() (*types.ToolResult, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.root == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "no workspace set",
		}, nil
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"root": s.root,
			"cwd":  s.cwd,
			"abs":  filepath.Join(s.root, s.cwd),
		},
	}, nil
}

// Chdir changes the current working directory.
func (s *Session) Chdir(args map[string]interface{}) (*types.ToolResult, error) {
	pathArg, ok := args["path"].(string)
	if !ok {
		return &types.ToolResult{
			OK:    false,
			Error: "path is required",
		}, nil
	}

	s.mu.Lock()
	defer s.mu.Unlock()

	if s.root == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "no workspace set",
		}, nil
	}

	// Resolve the new path
	var newCwd string
	if filepath.IsAbs(pathArg) {
		newCwd = pathArg
	} else {
		newCwd = filepath.Join(s.cwd, pathArg)
	}

	// Validate the path is within workspace
	absPath := filepath.Join(s.root, newCwd)
	if valid, err := s.validatePathLocked(absPath); !valid {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("invalid path: %v", err),
		}, nil
	}

	// Verify directory exists
	info, err := os.Stat(absPath)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to access directory: %v", err),
		}, nil
	}
	if !info.IsDir() {
		return &types.ToolResult{
			OK:    false,
			Error: "path is not a directory",
		}, nil
	}

	// Clean and set the new cwd
	s.cwd = filepath.Clean(newCwd)

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"cwd": s.cwd,
			"abs": absPath,
		},
	}, nil
}

// ValidatePath checks if a path is within the workspace and not blocked.
func (s *Session) ValidatePath(path string) (bool, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.validatePathLocked(path)
}

// validatePathLocked performs path validation (must hold lock).
func (s *Session) validatePathLocked(path string) (bool, error) {
	if s.root == "" {
		return false, fmt.Errorf("no workspace set")
	}

	// Resolve to absolute path
	var absPath string
	if filepath.IsAbs(path) {
		absPath = path
	} else {
		absPath = filepath.Join(s.root, s.cwd, path)
	}

	// Get real path (resolve symlinks)
	realPath, err := filepath.EvalSymlinks(absPath)
	if err != nil {
		// If file doesn't exist, check parent directory
		if os.IsNotExist(err) {
			parentDir := filepath.Dir(absPath)
			realPath, err = filepath.EvalSymlinks(parentDir)
			if err != nil {
				return false, fmt.Errorf("failed to resolve path: %w", err)
			}
			realPath = filepath.Join(realPath, filepath.Base(absPath))
		} else {
			return false, fmt.Errorf("failed to resolve path: %w", err)
		}
	}

	// Check if path is under workspace root
	realRoot, err := filepath.EvalSymlinks(s.root)
	if err != nil {
		return false, fmt.Errorf("failed to resolve workspace root: %w", err)
	}

	// Ensure realPath starts with realRoot
	rel, err := filepath.Rel(realRoot, realPath)
	if err != nil {
		return false, fmt.Errorf("failed to compute relative path: %w", err)
	}

	// Check for path traversal (relative path starting with ..)
	if strings.HasPrefix(rel, "..") {
		return false, fmt.Errorf("path escapes workspace root")
	}

	// Check blocked paths
	if s.config.IsPathBlocked(realPath) {
		return false, fmt.Errorf("path is blocked by policy")
	}

	return true, nil
}

// ResolvePath resolves a path relative to the workspace.
func (s *Session) ResolvePath(path string) (string, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	if s.root == "" {
		return "", fmt.Errorf("no workspace set")
	}

	var absPath string
	if filepath.IsAbs(path) {
		absPath = path
	} else {
		absPath = filepath.Join(s.root, s.cwd, path)
	}

	// Validate the path
	if valid, err := s.validatePathLocked(absPath); !valid {
		return "", err
	}

	return filepath.Clean(absPath), nil
}
