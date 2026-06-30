// Package tools implements MCP tool handlers.
package tools

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"strings"
	"time"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/types"
	"github.com/tldw/tldw-agent/internal/workspace"
)

// FSTools implements filesystem-related MCP tools.
type FSTools struct {
	config  *config.Config
	session *workspace.Session
}

// NewFSTools creates a new FSTools instance.
func NewFSTools(cfg *config.Config, session *workspace.Session) *FSTools {
	return &FSTools{
		config:  cfg,
		session: session,
	}
}

// FileEntry represents a file or directory entry.
type FileEntry struct {
	Name    string    `json:"name"`
	Type    string    `json:"type"` // "file" or "directory"
	Size    int64     `json:"size,omitempty"`
	ModTime time.Time `json:"mtime,omitempty"`
}

// List lists directory contents.
func (t *FSTools) List(args map[string]interface{}) (*types.ToolResult, error) {
	// Parse arguments
	path, _ := args["path"].(string)
	if path == "" {
		path = "."
	}

	depth := 1
	if d, ok := args["depth"].(float64); ok {
		depth = int(d)
	}

	includeHidden := false
	if h, ok := args["include_hidden"].(bool); ok {
		includeHidden = h
	}

	maxEntries := 1000
	if m, ok := args["max_entries"].(float64); ok {
		maxEntries = int(m)
	}

	// Resolve path
	absPath, err := t.session.ResolvePath(path)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: err.Error(),
		}, nil
	}

	// List entries
	entries := []FileEntry{}
	truncated := false

	err = t.walkDir(absPath, depth, includeHidden, maxEntries, &entries, &truncated)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to list directory: %v", err),
		}, nil
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"path":      path,
			"entries":   entries,
			"truncated": truncated,
		},
	}, nil
}

// walkDir recursively lists directory contents.
func (t *FSTools) walkDir(root string, maxDepth int, includeHidden bool, maxEntries int, entries *[]FileEntry, truncated *bool) error {
	return filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil // Skip entries we can't access
		}

		// Skip the root itself
		if path == root {
			return nil
		}

		// Check max entries
		if len(*entries) >= maxEntries {
			*truncated = true
			return filepath.SkipAll
		}

		// Calculate depth
		rel, _ := filepath.Rel(root, path)
		depth := strings.Count(rel, string(filepath.Separator)) + 1

		// Skip if too deep
		if depth > maxDepth {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Skip hidden files if not included
		name := d.Name()
		if !includeHidden && strings.HasPrefix(name, ".") {
			if d.IsDir() {
				return filepath.SkipDir
			}
			return nil
		}

		// Get file info
		info, err := d.Info()
		if err != nil {
			return nil // Skip entries we can't stat
		}

		entryType := "file"
		if d.IsDir() {
			entryType = "directory"
		}

		*entries = append(*entries, FileEntry{
			Name:    rel,
			Type:    entryType,
			Size:    info.Size(),
			ModTime: info.ModTime(),
		})

		return nil
	})
}

// Read reads file contents.
func (t *FSTools) Read(args map[string]interface{}) (*types.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "path is required",
		}, nil
	}

	// Resolve path
	absPath, err := t.session.ResolvePath(path)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: err.Error(),
		}, nil
	}

	// Check file size
	info, err := os.Stat(absPath)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to stat file: %v", err),
		}, nil
	}

	if info.IsDir() {
		return &types.ToolResult{
			OK:    false,
			Error: "path is a directory, not a file",
		}, nil
	}

	if info.Size() > t.config.Workspace.MaxFileSizeBytes {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("file too large: %d bytes (max %d)", info.Size(), t.config.Workspace.MaxFileSizeBytes),
		}, nil
	}

	// Parse line range
	startLine := 0
	endLine := 0
	if s, ok := args["start_line"].(float64); ok {
		startLine = int(s)
	}
	if e, ok := args["end_line"].(float64); ok {
		endLine = int(e)
	}

	// Read file
	file, err := os.Open(absPath)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to open file: %v", err),
		}, nil
	}
	defer file.Close()

	var lines []string
	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		if startLine > 0 && lineNum < startLine {
			continue
		}
		if endLine > 0 && lineNum > endLine {
			break
		}
		lines = append(lines, scanner.Text())
	}

	if err := scanner.Err(); err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to read file: %v", err),
		}, nil
	}

	content := strings.Join(lines, "\n")

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"path":       path,
			"content":    content,
			"line_count": lineNum,
			"size":       info.Size(),
		},
	}, nil
}

// Write writes content to a file.
func (t *FSTools) Write(args map[string]interface{}) (*types.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "path is required",
		}, nil
	}

	content, ok := args["content"].(string)
	if !ok {
		return &types.ToolResult{
			OK:    false,
			Error: "content is required",
		}, nil
	}

	// Resolve path
	absPath, err := t.session.ResolvePath(path)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: err.Error(),
		}, nil
	}

	// Ensure parent directory exists
	dir := filepath.Dir(absPath)
	if err := os.MkdirAll(dir, 0755); err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to create parent directory: %v", err),
		}, nil
	}

	// Write file
	if err := os.WriteFile(absPath, []byte(content), 0644); err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to write file: %v", err),
		}, nil
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"path":       path,
			"bytes":      len(content),
			"line_count": strings.Count(content, "\n") + 1,
		},
	}, nil
}

// ApplyPatch applies a unified diff patch.
func (t *FSTools) ApplyPatch(args map[string]interface{}) (*types.ToolResult, error) {
	patch, ok := args["patch"].(string)
	if !ok || patch == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "patch is required",
		}, nil
	}

	// Parse the unified diff
	// For now, return a placeholder - full implementation in Phase 2
	return &types.ToolResult{
		OK:    false,
		Error: "fs.apply_patch not yet fully implemented",
	}, nil
}

// Mkdir creates a directory.
func (t *FSTools) Mkdir(args map[string]interface{}) (*types.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "path is required",
		}, nil
	}

	// Resolve path
	absPath, err := t.session.ResolvePath(path)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: err.Error(),
		}, nil
	}

	// Create directory
	if err := os.MkdirAll(absPath, 0755); err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to create directory: %v", err),
		}, nil
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"path":    path,
			"created": true,
		},
	}, nil
}

// Delete deletes a file or directory.
func (t *FSTools) Delete(args map[string]interface{}) (*types.ToolResult, error) {
	path, ok := args["path"].(string)
	if !ok || path == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "path is required",
		}, nil
	}

	recursive := false
	if r, ok := args["recursive"].(bool); ok {
		recursive = r
	}

	// Resolve path
	absPath, err := t.session.ResolvePath(path)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: err.Error(),
		}, nil
	}

	// Check if path exists
	info, err := os.Stat(absPath)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("path does not exist: %v", err),
		}, nil
	}

	// Delete
	if info.IsDir() && recursive {
		if err := os.RemoveAll(absPath); err != nil {
			return &types.ToolResult{
				OK:    false,
				Error: fmt.Sprintf("failed to delete directory: %v", err),
			}, nil
		}
	} else {
		if err := os.Remove(absPath); err != nil {
			return &types.ToolResult{
				OK:    false,
				Error: fmt.Sprintf("failed to delete: %v", err),
			}, nil
		}
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"path":    path,
			"deleted": true,
		},
	}, nil
}
