package tools

import (
	"bytes"
	"fmt"
	"os/exec"
	"strings"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/types"
	"github.com/tldw/tldw-agent/internal/workspace"
)

// GitTools implements git-related MCP tools.
type GitTools struct {
	config  *config.Config
	session *workspace.Session
}

// NewGitTools creates a new GitTools instance.
func NewGitTools(cfg *config.Config, session *workspace.Session) *GitTools {
	return &GitTools{
		config:  cfg,
		session: session,
	}
}

// runGit runs a git command in the workspace.
func (t *GitTools) runGit(args ...string) (string, string, error) {
	cwd := t.session.AbsCwd()
	if cwd == "" {
		return "", "", fmt.Errorf("no workspace set")
	}

	cmd := exec.Command("git", args...)
	cmd.Dir = cwd

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	return stdout.String(), stderr.String(), err
}

// Status returns git repository status.
func (t *GitTools) Status(args map[string]interface{}) (*types.ToolResult, error) {
	// Check if we're in a git repo
	stdout, stderr, err := t.runGit("rev-parse", "--is-inside-work-tree")
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("not a git repository: %s", stderr),
		}, nil
	}

	if strings.TrimSpace(stdout) != "true" {
		return &types.ToolResult{
			OK:    false,
			Error: "not inside a git work tree",
		}, nil
	}

	// Get status
	stdout, stderr, err = t.runGit("status", "--porcelain", "-b")
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("git status failed: %s", stderr),
		}, nil
	}

	// Parse status
	lines := strings.Split(strings.TrimSpace(stdout), "\n")

	var branch string
	var staged, modified, untracked []string

	for _, line := range lines {
		if len(line) == 0 {
			continue
		}

		// Branch line starts with ##
		if strings.HasPrefix(line, "##") {
			branch = strings.TrimPrefix(line, "## ")
			continue
		}

		if len(line) < 3 {
			continue
		}

		status := line[:2]
		file := strings.TrimSpace(line[3:])

		// Index status (first char)
		switch status[0] {
		case 'A', 'M', 'D', 'R', 'C':
			staged = append(staged, file)
		}

		// Worktree status (second char)
		switch status[1] {
		case 'M', 'D':
			modified = append(modified, file)
		case '?':
			untracked = append(untracked, file)
		}
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"branch":    branch,
			"staged":    staged,
			"modified":  modified,
			"untracked": untracked,
			"clean":     len(staged) == 0 && len(modified) == 0 && len(untracked) == 0,
		},
	}, nil
}

// Diff shows git diff.
func (t *GitTools) Diff(args map[string]interface{}) (*types.ToolResult, error) {
	gitArgs := []string{"diff"}

	// Check if staged
	if staged, ok := args["staged"].(bool); ok && staged {
		gitArgs = append(gitArgs, "--staged")
	}

	// Add paths if specified
	if paths, ok := args["paths"].([]interface{}); ok {
		gitArgs = append(gitArgs, "--")
		for _, p := range paths {
			if s, ok := p.(string); ok {
				gitArgs = append(gitArgs, s)
			}
		}
	}

	stdout, stderr, err := t.runGit(gitArgs...)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("git diff failed: %s", stderr),
		}, nil
	}

	// Truncate if too large
	diff := stdout
	truncated := false
	maxSize := 100000 // 100KB
	if len(diff) > maxSize {
		diff = diff[:maxSize]
		truncated = true
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"diff":      diff,
			"truncated": truncated,
		},
	}, nil
}

// Log shows recent commits.
func (t *GitTools) Log(args map[string]interface{}) (*types.ToolResult, error) {
	count := 10
	if c, ok := args["count"].(float64); ok {
		count = int(c)
	}

	gitArgs := []string{"log", fmt.Sprintf("-n%d", count), "--pretty=format:%H|%an|%ae|%at|%s"}

	// Add path filter if specified
	if path, ok := args["path"].(string); ok && path != "" {
		gitArgs = append(gitArgs, "--", path)
	}

	stdout, stderr, err := t.runGit(gitArgs...)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("git log failed: %s", stderr),
		}, nil
	}

	// Parse commits
	var commits []map[string]interface{}
	lines := strings.Split(strings.TrimSpace(stdout), "\n")

	for _, line := range lines {
		if line == "" {
			continue
		}

		parts := strings.SplitN(line, "|", 5)
		if len(parts) < 5 {
			continue
		}

		commits = append(commits, map[string]interface{}{
			"hash":         parts[0],
			"author_name":  parts[1],
			"author_email": parts[2],
			"timestamp":    parts[3],
			"message":      parts[4],
		})
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"commits": commits,
			"count":   len(commits),
		},
	}, nil
}

// Branch shows branch information.
func (t *GitTools) Branch(args map[string]interface{}) (*types.ToolResult, error) {
	// Get current branch
	currentBranch, stderr, err := t.runGit("rev-parse", "--abbrev-ref", "HEAD")
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("git rev-parse failed: %s", stderr),
		}, nil
	}
	currentBranch = strings.TrimSpace(currentBranch)

	// Get all branches
	stdout, stderr, err := t.runGit("branch", "-a", "--format=%(refname:short)|%(upstream:short)|%(upstream:track)")
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("git branch failed: %s", stderr),
		}, nil
	}

	var branches []map[string]interface{}
	lines := strings.Split(strings.TrimSpace(stdout), "\n")

	for _, line := range lines {
		if line == "" {
			continue
		}

		parts := strings.Split(line, "|")
		name := parts[0]

		branch := map[string]interface{}{
			"name":    name,
			"current": name == currentBranch,
		}

		if len(parts) > 1 && parts[1] != "" {
			branch["upstream"] = parts[1]
		}
		if len(parts) > 2 && parts[2] != "" {
			branch["tracking"] = parts[2]
		}

		branches = append(branches, branch)
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"current":  currentBranch,
			"branches": branches,
		},
	}, nil
}

// Add stages files for commit.
func (t *GitTools) Add(args map[string]interface{}) (*types.ToolResult, error) {
	paths, ok := args["paths"].([]interface{})
	if !ok || len(paths) == 0 {
		return &types.ToolResult{
			OK:    false,
			Error: "paths is required",
		}, nil
	}

	gitArgs := []string{"add"}
	for _, p := range paths {
		if s, ok := p.(string); ok {
			gitArgs = append(gitArgs, s)
		}
	}

	stdout, stderr, err := t.runGit(gitArgs...)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("git add failed: %s %s", stderr, stdout),
		}, nil
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"staged": paths,
		},
	}, nil
}

// Commit creates a git commit.
func (t *GitTools) Commit(args map[string]interface{}) (*types.ToolResult, error) {
	message, ok := args["message"].(string)
	if !ok || message == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "message is required",
		}, nil
	}

	stdout, stderr, err := t.runGit("commit", "-m", message)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("git commit failed: %s %s", stderr, stdout),
		}, nil
	}

	// Get the commit hash
	hash, _, _ := t.runGit("rev-parse", "HEAD")
	hash = strings.TrimSpace(hash)

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"hash":    hash,
			"message": message,
		},
	}, nil
}
