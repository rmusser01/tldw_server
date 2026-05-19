package tools

import (
	"bufio"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"regexp"
	"strings"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/types"
	"github.com/tldw/tldw-agent/internal/workspace"
)

// SearchTools implements search-related MCP tools.
type SearchTools struct {
	config  *config.Config
	session *workspace.Session
}

// NewSearchTools creates a new SearchTools instance.
func NewSearchTools(cfg *config.Config, session *workspace.Session) *SearchTools {
	return &SearchTools{
		config:  cfg,
		session: session,
	}
}

// GrepMatch represents a single grep match.
type GrepMatch struct {
	Path    string `json:"path"`
	Line    int    `json:"line"`
	Column  int    `json:"column"`
	Preview string `json:"preview"`
}

// Grep searches file contents using a regex pattern.
func (t *SearchTools) Grep(args map[string]interface{}) (*types.ToolResult, error) {
	pattern, ok := args["pattern"].(string)
	if !ok || pattern == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "pattern is required",
		}, nil
	}

	// Parse optional arguments
	var searchPaths []string
	if paths, ok := args["paths"].([]interface{}); ok {
		for _, p := range paths {
			if s, ok := p.(string); ok {
				searchPaths = append(searchPaths, s)
			}
		}
	}

	globPattern := ""
	if g, ok := args["glob"].(string); ok {
		globPattern = g
	}

	caseSensitive := true
	if cs, ok := args["case_sensitive"].(bool); ok {
		caseSensitive = cs
	}

	maxResults := 100
	if m, ok := args["max_results"].(float64); ok {
		maxResults = int(m)
	}

	// Compile regex
	regexFlags := ""
	if !caseSensitive {
		regexFlags = "(?i)"
	}
	re, err := regexp.Compile(regexFlags + pattern)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("invalid regex pattern: %v", err),
		}, nil
	}

	// Default to workspace root if no paths specified
	if len(searchPaths) == 0 {
		searchPaths = []string{"."}
	}

	matches := []GrepMatch{}
	filesSearched := 0

	for _, searchPath := range searchPaths {
		absPath, err := t.session.ResolvePath(searchPath)
		if err != nil {
			continue // Skip invalid paths
		}

		err = filepath.WalkDir(absPath, func(path string, d fs.DirEntry, err error) error {
			if err != nil {
				return nil // Skip entries we can't access
			}

			// Skip directories
			if d.IsDir() {
				// Skip hidden directories
				if strings.HasPrefix(d.Name(), ".") {
					return filepath.SkipDir
				}
				// Skip common large directories
				if d.Name() == "node_modules" || d.Name() == "vendor" || d.Name() == "__pycache__" {
					return filepath.SkipDir
				}
				return nil
			}

			// Apply glob filter
			if globPattern != "" {
				matched, _ := filepath.Match(globPattern, d.Name())
				if !matched {
					return nil
				}
			}

			// Skip binary files (simple heuristic)
			if isBinaryFile(d.Name()) {
				return nil
			}

			// Search file
			fileMatches, err := t.searchFile(path, re, maxResults-len(matches))
			if err != nil {
				return nil // Skip files we can't read
			}

			// Convert paths to relative
			root := t.session.Root()
			for i := range fileMatches {
				relPath, _ := filepath.Rel(root, fileMatches[i].Path)
				fileMatches[i].Path = relPath
			}

			matches = append(matches, fileMatches...)
			filesSearched++

			// Stop if we have enough matches
			if len(matches) >= maxResults {
				return filepath.SkipAll
			}

			return nil
		})
		if err != nil && err != filepath.SkipAll {
			continue
		}
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"matches":        matches,
			"total_matches":  len(matches),
			"files_searched": filesSearched,
			"truncated":      len(matches) >= maxResults,
		},
	}, nil
}

// searchFile searches a single file for the pattern.
func (t *SearchTools) searchFile(path string, re *regexp.Regexp, maxMatches int) ([]GrepMatch, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	var matches []GrepMatch
	scanner := bufio.NewScanner(file)
	lineNum := 0

	for scanner.Scan() {
		lineNum++
		line := scanner.Text()

		// Find all matches in line
		locs := re.FindAllStringIndex(line, -1)
		for _, loc := range locs {
			if len(matches) >= maxMatches {
				return matches, nil
			}

			// Truncate preview if too long
			preview := line
			if len(preview) > 200 {
				start := loc[0] - 50
				if start < 0 {
					start = 0
				}
				end := loc[1] + 50
				if end > len(line) {
					end = len(line)
				}
				preview = line[start:end]
				if start > 0 {
					preview = "..." + preview
				}
				if end < len(line) {
					preview = preview + "..."
				}
			}

			matches = append(matches, GrepMatch{
				Path:    path,
				Line:    lineNum,
				Column:  loc[0] + 1, // 1-indexed
				Preview: preview,
			})
		}
	}

	return matches, scanner.Err()
}

// Glob finds files matching a glob pattern.
func (t *SearchTools) Glob(args map[string]interface{}) (*types.ToolResult, error) {
	pattern, ok := args["pattern"].(string)
	if !ok || pattern == "" {
		return &types.ToolResult{
			OK:    false,
			Error: "pattern is required",
		}, nil
	}

	basePath := "."
	if p, ok := args["path"].(string); ok && p != "" {
		basePath = p
	}

	maxResults := 100
	if m, ok := args["max_results"].(float64); ok {
		maxResults = int(m)
	}

	// Resolve base path
	absBasePath, err := t.session.ResolvePath(basePath)
	if err != nil {
		return &types.ToolResult{
			OK:    false,
			Error: err.Error(),
		}, nil
	}

	// Find matching files
	var matches []string
	truncated := false

	err = filepath.WalkDir(absBasePath, func(path string, d fs.DirEntry, err error) error {
		if err != nil {
			return nil
		}

		// Skip hidden directories
		if d.IsDir() && strings.HasPrefix(d.Name(), ".") {
			return filepath.SkipDir
		}

		// Skip common large directories
		if d.IsDir() {
			if d.Name() == "node_modules" || d.Name() == "vendor" || d.Name() == "__pycache__" {
				return filepath.SkipDir
			}
			return nil
		}

		// Check if name matches pattern
		matched, err := filepath.Match(pattern, d.Name())
		if err != nil {
			return nil
		}

		if matched {
			if len(matches) >= maxResults {
				truncated = true
				return filepath.SkipAll
			}

			// Convert to relative path
			root := t.session.Root()
			relPath, _ := filepath.Rel(root, path)
			matches = append(matches, relPath)
		}

		return nil
	})

	if err != nil && err != filepath.SkipAll {
		return &types.ToolResult{
			OK:    false,
			Error: fmt.Sprintf("failed to search: %v", err),
		}, nil
	}

	return &types.ToolResult{
		OK: true,
		Data: map[string]interface{}{
			"pattern":   pattern,
			"matches":   matches,
			"count":     len(matches),
			"truncated": truncated,
		},
	}, nil
}

// isBinaryFile checks if a file is likely binary based on extension.
func isBinaryFile(name string) bool {
	binaryExts := map[string]bool{
		".exe":   true,
		".dll":   true,
		".so":    true,
		".dylib": true,
		".bin":   true,
		".o":     true,
		".a":     true,
		".obj":   true,
		".png":   true,
		".jpg":   true,
		".jpeg":  true,
		".gif":   true,
		".bmp":   true,
		".ico":   true,
		".pdf":   true,
		".zip":   true,
		".tar":   true,
		".gz":    true,
		".7z":    true,
		".rar":   true,
		".woff":  true,
		".woff2": true,
		".ttf":   true,
		".eot":   true,
		".mp3":   true,
		".mp4":   true,
		".avi":   true,
		".mov":   true,
		".wav":   true,
		".flac":  true,
	}

	ext := strings.ToLower(filepath.Ext(name))
	return binaryExts[ext]
}
