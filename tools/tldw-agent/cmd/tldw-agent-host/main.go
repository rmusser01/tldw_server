// tldw-agent-host is a native messaging host for the tldw browser extension.
// It provides workspace-aware MCP tools for agentic coding/writing assistance.
package main

import (
	"log"
	"os"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/mcp"
	"github.com/tldw/tldw-agent/internal/native"
)

func main() {
	// Configure logging to stderr (stdout is reserved for native messaging)
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	// Load configuration
	cfg, err := config.Load()
	if err != nil {
		log.Printf("Warning: Could not load config, using defaults: %v", err)
		cfg = config.Default()
	}

	// Create MCP server with workspace tools
	mcpServer := mcp.NewServer(cfg)

	// Create native messaging handler
	handler := native.NewHandler(mcpServer, cfg)

	// Run the native messaging loop (reads from stdin, writes to stdout)
	if err := handler.Run(); err != nil {
		log.Fatalf("Native messaging handler error: %v", err)
	}
}
