package main

import (
	"log"
	"os"

	"github.com/tldw/tldw-agent/internal/guest"
)

func main() {
	root := os.Getenv("TLDW_AGENT_GUEST_WORKSPACE_ROOT")
	if root == "" {
		log.Fatal("TLDW_AGENT_GUEST_WORKSPACE_ROOT is required")
	}
	if _, err := guest.NewServer(root); err != nil {
		log.Fatalf("guest server init failed: %v", err)
	}
}
