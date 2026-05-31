package main

import (
	"context"
	"log"
	"os"

	"github.com/tldw/tldw-agent/internal/guest"
)

func main() {
	root := os.Getenv("TLDW_AGENT_GUEST_WORKSPACE_ROOT")
	if root == "" {
		log.Fatal("TLDW_AGENT_GUEST_WORKSPACE_ROOT is required")
	}
	server, err := guest.NewServer(root)
	if err != nil {
		log.Fatalf("guest server init failed: %v", err)
	}
	if guest.ShouldUseVSockMode() {
		client, err := guest.NewVSockClientFromEnv()
		if err != nil {
			log.Fatalf("guest vsock client init failed: %v", err)
		}
		if err := client.Run(context.Background(), server); err != nil {
			log.Fatalf("guest vsock client exited with error: %v", err)
		}
		return
	}
	if err := server.ServeStream(os.Stdin, os.Stdout); err != nil {
		log.Fatalf("guest server exited with error: %v", err)
	}
}
