package main

import (
	"log"
	"os"

	"github.com/tldw/tldw-agent/internal/acp"
	"github.com/tldw/tldw-agent/internal/config"
)

func main() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	cfg, err := config.Load()
	if err != nil {
		log.Printf("Warning: Could not load config, using defaults: %v", err)
		cfg = config.Default()
	}

	runner := acp.NewRunner(cfg)
	if err := runner.Run(os.Stdin, os.Stdout); err != nil {
		log.Fatalf("ACP runner error: %v", err)
	}
}
