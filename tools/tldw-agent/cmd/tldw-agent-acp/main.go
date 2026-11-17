package main

import (
	"errors"
	"flag"
	"log"
	"os"

	"github.com/tldw/tldw-agent/internal/acp"
	"github.com/tldw/tldw-agent/internal/config"
)

func main() {
	log.SetOutput(os.Stderr)
	log.SetFlags(log.Ldate | log.Ltime | log.Lshortfile)

	configPath := flag.String("config", "", "Runner YAML configuration file (defaults to host configuration)")
	flag.Parse()
	explicit := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "config" {
			explicit = true
		}
	})
	cfg, err := loadRunnerConfig(*configPath, explicit)
	if err != nil {
		if explicit {
			log.Fatal("Could not read explicit runner config; verify --config names readable valid YAML")
		}
		log.Printf("Warning: Could not load config, using defaults: %v", err)
		cfg = config.Default()
	}

	runner := acp.NewRunner(cfg)
	if err := runner.Run(os.Stdin, os.Stdout); err != nil {
		log.Fatalf("ACP runner error: %v", err)
	}
}

func loadRunnerConfig(path string, explicit bool) (*config.Config, error) {
	if !explicit {
		return config.Load()
	}
	if path == "" {
		return nil, errors.New("explicit runner config path must not be empty")
	}
	return config.LoadFromRequired(path)
}
