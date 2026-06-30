package acp

import (
	"bufio"
	"bytes"
	"fmt"
	"io"
)

const (
	// MaxMessageSize caps ACP stdio messages to 1MB.
	MaxMessageSize = 1024 * 1024
)

// ReadLineMessage reads a single JSON-RPC message delimited by a newline.
func ReadLineMessage(r *bufio.Reader) ([]byte, error) {
	for {
		line, err := r.ReadBytes('\n')
		if err != nil && err != io.EOF {
			return nil, fmt.Errorf("read message: %w", err)
		}
		if len(line) == 0 {
			return nil, err
		}

		if line[len(line)-1] == '\n' {
			line = line[:len(line)-1]
		}
		if len(line) > 0 && line[len(line)-1] == '\r' {
			line = line[:len(line)-1]
		}

		trimmed := bytes.TrimSpace(line)
		if len(trimmed) == 0 {
			if err == io.EOF {
				return nil, err
			}
			continue
		}

		if bytes.Contains(trimmed, []byte{'\n'}) {
			return nil, fmt.Errorf("message contains embedded newline")
		}
		if len(trimmed) > MaxMessageSize {
			return nil, fmt.Errorf("message length %d exceeds maximum %d", len(trimmed), MaxMessageSize)
		}

		return trimmed, nil
	}
}

// WriteLineMessage writes a single JSON-RPC message followed by a newline.
func WriteLineMessage(w io.Writer, data []byte) error {
	if len(data) == 0 {
		return fmt.Errorf("message is empty")
	}
	if bytes.Contains(data, []byte{'\n'}) {
		return fmt.Errorf("message contains embedded newline")
	}
	if len(data) > MaxMessageSize {
		return fmt.Errorf("message length %d exceeds maximum %d", len(data), MaxMessageSize)
	}

	if _, err := w.Write(append(data, '\n')); err != nil {
		return fmt.Errorf("write message: %w", err)
	}

	return nil
}
