package acp

import (
	"bufio"
	"bytes"
	"testing"
)

func TestReadLineMessage(t *testing.T) {
	input := []byte("{\"jsonrpc\":\"2.0\"}\n")
	reader := bufio.NewReader(bytes.NewReader(input))

	msg, err := ReadLineMessage(reader)
	if err != nil {
		t.Fatalf("ReadLineMessage error: %v", err)
	}

	if string(msg) != "{\"jsonrpc\":\"2.0\"}" {
		t.Fatalf("unexpected message: %s", string(msg))
	}
}

func TestWriteLineMessageRejectsNewline(t *testing.T) {
	var buf bytes.Buffer
	data := []byte("{\n}")

	if err := WriteLineMessage(&buf, data); err == nil {
		t.Fatalf("expected error for embedded newline")
	}
}
