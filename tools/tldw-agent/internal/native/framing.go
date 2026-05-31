// Package native implements the Chrome/Firefox native messaging protocol.
// The protocol uses length-prefixed JSON messages over stdin/stdout.
package native

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
)

const (
	// MaxMessageSize is the maximum allowed message size (1MB)
	MaxMessageSize = 1024 * 1024
)

// ReadMessage reads a native messaging message from the reader.
// The format is: 4-byte little-endian length prefix + JSON body.
func ReadMessage(r io.Reader) ([]byte, error) {
	// Read the 4-byte length prefix
	var length uint32
	if err := binary.Read(r, binary.LittleEndian, &length); err != nil {
		if err == io.EOF {
			return nil, err
		}
		return nil, fmt.Errorf("failed to read message length: %w", err)
	}

	// Validate length
	if length == 0 {
		return nil, fmt.Errorf("message length is zero")
	}
	if length > MaxMessageSize {
		return nil, fmt.Errorf("message length %d exceeds maximum %d", length, MaxMessageSize)
	}

	// Read the message body
	body := make([]byte, length)
	if _, err := io.ReadFull(r, body); err != nil {
		return nil, fmt.Errorf("failed to read message body: %w", err)
	}

	return body, nil
}

// WriteMessage writes a native messaging message to the writer.
// The format is: 4-byte little-endian length prefix + JSON body.
func WriteMessage(w io.Writer, data []byte) error {
	// Validate length
	length := uint32(len(data))
	if length > MaxMessageSize {
		return fmt.Errorf("message length %d exceeds maximum %d", length, MaxMessageSize)
	}

	// Write the 4-byte length prefix
	if err := binary.Write(w, binary.LittleEndian, length); err != nil {
		return fmt.Errorf("failed to write message length: %w", err)
	}

	// Write the message body
	if _, err := w.Write(data); err != nil {
		return fmt.Errorf("failed to write message body: %w", err)
	}

	return nil
}

// ReadJSON reads and unmarshals a JSON message from the reader.
func ReadJSON(r io.Reader, v interface{}) error {
	data, err := ReadMessage(r)
	if err != nil {
		return err
	}

	if err := json.Unmarshal(data, v); err != nil {
		return fmt.Errorf("failed to unmarshal message: %w", err)
	}

	return nil
}

// WriteJSON marshals and writes a JSON message to the writer.
func WriteJSON(w io.Writer, v interface{}) error {
	data, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("failed to marshal message: %w", err)
	}

	return WriteMessage(w, data)
}
