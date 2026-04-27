package guest

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"io"
)

type requestEnvelope struct {
	ProtocolVersion string `json:"protocol_version"`
	RequestID       string `json:"request_id"`
	Type            string `json:"type"`
}

func (s *Server) ServeStream(reader io.Reader, writer io.Writer) error {
	bufferedReader := bufio.NewReader(reader)
	for {
		line, err := bufferedReader.ReadBytes('\n')
		if err != nil && !errors.Is(err, io.EOF) {
			return err
		}

		payload := bytes.TrimSpace(line)
		if len(payload) == 0 {
			if errors.Is(err, io.EOF) {
				return nil
			}
			continue
		}

		response, handleErr := s.handleMessage(payload)
		if handleErr != nil {
			return handleErr
		}
		if _, writeErr := writer.Write(append(response, '\n')); writeErr != nil {
			return writeErr
		}

		if errors.Is(err, io.EOF) {
			return nil
		}
	}
}

func (s *Server) handleMessage(payload []byte) ([]byte, error) {
	var envelope requestEnvelope
	if err := json.Unmarshal(payload, &envelope); err != nil {
		return encodeMessage(&ErrorResponse{
			ProtocolVersion: ProtocolVersion,
			ErrorCode:       "invalid_request",
			Message:         "invalid JSON request",
		})
	}
	if envelope.ProtocolVersion != ProtocolVersion {
		return encodeMessage(&ErrorResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       envelope.RequestID,
			ErrorCode:       "protocol_mismatch",
			Message:         "unsupported guest protocol version",
		})
	}

	switch envelope.Type {
	case "handshake":
		var request HandshakeRequest
		if err := json.Unmarshal(payload, &request); err != nil {
			return encodeMessage(&ErrorResponse{
				ProtocolVersion: ProtocolVersion,
				RequestID:       envelope.RequestID,
				ErrorCode:       "invalid_request",
				Message:         "invalid handshake request",
			})
		}
		return encodeMessage(s.Handshake(request))
	case "ready":
		var request ReadyRequest
		if err := json.Unmarshal(payload, &request); err != nil {
			return encodeMessage(&ErrorResponse{
				ProtocolVersion: ProtocolVersion,
				RequestID:       envelope.RequestID,
				ErrorCode:       "invalid_request",
				Message:         "invalid ready request",
			})
		}
		return encodeMessage(s.Ready(request))
	case "heartbeat":
		var request HeartbeatRequest
		if err := json.Unmarshal(payload, &request); err != nil {
			return encodeMessage(&ErrorResponse{
				ProtocolVersion: ProtocolVersion,
				RequestID:       envelope.RequestID,
				ErrorCode:       "invalid_request",
				Message:         "invalid heartbeat request",
			})
		}
		return encodeMessage(s.Heartbeat(request))
	case "reconnect":
		var request ReconnectRequest
		if err := json.Unmarshal(payload, &request); err != nil {
			return encodeMessage(&ErrorResponse{
				ProtocolVersion: ProtocolVersion,
				RequestID:       envelope.RequestID,
				ErrorCode:       "invalid_request",
				Message:         "invalid reconnect request",
			})
		}
		return encodeMessage(s.Reconnect(request))
	case "exec":
		var request ExecRequest
		if err := json.Unmarshal(payload, &request); err != nil {
			return encodeMessage(&ErrorResponse{
				ProtocolVersion: ProtocolVersion,
				RequestID:       envelope.RequestID,
				ErrorCode:       "invalid_request",
				Message:         "invalid exec request",
			})
		}
		response, execErr := s.Exec(request)
		if execErr != nil {
			return encodeMessage(execErr)
		}
		return encodeMessage(response)
	default:
		return encodeMessage(&ErrorResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       envelope.RequestID,
			ErrorCode:       "invalid_request",
			Message:         "unsupported request type",
		})
	}
}

func encodeMessage(value any) ([]byte, error) {
	return json.Marshal(value)
}
