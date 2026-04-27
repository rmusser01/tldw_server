package guest

import (
	"bytes"
	"context"
	"errors"
	"os"
	"os/exec"
	"time"
)

func (s *Server) Exec(req ExecRequest) (*ExecResponse, *ErrorResponse) {
	if len(req.Argv) == 0 {
		return nil, &ErrorResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       req.RequestID,
			ErrorCode:       "invalid_request",
			Message:         "argv is required",
		}
	}

	cwd := s.session.Root()
	if req.Cwd != "" {
		resolved, err := s.session.ResolvePath(req.Cwd)
		if err != nil {
			return nil, &ErrorResponse{
				ProtocolVersion: ProtocolVersion,
				RequestID:       req.RequestID,
				ErrorCode:       "invalid_request",
				Message:         err.Error(),
			}
		}
		cwd = resolved
	}

	timeout := time.Duration(s.cfg.Execution.TimeoutMs) * time.Millisecond
	if req.TimeoutSec > 0 {
		timeout = time.Duration(req.TimeoutSec) * time.Second
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, req.Argv[0], req.Argv[1:]...)
	cmd.Dir = cwd
	cmd.Env = append(os.Environ(), flattenEnv(req.Env)...)

	var stdout bytes.Buffer
	var stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	err := cmd.Run()
	if ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, &ErrorResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       req.RequestID,
			ErrorCode:       "timeout_exceeded",
			Message:         "guest exec timed out",
		}
	}

	exitCode := 0
	if err != nil {
		var exitErr *exec.ExitError
		if errors.As(err, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			return nil, &ErrorResponse{
				ProtocolVersion: ProtocolVersion,
				RequestID:       req.RequestID,
				ErrorCode:       "exec_failed",
				Message:         err.Error(),
			}
		}
	}

	return &ExecResponse{
		ProtocolVersion: ProtocolVersion,
		RequestID:       req.RequestID,
		ExitCode:        exitCode,
		Stdout:          stdout.String(),
		Stderr:          stderr.String(),
	}, nil
}

func flattenEnv(values map[string]string) []string {
	if len(values) == 0 {
		return nil
	}
	flattened := make([]string, 0, len(values))
	for key, value := range values {
		if key == "" {
			continue
		}
		flattened = append(flattened, key+"="+value)
	}
	return flattened
}
