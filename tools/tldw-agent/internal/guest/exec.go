package guest

import (
	"bytes"
	"context"
	"errors"
	"io"
	"os"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	maxGuestOutputBytes = 256 * 1024 * 1024
	outputLimitExitCode = 137
)

type outputLimitReason string

const (
	outputLimitReasonNone   outputLimitReason = ""
	outputLimitReasonOutput outputLimitReason = "output_limit"
)

type outputStream int

const (
	outputStreamStdout outputStream = iota
	outputStreamStderr
)

type boundedExecOutput struct {
	mu           sync.Mutex
	limit        int
	stdout       []byte
	stderr       []byte
	stdoutSeen   int
	stderrSeen   int
	exceeded     bool
	cancelReason outputLimitReason
	cancel       context.CancelFunc
	kill         func()
}

type boundedStreamWriter struct {
	output *boundedExecOutput
	stream outputStream
}

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

	if req.MaxOutputBytes != nil && (*req.MaxOutputBytes <= 0 || *req.MaxOutputBytes > maxGuestOutputBytes) {
		return nil, &ErrorResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       req.RequestID,
			ErrorCode:       "invalid_request",
			Message:         "max_output_bytes out of range",
		}
	}

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	cmd := exec.CommandContext(ctx, req.Argv[0], req.Argv[1:]...)
	cmd.Dir = cwd
	cmd.Env = append(os.Environ(), flattenEnv(req.Env)...)

	if req.MaxOutputBytes != nil {
		return runExecWithOutputLimit(ctx, cancel, cmd, req, *req.MaxOutputBytes)
	}

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

func runExecWithOutputLimit(
	ctx context.Context,
	cancel context.CancelFunc,
	cmd *exec.Cmd,
	req ExecRequest,
	maxOutputBytes int,
) (*ExecResponse, *ErrorResponse) {
	limiter := &boundedExecOutput{
		limit:        maxOutputBytes,
		cancelReason: outputLimitReasonNone,
		cancel:       cancel,
	}
	limiter.kill = func() {
		terminateCommandProcess(cmd)
	}

	stdoutPipe, err := cmd.StdoutPipe()
	if err != nil {
		return nil, execFailedResponse(req.RequestID, err)
	}
	stderrPipe, err := cmd.StderrPipe()
	if err != nil {
		return nil, execFailedResponse(req.RequestID, err)
	}

	configureCommandProcessGroup(cmd)
	cmd.Cancel = func() error {
		terminateCommandProcess(cmd)
		return nil
	}

	if err := cmd.Start(); err != nil {
		return nil, execFailedResponse(req.RequestID, err)
	}

	var wg sync.WaitGroup
	wg.Add(2)
	go func() {
		defer wg.Done()
		_, _ = io.Copy(boundedStreamWriter{output: limiter, stream: outputStreamStdout}, stdoutPipe)
	}()
	go func() {
		defer wg.Done()
		_, _ = io.Copy(boundedStreamWriter{output: limiter, stream: outputStreamStderr}, stderrPipe)
	}()

	waitErr := cmd.Wait()
	wg.Wait()

	stdout, stderr, details, exceeded := limiter.response()
	if exceeded {
		return &ExecResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       req.RequestID,
			ExitCode:        outputLimitExitCode,
			Stdout:          stdout,
			Stderr:          stderr,
			Details:         details,
		}, nil
	}

	if ctx.Err() != nil && errors.Is(ctx.Err(), context.DeadlineExceeded) {
		return nil, &ErrorResponse{
			ProtocolVersion: ProtocolVersion,
			RequestID:       req.RequestID,
			ErrorCode:       "timeout_exceeded",
			Message:         "guest exec timed out",
		}
	}

	exitCode := 0
	if waitErr != nil {
		var exitErr *exec.ExitError
		if errors.As(waitErr, &exitErr) {
			exitCode = exitErr.ExitCode()
		} else {
			return nil, execFailedResponse(req.RequestID, waitErr)
		}
	}

	return &ExecResponse{
		ProtocolVersion: ProtocolVersion,
		RequestID:       req.RequestID,
		ExitCode:        exitCode,
		Stdout:          stdout,
		Stderr:          stderr,
		Details:         details,
	}, nil
}

func execFailedResponse(requestID string, err error) *ErrorResponse {
	return &ErrorResponse{
		ProtocolVersion: ProtocolVersion,
		RequestID:       requestID,
		ErrorCode:       "exec_failed",
		Message:         err.Error(),
	}
}

func (w boundedStreamWriter) Write(p []byte) (int, error) {
	if w.output == nil {
		return len(p), nil
	}
	w.output.write(w.stream, p)
	return len(p), nil
}

func (b *boundedExecOutput) write(stream outputStream, chunk []byte) {
	if len(chunk) == 0 {
		return
	}

	var shouldCancel bool
	b.mu.Lock()
	switch stream {
	case outputStreamStdout:
		b.stdoutSeen += len(chunk)
	case outputStreamStderr:
		b.stderrSeen += len(chunk)
	}

	remaining := b.limit - len(b.stdout) - len(b.stderr)
	if remaining > 0 {
		retained := chunk
		if len(retained) > remaining {
			retained = retained[:remaining]
		}
		switch stream {
		case outputStreamStdout:
			b.stdout = append(b.stdout, retained...)
		case outputStreamStderr:
			b.stderr = append(b.stderr, retained...)
		}
	}

	if b.stdoutSeen+b.stderrSeen > b.limit && !b.exceeded {
		b.exceeded = true
		b.cancelReason = outputLimitReasonOutput
		shouldCancel = true
	}
	cancel := b.cancel
	kill := b.kill
	b.mu.Unlock()

	if shouldCancel {
		if cancel != nil {
			cancel()
		}
		if kill != nil {
			kill()
		}
	}
}

func (b *boundedExecOutput) response() (string, string, map[string]string, bool) {
	b.mu.Lock()
	stdoutBytes := append([]byte(nil), b.stdout...)
	stderrBytes := append([]byte(nil), b.stderr...)
	stdoutSeen := b.stdoutSeen
	stderrSeen := b.stderrSeen
	exceeded := b.exceeded
	reason := b.cancelReason
	limit := b.limit
	b.mu.Unlock()

	stdout := sanitizeUTF8WithinLimit(stdoutBytes, limit)
	remaining := limit - len([]byte(stdout))
	if remaining < 0 {
		remaining = 0
	}
	stderr := sanitizeUTF8WithinLimit(stderrBytes, remaining)
	details := map[string]string{
		"guest_output_limit_bytes":    strconv.Itoa(limit),
		"guest_output_limit_exceeded": strconv.FormatBool(exceeded),
		"guest_stdout_bytes_observed": strconv.Itoa(stdoutSeen),
		"guest_stderr_bytes_observed": strconv.Itoa(stderrSeen),
		"guest_stdout_bytes_returned": strconv.Itoa(len([]byte(stdout))),
		"guest_stderr_bytes_returned": strconv.Itoa(len([]byte(stderr))),
	}
	if reason != outputLimitReasonNone {
		details["guest_output_kill_reason"] = string(reason)
	}
	return stdout, stderr, details, exceeded
}

func sanitizeUTF8WithinLimit(value []byte, maxBytes int) string {
	if maxBytes <= 0 || len(value) == 0 {
		return ""
	}
	if len(value) > maxBytes {
		value = value[:maxBytes]
	}
	text := strings.ToValidUTF8(string(value), "")
	for len([]byte(text)) > maxBytes {
		runes := []rune(text)
		if len(runes) == 0 {
			return ""
		}
		text = string(runes[:len(runes)-1])
	}
	return text
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
