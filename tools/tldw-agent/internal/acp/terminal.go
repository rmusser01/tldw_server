package acp

import (
	"context"
	"fmt"
	"io"
	"os"
	"os/exec"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"syscall"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/mcp/tools"
	"github.com/tldw/tldw-agent/internal/workspace"
)

type TerminalManager struct {
	config    *config.Config
	session   *workspace.Session
	commands  []tools.Command
	mu        sync.Mutex
	terminals map[string]*terminalProcess
	nextID    int64
}

type terminalProcess struct {
	id       string
	cmd      *exec.Cmd
	cancel   context.CancelFunc
	output   *cappedBuffer
	done     chan struct{}
	exitCode *int
	signal   *string
}

type cappedBuffer struct {
	mu        sync.Mutex
	buf       []byte
	limit     int
	truncated bool
}

func (b *cappedBuffer) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	b.buf = append(b.buf, p...)
	if b.limit > 0 && len(b.buf) > b.limit {
		over := len(b.buf) - b.limit
		b.buf = append([]byte{}, b.buf[over:]...)
		b.truncated = true
	}

	return len(p), nil
}

func (b *cappedBuffer) Snapshot() ([]byte, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()
	return append([]byte{}, b.buf...), b.truncated
}

func NewTerminalManager(cfg *config.Config, session *workspace.Session) *TerminalManager {
	commands := append([]tools.Command{}, tools.DefaultCommands()...)
	commands = append(commands, cfg.Execution.CustomCommands...)

	return &TerminalManager{
		config:    cfg,
		session:   session,
		commands:  commands,
		terminals: make(map[string]*terminalProcess),
	}
}

func (m *TerminalManager) Create(command string, args []string, cwd string, outputLimit int) (string, error) {
	if !m.config.Execution.Enabled {
		return "", fmt.Errorf("terminal execution disabled")
	}

	cmdDef, extraArgs, err := m.matchAllowlist(command, args)
	if err != nil {
		return "", err
	}
	for _, arg := range extraArgs {
		if containsShellMeta(arg) {
			return "", fmt.Errorf("argument %q contains disallowed characters", arg)
		}
	}

	fullCmd := cmdDef.Template
	if len(extraArgs) > 0 {
		fullCmd = fullCmd + " " + strings.Join(extraArgs, " ")
	}

	if cwd == "" {
		cwd = m.session.Root()
	}
	if cwd == "" {
		return "", fmt.Errorf("workspace root not set")
	}
	if !isAbsPath(cwd) {
		return "", fmt.Errorf("cwd must be absolute")
	}
	absCwd, err := m.session.ResolvePath(cwd)
	if err != nil {
		return "", fmt.Errorf("invalid cwd: %w", err)
	}

	limit := m.config.Execution.MaxOutputBytes
	if outputLimit > 0 && outputLimit < limit {
		limit = outputLimit
	}
	if limit <= 0 {
		limit = 1024 * 1024
	}

	ctx, cancel := context.WithCancel(context.Background())
	cmd := buildShellCommand(ctx, m.config.Execution.Shell, fullCmd)
	cmd.Dir = absCwd
	cmd.Env = append(os.Environ(), cmdDef.Env...)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		cancel()
		return "", fmt.Errorf("stdout pipe: %w", err)
	}
	stderr, err := cmd.StderrPipe()
	if err != nil {
		cancel()
		return "", fmt.Errorf("stderr pipe: %w", err)
	}

	if err := cmd.Start(); err != nil {
		cancel()
		return "", fmt.Errorf("start command: %w", err)
	}

	termID := fmt.Sprintf("term_%d", atomic.AddInt64(&m.nextID, 1))
	buffer := &cappedBuffer{limit: limit}
	proc := &terminalProcess{
		id:     termID,
		cmd:    cmd,
		cancel: cancel,
		output: buffer,
		done:   make(chan struct{}),
	}

	go streamOutput(buffer, stdout)
	go streamOutput(buffer, stderr)

	go func() {
		err := cmd.Wait()
		if err != nil {
			_ = err
		}
		code := cmd.ProcessState.ExitCode()
		proc.exitCode = &code
		if status, ok := cmd.ProcessState.Sys().(syscall.WaitStatus); ok && status.Signaled() {
			s := status.Signal().String()
			proc.signal = &s
		}
		close(proc.done)
	}()

	m.mu.Lock()
	m.terminals[termID] = proc
	m.mu.Unlock()

	return termID, nil
}

func (m *TerminalManager) Output(terminalID string) (string, bool, *TerminalExitStatus, error) {
	proc := m.get(terminalID)
	if proc == nil {
		return "", false, nil, fmt.Errorf("terminal not found")
	}

	data, truncated := proc.output.Snapshot()
	var exitStatus *TerminalExitStatus
	select {
	case <-proc.done:
		exitStatus = &TerminalExitStatus{ExitCode: proc.exitCode, Signal: proc.signal}
	default:
	}

	return string(data), truncated, exitStatus, nil
}

func (m *TerminalManager) WaitForExit(terminalID string) (*TerminalExitStatus, error) {
	proc := m.get(terminalID)
	if proc == nil {
		return nil, fmt.Errorf("terminal not found")
	}

	<-proc.done
	return &TerminalExitStatus{ExitCode: proc.exitCode, Signal: proc.signal}, nil
}

func (m *TerminalManager) Kill(terminalID string) error {
	proc := m.get(terminalID)
	if proc == nil {
		return fmt.Errorf("terminal not found")
	}

	proc.cancel()
	if proc.cmd.Process != nil {
		return proc.cmd.Process.Kill()
	}
	return nil
}

func (m *TerminalManager) Release(terminalID string) error {
	proc := m.get(terminalID)
	if proc == nil {
		return fmt.Errorf("terminal not found")
	}
	_ = m.Kill(terminalID)

	m.mu.Lock()
	delete(m.terminals, terminalID)
	m.mu.Unlock()

	return nil
}

func (m *TerminalManager) get(terminalID string) *terminalProcess {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.terminals[terminalID]
}

func (m *TerminalManager) matchAllowlist(command string, args []string) (tools.Command, []string, error) {
	requested := append([]string{command}, args...)
	for _, cmd := range m.commands {
		templateTokens := strings.Fields(cmd.Template)
		if len(templateTokens) == 0 {
			continue
		}
		if len(requested) < len(templateTokens) {
			continue
		}
		if !tokensMatchPrefix(templateTokens, requested) {
			continue
		}

		extra := requested[len(templateTokens):]
		if len(extra) > 0 {
			if !cmd.AllowArgs {
				continue
			}
			if cmd.MaxArgs > 0 && len(extra) > cmd.MaxArgs {
				continue
			}
		}
		return cmd, extra, nil
	}
	return tools.Command{}, nil, fmt.Errorf("command not in allowlist")
}

func tokensMatchPrefix(prefix []string, full []string) bool {
	if len(prefix) > len(full) {
		return false
	}
	for i := range prefix {
		if prefix[i] != full[i] {
			return false
		}
	}
	return true
}

func buildShellCommand(ctx context.Context, shell, command string) *exec.Cmd {
	if runtime.GOOS == "windows" {
		if shell == "auto" || shell == "" {
			shell = "powershell"
		}
		switch shell {
		case "powershell":
			return exec.CommandContext(ctx, "powershell", "-NoProfile", "-NonInteractive", "-Command", command)
		case "cmd":
			return exec.CommandContext(ctx, "cmd", "/c", command)
		default:
			return exec.CommandContext(ctx, shell, "-c", command)
		}
	}

	if shell == "auto" || shell == "" {
		shell = "sh"
	}
	return exec.CommandContext(ctx, shell, "-c", command)
}

func streamOutput(buffer *cappedBuffer, r io.Reader) {
	_, _ = io.Copy(buffer, r)
}

func containsShellMeta(s string) bool {
	metaChars := []string{
		";", "&", "|", "`", "$", "(", ")", "{", "}", "<", ">",
		"'", "\"", "\\", "\n", "\r",
	}

	for _, meta := range metaChars {
		if strings.Contains(s, meta) {
			return true
		}
	}

	return false
}

func isAbsPath(path string) bool {
	if runtime.GOOS == "windows" {
		return strings.HasPrefix(path, "\\\\") || (len(path) > 1 && path[1] == ':')
	}
	return strings.HasPrefix(path, "/")
}

// TerminalExitStatus mirrors ACP exit status structure.
type TerminalExitStatus struct {
	ExitCode *int    `json:"exitCode"`
	Signal   *string `json:"signal"`
}
