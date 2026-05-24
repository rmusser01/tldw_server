package acp

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/tldw/tldw-agent/internal/config"
	"github.com/tldw/tldw-agent/internal/mcp/tools"
	"github.com/tldw/tldw-agent/internal/workspace"
)

const (
	defaultProtocolVersion = 1
	runnerName             = "tldw-agent-runner"
	runnerVersion          = "0.1.0"
)

type Runner struct {
	cfg      *config.Config
	upstream *Conn

	sessions   map[string]*Session
	sessionsMu sync.Mutex
	spawnFunc  func(config.AgentConfig) (*Conn, *exec.Cmd, error)
	capsMu     sync.Mutex
	cachedCaps map[string]interface{}
}

type Session struct {
	id          string
	downstream  *Conn
	process     *exec.Cmd
	workspace   *workspace.Session
	fsTools     *tools.FSTools
	searchTools *tools.SearchTools
	gitTools    *tools.GitTools
	terminal    *TerminalManager
	runErr      <-chan error
}

func NewRunner(cfg *config.Config) *Runner {
	runner := &Runner{
		cfg:      cfg,
		sessions: make(map[string]*Session),
	}
	runner.spawnFunc = runner.spawnDownstream
	return runner
}

func (r *Runner) SetSpawnFunc(spawn func(config.AgentConfig) (*Conn, *exec.Cmd, error)) {
	r.spawnFunc = spawn
}

func (r *Runner) Run(stdin io.Reader, stdout io.Writer) error {
	r.upstream = NewConn(stdin, stdout)
	r.upstream.SetHandler(r.handleUpstreamRequest)
	r.upstream.SetNotificationHandler(r.handleUpstreamNotification)

	err := r.upstream.Run()
	r.shutdown()
	return err
}

func (r *Runner) handleUpstreamNotification(msg *RPCMessage) {
	// No upstream notifications are required for MVP.
}

func (r *Runner) handleUpstreamRequest(msg *RPCMessage) (*RPCResponse, error) {
	if msg.JSONRPC != "" && msg.JSONRPC != JSONRPCVersion {
		return NewErrorResponse(msg.ID, ErrInvalidReq, "unsupported jsonrpc version"), nil
	}

	switch msg.Method {
	case "initialize":
		return r.handleInitialize(msg)
	case "agent/list":
		return r.handleAgentList(msg)
	case "session/new":
		return r.handleSessionNew(msg)
	case "session/prompt":
		return r.handleSessionPrompt(msg)
	case "session/cancel":
		return r.handleSessionCancel(msg)
	case "_tldw/session/close":
		return r.handleSessionClose(msg)
	case "session/load":
		return NewErrorResponse(msg.ID, ErrMethodNotFound, "session/load not supported"), nil
	default:
		return NewErrorResponse(msg.ID, ErrMethodNotFound, "method not found"), nil
	}
}

func (r *Runner) handleInitialize(msg *RPCMessage) (*RPCResponse, error) {
	agentCapabilities := r.buildAgentCapabilities()
	result := map[string]interface{}{
		"protocolVersion":   defaultProtocolVersion,
		"agentCapabilities": agentCapabilities,
		"agentInfo": map[string]string{
			"name":    runnerName,
			"title":   "TLDW ACP Runner",
			"version": runnerVersion,
		},
		"authMethods": []interface{}{},
	}

	return NewResultResponse(msg.ID, result), nil
}

type agentListItem struct {
	Type           string `json:"type"`
	Name           string `json:"name"`
	Description    string `json:"description"`
	IsConfigured   bool   `json:"isConfigured"`
	RequiresAPIKey string `json:"requiresApiKey,omitempty"`
}

type agentListResponse struct {
	Agents           []agentListItem `json:"agents"`
	DefaultAgentType string          `json:"defaultAgentType"`
}

func (r *Runner) handleAgentList(msg *RPCMessage) (*RPCResponse, error) {
	entries, defaultType := r.getAgentEntries()
	items := make([]agentListItem, 0, len(entries))
	for _, entry := range entries {
		items = append(items, agentListItem{
			Type:           entry.Type,
			Name:           entry.Name,
			Description:    entry.Description,
			IsConfigured:   r.isAgentReady(entry),
			RequiresAPIKey: entry.RequiresAPIKey,
		})
	}
	return NewResultResponse(msg.ID, agentListResponse{
		Agents:           items,
		DefaultAgentType: defaultType,
	}), nil
}

type sessionNewParams struct {
	Cwd       string `json:"cwd"`
	AgentType string `json:"agentType"`
}

func (r *Runner) handleSessionNew(msg *RPCMessage) (*RPCResponse, error) {
	var params sessionNewParams
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid session/new params"), nil
	}
	if params.Cwd == "" || !filepath.IsAbs(params.Cwd) {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "cwd must be an absolute path"), nil
	}

	agentEntry, err := r.resolveAgentEntry(params.AgentType)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, err.Error()), nil
	}
	if agentEntry.Command == "" {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "agent.command is required"), nil
	}

	ws := workspace.NewSession(r.cfg)
	if err := ws.SetRoot(params.Cwd); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, fmt.Sprintf("invalid cwd: %v", err)), nil
	}

	agentCfg := config.AgentConfig{
		Command: agentEntry.Command,
		Args:    agentEntry.Args,
		Env:     expandAgentEnv(agentEntry.Env),
	}
	downstream, cmd, err := r.spawnFunc(agentCfg)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}

	runErr := make(chan error, 1)
	session := &Session{
		downstream:  downstream,
		process:     cmd,
		workspace:   ws,
		fsTools:     tools.NewFSTools(r.cfg, ws),
		searchTools: tools.NewSearchTools(r.cfg, ws),
		gitTools:    tools.NewGitTools(r.cfg, ws),
		terminal:    NewTerminalManager(r.cfg, ws),
		runErr:      runErr,
	}

	downstream.SetHandler(func(req *RPCMessage) (*RPCResponse, error) {
		return r.handleDownstreamRequest(session, req)
	})
	downstream.SetNotificationHandler(func(note *RPCMessage) {
		r.handleDownstreamNotification(session, note)
	})

	go func() {
		runErr <- downstream.Run()
	}()

	initParams := map[string]interface{}{
		"protocolVersion": defaultProtocolVersion,
		"clientCapabilities": map[string]interface{}{
			"fs": map[string]bool{
				"readTextFile":  true,
				"writeTextFile": true,
			},
			"terminal": r.cfg.Execution.Enabled,
		},
		"clientInfo": map[string]string{
			"name":    runnerName,
			"title":   "TLDW ACP Runner",
			"version": runnerVersion,
		},
	}

	initResp, err := downstream.Call(context.Background(), "initialize", initParams)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, fmt.Sprintf("downstream initialize failed: %v", err)), nil
	}
	if initResp != nil && initResp.Error != nil {
		return &RPCResponse{JSONRPC: JSONRPCVersion, ID: msg.ID, Error: initResp.Error}, nil
	}
	if initResp != nil && initResp.Result != nil {
		r.updateCachedCapabilities(initResp.Result)
	}

	downstreamParams, err := stripSessionRoutingParams(msg.Params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid session/new params"), nil
	}

	resp, err := downstream.CallRaw(context.Background(), "session/new", downstreamParams)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, fmt.Sprintf("downstream session/new failed: %v", err)), nil
	}
	if resp.Error != nil {
		return &RPCResponse{JSONRPC: JSONRPCVersion, ID: msg.ID, Error: resp.Error}, nil
	}

	var sessionResult struct {
		SessionID string `json:"sessionId"`
	}
	if err := json.Unmarshal(resp.Result, &sessionResult); err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, "invalid downstream session/new result"), nil
	}
	if sessionResult.SessionID == "" {
		return NewErrorResponse(msg.ID, ErrInternal, "missing downstream sessionId"), nil
	}

	session.id = sessionResult.SessionID
	r.sessionsMu.Lock()
	r.sessions[session.id] = session
	r.sessionsMu.Unlock()
	go r.watchSession(session.id, runErr)

	return NewResultResponse(msg.ID, json.RawMessage(resp.Result)), nil
}

func stripSessionRoutingParams(raw json.RawMessage) (json.RawMessage, error) {
	// RawMessage preserves downstream-owned values without decoding nested payloads.
	var params map[string]json.RawMessage
	if err := json.Unmarshal(raw, &params); err != nil {
		return nil, err
	}
	delete(params, "agentType")
	return json.Marshal(params)
}

type sessionPromptParams struct {
	SessionID string `json:"sessionId"`
}

func (r *Runner) handleSessionPrompt(msg *RPCMessage) (*RPCResponse, error) {
	var params sessionPromptParams
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid session/prompt params"), nil
	}

	session := r.getSession(params.SessionID)
	if session == nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "unknown session"), nil
	}

	resp, err := session.downstream.CallRaw(context.Background(), "session/prompt", msg.Params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, fmt.Sprintf("downstream session/prompt failed: %v", err)), nil
	}
	if resp.Error != nil {
		return &RPCResponse{JSONRPC: JSONRPCVersion, ID: msg.ID, Error: resp.Error}, nil
	}

	return NewResultResponse(msg.ID, json.RawMessage(resp.Result)), nil
}

func (r *Runner) handleSessionCancel(msg *RPCMessage) (*RPCResponse, error) {
	var params sessionPromptParams
	if err := json.Unmarshal(msg.Params, &params); err == nil {
		if session := r.getSession(params.SessionID); session != nil {
			_ = session.downstream.NotifyRaw("session/cancel", msg.Params)
		}
	}

	if len(msg.ID) == 0 || string(msg.ID) == "null" {
		return nil, nil
	}
	return NewResultResponse(msg.ID, nil), nil
}

func (r *Runner) handleSessionClose(msg *RPCMessage) (*RPCResponse, error) {
	var params sessionPromptParams
	if err := json.Unmarshal(msg.Params, &params); err != nil || params.SessionID == "" {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid session/close params"), nil
	}

	r.cleanupSession(params.SessionID)
	return NewResultResponse(msg.ID, nil), nil
}

func (r *Runner) getAgentEntries() ([]config.RegisteredAgent, string) {
	registry := r.cfg.Agents.Agents
	if len(registry) > 0 {
		return registry, defaultAgentType(registry, r.cfg.Agents.Default)
	}

	legacy := config.RegisteredAgent{
		Type:        "custom",
		Name:        "Custom Agent",
		Description: "Configured via tldw-agent agent.* settings",
		Command:     r.cfg.Agent.Command,
		Args:        r.cfg.Agent.Args,
		Env:         r.cfg.Agent.Env,
	}
	return []config.RegisteredAgent{legacy}, legacy.Type
}

func defaultAgentType(entries []config.RegisteredAgent, configuredDefault string) string {
	if len(entries) == 0 {
		return ""
	}
	if configuredDefault != "" && agentTypeExists(entries, configuredDefault) {
		for _, entry := range entries {
			if entry.Type == configuredDefault && strings.TrimSpace(entry.Command) != "" {
				return configuredDefault
			}
		}
	}
	for _, entry := range entries {
		if strings.TrimSpace(entry.Command) != "" {
			return entry.Type
		}
	}
	if configuredDefault != "" && agentTypeExists(entries, configuredDefault) {
		return configuredDefault
	}
	return entries[0].Type
}

func (r *Runner) resolveAgentEntry(agentType string) (config.RegisteredAgent, error) {
	entries, defaultType := r.getAgentEntries()
	targetType := agentType
	if targetType == "" {
		targetType = defaultType
	}
	for _, entry := range entries {
		if entry.Type == targetType {
			return entry, nil
		}
	}
	return config.RegisteredAgent{}, fmt.Errorf("unknown agent type: %s", targetType)
}

func agentTypeExists(entries []config.RegisteredAgent, agentType string) bool {
	for _, entry := range entries {
		if entry.Type == agentType {
			return true
		}
	}
	return false
}

func (r *Runner) buildAgentCapabilities() map[string]interface{} {
	base := defaultAgentCapabilities()
	cached := r.getCachedCapabilities()
	if cached == nil {
		if refreshed := r.refreshCapabilities(); refreshed != nil {
			cached = refreshed
		}
	}
	if cached == nil {
		return base
	}

	merged := copyMap(base)
	if promptCaps, ok := cached["promptCapabilities"]; ok {
		merged["promptCapabilities"] = promptCaps
	}
	if mcpCaps, ok := cached["mcpCapabilities"]; ok {
		merged["mcpCapabilities"] = mcpCaps
	}
	if sessionCaps, ok := cached["sessionCapabilities"]; ok {
		merged["sessionCapabilities"] = sessionCaps
	}
	merged["loadSession"] = false
	return merged
}

func defaultAgentCapabilities() map[string]interface{} {
	return map[string]interface{}{
		"loadSession": false,
		"promptCapabilities": map[string]bool{
			"image":           false,
			"audio":           false,
			"embeddedContext": false,
		},
		"mcpCapabilities": map[string]bool{
			"http": false,
			"sse":  false,
		},
		"sessionCapabilities": map[string]interface{}{},
	}
}

func (r *Runner) getCachedCapabilities() map[string]interface{} {
	r.capsMu.Lock()
	defer r.capsMu.Unlock()
	if r.cachedCaps == nil {
		return nil
	}
	return copyMap(r.cachedCaps)
}

func (r *Runner) updateCachedCapabilities(raw json.RawMessage) {
	caps := parseAgentCapabilities(raw)
	if caps == nil {
		return
	}
	r.capsMu.Lock()
	r.cachedCaps = caps
	r.capsMu.Unlock()
}

func (r *Runner) refreshCapabilities() map[string]interface{} {
	agentEntry, err := r.resolveAgentEntry("")
	if err != nil {
		return nil
	}
	agentCfg := config.AgentConfig{
		Command: agentEntry.Command,
		Args:    agentEntry.Args,
		Env:     expandAgentEnv(agentEntry.Env),
	}
	caps := r.probeAgentCapabilities(agentCfg, 5*time.Second)
	if caps == nil {
		return nil
	}
	r.capsMu.Lock()
	r.cachedCaps = caps
	r.capsMu.Unlock()
	return copyMap(caps)
}

func (r *Runner) isAgentReady(entry config.RegisteredAgent) bool {
	if strings.TrimSpace(entry.Command) == "" {
		return false
	}
	agentCfg := config.AgentConfig{
		Command: entry.Command,
		Args:    entry.Args,
		Env:     expandAgentEnv(entry.Env),
	}
	return r.probeAgentCapabilities(agentCfg, 2*time.Second) != nil
}

func expandAgentEnv(values []string) []string {
	if len(values) == 0 {
		return values
	}
	expanded := make([]string, len(values))
	for i, value := range values {
		expanded[i] = os.ExpandEnv(value)
	}
	return expanded
}

func (r *Runner) probeAgentCapabilities(agentCfg config.AgentConfig, timeout time.Duration) map[string]interface{} {
	downstream, cmd, err := r.spawnFunc(agentCfg)
	if err != nil {
		return nil
	}
	runErr := make(chan error, 1)
	go func() {
		runErr <- downstream.Run()
	}()

	ctx, cancel := context.WithTimeout(context.Background(), timeout)
	defer cancel()

	initParams := map[string]interface{}{
		"protocolVersion": defaultProtocolVersion,
		"clientCapabilities": map[string]interface{}{
			"fs": map[string]bool{
				"readTextFile":  true,
				"writeTextFile": true,
			},
			"terminal": r.cfg.Execution.Enabled,
		},
		"clientInfo": map[string]string{
			"name":    runnerName,
			"title":   "TLDW ACP Runner",
			"version": runnerVersion,
		},
	}

	resp, err := downstream.Call(ctx, "initialize", initParams)
	r.terminateProcess(cmd)
	select {
	case <-runErr:
	default:
	}
	if err != nil || resp == nil || resp.Error != nil {
		return nil
	}

	return parseAgentCapabilities(resp.Result)
}

func parseAgentCapabilities(raw json.RawMessage) map[string]interface{} {
	if raw == nil {
		return nil
	}
	var payload map[string]interface{}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil
	}
	caps, ok := payload["agentCapabilities"].(map[string]interface{})
	if !ok {
		return nil
	}
	if _, ok := caps["mcpCapabilities"]; !ok {
		if legacy, ok := caps["mcp"].(map[string]interface{}); ok {
			caps["mcpCapabilities"] = legacy
		}
	}
	return caps
}

func copyMap(src map[string]interface{}) map[string]interface{} {
	dst := make(map[string]interface{}, len(src))
	for key, value := range src {
		dst[key] = value
	}
	return dst
}

func (r *Runner) watchSession(sessionID string, runErr <-chan error) {
	if runErr == nil {
		return
	}
	<-runErr
	r.cleanupSession(sessionID)
}

func (r *Runner) cleanupSession(sessionID string) {
	r.sessionsMu.Lock()
	session := r.sessions[sessionID]
	delete(r.sessions, sessionID)
	r.sessionsMu.Unlock()
	if session == nil {
		return
	}
	r.terminateProcess(session.process)
}

func (r *Runner) terminateProcess(cmd *exec.Cmd) {
	if cmd == nil || cmd.Process == nil {
		return
	}
	_ = cmd.Process.Kill()
	_, _ = cmd.Process.Wait()
}

func (r *Runner) shutdown() {
	r.sessionsMu.Lock()
	ids := make([]string, 0, len(r.sessions))
	for sessionID := range r.sessions {
		ids = append(ids, sessionID)
	}
	r.sessionsMu.Unlock()
	for _, sessionID := range ids {
		r.cleanupSession(sessionID)
	}
}

func (r *Runner) handleDownstreamNotification(session *Session, msg *RPCMessage) {
	if r.upstream == nil {
		return
	}
	_ = r.upstream.SendMessage(msg)
}

func (r *Runner) handleDownstreamRequest(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	switch msg.Method {
	case "fs/read_text_file":
		return r.handleFSRead(session, msg)
	case "fs/write_text_file":
		return r.handleFSWrite(session, msg)
	case "search/grep":
		return r.handleSearchGrep(session, msg)
	case "search/glob":
		return r.handleSearchGlob(session, msg)
	case "git/status":
		return r.handleGitStatus(session, msg)
	case "git/diff":
		return r.handleGitDiff(session, msg)
	case "git/log":
		return r.handleGitLog(session, msg)
	case "git/branch":
		return r.handleGitBranch(session, msg)
	case "terminal/create":
		return r.handleTerminalCreate(session, msg)
	case "terminal/output":
		return r.handleTerminalOutput(session, msg)
	case "terminal/wait_for_exit":
		return r.handleTerminalWait(session, msg)
	case "terminal/kill":
		return r.handleTerminalKill(session, msg)
	case "terminal/release":
		return r.handleTerminalRelease(session, msg)
	case "session/request_permission":
		return r.handlePermissionRequest(session, msg)
	default:
		return NewErrorResponse(msg.ID, ErrMethodNotFound, "method not found"), nil
	}
}

func (r *Runner) handlePermissionRequest(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	if r.upstream == nil {
		fallback := map[string]interface{}{
			"outcome": map[string]interface{}{"outcome": "cancelled"},
		}
		return NewResultResponse(msg.ID, fallback), nil
	}

	resp, err := r.upstream.CallRaw(context.Background(), "session/request_permission", msg.Params)
	if err != nil || resp == nil {
		fallback := map[string]interface{}{
			"outcome": map[string]interface{}{"outcome": "cancelled"},
		}
		return NewResultResponse(msg.ID, fallback), nil
	}
	if resp.Error != nil {
		return NewResultResponse(msg.ID, map[string]interface{}{
			"outcome": map[string]interface{}{"outcome": "cancelled"},
		}), nil
	}

	return NewResultResponse(msg.ID, json.RawMessage(resp.Result)), nil
}

func (r *Runner) handleFSRead(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	var params struct {
		SessionID string `json:"sessionId"`
		Path      string `json:"path"`
		Line      int    `json:"line"`
		Limit     int    `json:"limit"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid fs/read_text_file params"), nil
	}
	if params.Path == "" || !filepath.IsAbs(params.Path) {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "path must be absolute"), nil
	}
	if params.SessionID != "" && session.id != params.SessionID {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "sessionId mismatch"), nil
	}

	args := map[string]interface{}{"path": params.Path}
	if params.Limit > 0 {
		startLine := params.Line
		if startLine <= 0 {
			startLine = 1
		}
		args["start_line"] = startLine
		args["end_line"] = startLine + params.Limit - 1
	} else if params.Line > 0 {
		args["start_line"] = params.Line
	}

	res, err := session.fsTools.Read(args)
	if err != nil || !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, "failed to read file"), nil
	}

	data, ok := res.Data.(map[string]interface{})
	if !ok {
		return NewErrorResponse(msg.ID, ErrInternal, "unexpected read result"), nil
	}
	content, _ := data["content"].(string)
	return NewResultResponse(msg.ID, map[string]interface{}{"content": content}), nil
}

func (r *Runner) handleFSWrite(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	var params struct {
		SessionID string `json:"sessionId"`
		Path      string `json:"path"`
		Content   string `json:"content"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid fs/write_text_file params"), nil
	}
	if params.Path == "" || !filepath.IsAbs(params.Path) {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "path must be absolute"), nil
	}
	if params.SessionID != "" && session.id != params.SessionID {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "sessionId mismatch"), nil
	}

	args := map[string]interface{}{"path": params.Path, "content": params.Content}
	res, err := session.fsTools.Write(args)
	if err != nil || !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, "failed to write file"), nil
	}

	return NewResultResponse(msg.ID, nil), nil
}

func (r *Runner) handleSearchGrep(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	params, err := parseParams(msg)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid search/grep params"), nil
	}
	if err := validateSessionID(params, session.id); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, err.Error()), nil
	}
	delete(params, "sessionId")

	res, err := session.searchTools.Grep(params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	if !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, res.Error), nil
	}
	return NewResultResponse(msg.ID, res.Data), nil
}

func (r *Runner) handleSearchGlob(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	params, err := parseParams(msg)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid search/glob params"), nil
	}
	if err := validateSessionID(params, session.id); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, err.Error()), nil
	}
	delete(params, "sessionId")

	res, err := session.searchTools.Glob(params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	if !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, res.Error), nil
	}
	return NewResultResponse(msg.ID, res.Data), nil
}

func (r *Runner) handleGitStatus(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	params, err := parseParams(msg)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid git/status params"), nil
	}
	if err := validateSessionID(params, session.id); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, err.Error()), nil
	}
	delete(params, "sessionId")

	res, err := session.gitTools.Status(params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	if !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, res.Error), nil
	}
	return NewResultResponse(msg.ID, res.Data), nil
}

func (r *Runner) handleGitDiff(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	params, err := parseParams(msg)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid git/diff params"), nil
	}
	if err := validateSessionID(params, session.id); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, err.Error()), nil
	}
	delete(params, "sessionId")

	res, err := session.gitTools.Diff(params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	if !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, res.Error), nil
	}
	return NewResultResponse(msg.ID, res.Data), nil
}

func (r *Runner) handleGitLog(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	params, err := parseParams(msg)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid git/log params"), nil
	}
	if err := validateSessionID(params, session.id); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, err.Error()), nil
	}
	delete(params, "sessionId")

	res, err := session.gitTools.Log(params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	if !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, res.Error), nil
	}
	return NewResultResponse(msg.ID, res.Data), nil
}

func (r *Runner) handleGitBranch(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	params, err := parseParams(msg)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid git/branch params"), nil
	}
	if err := validateSessionID(params, session.id); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, err.Error()), nil
	}
	delete(params, "sessionId")

	res, err := session.gitTools.Branch(params)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	if !res.OK {
		return NewErrorResponse(msg.ID, ErrInternal, res.Error), nil
	}
	return NewResultResponse(msg.ID, res.Data), nil
}

func (r *Runner) handleTerminalCreate(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	var params struct {
		SessionID       string   `json:"sessionId"`
		Command         string   `json:"command"`
		Args            []string `json:"args"`
		Cwd             string   `json:"cwd"`
		OutputByteLimit int      `json:"outputByteLimit"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid terminal/create params"), nil
	}
	if params.SessionID != "" && session.id != params.SessionID {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "sessionId mismatch"), nil
	}

	termID, err := session.terminal.Create(params.Command, params.Args, params.Cwd, params.OutputByteLimit)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}

	return NewResultResponse(msg.ID, map[string]string{"terminalId": termID}), nil
}

func (r *Runner) handleTerminalOutput(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	var params struct {
		SessionID  string `json:"sessionId"`
		TerminalID string `json:"terminalId"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid terminal/output params"), nil
	}
	if params.SessionID != "" && session.id != params.SessionID {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "sessionId mismatch"), nil
	}

	output, truncated, exitStatus, err := session.terminal.Output(params.TerminalID)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}

	result := map[string]interface{}{
		"output":    output,
		"truncated": truncated,
	}
	if exitStatus != nil {
		result["exitStatus"] = exitStatus
	}

	return NewResultResponse(msg.ID, result), nil
}

func (r *Runner) handleTerminalWait(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	var params struct {
		SessionID  string `json:"sessionId"`
		TerminalID string `json:"terminalId"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid terminal/wait_for_exit params"), nil
	}
	if params.SessionID != "" && session.id != params.SessionID {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "sessionId mismatch"), nil
	}

	status, err := session.terminal.WaitForExit(params.TerminalID)
	if err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}

	return NewResultResponse(msg.ID, map[string]interface{}{
		"exitCode": status.ExitCode,
		"signal":   status.Signal,
	}), nil
}

func (r *Runner) handleTerminalKill(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	var params struct {
		SessionID  string `json:"sessionId"`
		TerminalID string `json:"terminalId"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid terminal/kill params"), nil
	}
	if params.SessionID != "" && session.id != params.SessionID {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "sessionId mismatch"), nil
	}

	if err := session.terminal.Kill(params.TerminalID); err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	return NewResultResponse(msg.ID, nil), nil
}

func (r *Runner) handleTerminalRelease(session *Session, msg *RPCMessage) (*RPCResponse, error) {
	var params struct {
		SessionID  string `json:"sessionId"`
		TerminalID string `json:"terminalId"`
	}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "invalid terminal/release params"), nil
	}
	if params.SessionID != "" && session.id != params.SessionID {
		return NewErrorResponse(msg.ID, ErrInvalidParams, "sessionId mismatch"), nil
	}

	if err := session.terminal.Release(params.TerminalID); err != nil {
		return NewErrorResponse(msg.ID, ErrInternal, err.Error()), nil
	}
	return NewResultResponse(msg.ID, nil), nil
}

func parseParams(msg *RPCMessage) (map[string]interface{}, error) {
	if msg.Params == nil {
		return map[string]interface{}{}, nil
	}
	var params map[string]interface{}
	if err := json.Unmarshal(msg.Params, &params); err != nil {
		return nil, err
	}
	return params, nil
}

func validateSessionID(params map[string]interface{}, sessionID string) error {
	if params == nil {
		return nil
	}
	raw, ok := params["sessionId"]
	if !ok {
		return nil
	}
	value, ok := raw.(string)
	if !ok || value == "" {
		return nil
	}
	if value != sessionID {
		return fmt.Errorf("sessionId mismatch")
	}
	return nil
}

func (r *Runner) getSession(id string) *Session {
	r.sessionsMu.Lock()
	defer r.sessionsMu.Unlock()
	return r.sessions[id]
}

func (r *Runner) spawnDownstream(agent config.AgentConfig) (*Conn, *exec.Cmd, error) {
	cmd := exec.Command(agent.Command, agent.Args...)
	cmd.Env = append(os.Environ(), agent.Env...)
	cmd.Stderr = os.Stderr

	stdin, err := cmd.StdinPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("downstream stdin: %w", err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return nil, nil, fmt.Errorf("downstream stdout: %w", err)
	}

	if err := cmd.Start(); err != nil {
		return nil, nil, fmt.Errorf("start downstream: %w", err)
	}

	return NewConn(stdout, stdin), cmd, nil
}
