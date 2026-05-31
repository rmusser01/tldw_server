package acp

import (
	"bufio"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"sync"
	"sync/atomic"
)

type RequestHandler func(msg *RPCMessage) (*RPCResponse, error)
type NotificationHandler func(msg *RPCMessage)

var ErrConnectionClosed = errors.New("acp connection closed")

// Conn manages JSON-RPC communication over ACP stdio framing.
type Conn struct {
	reader *bufio.Reader
	writer io.Writer

	writeMu sync.Mutex

	pending   map[string]chan *RPCMessage
	pendingMu sync.Mutex
	nextID    int64
	closeErr  atomic.Value
	closeOnce sync.Once
	closed    chan struct{}

	handler      RequestHandler
	notification NotificationHandler
}

// NewConn creates a new ACP connection.
func NewConn(r io.Reader, w io.Writer) *Conn {
	return &Conn{
		reader:  bufio.NewReader(r),
		writer:  w,
		pending: make(map[string]chan *RPCMessage),
		closed:  make(chan struct{}),
	}
}

// SetHandler registers a request handler.
func (c *Conn) SetHandler(handler RequestHandler) {
	c.handler = handler
}

// SetNotificationHandler registers a notification handler.
func (c *Conn) SetNotificationHandler(handler NotificationHandler) {
	c.notification = handler
}

// Run starts the read loop and blocks until EOF or error.
func (c *Conn) Run() error {
	for {
		payload, err := ReadLineMessage(c.reader)
		if err != nil {
			if err == io.EOF {
				c.markClosed(io.EOF)
				return nil
			}
			c.markClosed(err)
			return err
		}

		var msg RPCMessage
		if err := json.Unmarshal(payload, &msg); err != nil {
			return fmt.Errorf("unmarshal message: %w", err)
		}

		if msg.Method != "" {
			if len(msg.ID) == 0 || string(msg.ID) == "null" {
				if c.notification != nil {
					c.notification(&msg)
				}
				continue
			}

			resp, err := c.handleRequest(&msg)
			if err != nil {
				resp = NewErrorResponse(msg.ID, ErrInternal, err.Error())
			}
			if resp != nil {
				if err := c.SendResponse(resp); err != nil {
					return err
				}
			}
			continue
		}

		if len(msg.ID) > 0 {
			c.deliverResponse(&msg)
		}
	}
}

// Call sends a request and waits for a response.
func (c *Conn) Call(ctx context.Context, method string, params interface{}) (*RPCMessage, error) {
	var rawParams json.RawMessage
	if params != nil {
		data, err := json.Marshal(params)
		if err != nil {
			return nil, fmt.Errorf("marshal params: %w", err)
		}
		rawParams = data
	}
	return c.CallRaw(ctx, method, rawParams)
}

// CallRaw sends a request with raw params and waits for a response.
func (c *Conn) CallRaw(ctx context.Context, method string, params json.RawMessage) (*RPCMessage, error) {
	select {
	case <-c.closed:
		return nil, c.connectionClosedErr()
	default:
	}

	id := atomic.AddInt64(&c.nextID, 1)
	idRaw := json.RawMessage(fmt.Sprintf("%d", id))

	msg := &RPCMessage{
		JSONRPC: JSONRPCVersion,
		ID:      idRaw,
		Method:  method,
		Params:  params,
	}

	respCh := make(chan *RPCMessage, 1)
	key := string(idRaw)
	c.pendingMu.Lock()
	c.pending[key] = respCh
	c.pendingMu.Unlock()

	if err := c.SendMessage(msg); err != nil {
		c.pendingMu.Lock()
		delete(c.pending, key)
		c.pendingMu.Unlock()
		return nil, err
	}

	select {
	case <-ctx.Done():
		c.pendingMu.Lock()
		delete(c.pending, key)
		c.pendingMu.Unlock()
		return nil, ctx.Err()
	case <-c.closed:
		c.pendingMu.Lock()
		delete(c.pending, key)
		c.pendingMu.Unlock()
		return nil, c.connectionClosedErr()
	case resp, ok := <-respCh:
		if !ok {
			return nil, c.connectionClosedErr()
		}
		return resp, nil
	}
}

// Notify sends a JSON-RPC notification.
func (c *Conn) Notify(method string, params interface{}) error {
	var rawParams json.RawMessage
	if params != nil {
		data, err := json.Marshal(params)
		if err != nil {
			return fmt.Errorf("marshal params: %w", err)
		}
		rawParams = data
	}
	return c.NotifyRaw(method, rawParams)
}

// NotifyRaw sends a JSON-RPC notification with raw params.
func (c *Conn) NotifyRaw(method string, params json.RawMessage) error {
	msg := &RPCMessage{
		JSONRPC: JSONRPCVersion,
		Method:  method,
		Params:  params,
	}
	return c.SendMessage(msg)
}

// SendResponse sends a JSON-RPC response.
func (c *Conn) SendResponse(resp *RPCResponse) error {
	if resp.JSONRPC == "" {
		resp.JSONRPC = JSONRPCVersion
	}
	return c.send(resp)
}

// SendMessage sends a raw JSON-RPC message.
func (c *Conn) SendMessage(msg *RPCMessage) error {
	if msg.JSONRPC == "" {
		msg.JSONRPC = JSONRPCVersion
	}
	return c.send(msg)
}

func (c *Conn) send(msg interface{}) error {
	data, err := json.Marshal(msg)
	if err != nil {
		return fmt.Errorf("marshal message: %w", err)
	}

	c.writeMu.Lock()
	defer c.writeMu.Unlock()
	return WriteLineMessage(c.writer, data)
}

func (c *Conn) handleRequest(msg *RPCMessage) (*RPCResponse, error) {
	if c.handler == nil {
		return NewErrorResponse(msg.ID, ErrMethodNotFound, "method not found"), nil
	}
	return c.handler(msg)
}

func (c *Conn) deliverResponse(msg *RPCMessage) {
	key := string(msg.ID)
	c.pendingMu.Lock()
	ch, ok := c.pending[key]
	if ok {
		delete(c.pending, key)
	}
	c.pendingMu.Unlock()
	if ok {
		ch <- msg
	}
}

func (c *Conn) markClosed(err error) {
	if err == nil {
		err = io.EOF
	}
	c.closeOnce.Do(func() {
		c.closeErr.Store(err)
		c.pendingMu.Lock()
		pending := c.pending
		c.pending = make(map[string]chan *RPCMessage)
		c.pendingMu.Unlock()
		for _, ch := range pending {
			close(ch)
		}
		close(c.closed)
	})
}

func (c *Conn) connectionClosedErr() error {
	value := c.closeErr.Load()
	if value == nil {
		return ErrConnectionClosed
	}
	inner, ok := value.(error)
	if !ok || inner == nil {
		return ErrConnectionClosed
	}
	if errors.Is(inner, ErrConnectionClosed) {
		return inner
	}
	return fmt.Errorf("%w: %v", ErrConnectionClosed, inner)
}
