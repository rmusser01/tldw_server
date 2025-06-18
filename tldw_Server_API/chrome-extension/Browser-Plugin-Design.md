# TLDW Server Browser Plugin Design Document

## Overview

The TLDW Server Browser Plugin is a cross-browser extension that provides seamless integration between web browsers and the TLDW (Too Long; Didn't Watch) Server API. It enables users to interact with the server's chat, prompts, characters, and media processing capabilities directly from their browser.

## Architecture

### Core Design Principles

1. **Cross-Browser Compatibility**: Single codebase supporting Chrome (V2/V3), Firefox, Edge, and other Chromium browsers
2. **Modular Architecture**: Clear separation between UI, business logic, and API communication
3. **Security First**: All API communications use secure authentication tokens
4. **User Experience**: Intuitive interface with minimal clicks to accomplish tasks
5. **Performance**: Efficient background processing without impacting browser performance

### Component Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Browser Extension                         │
├─────────────────┬───────────────────┬──────────────────────────┤
│   Popup UI      │  Background Worker │   Content Scripts        │
│                 │                    │                          │
│ • Chat Interface│ • API Proxy       │ • Text Selection         │
│ • Prompts List  │ • Context Menus   │ • Page Interaction       │
│ • Characters    │ • Message Routing │ • Floating Actions       │
│ • Media Upload  │ • Auth Management │ • Element Selection      │
└────────┬────────┴─────────┬─────────┴────────┬─────────────────┘
         │                  │                   │
         └──────────────────┴───────────────────┘
                            │
                    ┌───────▼────────┐
                    │   API Client   │
                    │                │
                    │ • HTTP Requests│
                    │ • Auth Headers │
                    │ • Error Handle │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  TLDW Server   │
                    │     API        │
                    └────────────────┘
```

### File Structure

```
chrome-extension/
├── manifest.json          # Chrome V3 manifest (default)
├── manifest-v2.json       # Firefox/Chrome V2 manifest
├── build.js              # Build script for generating browser-specific versions
│
├── html/
│   ├── popup.html        # Main extension popup interface
│   └── options.html      # Settings/configuration page
│
├── css/
│   ├── popup.css         # Styles for popup interface
│   └── options.css       # Styles for options page
│
├── js/
│   ├── popup.js          # Popup interface logic
│   ├── options.js        # Options page logic
│   ├── background.js     # Chrome V3 service worker
│   ├── background-v2.js  # V2 background script
│   ├── content.js        # Content script for web pages
│   ├── browser-polyfill.js # Cross-browser compatibility layer
│   ├── compat-utils.js   # Compatibility utility functions
│   └── utils/
│       └── api.js        # API client for server communication
│
├── icons/                # Extension icons (16x16, 48x48, 128x128)
├── tests/                # Jest test suite
└── dist/                 # Built extensions (generated)
```

## Features

### 1. Chat Integration

**Purpose**: Enable AI-powered conversations with various LLM models

**Components**:
- Model selection dropdown (GPT-4, Claude, etc.)
- Character context selection
- Message history display
- Real-time streaming support
- Conversation management

**User Flow**:
1. User clicks extension icon → Popup opens
2. Select model and optional character
3. Type message and send
4. Receive AI response
5. Continue conversation with context

### 2. Prompts Management

**Purpose**: Create, browse, and apply prompt templates

**Features**:
- Browse saved prompts with search
- Create new prompts from selected text
- Apply prompts to chat conversations
- Export prompts to Markdown/CSV
- Keyword tagging system

**API Endpoints**:
- `GET/POST /api/v1/prompts/`
- `GET /api/v1/prompts/search`
- `GET /api/v1/prompts/export`

### 3. Character System

**Purpose**: Apply character personalities and contexts to conversations

**Features**:
- Character gallery with avatars
- Import character cards (PNG, WEBP, JSON, MD)
- Search characters by name/description
- Apply character to chat session
- Version control for characters

**Storage**: Character data including images stored as base64

### 4. Media Processing

**Purpose**: Process various media types through the TLDW server

**Supported Types**:
- Videos (MP4, AVI, MOV)
- Audio (MP3, WAV, M4A)
- Documents (PDF, EPUB, DOC, DOCX)
- Web pages (via URL)

**Processing Options**:
- Transcription for audio/video
- Text extraction for documents
- Chunking strategies
- Summarization

### 5. Web Page Interaction

**Purpose**: Seamlessly integrate with web content

**Features**:
- Text selection with floating action button
- Right-click context menu integration
- Keyboard shortcuts (Ctrl+Shift+T, Ctrl+Shift+C)
- Element selection mode
- Quick actions menu

## Configuration

### Server Configuration

**Storage Location**: Chrome sync storage

**Settings**:
```javascript
{
  serverUrl: "http://localhost:8000",     // TLDW server URL
  apiToken: "bearer-token-here",          // Authentication token
  defaultModel: "gpt-4",                  // Default chat model
  defaultTemperature: 0.7,                // Model temperature
  maxTokens: 1000,                        // Max response tokens
  autoLoadChats: false,                   // Load chats on startup
  streamResponses: false,                 // Enable streaming
  showNotifications: true,                // Desktop notifications
  showFloatingButton: true                // Show selection button
}
```

### Authentication

**Method**: Bearer token authentication

**Header Format**:
```
Token: Bearer <API_BEARER_TOKEN>
```

**Token Source**: Server's `API_BEARER` environment variable

### Permissions

**Chrome V3 Permissions**:
- `storage`: Save user preferences
- `contextMenus`: Right-click menu items
- `activeTab`: Current tab access
- `scripting`: Inject content scripts

**Firefox/V2 Additional**:
- `tabs`: Tab management
- `<all_urls>`: Content script injection

## API Communication

### API Client Architecture

**Location**: `/js/utils/api.js`

**Features**:
- Singleton pattern for consistent state
- Automatic initialization on first use
- Promise-based interface
- Error handling with retry logic
- Request/response interceptors

**Example Usage**:
```javascript
// Initialize (happens automatically)
await apiClient.init();

// Make API call
const response = await apiClient.createChatCompletion({
  model: "gpt-4",
  messages: [{ role: "user", content: "Hello" }]
});
```

### Request Flow

1. **User Action** → Popup/Content Script
2. **API Call** → API Client prepares request
3. **Authentication** → Bearer token added to headers
4. **Request** → Sent to TLDW server
5. **Response** → Processed and returned
6. **UI Update** → Results displayed to user

### Error Handling

**Network Errors**:
- Automatic retry with exponential backoff
- User notification on persistent failure
- Fallback to cached data when available

**API Errors**:
- 401: Prompt for new token
- 429: Rate limit handling with retry
- 500+: Server error notification

## Cross-Browser Compatibility

### Manifest Versions

**Chrome V3** (manifest.json):
- Service worker background script
- `action` API for toolbar button
- Declarative net request

**Firefox/Chrome V2** (manifest-v2.json):
- Background pages (persistent: false)
- `browser_action` API
- WebRequest API

### Browser Polyfill

**Purpose**: Unified API across browsers

**Implementation**:
- Wraps Chrome callbacks with Promises
- Provides `browser` namespace for Firefox
- Handles API differences transparently

**Usage**:
```javascript
// Works in both Chrome and Firefox
const tabs = await browser.tabs.query({ active: true });
```

### Build System

**Build Script**: `build.js`

**Outputs**:
- `dist/chrome-v3/`: Chrome 88+ (Manifest V3)
- `dist/chrome-v2/`: Chrome <88 (Manifest V2)
- `dist/firefox/`: Firefox (Manifest V2)

**Build Process**:
1. Copy common files
2. Apply manifest version
3. Update API calls for browser
4. Generate zip files

## Security Considerations

### Data Protection

1. **Token Storage**: Stored in encrypted browser storage
2. **HTTPS Only**: Enforced for production servers
3. **Content Security**: No inline scripts or eval()
4. **Origin Validation**: Verify message senders

### Permissions Model

- Minimal permissions requested
- Host permissions only for configured server
- Optional permissions for enhanced features
- User consent for sensitive operations

## Performance Optimization

### Background Worker

- Event-based activation (V3)
- Lazy loading of resources
- Connection pooling for API calls
- Debounced status checks

### Content Scripts

- Injected only when needed
- Minimal DOM manipulation
- Event delegation for efficiency
- Cleanup on navigation

### Storage

- Chunked storage for large data
- IndexedDB for media cache
- Periodic cleanup of old data
- Compression for character images

## Testing Strategy

### Unit Tests

- API client methods
- UI component logic
- Background worker functions
- Content script features

### Integration Tests

- Full API workflows
- Cross-component communication
- Error scenarios
- Browser compatibility

### Manual Testing

- Browser-specific features
- Permission flows
- UI responsiveness
- Extension updates

## Deployment

### Development

```bash
# Install dependencies
npm install

# Generate icons
Open icons/icon-generator.html

# Build all versions
npm run build

# Run tests
npm test
```

### Distribution

**Chrome Web Store**:
- Use `dist/chrome-v3/`
- Create ZIP file
- Submit for review

**Firefox Add-ons**:
- Use `dist/firefox/`
- Sign with Mozilla
- Submit for review

**Self-Hosting**:
- Provide download links
- Include installation instructions
- Document server requirements

## Future Enhancements

### Planned Features

1. **Offline Mode**: Cache responses for offline access
2. **Batch Processing**: Queue multiple media files
3. **Collaboration**: Share prompts and characters
4. **Mobile Support**: Firefox mobile compatibility
5. **Voice Input**: Speech-to-text for chat

### API Extensions

1. **WebSocket Support**: Real-time updates
2. **File Streaming**: Large file uploads
3. **Workspace Sync**: Cross-device settings
4. **Analytics**: Usage tracking (optional)

## Troubleshooting

### Common Issues

**Extension Not Loading**:
- Check manifest version matches browser
- Verify all files are present
- Look for console errors

**API Connection Failed**:
- Verify server is running
- Check API token is valid
- Confirm server URL is correct

**Features Not Working**:
- Clear extension cache
- Check permissions granted
- Update to latest version

### Debug Mode

Enable verbose logging:
```javascript
// In background script
const DEBUG = true;
```

View logs:
- Chrome: chrome://extensions → Background page
- Firefox: about:debugging → Inspect

## Conclusion

The TLDW Server Browser Plugin provides a robust, cross-browser solution for integrating AI capabilities into the web browsing experience. Its modular design, comprehensive API client, and careful attention to browser compatibility ensure a seamless user experience across platforms while maintaining security and performance standards.