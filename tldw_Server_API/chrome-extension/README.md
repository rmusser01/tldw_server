# TLDW Server Browser Extension

A cross-browser extension that provides seamless integration with your TLDW (Too Long; Didn't Watch) Server for chat, prompts, characters, and media processing. Works with Chrome, Firefox, Edge, and other Chromium-based browsers.

## Features

### 🤖 Chat Integration
- Send messages to various LLM models (GPT-4, Claude, etc.)
- Apply character contexts to conversations
- Use prompt templates for consistent interactions
- Maintain conversation history

### 📝 Prompt Management
- Browse and search through saved prompts
- Create new prompts directly from the extension
- Export prompts to Markdown or CSV
- Apply prompts to chat conversations

### 👥 Character Support
- Browse character library with images and descriptions
- Import character cards (PNG, WEBP, JSON, MD formats)
- Apply character contexts to chat sessions
- Search characters by name or description

### 🎬 Media Processing
- Process URLs directly from the browser
- Upload and process various file types:
  - Videos and audio files
  - PDFs and EPUBs
  - Documents (DOC, DOCX)
- View and manage processed media items
- Quick summarization of media content

### 🌐 Web Page Interaction
- Select text on any webpage to send to chat
- Right-click context menu for quick actions
- Floating action button for selected text
- Keyboard shortcuts for rapid access

## Installation

### Building the Extension

1. Install dependencies:
   ```bash
   npm install
   ```

2. Generate icons:
   - Open `/chrome-extension/icons/icon-generator.html` in your browser
   - Click on each icon to download and save them in the `icons` folder

3. Build for your browser:
   ```bash
   # Build all versions
   npm run build
   
   # Build specific version
   npm run build:chrome-v3  # Chrome 88+
   npm run build:chrome-v2  # Chrome 87 and older
   npm run build:firefox    # Firefox
   ```

### Installing in Chrome (Manifest V3)
1. Open Chrome and navigate to `chrome://extensions/`
2. Enable "Developer mode" in the top right
3. Click "Load unpacked" and select the `dist/chrome-v3` folder
4. The extension will appear in your browser toolbar

### Installing in Firefox
1. Open Firefox and navigate to `about:debugging`
2. Click "This Firefox" in the left sidebar
3. Click "Load Temporary Add-on"
4. Navigate to `dist/firefox` and select the `manifest.json` file
5. The extension will appear in your browser toolbar

### Installing in Chrome (Manifest V2) or Edge
1. Open your browser and navigate to the extensions page
   - Chrome: `chrome://extensions/`
   - Edge: `edge://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" and select the `dist/chrome-v2` folder
4. The extension will appear in your browser toolbar

## Configuration

1. Click the extension icon and then click "Settings" at the bottom
2. Configure your TLDW server settings:
   - **Server URL**: Your TLDW server address (e.g., `http://localhost:8000`)
   - **API Token**: Your Bearer token from the server's `API_BEARER` environment variable
3. Test the connection to ensure everything is working
4. Configure default settings for chat and behavior preferences

## Usage

### Quick Start
1. Click the extension icon in your toolbar to open the popup
2. Select a tab for the feature you want to use
3. For chat, select a model and start typing
4. For media, paste a URL or upload a file

### Keyboard Shortcuts
- `Ctrl+Shift+T` (Windows/Linux) or `Cmd+Shift+T` (Mac): Open extension popup
- `Ctrl+Shift+C` (Windows/Linux) or `Cmd+Shift+C` (Mac): Send selected text to chat

### Context Menu Actions
Right-click on selected text or media to:
- Send to TLDW Chat
- Process as Media
- Save as Prompt

### Web Page Interaction
1. Select any text on a webpage
2. A floating button will appear
3. Click it to see quick action options
4. Choose an action to process the selected content

## API Endpoints Used

The extension interacts with the following TLDW server endpoints:

### Chat API
- `POST /api/v1/chat/completions` - Create chat completions

### Prompts API
- `GET/POST /api/v1/prompts/` - List and create prompts
- `GET /api/v1/prompts/export` - Export prompts
- `POST /api/v1/prompts/search` - Search prompts

### Characters API
- `GET /api/v1/characters/` - List characters
- `POST /api/v1/characters/import` - Import character cards
- `GET /api/v1/characters/search/` - Search characters

### Media API
- `GET /api/v1/media/` - List media items
- `POST /api/v1/media/add` - Add media from URL
- `POST /api/v1/media/process-*` - Process various file types
- `POST /api/v1/media/ingest-web-content` - Ingest web content

## Security Notes

- API tokens are stored securely in Chrome's sync storage
- All API requests include proper authentication headers
- The extension only requests necessary permissions
- No data is sent to third parties

## Troubleshooting

### Connection Issues
1. Verify your server is running and accessible
2. Check that your API token is correct
3. Ensure the server URL includes the protocol (http:// or https://)
4. Test the connection from the settings page

### Missing Features
If certain features aren't working:
1. Check the browser console for errors (F12 → Console tab)
2. Ensure you have the latest version of the extension
3. Verify your server supports the required endpoints

## Browser Compatibility

| Browser | Minimum Version | Manifest Version | Notes |
|---------|----------------|------------------|-------|
| Chrome | 88+ | V3 | Full support |
| Chrome | 42-87 | V2 | Use chrome-v2 build |
| Firefox | 78+ | V2 | Full support |
| Edge | 79+ | V2/V3 | Based on Chromium |
| Opera | 75+ | V2/V3 | Based on Chromium |
| Brave | All | V2/V3 | Based on Chromium |

## Development

To modify the extension:
1. Edit the source files in the `chrome-extension` directory
2. Run `npm run build` to generate distribution files
3. Reload the extension in your browser
4. Test your changes

### File Structure
```
chrome-extension/
├── manifest.json          # Chrome V3 manifest
├── manifest-v2.json       # Firefox/Chrome V2 manifest
├── build.js              # Build script
├── html/
│   ├── popup.html        # Main popup interface
│   └── options.html      # Settings page
├── js/
│   ├── popup.js          # Popup functionality
│   ├── options.js        # Settings page logic
│   ├── background.js     # Chrome V3 service worker
│   ├── background-v2.js  # V2 background script
│   ├── content.js        # Content script
│   ├── browser-polyfill.js # Cross-browser compatibility
│   ├── compat-utils.js   # Compatibility utilities
│   └── utils/
│       └── api.js        # API client
├── css/
│   ├── popup.css         # Popup styles
│   └── options.css       # Settings page styles
├── icons/                # Extension icons
├── tests/                # Test files
└── dist/                 # Built extensions (after build)
    ├── chrome-v3/        # Chrome V3 build
    ├── chrome-v2/        # Chrome V2 build
    └── firefox/          # Firefox build
```

### Cross-Browser Development Tips

1. **Use the browser polyfill**: Always use `browserAPI` instead of direct `chrome` calls
2. **Test in multiple browsers**: Firefox and Chrome have subtle differences
3. **Check permissions**: Firefox may require additional permissions
4. **Manifest differences**: V2 uses `browser_action`, V3 uses `action`
5. **Background scripts**: V2 uses persistent background pages, V3 uses service workers

## Testing

The extension includes comprehensive test coverage using Jest.

### Setup

```bash
# Install dependencies
npm install
```

### Running Tests

```bash
# Run all tests
npm test

# Run tests in watch mode
npm run test:watch

# Run tests with coverage report
npm run test:coverage

# Run only unit tests
npm run test:unit

# Run only integration tests
npm run test:integration
```

### Test Structure

```
tests/
├── setup.js              # Jest configuration and global mocks
├── unit/                 # Unit tests for individual components
│   ├── api.test.js       # API client tests
│   ├── popup.test.js     # Popup interface tests
│   ├── background.test.js # Background service worker tests
│   ├── content.test.js   # Content script tests
│   └── options.test.js   # Options page tests
└── integration/          # Integration tests
    └── api-integration.test.js # Full API workflow tests
```

### Test Coverage

The test suite covers:

- **API Client**: All CRUD operations, error handling, streaming responses
- **Popup Interface**: Tab switching, chat functionality, prompt management, character selection
- **Background Worker**: Context menus, keyboard shortcuts, message handling, connection monitoring
- **Content Script**: Text selection, floating button, quick actions, element highlighting
- **Options Page**: Settings management, import/export, connection testing
- **Integration**: Complete user workflows, concurrent operations, error scenarios

### Writing New Tests

When adding new features:

1. Write unit tests for individual functions/components
2. Add integration tests for complete workflows
3. Ensure minimum 70% code coverage
4. Mock external dependencies (Chrome APIs, fetch)
5. Test both success and error scenarios

### Mocking Chrome APIs

The test setup includes comprehensive Chrome API mocks in `tests/setup.js`. These mocks simulate:
- Storage API (sync and local)
- Runtime messaging
- Context menus
- Tabs API
- Downloads
- Notifications

### Debugging Tests

To debug a specific test:
1. Add `debugger` statement in your test
2. Run: `node --inspect-brk node_modules/.bin/jest --runInBand`
3. Open Chrome DevTools for Node.js debugging

## License

This extension is part of the TLDW Server project and follows the same license terms.