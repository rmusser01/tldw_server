// Content script for TLDW Server Assistant
// Runs on all web pages to enable interaction with page content

// Browser API compatibility
const browserAPI = (typeof browser !== 'undefined') ? browser : chrome;

// Listen for messages from the background script
browserAPI.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'getSelection') {
    const selectedText = window.getSelection().toString();
    sendResponse({ text: selectedText });
  } else if (request.action === 'getPageInfo') {
    sendResponse({
      title: document.title,
      url: window.location.href,
      selectedText: window.getSelection().toString(),
      pageContent: document.body.innerText.substring(0, 5000) // First 5000 chars
    });
  }
});

// Add floating action button for quick access
let floatingButton = null;
let selectionTimeout = null;

// Create floating button
function createFloatingButton() {
  const button = document.createElement('div');
  button.className = 'tldw-floating-button';
  button.innerHTML = `
    <svg width="24" height="24" viewBox="0 0 24 24" fill="white">
      <path d="M20 2H4c-1.1 0-2 .9-2 2v18l4-4h14c1.1 0 2-.9 2-2V4c0-1.1-.9-2-2-2z"/>
    </svg>
  `;
  button.style.cssText = `
    position: fixed;
    bottom: 20px;
    right: 20px;
    width: 48px;
    height: 48px;
    background: #3498db;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    cursor: pointer;
    box-shadow: 0 2px 10px rgba(0,0,0,0.3);
    z-index: 9999;
    transition: all 0.3s ease;
    display: none;
  `;
  
  button.addEventListener('mouseenter', () => {
    button.style.transform = 'scale(1.1)';
  });
  
  button.addEventListener('mouseleave', () => {
    button.style.transform = 'scale(1)';
  });
  
  button.addEventListener('click', handleFloatingButtonClick);
  
  return button;
}

// Handle floating button click
function handleFloatingButtonClick(e) {
  e.stopPropagation();
  
  const selectedText = window.getSelection().toString();
  if (selectedText) {
    showQuickActions(e.pageX, e.pageY, selectedText);
  } else {
    chrome.runtime.sendMessage({ action: 'openPopup' });
  }
}

// Show quick actions menu
function showQuickActions(x, y, text) {
  removeQuickActions();
  
  const menu = document.createElement('div');
  menu.className = 'tldw-quick-actions';
  menu.style.cssText = `
    position: fixed;
    left: ${x}px;
    top: ${y}px;
    background: white;
    border: 1px solid #ddd;
    border-radius: 4px;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    z-index: 10000;
    padding: 8px 0;
    min-width: 180px;
  `;
  
  const actions = [
    { label: 'Send to Chat', action: 'sendToChat' },
    { label: 'Save as Prompt', action: 'saveAsPrompt' },
    { label: 'Process as Media', action: 'processAsMedia' }
  ];
  
  actions.forEach(item => {
    const button = document.createElement('div');
    button.textContent = item.label;
    button.style.cssText = `
      padding: 8px 16px;
      cursor: pointer;
      transition: background 0.2s;
    `;
    button.addEventListener('mouseenter', () => {
      button.style.background = '#f5f5f5';
    });
    button.addEventListener('mouseleave', () => {
      button.style.background = 'white';
    });
    button.addEventListener('click', () => {
      handleQuickAction(item.action, text);
      removeQuickActions();
    });
    menu.appendChild(button);
  });
  
  document.body.appendChild(menu);
  
  // Remove menu when clicking outside
  setTimeout(() => {
    document.addEventListener('click', removeQuickActions);
  }, 100);
}

function removeQuickActions() {
  const menu = document.querySelector('.tldw-quick-actions');
  if (menu) {
    menu.remove();
  }
  document.removeEventListener('click', removeQuickActions);
}

function handleQuickAction(action, text) {
  browserAPI.runtime.sendMessage({
    action: 'processSelection',
    data: {
      type: action,
      text: text,
      url: window.location.href,
      title: document.title
    }
  });
}

// Monitor text selection
document.addEventListener('mouseup', () => {
  clearTimeout(selectionTimeout);
  selectionTimeout = setTimeout(() => {
    const selectedText = window.getSelection().toString().trim();
    if (selectedText && selectedText.length > 10) {
      if (floatingButton) {
        floatingButton.style.display = 'flex';
      }
    } else {
      if (floatingButton) {
        floatingButton.style.display = 'none';
      }
    }
  }, 500);
});

// Initialize content script
function initContentScript() {
  // Check if we should show floating button
  browserAPI.storage.sync.get(['showFloatingButton'], (result) => {
    if (result.showFloatingButton !== false) {
      floatingButton = createFloatingButton();
      document.body.appendChild(floatingButton);
    }
  });
  
  // Add keyboard shortcuts
  document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Shift + T: Send selection to chat
    if ((e.ctrlKey || e.metaKey) && e.shiftKey && e.key === 'T') {
      e.preventDefault();
      const selectedText = window.getSelection().toString();
      if (selectedText) {
        handleQuickAction('sendToChat', selectedText);
      }
    }
  });
}

// Highlight elements on hover (for future element selection feature)
let highlightedElement = null;

function enableElementHighlight() {
  document.addEventListener('mouseover', highlightElement);
  document.addEventListener('mouseout', unhighlightElement);
  document.addEventListener('click', selectElement);
}

function disableElementHighlight() {
  document.removeEventListener('mouseover', highlightElement);
  document.removeEventListener('mouseout', unhighlightElement);
  document.removeEventListener('click', selectElement);
  unhighlightElement();
}

function highlightElement(e) {
  if (highlightedElement) {
    unhighlightElement();
  }
  
  highlightedElement = e.target;
  highlightedElement.style.outline = '2px solid #3498db';
  highlightedElement.style.outlineOffset = '2px';
}

function unhighlightElement() {
  if (highlightedElement) {
    highlightedElement.style.outline = '';
    highlightedElement.style.outlineOffset = '';
    highlightedElement = null;
  }
}

function selectElement(e) {
  e.preventDefault();
  e.stopPropagation();
  
  if (highlightedElement) {
    const content = highlightedElement.innerText || highlightedElement.textContent;
    handleQuickAction('processAsMedia', content);
    disableElementHighlight();
  }
}

// Listen for commands from extension
browserAPI.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === 'enableElementSelection') {
    enableElementHighlight();
    sendResponse({ success: true });
  } else if (request.action === 'disableElementSelection') {
    disableElementHighlight();
    sendResponse({ success: true });
  }
});

// Initialize when DOM is ready
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initContentScript);
} else {
  initContentScript();
}