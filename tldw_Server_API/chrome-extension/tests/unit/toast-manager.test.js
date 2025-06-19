/**
 * @jest-environment jsdom
 */

import { waitForAsync, cleanupDOM } from '../utils/helpers.js';
import '../utils/matchers.js';

describe('Toast Manager', () => {
  let ToastManager;
  let toastManager;

  beforeEach(() => {
    // Clean DOM
    document.body.innerHTML = '';
    
    // Mock ToastManager class
    ToastManager = class {
      constructor() {
        this.container = null;
        this.toasts = new Set();
      }

      init() {
        this.container = document.getElementById('toast-container');
        if (!this.container) {
          this.container = document.createElement('div');
          this.container.id = 'toast-container';
          this.container.className = 'toast-container';
          document.body.appendChild(this.container);
        }
      }

      show(message, type = 'info', duration = 3000) {
        if (!this.container) return;

        const toast = document.createElement('div');
        toast.className = `toast toast-${type}`;
        
        const icon = document.createElement('span');
        icon.className = 'toast-icon';
        icon.textContent = this.getIcon(type);
        
        const messageEl = document.createElement('span');
        messageEl.className = 'toast-message';
        messageEl.textContent = message;
        
        const closeButton = document.createElement('button');
        closeButton.className = 'toast-close';
        closeButton.textContent = '×';
        closeButton.onclick = () => this.hide(toast);
        
        toast.appendChild(icon);
        toast.appendChild(messageEl);
        toast.appendChild(closeButton);
        
        this.container.appendChild(toast);
        this.toasts.add(toast);
        
        // Auto-hide after duration
        if (duration > 0) {
          setTimeout(() => this.hide(toast), duration);
        }
        
        return toast;
      }

      hide(toast) {
        if (!toast || !this.toasts.has(toast)) return;
        
        toast.classList.add('toast-hiding');
        setTimeout(() => {
          if (toast.parentNode) {
            toast.parentNode.removeChild(toast);
          }
          this.toasts.delete(toast);
        }, 300);
      }

      success(message, duration) {
        return this.show(message, 'success', duration);
      }

      error(message, duration) {
        return this.show(message, 'error', duration);
      }

      warning(message, duration) {
        return this.show(message, 'warning', duration);
      }

      info(message, duration) {
        return this.show(message, 'info', duration);
      }

      loading(message) {
        return this.show(message, 'loading', 0);
      }

      clear() {
        this.toasts.forEach(toast => this.hide(toast));
      }

      getIcon(type) {
        const icons = {
          success: '✓',
          error: '✕',
          warning: '⚠',
          info: 'ℹ',
          loading: '⟳'
        };
        return icons[type] || icons.info;
      }
    };

    toastManager = new ToastManager();
  });

  afterEach(() => {
    cleanupDOM();
    jest.clearAllMocks();
  });

  describe('Initialization', () => {
    test('should create container if not exists', () => {
      expect(document.getElementById('toast-container')).toBeNull();
      
      toastManager.init();
      
      const container = document.getElementById('toast-container');
      expect(container).toBeDefined();
      expect(container.className).toBe('toast-container');
    });

    test('should use existing container if present', () => {
      const existingContainer = document.createElement('div');
      existingContainer.id = 'toast-container';
      existingContainer.className = 'existing-container';
      document.body.appendChild(existingContainer);
      
      toastManager.init();
      
      expect(toastManager.container).toBe(existingContainer);
      expect(toastManager.container.className).toBe('existing-container');
    });
  });

  describe('Toast Creation', () => {
    beforeEach(() => {
      toastManager.init();
    });

    test('should create success toast', () => {
      const toast = toastManager.success('Operation successful');
      
      expect(toast).toBeDefined();
      expect(toast.className).toContain('toast-success');
      expect(document).toHaveToastNotification('success', 'Operation successful');
    });

    test('should create error toast', () => {
      const toast = toastManager.error('Operation failed');
      
      expect(toast.className).toContain('toast-error');
      expect(document).toHaveToastNotification('error', 'Operation failed');
    });

    test('should create warning toast', () => {
      const toast = toastManager.warning('Be careful!');
      
      expect(toast.className).toContain('toast-warning');
      expect(document).toHaveToastNotification('warning', 'Be careful!');
    });

    test('should create info toast', () => {
      const toast = toastManager.info('For your information');
      
      expect(toast.className).toContain('toast-info');
      expect(document).toHaveToastNotification('info', 'For your information');
    });

    test('should create loading toast without auto-hide', () => {
      const toast = toastManager.loading('Processing...');
      
      expect(toast.className).toContain('toast-loading');
      expect(document).toHaveToastNotification('loading', 'Processing...');
    });

    test('should include correct icons', () => {
      const toasts = {
        success: toastManager.success('Success'),
        error: toastManager.error('Error'),
        warning: toastManager.warning('Warning'),
        info: toastManager.info('Info'),
        loading: toastManager.loading('Loading')
      };

      expect(toasts.success.querySelector('.toast-icon').textContent).toBe('✓');
      expect(toasts.error.querySelector('.toast-icon').textContent).toBe('✕');
      expect(toasts.warning.querySelector('.toast-icon').textContent).toBe('⚠');
      expect(toasts.info.querySelector('.toast-icon').textContent).toBe('ℹ');
      expect(toasts.loading.querySelector('.toast-icon').textContent).toBe('⟳');
    });

    test('should include close button', () => {
      const toast = toastManager.info('Test message');
      const closeButton = toast.querySelector('.toast-close');
      
      expect(closeButton).toBeDefined();
      expect(closeButton.textContent).toBe('×');
    });

    test('should not create toast without container', () => {
      toastManager.container = null;
      
      const toast = toastManager.info('Test');
      
      expect(toast).toBeUndefined();
    });
  });

  describe('Toast Auto-hide', () => {
    beforeEach(() => {
      toastManager.init();
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    test('should auto-hide after default duration', () => {
      const toast = toastManager.info('Auto-hide test');
      
      expect(toastManager.toasts.has(toast)).toBe(true);
      
      // Fast-forward default duration (3000ms)
      jest.advanceTimersByTime(3000);
      
      expect(toast.classList.contains('toast-hiding')).toBe(true);
      
      // Fast-forward animation duration
      jest.advanceTimersByTime(300);
      
      expect(toastManager.toasts.has(toast)).toBe(false);
      expect(toast.parentNode).toBeNull();
    });

    test('should respect custom duration', () => {
      const toast = toastManager.info('Custom duration', 5000);
      
      jest.advanceTimersByTime(3000);
      expect(toastManager.toasts.has(toast)).toBe(true);
      
      jest.advanceTimersByTime(2000);
      expect(toast.classList.contains('toast-hiding')).toBe(true);
    });

    test('should not auto-hide loading toasts', () => {
      const toast = toastManager.loading('Loading...');
      
      jest.advanceTimersByTime(10000);
      
      expect(toastManager.toasts.has(toast)).toBe(true);
      expect(toast.parentNode).toBe(toastManager.container);
    });

    test('should not auto-hide with duration 0', () => {
      const toast = toastManager.show('No auto-hide', 'info', 0);
      
      jest.advanceTimersByTime(10000);
      
      expect(toastManager.toasts.has(toast)).toBe(true);
    });
  });

  describe('Toast Manual Hide', () => {
    beforeEach(() => {
      toastManager.init();
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    test('should hide toast on close button click', () => {
      const toast = toastManager.info('Click to close');
      const closeButton = toast.querySelector('.toast-close');
      
      closeButton.click();
      
      expect(toast.classList.contains('toast-hiding')).toBe(true);
      
      jest.advanceTimersByTime(300);
      
      expect(toastManager.toasts.has(toast)).toBe(false);
      expect(toast.parentNode).toBeNull();
    });

    test('should hide specific toast', () => {
      const toast1 = toastManager.info('Toast 1');
      const toast2 = toastManager.info('Toast 2');
      
      toastManager.hide(toast1);
      
      jest.advanceTimersByTime(300);
      
      expect(toastManager.toasts.has(toast1)).toBe(false);
      expect(toastManager.toasts.has(toast2)).toBe(true);
    });

    test('should not hide invalid toast', () => {
      const fakeToast = document.createElement('div');
      
      // Should not throw
      expect(() => toastManager.hide(fakeToast)).not.toThrow();
      expect(() => toastManager.hide(null)).not.toThrow();
    });
  });

  describe('Multiple Toasts', () => {
    beforeEach(() => {
      toastManager.init();
    });

    test('should handle multiple toasts', () => {
      const toast1 = toastManager.success('First');
      const toast2 = toastManager.error('Second');
      const toast3 = toastManager.info('Third');
      
      expect(toastManager.toasts.size).toBe(3);
      expect(toastManager.container.children.length).toBe(3);
    });

    test('should stack toasts vertically', () => {
      const toasts = [
        toastManager.info('Toast 1'),
        toastManager.info('Toast 2'),
        toastManager.info('Toast 3')
      ];
      
      // Check they're all in the container
      toasts.forEach(toast => {
        expect(toast.parentNode).toBe(toastManager.container);
      });
      
      // They should appear in order
      expect(toastManager.container.children[0]).toBe(toasts[0]);
      expect(toastManager.container.children[1]).toBe(toasts[1]);
      expect(toastManager.container.children[2]).toBe(toasts[2]);
    });

    test('should clear all toasts', () => {
      toastManager.success('Toast 1');
      toastManager.error('Toast 2');
      toastManager.info('Toast 3');
      
      expect(toastManager.toasts.size).toBe(3);
      
      toastManager.clear();
      
      // All should be marked for hiding
      toastManager.toasts.forEach(toast => {
        expect(toast.classList.contains('toast-hiding')).toBe(true);
      });
    });
  });

  describe('Edge Cases', () => {
    beforeEach(() => {
      toastManager.init();
    });

    test('should handle empty messages', () => {
      const toast = toastManager.info('');
      
      expect(toast).toBeDefined();
      expect(toast.querySelector('.toast-message').textContent).toBe('');
    });

    test('should handle very long messages', () => {
      const longMessage = 'A'.repeat(1000);
      const toast = toastManager.info(longMessage);
      
      expect(toast.querySelector('.toast-message').textContent).toBe(longMessage);
    });

    test('should handle special characters', () => {
      const specialMessage = '<script>alert("XSS")</script> & "quotes" \'single\'';
      const toast = toastManager.info(specialMessage);
      
      // Should be text content, not HTML
      expect(toast.querySelector('.toast-message').textContent).toBe(specialMessage);
      expect(toast.querySelector('script')).toBeNull();
    });

    test('should handle rapid creation and deletion', () => {
      jest.useFakeTimers();
      
      // Create many toasts rapidly
      for (let i = 0; i < 10; i++) {
        toastManager.info(`Toast ${i}`, 100);
      }
      
      expect(toastManager.toasts.size).toBe(10);
      
      // Advance time to hide them all
      jest.advanceTimersByTime(400);
      
      expect(toastManager.toasts.size).toBe(0);
      
      jest.useRealTimers();
    });

    test('should handle undefined type', () => {
      const toast = toastManager.show('Test', undefined);
      
      expect(toast.className).toContain('toast-info'); // Should default to info
    });
  });

  describe('Memory Management', () => {
    beforeEach(() => {
      toastManager.init();
      jest.useFakeTimers();
    });

    afterEach(() => {
      jest.useRealTimers();
    });

    test('should clean up references after hiding', () => {
      const toast = toastManager.info('Memory test');
      const initialSize = toastManager.toasts.size;
      
      toastManager.hide(toast);
      jest.advanceTimersByTime(300);
      
      expect(toastManager.toasts.size).toBe(initialSize - 1);
      expect(toastManager.toasts.has(toast)).toBe(false);
    });

    test('should not leak memory with auto-hide', () => {
      // Create many toasts
      for (let i = 0; i < 100; i++) {
        toastManager.info(`Toast ${i}`, 100);
      }
      
      expect(toastManager.toasts.size).toBe(100);
      
      // Let them all auto-hide
      jest.advanceTimersByTime(500);
      
      expect(toastManager.toasts.size).toBe(0);
      expect(toastManager.container.children.length).toBe(0);
    });
  });
});