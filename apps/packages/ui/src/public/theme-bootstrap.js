// Set theme class synchronously to prevent flash of white background
(function() {
  var theme = null;
  try {
    theme = localStorage.getItem('theme');
  } catch (error) {
    if (!error || error.name !== 'SecurityError') {
      throw error;
    }
    // Storage may be blocked by browser policy; use the system preference instead.
  }
  if (theme === 'dark' || (!theme && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
    document.documentElement.classList.add('dark');
  }
})();
