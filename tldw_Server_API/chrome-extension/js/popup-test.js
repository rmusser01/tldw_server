// Minimal test file
console.log('popup-test.js loaded');

document.addEventListener('DOMContentLoaded', () => {
  console.log('DOMContentLoaded fired in test file');
  
  // Try to find tab buttons
  const tabButtons = document.querySelectorAll('.tab-button');
  console.log('Found tab buttons:', tabButtons.length);
  
  // Try to add click listener to first button
  if (tabButtons.length > 0) {
    tabButtons[0].addEventListener('click', () => {
      console.log('Tab button clicked!');
      alert('Tab button clicked!');
    });
  }
  
  // Try settings button
  const settingsBtn = document.getElementById('openOptions');
  if (settingsBtn) {
    console.log('Found settings button');
    settingsBtn.addEventListener('click', (e) => {
      e.preventDefault();
      console.log('Settings clicked');
      alert('Settings clicked!');
    });
  }
});

// Also log immediately
console.log('popup-test.js executing');