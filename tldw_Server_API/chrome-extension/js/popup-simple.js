console.log('popup-simple.js loaded');

document.addEventListener('DOMContentLoaded', () => {
  console.log('DOMContentLoaded in simple popup');
  
  const btn = document.getElementById('testBtn');
  if (btn) {
    btn.addEventListener('click', () => {
      console.log('Button clicked');
      document.getElementById('output').textContent = 'Button was clicked!';
    });
  }
});