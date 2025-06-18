# Loading the Extension in Different Browsers

## Quick Guide

### For Firefox:
1. Use `manifest-v2.json` (NOT `manifest.json`)
2. Either:
   - Build first: `npm run build:firefox` and load from `dist/firefox/`
   - Or manually: Copy `manifest-v2.json` to `manifest.json` in your working directory

### For Chrome (version 88+):
1. Use `manifest.json` (the default one with manifest_version: 3)
2. Either:
   - Build first: `npm run build:chrome-v3` and load from `dist/chrome-v3/`
   - Or load directly from the source directory

### For Chrome (version 87 and older):
1. Use `manifest-v2.json`
2. Build first: `npm run build:chrome-v2` and load from `dist/chrome-v2/`

## Detailed Instructions

### Firefox Development (Temporary Load)
```bash
# Option 1: Use the build system
npm run build:firefox
# Then in Firefox: about:debugging → This Firefox → Load Temporary Add-on → Select dist/firefox/manifest.json

# Option 2: Manual setup for development
cp manifest-v2.json manifest.json
# Then in Firefox: about:debugging → This Firefox → Load Temporary Add-on → Select manifest.json
```

### Chrome Development
```bash
# For Chrome 88+
# Just load the extension directory as-is from chrome://extensions/

# For older Chrome
npm run build:chrome-v2
# Then load from dist/chrome-v2/
```

## Common Issues

### "background.service_worker is currently disabled"
- **Cause**: You're loading the V3 manifest (manifest.json) in Firefox
- **Fix**: Use manifest-v2.json instead

### "Cannot read property 'browserAction' of undefined"
- **Cause**: Code is trying to use V2 APIs in a V3 context or vice versa
- **Fix**: Make sure you're using the correct build for your browser

### Extension doesn't appear after loading
- **Cause**: Missing icons or permission issues
- **Fix**: Generate icons first using icons/icon-generator.html

## Testing Your Setup

After loading the extension:
1. Click the extension icon - the popup should open
2. Right-click on a webpage - you should see "Send to TLDW Chat" options
3. Go to extension options/settings - the options page should load

If any of these don't work, check the browser console for errors (about:debugging in Firefox, chrome://extensions/ in Chrome with Developer mode on).