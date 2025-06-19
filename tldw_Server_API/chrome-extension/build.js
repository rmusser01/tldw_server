#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Build configuration
const BUILD_DIR = 'dist';
const CHROME_V3_DIR = path.join(BUILD_DIR, 'chrome-v3');
const CHROME_V2_DIR = path.join(BUILD_DIR, 'chrome-v2');
const FIREFOX_DIR = path.join(BUILD_DIR, 'firefox');

// Files to copy
const FILES_TO_COPY = [
  'html/',
  'css/',
  'icons/',
  'js/browser-polyfill.js',
  'js/compat-utils.js',
  'js/content.js',
  'js/options.js',
  'js/popup.js',
  'js/utils/',
  'js/background-v2.js'
];

// Clean build directory
function cleanBuildDir() {
  console.log('Cleaning build directory...');
  if (fs.existsSync(BUILD_DIR)) {
    fs.rmSync(BUILD_DIR, { recursive: true, force: true });
  }
  fs.mkdirSync(BUILD_DIR);
  fs.mkdirSync(CHROME_V3_DIR);
  fs.mkdirSync(CHROME_V2_DIR);
  fs.mkdirSync(FIREFOX_DIR);
}

// Copy files recursively
function copyRecursive(src, dest) {
  const exists = fs.existsSync(src);
  const stats = exists && fs.statSync(src);
  const isDirectory = exists && stats.isDirectory();
  
  if (isDirectory) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    fs.readdirSync(src).forEach(childItemName => {
      copyRecursive(
        path.join(src, childItemName),
        path.join(dest, childItemName)
      );
    });
  } else {
    const destDir = path.dirname(dest);
    if (!fs.existsSync(destDir)) {
      fs.mkdirSync(destDir, { recursive: true });
    }
    fs.copyFileSync(src, dest);
  }
}

// Copy common files
function copyCommonFiles(targetDir) {
  console.log(`Copying files to ${targetDir}...`);
  
  FILES_TO_COPY.forEach(file => {
    const src = path.join('.', file);
    const dest = path.join(targetDir, file);
    
    if (fs.existsSync(src)) {
      copyRecursive(src, dest);
    }
  });
}

// Build Chrome V3 version
function buildChromeV3() {
  console.log('\nBuilding Chrome V3 extension...');
  
  // Copy common files
  copyCommonFiles(CHROME_V3_DIR);
  
  // Copy V3 specific files
  fs.copyFileSync('manifest.json', path.join(CHROME_V3_DIR, 'manifest.json'));
  fs.copyFileSync('js/background.js', path.join(CHROME_V3_DIR, 'js/background.js'));
  
  console.log('Chrome V3 build complete!');
}

// Build Chrome V2 version
function buildChromeV2() {
  console.log('\nBuilding Chrome V2 extension...');
  
  // Copy common files
  copyCommonFiles(CHROME_V2_DIR);
  
  // Copy V2 specific files
  fs.copyFileSync('manifest-v2.json', path.join(CHROME_V2_DIR, 'manifest.json'));
  
  console.log('Chrome V2 build complete!');
}

// Build Firefox version
function buildFirefox() {
  console.log('\nBuilding Firefox extension...');
  
  // Copy common files
  copyCommonFiles(FIREFOX_DIR);
  
  // Copy V2 manifest (Firefox uses V2)
  fs.copyFileSync('manifest-v2.json', path.join(FIREFOX_DIR, 'manifest.json'));
  
  // Skip updating scripts for Firefox - they already have browser API compatibility
  // updateScriptsForFirefox(FIREFOX_DIR);
  
  console.log('Firefox build complete!');
  console.log('To load in Firefox: about:debugging → This Firefox → Load Temporary Add-on');
  console.log(`Select: ${path.join(FIREFOX_DIR, 'manifest.json')}`);
}

// Update scripts for Firefox compatibility
function updateScriptsForFirefox(targetDir) {
  const scriptsToUpdate = [
    // popup.js already has browser API compatibility built in
    // 'js/popup.js',
    'js/options.js',
    'js/content.js'
  ];
  
  scriptsToUpdate.forEach(scriptPath => {
    const fullPath = path.join(targetDir, scriptPath);
    if (fs.existsSync(fullPath)) {
      let content = fs.readFileSync(fullPath, 'utf8');
      
      // Add compat-utils import at the beginning if not already present
      if (!content.includes('compat-utils')) {
        const compatImport = `// Import compatibility utilities\n` +
          `const { getBrowserAPI, storage, tabs, runtime, downloads, notifications } = ` +
          `typeof module !== 'undefined' ? require('./compat-utils.js') : window;\n\n`;
        
        content = compatImport + content;
        
        // Replace direct chrome API calls
        content = content.replace(/chrome\.storage/g, 'storage');
        content = content.replace(/chrome\.tabs/g, 'tabs');
        content = content.replace(/chrome\.runtime/g, 'runtime');
        content = content.replace(/chrome\.downloads/g, 'downloads');
        content = content.replace(/chrome\.notifications/g, 'notifications');
      }
      
      fs.writeFileSync(fullPath, content);
    }
  });
}

// Create zip files
function createZipFiles() {
  console.log('\nCreating zip files...');
  
  const zipCommands = [
    { dir: CHROME_V3_DIR, output: 'tldw-assistant-chrome-v3.zip' },
    { dir: CHROME_V2_DIR, output: 'tldw-assistant-chrome-v2.zip' },
    { dir: FIREFOX_DIR, output: 'tldw-assistant-firefox.zip' }
  ];
  
  zipCommands.forEach(({ dir, output }) => {
    const outputPath = path.join(BUILD_DIR, output);
    try {
      execSync(`cd ${dir} && zip -r ../${output} ./*`, { stdio: 'inherit' });
      console.log(`Created ${output}`);
    } catch (error) {
      console.error(`Failed to create ${output}:`, error.message);
    }
  });
}

// Main build function
function build() {
  console.log('Starting extension build process...');
  
  cleanBuildDir();
  buildChromeV3();
  buildChromeV2();
  buildFirefox();
  
  // Create zip files if zip command is available
  try {
    execSync('which zip', { stdio: 'ignore' });
    createZipFiles();
  } catch {
    console.log('\nZip command not found. Skipping zip file creation.');
  }
  
  console.log('\nBuild complete! Extensions are in the dist/ directory.');
  console.log('- Chrome V3: dist/chrome-v3/');
  console.log('- Chrome V2: dist/chrome-v2/');
  console.log('- Firefox: dist/firefox/');
}

// Run build
build();