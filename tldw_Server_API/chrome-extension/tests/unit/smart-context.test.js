/**
 * @jest-environment jsdom
 */

import { waitForAsync, createMockBrowserAPI } from '../utils/helpers.js';
import { createMockTab } from '../utils/factories.js';

describe('Smart Context Detection System', () => {
  let SmartContextDetector;
  let detector;

  beforeEach(() => {
    // Mock the SmartContextDetector class
    SmartContextDetector = class {
      constructor() {
        this.videoPatterns = [
          /youtube\.com\/watch/i,
          /youtu\.be\//i,
          /vimeo\.com\//i,
          /twitch\.tv\//i,
          /dailymotion\.com\//i,
          /facebook\.com\/.*\/videos\//i,
          /instagram\.com\/(p|tv|reel)\//i,
          /tiktok\.com\/@.*\/video\//i,
          /twitter\.com\/.*\/status\/.*\/video\//i,
          /reddit\.com\/r\/.*\/comments\/.*\/(v\.redd\.it|youtube)/i
        ];
        
        this.audioPatterns = [
          /spotify\.com\/(track|album|playlist|episode|show)/i,
          /soundcloud\.com\//i,
          /anchor\.fm\//i,
          /podcasts\.apple\.com\//i,
          /podcasts\.google\.com\//i,
          /overcast\.fm\//i,
          /castbox\.fm\//i,
          /stitcher\.com\//i
        ];
        
        this.articlePatterns = [
          /medium\.com\//i,
          /dev\.to\//i,
          /hashnode\./i,
          /substack\.com\//i,
          /github\.com\/.*\/blob\//i,
          /stackoverflow\.com\/questions\//i,
          /reddit\.com\/r\/.*\/comments\//i,
          /news\.ycombinator\.com\//i,
          /wikipedia\.org\/wiki\//i,
          /arxiv\.org\/abs\//i
        ];
        
        this.documentPatterns = [
          /\.pdf($|\?)/i,
          /docs\.google\.com\/(document|spreadsheets|presentation)/i,
          /drive\.google\.com\/file\/.*\/view/i,
          /dropbox\.com\/.*\.pdf/i,
          /onedrive\.live\.com\/.*\.pdf/i
        ];
      }
      
      detectContentType(url, title = '', pageContent = '') {
        const urlLower = url.toLowerCase();
        
        // Check for video content
        if (this.videoPatterns.some(pattern => pattern.test(url))) {
          return {
            type: 'video',
            confidence: 0.9,
            suggestedAction: 'process-videos',
            icon: '🎥',
            description: 'Video content detected'
          };
        }
        
        // Check for audio content
        if (this.audioPatterns.some(pattern => pattern.test(url))) {
          return {
            type: 'audio',
            confidence: 0.9,
            suggestedAction: 'process-audios',
            icon: '🎵',
            description: 'Audio content detected'
          };
        }
        
        // Check for document content
        if (this.documentPatterns.some(pattern => pattern.test(url))) {
          return {
            type: 'document',
            confidence: 0.85,
            suggestedAction: 'process-documents',
            icon: '📄',
            description: 'Document content detected'
          };
        }
        
        // Check for article content
        if (this.articlePatterns.some(pattern => pattern.test(url))) {
          return {
            type: 'article',
            confidence: 0.8,
            suggestedAction: 'process-url',
            icon: '📰',
            description: 'Article content detected'
          };
        }
        
        // Advanced content analysis
        return this.analyzePageContent(url, title, pageContent);
      }
      
      analyzePageContent(url, title, pageContent) {
        let confidence = 0.5;
        let type = 'webpage';
        let suggestedAction = 'process-url';
        let icon = '🌐';
        let description = 'Web page content';
        
        // Check for code content
        if (this.hasCodeContent(url, title, pageContent)) {
          return {
            type: 'code',
            confidence: 0.75,
            suggestedAction: 'save-as-prompt',
            icon: '💻',
            description: 'Code content detected'
          };
        }
        
        // Check for long-form content
        if (pageContent.length > 3000) {
          confidence = 0.7;
          type = 'article';
          icon = '📖';
          description = 'Long-form content detected';
        }
        
        // Check for social media content
        if (this.isSocialMediaContent(url)) {
          return {
            type: 'social',
            confidence: 0.6,
            suggestedAction: 'send-to-chat',
            icon: '💬',
            description: 'Social media content'
          };
        }
        
        return { type, confidence, suggestedAction, icon, description };
      }
      
      hasCodeContent(url, title, content) {
        const codeIndicators = [
          /github\.com/i,
          /gitlab\.com/i,
          /bitbucket\.org/i,
          /codepen\.io/i,
          /jsfiddle\.net/i,
          /stackoverflow\.com/i,
          /\b(function|class|import|export|const|let|var)\b/i,
          /\b(def|class|import|from)\b/i, // Python
          /\b(public|private|protected|static)\b/i, // Java/C#
          /<\?php/i,
          /\{\s*\}/i,
          /\[\s*\]/i
        ];
        
        return codeIndicators.some(pattern => 
          pattern.test(url) || pattern.test(title) || pattern.test(content)
        );
      }
      
      isSocialMediaContent(url) {
        const socialPatterns = [
          /twitter\.com/i,
          /facebook\.com/i,
          /instagram\.com/i,
          /linkedin\.com/i,
          /tiktok\.com/i,
          /discord\.com/i,
          /telegram\.org/i
        ];
        
        return socialPatterns.some(pattern => pattern.test(url));
      }
      
      getSuggestedActions(context) {
        const actions = [];
        
        switch (context.type) {
          case 'video':
            actions.push(
              { label: 'Process Video', action: 'process-videos', primary: true },
              { label: 'Send to Chat', action: 'send-to-chat' },
              { label: 'Save URL', action: 'save-url' }
            );
            break;
            
          case 'audio':
            actions.push(
              { label: 'Process Audio', action: 'process-audios', primary: true },
              { label: 'Send to Chat', action: 'send-to-chat' },
              { label: 'Save URL', action: 'save-url' }
            );
            break;
            
          case 'document':
            actions.push(
              { label: 'Process Document', action: 'process-documents', primary: true },
              { label: 'Send to Chat', action: 'send-to-chat' }
            );
            break;
            
          case 'article':
            actions.push(
              { label: 'Process Article', action: 'process-url', primary: true },
              { label: 'Send to Chat', action: 'send-to-chat' },
              { label: 'Save as Prompt', action: 'save-as-prompt' }
            );
            break;
            
          case 'code':
            actions.push(
              { label: 'Save as Prompt', action: 'save-as-prompt', primary: true },
              { label: 'Send to Chat', action: 'send-to-chat' },
              { label: 'Process URL', action: 'process-url' }
            );
            break;
            
          case 'social':
            actions.push(
              { label: 'Send to Chat', action: 'send-to-chat', primary: true },
              { label: 'Save as Prompt', action: 'save-as-prompt' }
            );
            break;
            
          default:
            actions.push(
              { label: 'Process URL', action: 'process-url', primary: true },
              { label: 'Send to Chat', action: 'send-to-chat' },
              { label: 'Save as Prompt', action: 'save-as-prompt' }
            );
        }
        
        return actions;
      }
    };

    detector = new SmartContextDetector();
  });

  describe('Video Content Detection', () => {
    test.each([
      ['https://www.youtube.com/watch?v=dQw4w9WgXcQ', 'YouTube video'],
      ['https://youtu.be/dQw4w9WgXcQ', 'YouTube short URL'],
      ['https://vimeo.com/123456789', 'Vimeo video'],
      ['https://www.twitch.tv/username', 'Twitch stream'],
      ['https://www.dailymotion.com/video/x123456', 'Dailymotion video'],
      ['https://www.facebook.com/username/videos/123456789', 'Facebook video'],
      ['https://www.instagram.com/p/ABC123/', 'Instagram post'],
      ['https://www.instagram.com/reel/ABC123/', 'Instagram reel'],
      ['https://www.tiktok.com/@username/video/123456789', 'TikTok video'],
      ['https://twitter.com/user/status/123456789/video/1', 'Twitter video'],
      ['https://www.reddit.com/r/videos/comments/abc123/v.redd.it', 'Reddit video']
    ])('should detect video content from %s', (url, description) => {
      const result = detector.detectContentType(url);
      
      expect(result.type).toBe('video');
      expect(result.confidence).toBeGreaterThanOrEqual(0.9);
      expect(result.suggestedAction).toBe('process-videos');
      expect(result.icon).toBe('🎥');
    });
  });

  describe('Audio Content Detection', () => {
    test.each([
      ['https://open.spotify.com/track/123456789', 'Spotify track'],
      ['https://open.spotify.com/album/123456789', 'Spotify album'],
      ['https://open.spotify.com/playlist/123456789', 'Spotify playlist'],
      ['https://open.spotify.com/episode/123456789', 'Spotify podcast episode'],
      ['https://soundcloud.com/artist/track', 'SoundCloud track'],
      ['https://anchor.fm/podcast/episodes/episode-123', 'Anchor podcast'],
      ['https://podcasts.apple.com/podcast/id123456789', 'Apple Podcast'],
      ['https://podcasts.google.com/feed/abc123', 'Google Podcast'],
      ['https://overcast.fm/+ABC123', 'Overcast podcast'],
      ['https://castbox.fm/episode/id123456', 'Castbox episode'],
      ['https://www.stitcher.com/show/podcast-name', 'Stitcher podcast']
    ])('should detect audio content from %s', (url, description) => {
      const result = detector.detectContentType(url);
      
      expect(result.type).toBe('audio');
      expect(result.confidence).toBeGreaterThanOrEqual(0.9);
      expect(result.suggestedAction).toBe('process-audios');
      expect(result.icon).toBe('🎵');
    });
  });

  describe('Document Content Detection', () => {
    test.each([
      ['https://example.com/document.pdf', 'Direct PDF link'],
      ['https://example.com/file.pdf?version=2', 'PDF with query params'],
      ['https://docs.google.com/document/d/123456/edit', 'Google Docs'],
      ['https://docs.google.com/spreadsheets/d/123456/edit', 'Google Sheets'],
      ['https://docs.google.com/presentation/d/123456/edit', 'Google Slides'],
      ['https://drive.google.com/file/d/123456/view', 'Google Drive file'],
      ['https://www.dropbox.com/s/abc123/document.pdf', 'Dropbox PDF'],
      ['https://onedrive.live.com/view.aspx?resid=123456!789&app=WordPdf', 'OneDrive PDF']
    ])('should detect document content from %s', (url, description) => {
      const result = detector.detectContentType(url);
      
      expect(result.type).toBe('document');
      expect(result.confidence).toBeGreaterThanOrEqual(0.85);
      expect(result.suggestedAction).toBe('process-documents');
      expect(result.icon).toBe('📄');
    });
  });

  describe('Article Content Detection', () => {
    test.each([
      ['https://medium.com/@author/article-title-123456', 'Medium article'],
      ['https://dev.to/author/article-slug', 'Dev.to article'],
      ['https://hashnode.dev/article-title', 'Hashnode article'],
      ['https://author.substack.com/p/article-title', 'Substack article'],
      ['https://github.com/owner/repo/blob/main/README.md', 'GitHub file'],
      ['https://stackoverflow.com/questions/123456/question-title', 'Stack Overflow'],
      ['https://www.reddit.com/r/programming/comments/abc123/post_title/', 'Reddit post'],
      ['https://news.ycombinator.com/item?id=123456', 'Hacker News'],
      ['https://en.wikipedia.org/wiki/Article_Title', 'Wikipedia'],
      ['https://arxiv.org/abs/2101.12345', 'arXiv paper']
    ])('should detect article content from %s', (url, description) => {
      const result = detector.detectContentType(url);
      
      expect(result.type).toBe('article');
      expect(result.confidence).toBeGreaterThanOrEqual(0.8);
      expect(result.suggestedAction).toBe('process-url');
      expect(result.icon).toBe('📰');
    });
  });

  describe('Code Content Detection', () => {
    test('should detect code from GitHub URLs', () => {
      const result = detector.detectContentType('https://github.com/user/repo');
      
      expect(result.type).toBe('code');
      expect(result.confidence).toBe(0.75);
      expect(result.suggestedAction).toBe('save-as-prompt');
      expect(result.icon).toBe('💻');
    });

    test('should detect code from page content', () => {
      const codeContent = `
        function hello() {
          console.log("Hello, world!");
        }
        
        class MyClass {
          constructor() {
            this.value = 42;
          }
        }
      `;
      
      const result = detector.detectContentType('https://example.com', 'Code Example', codeContent);
      
      expect(result.type).toBe('code');
      expect(result.suggestedAction).toBe('save-as-prompt');
    });

    test('should detect various programming languages', () => {
      const pythonCode = 'def hello():\n    print("Hello")\n\nfrom module import something';
      const javaCode = 'public class Main { private static void main(String[] args) {} }';
      const phpCode = '<?php echo "Hello"; ?>';
      
      expect(detector.hasCodeContent('', '', pythonCode)).toBe(true);
      expect(detector.hasCodeContent('', '', javaCode)).toBe(true);
      expect(detector.hasCodeContent('', '', phpCode)).toBe(true);
    });
  });

  describe('Social Media Detection', () => {
    test.each([
      ['https://twitter.com/username/status/123456', 'Twitter'],
      ['https://www.facebook.com/username/posts/123456', 'Facebook'],
      ['https://www.instagram.com/username/', 'Instagram'],
      ['https://www.linkedin.com/posts/username-123456', 'LinkedIn'],
      ['https://www.tiktok.com/@username', 'TikTok'],
      ['https://discord.com/channels/123456/789012', 'Discord'],
      ['https://t.me/channelname', 'Telegram']
    ])('should detect social media content from %s', (url, platform) => {
      const result = detector.detectContentType(url);
      
      expect(result.type).toBe('social');
      expect(result.confidence).toBe(0.6);
      expect(result.suggestedAction).toBe('send-to-chat');
      expect(result.icon).toBe('💬');
    });
  });

  describe('Long-form Content Detection', () => {
    test('should detect long-form content based on length', () => {
      const longContent = 'A'.repeat(3001);
      const shortContent = 'A'.repeat(2999);
      
      const longResult = detector.detectContentType('https://example.com', 'Article', longContent);
      const shortResult = detector.detectContentType('https://example.com', 'Article', shortContent);
      
      expect(longResult.type).toBe('article');
      expect(longResult.confidence).toBe(0.7);
      expect(longResult.icon).toBe('📖');
      
      expect(shortResult.type).toBe('webpage');
      expect(shortResult.confidence).toBe(0.5);
    });
  });

  describe('Suggested Actions', () => {
    test('should return appropriate actions for video content', () => {
      const context = { type: 'video' };
      const actions = detector.getSuggestedActions(context);
      
      expect(actions).toHaveLength(3);
      expect(actions[0]).toEqual({
        label: 'Process Video',
        action: 'process-videos',
        primary: true
      });
      expect(actions.map(a => a.action)).toContain('send-to-chat');
      expect(actions.map(a => a.action)).toContain('save-url');
    });

    test('should return appropriate actions for code content', () => {
      const context = { type: 'code' };
      const actions = detector.getSuggestedActions(context);
      
      expect(actions[0]).toEqual({
        label: 'Save as Prompt',
        action: 'save-as-prompt',
        primary: true
      });
    });

    test('should return default actions for unknown content', () => {
      const context = { type: 'unknown' };
      const actions = detector.getSuggestedActions(context);
      
      expect(actions[0]).toEqual({
        label: 'Process URL',
        action: 'process-url',
        primary: true
      });
    });

    test('should mark primary action correctly', () => {
      const types = ['video', 'audio', 'document', 'article', 'code', 'social', 'webpage'];
      
      types.forEach(type => {
        const actions = detector.getSuggestedActions({ type });
        const primaryActions = actions.filter(a => a.primary);
        
        expect(primaryActions).toHaveLength(1);
        expect(actions[0].primary).toBe(true);
      });
    });
  });

  describe('Edge Cases', () => {
    test('should handle empty inputs', () => {
      const result = detector.detectContentType('', '', '');
      
      expect(result.type).toBe('webpage');
      expect(result.confidence).toBe(0.5);
      expect(result.suggestedAction).toBe('process-url');
    });

    test('should handle malformed URLs', () => {
      const result = detector.detectContentType('not-a-url', 'Title', 'Content');
      
      expect(result).toBeDefined();
      expect(result.type).toBeDefined();
    });

    test('should prioritize URL patterns over content analysis', () => {
      // Even with code content, YouTube URL should be detected as video
      const codeContent = 'function test() { return true; }';
      const result = detector.detectContentType(
        'https://www.youtube.com/watch?v=123',
        'Coding Tutorial',
        codeContent
      );
      
      expect(result.type).toBe('video');
      expect(result.suggestedAction).toBe('process-videos');
    });

    test('should handle mixed content indicators', () => {
      // GitHub repo with PDF in path
      const result = detector.detectContentType(
        'https://github.com/user/repo/blob/main/docs/guide.pdf'
      );
      
      // Should prioritize document detection
      expect(result.type).toBe('document');
    });
  });

  describe('Confidence Levels', () => {
    test('should assign appropriate confidence levels', () => {
      const testCases = [
        { url: 'https://youtube.com/watch?v=123', expectedConfidence: 0.9 },
        { url: 'https://example.com/doc.pdf', expectedConfidence: 0.85 },
        { url: 'https://medium.com/article', expectedConfidence: 0.8 },
        { url: 'https://github.com/repo', expectedConfidence: 0.75 },
        { url: 'https://twitter.com/status', expectedConfidence: 0.6 },
        { url: 'https://example.com', expectedConfidence: 0.5 }
      ];
      
      testCases.forEach(({ url, expectedConfidence }) => {
        const result = detector.detectContentType(url);
        expect(result.confidence).toBe(expectedConfidence);
      });
    });

    test('should increase confidence for long-form generic content', () => {
      const longContent = 'Lorem ipsum '.repeat(500);
      const result = detector.detectContentType('https://example.com', 'Article', longContent);
      
      expect(result.confidence).toBe(0.7);
      expect(result.type).toBe('article');
    });
  });
});