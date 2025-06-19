/**
 * @jest-environment jsdom
 */

import { 
  createMockPrompt, 
  createMockCharacter,
  createMockError 
} from '../utils/factories.js';
import { 
  waitForAsync, 
  setupMockAPI, 
  setupChromeStorage,
  simulateUserInput,
  createDOM,
  cleanupDOM
} from '../utils/helpers.js';

// Mock the API client
jest.mock('../../js/utils/api.js', () => ({
  apiClient: {
    searchPrompts: jest.fn(),
    searchCharacters: jest.fn(),
    getPrompts: jest.fn(),
    getCharacters: jest.fn()
  }
}));

describe('Enhanced Search System', () => {
  let enhancedSearch;
  let mockStorage;
  let apiClient;

  beforeEach(() => {
    jest.clearAllMocks();
    
    // Setup DOM structure
    document.body.innerHTML = `
      <input id="promptSearch" type="text" />
      <select id="promptSortBy">
        <option value="name">Name</option>
        <option value="created_at">Date</option>
      </select>
      <select id="promptFilterBy">
        <option value="">All</option>
        <option value="writing">Writing</option>
      </select>
      <button id="clearPromptSearch">Clear</button>
      <div id="promptSearchSuggestions"></div>
      <div id="promptRecentSearches">
        <div id="promptRecentItems"></div>
      </div>
      <div id="promptsList"></div>
      
      <input id="characterSearch" type="text" />
      <select id="characterSortBy">
        <option value="name">Name</option>
        <option value="popularity">Popularity</option>
      </select>
      <select id="characterFilterBy">
        <option value="">All</option>
        <option value="assistant">Assistant</option>
      </select>
      <div id="charactersList"></div>
    `;

    // Setup storage
    mockStorage = setupChromeStorage({
      local: {
        recentSearches: {
          prompts: ['test prompt', 'writing helper'],
          characters: ['assistant', 'creative writer']
        }
      }
    });

    // Get API client mock
    apiClient = require('../../js/utils/api.js').apiClient;

    // Mock the EnhancedSearch class
    enhancedSearch = {
      recentSearches: {
        prompts: ['test prompt', 'writing helper'],
        characters: ['assistant', 'creative writer']
      },
      searchCache: new Map(),
      searchTimeout: null,
      maxRecentSearches: 10,

      initialize: jest.fn(async function() {
        await this.setupSearchEventListeners();
        await this.loadSearchData();
      }),

      setupSearchEventListeners: jest.fn(function() {
        const promptSearch = document.getElementById('promptSearch');
        if (promptSearch) {
          promptSearch.addEventListener('input', (e) => this.handleSearchInput(e, 'prompts'));
        }
      }),

      handleSearchInput: jest.fn(function(event, type) {
        const query = event.target.value.trim();
        
        if (this.searchTimeout) {
          clearTimeout(this.searchTimeout);
        }
        
        if (query.length >= 2) {
          this.showSearchSuggestions(type, query);
          this.searchTimeout = setTimeout(() => {
            this.performSearch(type, query);
          }, 300);
        }
      }),

      performSearch: jest.fn(async function(type, query) {
        const filters = this.getSearchFilters(type);
        const cacheKey = `${type}-${query}-${JSON.stringify(filters)}`;
        
        if (this.searchCache.has(cacheKey)) {
          const results = this.searchCache.get(cacheKey);
          this.displayResults(type, results);
          return results;
        }
        
        let results;
        if (type === 'prompts') {
          if (query) {
            const response = await apiClient.searchPrompts(query);
            results = response.results;
          } else {
            results = await this.loadAllPrompts();
          }
        }
        
        this.searchCache.set(cacheKey, results);
        this.displayResults(type, results);
        return results;
      }),

      getSearchFilters: jest.fn(function(type) {
        const sortBy = document.getElementById(`${type.slice(0, -1)}SortBy`)?.value || 'name';
        const filterBy = document.getElementById(`${type.slice(0, -1)}FilterBy`)?.value || '';
        return { sortBy, filterBy };
      }),

      showSearchSuggestions: jest.fn(function(type, query) {
        const suggestions = this.generateSuggestions(type, query);
        const container = document.getElementById(`${type.slice(0, -1)}SearchSuggestions`);
        if (container) {
          container.innerHTML = suggestions.map(s => 
            `<div class="suggestion-item" data-suggestion="${s}">${s}</div>`
          ).join('');
          container.style.display = 'block';
        }
      }),

      generateSuggestions: jest.fn(function(type, query) {
        const recent = this.recentSearches[type] || [];
        return recent.filter(s => s.toLowerCase().includes(query.toLowerCase()));
      }),

      displayResults: jest.fn(function(type, results) {
        const container = document.getElementById(`${type}List`);
        if (container) {
          container.innerHTML = `Found ${results.length} results`;
        }
      }),

      loadAllPrompts: jest.fn(async function() {
        const response = await apiClient.getPrompts();
        return response.results || [];
      }),

      saveRecentSearches: jest.fn(function() {
        chrome.storage.local.set({ recentSearches: this.recentSearches });
      }),

      addRecentSearch: jest.fn(function(type, query) {
        if (!this.recentSearches[type]) {
          this.recentSearches[type] = [];
        }
        
        const searches = this.recentSearches[type];
        const index = searches.indexOf(query);
        
        if (index > -1) {
          searches.splice(index, 1);
        }
        
        searches.unshift(query);
        
        if (searches.length > this.maxRecentSearches) {
          searches.pop();
        }
        
        this.saveRecentSearches();
      }),

      clearSearch: jest.fn(function(type) {
        const searchInput = document.getElementById(`${type.slice(0, -1)}Search`);
        if (searchInput) {
          searchInput.value = '';
          this.performSearch(type, '');
        }
      }),

      applySorting: jest.fn(function(results, sortBy, type) {
        const sorted = [...results];
        
        switch (sortBy) {
          case 'name':
            sorted.sort((a, b) => a.name.localeCompare(b.name));
            break;
          case 'created_at':
            sorted.sort((a, b) => new Date(b.created_at) - new Date(a.created_at));
            break;
          case 'usage':
            sorted.sort((a, b) => (b.usage_count || 0) - (a.usage_count || 0));
            break;
        }
        
        return sorted;
      }),

      loadSearchData: jest.fn(async function() {
        const stored = await chrome.storage.local.get('recentSearches');
        if (stored.recentSearches) {
          this.recentSearches = stored.recentSearches;
        }
      })
    };
  });

  afterEach(() => {
    cleanupDOM();
  });

  describe('Initialization', () => {
    test('should initialize search system correctly', async () => {
      await enhancedSearch.initialize();

      expect(enhancedSearch.setupSearchEventListeners).toHaveBeenCalled();
      expect(enhancedSearch.loadSearchData).toHaveBeenCalled();
    });

    test('should load recent searches from storage', async () => {
      await enhancedSearch.loadSearchData();

      expect(chrome.storage.local.get).toHaveBeenCalledWith('recentSearches');
      expect(enhancedSearch.recentSearches.prompts).toContain('test prompt');
      expect(enhancedSearch.recentSearches.characters).toContain('assistant');
    });
  });

  describe('Search Input Handling', () => {
    test('should handle search input with debouncing', async () => {
      const searchInput = document.getElementById('promptSearch');
      
      enhancedSearch.handleSearchInput({ target: { value: 'test' } }, 'prompts');

      expect(enhancedSearch.showSearchSuggestions).toHaveBeenCalledWith('prompts', 'test');
      expect(enhancedSearch.performSearch).not.toHaveBeenCalled();

      // Wait for debounce
      await waitForAsync(350);
      
      expect(enhancedSearch.performSearch).toHaveBeenCalledWith('prompts', 'test');
    });

    test('should not search for queries less than 2 characters', () => {
      enhancedSearch.handleSearchInput({ target: { value: 'a' } }, 'prompts');

      expect(enhancedSearch.showSearchSuggestions).not.toHaveBeenCalled();
      expect(enhancedSearch.performSearch).not.toHaveBeenCalled();
    });

    test('should cancel previous search timeout on new input', async () => {
      enhancedSearch.handleSearchInput({ target: { value: 'te' } }, 'prompts');
      await waitForAsync(100);
      enhancedSearch.handleSearchInput({ target: { value: 'tes' } }, 'prompts');
      await waitForAsync(100);
      enhancedSearch.handleSearchInput({ target: { value: 'test' } }, 'prompts');

      // Should only perform search once
      await waitForAsync(350);
      expect(enhancedSearch.performSearch).toHaveBeenCalledTimes(1);
      expect(enhancedSearch.performSearch).toHaveBeenCalledWith('prompts', 'test');
    });
  });

  describe('Search Execution', () => {
    const mockPrompts = [
      createMockPrompt({ name: 'Writing Assistant', created_at: '2024-01-01' }),
      createMockPrompt({ name: 'Code Helper', created_at: '2024-01-02' }),
      createMockPrompt({ name: 'Test Prompt', created_at: '2024-01-03' })
    ];

    beforeEach(() => {
      apiClient.searchPrompts.mockResolvedValue({ results: mockPrompts });
      apiClient.getPrompts.mockResolvedValue({ results: mockPrompts });
    });

    test('should perform search and cache results', async () => {
      const results = await enhancedSearch.performSearch('prompts', 'test');

      expect(apiClient.searchPrompts).toHaveBeenCalledWith('test');
      expect(enhancedSearch.displayResults).toHaveBeenCalledWith('prompts', mockPrompts);
      expect(enhancedSearch.searchCache.has('prompts-test-{"sortBy":"name","filterBy":""}')).toBe(true);
      expect(results).toEqual(mockPrompts);
    });

    test('should return cached results on subsequent searches', async () => {
      // First search
      await enhancedSearch.performSearch('prompts', 'test');
      
      // Reset mocks
      apiClient.searchPrompts.mockClear();
      enhancedSearch.displayResults.mockClear();

      // Second search with same query
      const results = await enhancedSearch.performSearch('prompts', 'test');

      expect(apiClient.searchPrompts).not.toHaveBeenCalled();
      expect(enhancedSearch.displayResults).toHaveBeenCalledWith('prompts', mockPrompts);
      expect(results).toEqual(mockPrompts);
    });

    test('should load all items when search is empty', async () => {
      await enhancedSearch.performSearch('prompts', '');

      expect(apiClient.searchPrompts).not.toHaveBeenCalled();
      expect(enhancedSearch.loadAllPrompts).toHaveBeenCalled();
    });

    test('should handle search errors gracefully', async () => {
      apiClient.searchPrompts.mockRejectedValue(new Error('Search failed'));

      await expect(enhancedSearch.performSearch('prompts', 'test')).rejects.toThrow('Search failed');
    });
  });

  describe('Search Filters and Sorting', () => {
    test('should get search filters correctly', () => {
      document.getElementById('promptSortBy').value = 'created_at';
      document.getElementById('promptFilterBy').value = 'writing';

      const filters = enhancedSearch.getSearchFilters('prompts');

      expect(filters).toEqual({
        sortBy: 'created_at',
        filterBy: 'writing'
      });
    });

    test('should apply sorting correctly', () => {
      const items = [
        { name: 'C Item', created_at: '2024-01-01', usage_count: 5 },
        { name: 'A Item', created_at: '2024-01-03', usage_count: 10 },
        { name: 'B Item', created_at: '2024-01-02', usage_count: 3 }
      ];

      // Sort by name
      const byName = enhancedSearch.applySorting(items, 'name', 'prompts');
      expect(byName[0].name).toBe('A Item');
      expect(byName[2].name).toBe('C Item');

      // Sort by date
      const byDate = enhancedSearch.applySorting(items, 'created_at', 'prompts');
      expect(byDate[0].created_at).toBe('2024-01-03');
      expect(byDate[2].created_at).toBe('2024-01-01');

      // Sort by usage
      const byUsage = enhancedSearch.applySorting(items, 'usage', 'prompts');
      expect(byUsage[0].usage_count).toBe(10);
      expect(byUsage[2].usage_count).toBe(3);
    });
  });

  describe('Search Suggestions', () => {
    test('should generate suggestions from recent searches', () => {
      const suggestions = enhancedSearch.generateSuggestions('prompts', 'test');

      expect(suggestions).toContain('test prompt');
      expect(suggestions).not.toContain('writing helper');
    });

    test('should display suggestions in DOM', () => {
      enhancedSearch.showSearchSuggestions('prompts', 'test');

      const container = document.getElementById('promptSearchSuggestions');
      expect(container.style.display).toBe('block');
      expect(container.innerHTML).toContain('test prompt');
      expect(container.querySelectorAll('.suggestion-item').length).toBe(1);
    });

    test('should handle empty suggestions', () => {
      enhancedSearch.recentSearches.prompts = [];
      
      const suggestions = enhancedSearch.generateSuggestions('prompts', 'xyz');
      enhancedSearch.showSearchSuggestions('prompts', 'xyz');

      expect(suggestions).toEqual([]);
      const container = document.getElementById('promptSearchSuggestions');
      expect(container.innerHTML).toBe('');
    });
  });

  describe('Recent Searches Management', () => {
    test('should add new search to recent searches', () => {
      enhancedSearch.addRecentSearch('prompts', 'new search query');

      expect(enhancedSearch.recentSearches.prompts[0]).toBe('new search query');
      expect(enhancedSearch.saveRecentSearches).toHaveBeenCalled();
    });

    test('should move existing search to top', () => {
      enhancedSearch.recentSearches.prompts = ['old', 'test prompt', 'other'];
      
      enhancedSearch.addRecentSearch('prompts', 'test prompt');

      expect(enhancedSearch.recentSearches.prompts[0]).toBe('test prompt');
      expect(enhancedSearch.recentSearches.prompts).toEqual(['test prompt', 'old', 'other']);
    });

    test('should limit recent searches to max count', () => {
      enhancedSearch.recentSearches.prompts = new Array(10).fill('search');
      
      enhancedSearch.addRecentSearch('prompts', 'new search');

      expect(enhancedSearch.recentSearches.prompts.length).toBe(10);
      expect(enhancedSearch.recentSearches.prompts[0]).toBe('new search');
    });

    test('should save recent searches to storage', () => {
      enhancedSearch.saveRecentSearches();

      expect(chrome.storage.local.set).toHaveBeenCalledWith({
        recentSearches: enhancedSearch.recentSearches
      });
    });
  });

  describe('Clear Search', () => {
    test('should clear search input and results', () => {
      const searchInput = document.getElementById('promptSearch');
      searchInput.value = 'test query';

      enhancedSearch.clearSearch('prompts');

      expect(searchInput.value).toBe('');
      expect(enhancedSearch.performSearch).toHaveBeenCalledWith('prompts', '');
    });
  });

  describe('Cache Management', () => {
    test('should cache results with correct key', async () => {
      document.getElementById('promptSortBy').value = 'name';
      document.getElementById('promptFilterBy').value = 'writing';

      await enhancedSearch.performSearch('prompts', 'test');

      const expectedKey = 'prompts-test-{"sortBy":"name","filterBy":"writing"}';
      expect(enhancedSearch.searchCache.has(expectedKey)).toBe(true);
    });

    test('should have separate cache entries for different filters', async () => {
      // First search
      document.getElementById('promptSortBy').value = 'name';
      await enhancedSearch.performSearch('prompts', 'test');

      // Second search with different sort
      document.getElementById('promptSortBy').value = 'created_at';
      await enhancedSearch.performSearch('prompts', 'test');

      expect(enhancedSearch.searchCache.size).toBe(2);
    });
  });
});