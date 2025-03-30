# DDG_WebSearch.py
# Description: DuckDuckGo Web Search API Unit Test
#
# Imports
import unittest
from typing import List, Dict
#
# 3rd party imports
#
# Local imports
from PoC_Version.App_Function_Libraries.Web_Scraping.WebSearch_APIs import search_web_ddg
#
#######################################################################################################################
#
# Functions:

def example_usage():
    """Example usage of the DuckDuckGo search function"""
    try:
        # Basic search
        results = search_web_ddg("Python programming")
        print(f"Found {len(results)} results for 'Python programming'")

        # Print first 3 results
        for i, result in enumerate(results[:3], 1):
            print(f"\nResult {i}:")
            print(f"Title: {result['title']}")
            print(f"URL: {result['href']}")
            print(f"Description: {result['body'][:150]}...")

        # Search with different parameters
        limited_results = search_web_ddg(
            keywords="artificial intelligence news",
            region="us-en",
            safesearch="on",
            max_results=5
        )
        print(f"\nFound {len(limited_results)} limited results")

    except search_web_ddg as e:
        print(f"Search failed: {e}")

class TestDuckDuckGoSearch(unittest.TestCase):
    """Test cases for DuckDuckGo search function"""

    def test_basic_search(self):
        """Test basic search functionality"""
        results = search_web_ddg("Python programming")
        self.assertIsInstance(results, list)
        self.assertTrue(len(results) > 0)

        # Check result structure
        first_result = results[0]
        self.assertIn('title', first_result)
        self.assertIn('href', first_result)
        self.assertIn('body', first_result)

        # Check data types
        self.assertIsInstance(first_result['title'], str)
        self.assertIsInstance(first_result['href'], str)
        self.assertIsInstance(first_result['body'], str)

        # Check for non-empty values
        self.assertTrue(first_result['title'])
        self.assertTrue(first_result['href'])
        self.assertTrue(first_result['body'])

    def test_max_results(self):
        """Test max_results parameter"""
        max_results = 5
        results = search_web_ddg("Python programming", max_results=max_results)
        self.assertLessEqual(len(results), max_results)

    def test_invalid_region(self):
        """Test invalid region handling"""
        results = search_web_ddg("Python", region="invalid-region")
        self.assertIsInstance(results, list)  # Should still return results with default region

    def test_result_uniqueness(self):
        """Test that results are unique"""
        results = search_web_ddg("Python programming", max_results=10)
        urls = [result['href'] for result in results]
        unique_urls = set(urls)
        self.assertEqual(len(urls), len(unique_urls))

    def test_url_normalization(self):
        """Test URL normalization"""
        results = search_web_ddg("Python programming")
        for result in results:
            self.assertTrue(result['href'].startswith(('http://', 'https://')))

    def validate_search_results(self, results: List[Dict[str, str]]) -> bool:
        """Helper method to validate search results structure"""
        if not results:
            return False

        required_keys = {'title', 'href', 'body'}
        for result in results:
            if not all(key in result for key in required_keys):
                return False
            if not all(isinstance(result[key], str) for key in required_keys):
                return False
            if not all(result[key].strip() for key in required_keys):
                return False
        return True


def run_tests():
    """Run all tests"""
    unittest.main(argv=[''], exit=False)


if __name__ == "__main__":
    # Example usage
    print("Running example search:")
    example_usage()

    print("\nRunning tests:")
    run_tests()

#
# End of DDG_WebSearch.py
#######################################################################################################################
