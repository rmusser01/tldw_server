import os
import sys
import unittest
from unittest.mock import patch

# Adjust the path to the parent directory of App_Function_Libraries
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(current_dir, '..', '..'))
sys.path.append(parent_dir)

# Now import the necessary modules
from PoC_Version.App_Function_Libraries.RAG.RAG_Library_2 import enhanced_rag_pipeline


class TestEnhancedRagPipeline(unittest.TestCase):

    @patch('App_Function_Libraries.RAG.RAG_Library_2.fetch_relevant_media_ids')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.perform_vector_search')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.perform_full_text_search')
    @patch('App_Function_Libraries.RAG.RAG_Library_2.generate_answer')
    def test_enhanced_rag_pipeline(self, mock_generate_answer, mock_fts_search, mock_vector_search,
                                   mock_fetch_keywords):
        """Test the enhanced_rag_pipeline function with less strict string matching"""

        # Setup mock data
        query = "What is the capital of France?"
        keywords = "geography, cities"
        api_choice = "OpenAI"

        # Mock the relevant_media_ids fetched by keywords
        mock_fetch_keywords.return_value = [1, 2, 3]

        # Mock vector search results
        mock_vector_search.return_value = [
            {"content": "Paris is the capital of France."}
        ]

        # Mock full-text search results
        mock_fts_search.return_value = [
            {"content": "The capital of France is Paris."}
        ]

        # Mock the API response from the generate_answer function
        mock_generate_answer.return_value = "Paris is the capital of France."

        # Call the enhanced_rag_pipeline function
        result = enhanced_rag_pipeline(query=query, api_choice=api_choice, keywords=keywords)

        # Validate that the vector search and full-text search are called with expected arguments
        mock_vector_search.assert_called_once_with(query, ['1', '2', '3'], top_k=10)
        mock_fts_search.assert_called_once_with(query, 'Media DB', ['1', '2', '3'], 10)

        # Instead of checking exact string match, check that both pieces are in the context
        call_args = mock_generate_answer.call_args[0]  # Get the args the mock was called with
        actual_context = call_args[1]  # Get the context string

        # Verify each piece of content is in the context
        self.assertIn("Paris is the capital of France.", actual_context)
        self.assertIn("The capital of France is Paris.", actual_context)
        self.assertEqual(call_args[0], api_choice)  # API choice should match
        self.assertEqual(call_args[2], query)  # Query should match

        # Validate the result structure
        self.assertIn("answer", result)
        self.assertIn("context", result)
        self.assertEqual(result["answer"], "Paris is the capital of France.")

        # Verify both pieces are in the result context too
        self.assertIn("Paris is the capital of France.", result["context"])
        self.assertIn("The capital of France is Paris.", result["context"])

if __name__ == '__main__':
    unittest.main()
