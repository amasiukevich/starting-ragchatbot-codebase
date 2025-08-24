"""
Comprehensive unit tests for CourseSearchTool.execute() method.

These tests are designed to identify the root cause of "query failed" errors
by thoroughly testing all aspects of the search tool functionality.
"""

import os
import sys
from unittest.mock import Mock, patch

import pytest

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from search_tools import CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestCourseSearchToolExecute:
    """Test the execute method of CourseSearchTool in detail"""

    def test_successful_search_with_results(
        self, mock_vector_store, successful_search_results
    ):
        """Test normal successful search that returns results"""
        # Setup
        mock_vector_store.search.return_value = successful_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        # Execute
        result = search_tool.execute("What is Python?")

        # Verify vector store was called correctly
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?", course_name=None, lesson_number=None
        )

        # Verify result format
        assert "[Python Fundamentals - Lesson 1]" in result
        assert "Python is a high-level programming language" in result
        assert "[Python Fundamentals - Lesson 2]" in result
        assert "Variables in Python are used to store data" in result

        # Verify sources are tracked
        assert len(search_tool.last_sources) == 2
        assert search_tool.last_sources[0]["text"] == "Python Fundamentals - Lesson 1"
        assert search_tool.last_sources[1]["text"] == "Python Fundamentals - Lesson 2"

    def test_search_with_course_name_filter(
        self, mock_vector_store, successful_search_results
    ):
        """Test search with course name filtering"""
        mock_vector_store.search.return_value = successful_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute(
            "What is Python?", course_name="Python Fundamentals"
        )

        # Verify course name was passed to vector store
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?",
            course_name="Python Fundamentals",
            lesson_number=None,
        )

        assert "[Python Fundamentals" in result

    def test_search_with_lesson_number_filter(
        self, mock_vector_store, successful_search_results
    ):
        """Test search with lesson number filtering"""
        mock_vector_store.search.return_value = successful_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("What is Python?", lesson_number=1)

        # Verify lesson number was passed to vector store
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?", course_name=None, lesson_number=1
        )

        assert "Lesson 1" in result

    def test_search_with_both_filters(
        self, mock_vector_store, successful_search_results
    ):
        """Test search with both course name and lesson number filters"""
        mock_vector_store.search.return_value = successful_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute(
            "What is Python?", course_name="Python Fundamentals", lesson_number=1
        )

        # Verify both filters were passed
        mock_vector_store.search.assert_called_once_with(
            query="What is Python?", course_name="Python Fundamentals", lesson_number=1
        )

    def test_empty_search_results(self, mock_vector_store, empty_search_results):
        """Test handling of empty search results"""
        mock_vector_store.search.return_value = empty_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("nonexistent content")

        assert result == "No relevant content found."

    def test_empty_results_with_course_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test empty results message includes course filter info"""
        mock_vector_store.search.return_value = empty_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("nonexistent", course_name="Python")

        assert result == "No relevant content found in course 'Python'."

    def test_empty_results_with_lesson_filter(
        self, mock_vector_store, empty_search_results
    ):
        """Test empty results message includes lesson filter info"""
        mock_vector_store.search.return_value = empty_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("nonexistent", lesson_number=5)

        assert result == "No relevant content found in lesson 5."

    def test_empty_results_with_both_filters(
        self, mock_vector_store, empty_search_results
    ):
        """Test empty results message includes both filter info"""
        mock_vector_store.search.return_value = empty_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute(
            "nonexistent", course_name="Python", lesson_number=5
        )

        assert result == "No relevant content found in course 'Python' in lesson 5."

    def test_vector_store_error_handling(self, mock_vector_store, error_search_results):
        """Test handling of vector store errors - CRITICAL for debugging 'query failed'"""
        mock_vector_store.search.return_value = error_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("test query")

        # This is likely where "query failed" or similar errors originate
        assert result == "Database connection failed"

    def test_vector_store_exception_handling(self, mock_vector_store):
        """Test handling when vector store raises an exception"""
        mock_vector_store.search.side_effect = Exception("ChromaDB connection error")
        search_tool = CourseSearchTool(mock_vector_store)

        # This should not raise an exception but return an error message
        result = search_tool.execute("test query")

        # If this raises an exception, it indicates poor error handling
        assert isinstance(result, str)

    def test_results_formatting_without_lesson_numbers(self, mock_vector_store):
        """Test formatting when metadata lacks lesson numbers"""
        results_without_lessons = SearchResults(
            documents=["General course information"],
            metadata=[{"course_title": "Python Fundamentals"}],  # No lesson_number
            distances=[0.1],
        )
        mock_vector_store.search.return_value = results_without_lessons
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("test query")

        assert "[Python Fundamentals]" in result
        assert "General course information" in result
        assert "Lesson" not in result

    def test_results_formatting_with_missing_course_title(self, mock_vector_store):
        """Test formatting when metadata lacks course title"""
        results_missing_title = SearchResults(
            documents=["Some content"],
            metadata=[{"lesson_number": 1}],  # No course_title
            distances=[0.1],
        )
        mock_vector_store.search.return_value = results_missing_title
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("test query")

        assert "[unknown - Lesson 1]" in result
        assert "Some content" in result

    def test_lesson_link_retrieval(self, mock_vector_store, successful_search_results):
        """Test that lesson links are retrieved and stored in sources"""
        mock_vector_store.search.return_value = successful_search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("test query")

        # Verify lesson link was requested
        mock_vector_store.get_lesson_link.assert_called()

        # Verify link is stored in sources
        assert len(search_tool.last_sources) > 0
        assert search_tool.last_sources[0]["link"] == "http://example.com/lesson1"

    def test_lesson_link_not_retrieved_without_lesson_number(self, mock_vector_store):
        """Test that lesson links are not retrieved when no lesson number"""
        results_without_lessons = SearchResults(
            documents=["General content"],
            metadata=[{"course_title": "Python Fundamentals"}],  # No lesson_number
            distances=[0.1],
        )
        mock_vector_store.search.return_value = results_without_lessons
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("test query")

        # Verify lesson link was NOT requested
        mock_vector_store.get_lesson_link.assert_not_called()

        # Verify source has no link
        assert search_tool.last_sources[0]["link"] is None

    def test_sources_reset_between_searches(
        self, mock_vector_store, successful_search_results
    ):
        """Test that sources are properly reset between searches"""
        mock_vector_store.search.return_value = successful_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        # First search
        search_tool.execute("first query")
        first_sources_count = len(search_tool.last_sources)

        # Second search with different results
        single_result = SearchResults(
            documents=["Single result"],
            metadata=[{"course_title": "Python", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = single_result
        search_tool.execute("second query")

        # Verify sources were replaced, not appended
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "Python - Lesson 1"

    def test_multiple_results_formatting(self, mock_vector_store):
        """Test proper formatting of multiple search results"""
        multi_results = SearchResults(
            documents=[
                "First piece of content about Python",
                "Second piece about variables",
                "Third piece about functions",
            ],
            metadata=[
                {"course_title": "Python Basics", "lesson_number": 1},
                {"course_title": "Python Basics", "lesson_number": 2},
                {"course_title": "Advanced Python", "lesson_number": 1},
            ],
            distances=[0.1, 0.2, 0.3],
        )
        mock_vector_store.search.return_value = multi_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("Python content")

        # Verify all results are included and properly separated
        lines = result.split("\n\n")
        assert len(lines) == 3

        assert "[Python Basics - Lesson 1]" in lines[0]
        assert "First piece of content" in lines[0]

        assert "[Python Basics - Lesson 2]" in lines[1]
        assert "Second piece about variables" in lines[1]

        assert "[Advanced Python - Lesson 1]" in lines[2]
        assert "Third piece about functions" in lines[2]

        # Verify all sources are tracked
        assert len(search_tool.last_sources) == 3


class TestCourseSearchToolIntegration:
    """Integration tests for CourseSearchTool with ToolManager"""

    def test_tool_definition(self):
        """Test that tool definition is correctly formatted"""
        search_tool = CourseSearchTool(Mock())
        definition = search_tool.get_tool_definition()

        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        assert "query" in definition["input_schema"]["properties"]
        assert "course_name" in definition["input_schema"]["properties"]
        assert "lesson_number" in definition["input_schema"]["properties"]

    def test_tool_manager_registration(self, mock_vector_store):
        """Test that search tool can be registered with ToolManager"""
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)

        tool_manager.register_tool(search_tool)

        assert "search_course_content" in tool_manager.tools
        definitions = tool_manager.get_tool_definitions()
        assert len(definitions) == 1
        assert definitions[0]["name"] == "search_course_content"

    def test_tool_manager_execution(self, mock_vector_store, successful_search_results):
        """Test tool execution through ToolManager"""
        mock_vector_store.search.return_value = successful_search_results

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        result = tool_manager.execute_tool(
            "search_course_content", query="test query", course_name="Python"
        )

        assert "[Python Fundamentals" in result

    def test_tool_manager_error_handling(self, mock_vector_store, error_search_results):
        """Test error propagation through ToolManager"""
        mock_vector_store.search.return_value = error_search_results

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        result = tool_manager.execute_tool("search_course_content", query="test")

        # Error should propagate through ToolManager
        assert result == "Database connection failed"

    def test_sources_retrieval_through_tool_manager(
        self, mock_vector_store, successful_search_results
    ):
        """Test that sources can be retrieved after tool execution"""
        mock_vector_store.search.return_value = successful_search_results

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Execute search
        tool_manager.execute_tool("search_course_content", query="test")

        # Retrieve sources
        sources = tool_manager.get_last_sources()
        assert len(sources) == 2
        assert sources[0]["text"] == "Python Fundamentals - Lesson 1"

        # Reset sources
        tool_manager.reset_sources()
        sources_after_reset = tool_manager.get_last_sources()
        assert len(sources_after_reset) == 0


class TestCourseSearchToolErrorConditions:
    """Test various error conditions that could cause 'query failed'"""

    def test_none_query_handling(self, mock_vector_store):
        """Test handling of None query"""
        search_tool = CourseSearchTool(mock_vector_store)

        # This should not crash the application
        try:
            result = search_tool.execute(None)
            # If it returns something, it should be a string
            assert isinstance(result, str)
        except Exception as e:
            # If it raises an exception, that's a bug we need to fix
            pytest.fail(f"Tool should handle None query gracefully, but raised: {e}")

    def test_empty_string_query(self, mock_vector_store, empty_search_results):
        """Test handling of empty string query"""
        mock_vector_store.search.return_value = empty_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        result = search_tool.execute("")
        assert isinstance(result, str)

    def test_very_long_query(self, mock_vector_store, successful_search_results):
        """Test handling of very long queries"""
        mock_vector_store.search.return_value = successful_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        long_query = "What is Python? " * 1000  # Very long query
        result = search_tool.execute(long_query)

        # Should still work
        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once()

    def test_special_characters_in_query(
        self, mock_vector_store, successful_search_results
    ):
        """Test handling of special characters in queries"""
        mock_vector_store.search.return_value = successful_search_results
        search_tool = CourseSearchTool(mock_vector_store)

        special_query = "What is Python? <>{}[]()!@#$%^&*"
        result = search_tool.execute(special_query)

        assert isinstance(result, str)
        mock_vector_store.search.assert_called_once_with(
            query=special_query, course_name=None, lesson_number=None
        )
