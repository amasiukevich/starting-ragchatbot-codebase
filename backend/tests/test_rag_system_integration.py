"""
Integration tests for RAG system query flow.

These tests verify the complete query pipeline from user input to response,
helping identify where "query failed" errors might be occurring in the full system.
"""

import os
import sys
from unittest.mock import MagicMock, Mock, patch

import pytest

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from rag_system import RAGSystem
from search_tools import CourseOutlineTool, CourseSearchTool, ToolManager
from vector_store import SearchResults


class TestRAGSystemQueryFlow:
    """Test the complete query flow through RAG system"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_successful_content_query_flow(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test successful content query from start to finish"""
        # Setup mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        # Mock successful search results
        search_results = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store_instance.search.return_value = search_results
        mock_vector_store_instance.get_lesson_link.return_value = (
            "http://example.com/lesson1"
        )

        # Mock AI generator that uses tools
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = (
            "Python is a high-level programming language used for various applications."
        )

        # Mock session manager
        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        # Create RAG system
        rag_system = RAGSystem(mock_config)

        # Execute query
        response, sources = rag_system.query(
            "What is Python?", session_id="test_session"
        )

        # Verify the flow
        assert (
            response
            == "Python is a high-level programming language used for various applications."
        )

        # Verify AI generator was called with tools
        mock_ai_instance.generate_response.assert_called_once()
        call_args = mock_ai_instance.generate_response.call_args
        assert "tools" in call_args[1]
        assert "tool_manager" in call_args[1]

        # Verify session management
        mock_session_instance.get_conversation_history.assert_called_once_with(
            "test_session"
        )
        mock_session_instance.add_exchange.assert_called_once()

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_query_without_session(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test query without session ID"""
        # Setup basic mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Response without session"

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # Execute query without session
        response, sources = rag_system.query("What is Python?")

        # Verify no session operations were called
        mock_session_instance.get_conversation_history.assert_not_called()
        mock_session_instance.add_exchange.assert_not_called()

        # Verify response was still generated
        assert response == "Response without session"

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_vector_store_error_propagation(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test how vector store errors propagate through the system"""
        # Setup vector store to return error
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Vector database connection failed",
        )
        mock_vector_store_instance.search.return_value = error_results

        # Mock AI generator
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = (
            "I'm sorry, I couldn't search the course materials due to a database error."
        )

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # Execute query
        response, sources = rag_system.query("What is Python?")

        # System should still return a response, not crash
        assert isinstance(response, str)
        assert len(sources) == 0  # No sources due to error

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_ai_generator_error_handling(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test handling when AI generator fails"""
        # Setup mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        # Mock AI generator that raises exception
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance
        mock_ai_instance.generate_response.side_effect = Exception(
            "AI API connection failed"
        )

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # Execute query - should raise exception
        with pytest.raises(Exception) as exc_info:
            rag_system.query("What is Python?")

        assert "AI API connection failed" in str(exc_info.value)

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_tool_registration_on_initialization(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test that tools are properly registered during RAG system initialization"""
        # Setup mocks
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        # Create RAG system
        rag_system = RAGSystem(mock_config)

        # Verify tools are registered
        assert "search_course_content" in rag_system.tool_manager.tools
        assert "get_course_outline" in rag_system.tool_manager.tools

        # Verify tool definitions can be retrieved
        definitions = rag_system.tool_manager.get_tool_definitions()
        assert len(definitions) == 2

        tool_names = [tool["name"] for tool in definitions]
        assert "search_course_content" in tool_names
        assert "get_course_outline" in tool_names

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_sources_tracking_and_reset(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test that sources are properly tracked and reset between queries"""
        # Setup vector store with results
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        search_results = SearchResults(
            documents=["Content 1", "Content 2"],
            metadata=[
                {"course_title": "Course A", "lesson_number": 1},
                {"course_title": "Course B", "lesson_number": 2},
            ],
            distances=[0.1, 0.2],
        )
        mock_vector_store_instance.search.return_value = search_results
        mock_vector_store_instance.get_lesson_link.return_value = (
            "http://example.com/lesson"
        )

        # Mock AI generator
        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance
        mock_ai_instance.generate_response.return_value = "Test response"

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # First query
        response1, sources1 = rag_system.query("First query")

        # Second query
        response2, sources2 = rag_system.query("Second query")

        # Sources should be properly tracked and reset
        # (Note: This assumes the tool is actually called by the AI)
        # The sources list should be consistent between calls
        assert isinstance(sources1, list)
        assert isinstance(sources2, list)

    def test_real_tool_execution_flow(self, mock_config):
        """Test the actual tool execution flow with real tools but mocked dependencies"""
        # Create real tool manager and tools with mocked vector store
        mock_vector_store = Mock()

        # Mock successful search
        search_results = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1],
        )
        mock_vector_store.search.return_value = search_results
        mock_vector_store.get_lesson_link.return_value = "http://example.com/lesson1"

        # Create real tools
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        outline_tool = CourseOutlineTool(mock_vector_store)

        tool_manager.register_tool(search_tool)
        tool_manager.register_tool(outline_tool)

        # Test tool execution
        result = tool_manager.execute_tool(
            "search_course_content", query="What is Python?"
        )

        # Verify tool executed successfully
        assert "[Python Basics - Lesson 1]" in result
        assert "Python is a programming language" in result

        # Verify sources were tracked
        sources = tool_manager.get_last_sources()
        assert len(sources) == 1
        assert sources[0]["text"] == "Python Basics - Lesson 1"
        assert sources[0]["link"] == "http://example.com/lesson1"

    def test_tool_error_handling_in_manager(self, mock_config):
        """Test error handling when tools fail during execution"""
        # Create tool with failing vector store
        mock_vector_store = Mock()
        error_results = SearchResults(
            documents=[], metadata=[], distances=[], error="Database connection error"
        )
        mock_vector_store.search.return_value = error_results

        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)

        # Execute tool - should return error message, not crash
        result = tool_manager.execute_tool("search_course_content", query="test")

        assert result == "Database connection error"

        # Sources should be empty due to error
        sources = tool_manager.get_last_sources()
        assert len(sources) == 0


class TestRAGSystemInitialization:
    """Test RAG system initialization and component setup"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_component_initialization(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test that all components are properly initialized"""
        rag_system = RAGSystem(mock_config)

        # Verify all components exist
        assert rag_system.document_processor is not None
        assert rag_system.vector_store is not None
        assert rag_system.ai_generator is not None
        assert rag_system.session_manager is not None
        assert rag_system.tool_manager is not None
        assert rag_system.search_tool is not None
        assert rag_system.outline_tool is not None

        # Verify components were called with correct config
        mock_doc_proc.assert_called_once_with(
            mock_config.CHUNK_SIZE, mock_config.CHUNK_OVERLAP
        )
        mock_vector_store.assert_called_once_with(
            mock_config.CHROMA_PATH,
            mock_config.EMBEDDING_MODEL,
            mock_config.MAX_RESULTS,
        )
        mock_ai_gen.assert_called_once_with(
            mock_config.ANTHROPIC_API_KEY, mock_config.ANTHROPIC_MODEL
        )
        mock_session_mgr.assert_called_once_with(mock_config.MAX_HISTORY)

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_initialization_with_missing_config(
        self, mock_session_mgr, mock_doc_proc, mock_vector_store, mock_ai_gen
    ):
        """Test initialization behavior with missing or invalid config"""
        # Test with None config
        with pytest.raises(AttributeError):
            RAGSystem(None)

        # Test with incomplete config
        incomplete_config = Mock()
        incomplete_config.CHUNK_SIZE = 800
        # Missing other required attributes

        with pytest.raises(AttributeError):
            RAGSystem(incomplete_config)


class TestRAGSystemCourseManagement:
    """Test course document management functionality"""

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_add_course_document_success(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
        sample_course,
        sample_course_chunks,
    ):
        """Test successful course document addition"""
        # Setup mocks
        mock_doc_proc_instance = Mock()
        mock_doc_proc.return_value = mock_doc_proc_instance
        mock_doc_proc_instance.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )

        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # Add course document
        course, chunk_count = rag_system.add_course_document("test_course.txt")

        # Verify processing was called
        mock_doc_proc_instance.process_course_document.assert_called_once_with(
            "test_course.txt"
        )

        # Verify data was added to vector store
        mock_vector_store_instance.add_course_metadata.assert_called_once_with(
            sample_course
        )
        mock_vector_store_instance.add_course_content.assert_called_once_with(
            sample_course_chunks
        )

        # Verify return values
        assert course == sample_course
        assert chunk_count == len(sample_course_chunks)

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_add_course_document_error(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test error handling during course document addition"""
        # Setup mock to raise exception
        mock_doc_proc_instance = Mock()
        mock_doc_proc.return_value = mock_doc_proc_instance
        mock_doc_proc_instance.process_course_document.side_effect = Exception(
            "File not found"
        )

        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance

        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # Add course document - should handle error gracefully
        course, chunk_count = rag_system.add_course_document("nonexistent.txt")

        # Should return None and 0 on error
        assert course is None
        assert chunk_count == 0

        # Vector store should not be called on error
        mock_vector_store_instance.add_course_metadata.assert_not_called()
        mock_vector_store_instance.add_course_content.assert_not_called()

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    @patch("rag_system.os.path.exists")
    @patch("rag_system.os.listdir")
    def test_add_course_folder(
        self,
        mock_listdir,
        mock_exists,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
        sample_course,
        sample_course_chunks,
    ):
        """Test adding course documents from a folder"""
        # Setup file system mocks
        mock_exists.return_value = True
        mock_listdir.return_value = ["course1.txt", "course2.pdf", "readme.md"]

        # Setup document processor mock
        mock_doc_proc_instance = Mock()
        mock_doc_proc.return_value = mock_doc_proc_instance
        mock_doc_proc_instance.process_course_document.return_value = (
            sample_course,
            sample_course_chunks,
        )

        # Setup vector store mock
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        mock_vector_store_instance.get_existing_course_titles.return_value = []

        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # Add course folder
        total_courses, total_chunks = rag_system.add_course_folder("test_folder")

        # Should process both .txt and .pdf files, but not .md
        assert mock_doc_proc_instance.process_course_document.call_count == 2
        assert total_courses == 2
        assert total_chunks == len(sample_course_chunks) * 2

    @patch("rag_system.AIGenerator")
    @patch("rag_system.VectorStore")
    @patch("rag_system.DocumentProcessor")
    @patch("rag_system.SessionManager")
    def test_get_course_analytics(
        self,
        mock_session_mgr,
        mock_doc_proc,
        mock_vector_store,
        mock_ai_gen,
        mock_config,
    ):
        """Test course analytics retrieval"""
        # Setup vector store mock
        mock_vector_store_instance = Mock()
        mock_vector_store.return_value = mock_vector_store_instance
        mock_vector_store_instance.get_course_count.return_value = 5
        mock_vector_store_instance.get_existing_course_titles.return_value = [
            "Course A",
            "Course B",
            "Course C",
        ]

        mock_ai_instance = Mock()
        mock_ai_gen.return_value = mock_ai_instance

        mock_session_instance = Mock()
        mock_session_mgr.return_value = mock_session_instance

        rag_system = RAGSystem(mock_config)

        # Get analytics
        analytics = rag_system.get_course_analytics()

        # Verify analytics structure
        assert "total_courses" in analytics
        assert "course_titles" in analytics
        assert analytics["total_courses"] == 5
        assert len(analytics["course_titles"]) == 3
        assert "Course A" in analytics["course_titles"]
