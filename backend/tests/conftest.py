"""
Shared test fixtures and configuration for RAG chatbot tests.
"""

import os
import sys
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest

# Add the backend directory to the Python path so we can import modules
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(
            lesson_number=1,
            title="Introduction to Python",
            lesson_link="http://example.com/lesson1",
        ),
        Lesson(
            lesson_number=2,
            title="Variables and Data Types",
            lesson_link="http://example.com/lesson2",
        ),
        Lesson(
            lesson_number=3,
            title="Control Structures",
            lesson_link="http://example.com/lesson3",
        ),
    ]
    return Course(
        title="Python Fundamentals",
        instructor="Dr. Python",
        course_link="http://example.com/course",
        lessons=lessons,
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language known for its simplicity.",
            course_title="Python Fundamentals",
            lesson_number=1,
            chunk_index=0,
        ),
        CourseChunk(
            content="Variables in Python are used to store data values.",
            course_title="Python Fundamentals",
            lesson_number=2,
            chunk_index=1,
        ),
        CourseChunk(
            content="Control structures like if-else statements control program flow.",
            course_title="Python Fundamentals",
            lesson_number=3,
            chunk_index=2,
        ),
    ]


@pytest.fixture
def successful_search_results():
    """Create successful search results for testing"""
    return SearchResults(
        documents=[
            "Python is a high-level programming language known for its simplicity.",
            "Variables in Python are used to store data values.",
        ],
        metadata=[
            {
                "course_title": "Python Fundamentals",
                "lesson_number": 1,
                "chunk_index": 0,
            },
            {
                "course_title": "Python Fundamentals",
                "lesson_number": 2,
                "chunk_index": 1,
            },
        ],
        distances=[0.1, 0.2],
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults(
        documents=[], metadata=[], distances=[], error="Database connection failed"
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(
        documents=[], metadata=[], distances=[]
    )
    mock_store.get_lesson_link.return_value = "http://example.com/lesson1"
    mock_store._resolve_course_name.return_value = "Python Fundamentals"
    mock_store.get_all_courses_metadata.return_value = []
    return mock_store


@pytest.fixture
def mock_anthropic_client():
    """Create a mock Anthropic client for testing"""
    mock_client = Mock()

    # Default response for non-tool calls
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = "This is a test AI response"
    mock_response.stop_reason = "end_turn"

    mock_client.messages.create.return_value = mock_response
    return mock_client


@pytest.fixture
def mock_tool_use_response():
    """Create a mock Anthropic response that uses tools"""
    mock_response = Mock()

    # Mock tool use content block
    tool_block = Mock()
    tool_block.type = "tool_use"
    tool_block.name = "search_course_content"
    tool_block.id = "tool_123"
    tool_block.input = {"query": "test query", "course_name": "Python"}

    mock_response.content = [tool_block]
    mock_response.stop_reason = "tool_use"

    return mock_response


@pytest.fixture
def mock_final_response():
    """Create a mock final response after tool execution"""
    mock_response = Mock()
    mock_response.content = [Mock()]
    mock_response.content[0].text = (
        "Based on the course content, Python is a programming language."
    )
    mock_response.stop_reason = "end_turn"
    return mock_response


@pytest.fixture
def sample_tool_definitions():
    """Create sample tool definitions for testing"""
    return [
        {
            "name": "search_course_content",
            "description": "Search course materials",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "course_name": {"type": "string"},
                },
                "required": ["query"],
            },
        }
    ]


@pytest.fixture
def mock_config():
    """Create a mock config object for testing"""
    config = Mock()
    config.ANTHROPIC_API_KEY = "test_api_key"
    config.ANTHROPIC_MODEL = "claude-3-sonnet-20240229"
    config.CHUNK_SIZE = 800
    config.CHUNK_OVERLAP = 100
    config.MAX_RESULTS = 5
    config.MAX_HISTORY = 2
    config.CHROMA_PATH = "./test_chroma_db"
    config.EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    return config
