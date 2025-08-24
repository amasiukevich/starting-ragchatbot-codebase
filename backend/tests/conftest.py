"""
Shared test fixtures and configuration for RAG chatbot tests.
"""
import pytest
import sys
import os
import tempfile
import shutil
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, List, Any
from fastapi.testclient import TestClient

# Add the backend directory to the Python path so we can import modules
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from vector_store import SearchResults
from models import Course, Lesson, CourseChunk


@pytest.fixture
def sample_course():
    """Create a sample course for testing"""
    lessons = [
        Lesson(lesson_number=1, title="Introduction to Python", lesson_link="http://example.com/lesson1"),
        Lesson(lesson_number=2, title="Variables and Data Types", lesson_link="http://example.com/lesson2"),
        Lesson(lesson_number=3, title="Control Structures", lesson_link="http://example.com/lesson3")
    ]
    return Course(
        title="Python Fundamentals",
        instructor="Dr. Python",
        course_link="http://example.com/course",
        lessons=lessons
    )


@pytest.fixture
def sample_course_chunks():
    """Create sample course chunks for testing"""
    return [
        CourseChunk(
            content="Python is a high-level programming language known for its simplicity.",
            course_title="Python Fundamentals",
            lesson_number=1,
            chunk_index=0
        ),
        CourseChunk(
            content="Variables in Python are used to store data values.",
            course_title="Python Fundamentals", 
            lesson_number=2,
            chunk_index=1
        ),
        CourseChunk(
            content="Control structures like if-else statements control program flow.",
            course_title="Python Fundamentals",
            lesson_number=3,
            chunk_index=2
        )
    ]


@pytest.fixture
def successful_search_results():
    """Create successful search results for testing"""
    return SearchResults(
        documents=[
            "Python is a high-level programming language known for its simplicity.",
            "Variables in Python are used to store data values."
        ],
        metadata=[
            {
                "course_title": "Python Fundamentals",
                "lesson_number": 1,
                "chunk_index": 0
            },
            {
                "course_title": "Python Fundamentals", 
                "lesson_number": 2,
                "chunk_index": 1
            }
        ],
        distances=[0.1, 0.2]
    )


@pytest.fixture
def empty_search_results():
    """Create empty search results for testing"""
    return SearchResults(documents=[], metadata=[], distances=[])


@pytest.fixture
def error_search_results():
    """Create error search results for testing"""
    return SearchResults(
        documents=[], 
        metadata=[], 
        distances=[], 
        error="Database connection failed"
    )


@pytest.fixture
def mock_vector_store():
    """Create a mock vector store for testing"""
    mock_store = Mock()
    mock_store.search.return_value = SearchResults(documents=[], metadata=[], distances=[])
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
    mock_response.content[0].text = "Based on the course content, Python is a programming language."
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
                    "course_name": {"type": "string"}
                },
                "required": ["query"]
            }
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


@pytest.fixture
def temp_chroma_db():
    """Create a temporary ChromaDB directory for testing"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_rag_system():
    """Create a mock RAG system for API testing"""
    mock_rag = Mock()
    mock_rag.query.return_value = (
        "This is a test response from the RAG system.",
        [{"course_title": "Test Course", "lesson_number": 1, "content": "Test content"}]
    )
    mock_rag.get_course_analytics.return_value = {
        "total_courses": 2,
        "course_titles": ["Python Fundamentals", "Web Development"]
    }
    mock_rag.session_manager.create_session.return_value = "test_session_123"
    mock_rag.session_manager.clear_session.return_value = None
    mock_rag.add_course_folder.return_value = (2, 10)  # courses, chunks
    return mock_rag


@pytest.fixture
def test_app(mock_rag_system):
    """Create a test FastAPI app without static file mounting issues"""
    from fastapi import FastAPI, HTTPException
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.middleware.trustedhost import TrustedHostMiddleware
    from pydantic import BaseModel
    from typing import List, Optional, Dict, Any
    
    # Create test app
    app = FastAPI(title="Test Course Materials RAG System")
    
    # Add middleware
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        expose_headers=["*"],
    )
    
    # Pydantic models
    class QueryRequest(BaseModel):
        query: str
        session_id: Optional[str] = None

    class ResetSessionRequest(BaseModel):
        session_id: str

    class QueryResponse(BaseModel):
        answer: str
        sources: List[Dict[str, Any]]
        session_id: str

    class CourseStats(BaseModel):
        total_courses: int
        course_titles: List[str]
    
    # API endpoints
    @app.post("/api/query", response_model=QueryResponse)
    async def query_documents(request: QueryRequest):
        try:
            session_id = request.session_id
            if not session_id:
                session_id = mock_rag_system.session_manager.create_session()
            
            answer, sources = mock_rag_system.query(request.query, session_id)
            
            return QueryResponse(
                answer=answer,
                sources=sources,
                session_id=session_id
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/courses", response_model=CourseStats)
    async def get_course_stats():
        try:
            analytics = mock_rag_system.get_course_analytics()
            return CourseStats(
                total_courses=analytics["total_courses"],
                course_titles=analytics["course_titles"]
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/reset-session")
    async def reset_session(request: ResetSessionRequest):
        try:
            mock_rag_system.session_manager.clear_session(request.session_id)
            return {"success": True, "message": "Session reset successfully"}
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
    
    # Health check endpoint
    @app.get("/")
    async def root():
        return {"message": "RAG System API is running"}
    
    return app


@pytest.fixture
def test_client(test_app):
    """Create a test client for the FastAPI app"""
    return TestClient(test_app)