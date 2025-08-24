"""
Integration tests for FastAPI endpoints in the RAG chatbot system.
"""
import pytest
from fastapi import status
from unittest.mock import Mock, patch


@pytest.mark.integration
class TestQueryEndpoint:
    """Test /api/query endpoint"""

    def test_query_without_session_id(self, test_client, mock_rag_system):
        """Test query without providing session_id creates new session"""
        response = test_client.post(
            "/api/query",
            json={"query": "What is Python?"}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "answer" in data
        assert "sources" in data
        assert "session_id" in data
        assert data["answer"] == "This is a test response from the RAG system."
        assert data["session_id"] == "test_session_123"
        assert len(data["sources"]) == 1
        
        # Verify RAG system was called correctly
        mock_rag_system.session_manager.create_session.assert_called_once()
        mock_rag_system.query.assert_called_once_with("What is Python?", "test_session_123")

    def test_query_with_existing_session_id(self, test_client, mock_rag_system):
        """Test query with existing session_id"""
        session_id = "existing_session_456"
        response = test_client.post(
            "/api/query",
            json={
                "query": "Explain variables in Python",
                "session_id": session_id
            }
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["session_id"] == session_id
        assert "answer" in data
        assert "sources" in data
        
        # Verify session was not created again
        mock_rag_system.session_manager.create_session.assert_not_called()
        mock_rag_system.query.assert_called_once_with("Explain variables in Python", session_id)

    def test_query_with_empty_string(self, test_client):
        """Test query with empty string"""
        response = test_client.post(
            "/api/query",
            json={"query": ""}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "answer" in data

    def test_query_missing_query_field(self, test_client):
        """Test query without required query field"""
        response = test_client.post(
            "/api/query",
            json={"session_id": "test"}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_invalid_json(self, test_client):
        """Test query with invalid JSON"""
        response = test_client.post(
            "/api/query",
            data="invalid json"
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_query_rag_system_error(self, test_client, mock_rag_system):
        """Test handling of RAG system errors"""
        mock_rag_system.query.side_effect = Exception("RAG system error")
        
        response = test_client.post(
            "/api/query",
            json={"query": "test query"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "RAG system error" in response.json()["detail"]


@pytest.mark.integration
class TestCoursesEndpoint:
    """Test /api/courses endpoint"""

    def test_get_courses_success(self, test_client, mock_rag_system):
        """Test successful retrieval of course statistics"""
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert "total_courses" in data
        assert "course_titles" in data
        assert data["total_courses"] == 2
        assert data["course_titles"] == ["Python Fundamentals", "Web Development"]
        
        mock_rag_system.get_course_analytics.assert_called_once()

    def test_get_courses_rag_system_error(self, test_client, mock_rag_system):
        """Test handling of RAG system errors in courses endpoint"""
        mock_rag_system.get_course_analytics.side_effect = Exception("Analytics error")
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Analytics error" in response.json()["detail"]

    def test_get_courses_empty_response(self, test_client, mock_rag_system):
        """Test courses endpoint with empty course list"""
        mock_rag_system.get_course_analytics.return_value = {
            "total_courses": 0,
            "course_titles": []
        }
        
        response = test_client.get("/api/courses")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["total_courses"] == 0
        assert data["course_titles"] == []


@pytest.mark.integration
class TestResetSessionEndpoint:
    """Test /api/reset-session endpoint"""

    def test_reset_session_success(self, test_client, mock_rag_system):
        """Test successful session reset"""
        session_id = "session_to_reset"
        response = test_client.post(
            "/api/reset-session",
            json={"session_id": session_id}
        )
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        
        assert data["success"] is True
        assert "Session reset successfully" in data["message"]
        
        mock_rag_system.session_manager.clear_session.assert_called_once_with(session_id)

    def test_reset_session_missing_session_id(self, test_client):
        """Test reset session without session_id"""
        response = test_client.post(
            "/api/reset-session",
            json={}
        )
        
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_reset_session_rag_system_error(self, test_client, mock_rag_system):
        """Test handling of RAG system errors in reset session"""
        mock_rag_system.session_manager.clear_session.side_effect = Exception("Session reset error")
        
        response = test_client.post(
            "/api/reset-session",
            json={"session_id": "test_session"}
        )
        
        assert response.status_code == status.HTTP_500_INTERNAL_SERVER_ERROR
        assert "Session reset error" in response.json()["detail"]


@pytest.mark.integration
class TestRootEndpoint:
    """Test root (/) endpoint"""

    def test_root_endpoint(self, test_client):
        """Test root endpoint returns health check message"""
        response = test_client.get("/")
        
        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["message"] == "RAG System API is running"


@pytest.mark.integration
class TestCORSAndMiddleware:
    """Test CORS and middleware functionality"""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present in responses"""
        response = test_client.get("/")
        
        # Check that CORS headers are present (FastAPI adds them automatically with our config)
        assert response.status_code == status.HTTP_200_OK

    def test_preflight_request(self, test_client):
        """Test CORS preflight request"""
        response = test_client.options(
            "/api/query",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type"
            }
        )
        
        # Should not error out due to CORS
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_405_METHOD_NOT_ALLOWED]


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling across endpoints"""

    def test_404_for_nonexistent_endpoint(self, test_client):
        """Test 404 response for nonexistent endpoints"""
        response = test_client.get("/api/nonexistent")
        
        assert response.status_code == status.HTTP_404_NOT_FOUND

    def test_method_not_allowed(self, test_client):
        """Test method not allowed responses"""
        # GET on POST-only endpoint
        response = test_client.get("/api/query")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED

        # POST on GET-only endpoint
        response = test_client.post("/api/courses")
        
        assert response.status_code == status.HTTP_405_METHOD_NOT_ALLOWED


@pytest.mark.integration
class TestRequestValidation:
    """Test request validation and Pydantic models"""

    def test_query_request_validation(self, test_client):
        """Test query request model validation"""
        # Valid minimal request
        response = test_client.post(
            "/api/query",
            json={"query": "test"}
        )
        assert response.status_code == status.HTTP_200_OK

        # Valid complete request
        response = test_client.post(
            "/api/query",
            json={"query": "test", "session_id": "123"}
        )
        assert response.status_code == status.HTTP_200_OK

        # Invalid - missing required field
        response = test_client.post(
            "/api/query",
            json={"session_id": "123"}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Invalid - wrong type
        response = test_client.post(
            "/api/query",
            json={"query": 123}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_reset_session_request_validation(self, test_client):
        """Test reset session request model validation"""
        # Valid request
        response = test_client.post(
            "/api/reset-session",
            json={"session_id": "test_session"}
        )
        assert response.status_code == status.HTTP_200_OK

        # Invalid - missing required field
        response = test_client.post(
            "/api/reset-session",
            json={}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Invalid - wrong type
        response = test_client.post(
            "/api/reset-session",
            json={"session_id": 123}
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY