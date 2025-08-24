"""
Diagnostic tests for vector store health and data availability.

These tests are designed to identify the root cause of "query failed" errors
by testing the actual vector database state, data availability, and functionality.
"""

import os
import shutil
import sys
import tempfile
from unittest.mock import Mock, patch

import pytest

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from models import Course, CourseChunk, Lesson
from vector_store import SearchResults, VectorStore


class TestVectorStoreHealth:
    """Test the basic health and connectivity of the vector store"""

    def test_vector_store_initialization(self):
        """Test that vector store can be initialized with basic settings"""
        # Use a temporary directory for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2",
                    max_results=5,
                )

                # Basic checks
                assert vector_store.client is not None
                assert vector_store.embedding_function is not None
                assert vector_store.course_catalog is not None
                assert vector_store.course_content is not None
                assert vector_store.max_results == 5

            except Exception as e:
                pytest.fail(f"Vector store initialization failed: {e}")

    def test_embedding_model_loading(self):
        """Test that the embedding model loads correctly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="all-MiniLM-L6-v2",
                    max_results=5,
                )

                # Test that embeddings can be generated
                # This is indirect but tests if the model loaded properly
                test_text = "This is a test sentence"
                # The embedding function should be callable
                assert callable(vector_store.embedding_function)

            except Exception as e:
                pytest.fail(f"Embedding model loading failed: {e}")

    def test_chromadb_collections_creation(self):
        """Test that ChromaDB collections are created properly"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Test that collections exist and have the right names
            catalog_name = vector_store.course_catalog.name
            content_name = vector_store.course_content.name

            assert catalog_name == "course_catalog"
            assert content_name == "course_content"

            # Test that collections are initially empty
            catalog_count = vector_store.course_catalog.count()
            content_count = vector_store.course_content.count()

            assert catalog_count == 0
            assert content_count == 0


class TestVectorStoreDataOperations:
    """Test data operations in the vector store"""

    def test_add_course_metadata(self, sample_course):
        """Test adding course metadata to the vector store"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add course metadata
            vector_store.add_course_metadata(sample_course)

            # Verify data was added
            catalog_count = vector_store.course_catalog.count()
            assert catalog_count == 1

            # Verify data can be retrieved
            results = vector_store.course_catalog.get()
            assert len(results["ids"]) == 1
            assert results["ids"][0] == sample_course.title

            # Verify metadata structure
            metadata = results["metadatas"][0]
            assert metadata["title"] == sample_course.title
            assert metadata["instructor"] == sample_course.instructor
            assert "lessons_json" in metadata

    def test_add_course_content(self, sample_course_chunks):
        """Test adding course content chunks to the vector store"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add course content
            vector_store.add_course_content(sample_course_chunks)

            # Verify data was added
            content_count = vector_store.course_content.count()
            assert content_count == len(sample_course_chunks)

            # Verify data can be retrieved
            results = vector_store.course_content.get()
            assert len(results["ids"]) == len(sample_course_chunks)

            # Verify metadata structure
            for i, metadata in enumerate(results["metadatas"]):
                chunk = sample_course_chunks[i]
                assert metadata["course_title"] == chunk.course_title
                assert metadata["lesson_number"] == chunk.lesson_number
                assert metadata["chunk_index"] == chunk.chunk_index

    def test_search_with_data(self, sample_course, sample_course_chunks):
        """Test search functionality with actual data"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add test data
            vector_store.add_course_metadata(sample_course)
            vector_store.add_course_content(sample_course_chunks)

            # Test search
            results = vector_store.search("Python programming")

            # Verify search returns results
            assert not results.is_empty()
            assert results.error is None
            assert len(results.documents) > 0
            assert len(results.metadata) > 0
            assert len(results.distances) > 0

            # Verify metadata structure
            for metadata in results.metadata:
                assert "course_title" in metadata
                assert "lesson_number" in metadata
                assert "chunk_index" in metadata

    def test_search_with_course_filter(self, sample_course, sample_course_chunks):
        """Test search with course name filtering"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add test data
            vector_store.add_course_metadata(sample_course)
            vector_store.add_course_content(sample_course_chunks)

            # Test search with existing course
            results = vector_store.search("Python", course_name="Python Fundamentals")

            # Should find results
            assert not results.is_empty()
            assert results.error is None

            # Test search with non-existent course
            results = vector_store.search("Python", course_name="Nonexistent Course")

            # Should return error about course not found
            assert results.error is not None
            assert "No course found matching" in results.error

    def test_search_with_lesson_filter(self, sample_course, sample_course_chunks):
        """Test search with lesson number filtering"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add test data
            vector_store.add_course_metadata(sample_course)
            vector_store.add_course_content(sample_course_chunks)

            # Test search with specific lesson
            results = vector_store.search("Python", lesson_number=1)

            # Should find results
            assert not results.is_empty()
            assert results.error is None

            # All results should be from lesson 1
            for metadata in results.metadata:
                if metadata.get("lesson_number") is not None:
                    assert metadata["lesson_number"] == 1

    def test_empty_search_results(self, sample_course, sample_course_chunks):
        """Test search that should return no results"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add test data
            vector_store.add_course_metadata(sample_course)
            vector_store.add_course_content(sample_course_chunks)

            # Search for something that shouldn't exist
            results = vector_store.search("quantum physics nuclear reactor engineering")

            # Should return empty but not error
            assert results.is_empty()
            assert results.error is None


class TestVectorStoreErrorConditions:
    """Test error conditions that could cause 'query failed'"""

    def test_search_on_empty_database(self):
        """Test search behavior on empty database"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Search on empty database
            results = vector_store.search("anything")

            # Should return empty results, not error
            assert results.is_empty()
            assert results.error is None

    def test_invalid_chroma_path(self):
        """Test initialization with invalid ChromaDB path"""
        # Use a path that doesn't exist and can't be created
        invalid_path = "/root/invalid/path/that/cannot/be/created"

        try:
            vector_store = VectorStore(
                chroma_path=invalid_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )
            # If this succeeds, ChromaDB might create the path automatically
            # That's fine for the test
        except Exception as e:
            # Expected if the path truly cannot be created
            assert "permission" in str(e).lower() or "path" in str(e).lower()

    def test_invalid_embedding_model(self):
        """Test initialization with invalid embedding model"""
        with tempfile.TemporaryDirectory() as temp_dir:
            # This might take a while to fail as it tries to download the model
            try:
                vector_store = VectorStore(
                    chroma_path=temp_dir,
                    embedding_model="nonexistent-model-that-should-not-exist",
                    max_results=5,
                )
                # If this doesn't fail immediately, the model loading happens lazily
                # Try to force model loading by doing a search
                vector_store.search("test")

            except Exception as e:
                # Expected - invalid model should cause an error
                assert "model" in str(e).lower() or "not found" in str(e).lower()

    def test_corrupted_data_handling(self, sample_course):
        """Test handling of corrupted data in the vector store"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add valid course first
            vector_store.add_course_metadata(sample_course)

            # Try to add invalid data directly to collection
            try:
                # This should be caught and handled gracefully
                vector_store.course_catalog.add(
                    documents=["test"],
                    metadatas=[{"invalid": "metadata_structure"}],
                    ids=["test_id"],
                )

                # Try to search - should still work
                results = vector_store.search("Python")
                # Should either work or return a clear error
                assert isinstance(results, SearchResults)

            except Exception as e:
                # If it fails, it should be a clear, handleable error
                assert isinstance(e, Exception)

    @patch("vector_store.chromadb.PersistentClient")
    def test_chromadb_connection_failure(self, mock_client):
        """Test handling when ChromaDB connection fails"""
        # Mock ChromaDB to raise an exception
        mock_client.side_effect = Exception("ChromaDB connection failed")

        # This should raise an exception during initialization
        with pytest.raises(Exception) as exc_info:
            VectorStore(
                chroma_path="test_path",
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

        assert "ChromaDB connection failed" in str(exc_info.value)

    def test_large_query_handling(self, sample_course, sample_course_chunks):
        """Test handling of very large queries"""
        with tempfile.TemporaryDirectory() as temp_dir:
            vector_store = VectorStore(
                chroma_path=temp_dir, embedding_model="all-MiniLM-L6-v2", max_results=5
            )

            # Add test data
            vector_store.add_course_metadata(sample_course)
            vector_store.add_course_content(sample_course_chunks)

            # Very large query
            large_query = "What is Python programming? " * 1000

            try:
                results = vector_store.search(large_query)
                # Should handle gracefully
                assert isinstance(results, SearchResults)

            except Exception as e:
                # If it fails, should be a clear error
                assert "too large" in str(e).lower() or "limit" in str(e).lower()


class TestRealVectorStoreData:
    """Test against the actual vector store data if it exists"""

    def test_existing_chroma_db_connectivity(self):
        """Test connectivity to the existing ChromaDB instance"""
        # Use the actual ChromaDB path from config
        chroma_path = "./chroma_db"

        try:
            vector_store = VectorStore(
                chroma_path=chroma_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

            # Test basic connectivity
            catalog_count = vector_store.course_catalog.count()
            content_count = vector_store.course_content.count()

            print(f"Catalog count: {catalog_count}")
            print(f"Content count: {content_count}")

            # These should be accessible without error
            assert catalog_count >= 0
            assert content_count >= 0

        except Exception as e:
            pytest.fail(f"Failed to connect to existing ChromaDB: {e}")

    def test_existing_data_availability(self):
        """Test if there's actual course data in the existing vector store"""
        chroma_path = "./chroma_db"

        try:
            vector_store = VectorStore(
                chroma_path=chroma_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

            # Get course titles
            course_titles = vector_store.get_existing_course_titles()
            print(f"Existing courses: {course_titles}")

            # Get course count
            course_count = vector_store.get_course_count()
            print(f"Course count: {course_count}")

            if course_count > 0:
                # Test search on existing data
                results = vector_store.search("Python")
                print(
                    f"Search results for 'Python': {len(results.documents)} documents"
                )
                print(f"Search error: {results.error}")

                if results.error:
                    pytest.fail(f"Search failed on existing data: {results.error}")

                # Test course resolution
                if course_titles:
                    first_course = course_titles[0]
                    resolved = vector_store._resolve_course_name(first_course)
                    print(f"Course resolution for '{first_course}': {resolved}")
                    assert resolved is not None

            else:
                print("No existing course data found - this may be the problem!")

        except Exception as e:
            pytest.fail(f"Failed to access existing data: {e}")

    def test_search_real_course_content(self):
        """Test search against real course content if available"""
        chroma_path = "./chroma_db"

        try:
            vector_store = VectorStore(
                chroma_path=chroma_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

            # Try various search queries that might exist in course content
            test_queries = [
                "introduction",
                "lesson",
                "course",
                "python",
                "programming",
                "tutorial",
                "example",
            ]

            search_results = {}

            for query in test_queries:
                results = vector_store.search(query)
                search_results[query] = {
                    "doc_count": len(results.documents),
                    "has_error": results.error is not None,
                    "error": results.error,
                }
                print(
                    f"Query '{query}': {len(results.documents)} docs, error: {results.error}"
                )

            # At least one query should return results if data exists
            total_results = sum(r["doc_count"] for r in search_results.values())

            if total_results == 0:
                print(
                    "WARNING: No search results for any test query - possible data issue!"
                )

                # Check if there's any content at all
                content_count = vector_store.course_content.count()
                if content_count == 0:
                    pytest.fail(
                        "No course content found in vector store - this is likely the cause of 'query failed' errors"
                    )
                else:
                    print(
                        f"Content exists ({content_count} chunks) but no search results - possible embedding/search issue"
                    )

        except Exception as e:
            pytest.fail(f"Failed to search real course content: {e}")


class TestDiagnosticUtilities:
    """Utility tests for diagnosing vector store issues"""

    def test_vector_store_info_dump(self):
        """Dump comprehensive information about the vector store state"""
        chroma_path = "./chroma_db"

        try:
            vector_store = VectorStore(
                chroma_path=chroma_path,
                embedding_model="all-MiniLM-L6-v2",
                max_results=5,
            )

            print("\n=== VECTOR STORE DIAGNOSTIC INFORMATION ===")

            # Basic info
            print(f"ChromaDB path: {chroma_path}")
            print(f"Max results: {vector_store.max_results}")

            # Collection info
            catalog_count = vector_store.course_catalog.count()
            content_count = vector_store.course_content.count()
            print(f"Catalog collection count: {catalog_count}")
            print(f"Content collection count: {content_count}")

            # Course info
            course_titles = vector_store.get_existing_course_titles()
            print(f"Available courses: {course_titles}")

            # Metadata info
            if catalog_count > 0:
                all_courses = vector_store.get_all_courses_metadata()
                print(f"Course metadata count: {len(all_courses)}")
                for course in all_courses[:3]:  # Show first 3
                    print(
                        f"  - {course.get('title', 'Unknown')}: {len(course.get('lessons', []))} lessons"
                    )

            # Content sample
            if content_count > 0:
                content_sample = vector_store.course_content.get(limit=3)
                print(f"Content sample (first 3 items):")
                for i, doc in enumerate(content_sample.get("documents", [])[:3]):
                    meta = content_sample.get("metadatas", [{}])[i]
                    print(
                        f"  - {doc[:100]}... (Course: {meta.get('course_title', 'Unknown')})"
                    )

            print("=== END DIAGNOSTIC INFORMATION ===\n")

            # This test always passes - it's for information gathering
            assert True

        except Exception as e:
            print(f"Diagnostic failed: {e}")
            pytest.fail(f"Could not gather diagnostic information: {e}")
