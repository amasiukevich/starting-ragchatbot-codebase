import pytest
from unittest.mock import Mock, MagicMock
from search_tools import Tool, CourseSearchTool, CourseOutlineTool, ToolManager
from vector_store import SearchResults


class MockTool(Tool):
    def get_tool_definition(self):
        return {"name": "mock_tool", "description": "A mock tool"}
    
    def execute(self, **kwargs):
        return "mock result"


class TestTool:
    def test_tool_is_abstract(self):
        with pytest.raises(TypeError):
            Tool()


class TestCourseSearchTool:
    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock()
        mock_store.get_lesson_link.return_value = "http://example.com/lesson1"
        return mock_store
    
    @pytest.fixture
    def search_tool(self, mock_vector_store):
        return CourseSearchTool(mock_vector_store)
    
    def test_get_tool_definition(self, search_tool):
        definition = search_tool.get_tool_definition()
        
        assert definition["name"] == "search_course_content"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["query"]
        
        properties = definition["input_schema"]["properties"]
        assert "query" in properties
        assert "course_name" in properties
        assert "lesson_number" in properties
    
    def test_execute_with_error(self, search_tool, mock_vector_store):
        error_results = SearchResults(
            documents=[],
            metadata=[],
            distances=[],
            error="Database connection failed"
        )
        mock_vector_store.search.return_value = error_results
        
        result = search_tool.execute("test query")
        assert result == "Database connection failed"
    
    def test_execute_with_empty_results(self, search_tool, mock_vector_store):
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("test query")
        assert result == "No relevant content found."
    
    def test_execute_with_course_filter(self, search_tool, mock_vector_store):
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("test query", course_name="Python")
        assert result == "No relevant content found in course 'Python'."
    
    def test_execute_with_lesson_filter(self, search_tool, mock_vector_store):
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("test query", lesson_number=1)
        assert result == "No relevant content found in lesson 1."
    
    def test_execute_with_both_filters(self, search_tool, mock_vector_store):
        empty_results = SearchResults(documents=[], metadata=[], distances=[])
        mock_vector_store.search.return_value = empty_results
        
        result = search_tool.execute("test query", course_name="Python", lesson_number=1)
        assert result == "No relevant content found in course 'Python' in lesson 1."
    
    def test_execute_with_results(self, search_tool, mock_vector_store):
        results = SearchResults(
            documents=["This is content about Python"],
            metadata=[{
                "course_title": "Python Basics",
                "lesson_number": 1
            }],
            distances=[0.1]
        )
        mock_vector_store.search.return_value = results
        
        result = search_tool.execute("test query")
        
        assert "[Python Basics - Lesson 1]" in result
        assert "This is content about Python" in result
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "Python Basics - Lesson 1"
        assert search_tool.last_sources[0]["link"] == "http://example.com/lesson1"
    
    def test_execute_with_no_lesson_number(self, search_tool, mock_vector_store):
        results = SearchResults(
            documents=["General course content"],
            metadata=[{
                "course_title": "Python Basics"
            }],
            distances=[0.2]
        )
        mock_vector_store.search.return_value = results
        
        result = search_tool.execute("test query")
        
        assert "[Python Basics]" in result
        assert "General course content" in result
        assert len(search_tool.last_sources) == 1
        assert search_tool.last_sources[0]["text"] == "Python Basics"
        assert search_tool.last_sources[0]["link"] is None


class TestCourseOutlineTool:
    @pytest.fixture
    def mock_vector_store(self):
        mock_store = Mock()
        mock_store._resolve_course_name.return_value = "Python Basics Course"
        mock_store.get_all_courses_metadata.return_value = [
            {
                "title": "Python Basics Course",
                "course_link": "http://example.com/python",
                "lessons": [
                    {"lesson_number": 1, "lesson_title": "Introduction"},
                    {"lesson_number": 2, "lesson_title": "Variables"}
                ]
            }
        ]
        return mock_store
    
    @pytest.fixture
    def outline_tool(self, mock_vector_store):
        return CourseOutlineTool(mock_vector_store)
    
    def test_get_tool_definition(self, outline_tool):
        definition = outline_tool.get_tool_definition()
        
        assert definition["name"] == "get_course_outline"
        assert "description" in definition
        assert "input_schema" in definition
        assert definition["input_schema"]["required"] == ["course_name"]
        
        properties = definition["input_schema"]["properties"]
        assert "course_name" in properties
    
    def test_execute_course_not_found(self, outline_tool, mock_vector_store):
        mock_vector_store._resolve_course_name.return_value = None
        
        result = outline_tool.execute("NonExistent")
        assert result == "No course found matching 'NonExistent'"
    
    def test_execute_metadata_not_found(self, outline_tool, mock_vector_store):
        mock_vector_store._resolve_course_name.return_value = "Python Basics Course"
        mock_vector_store.get_all_courses_metadata.return_value = []
        
        result = outline_tool.execute("Python")
        assert result == "No course found matching 'Python'"
    
    def test_execute_successful(self, outline_tool):
        result = outline_tool.execute("Python")
        
        assert "**Python Basics Course**" in result
        assert "🔗 [View Course](http://example.com/python)" in result
        assert "**Lessons:**" in result
        assert "1. Introduction" in result
        assert "2. Variables" in result
    
    def test_execute_no_link(self, outline_tool, mock_vector_store):
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Python Basics Course",
                "course_link": "",
                "lessons": [
                    {"lesson_number": 1, "lesson_title": "Introduction"}
                ]
            }
        ]
        
        result = outline_tool.execute("Python")
        
        assert "**Python Basics Course**" in result
        assert "🔗" not in result
        assert "1. Introduction" in result
    
    def test_execute_no_lessons(self, outline_tool, mock_vector_store):
        mock_vector_store.get_all_courses_metadata.return_value = [
            {
                "title": "Python Basics Course",
                "course_link": "http://example.com/python",
                "lessons": []
            }
        ]
        
        result = outline_tool.execute("Python")
        
        assert "**Python Basics Course**" in result
        assert "*No lessons available*" in result


class TestToolManager:
    @pytest.fixture
    def tool_manager(self):
        return ToolManager()
    
    @pytest.fixture
    def mock_tool(self):
        tool = MockTool()
        tool.last_sources = [{"text": "test source", "link": "http://test.com"}]
        return tool
    
    def test_register_tool(self, tool_manager, mock_tool):
        tool_manager.register_tool(mock_tool)
        assert "mock_tool" in tool_manager.tools
    
    def test_register_tool_without_name(self, tool_manager):
        bad_tool = Mock()
        bad_tool.get_tool_definition.return_value = {"description": "bad tool"}
        
        with pytest.raises(ValueError, match="Tool must have a 'name' in its definition"):
            tool_manager.register_tool(bad_tool)
    
    def test_get_tool_definitions(self, tool_manager, mock_tool):
        tool_manager.register_tool(mock_tool)
        definitions = tool_manager.get_tool_definitions()
        
        assert len(definitions) == 1
        assert definitions[0]["name"] == "mock_tool"
    
    def test_execute_tool(self, tool_manager, mock_tool):
        tool_manager.register_tool(mock_tool)
        result = tool_manager.execute_tool("mock_tool", param1="value1")
        
        assert result == "mock result"
    
    def test_execute_nonexistent_tool(self, tool_manager):
        result = tool_manager.execute_tool("nonexistent_tool")
        assert result == "Tool 'nonexistent_tool' not found"
    
    def test_get_last_sources(self, tool_manager, mock_tool):
        tool_manager.register_tool(mock_tool)
        sources = tool_manager.get_last_sources()
        
        assert len(sources) == 1
        assert sources[0]["text"] == "test source"
        assert sources[0]["link"] == "http://test.com"
    
    def test_get_last_sources_empty(self, tool_manager):
        sources = tool_manager.get_last_sources()
        assert sources == []
    
    def test_reset_sources(self, tool_manager, mock_tool):
        tool_manager.register_tool(mock_tool)
        
        assert len(tool_manager.get_last_sources()) == 1
        
        tool_manager.reset_sources()
        
        assert len(tool_manager.get_last_sources()) == 0
        assert mock_tool.last_sources == []