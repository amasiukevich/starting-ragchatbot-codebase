"""
Unit tests for AIGenerator tool calling mechanism.

These tests focus on identifying issues in the AI tool calling flow
that could cause "query failed" errors or prevent proper tool execution.
"""
import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock

# Add backend to path
backend_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, backend_dir)

from ai_generator import AIGenerator
from search_tools import ToolManager, CourseSearchTool


class TestAIGeneratorToolCalling:
    """Test tool calling functionality in AIGenerator"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tools_passed_to_api(self, mock_anthropic, sample_tool_definitions):
        """Test that tools are correctly passed to Anthropic API"""
        # Setup mock client
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        # Create AIGenerator
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # Generate response with tools
        result = ai_gen.generate_response(
            query="What is Python?",
            tools=sample_tool_definitions
        )
        
        # Verify API was called with tools
        call_args = mock_client.messages.create.call_args
        assert call_args[1]["tools"] == sample_tool_definitions
        assert call_args[1]["tool_choice"] == {"type": "auto"}
        assert result == "Test response"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_no_tools_provided(self, mock_anthropic):
        """Test normal operation when no tools are provided"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response without tools"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        result = ai_gen.generate_response(query="What is Python?")
        
        # Verify no tools were passed
        call_args = mock_client.messages.create.call_args
        assert "tools" not in call_args[1]
        assert "tool_choice" not in call_args[1]
        assert result == "Test response without tools"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_use_triggers_execution(self, mock_anthropic, mock_tool_use_response, mock_final_response):
        """Test that tool_use stop reason triggers tool execution"""
        mock_client = Mock()
        # First call returns tool_use, second returns final response
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Create mock tool manager
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        result = ai_gen.generate_response(
            query="What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content",
            query="test query",
            course_name="Python"
        )
        
        # Verify final response was returned
        assert result == "Based on the course content, Python is a programming language."
        
        # Verify two API calls were made
        assert mock_client.messages.create.call_count == 2
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_use_without_tool_manager(self, mock_anthropic, mock_tool_use_response):
        """Test handling when tool_use occurs but no tool_manager provided"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_tool_use_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # This should handle gracefully - tool_use with no tool_manager
        result = ai_gen.generate_response(
            query="What is Python?",
            tools=[{"name": "search_course_content"}]
            # Note: no tool_manager provided
        )
        
        # Should not crash and should return something meaningful
        assert isinstance(result, str)
        # Only one API call should be made
        assert mock_client.messages.create.call_count == 1
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_execution_error_handling(self, mock_anthropic, mock_tool_use_response, mock_final_response):
        """Test handling when tool execution fails"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Mock tool manager that raises an exception
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = Exception("Tool execution failed")
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # This should handle the tool execution error gracefully
        try:
            result = ai_gen.generate_response(
                query="What is Python?",
                tools=[{"name": "search_course_content"}],
                tool_manager=mock_tool_manager
            )
            # If no exception is raised, verify it's handled properly
            assert isinstance(result, str)
        except Exception as e:
            # If exception propagates, that's a potential source of "query failed"
            pytest.fail(f"Tool execution error should be handled gracefully, but raised: {e}")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_anthropic_api_error_handling(self, mock_anthropic):
        """Test handling when Anthropic API fails"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("API connection failed")
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # This should raise an exception that can be caught by higher-level code
        with pytest.raises(Exception) as exc_info:
            ai_gen.generate_response(query="What is Python?")
        
        assert "API connection failed" in str(exc_info.value)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_multiple_tool_calls_in_response(self, mock_anthropic, mock_final_response):
        """Test handling multiple tool calls in a single response"""
        # Create response with multiple tool use blocks
        mock_response = Mock()
        tool_block1 = Mock()
        tool_block1.type = "tool_use"
        tool_block1.name = "search_course_content"
        tool_block1.id = "tool_1"
        tool_block1.input = {"query": "Python basics"}
        
        tool_block2 = Mock()
        tool_block2.type = "tool_use"
        tool_block2.name = "get_course_outline"
        tool_block2.id = "tool_2"
        tool_block2.input = {"course_name": "Python"}
        
        mock_response.content = [tool_block1, tool_block2]
        mock_response.stop_reason = "tool_use"
        
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.side_effect = ["Result 1", "Result 2"]
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        result = ai_gen.generate_response(
            query="Tell me about Python course",
            tools=[{"name": "search_course_content"}, {"name": "get_course_outline"}],
            tool_manager=mock_tool_manager
        )
        
        # Verify both tools were executed
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="Python basics")
        mock_tool_manager.execute_tool.assert_any_call("get_course_outline", course_name="Python")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_tool_result_format(self, mock_anthropic, mock_tool_use_response, mock_final_response):
        """Test that tool results are formatted correctly for API"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        mock_tool_manager = Mock()
        mock_tool_manager.execute_tool.return_value = "Tool execution result"
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        ai_gen.generate_response(
            query="What is Python?",
            tools=[{"name": "search_course_content"}],
            tool_manager=mock_tool_manager
        )
        
        # Check the second API call (final response) for proper tool result format
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        
        # Should have 3 messages: user, assistant (tool use), user (tool results)
        assert len(messages) == 3
        
        # Check tool result message format
        tool_result_message = messages[2]
        assert tool_result_message["role"] == "user"
        assert "content" in tool_result_message
        
        tool_result_content = tool_result_message["content"][0]
        assert tool_result_content["type"] == "tool_result"
        assert tool_result_content["tool_use_id"] == "tool_123"
        assert tool_result_content["content"] == "Tool execution result"
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_conversation_history_with_tools(self, mock_anthropic, mock_final_response):
        """Test that conversation history is preserved when using tools"""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_final_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        conversation_history = "User: Hello\nAssistant: Hi there!"
        
        ai_gen.generate_response(
            query="What is Python?",
            conversation_history=conversation_history,
            tools=[{"name": "search_course_content"}]
        )
        
        # Verify conversation history is included in system prompt
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation:" in system_content
        assert "Hello" in system_content
        assert "Hi there!" in system_content
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_system_prompt_content(self, mock_anthropic):
        """Test that system prompt contains proper tool usage instructions"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Test response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        ai_gen.generate_response(
            query="What is Python?",
            tools=[{"name": "search_course_content"}]
        )
        
        call_args = mock_client.messages.create.call_args
        system_content = call_args[1]["system"]
        
        # Verify key instruction elements
        assert "Content Search Tool" in system_content
        assert "Course Outline Tool" in system_content
        assert "Up to 2 tool usage rounds" in system_content
        assert "One tool use per round maximum" in system_content
        assert "course materials" in system_content.lower()


class TestAIGeneratorIntegrationWithTools:
    """Integration tests for AIGenerator with real ToolManager and tools"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_integration_with_tool_manager(self, mock_anthropic, mock_tool_use_response, mock_final_response, mock_vector_store):
        """Test full integration with ToolManager and CourseSearchTool"""
        # Setup Anthropic mock
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Setup vector store mock
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=["Python is a programming language"],
            metadata=[{"course_title": "Python Basics", "lesson_number": 1}],
            distances=[0.1]
        )
        
        # Create real tool manager and search tool
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        # Create AI generator
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # Execute full flow
        result = ai_gen.generate_response(
            query="What is Python?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # Verify the search was executed
        mock_vector_store.search.assert_called_once()
        
        # Verify final response
        assert result == "Based on the course content, Python is a programming language."
    
    @patch('ai_generator.anthropic.Anthropic') 
    def test_tool_error_propagation(self, mock_anthropic, mock_tool_use_response, mock_final_response, mock_vector_store):
        """Test that tool errors are properly handled and propagated"""
        # Setup Anthropic mock
        mock_client = Mock()
        mock_client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        mock_anthropic.return_value = mock_client
        
        # Setup vector store to return error
        from vector_store import SearchResults
        mock_vector_store.search.return_value = SearchResults(
            documents=[], metadata=[], distances=[], error="Vector store connection failed"
        )
        
        # Create tool manager with real search tool
        tool_manager = ToolManager()
        search_tool = CourseSearchTool(mock_vector_store)
        tool_manager.register_tool(search_tool)
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # Execute - this should not crash
        result = ai_gen.generate_response(
            query="What is Python?",
            tools=tool_manager.get_tool_definitions(),
            tool_manager=tool_manager
        )
        
        # The error should be included in the tool result sent to Claude
        second_call_args = mock_client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        tool_result_content = messages[2]["content"][0]["content"]
        
        # The error message should be passed to Claude
        assert "Vector store connection failed" in tool_result_content
        
        # Claude should still return a response
        assert result == "Based on the course content, Python is a programming language."


class TestAIGeneratorErrorConditions:
    """Test error conditions that could cause 'query failed'"""
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_invalid_api_key(self, mock_anthropic):
        """Test handling of invalid API key"""
        mock_anthropic.side_effect = Exception("Invalid API key")
        
        # This should raise an exception during initialization
        with pytest.raises(Exception):
            AIGenerator("invalid_key", "claude-3-sonnet-20240229")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_malformed_tool_definitions(self, mock_anthropic):
        """Test handling of malformed tool definitions"""
        mock_client = Mock()
        mock_client.messages.create.side_effect = Exception("Invalid tool schema")
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # Malformed tool definition
        bad_tools = [{"invalid": "tool_def"}]
        
        with pytest.raises(Exception) as exc_info:
            ai_gen.generate_response(
                query="What is Python?",
                tools=bad_tools
            )
        
        assert "Invalid tool schema" in str(exc_info.value)
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_response_without_content(self, mock_anthropic):
        """Test handling of malformed API response"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = []  # Empty content
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # This should handle the malformed response gracefully
        with pytest.raises(IndexError):
            ai_gen.generate_response(query="What is Python?")
    
    @patch('ai_generator.anthropic.Anthropic')
    def test_very_long_conversation_history(self, mock_anthropic):
        """Test handling of very long conversation history"""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.content = [Mock()]
        mock_response.content[0].text = "Response"
        mock_response.stop_reason = "end_turn"
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client
        
        ai_gen = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
        
        # Very long conversation history
        long_history = "User: Question\nAssistant: Answer\n" * 1000
        
        # Should still work without crashing
        result = ai_gen.generate_response(
            query="What is Python?",
            conversation_history=long_history
        )
        
        assert result == "Response"