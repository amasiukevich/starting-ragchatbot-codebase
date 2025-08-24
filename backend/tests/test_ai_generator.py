import pytest
from unittest.mock import Mock, patch, MagicMock
from ai_generator import AIGenerator


class TestAIGenerator:
    @pytest.fixture
    def mock_anthropic_response(self):
        response = Mock()
        response.content = [Mock()]
        response.content[0].text = "Test AI response"
        response.stop_reason = "end_turn"
        return response
    
    @pytest.fixture
    def mock_tool_use_response(self):
        response = Mock()
        
        # Mock content blocks for tool use
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test query"}
        
        response.content = [tool_block]
        response.stop_reason = "tool_use"
        return response
    
    @pytest.fixture
    def mock_final_response(self):
        response = Mock()
        response.content = [Mock()]
        response.content[0].text = "Final AI response with tool results"
        response.stop_reason = "end_turn"
        return response
    
    @pytest.fixture
    def mock_tool_manager(self):
        manager = Mock()
        manager.execute_tool.return_value = "Tool execution result"
        return manager
    
    @pytest.fixture
    def ai_generator(self):
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            generator = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
            generator.client = mock_client
            return generator
    
    def test_init(self):
        with patch('ai_generator.anthropic.Anthropic') as mock_anthropic:
            mock_client = Mock()
            mock_anthropic.return_value = mock_client
            
            generator = AIGenerator("test_api_key", "claude-3-sonnet-20240229")
            
            mock_anthropic.assert_called_once_with(api_key="test_api_key")
            assert generator.model == "claude-3-sonnet-20240229"
            assert generator.base_params["model"] == "claude-3-sonnet-20240229"
            assert generator.base_params["temperature"] == 0
            assert generator.base_params["max_tokens"] == 800
    
    def test_generate_response_simple(self, ai_generator, mock_anthropic_response):
        ai_generator.client.messages.create.return_value = mock_anthropic_response
        
        result = ai_generator.generate_response("What is Python?")
        
        assert result == "Test AI response"
        
        # Verify the API call
        call_args = ai_generator.client.messages.create.call_args
        assert call_args[1]["model"] == "claude-3-sonnet-20240229"
        assert call_args[1]["temperature"] == 0
        assert call_args[1]["max_tokens"] == 800
        assert call_args[1]["messages"][0]["content"] == "What is Python?"
        assert call_args[1]["messages"][0]["role"] == "user"
        assert AIGenerator.SYSTEM_PROMPT in call_args[1]["system"]
    
    def test_generate_response_with_history(self, ai_generator, mock_anthropic_response):
        ai_generator.client.messages.create.return_value = mock_anthropic_response
        history = "Previous conversation content"
        
        result = ai_generator.generate_response("What is Python?", conversation_history=history)
        
        assert result == "Test AI response"
        
        # Verify history is included in system prompt
        call_args = ai_generator.client.messages.create.call_args
        system_content = call_args[1]["system"]
        assert "Previous conversation content" in system_content
        assert AIGenerator.SYSTEM_PROMPT in system_content
    
    def test_generate_response_with_tools(self, ai_generator, mock_anthropic_response):
        ai_generator.client.messages.create.return_value = mock_anthropic_response
        tools = [{"name": "test_tool", "description": "A test tool"}]
        
        result = ai_generator.generate_response("What is Python?", tools=tools)
        
        assert result == "Test AI response"
        
        # Verify tools are included
        call_args = ai_generator.client.messages.create.call_args
        assert call_args[1]["tools"] == tools
        assert call_args[1]["tool_choice"] == {"type": "auto"}
    
    def test_generate_response_with_tool_use(self, ai_generator, mock_tool_use_response, 
                                           mock_final_response, mock_tool_manager):
        # First call returns tool use, second call returns final response
        ai_generator.client.messages.create.side_effect = [mock_tool_use_response, mock_final_response]
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "What is Python?", 
            tools=tools, 
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final AI response with tool results"
        
        # Verify tool was executed
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="test query"
        )
        
        # Verify two API calls were made
        assert ai_generator.client.messages.create.call_count == 2
    
    def test_handle_tool_execution(self, ai_generator, mock_tool_use_response, 
                                 mock_final_response, mock_tool_manager):
        # Mock the base parameters
        base_params = {
            "model": "claude-3-sonnet-20240229",
            "temperature": 0,
            "max_tokens": 800,
            "messages": [{"role": "user", "content": "What is Python?"}],
            "system": "System prompt content"
        }
        
        ai_generator.client.messages.create.return_value = mock_final_response
        
        result = ai_generator._handle_tool_execution(
            mock_tool_use_response, 
            base_params, 
            mock_tool_manager
        )
        
        assert result == "Final AI response with tool results"
        
        # Verify tool execution
        mock_tool_manager.execute_tool.assert_called_once_with(
            "search_course_content", 
            query="test query"
        )
        
        # Verify final API call structure
        call_args = ai_generator.client.messages.create.call_args
        messages = call_args[1]["messages"]
        
        # Should have original user message, assistant tool use, and tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"
        assert messages[1]["role"] == "assistant"
        assert messages[2]["role"] == "user"
        
        # Check tool results format
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Tool execution result"
    
    def test_system_prompt_content(self):
        # Test that the system prompt contains expected instructions
        system_prompt = AIGenerator.SYSTEM_PROMPT
        
        assert "course materials" in system_prompt.lower()
        assert "tool usage" in system_prompt.lower()
        assert "content search tool" in system_prompt.lower()
        assert "course outline tool" in system_prompt.lower()
        assert "up to 2 tool usage rounds" in system_prompt.lower()
        assert "one tool use per round maximum" in system_prompt.lower()
        assert "brief, concise and focused" in system_prompt.lower()
    
    def test_generate_response_no_tool_manager_with_tool_use(self, ai_generator, mock_tool_use_response):
        # If tool_manager is None but response requires tool use, should handle gracefully
        ai_generator.client.messages.create.return_value = mock_tool_use_response
        
        tools = [{"name": "test_tool", "description": "A test tool"}]
        
        # This should not raise an exception, but behavior depends on implementation
        # Since _handle_tool_execution is only called when tool_manager exists,
        # this would likely return the raw response content
        result = ai_generator.generate_response("What is Python?", tools=tools)
        
        # The implementation would need to handle this case appropriately
        # This test ensures the method doesn't crash
        assert result is not None
    
    def test_sequential_tool_calling_two_rounds(self, ai_generator, mock_tool_manager):
        # Mock responses for 2-round tool calling
        first_tool_response = Mock()
        first_tool_block = Mock()
        first_tool_block.type = "tool_use"
        first_tool_block.name = "search_course_content"
        first_tool_block.id = "tool_123"
        first_tool_block.input = {"query": "first search"}
        first_tool_response.content = [first_tool_block]
        first_tool_response.stop_reason = "tool_use"
        
        second_tool_response = Mock()
        second_tool_block = Mock()
        second_tool_block.type = "tool_use"
        second_tool_block.name = "search_course_outline"
        second_tool_block.id = "tool_456"
        second_tool_block.input = {"course_name": "follow-up search"}
        second_tool_response.content = [second_tool_block]
        second_tool_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final synthesized response"
        final_response.stop_reason = "end_turn"
        
        # Configure side effects: first round tool use, second round tool use, final response
        ai_generator.client.messages.create.side_effect = [
            first_tool_response,
            second_tool_response, 
            final_response
        ]
        
        # Configure tool manager to return different results for each call
        mock_tool_manager.execute_tool.side_effect = [
            "First tool result",
            "Second tool result"
        ]
        
        tools = [
            {"name": "search_course_content", "description": "Search content"},
            {"name": "search_course_outline", "description": "Search outline"}
        ]
        
        result = ai_generator.generate_response(
            "Complex query requiring two searches", 
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final synthesized response"
        
        # Verify two tool executions occurred
        assert mock_tool_manager.execute_tool.call_count == 2
        mock_tool_manager.execute_tool.assert_any_call("search_course_content", query="first search")
        mock_tool_manager.execute_tool.assert_any_call("search_course_outline", course_name="follow-up search")
        
        # Verify three API calls were made (2 rounds + final)
        assert ai_generator.client.messages.create.call_count == 3
    
    def test_early_termination_after_first_round(self, ai_generator, mock_tool_manager):
        # First call uses tool, second call provides final answer without tool use
        first_tool_response = Mock()
        first_tool_block = Mock()
        first_tool_block.type = "tool_use"
        first_tool_block.name = "search_course_content"
        first_tool_block.id = "tool_123"
        first_tool_block.input = {"query": "test query"}
        first_tool_response.content = [first_tool_block]
        first_tool_response.stop_reason = "tool_use"
        
        second_response = Mock()
        second_response.content = [Mock()]
        second_response.content[0].text = "Final answer after first round"
        second_response.stop_reason = "end_turn"
        
        ai_generator.client.messages.create.side_effect = [first_tool_response, second_response]
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Query that only needs one tool call", 
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Final answer after first round"
        
        # Verify only one tool execution
        assert mock_tool_manager.execute_tool.call_count == 1
        
        # Verify two API calls (first round + final answer)
        assert ai_generator.client.messages.create.call_count == 2
    
    def test_max_rounds_exceeded(self, ai_generator, mock_tool_manager):
        # Both rounds use tools, then force final response
        tool_response = Mock()
        tool_block = Mock()
        tool_block.type = "tool_use"
        tool_block.name = "search_course_content"
        tool_block.id = "tool_123"
        tool_block.input = {"query": "test query"}
        tool_response.content = [tool_block]
        tool_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Final response after max rounds"
        final_response.stop_reason = "end_turn"
        
        # First two calls return tool use, third is final response
        ai_generator.client.messages.create.side_effect = [
            tool_response,
            tool_response,
            final_response
        ]
        
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Query that wants to use tools repeatedly",
            tools=tools,
            tool_manager=mock_tool_manager,
            max_rounds=2
        )
        
        assert result == "Final response after max rounds"
        
        # Verify two tool executions (max rounds)
        assert mock_tool_manager.execute_tool.call_count == 2
        
        # Verify three API calls (2 rounds + final)
        assert ai_generator.client.messages.create.call_count == 3
    
    def test_conversation_state_maintenance(self, ai_generator, mock_tool_manager):
        # Test that conversation state is properly maintained across rounds
        first_tool_response = Mock()
        first_tool_block = Mock()
        first_tool_block.type = "tool_use"
        first_tool_block.name = "search_course_content"
        first_tool_block.id = "tool_123"
        first_tool_block.input = {"query": "first search"}
        first_tool_response.content = [first_tool_block]
        first_tool_response.stop_reason = "tool_use"
        
        final_response = Mock()
        final_response.content = [Mock()]
        final_response.content[0].text = "Response with context"
        final_response.stop_reason = "end_turn"
        
        ai_generator.client.messages.create.side_effect = [first_tool_response, final_response]
        mock_tool_manager.execute_tool.return_value = "Tool result"
        
        tools = [{"name": "search_course_content", "description": "Search tool"}]
        
        result = ai_generator.generate_response(
            "Test query",
            tools=tools,
            tool_manager=mock_tool_manager
        )
        
        assert result == "Response with context"
        
        # Verify that the second API call has accumulated message history
        second_call_args = ai_generator.client.messages.create.call_args_list[1]
        messages = second_call_args[1]["messages"]
        
        # Should have: original user message, assistant tool use, tool results
        assert len(messages) == 3
        assert messages[0]["role"] == "user"  # Original query
        assert messages[1]["role"] == "assistant"  # Tool use response
        assert messages[2]["role"] == "user"  # Tool results
        
        # Verify tool results format
        tool_results = messages[2]["content"]
        assert len(tool_results) == 1
        assert tool_results[0]["type"] == "tool_result"
        assert tool_results[0]["tool_use_id"] == "tool_123"
        assert tool_results[0]["content"] == "Tool result"
    
    def test_no_tools_direct_response(self, ai_generator, mock_anthropic_response):
        # Test that queries without tools work as before
        ai_generator.client.messages.create.return_value = mock_anthropic_response
        
        result = ai_generator.generate_response("What is Python?")
        
        assert result == "Test AI response"
        assert ai_generator.client.messages.create.call_count == 1