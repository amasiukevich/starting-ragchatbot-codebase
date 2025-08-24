import anthropic
from typing import List, Optional, Dict, Any

class AIGenerator:
    """Handles interactions with Anthropic's Claude API for generating responses"""
    
    # Static system prompt to avoid rebuilding on each call
    SYSTEM_PROMPT = """ You are an AI assistant specialized in course materials and educational content with access to comprehensive search and outline tools for course information.

Tool Usage Rules:
- **Up to 2 tool usage rounds per user query**
- **One tool use per round maximum** 
- **Content Search Tool**: Use for questions about specific course content or detailed educational materials
- **Course Outline Tool**: Use for questions about course structure, lesson lists, or course overviews
- Use tools when you need specific course information that you don't already know
- Chain tool calls logically: first search → analyze results → follow-up search if needed for additional information
- Synthesize all tool results into accurate, fact-based responses
- If tools yield no results, state this clearly without offering alternatives

Response Protocol:
- **General knowledge questions**: Answer using existing knowledge without using tools
- **Course-specific content questions**: Use content search tool first, then answer based on results
- **Course outline/structure questions**: Use course outline tool first, then answer based on results  
- **Complex queries**: Use first tool call for primary information, second tool call (if needed) for additional details or clarification
- **Provide final answer** when you have sufficient information to fully address the user's question
- **No meta-commentary**:
 - Provide direct answers only — no reasoning process, tool explanations, or question-type analysis
 - Do not mention "based on the search results" or "based on the tool results"

For outline queries, always include:
- Course title
- Course link (if available)
- Complete lesson list with numbers and titles

All responses must be:
1. **Brief, Concise and focused** - Get to the point quickly
2. **Educational** - Maintain instructional value
3. **Clear** - Use accessible language
4. **Example-supported** - Include relevant examples when they aid understanding
Provide only the direct answer to what was asked.
"""
    
    def __init__(self, api_key: str, model: str):
        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        
        # Pre-build base API parameters
        self.base_params = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": 800
        }
    
    def generate_response(self, query: str,
                         conversation_history: Optional[str] = None,
                         tools: Optional[List] = None,
                         tool_manager=None,
                         max_rounds: int = 2) -> str:
        """
        Generate AI response with optional tool usage and conversation context.
        Supports up to max_rounds of sequential tool calling.
        
        Args:
            query: The user's question or request
            conversation_history: Previous messages for context
            tools: Available tools the AI can use
            tool_manager: Manager to execute tools
            max_rounds: Maximum number of tool calling rounds (default 2)
            
        Returns:
            Generated response as string
        """
        
        # Build system content efficiently - avoid string ops when possible
        system_content = (
            f"{self.SYSTEM_PROMPT}\n\nPrevious conversation:\n{conversation_history}"
            if conversation_history 
            else self.SYSTEM_PROMPT
        )
        
        # Initialize conversation state
        messages = [{"role": "user", "content": query}]
        current_round = 1
        
        while current_round <= max_rounds:
            # Prepare API call parameters for current round
            api_params = {
                **self.base_params,
                "messages": messages.copy(),
                "system": system_content
            }
            
            # Add tools if available (keep tools available for all rounds)
            if tools:
                api_params["tools"] = tools
                api_params["tool_choice"] = {"type": "auto"}
            
            # Get response from Claude
            response = self.client.messages.create(**api_params)
            
            # Handle tool execution if needed
            if response.stop_reason == "tool_use":
                if not tool_manager:
                    # No tool manager available but tool use requested
                    # Check if we have actual text content (not tool blocks)
                    if (response.content and 
                        hasattr(response.content[0], 'text') and 
                        hasattr(response.content[0], 'type') and 
                        getattr(response.content[0], 'type', None) != 'tool_use'):
                        return response.content[0].text
                    else:
                        return "Tool requested but no tool manager available"
                
                try:
                    # Execute tools and get results
                    messages, _ = self._execute_tools_for_round(
                        response, messages, tool_manager
                    )
                    
                    # Check if this is the final round
                    if current_round >= max_rounds:
                        # Final round - make one more call without tools for final response
                        return self._make_final_response(messages, system_content)
                    
                    # Continue to next round
                    current_round += 1
                except Exception as e:
                    # Tool execution failed - handle gracefully
                    return f"Tool execution failed: {str(e)}. Unable to provide complete response."
            else:
                # No tool use - return response
                return response.content[0].text
        
        # Fallback: shouldn't reach here, but return last response
        return response.content[0].text
    
    def _execute_tools_for_round(self, response, messages: List[Dict], tool_manager):
        """
        Execute tool calls for current round and update conversation state.
        
        Args:
            response: API response containing tool use requests
            messages: Current conversation messages
            tool_manager: Manager to execute tools
            
        Returns:
            Tuple of (updated_messages, tool_results)
        """
        # Add AI's tool use response to conversation
        messages.append({"role": "assistant", "content": response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as user message for next round
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        return messages, tool_results
    
    def _make_final_response(self, messages: List[Dict], system_content: str) -> str:
        """
        Make final API call without tools to get synthesized response.
        
        Args:
            messages: Current conversation messages
            system_content: System prompt content
            
        Returns:
            Final response text
        """
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": system_content
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text
    
    def _handle_tool_execution(self, initial_response, base_params: Dict[str, Any], tool_manager):
        """
        Handle execution of tool calls and get follow-up response.
        
        Args:
            initial_response: The response containing tool use requests
            base_params: Base API parameters
            tool_manager: Manager to execute tools
            
        Returns:
            Final response text after tool execution
        """
        # Start with existing messages
        messages = base_params["messages"].copy()
        
        # Add AI's tool use response
        messages.append({"role": "assistant", "content": initial_response.content})
        
        # Execute all tool calls and collect results
        tool_results = []
        for content_block in initial_response.content:
            if content_block.type == "tool_use":
                tool_result = tool_manager.execute_tool(
                    content_block.name, 
                    **content_block.input
                )
                
                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": content_block.id,
                    "content": tool_result
                })
        
        # Add tool results as single message
        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        
        # Prepare final API call without tools
        final_params = {
            **self.base_params,
            "messages": messages,
            "system": base_params["system"]
        }
        
        # Get final response
        final_response = self.client.messages.create(**final_params)
        return final_response.content[0].text