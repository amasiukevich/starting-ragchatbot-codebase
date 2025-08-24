Refactor @backend/ai_generator to support sequential tool calling where Claude can make up to 2 tool calls in separate AI rounds.

Current behavior:
- Claude makes 1 tool call -> tools are removed from the API params -> final response
- If Claude wants another tool call after seeing results - it can't (gets empty response)


Desired Behavior:
- Each tool call should be a separate API request where Claude can reason about previous results
- Support for complex queries requiring multiple searches for comparisons, multi-part questions, or when informations from different courses/lessons is needed


Example flow:
- User: "Search for the course that discusses the same topic as lesson 4 of course X"
- Claude: "Get the search outline for Course X -> gets the title of lesson 4"
- Claude: Uses the title to search for a course that discusses the same topic -> returns course information
- Claude: Provides complete answer


Requirements:
- Max 2 seaquential rounds per user query
- Terminate when: (a) Rounds completed, (b) Claude response doesn't have any tool uses, (c) tool call fails
- Preserve the conversation context between rounds
- Handle tool execution errors gracefully

Notes:
- update the system prompt in @backend/ai_generator.py
- update the test in @backend/tests/test_ai_generator.py
- Write tests to verify the external behavior (API calls made, tools executed, results returned) rather than internal state details


Use two parallel subagents to brainstorm possible plans. Do not implement any code yet.