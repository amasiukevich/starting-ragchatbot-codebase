# RAG Chatbot Query Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 FRONTEND                                        │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ User Types Query│
    │   "What is...?" │
    └─────────┬───────┘
              │ Enter key / Click send
              ▼
    ┌─────────────────┐
    │  sendMessage()  │
    │ - Disable UI    │
    │ - Show loading  │
    │ - Add user msg  │
    └─────────┬───────┘
              │ 
              ▼
    ┌─────────────────┐
    │  HTTP POST      │
    │  /api/query     │
    │  {              │
    │    query: "...", │
    │    session_id   │
    │  }              │
    └─────────┬───────┘
              │
              │ JSON over HTTP
              ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                                 BACKEND                                         │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ FastAPI         │
    │ @app.post       │
    │ /api/query      │
    └─────────┬───────┘
              │ Validate QueryRequest
              ▼
    ┌─────────────────┐
    │ Session Check   │
    │ if no session:  │
    │   create_session│
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ rag_system.     │
    │ query()         │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐         ┌─────────────────┐
    │ Get History     │◄────────┤ SessionManager  │
    │ from session    │         │ - In-memory     │
    └─────────┬───────┘         │ - 2 exchanges   │
              │                 │ - String format │
              ▼                 └─────────────────┘
    ┌─────────────────┐
    │ Wrap in Prompt  │
    │ "Answer this... │
    │  {user_query}"  │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ ai_generator.   │
    │ generate_       │
    │ response()      │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Build System    │
    │ Prompt +        │
    │ History         │
    └─────────┬───────┘
              │
              ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                             CLAUDE API CALL                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ Claude Sonnet 4 │
    │ + Tools         │
    │ + System Prompt │
    │ + History       │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐         ┌─────────────────┐
    │ Tool Decision   │────────►│ Search Tool?    │
    │ Use search?     │  YES    │ - Vector search │
    └─────────┬───────┘         │ - ChromaDB      │
              │ NO              │ - Embeddings    │
              ▼                 └─────────────────┘
    ┌─────────────────┐                │
    │ Direct Answer   │◄───────────────┘
    │ Generated       │ Tool results
    └─────────┬───────┘
              │
              ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                            RESPONSE ASSEMBLY                                    │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ Get Sources     │
    │ from ToolManager│
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Update Session  │
    │ add_exchange()  │
    │ (query+response)│
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Return JSON     │
    │ {               │
    │   answer: "..." │
    │   sources: []   │
    │   session_id    │
    │ }               │
    └─────────┬───────┘
              │
              │ HTTP Response
              ▼

┌─────────────────────────────────────────────────────────────────────────────────┐
│                           FRONTEND RESPONSE                                     │
└─────────────────────────────────────────────────────────────────────────────────┘

    ┌─────────────────┐
    │ Receive JSON    │
    │ response.json() │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Update Session  │
    │ currentSessionId│
    │ = data.session_id│
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Remove Loading  │
    │ Parse Markdown  │
    │ Display Answer  │
    └─────────┬───────┘
              │
              ▼
    ┌─────────────────┐
    │ Re-enable UI    │
    │ Ready for next  │
    │ query           │
    └─────────────────┘

```

## Key Components Flow:

**Data Transformations:**
- User text → JSON payload → Instruction prompt → Claude API → JSON response → HTML display

**Session Management:**
- First request: `null` → `"session_1"` created
- Subsequent: `"session_1"` → maintains 2-exchange history

**Tool Integration:**
- Claude decides autonomously whether to search
- Vector search via ChromaDB if needed
- Sources tracked by ToolManager

**Error Handling:**
- Network errors → Error message display
- API errors → HTTP 500 → Frontend error handling