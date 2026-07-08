# Agentic Gmail Chatbot Design

## Purpose

Build an interview-ready chatbot that demonstrates both retrieval-augmented generation and agentic AI. The existing Streamlit RAG chatbot remains available for document-grounded question answering. A new Agent mode adds real Gmail API actions so the assistant can search/read email context, draft messages, and send only after explicit human approval.

## Goals

- Keep the current RAG flow: load local and uploaded documents, chunk them, embed them, retrieve relevant context, and answer with visible source context.
- Add a real Gmail-backed agent flow for one local Gmail account.
- Let the assistant search/read Gmail threads when needed.
- Let the assistant propose email drafts with recipient, subject, body, and rationale.
- Require a visible approval button before any email is sent.
- Document the core concepts so the user can explain RAG, agents, tools, approval gates, and Gmail OAuth during an interview review.

## Non-Goals

- Multi-user OAuth.
- Background jobs or persistent approval databases.
- Gmail labels, archiving, deleting, attachments, or calendar actions.
- Fully autonomous sending.
- Production deployment hardening.

## User Experience

The app has two clear modes:

1. RAG Chatbot: the user asks questions about loaded documents and sees the answer plus source context.
2. Agentic Gmail Chatbot: the user asks for email-related help, such as "Find the latest email from Alex about invoices and draft a reply." The app can search Gmail, inspect thread snippets, and produce a proposed draft.

When the agent proposes an email, Streamlit stores it as a pending action and renders an approval panel. The panel shows:

- To
- Subject
- Body
- Whether it is a new email or reply
- Context or thread information used
- A Send button
- A Cancel button

The email is sent only when the user clicks Send. The LLM never receives a direct "send now" capability.

## Architecture

### Streamlit UI

`app.py` remains the entry point. It owns the chat mode selection, message display, file upload, approval panel, and session state. It delegates document loading to `rag_utils.py`, Gmail operations to a new Gmail module, and agent orchestration to a new agent module.

### RAG Utilities

`rag_utils.py` continues to handle supported document types and conversion into LangChain `Document` objects. It should stay small and testable.

### Gmail Client

A new `gmail_utils.py` module handles Google OAuth and Gmail API calls. It is responsible for:

- Loading `credentials.json`.
- Creating or refreshing `token.json`.
- Building the Gmail API service.
- Searching messages with Gmail query syntax.
- Reading message/thread details.
- Creating and sending MIME email messages.

OAuth files stay local and are not committed.

### Agent Orchestration

A new `agent_utils.py` module converts a user request into an agent result. For v1, it can use a conservative structured flow rather than unrestricted tool execution:

- Interpret whether the request needs Gmail search.
- Search/read relevant Gmail context when needed.
- Ask the LLM to generate a structured draft proposal.
- Validate that required fields are present.
- Return either a normal assistant answer or a pending email action.

This keeps the demo agentic because it uses tools and chooses actions, while keeping the send boundary deterministic and safe.

### Documentation

Add documentation for:

- Setup steps for Google Cloud OAuth credentials.
- Required environment variables.
- How to run the app.
- Interview concepts: RAG, embeddings, vector database, retrieval, agents, tools, tool approval, OAuth scopes, and human-in-the-loop safety.

## Data Flow

### RAG Chat Flow

1. User uploads additional documents when needed.
2. App loads default and uploaded documents.
3. Text splitter creates chunks.
4. Chroma stores chunk embeddings.
5. Retriever returns top relevant chunks for the user question.
6. LLM answers using only retrieved context.
7. App displays answer and source context.

### Agentic Gmail Flow

1. User enters an email-related instruction in Agent mode.
2. Agent decides whether Gmail search/read is needed.
3. Gmail client authenticates the local user if needed.
4. Gmail client searches and reads matching messages or threads.
5. LLM drafts a structured email proposal using the user request and email context.
6. App stores the proposal as `pending_email_action`.
7. App renders the approval panel.
8. User clicks Send.
9. Gmail client sends the email.
10. App records the result in session history.

## Gmail Permissions

The app needs Gmail API access for reading/searching email and sending email. It requests only the scopes required for v1:

- `https://www.googleapis.com/auth/gmail.readonly` for Gmail search/read.
- `https://www.googleapis.com/auth/gmail.send` for approved sends.

V1 does not create Gmail-hosted drafts and does not request Gmail draft scope. Streamlit holds the pending draft locally until approval.

## Safety Rules

- The LLM cannot directly call the send function.
- Sending requires a user click in the Streamlit UI.
- The approval panel must show the exact email fields before sending.
- Missing recipient, subject, or body blocks sending.
- Gmail credentials and tokens are local files and must be ignored by git.
- Errors from Gmail should be shown without exposing secrets or raw tokens.

## Error Handling

- Missing `GROQ_API_KEY`: show a setup error before LLM calls.
- Missing Gmail credentials: show setup instructions.
- Expired OAuth token: refresh automatically when possible.
- Gmail API failure: show a concise error and keep the pending draft unsent.
- LLM returns invalid draft JSON: show a recoverable error and ask the user to refine the request.
- No Gmail search results: explain that no matching thread was found and offer to draft a new email from the available request.

## Testing

Add unit tests for pure helpers:

- Gmail MIME message creation.
- Gmail result normalization.
- Agent draft validation.
- Existing document loading behavior.

Manual verification should cover:

- RAG mode still answers from documents.
- Agent mode can complete Gmail OAuth.
- Agent mode can search/read a real Gmail thread.
- Agent mode can generate a pending draft.
- Send only happens after the approval button is clicked.

## Demo Script

1. Show RAG mode and ask a question about the sample refund policy.
2. Explain chunking, embeddings, Chroma, retrieval, and grounded generation.
3. Switch to Agent mode.
4. Ask the assistant to find a real email thread and draft a reply.
5. Show the Gmail search/read context.
6. Show the pending email approval panel.
7. Explain that this is agentic because the assistant uses tools and prepares actions.
8. Send only after approval, or cancel to demonstrate the safety gate.

## Implementation Boundaries

This design should be implemented in small steps:

1. Add Gmail dependencies and local credential ignores.
2. Build and test Gmail helper functions.
3. Build and test agent draft validation.
4. Integrate Agent mode into Streamlit.
5. Add README and concept documentation.
6. Run automated tests and a local Streamlit smoke test.
