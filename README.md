# RAG + Agentic Gmail Chatbot

Interview demo project that shows two AI chatbot patterns:

- **RAG Chatbot**: answers questions from indexed documents using retrieval-augmented generation.
- **Agentic Gmail Chatbot**: chats normally, searches Gmail when needed, drafts emails, and sends only after human approval.

## Features

- Streamlit chat UI with two modes.
- Document loading for `.txt` and `.md` files.
- Chroma vector retrieval with Hugging Face embeddings.
- Groq LLM responses through LangChain.
- Real Gmail API OAuth for one local Gmail account.
- Gmail search/read support.
- Approval-gated email sending.
- Tests for RAG helpers, Gmail helpers, and agent routing.

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```bash
GROQ_API_KEY=your_groq_api_key
```

On Streamlit Cloud, add the same value in **Manage app → Settings → Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

## Gmail API Setup

1. Create a Google Cloud project.
2. Enable the Gmail API.
3. Configure OAuth consent screen.
4. Create an OAuth Client ID with application type **Desktop app**.
5. Download the OAuth file and rename it to `credentials.json`.
6. Place `credentials.json` in the project root.
7. Run:

```bash
python gmail_auth_check.py
```

After browser approval, Google creates `token.json` locally.

Do not commit `.env`, `credentials.json`, or `token.json`.

## Run

```bash
streamlit run app.py
```

Open:

```txt
http://localhost:8501
```

## Demo Prompts

RAG mode:

```txt
What is the refund policy?
What can you do?
Explain RAG.
```

Agentic Gmail mode:

```txt
What is agentic AI?
Summarize my latest interview emails.
Draft a polite reply to my latest interview email saying thank you.
In the body, add that I am available tomorrow.
```

The app sends email only after the user clicks **Send Email** in the approval panel.

## Tests

```bash
python -m unittest discover -s tests
```

## Key Concepts

- **RAG**: retrieves relevant document chunks and gives them to the LLM as context.
- **Vector database**: stores embeddings so semantically similar text can be found.
- **Agentic AI**: uses tools, such as Gmail search and send, to complete tasks.
- **Human-in-the-loop approval**: the LLM can draft an action, but the user approves before the real side effect happens.
