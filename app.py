import os
from dotenv import load_dotenv

import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import CharacterTextSplitter

from rag_utils import (
    SUPPORTED_FILE_TYPES,
    describe_rag_context,
    documents_from_file_paths,
    documents_from_uploaded_files,
    is_context_overview_question,
)
from agent_utils import revise_email_draft, run_agentic_chat
from config_utils import get_config_value
from gmail_utils import get_gmail_service, send_email

load_dotenv()

st.title("RAG + Agentic Gmail Chatbot")

mode = st.sidebar.radio(
    "Mode",
    ["RAG Chatbot", "Agentic Gmail Chatbot"],
)

def get_llm():
    api_key = get_config_value("GROQ_API_KEY", st.secrets)
    if not api_key:
        st.error(
            "Missing GROQ_API_KEY. Add it to Streamlit Cloud secrets or "
            "your local .env file."
        )
        st.stop()

    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=api_key,
    )


def load_documents(uploaded_files=None):
    file_paths = [
        "data/sample.txt",
    ]

    documents = documents_from_file_paths(file_paths)
    documents.extend(documents_from_uploaded_files(uploaded_files))

    return documents


@st.cache_resource(show_spinner="Loading embedding model (first run only)...")
def get_embeddings():
    # Loading the model + torch is the expensive step; cache it so it happens
    # once per app process instead of on every message.
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
    )


def create_vector_database(documents):
    embeddings = get_embeddings()

    splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(documents)

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    return vector_db


@st.cache_resource(show_spinner="Indexing knowledge base (first run only)...")
def get_base_vector_database():
    # The bundled sample never changes, so build its index once and reuse it.
    return create_vector_database(load_documents())


def show_rag_chatbot():
    uploaded_files = st.sidebar.file_uploader(
        "Add knowledge files",
        type=SUPPORTED_FILE_TYPES,
        accept_multiple_files=True,
    )

    if uploaded_files:
        st.sidebar.success(f"Using {len(uploaded_files)} uploaded file(s)")

    question = st.chat_input("Ask something about the document")

    if question:
        st.chat_message("user").write(question)

        documents = load_documents(uploaded_files)

        if is_context_overview_question(question):
            answer = describe_rag_context(documents)
            st.chat_message("assistant").write(answer)
            return

        llm = get_llm()
        if uploaded_files:
            # User added their own files: build a fresh index for this session.
            vector_db = create_vector_database(documents)
        else:
            # No uploads: reuse the cached index of the bundled sample.
            vector_db = get_base_vector_database()
        retriever = vector_db.as_retriever(search_kwargs={"k": 3})
        relevant_docs = retriever.invoke(question)

        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        sources = "\n".join(
            f"- {document.metadata.get('source', 'Unknown source')}"
            for document in documents
        )

        prompt = f"""
You are a helpful RAG chatbot.

You can respond like a normal chatbot for greetings, general conversation,
and questions about what you can do.

When the user asks about the uploaded or indexed documents, answer using only
the retrieved context below. If the document answer is not in the context, say:
"I don't know based on the provided document."

Available context sources:
{sources}

Context:
{context}

Question:
{question}
"""

        response = llm.invoke(prompt)

        st.chat_message("assistant").write(response.content)

        with st.expander("Source context"):
            st.write(context)


def _initialize_agent_state():
    if "agent_messages" not in st.session_state:
        st.session_state.agent_messages = []
    if "pending_email_action" not in st.session_state:
        st.session_state.pending_email_action = None


def _render_agent_history():
    for message in st.session_state.agent_messages:
        st.chat_message(message["role"]).write(message["content"])


def _render_pending_email_action():
    pending = st.session_state.pending_email_action
    if not pending:
        return

    draft = pending["draft"]

    st.subheader("Pending Email Approval")
    st.caption("Review the exact email below. It will only send if you approve.")

    draft_version = pending.get("version", 0)
    edited_to = st.text_input(
        "To",
        value=draft.get("to", ""),
        key=f"pending_to_{draft_version}",
    )
    edited_subject = st.text_input(
        "Subject",
        value=draft.get("subject", ""),
        key=f"pending_subject_{draft_version}",
    )
    edited_body = st.text_area(
        "Body",
        value=draft.get("body", ""),
        height=220,
        key=f"pending_body_{draft_version}",
    )

    with st.expander("Agent rationale and Gmail context"):
        st.write(draft.get("rationale", "No rationale provided."))
        st.write(f"Gmail query: `{pending.get('query', '')}`")
        st.text(pending.get("context", ""))

    send_col, cancel_col = st.columns(2)

    with send_col:
        if st.button("Send Email", type="primary"):
            if not edited_to.strip() or not edited_subject.strip() or not edited_body.strip():
                st.error("Recipient, subject, and body are required before sending.")
                return

            try:
                service = get_gmail_service()
                result = send_email(
                    service,
                    to=edited_to,
                    subject=edited_subject,
                    body=edited_body,
                )
            except Exception as exc:
                st.error(f"Could not send email: {exc}")
                return

            st.session_state.agent_messages.append(
                {
                    "role": "assistant",
                    "content": f"Email sent successfully. Gmail message ID: {result.get('id')}",
                }
            )
            st.session_state.pending_email_action = None
            st.rerun()

    with cancel_col:
        if st.button("Cancel Draft"):
            st.session_state.agent_messages.append(
                {
                    "role": "assistant",
                    "content": "Draft cancelled. No email was sent.",
                }
            )
            st.session_state.pending_email_action = None
            st.rerun()


def show_agentic_gmail_chatbot():
    _initialize_agent_state()

    st.info(
        "This mode uses the real Gmail API. The agent can search/read Gmail "
        "and draft an email, but sending requires your button click."
    )

    _render_agent_history()
    _render_pending_email_action()

    user_request = st.chat_input(
        "Ask the Gmail agent, e.g. 'Find my latest email about the meeting and draft a reply'"
    )

    if not user_request:
        return

    st.session_state.agent_messages.append(
        {"role": "user", "content": user_request}
    )
    st.chat_message("user").write(user_request)
    llm = get_llm()

    if st.session_state.pending_email_action:
        try:
            pending = st.session_state.pending_email_action
            revised_draft = revise_email_draft(
                pending["draft"],
                user_request,
                llm,
            )
        except Exception as exc:
            error_message = f"Draft edit error: {exc}"
            st.session_state.agent_messages.append(
                {"role": "assistant", "content": error_message}
            )
            st.chat_message("assistant").write(error_message)
            return

        pending["draft"] = revised_draft
        pending["version"] = pending.get("version", 0) + 1
        st.session_state.pending_email_action = pending

        assistant_message = (
            "I updated the pending email draft. "
            "Please review the approval panel before sending."
        )
        st.session_state.agent_messages.append(
            {"role": "assistant", "content": assistant_message}
        )
        st.chat_message("assistant").write(assistant_message)
        st.rerun()

    try:
        result = run_agentic_chat(user_request, llm)
    except Exception as exc:
        error_message = f"Agent error: {exc}"
        st.session_state.agent_messages.append(
            {"role": "assistant", "content": error_message}
        )
        st.chat_message("assistant").write(error_message)
        return

    if result["type"] == "chat":
        assistant_message = result["content"]

        st.session_state.agent_messages.append(
            {"role": "assistant", "content": assistant_message}
        )
        st.chat_message("assistant").write(assistant_message)
        return

    if result["type"] == "email_draft":
        assistant_message = (
            "I prepared an email draft. "
            "Please review the pending email approval panel before sending."
        )

        st.session_state.agent_messages.append(
            {"role": "assistant", "content": assistant_message}
        )
        result["version"] = 0
        st.session_state.pending_email_action = result

        st.chat_message("assistant").write(assistant_message)
        st.rerun()


if mode == "RAG Chatbot":
    show_rag_chatbot()
else:
    show_agentic_gmail_chatbot()
