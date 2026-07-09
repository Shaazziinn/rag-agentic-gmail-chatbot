from langchain_core.documents import Document


SUPPORTED_FILE_TYPES = ["txt", "md"]
CONTEXT_OVERVIEW_PATTERNS = [
    "what is this",
    "what this is",
    "what does this do",
    "what can you do",
    "what context",
    "context attached",
    "attached context",
    "which files",
    "what files",
    "uploaded files",
    "indexed files",
    "source files",
]


def documents_from_file_paths(file_paths):
    documents = []

    for file_path in file_paths:
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()

        documents.append(
            Document(
                page_content=text,
                metadata={"source": file_path},
            )
        )

    return documents


def documents_from_uploaded_files(uploaded_files):
    if not uploaded_files:
        return []

    documents = []

    for uploaded_file in uploaded_files:
        text = uploaded_file.getvalue().decode("utf-8")
        documents.append(
            Document(
                page_content=text,
                metadata={"source": uploaded_file.name},
            )
        )

    return documents


def is_context_overview_question(question):
    normalized_question = question.strip().lower()
    return any(
        pattern in normalized_question
        for pattern in CONTEXT_OVERVIEW_PATTERNS
    )


def describe_rag_context(documents):
    sources = sorted(
        {
            document.metadata.get("source", "Unknown source")
            for document in documents
        }
    )

    source_lines = "\n".join(f"- {source}" for source in sources)

    return f"""This is a RAG chatbot. RAG means Retrieval-Augmented Generation.

It answers questions by using the knowledge files currently attached to this session. It splits those files into smaller chunks, creates embeddings, retrieves relevant chunks from Chroma, and sends that context to the Groq language model.

Current context sources:
{source_lines}

You can ask questions about the content in these files. If an answer is not present in the attached context, the chatbot should say that it does not know based on the provided document."""
