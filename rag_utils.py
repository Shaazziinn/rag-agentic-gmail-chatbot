from langchain_core.documents import Document


SUPPORTED_FILE_TYPES = ["txt", "md"]


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
