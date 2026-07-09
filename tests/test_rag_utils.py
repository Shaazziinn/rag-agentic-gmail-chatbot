import io
import tempfile
import unittest
from pathlib import Path

from rag_utils import (
    describe_rag_context,
    documents_from_file_paths,
    documents_from_uploaded_files,
    is_context_overview_question,
)


class UploadedFileStub:
    def __init__(self, name, content):
        self.name = name
        self._buffer = io.BytesIO(content)

    def getvalue(self):
        return self._buffer.getvalue()


class RagUtilsTest(unittest.TestCase):
    def test_documents_from_file_paths_keeps_source_metadata(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "notes.md"
            path.write_text("# Notes\nRefunds take 5 days.", encoding="utf-8")

            documents = documents_from_file_paths([str(path)])

        self.assertEqual(len(documents), 1)
        self.assertEqual(documents[0].page_content, "# Notes\nRefunds take 5 days.")
        self.assertTrue(documents[0].metadata["source"].endswith("notes.md"))

    def test_documents_from_uploaded_files_decodes_text_and_tracks_source(self):
        uploaded_file = UploadedFileStub(
            "GUIDE.md",
            b"# Guide\nUse the local project instructions.",
        )

        documents = documents_from_uploaded_files([uploaded_file])

        self.assertEqual(len(documents), 1)
        self.assertEqual(
            documents[0].page_content,
            "# Guide\nUse the local project instructions.",
        )
        self.assertEqual(documents[0].metadata["source"], "GUIDE.md")

    def test_is_context_overview_question_detects_reviewer_style_questions(self):
        self.assertTrue(is_context_overview_question("What is this?"))
        self.assertTrue(is_context_overview_question("What context is attached?"))
        self.assertTrue(is_context_overview_question("Which files are uploaded?"))
        self.assertFalse(is_context_overview_question("What is the refund policy?"))

    def test_describe_rag_context_lists_sources_and_purpose(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "sample.txt"
            path.write_text("Refunds are allowed within 30 days.", encoding="utf-8")
            documents = documents_from_file_paths([str(path)])

        description = describe_rag_context(documents)

        self.assertIn("RAG chatbot", description)
        self.assertIn("sample.txt", description)
        self.assertIn("retrieves relevant chunks", description)


if __name__ == "__main__":
    unittest.main()
