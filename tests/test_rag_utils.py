import io
import tempfile
import unittest
from pathlib import Path

from rag_utils import documents_from_file_paths, documents_from_uploaded_files


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


if __name__ == "__main__":
    unittest.main()
