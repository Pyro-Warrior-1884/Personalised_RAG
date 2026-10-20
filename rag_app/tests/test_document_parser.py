"""Tests for the format specific document parsers."""

from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import SimpleTestCase

from rag_app.services.document_parser import load_file

from .fakes import make_docx, make_html, make_pdf, make_xlsx


class DocumentParserTests(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)

    def _write(self, name: str, content: str = "") -> Path:
        path = self.root / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_parses_plain_text(self):
        path = self._write("notes.txt", "The sky is blue.")
        result = load_file(path, "notes.txt")
        self.assertTrue(result.ok)
        self.assertEqual(result.file_type, "text")
        self.assertIn("sky is blue", result.documents[0].page_content)
        self.assertEqual(result.documents[0].metadata["relative_path"], "notes.txt")

    def test_parses_markdown_and_records_headings(self):
        path = self._write("readme.md", "# Title\n\nSome text\n")
        result = load_file(path, "readme.md")
        self.assertTrue(result.ok)
        self.assertEqual(result.file_type, "markdown")
        self.assertIn("Title", result.documents[0].metadata.get("headings", ""))

    def test_parses_source_code_and_records_the_language(self):
        path = self._write("app.py", "def main():\n    return 1\n")
        result = load_file(path, "app.py")
        self.assertTrue(result.ok)
        self.assertEqual(result.file_type, "code")
        self.assertEqual(result.documents[0].metadata["language"], "py")

    def test_parses_csv_with_header_and_row_context(self):
        path = self._write("people.csv", "name,age\nAda,36\nAlan,41\n")
        result = load_file(path, "people.csv")
        self.assertTrue(result.ok)
        content = result.documents[0].page_content
        self.assertIn("name | age", content)
        self.assertIn("Row 2: Ada | 36", content)
        self.assertEqual(result.documents[0].metadata["row_end"], 3)

    def test_parses_html_and_strips_scripts(self):
        path = make_html(self.root / "page.html", "Example title", "Real body text.")
        result = load_file(path, "page.html")
        self.assertTrue(result.ok)
        content = result.documents[0].page_content
        self.assertIn("Real body text.", content)
        self.assertNotIn("alert(", content)
        self.assertEqual(result.documents[0].metadata["page_title"], "Example title")

    def test_parses_docx_with_paragraphs(self):
        path = make_docx(self.root / "report.docx", ["First paragraph", "Second paragraph"])
        result = load_file(path, "report.docx")
        self.assertTrue(result.ok)
        self.assertIn("First paragraph", result.documents[0].page_content)
        self.assertEqual(result.file_type, "word")

    def test_parses_excel_with_sheet_and_row_metadata(self):
        path = make_xlsx(self.root / "book.xlsx", [["name", "age"], ["Ada", "36"]])
        result = load_file(path, "book.xlsx")
        self.assertTrue(result.ok)
        document = result.documents[0]
        self.assertEqual(document.metadata["sheet"], "Data")
        self.assertEqual(document.metadata["workbook"], "book.xlsx")
        self.assertIn("Ada", document.page_content)

    def test_parses_pdf_with_page_numbers(self):
        path = make_pdf(self.root / "manual.pdf", ["Page one content", "Page two content"])
        result = load_file(path, "manual.pdf")
        self.assertTrue(result.ok)
        self.assertEqual(len(result.documents), 2)
        self.assertEqual(result.documents[0].metadata["page_number"], 1)
        self.assertEqual(result.documents[1].metadata["page_number"], 2)
        self.assertIn("Page one content", result.documents[0].page_content)

    def test_reports_an_unsupported_file(self):
        path = self._write("archive.zip", "not really a zip")
        result = load_file(path, "archive.zip")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "unsupported")

    def test_reports_an_empty_file(self):
        path = self._write("empty.txt", "")
        result = load_file(path, "empty.txt")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "empty")

    def test_reports_a_corrupt_pdf_without_raising(self):
        path = self.root / "broken.pdf"
        path.write_bytes(b"%PDF-1.4 this is not a real pdf")
        result = load_file(path, "broken.pdf")
        self.assertFalse(result.ok)
        self.assertIn(result.reason, {"corrupt", "empty"})

    def test_reports_a_missing_file(self):
        result = load_file(self.root / "nope.txt", "nope.txt")
        self.assertFalse(result.ok)
        self.assertEqual(result.reason, "missing")
