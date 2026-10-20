"""Tests for the source manager: path validation and repository URL checks."""

from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import SimpleTestCase, override_settings

from rag_app.services.source_manager import (
    SourceValidationError,
    validate_github_url,
    validate_local_directory,
)


class LocalDirectoryValidationTests(SimpleTestCase):
    """A local path must exist, be a directory and live inside an approved root."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.nested = self.root / "docs" / "notes"
        self.nested.mkdir(parents=True)
        self.file = self.root / "notes.txt"
        self.file.write_text("hello", encoding="utf-8")
        self.settings = override_settings(ALLOWED_SOURCE_ROOTS=[self.root])
        self.settings.enable()
        self.addCleanup(self.settings.disable)
        self.addCleanup(self._tmp.cleanup)

    def test_accepts_a_directory_inside_an_approved_root(self):
        resolved = validate_local_directory(str(self.nested))
        self.assertEqual(resolved, self.nested)

    def test_accepts_the_approved_root_itself(self):
        self.assertEqual(validate_local_directory(str(self.root)), self.root)

    def test_rejects_a_missing_path(self):
        with self.assertRaises(SourceValidationError) as ctx:
            validate_local_directory(str(self.root / "does-not-exist"))
        self.assertIn("does not exist", str(ctx.exception).lower())

    def test_rejects_a_file(self):
        with self.assertRaises(SourceValidationError) as ctx:
            validate_local_directory(str(self.file))
        self.assertIn("not a directory", str(ctx.exception).lower())

    def test_rejects_an_empty_path(self):
        with self.assertRaises(SourceValidationError):
            validate_local_directory("   ")

    def test_rejects_path_traversal_outside_the_root(self):
        outside = self.root.parent / "elsewhere"
        outside.mkdir(exist_ok=True)
        with self.assertRaises(SourceValidationError) as ctx:
            validate_local_directory(str(self.root / ".." / "elsewhere"))
        self.assertIn("outside the approved roots", str(ctx.exception))

    def test_rejects_a_path_in_a_different_root(self):
        other = tempfile.TemporaryDirectory()
        self.addCleanup(other.cleanup)
        with self.assertRaises(SourceValidationError):
            validate_local_directory(str(Path(other.name)))

    def test_rejects_everything_when_no_root_is_configured(self):
        with override_settings(ALLOWED_SOURCE_ROOTS=[]):
            with self.assertRaises(SourceValidationError) as ctx:
                validate_local_directory(str(self.nested))
        self.assertIn("approved source roots", str(ctx.exception))

    def test_expands_a_home_relative_path(self):
        with override_settings(ALLOWED_SOURCE_ROOTS=[Path.home()]):
            resolved = validate_local_directory("~")
            self.assertEqual(resolved, Path.home().resolve())


class GithubUrlValidationTests(SimpleTestCase):
    """Only public GitHub repository URLs are accepted."""

    def test_accepts_a_repository_url(self):
        self.assertEqual(
            validate_github_url("https://github.com/krishnaik06/RAG-Tutorials"),
            "https://github.com/krishnaik06/RAG-Tutorials.git",
        )

    def test_accepts_a_dot_git_url(self):
        self.assertEqual(
            validate_github_url("https://github.com/owner/repo.git"),
            "https://github.com/owner/repo.git",
        )

    def test_rejects_http(self):
        with self.assertRaises(SourceValidationError):
            validate_github_url("http://github.com/owner/repo")

    def test_rejects_other_hosts(self):
        for url in (
            "https://gitlab.com/owner/repo",
            "https://raw.githubusercontent.com/owner/repo",
            "https://example.com/owner/repo",
        ):
            with self.assertRaises(SourceValidationError):
                validate_github_url(url)

    def test_rejects_localhost_and_private_endpoints(self):
        for url in (
            "https://localhost/owner/repo",
            "https://127.0.0.1/owner/repo",
            "https://192.168.1.10/owner/repo",
        ):
            with self.assertRaises(SourceValidationError):
                validate_github_url(url)

    def test_rejects_credentials_in_the_url(self):
        with self.assertRaises(SourceValidationError):
            validate_github_url("https://user:pass@github.com/owner/repo")

    def test_rejects_non_standard_ports(self):
        with self.assertRaises(SourceValidationError):
            validate_github_url("https://github.com:8443/owner/repo")

    def test_rejects_malformed_paths(self):
        for url in (
            "https://github.com/",
            "https://github.com/owner",
            "https://github.com/owner/repo/tree/main",
            "https://github.com/owner/repo/issues",
        ):
            with self.assertRaises(SourceValidationError):
                validate_github_url(url)
