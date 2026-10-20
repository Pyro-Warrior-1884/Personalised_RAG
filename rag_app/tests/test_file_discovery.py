"""Tests for the file discovery rules."""

from __future__ import annotations

import tempfile
from pathlib import Path

from django.test import SimpleTestCase, override_settings

from rag_app.services.file_discovery import (
    discover_supported_files,
    is_ignored,
    is_supported_file,
)


class FileDiscoveryTests(SimpleTestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name).resolve()
        self.addCleanup(self._tmp.cleanup)

        (self.root / "notes.md").write_text("# Notes", encoding="utf-8")
        (self.root / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")
        (self.root / "script.py").write_text("print('hi')", encoding="utf-8")
        (self.root / "archive.zip").write_bytes(b"PK\x03\x04")

        (self.root / "node_modules" / "pkg").mkdir(parents=True)
        (self.root / "node_modules" / "pkg" / "index.js").write_text("x", encoding="utf-8")

        (self.root / ".git").mkdir()
        (self.root / ".git" / "config").write_text("x", encoding="utf-8")

        (self.root / ".hidden").mkdir()
        (self.root / ".hidden" / "secret.md").write_text("x", encoding="utf-8")

        (self.root / ".env").write_text("SECRET=1", encoding="utf-8")
        (self.root / "server.pem").write_text("-----BEGIN PRIVATE KEY-----", encoding="utf-8")
        (self.root / "credentials.json").write_text("{}", encoding="utf-8")

    def test_finds_supported_files_only(self):
        found = discover_supported_files(self.root)
        self.assertIn("notes.md", found)
        self.assertIn("data.csv", found)
        self.assertIn("script.py", found)
        self.assertNotIn("archive.zip", found)

    def test_skips_ignored_and_hidden_directories(self):
        found = discover_supported_files(self.root)
        self.assertNotIn("node_modules/pkg/index.js", found)
        self.assertNotIn(".git/config", found)
        self.assertNotIn(".hidden/secret.md", found)

    def test_skips_secrets_and_environment_files(self):
        found = discover_supported_files(self.root)
        self.assertNotIn(".env", found)
        self.assertNotIn("server.pem", found)
        self.assertNotIn("credentials.json", found)

    def test_returns_sorted_relative_posix_paths(self):
        found = discover_supported_files(self.root)
        self.assertEqual(found, sorted(found))
        for path in found:
            self.assertNotIn("\\", path)
            self.assertFalse(Path(path).is_absolute())

    def test_does_not_follow_symlinked_directories(self):
        outside = self.root.parent / "outside-target"
        outside.mkdir(exist_ok=True)
        (outside / "leak.md").write_text("secret", encoding="utf-8")
        try:
            (self.root / "linked").symlink_to(outside, target_is_directory=True)
        except (OSError, NotImplementedError):  # pragma: no cover - platform dependent
            self.skipTest("symlinks are not supported here")
        found = discover_supported_files(self.root)
        self.assertNotIn("linked/leak.md", found)

    def test_respects_the_maximum_file_count(self):
        with override_settings(MAX_SOURCE_FILES=1):
            found = discover_supported_files(self.root)
        self.assertLessEqual(len(found), 1)

    def test_respects_the_maximum_file_size(self):
        (self.root / "big.md").write_text("x" * 5000, encoding="utf-8")
        with override_settings(MAX_FILE_SIZE_BYTES=1000):
            found = discover_supported_files(self.root)
        self.assertNotIn("big.md", found)

    def test_is_ignored_flags_hidden_and_sensitive_names(self):
        self.assertTrue(is_ignored(Path("/tmp/.env")))
        self.assertTrue(is_ignored(Path("/tmp/node_modules")))
        self.assertTrue(is_ignored(Path("/tmp/key.pem")))
        self.assertFalse(is_ignored(Path("/tmp/notes.md")))

    def test_is_supported_file_uses_the_extension_allowlist(self):
        self.assertTrue(is_supported_file(Path("a/b/file.md")))
        self.assertFalse(is_supported_file(Path("a/b/file.exe")))
