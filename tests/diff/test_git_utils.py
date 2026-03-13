"""Tests for git_utils.py — mock subprocess for git operations."""

import unittest
from unittest.mock import patch, MagicMock
import subprocess

from cylint.diff.git_utils import get_base_source, get_changed_files


class TestGetBaseSource(unittest.TestCase):
    @patch("cylint.diff.git_utils.subprocess.run")
    def test_file_exists(self, mock_run):
        mock_run.return_value = MagicMock(stdout="print('hello')\n")
        result = get_base_source("origin/main", "src/foo.py")
        self.assertEqual(result, "print('hello')\n")
        mock_run.assert_called_once()

    @patch("cylint.diff.git_utils.subprocess.run")
    def test_new_file_returns_none(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            1, "git", stderr="fatal: path 'src/new.py' does not exist"
        )
        result = get_base_source("origin/main", "src/new.py")
        self.assertIsNone(result)

    @patch("cylint.diff.git_utils.subprocess.run")
    def test_other_error_raises(self, mock_run):
        mock_run.side_effect = subprocess.CalledProcessError(
            128, "git", stderr="fatal: not a git repository"
        )
        with self.assertRaises(subprocess.CalledProcessError):
            get_base_source("origin/main", "src/foo.py")


class TestGetChangedFiles(unittest.TestCase):
    @patch("cylint.diff.git_utils.subprocess.run")
    def test_modified_files(self, mock_run):
        mock_run.return_value = MagicMock(stdout="M\tsrc/foo.py\nM\tsrc/bar.py\n")
        files = get_changed_files("origin/main")
        self.assertEqual(len(files), 2)
        self.assertEqual(files[0].status, "M")
        self.assertEqual(files[0].path, "src/foo.py")

    @patch("cylint.diff.git_utils.subprocess.run")
    def test_added_file(self, mock_run):
        mock_run.return_value = MagicMock(stdout="A\tsrc/new.py\n")
        files = get_changed_files("origin/main")
        self.assertEqual(files[0].status, "A")
        self.assertIsNone(files[0].old_path)

    @patch("cylint.diff.git_utils.subprocess.run")
    def test_deleted_file(self, mock_run):
        mock_run.return_value = MagicMock(stdout="D\tsrc/old.py\n")
        files = get_changed_files("origin/main")
        self.assertEqual(files[0].status, "D")

    @patch("cylint.diff.git_utils.subprocess.run")
    def test_renamed_file(self, mock_run):
        mock_run.return_value = MagicMock(stdout="R100\tsrc/old.py\tsrc/new.py\n")
        files = get_changed_files("origin/main")
        self.assertEqual(files[0].status, "R")
        self.assertEqual(files[0].path, "src/new.py")
        self.assertEqual(files[0].old_path, "src/old.py")

    @patch("cylint.diff.git_utils.subprocess.run")
    def test_empty_output(self, mock_run):
        mock_run.return_value = MagicMock(stdout="")
        files = get_changed_files("origin/main")
        self.assertEqual(len(files), 0)


if __name__ == "__main__":
    unittest.main()
