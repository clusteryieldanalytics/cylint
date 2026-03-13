"""Tests for orchestrator.py — end-to-end CI flow."""

import json
import os
import tempfile
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

from cylint.ci.orchestrator import CIOrchestrator, CIResult


def _write_py(directory: str, name: str, content: str) -> str:
    """Write a .py file and return its path."""
    path = os.path.join(directory, name)
    with open(path, "w") as f:
        f.write(content)
    return path


class TestCIResultSerialization(unittest.TestCase):
    def test_to_json(self):
        result = CIResult()
        data = json.loads(result.to_json())
        self.assertIn("findings", data)
        self.assertIn("stats", data)
        self.assertIn("comment", data)

    def test_to_dict_keys(self):
        result = CIResult()
        d = result.to_dict()
        expected = {
            "findings", "changeClassifications", "enrichedFindings",
            "changeFindings", "planFindings", "comment", "stats",
        }
        self.assertEqual(set(d.keys()), expected)


class TestOrchestratorLintOnly(unittest.TestCase):
    """Test cy ci without enrichment (free tier)."""

    def test_lint_clean_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "clean.py", "x = 1\n")
            orch = CIOrchestrator(paths=[tmpdir])
            result = orch.run()
            self.assertEqual(result.stats["linterFindings"], 0)
            self.assertFalse(result.stats["enriched"])

    def test_lint_file_with_findings(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "bad.py", '''\
df = spark.table("orders")
df.collect()
''')
            orch = CIOrchestrator(paths=[tmpdir])
            result = orch.run()
            self.assertGreater(result.stats["linterFindings"], 0)
            self.assertGreater(len(result.findings), 0)
            self.assertFalse(result.stats["enriched"])

    def test_comment_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "bad.py", '''\
df = spark.table("orders")
df.collect()
''')
            orch = CIOrchestrator(paths=[tmpdir])
            result = orch.run()
            self.assertIsNotNone(result.comment)
            self.assertIn("Cluster Yield", result.comment.markdown)

    def test_annotations_generated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "bad.py", '''\
df = spark.table("orders")
df.collect()
''')
            orch = CIOrchestrator(paths=[tmpdir])
            result = orch.run()
            self.assertGreater(len(result.comment.annotations), 0)
            ann = result.comment.annotations[0]
            self.assertIn("file", ann)
            self.assertIn("line", ann)
            self.assertIn("level", ann)

    def test_no_findings_comment(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "clean.py", "x = 1\n")
            orch = CIOrchestrator(paths=[tmpdir])
            result = orch.run()
            self.assertIn("No findings", result.comment.markdown)

    def test_json_output_valid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "bad.py", '''\
df = spark.table("orders")
df.collect()
''')
            orch = CIOrchestrator(paths=[tmpdir])
            result = orch.run()
            data = json.loads(result.to_json())
            self.assertIn("findings", data)
            self.assertIn("comment", data)
            self.assertIn("stats", data)


class TestOrchestratorWithEnrichment(unittest.TestCase):
    """Test cy ci with a mock enrichment server."""

    def _start_server(self, handler_cls):
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        return server

    def test_enrichment_success(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)

                body = json.dumps({
                    "findings": [{"rule": "CY001", "file": "bad.py", "line": 2,
                                  "message": "collect", "savings": 500}],
                    "planFindings": [],
                    "changeFindings": [],
                    "matchStats": {"fingerprintMatchRate": 1.0},
                }).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        server = self._start_server(Handler)
        try:
            host, port = server.server_address
            with tempfile.TemporaryDirectory() as tmpdir:
                _write_py(tmpdir, "bad.py", '''\
df = spark.table("orders")
df.collect()
''')
                orch = CIOrchestrator(
                    paths=[tmpdir],
                    api_key="test-key",
                    environment="prod",
                    base_url=f"http://{host}:{port}",
                )
                result = orch.run()
                self.assertTrue(result.stats["enriched"])
                self.assertEqual(len(result.enriched_findings), 1)
        finally:
            server.shutdown()

    def test_enrichment_failure_degrades_gracefully(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(500)
                self.end_headers()
                self.wfile.write(b"Internal Server Error")

            def log_message(self, *args):
                pass

        server = self._start_server(Handler)
        try:
            host, port = server.server_address
            with tempfile.TemporaryDirectory() as tmpdir:
                _write_py(tmpdir, "bad.py", '''\
df = spark.table("orders")
df.collect()
''')
                orch = CIOrchestrator(
                    paths=[tmpdir],
                    api_key="test-key",
                    environment="prod",
                    base_url=f"http://{host}:{port}",
                )
                result = orch.run()
                # Should still have linter findings even though enrichment failed
                self.assertFalse(result.stats["enriched"])
                self.assertGreater(result.stats["linterFindings"], 0)
                self.assertGreater(len(result.findings), 0)
        finally:
            server.shutdown()


class TestOrchestratorNotebook(unittest.TestCase):
    """Test notebook detection in orchestrator."""

    def test_notebook_cell_maps_built(self):
        notebook_source = """\
# Databricks notebook source
# COMMAND ----------
df = spark.table("orders")
df.collect()
# COMMAND ----------
df.show()
"""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_py(tmpdir, "notebook.py", notebook_source)
            orch = CIOrchestrator(paths=[tmpdir])
            result = orch.run()
            self.assertGreater(result.stats["linterFindings"], 0)


if __name__ == "__main__":
    unittest.main()
