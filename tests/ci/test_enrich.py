"""Tests for enrich.py — request assembly, HTTP client, response processing."""

import json
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from unittest.mock import patch

from cylint.ci.enrich import (
    EnrichRequest,
    EnrichResponse,
    convert_finding,
    convert_changed_lines,
    post_enrich,
    resolve_provenance,
)
from cylint.models import Finding, Severity


class TestConvertFinding(unittest.TestCase):
    def _make_finding(self, **kwargs):
        defaults = dict(
            rule_id="CY001",
            severity=Severity.WARNING,
            message="test message",
            filepath="test.py",
            line=10,
        )
        defaults.update(kwargs)
        return Finding(**defaults)

    def test_plain_py_no_cell_map(self):
        f = self._make_finding()
        result = convert_finding(f, cell_map=None)
        self.assertEqual(result["rule"], "CY001")
        self.assertEqual(result["file"], "test.py")
        self.assertEqual(result["line"], 10)
        self.assertNotIn("cellFingerprint", result)

    def test_notebook_with_cell_map(self):
        cell_map = {"abc123": 5}  # cell starts at line 5
        f = self._make_finding(line=7)
        result = convert_finding(f, cell_map=cell_map)
        self.assertEqual(result["cellFingerprint"], "abc123")
        self.assertEqual(result["cellLine"], 3)
        self.assertEqual(result["absoluteLine"], 7)


class TestConvertChangedLines(unittest.TestCase):
    def test_plain_py(self):
        result = convert_changed_lines([10, 20], "test.py", cell_map=None)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["file"], "test.py")
        self.assertEqual(result[0]["line"], 10)

    def test_notebook(self):
        cell_map = {"abc123": 5}
        result = convert_changed_lines([7], "nb.py", cell_map)
        self.assertEqual(result[0]["cellFingerprint"], "abc123")
        self.assertEqual(result[0]["cellLine"], 3)


class TestResolveProvenance(unittest.TestCase):
    def test_trigger_line_resolved(self):
        cell_map = {"fp1": 10}
        finding = {
            "triggerCellFingerprint": "fp1",
            "triggerLine": 3,
        }
        result = resolve_provenance(finding, cell_map)
        self.assertEqual(result["triggerLineAbsolute"], 12)

    def test_construction_lines_resolved(self):
        cell_map = {"fp1": 10, "fp2": 30}
        finding = {
            "triggerCellFingerprint": "fp1",
            "triggerLine": 1,
            "constructionLines": [
                {"cellFingerprint": "fp1", "lines": [1, 2]},
                {"cellFingerprint": "fp2", "lines": [5]},
            ],
        }
        result = resolve_provenance(finding, cell_map)
        self.assertEqual(result["constructionLinesAbsolute"], [10, 11, 34])
        self.assertEqual(result["constructionSpanStart"], 10)
        self.assertEqual(result["constructionSpanEnd"], 34)

    def test_unknown_fingerprint(self):
        cell_map = {"fp1": 10}
        finding = {
            "triggerCellFingerprint": "unknown",
            "triggerLine": 1,
        }
        result = resolve_provenance(finding, cell_map)
        self.assertIsNone(result["triggerLineAbsolute"])


class TestPostEnrich(unittest.TestCase):
    """Test post_enrich with a local mock HTTP server."""

    def _start_server(self, handler_cls):
        server = HTTPServer(("127.0.0.1", 0), handler_cls)
        thread = Thread(target=server.serve_forever)
        thread.daemon = True
        thread.start()
        return server

    def test_success(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                # Read request body to avoid connection reset
                length = int(self.headers.get("Content-Length", 0))
                self.rfile.read(length)

                body = json.dumps({
                    "findings": [{"rule": "CY001", "savings": 500}],
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
            req = EnrichRequest(
                files=[{"path": "test.py"}],
                linter_findings=[{"rule": "CY001"}],
                environment="prod",
            )
            resp = post_enrich(req, "test-key", base_url=f"http://{host}:{port}")
            self.assertIsNotNone(resp)
            self.assertEqual(len(resp.findings), 1)
            self.assertEqual(resp.match_stats["fingerprintMatchRate"], 1.0)
        finally:
            server.shutdown()

    def test_server_error_returns_none(self):
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
            req = EnrichRequest(environment="prod")
            resp = post_enrich(req, "test-key", base_url=f"http://{host}:{port}")
            self.assertIsNone(resp)
        finally:
            server.shutdown()

    def test_timeout_returns_none(self):
        req = EnrichRequest(environment="prod")
        # Use an unreachable address to trigger timeout
        resp = post_enrich(
            req, "test-key",
            base_url="http://192.0.2.1",  # RFC 5737 TEST-NET
            timeout=1,
        )
        self.assertIsNone(resp)

    def test_auth_failure_returns_none(self):
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.send_response(401)
                self.end_headers()
                self.wfile.write(b"Unauthorized")

            def log_message(self, *args):
                pass

        server = self._start_server(Handler)
        try:
            host, port = server.server_address
            req = EnrichRequest(environment="prod")
            resp = post_enrich(req, "bad-key", base_url=f"http://{host}:{port}")
            self.assertIsNone(resp)
        finally:
            server.shutdown()


if __name__ == "__main__":
    unittest.main()
