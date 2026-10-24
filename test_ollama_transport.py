"""Offline loopback tests exercise the real subprocess and HTTP socket lifecycle."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import socket
import subprocess
import threading
import time
import unittest
from unittest.mock import Mock, patch
from urllib import request

import ollama_transport
from ollama_transport import NoRedirect, TransportFailure, bounded_http_request


@contextmanager
def server(mode):
    stopped = threading.Event()
    disconnected = threading.Event()
    observed = {"requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_POST(self):
            observed["requests"] += 1
            observed["body"] = self.rfile.read(int(self.headers["Content-Length"]))
            if mode == "stall_headers":
                # No response at all. recv detects EOF when the killed worker's
                # socket closes; the server does not wait for a Python thread.
                self.connection.settimeout(3)
                try:
                    disconnected.set() if self.connection.recv(1) == b"" else None
                except OSError:
                    disconnected.set()
                return
            if mode == "redirect":
                self.send_response(302)
                self.send_header("Location", "/unexpected-second-request")
                self.end_headers()
                return
            self.send_response(200)
            body = b"success" if mode == "success" else b"x" * 20_000
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            try:
                if mode == "drip":
                    for _ in range(20_000):
                        if stopped.wait(.02):
                            return
                        self.wfile.write(b"x")
                        self.wfile.flush()
                else:
                    self.wfile.write(body)
                    self.wfile.flush()
            except OSError:
                disconnected.set()

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    thread = threading.Thread(target=lambda: httpd.serve_forever(poll_interval=.02), daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{httpd.server_port}/api/generate", observed, disconnected
    finally:
        stopped.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(timeout=2)


class TransportTests(unittest.TestCase):
    def req(self, endpoint):
        return request.Request(endpoint, data=b'{"synthetic":true}', method="POST")

    def test_real_child_completes_single_request_and_is_reaped(self):
        with server("success") as (endpoint, observed, _):
            metadata = {}
            result = bounded_http_request(self.req(endpoint), 3, 16_384, metadata)
            self.assertEqual(result, b"success")
            self.assertEqual(observed["requests"], 1)
            self.assertEqual(observed["body"], b'{"synthetic":true}')
            self.assertTrue(metadata["worker_stopped"])
            self.assertEqual(metadata["worker_returncode"], 0)

    def test_stall_and_drip_cannot_outlive_total_budget(self):
        for mode in ["stall_headers", "drip"]:
            with self.subTest(mode=mode), server(mode) as (endpoint, observed, disconnected):
                metadata = {}
                start = time.monotonic()
                with self.assertRaisesRegex(TransportFailure, "total response deadline"):
                    bounded_http_request(self.req(endpoint), .6, 16_384, metadata)
                elapsed = time.monotonic() - start
                self.assertLess(elapsed, 1.6, "worker cleanup exceeded the generous offline test allowance")
                self.assertTrue(metadata["deadline_exceeded"])
                self.assertTrue(metadata["worker_stopped"])
                self.assertIsNotNone(metadata["worker_returncode"])
                self.assertTrue(disconnected.wait(2), "HTTP server did not observe the worker disconnect")
                self.assertEqual(observed["requests"], 1)

    def test_oversize_response_remains_bounded(self):
        with server("oversize") as (endpoint, observed, _):
            metadata = {}
            result = bounded_http_request(self.req(endpoint), 3, 16_384, metadata)
            self.assertEqual(len(result), 16_385)
            self.assertEqual(observed["requests"], 1)
            self.assertTrue(metadata["worker_stopped"])

    def test_redirect_does_not_make_second_request(self):
        with server("redirect") as (endpoint, observed, _):
            metadata = {}
            with self.assertRaisesRegex(TransportFailure, "Redirect"):
                bounded_http_request(self.req(endpoint), 3, 16_384, metadata)
            self.assertEqual(observed["requests"], 1)
            self.assertTrue(metadata["worker_stopped"])

    def test_unapproved_destination_never_starts_worker(self):
        with patch("ollama_transport.subprocess.Popen") as popen:
            with self.assertRaisesRegex(TransportFailure, "Unapproved"):
                bounded_http_request(self.req("https://elsewhere.invalid/api/generate"), 1, 100)
            popen.assert_not_called()

    def test_cleanup_timeout_is_explicit_and_keeps_elapsed_metadata(self):
        worker = Mock(pid=123, args=["synthetic-worker"], returncode=None)
        worker.poll.return_value = None
        worker.communicate.side_effect = [
            subprocess.TimeoutExpired(worker.args, .1),
            subprocess.TimeoutExpired(worker.args, 2),
        ]
        metadata = {}
        with patch("ollama_transport.subprocess.Popen", return_value=worker):
            with self.assertRaisesRegex(TransportFailure, "cleanup could not be confirmed"):
                bounded_http_request(self.req(ollama_transport.ENDPOINT), .1, 100, metadata)
        worker.kill.assert_called_once()
        self.assertTrue(metadata["deadline_exceeded"])
        self.assertTrue(metadata["cleanup_failed"])
        self.assertFalse(metadata["worker_stopped"])
        self.assertEqual(metadata["cleanup_status"], "uncertain")
        self.assertIn("transport_elapsed_ms", metadata)

    def test_default_opener_disables_environment_proxy(self):
        packet = {"endpoint": ollama_transport.ENDPOINT, "timeout": 1,
                  "max_bytes": 100, "body": "e30="}
        from io import BytesIO, StringIO
        with patch("ollama_transport.sys.stdin") as stdin, patch("ollama_transport.sys.stdout", new_callable=StringIO), patch("ollama_transport.request.build_opener") as build_opener:
            stdin.buffer = BytesIO(json.dumps(packet).encode())
            response = build_opener.return_value.open.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = b"{}"
            ollama_transport._worker()
            handlers = build_opener.call_args.args
            self.assertEqual(handlers[0].proxies, {})
            self.assertIsInstance(handlers[1], NoRedirect)


if __name__ == "__main__":
    unittest.main()
