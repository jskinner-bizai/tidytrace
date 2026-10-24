"""Killable, single-request HTTP transport. Production destination stays fixed."""
from __future__ import annotations

import base64
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from urllib import request
from urllib.parse import urlsplit

ENDPOINT = "http://127.0.0.1:11434/api/generate"
MAX_REQUEST_BYTES = 16_384
MAX_RESPONSE_BYTES = 16_384


class TransportFailure(RuntimeError):
    pass


class NoRedirect(request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise TransportFailure("Redirect from the approved local endpoint is forbidden")


def _allowed_endpoint(endpoint):
    # Loopback exists solely for offline transport tests; the adapter never
    # accepts a caller-provided destination and always uses ENDPOINT.
    parsed = urlsplit(endpoint)
    return endpoint == ENDPOINT or (
        parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "::1"}
        and parsed.username is None and parsed.password is None
        and not parsed.query and not parsed.fragment
    )


def bounded_http_request(req, timeout, max_bytes, metadata=None):
    """Return raw bytes, killing and reaping the HTTP worker at the total budget.

    The budget includes process startup and response read. OS process creation
    and termination are not real-time operations; cleanup can add scheduling
    latency. No request thread survives a returned timeout.
    """
    if not _allowed_endpoint(req.full_url) or req.method != "POST":
        raise TransportFailure("Unapproved transport destination or method")
    if not isinstance(req.data, bytes) or len(req.data) > MAX_REQUEST_BYTES:
        raise TransportFailure("Request exceeds the fixed synthetic byte bound")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise TransportFailure("A positive finite request deadline is required")
    if type(max_bytes) is not int or not 0 < max_bytes <= MAX_RESPONSE_BYTES:
        raise TransportFailure("Invalid response byte bound")
    started = time.monotonic()
    packet = json.dumps({"endpoint": req.full_url, "timeout": timeout,
                         "max_bytes": max_bytes,
                         "body": base64.b64encode(req.data).decode("ascii")}).encode()
    worker = None
    details = metadata if metadata is not None else {}
    details.update({"transport": "killable_subprocess", "deadline_seconds": timeout,
                    "worker_stopped": False})
    try:
        worker = subprocess.Popen(
            [sys.executable, "-I", str(Path(__file__).resolve()), "--worker"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            shell=False, close_fds=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        details["worker_pid"] = worker.pid
        remaining = timeout - (time.monotonic() - started)
        if remaining <= 0:
            raise subprocess.TimeoutExpired(worker.args, timeout)
        stdout, stderr = worker.communicate(packet, timeout=remaining)
        if time.monotonic() - started > timeout:
            raise subprocess.TimeoutExpired(worker.args, timeout)
        if worker.returncode:
            # Worker emits only a bounded exception type/message, no request.
            raise TransportFailure("Local HTTP worker failed: " + stderr[:1024].decode("utf-8", errors="replace"))
        if len(stdout) > 24_000:
            raise TransportFailure("Invalid worker output size")
        try:
            envelope = json.loads(stdout)
            if set(envelope) != {"status", "body"} or type(envelope["status"]) is not int:
                raise ValueError("Invalid worker envelope")
            if envelope["status"] != 200:
                raise TransportFailure(f"Local model returned HTTP {envelope['status']}")
            body = base64.b64decode(envelope["body"], validate=True)
        except (ValueError, TypeError, KeyError) as exc:
            raise TransportFailure("Invalid local HTTP worker output") from exc
        return body
    except subprocess.TimeoutExpired as exc:
        details["deadline_exceeded"] = True
        raise TransportFailure("Local model exceeded the total response deadline") from exc
    finally:
        cleanup_error = None
        try:
            if worker is not None:
                try:
                    if worker.poll() is None:
                        worker.kill()
                    # Confirm reaping before reporting ordinary success/timeout.
                    worker.communicate(timeout=2)
                except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
                    cleanup_error = exc
                    details.update({"cleanup_failed": True,
                                    "cleanup_error_type": type(exc).__name__})
                finally:
                    details["worker_stopped"] = worker.poll() is not None
                    details["worker_returncode"] = worker.returncode
                    details["cleanup_status"] = "confirmed" if details["worker_stopped"] and cleanup_error is None else "uncertain"
        finally:
            details["transport_elapsed_ms"] = round((time.monotonic() - started) * 1000, 3)
        if cleanup_error is not None:
            raise TransportFailure("Local HTTP worker cleanup could not be confirmed; no retry or fallback") from cleanup_error


def _worker():
    packet = json.loads(sys.stdin.buffer.read(25_000))
    endpoint, timeout, max_bytes = packet["endpoint"], packet["timeout"], packet["max_bytes"]
    body = base64.b64decode(packet["body"], validate=True)
    if not _allowed_endpoint(endpoint) or len(body) > MAX_REQUEST_BYTES:
        raise TransportFailure("Unapproved worker input")
    if not 0 < max_bytes <= MAX_RESPONSE_BYTES or not math.isfinite(timeout) or timeout <= 0:
        raise TransportFailure("Invalid worker bounds")
    opener = request.build_opener(request.ProxyHandler({}), NoRedirect()).open
    req = request.Request(endpoint, data=body, method="POST",
                          headers={"Content-Type": "application/json", "Accept": "application/json"})
    with opener(req, timeout=timeout) as response:
        raw = response.read(max_bytes + 1)
        output = {"status": response.status, "body": base64.b64encode(raw).decode("ascii")}
    sys.stdout.write(json.dumps(output))


if __name__ == "__main__":
    if sys.argv[1:] != ["--worker"]:
        raise SystemExit("This private transport worker is not an application command")
    try:
        _worker()
    except Exception as exc:
        sys.stderr.write((type(exc).__name__ + ": " + str(exc))[:1024])
        raise SystemExit(1)
