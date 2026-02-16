"""Tiny dependency-free HTTP API for MVP predictions.

Endpoints:
- GET /health
- GET /predict
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from predict_mvp import predict_latest

HOST = "127.0.0.1"
PORT = 8000
META_PATH = Path("artifacts/meta.json")


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        if self.path == "/predict":
            try:
                result = predict_latest()
                metrics = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
                self._send_json(200, {"prediction": result, "metrics": metrics})
            except Exception as exc:
                self._send_json(500, {"error": str(exc)})
            return

        self._send_json(404, {"error": "not found"})


def run() -> None:
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Serving MVP API on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
