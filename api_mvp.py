"""Simple HTTP API for real-data-capable direction predictions.

Endpoints:
- GET /health
- GET /predict?symbol=AAPL&start=2016-01-01&end=2026-01-01
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from predict_mvp import predict_latest

HOST = "127.0.0.1"
PORT = 8000
META_PATH = Path("artifacts/meta.json")


class Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send(200, {"status": "ok"})
            return

        if parsed.path == "/predict":
            q = parse_qs(parsed.query)
            symbol = (q.get("symbol") or [None])[0]
            start = (q.get("start") or ["2016-01-01"])[0]
            end = (q.get("end") or ["2100-01-01"])[0]
            try:
                pred = predict_latest(symbol=symbol, start=start, end=end)
                metrics = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
                self._send(200, {"prediction": pred, "metrics": metrics})
            except Exception as exc:
                self._send(500, {"error": str(exc)})
            return

        self._send(404, {"error": "not found"})


def run() -> None:
    server = HTTPServer((HOST, PORT), Handler)
    print(f"Serving on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
