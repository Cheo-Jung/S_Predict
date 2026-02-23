"""Tiny dependency-free HTTP API + GUI for MVP predictions.

Endpoints:
- GET /health
- GET /predict
- GET /meta
- GET /
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from predict_mvp import predict_latest

HOST = "127.0.0.1"
PORT = 8000
META_PATH = Path("artifacts/meta.json")

HTML_PAGE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Stock Predictor MVP</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f7fb;
      --card: #ffffff;
      --text: #10243e;
      --muted: #55708f;
      --accent: #245ee6;
      --accent-2: #1fa37f;
      --danger: #b0383c;
      --border: #dbe5f1;
      --shadow: 0 8px 30px rgba(16, 36, 62, 0.08);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      font-family: Inter, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
      background: radial-gradient(circle at top right, #eef3ff 0%, var(--bg) 45%);
      color: var(--text);
      line-height: 1.5;
    }

    .container {
      width: min(1100px, 92vw);
      margin: 2rem auto 3rem;
    }

    .header {
      display: flex;
      justify-content: space-between;
      align-items: center;
      margin-bottom: 1.25rem;
      gap: 1rem;
      flex-wrap: wrap;
    }

    h1 {
      margin: 0;
      font-size: clamp(1.5rem, 3vw, 2.1rem);
      letter-spacing: -0.02em;
    }

    .subtitle {
      margin: 0.25rem 0 0;
      color: var(--muted);
      font-size: 0.98rem;
    }

    .status-pill {
      display: inline-flex;
      align-items: center;
      gap: 0.5rem;
      padding: 0.5rem 0.8rem;
      border: 1px solid var(--border);
      border-radius: 999px;
      background: var(--card);
      font-size: 0.9rem;
      box-shadow: var(--shadow);
    }

    .dot {
      width: 0.65rem;
      height: 0.65rem;
      border-radius: 999px;
      background: #9caec2;
    }

    .dot.ok { background: var(--accent-2); }
    .dot.err { background: var(--danger); }

    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 1rem;
    }

    .card {
      background: var(--card);
      border: 1px solid var(--border);
      border-radius: 16px;
      box-shadow: var(--shadow);
      padding: 1rem 1.1rem;
    }

    .card h2 {
      margin: 0 0 0.8rem;
      font-size: 1.1rem;
    }

    .prediction {
      display: flex;
      align-items: baseline;
      gap: 0.6rem;
      margin: 0.2rem 0 0.8rem;
    }

    .prediction .value {
      font-size: 2rem;
      font-weight: 700;
      letter-spacing: -0.03em;
    }

    .prediction .badge {
      font-size: 0.82rem;
      border: 1px solid var(--border);
      color: var(--muted);
      border-radius: 999px;
      padding: 0.2rem 0.55rem;
    }

    .kv {
      margin: 0;
      display: grid;
      grid-template-columns: 1fr auto;
      gap: 0.4rem 0.6rem;
      font-size: 0.93rem;
    }

    .kv dt { color: var(--muted); }
    .kv dd { margin: 0; font-variant-numeric: tabular-nums; }

    button {
      border: none;
      background: var(--accent);
      color: white;
      border-radius: 10px;
      padding: 0.6rem 0.9rem;
      font-weight: 600;
      cursor: pointer;
    }

    button:hover { filter: brightness(1.07); }

    .raw {
      margin-top: 1rem;
      background: #0f2137;
      color: #d9e6f6;
      border-radius: 14px;
      padding: 0.9rem;
      max-height: 260px;
      overflow: auto;
      font-size: 0.82rem;
      white-space: pre-wrap;
      border: 1px solid #203a5f;
    }
  </style>
</head>
<body>
  <main class="container">
    <section class="header">
      <div>
        <h1>Stock Predictor MVP</h1>
        <p class="subtitle">Simple dashboard for health, latest prediction, and model metrics.</p>
      </div>
      <div class="status-pill"><span id="health-dot" class="dot"></span><span id="health-text">Checking API…</span></div>
    </section>

    <section class="grid">
      <article class="card">
        <h2>Latest Prediction</h2>
        <div class="prediction">
          <div id="pred-main" class="value">--</div>
          <span id="pred-task" class="badge">task</span>
        </div>
        <dl class="kv" id="prediction-kv"></dl>
        <p style="margin-top:0.9rem;"><button id="refresh-btn">Refresh Data</button></p>
      </article>

      <article class="card">
        <h2>Evaluation Metrics</h2>
        <dl class="kv" id="metrics-kv"></dl>
      </article>
    </section>

    <section class="raw" id="raw-json">Loading response…</section>
  </main>

  <script>
    const healthDot = document.getElementById('health-dot');
    const healthText = document.getElementById('health-text');
    const predMain = document.getElementById('pred-main');
    const predTask = document.getElementById('pred-task');
    const predictionKv = document.getElementById('prediction-kv');
    const metricsKv = document.getElementById('metrics-kv');
    const rawJson = document.getElementById('raw-json');

    function formatValue(value) {
      if (typeof value === 'number') {
        return Math.abs(value) < 1 ? value.toFixed(4) : value.toLocaleString();
      }
      return String(value);
    }

    function renderKv(element, obj) {
      element.innerHTML = '';
      if (!obj || Object.keys(obj).length === 0) {
        element.innerHTML = '<dt>No data</dt><dd>—</dd>';
        return;
      }
      for (const [k, v] of Object.entries(obj)) {
        const dt = document.createElement('dt');
        dt.textContent = k.replaceAll('_', ' ');
        const dd = document.createElement('dd');
        dd.textContent = formatValue(v);
        element.appendChild(dt);
        element.appendChild(dd);
      }
    }

    async function checkHealth() {
      try {
        const res = await fetch('/health');
        if (!res.ok) throw new Error('health not ok');
        const data = await res.json();
        healthDot.className = 'dot ok';
        healthText.textContent = data.status === 'ok' ? 'API healthy' : 'API responded';
      } catch {
        healthDot.className = 'dot err';
        healthText.textContent = 'API unreachable';
      }
    }

    async function loadData() {
      predMain.textContent = '...';
      try {
        const res = await fetch('/predict');
        const payload = await res.json();

        if (!res.ok || payload.error) {
          throw new Error(payload.error || 'Unknown server error');
        }

        const prediction = payload.prediction || {};
        const metrics = payload.metrics || {};

        predTask.textContent = prediction.task || 'unknown';

        if (prediction.task === 'direction') {
          predMain.textContent = prediction.predicted_direction || '--';
        } else if (prediction.task === 'return') {
          const value = prediction.predicted_next_day_return;
          predMain.textContent = typeof value === 'number' ? `${(value * 100).toFixed(2)}%` : '--';
        } else {
          predMain.textContent = '--';
        }

        renderKv(predictionKv, prediction);
        renderKv(metricsKv, metrics);
        rawJson.textContent = JSON.stringify(payload, null, 2);
      } catch (err) {
        predMain.textContent = 'Error';
        rawJson.textContent = `Failed to load data: ${err.message}`;
        predictionKv.innerHTML = '<dt>Status</dt><dd>Could not fetch prediction</dd>';
      }
    }

    document.getElementById('refresh-btn').addEventListener('click', loadData);

    checkHealth();
    loadData();
  </script>
</body>
</html>
"""


class Handler(BaseHTTPRequestHandler):
    def _send_json(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_html(self, html: str) -> None:
        body = html.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self._send_html(HTML_PAGE)
            return

        if self.path == "/health":
            self._send_json(200, {"status": "ok"})
            return

        if self.path == "/meta":
            metrics = json.loads(META_PATH.read_text()) if META_PATH.exists() else {}
            self._send_json(200, {"metrics": metrics})
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
    print(f"Serving MVP API + GUI on http://{HOST}:{PORT}")
    server.serve_forever()


if __name__ == "__main__":
    run()
