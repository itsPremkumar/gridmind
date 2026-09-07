"""REST API server for GridMind."""
from __future__ import annotations

from typing import Any


def create_app(grid_manager=None, optimizer=None, predictor=None, detector=None):
    """Create a simple WSGI/HTTP app for grid control."""
    from http.server import HTTPServer, BaseHTTPRequestHandler
    import json

    class GridHandler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/stats":
                stats = grid_manager.get_stats() if grid_manager else {}
                self._respond(200, stats)
            elif self.path == "/health":
                self._respond(200, {"status": "ok"})
            else:
                self._respond(404, {"error": "not found"})

        def do_POST(self):
            if self.path == "/optimize":
                if optimizer:
                    result = optimizer.optimize()
                    self._respond(200, {
                        "iterations": result.iterations,
                        "initial_cost": result.initial_cost,
                        "final_cost": result.final_cost,
                        "converged": result.converged,
                    })
                else:
                    self._respond(503, {"error": "optimizer not available"})
            else:
                self._respond(404, {"error": "not found"})

        def _respond(self, status: int, body: dict):
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, format, *args):
            pass  # Suppress logs

    return HTTPServer(("0.0.0.0", 8080), GridHandler)
