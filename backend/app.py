from __future__ import annotations

from backend.api.server import app


def main() -> None:
    app.run(host="0.0.0.0", port=8000, debug=False, threaded=True)


__all__ = ["app", "main"]
