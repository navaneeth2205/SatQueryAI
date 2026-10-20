#!/usr/bin/env python3
"""
Launcher script for SatQuery AI Interactive Web Demo.

Starts the Flask local server providing the web-based satellite analyst
dashboard with task routing, multitemporal change detection overlays,
and multimodal land-cover classification.

Usage:
    python scripts/run_demo_app.py [--host 127.0.0.1] [--port 5000] [--debug]
"""

import argparse
import sys
from pathlib import Path

# Safe Windows stdout encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from satquery.demo.app import create_app


def parse_args():
    parser = argparse.ArgumentParser(
        description="Launch SatQuery AI Interactive Satellite Analyst Demo."
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host address to bind the web server (default: 127.0.0.1)."
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5000,
        help="Port to listen on (default: 5000)."
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable Flask debug mode (default: False)."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 70)
    print("       [SatQuery AI] - Interactive Satellite Analyst Demo")
    print("=" * 70)
    print(f" * Server address: http://{args.host}:{args.port}")
    print(" * Ready for queries: Change Detection & Land Cover Classification")
    print(" * Pre-configured real satellite pairs from OSCD and BigEarthNet-14k")
    print(" * Press CTRL+C to stop the server.")
    print("=" * 70)

    app = create_app()
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
