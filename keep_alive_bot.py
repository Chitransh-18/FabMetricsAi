"""
FabMetrics AI — High-Availability Keep-Alive Knock Bot
======================================================
Prevents free-tier cloud deployments (Render, Railway, Fly.io, HuggingFace Spaces)
from spinning down or sleeping after inactivity periods.

Periodically sends lightweight HTTP health-check knocks to keep the container warm,
the PyTorch model memory-mapped, and the SQLite database active.

Usage:
------
# Knock every 10 minutes (600 seconds) against a deployed website:
python keep_alive_bot.py --url https://your-fabmetrics-service.onrender.com --interval 600

# Knock once for diagnostic check:
python keep_alive_bot.py --url http://127.0.0.1:8000 --once
"""

from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
import urllib.request
import urllib.error


def ping_endpoint(base_url: str, timeout: int = 15) -> dict:
    """Send a lightweight HTTP request to the target server's health endpoint."""
    clean_url = base_url.rstrip("/")
    # Try /api/health first, then /health
    targets = [f"{clean_url}/api/health", f"{clean_url}/health", clean_url]
    
    last_err = None
    for target in targets:
        req = urllib.request.Request(
            target,
            headers={
                "User-Agent": "FabMetrics-KeepAliveBot/4.0 (Anti-Sleep Daemon)",
                "Accept": "application/json"
            }
        )
        start_t = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                elapsed_ms = (time.perf_counter() - start_t) * 1000.0
                status_code = resp.getcode()
                body = resp.read(512).decode("utf-8", errors="ignore")
                return {
                    "success": True,
                    "target": target,
                    "status_code": status_code,
                    "latency_ms": round(elapsed_ms, 2),
                    "body": body[:120]
                }
        except urllib.error.HTTPError as http_err:
            elapsed_ms = (time.perf_counter() - start_t) * 1000.0
            return {
                "success": False,
                "target": target,
                "status_code": http_err.code,
                "latency_ms": round(elapsed_ms, 2),
                "error": f"HTTP {http_err.code}: {http_err.reason}"
            }
        except Exception as e:
            last_err = str(e)
            continue

    return {
        "success": False,
        "target": targets[0],
        "status_code": 0,
        "latency_ms": 0.0,
        "error": last_err or "Unknown connection failure"
    }


def run_bot(url: str, interval: int = 600, max_retries: int = 3, once: bool = False):
    print("=" * 78)
    print("FabMetrics AI — High-Availability Keep-Alive Knock Bot")
    print(f"Target URL     : {url}")
    print(f"Knock Interval : {interval} seconds ({interval / 60:.1f} minutes)")
    print(f"Mode           : {'Single Diagnostic Knock' if once else 'Continuous Background Sentinel'}")
    print("=" * 78)

    knocks_sent = 0
    knocks_successful = 0
    consecutive_failures = 0

    while True:
        knocks_sent += 1
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        result = ping_endpoint(url)

        if result["success"]:
            knocks_successful += 1
            consecutive_failures = 0
            print(f"[{now_str}] [KNOCK #{knocks_sent:04d}] OK (HTTP {result['status_code']}) | Latency: {result['latency_ms']:>6.1f}ms | Target: {result['target']}")
        else:
            consecutive_failures += 1
            err_msg = result.get('error', 'Unreachable')
            print(f"[{now_str}] [KNOCK #{knocks_sent:04d}] FAILED ({err_msg}) | Target: {result['target']}")

        if once:
            break

        time.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FabMetrics AI Keep-Alive Knock Bot")
    parser.add_argument(
        "--url",
        type=str,
        default=os.environ.get("FABMETRICS_APP_URL", "http://127.0.0.1:8000"),
        help="Target base URL of deployed or local FabMetrics AI app"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=600,
        help="Knock interval in seconds (default: 600 = 10 minutes)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Send a single diagnostic knock and exit"
    )
    args = parser.parse_args()

    run_bot(url=args.url, interval=args.interval, once=args.once)
