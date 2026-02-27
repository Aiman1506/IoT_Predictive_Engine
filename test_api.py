"""
FactoryGuard AI - API Test Client
===================================
Quick sanity test for the Flask REST API.

Usage:
  1. Start server: python serve.py
  2. In another terminal: python test_api.py
"""

import http.client
import json
import sys
import time
import urllib.request
import urllib.error
from urllib.parse import urlparse

BASE_URL = "http://localhost:5000"
_HOST = "localhost"
_PORT = 5000


def post_json(url: str, payload: dict) -> tuple:
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url, data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())
    except Exception as e:
        return None, {"error": str(e)}


def get_json(url: str) -> tuple:
    try:
        with urllib.request.urlopen(url, timeout=5) as resp:
            return resp.status, json.loads(resp.read())
    except Exception as e:
        return None, {"error": str(e)}


def _latency_test(payload: dict, n: int = 20) -> tuple[float, float, float, float]:
    """
    Measure latency two ways:
      1. Round-trip wall-clock time (new connection per request — Werkzeug
         dev server sends Connection: close so keep-alive is unavailable).
      2. Server-side inference time reported in the JSON response body
         (eliminates TCP setup cost, reflects production inference speed).
    """
    body = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    roundtrip_ms = []
    server_ms = []

    for _ in range(n):
        conn = http.client.HTTPConnection(_HOST, _PORT, timeout=5)
        try:
            t0 = time.perf_counter()
            conn.request("POST", "/predict", body=body, headers=headers)
            resp = conn.getresponse()
            raw = resp.read()
            roundtrip_ms.append((time.perf_counter() - t0) * 1000)
            try:
                server_ms.append(json.loads(raw)["response_time_ms"])
            except Exception:
                pass
        except (ConnectionRefusedError, OSError) as e:
            print(f"    [!] Server unreachable during latency test: {e}")
            print("        Make sure `python serve.py` is still running.")
            return 0.0, 0.0, 0.0, 0.0
        finally:
            conn.close()

    avg_rt = sum(roundtrip_ms) / len(roundtrip_ms)
    max_rt = max(roundtrip_ms)
    avg_sv = sum(server_ms) / len(server_ms) if server_ms else 0.0
    max_sv = max(server_ms) if server_ms else 0.0
    return avg_rt, max_rt, avg_sv, max_sv


def run_tests():
    print("=" * 60)
    print("FactoryGuard AI - API Tests")
    print("=" * 60)

    # Health check
    status, body = get_json(f"{BASE_URL}/health")
    print(f"\n[1] GET /health → {status}")
    print(f"    {body}")

    # Normal reading
    normal_payload = {"temperature": 55.0, "vibration": 2.0, "pressure": 6.0}
    status, body = post_json(f"{BASE_URL}/predict", normal_payload)
    print(f"\n[2] POST /predict (normal reading) → {status}")
    print(f"    Payload: {normal_payload}")
    print(f"    Response: {body}")

    # High-risk reading
    risky_payload = {"temperature": 89.0, "vibration": 8.5, "pressure": 3.2}
    status, body = post_json(f"{BASE_URL}/predict", risky_payload)
    print(f"\n[3] POST /predict (high-risk reading) → {status}")
    print(f"    Payload: {risky_payload}")
    print(f"    Response: {body}")

    # Missing field
    bad_payload = {"temperature": 55.0}
    status, body = post_json(f"{BASE_URL}/predict", bad_payload)
    print(f"\n[4] POST /predict (missing fields) → {status}")
    print(f"    {body}")

    # Model info
    status, body = get_json(f"{BASE_URL}/model-info")
    print(f"\n[5] GET /model-info → {status}")
    print(f"    Version: {body.get('version')}")
    print(f"    Features: {body.get('n_features')}")

    # Latency test — two measurements: round-trip and server-side inference
    print(f"\n[6] Latency test (20 predictions)...")
    avg_rt, max_rt, avg_sv, max_sv = _latency_test(normal_payload, n=20)
    if avg_sv > 0:
        print(f"    Round-trip  (incl. TCP):  Avg {avg_rt:.1f} ms | Max {max_rt:.1f} ms")
        print(f"    Server-side (model only): Avg {avg_sv:.1f} ms | Max {max_sv:.1f} ms")
        print(f"    Target (<50ms inference): {'PASS' if avg_sv < 50 else 'FAIL'}")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
