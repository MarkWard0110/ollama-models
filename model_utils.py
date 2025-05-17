import requests
import os
import csv

API_BASE = "http://quorra.homelan.binaryward.com:11434"

def fetch_installed_models():
    try:
        resp = requests.get(f"{API_BASE}/api/tags", timeout=10)
        data = resp.json()
        return data.get("models", [])
    except Exception:
        return []

def fetch_max_context_size(model_name):
    try:
        resp = requests.post(f"{API_BASE}/api/show", json={"model": model_name}, timeout=10)
        info = resp.json().get("model_info", {})
        for key, value in info.items():
            if key.endswith(".context_length"):
                return value
        return 2048
    except Exception:
        return 2048

def try_model_call(model_name, context_size):
    payload_chat = {
        "model": model_name,
        "prompt": "What is the capital of France?",
        "stream": False,
        "options": {
            "num_ctx": context_size,
            "max_tokens": 64,
        }
    }
    try:
        requests.post(f"{API_BASE}/api/chat", json=payload_chat, timeout=1200)
        return True
    except Exception:
        embed_payload = {"model": model_name, "input": "Test"}
        try:
            requests.post(f"{API_BASE}/api/embed", json=embed_payload, timeout=1200)
            return True
        except Exception:
            return False

def fetch_memory_usage(model_name):
    try:
        resp = requests.get(f"{API_BASE}/api/ps", timeout=10)
        for m in resp.json().get("models", []):
            if m.get("model") == model_name:
                return m.get("size", 0), m.get("size_vram", 0)
        return 0, 0
    except Exception:
        return 0, 0

def format_size(num_bytes):
    units = ["B", "KiB", "MiB", "GiB", "TiB", "PiB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f}{units[idx]}"
