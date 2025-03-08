#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import math
import csv
import sys

API_BASE = "http://localhost:11434"

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
        return info.get("llama.context_length", 2048)
    except Exception:
        return 2048

def try_model_call(model_name, context_size):
    # ...existing code...
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
        requests.post(f"{API_BASE}/api/chat", json=payload_chat, timeout=10)
        return True
    except Exception:
        # ...existing code...
        embed_payload = {"model": model_name, "input": "Test"}
        try:
            requests.post(f"{API_BASE}/api/embed", json=embed_payload, timeout=10)
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

def main():
    models = fetch_installed_models()
    usage_file = open("context_usage.csv", "w", newline="")
    fit_file = open("max_context.csv", "w", newline="")
    usage_writer = csv.writer(usage_file)
    fit_writer = csv.writer(fit_file)
    usage_writer.writerow(["model_name", "context_size", "memory_allocated"])
    fit_writer.writerow(["model_name", "max_context_size"])

    for m in models:
        name = m.get("name")
        print(f"Processing model: {name}")
        max_ctx = fetch_max_context_size(name)
        print(f"Maximum reported context size: {max_ctx}")
        last_fit = 2048
        ctx = 2048
        while ctx <= max_ctx:
            print(f"Testing context size: {ctx}")
            success = try_model_call(name, ctx)
            if not success: 
                print(f"Model call failed for {name} at context size {ctx}, stopping.")
                break
            size, size_vram = fetch_memory_usage(name)
            print(f"Total allocated: {size}, In VRAM: {size_vram}")
            usage_writer.writerow([name, ctx, size])
            if size_vram < size:
                print(f"Model offloaded memory at {ctx}, stopping.")
                break
            last_fit = ctx
            ctx *= 2
        print(f"Max context size fully in VRAM for {name} is {last_fit}")
        fit_writer.writerow([name, last_fit])

    usage_file.close()
    fit_file.close()

if __name__ == "__main__":
    main()
