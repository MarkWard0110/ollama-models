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
        for key, value in info.items():
            if key.endswith(".context_length"):
                return value
        return 2048
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

def fits_in_vram(model_name, context_size):
    print(f"Attempting to fit model '{model_name}' at context size {context_size} in VRAM.")
    if not try_model_call(model_name, context_size):
        print(f"Failed model call for {model_name} at context size {context_size}.")
        return False
    size, size_vram = fetch_memory_usage(model_name)
    print(f"Memory usage for {model_name} at {context_size}: total={size}, VRAM={size_vram}")
    return (size_vram >= size)

def find_max_fit_in_vram(model_name, max_ctx):
    best_fit = 0
    # Exponential search to find an upper bound
    start = 2048
    if not fits_in_vram(model_name, start):
        print(f"{model_name} cannot fit in VRAM even at 2048.")
        return 0
    high = start
    while high <= max_ctx and fits_in_vram(model_name, high):
        high *= 2
    left, right = high // 2, min(high, max_ctx)
    # Binary search in [left, right]
    while left <= right:
        mid = (left + right) // 2
        if fits_in_vram(model_name, mid):
            best_fit = mid
            left = mid + 1
        else:
            right = mid - 1
    print(f"Highest context size fitting in VRAM so far for {model_name}: {best_fit}")
    return best_fit

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
        ctx = 2048
        while ctx <= max_ctx:
            success = try_model_call(name, ctx)
            if success:
                size, size_vram = fetch_memory_usage(name)
                print(f"Measured at 2^n = {ctx}, total allocated: {size}, VRAM: {size_vram}")
                usage_writer.writerow([name, ctx, size])
                usage_file.flush()
            else:
                print(f"Failed chat/embed call for {name} at 2^n size {ctx}")
            ctx *= 2

        best_fit = find_max_fit_in_vram(name, max_ctx)
        if best_fit >= 2048:
            print(f"Max context size fully in VRAM for {name} is {best_fit}")
            fit_writer.writerow([name, best_fit])
            fit_file.flush()

    usage_file.close()
    fit_file.close()

if __name__ == "__main__":
    main()
