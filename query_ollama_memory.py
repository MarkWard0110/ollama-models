#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import requests
import math
import csv
import sys

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
        requests.post(f"{API_BASE}/api/chat", json=payload_chat, timeout=1200)
        return True
    except Exception:
        # ...existing code...
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

def fits_in_vram(model_name, context_size):
    if not try_model_call(model_name, context_size):
        print(f"Failed model call for {model_name} at context size {context_size}.")
        return False
    size, size_vram = fetch_memory_usage(model_name)
    size_hr = format_size(size)
    size_vram_hr = format_size(size_vram)
    print(f"Memory usage for {model_name} at {context_size}: total={size_hr}, VRAM={size_vram_hr}")
    return (size_vram >= size)

def find_max_fit_in_vram(model_name, max_ctx):
    print(f"Finding max context size fitting in VRAM for {model_name}...")
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

def read_existing_data(usage_path, fit_path):
    usage_set = set()
    fit_models = set()
    if os.path.isfile(usage_path):
        with open(usage_path, "r", newline="") as uf:
            r = csv.reader(uf)
            next(r, None)  # skip header
            for row in r:
                if len(row) >= 2:
                    usage_set.add((row[0], int(row[1])))
    if os.path.isfile(fit_path):
        with open(fit_path, "r", newline="") as ff:
            r = csv.reader(ff)
            next(r, None)  # skip header
            for row in r:
                if len(row) >= 1:
                    fit_models.add(row[0])
    return usage_set, fit_models

def format_size(num_bytes):
    units = ["B", "KB", "MB", "GB", "TB", "PB"]
    size = float(num_bytes)
    idx = 0
    while size >= 1024 and idx < len(units) - 1:
        size /= 1024
        idx += 1
    return f"{size:.1f}{units[idx]}"

def main():
    usage_path = "context_usage.csv"
    fit_path = "max_context.csv"

    usage_set, fit_models = read_existing_data(usage_path, fit_path)

    # Open usage file in append mode if it exists, else create and write header
    usage_exists = os.path.isfile(usage_path)
    usage_file = open(usage_path, 'a' if usage_exists else 'w', newline="")
    usage_writer = csv.writer(usage_file)
    if not usage_exists:
        usage_writer.writerow(["model_name", "context_size", "memory_allocated"])

    # Open fit file in append mode similarly
    fit_exists = os.path.isfile(fit_path)
    fit_file = open(fit_path, 'a' if fit_exists else 'w', newline="")
    fit_writer = csv.writer(fit_file)
    if not fit_exists:
        fit_writer.writerow(["model_name", "max_context_size", "is_model_max"])

    models = fetch_installed_models()

    for m in models:
        name = m.get("name")
        if name in fit_models:
            print(f"Skipping {name}: already has a max_context entry.")
            continue
        print(f"Processing model: {name}")
        max_ctx = fetch_max_context_size(name)
        print(f"Maximum reported context size: {max_ctx}")

        ctx = 2048
        while ctx <= max_ctx:
            if (name, ctx) in usage_set:
                print(f"Skipping model {name} at 2^n={ctx}: already tested.")
                ctx *= 2
                continue
            success = try_model_call(name, ctx)
            if success:
                size, size_vram = fetch_memory_usage(name)
                size_hr = format_size(size)
                size_vram_hr = format_size(size_vram)
                print(f"Measured at 2^n = {ctx}, total allocated: {size_hr}, VRAM: {size_vram_hr}")
                usage_writer.writerow([name, ctx, size_hr])
                usage_file.flush()
                usage_set.add((name, ctx))
            else:
                print(f"Failed chat/embed call for {name} at 2^n size {ctx}")
            ctx *= 2

        best_fit = find_max_fit_in_vram(name, max_ctx)
        if best_fit >= 2048:
            is_model_max = (best_fit == max_ctx)
            print(f"Max context size fully in VRAM for {name} is {best_fit}")
            fit_writer.writerow([name, best_fit, is_model_max])
            fit_file.flush()
            fit_models.add(name)

    usage_file.close()
    fit_file.close()

if __name__ == "__main__":
    main()