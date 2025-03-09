#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
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

def main():
    usage_path = "context_usage.csv"
    fit_path = "max_context.csv"
    usage_set, fit_models = read_existing_data(usage_path, fit_path)

    # We'll collect old rows plus new rows to re-write them at the end
    old_usage_rows = []
    old_fit_rows = []
    # read them again for rewriting
    if os.path.isfile(usage_path):
        with open(usage_path, "r", newline="") as uf:
            r = csv.reader(uf)
            header = next(r, None)
            for row in r:
                old_usage_rows.append(row)
    if os.path.isfile(fit_path):
        with open(fit_path, "r", newline="") as ff:
            r = csv.reader(ff)
            header = next(r, None)
            for row in r:
                old_fit_rows.append(row)

    models = fetch_installed_models()
    new_usage_rows = []
    new_fit_rows = []

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
                print(f"Measured at 2^n = {ctx}, total allocated: {size}, VRAM: {size_vram}")
                new_usage_rows.append([name, ctx, size])
                usage_set.add((name, ctx))
            else:
                print(f"Failed chat/embed call for {name} at 2^n size {ctx}")
            ctx *= 2

        best_fit = find_max_fit_in_vram(name, max_ctx)
        if best_fit >= 2048:
            print(f"Max context size fully in VRAM for {name} is {best_fit}")
            new_fit_rows.append([name, best_fit])
            fit_models.add(name)

    # Re-write the usage file with old rows + new rows
    with open(usage_path, "w", newline="") as uf:
        w = csv.writer(uf)
        w.writerow(["model_name", "context_size", "memory_allocated"])
        for row in old_usage_rows:
            w.writerow(row)
        for row in new_usage_rows:
            w.writerow(row)

    # Re-write the fit file with old rows + new rows
    with open(fit_path, "w", newline="") as ff:
        w = csv.writer(ff)
        w.writerow(["model_name", "max_context_size"])
        for row in old_fit_rows:
            w.writerow(row)
        for row in new_fit_rows:
            w.writerow(row)

if __name__ == "__main__":
    main()
