#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
import os

CONFIG_FILE = "selected_tags.conf"

def load_config():
    selected = set()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    selected.add(line)
    return selected

def sync_ollama():
    print("Loading selected models from config...")
    selected = load_config()
    print(f"Found {len(selected)} models in config.")

    print("Fetching currently installed models from Ollama...")
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=10)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("Error fetching Ollama tags:", e)
        return

    installed_models = set(m["name"] for m in data.get("models", []))
    print("Determining new and removed models...")
    new_models = selected - installed_models
    removed_models = installed_models - selected

    for model in sorted(selected):
        print(f"Pulling or updating model: {model}")
        try:
            requests.post("http://localhost:11434/api/pull", json={"model": model}, timeout=30)
        except Exception as e:
            print(f"Error pulling {model}:", e)

    for model in sorted(removed_models):
        print(f"Deleting model: {model}")
        try:
            requests.delete("http://localhost:11434/api/delete", json={"model": model}, timeout=10)
        except Exception as e:
            print(f"Error deleting {model}:", e)

    print("New models:", sorted(new_models))
    print("Deleted models:", sorted(removed_models))

def main():
    sync_ollama()

if __name__ == "__main__":
    main()
