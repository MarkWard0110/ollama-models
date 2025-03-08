#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests

CONFIG_FILE = "selected_tags.conf"

def main():
    # Fetch tags from Ollama API
    response = requests.get("http://localhost:11434/api/tags")
    response.raise_for_status()
    data = response.json()
    model_data = data.get("models", [])

    models = []
    for item in model_data:
        if isinstance(item, dict) and item.get("model"):
            models.append(item["model"])

    # Update selected_tags.conf with retrieved models
    with open(CONFIG_FILE, "w") as f:
        for model in sorted(set(models)):
            f.write(f"{model}\n")

if __name__ == "__main__":
    main()
