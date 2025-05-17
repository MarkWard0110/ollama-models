import os
import csv
from model_utils import fetch_installed_models, fetch_max_context_size, try_model_call, fetch_memory_usage, format_size

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
    start = 2048
    if not fits_in_vram(model_name, start):
        print(f"{model_name} cannot fit in VRAM even at 2048.")
        return 0
    high = start
    while high <= max_ctx and fits_in_vram(model_name, high):
        high *= 2
    left, right = high // 2, min(high, max_ctx)
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
    fit_path = "max_context.csv"
    fit_models = set()
    fit_rows = []

    # Read existing fit data (skip header)
    if os.path.isfile(fit_path):
        with open(fit_path, "r", newline="") as ff:
            r = csv.reader(ff)
            next(r, None)
            for row in r:
                if len(row) >= 3:
                    fit_rows.append(row)
                if len(row) >= 1:
                    fit_models.add(row[0])

    models = fetch_installed_models()

    for m in models:
        name = m.get("name")
        if name in fit_models:
            print(f"Skipping {name}: already has a max_context entry.")
            continue
        print(f"Processing model: {name}")
        max_ctx = fetch_max_context_size(name)
        print(f"Maximum reported context size: {max_ctx}")
        best_fit = find_max_fit_in_vram(name, max_ctx)
        if best_fit >= 2048:
            is_model_max = (best_fit == max_ctx)
            print(f"Max context size fully in VRAM for {name} is {best_fit}")
            fit_rows.append([name, best_fit, is_model_max])
            fit_models.add(name)

    # Sort and write fit file
    fit_rows.sort(key=lambda row: row[0])
    with open(fit_path, 'w', newline="") as fit_file:
        fit_writer = csv.writer(fit_file)
        fit_writer.writerow(["model_name", "max_context_size", "is_model_max"])
        for row in fit_rows:
            fit_writer.writerow(row)

if __name__ == "__main__":
    main()
