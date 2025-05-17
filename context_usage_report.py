import os
import csv
from model_utils import fetch_installed_models, fetch_max_context_size, try_model_call, fetch_memory_usage, format_size

def main():
    usage_path = "context_usage.csv"
    usage_set = set()
    usage_rows = []

    # Read existing usage data (skip header)
    if os.path.isfile(usage_path):
        with open(usage_path, "r", newline="") as uf:
            r = csv.reader(uf)
            next(r, None)
            for row in r:
                if len(row) >= 3:
                    usage_rows.append(row)
                if len(row) >= 2:
                    usage_set.add((row[0], int(row[1])))

    models = fetch_installed_models()

    for m in models:
        name = m.get("name")
        max_ctx = fetch_max_context_size(name)
        print(f"Processing model: {name}")
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
                usage_rows.append([name, ctx, size_hr])
                usage_set.add((name, ctx))
            else:
                print(f"Failed chat/embed call for {name} at 2^n size {ctx}")
            ctx *= 2

    # Sort and write usage file
    usage_rows.sort(key=lambda row: row[0])
    with open(usage_path, 'w', newline="") as usage_file:
        usage_writer = csv.writer(usage_file)
        usage_writer.writerow(["model_name", "context_size", "memory_allocated"])
        for row in usage_rows:
            usage_writer.writerow(row)

if __name__ == "__main__":
    main()
