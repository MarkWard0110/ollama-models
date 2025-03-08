#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import curses

CONFIG_FILE = "selected_tags.conf"

def load_models(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Build a dict: { model_name: { param_size: [tags], ...}, ... }
    models_dict = {}
    for model_entry in data:
        name = model_entry.get("name")
        tags = model_entry.get("tags", [])
        sizes = {}
        for tag in tags:
            size = tag.get("parameter_size")
            if size not in sizes:
                sizes[size] = []
            sizes[size].append(tag.get("name", ""))
        models_dict[name] = sizes
    return models_dict

def load_config():
    selected = set()
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    selected.add(line)
    return selected

def save_config(selected):
    with open(CONFIG_FILE, "w") as f:
        for item in sorted(selected):
            f.write(f"{item}\n")
    print("Configuration saved.")

def draw_menu(stdscr, title, items, current_idx, start_idx):
    stdscr.clear()
    title = title.strip('\n')
    stdscr.addstr(0, 0, title)
    height, width = stdscr.getmaxyx()
    # Reserve two lines for title/instructions
    max_viewable = height - 2
    # Display visible slice from items
    visible_items = items[start_idx : start_idx + max_viewable]
    for i, item in enumerate(visible_items):
        row = i + 2
        display_str = f"> {item}" if (start_idx + i) == current_idx else f"  {item}"
        stdscr.addstr(row, 2, display_str[: width - 3])
    stdscr.refresh()

def interactive_menu_select(stdscr, title, items):
    title = title.strip('\n')
    idx = 0
    start_idx = 0
    while True:
        draw_menu(stdscr, title, items, idx, start_idx)
        key = stdscr.getch()
        if key == curses.KEY_UP and idx > 0:
            idx -= 1
            if idx < start_idx:
                start_idx -= 1
        elif key == curses.KEY_DOWN and idx < len(items) - 1:
            idx += 1
            height, _ = stdscr.getmaxyx()
            max_viewable = height - 2
            if idx >= start_idx + max_viewable:
                start_idx += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            return items[idx]
        elif key in (27, ord('q')):
            return None

def interactive_toggle_tags(stdscr, model, size, tags, selected):
    idx = 0
    start_idx = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Model: {model}, Size: {size}")
        height, width = stdscr.getmaxyx()
        max_viewable = height - 4  # Reserve some lines for instructions
        visible_tags = tags[start_idx : start_idx + max_viewable]
        for i, tag in enumerate(visible_tags):
            record = f"{model}:{tag}"
            mark = "[X]" if record in selected else "[ ]"
            prefix = "> " if (start_idx + i) == idx else "  "
            stdscr.addstr(i + 2, 2, f"{prefix}{mark} {tag}"[: width - 3])
        stdscr.addstr(len(visible_tags) + 3, 2, "[Enter] Toggle | [s] Save & Back | [q] Back w/o Save")
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and idx > 0:
            idx -= 1
            if idx < start_idx:
                start_idx -= 1
        elif key == curses.KEY_DOWN and idx < len(tags) - 1:
            idx += 1
            if idx >= start_idx + max_viewable:
                start_idx += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            record = f"{model}:{tags[idx]}"
            if record in selected:
                selected.remove(record)
            else:
                selected.add(record)
        elif key in (ord('s'), ord('S')):
            save_config(selected)
            return
        elif key in (27, ord('q')):
            return

def interactive_view_config(stdscr, selected):
    temp_selected = set(selected)
    tags = sorted(temp_selected)
    idx = 0
    start_idx = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "View Current Config")
        height, width = stdscr.getmaxyx()
        max_viewable = height - 4
        visible_tags = tags[start_idx : start_idx + max_viewable]
        for i, tag in enumerate(visible_tags):
            mark = "[X]" if tag in temp_selected else "[ ]"
            prefix = "> " if (start_idx + i) == idx else "  "
            stdscr.addstr(i + 2, 2, f"{prefix}{mark} {tag}"[: width - 3])
        stdscr.addstr(len(visible_tags) + 3, 2, "[Enter] Unselect | [s] Save & Back | [q] Back w/o Save")
        stdscr.refresh()

        key = stdscr.getch()
        if key == curses.KEY_UP and idx > 0:
            idx -= 1
            if idx < start_idx:
                start_idx -= 1
        elif key == curses.KEY_DOWN and idx < len(tags) - 1:
            idx += 1
            if idx >= start_idx + max_viewable:
                start_idx += 1
        elif key in (curses.KEY_ENTER, 10, 13):
            if tags[idx] in temp_selected:
                temp_selected.remove(tags[idx])
            else:
                temp_selected.add(tags[idx])
        elif key in (ord('s'), ord('S')):
            selected.clear()
            selected.update(temp_selected)
            save_config(selected)
            return
        elif key in (27, ord('q')):
            return

def run(stdscr):
    json_path = "ollama_models.json"
    models_data = load_models(json_path)
    selected = load_config()

    while True:
        choice = interactive_menu_select(
            stdscr,
            "Main Menu:\nSelect an option (↑/↓, Enter, q=quit)",
            ["Modify config", "View current config", "Quit"],
        )
        if choice == "Modify config":
            while True:
                model_list = sorted(models_data.keys())
                model = interactive_menu_select(stdscr, "\nSelect a model (↑/↓, Enter, q to quit):", model_list)
                if not model:
                    break
                sizes_dict = models_data[model]

                while True:
                    size_list = sorted(s for s in sizes_dict.keys() if s)
                    title = f"\nSelect parameter size for {model}"
                    chosen_size = interactive_menu_select(stdscr, title, size_list)
                    if not chosen_size:
                        break
                    tags_list = sorted(sizes_dict[chosen_size])
                    interactive_toggle_tags(stdscr, model, chosen_size, tags_list, selected)
        elif choice == "View current config":
            interactive_view_config(stdscr, selected)
        else:
            break

def main():
    curses.wrapper(run)

if __name__ == "__main__":
    main()
