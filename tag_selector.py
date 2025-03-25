#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import json
import os
import curses
import sys

# Use an absolute path for the config file
CONFIG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selected_tags.conf")

def load_models(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)
    # Build a dict: { model_name: { param_size: [tags], ...}, ... }
    models_dict = {}
    for model_entry in data:
        name = model_entry.get("name")
        tags = model_entry.get("tags", [])
        sizes = {}
        # Store the model's capabilities and sizes from the JSON
        model_capabilities = model_entry.get("capabilities", [])
        model_sizes = model_entry.get("sizes", [])
        
        for tag in tags:
            size = tag.get("parameter_size")
            if size not in sizes:
                sizes[size] = []
            sizes[size].append({
                "name": tag.get("name", ""),
                "size": tag.get("size", "")
            })
        
        # Store the model with its sizes, tags, and metadata
        models_dict[name] = {
            "sizes_dict": sizes,
            "capabilities": model_capabilities,
            "size_list": model_sizes
        }
    return models_dict

def load_config():
    selected = set()
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        selected.add(line)
        except Exception as e:
            print(f"Error loading config: {e}")
    return selected

def save_config(selected):
    try:
        with open(CONFIG_FILE, "w") as f:
            for item in sorted(selected):
                f.write(f"{item}\n")
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def show_message(stdscr, message, timeout=1000):
    """Show a temporary message to the user"""
    height, width = stdscr.getmaxyx()
    y = height - 1
    # Clear the line
    stdscr.addstr(y, 0, " " * (width - 1))
    # Show the message
    stdscr.addstr(y, 0, message[:width-1])
    stdscr.refresh()
    curses.napms(timeout)  # Show message for specified time (milliseconds)

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
    changes_made = False
    
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Model: {model}, Size: {size}")
        height, width = stdscr.getmaxyx()
        max_viewable = height - 4  # Reserve some lines for instructions
        visible_tags = tags[start_idx : start_idx + max_viewable]
        for i, tag in enumerate(visible_tags):
            record = f"{model}:{tag['name']}"
            mark = "[X]" if record in selected else "[ ]"
            prefix = "> " if (start_idx + i) == idx else "  "
            stdscr.addstr(i + 2, 2, f"{prefix}{mark} {tag['name']} ({tag['size']})"[: width - 3])
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
        elif key in (curses.KEY_ENTER, 10, 13, ord(' ')):
            record = f"{model}:{tags[idx]['name']}"
            if record in selected:
                selected.remove(record)
            else:
                selected.add(record)
            changes_made = True
        elif key in (ord('s'), ord('S')):
            if save_config(selected):
                show_message(stdscr, "Configuration saved.")
            return
        elif key in (27, ord('q')):
            if changes_made:
                # Ask to save changes
                stdscr.clear()
                stdscr.addstr(0, 0, "Save changes? (y/n)")
                stdscr.refresh()
                while True:
                    resp = stdscr.getch()
                    if resp in (ord('y'), ord('Y')):
                        if save_config(selected):
                            show_message(stdscr, "Configuration saved.")
                        return
                    elif resp in (ord('n'), ord('N')):
                        return
            else:
                return

def interactive_view_config(stdscr, selected, models_data):
    temp_selected = set(selected)
    tags = sorted(temp_selected)
    changes_made = False
    
    def get_tag_size(m_data, model, tag_name):
        for size_key, tag_list in m_data[model].items():
            for t in tag_list:
                if t["name"] == tag_name:
                    return t["size"]
        return "N/A"

    idx = 0
    start_idx = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "View Current Config")
        height, width = stdscr.getmaxyx()
        max_viewable = height - 4
        visible_tags = tags[start_idx : start_idx + max_viewable]
        for i, tag in enumerate(visible_tags):
            model_tag = tag.split(":", 1)
            if len(model_tag) == 2:
                the_model, t_name = model_tag
                tag_size = get_tag_size(models_data, the_model, t_name)
            else:
                tag_size = "N/A"
            mark = "[X]" if tag in temp_selected else "[ ]"
            prefix = "> " if (start_idx + i) == idx else "  "
            stdscr.addstr(
                i + 2,
                2,
                f"{prefix}{mark} {tag} ({tag_size})"[: width - 3]
            )
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
        elif key in (curses.KEY_ENTER, 10, 13, ord(' ')):
            if tags[idx] in temp_selected:
                temp_selected.remove(tags[idx])
            else:
                temp_selected.add(tags[idx])
            changes_made = True
        elif key in (ord('s'), ord('S')):
            selected.clear()
            selected.update(temp_selected)
            if save_config(selected):
                show_message(stdscr, "Configuration saved.")
            return
        elif key in (27, ord('q')):
            if changes_made:
                # Ask to save changes
                stdscr.clear()
                stdscr.addstr(0, 0, "Save changes? (y/n)")
                stdscr.refresh()
                while True:
                    resp = stdscr.getch()
                    if resp in (ord('y'), ord('Y')):
                        selected.clear()
                        selected.update(temp_selected)
                        if save_config(selected):
                            show_message(stdscr, "Configuration saved.")
                        return
                    elif resp in (ord('n'), ord('N')):
                        return
            else:
                return

def get_model_info_display(model_name, model_data):
    """Create a display string with model information"""
    # Get the sizes and capabilities directly from the JSON data
    sizes_str = ",".join(model_data["size_list"]) if model_data["size_list"] else "unknown"
    capabilities_str = ",".join(model_data["capabilities"]) if model_data["capabilities"] else ""
    
    # Format the display string
    if capabilities_str:
        return f"{model_name} ({sizes_str}) ({capabilities_str})"
    else:
        return f"{model_name} ({sizes_str})"

def run(stdscr):
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    stdscr.attrset(curses.color_pair(1) | curses.A_BOLD)
    json_path = "ollama_models.json"
    models_data = load_models(json_path)
    selected = load_config()

    # Add a handler for program exit to save any pending changes
    def exit_handler():
        if 'selected' in locals() and selected:
            save_config(selected)
    
    import atexit
    atexit.register(exit_handler)

    while True:
        choice = interactive_menu_select(
            stdscr,
            "Main Menu:\nSelect an option (↑/↓, Enter, q=quit)",
            ["Select model", "Edit current config", "Quit"],
        )
        if choice == "Select model":
            while True:
                # Create display strings with additional information
                model_display_list = []
                model_name_map = {}  # Map display strings back to model names
                
                for model_name in sorted(models_data.keys()):
                    display_str = get_model_info_display(model_name, models_data[model_name])
                    model_display_list.append(display_str)
                    model_name_map[display_str] = model_name
                
                model_display = interactive_menu_select(stdscr, "\nSelect a model (↑/↓, Enter, q to quit):", model_display_list)
                if not model_display:
                    break
                    
                # Get the actual model name from our mapping
                model = model_name_map[model_display]
                sizes_dict = models_data[model]["sizes_dict"]
                
                while True:
                    size_list = sorted(s for s in sizes_dict.keys() if s)
                    title = f"\nSelect parameter size for {model}"
                    chosen_size = interactive_menu_select(stdscr, title, size_list)
                    if not chosen_size:
                        break
                    tags_list = sorted(sizes_dict[chosen_size], key=lambda x: x['name'])
                    interactive_toggle_tags(stdscr, model, chosen_size, tags_list, selected)
        elif choice == "Edit current config":
            # Update interactive_view_config call to work with new data structure
            interactive_view_config(stdscr, selected, {name: data["sizes_dict"] for name, data in models_data.items()})
        else:
            break

def main():
    curses.wrapper(run)

if __name__ == "__main__":
    main()
