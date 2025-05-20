"""
Model tag selector functionality.
"""
import json
import os
import sys
import logging

try:
    import curses
except ImportError:
    print("Warning: Curses module not available. Interactive tag selection may not work properly.", file=sys.stderr)
    # Creating a mock curses module with minimal functionality to prevent crashes
    class MockCurses:
        def __getattr__(self, name):
            return lambda *args, **kwargs: None
    curses = MockCurses()

logger = logging.getLogger("ollama_models.core.tag_selector")

def load_models(json_path):
    """
    Load model data from JSON file.
    
    Args:
        json_path (str): Path to the JSON file
        
    Returns:
        dict: Dictionary of models and their tags
    """
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

def load_config(config_file):
    """
    Load selected tags from config file.
    
    Args:
        config_file (str): Path to the config file
        
    Returns:
        set: Set of selected tags
    """
    selected = set()
    if os.path.exists(config_file):
        try:
            with open(config_file, "r") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        selected.add(line)
        except Exception as e:
            logger.error(f"Error loading config: {e}")
    return selected

def save_config(selected, config_file):
    """
    Save selected tags to config file.
    
    Args:
        selected (set): Set of selected tags
        config_file (str): Path to the config file
        
    Returns:
        bool: True if successful, False otherwise
    """
    try:
        with open(config_file, "w") as f:
            for item in sorted(selected):
                f.write(f"{item}\n")
        return True
    except Exception as e:
        logger.error(f"Error saving config: {e}")
        return False

def show_message(stdscr, message, timeout=1000):
    """
    Show a temporary message to the user.
    
    Args:
        stdscr (window): Curses window
        message (str): Message to show
        timeout (int): Timeout in milliseconds
    """
    height, width = stdscr.getmaxyx()
    y = height - 1
    # Clear the line
    stdscr.addstr(y, 0, " " * (width - 1))
    # Show the message
    stdscr.addstr(y, 0, message[:width-1])
    stdscr.refresh()
    curses.napms(timeout)  # Show message for specified time (milliseconds)

def draw_menu(stdscr, title, items, current_idx, start_idx):
    """
    Draw a menu on the screen.
    
    Args:
        stdscr (window): Curses window
        title (str): Menu title
        items (list): Menu items
        current_idx (int): Index of the current item
        start_idx (int): Index of the first item to display
    """
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

def interactive_menu_select(stdscr, title, items, initial_idx=0):
    """
    Show an interactive menu for selection.
    
    Args:
        stdscr (window): Curses window
        title (str): Menu title
        items (list): Menu items
        initial_idx (int): Index to start selection at
    Returns:
        str or None: Selected item or None if cancelled
        int: The index of the selected/cancelled item
    """
    title = title.strip('\n')
    idx = initial_idx
    height, _ = stdscr.getmaxyx()
    max_viewable = height - 2
    # Center the selected item if possible
    start_idx = max(0, min(idx - max_viewable // 2, len(items) - max_viewable)) if len(items) > max_viewable else 0
    while True:
        draw_menu(stdscr, title, items, idx, start_idx)
        key = stdscr.getch()
        if key == curses.KEY_UP and idx > 0:
            idx -= 1
            if idx < start_idx:
                start_idx = max(0, start_idx - 1)
        elif key == curses.KEY_DOWN and idx < len(items) - 1:
            idx += 1
            if idx >= start_idx + max_viewable:
                start_idx = min(start_idx + 1, len(items) - max_viewable)
        elif key in (curses.KEY_ENTER, 10, 13):
            return items[idx], idx
        elif key in (27, ord('q')):
            return None, idx

def interactive_toggle_tags(stdscr, model, size, tags, selected):
    """
    Interactive menu for toggling tags.
    
    Args:
        stdscr (window): Curses window
        model (str): Model name
        size (str): Parameter size
        tags (list): List of tags
        selected (set): Set of selected tags
        
    Returns:
        tuple: (changes_made, selected, go_back)
    """
    idx = 0
    start_idx = 0
    changes_made = False
    
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, f"Model: {model}, Size: {size}")
        height, width = stdscr.getmaxyx()
        max_viewable = height - 4  # Reserve some lines for instructions
        
        # Show instructions
        stdscr.addstr(height-2, 0, "Space: toggle selection | Enter: done | q/esc: back")
        
        # Display visible slice from items
        visible_items = tags[start_idx : start_idx + max_viewable]
        for i, tag in enumerate(visible_items):
            row = i + 2
            tag_name = tag["name"]
            record = f"{model}:{tag_name}"
            is_selected = record in selected
            prefix = "[X]" if is_selected else "[ ]"
            cursor = ">" if (start_idx + i) == idx else " "
            display_str = f"{cursor} {prefix} {tag_name} ({tag['size']})"
            if (start_idx + i) == idx:
                stdscr.attron(curses.A_REVERSE)
                stdscr.addstr(row, 2, display_str[:width-3])
                stdscr.attroff(curses.A_REVERSE)
            else:
                stdscr.addstr(row, 2, display_str[:width-3])
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
        elif key == ord(' '):  # Toggle selection
            tag_name = tags[idx]["name"]
            record = f"{model}:{tag_name}"
            if record in selected:
                selected.remove(record)
            else:
                selected.add(record)
            changes_made = True
        elif key in (curses.KEY_ENTER, 10, 13):  # Enter key
            return changes_made, selected, False
        elif key in (27, ord('q')):  # Escape or q
            return changes_made, selected, True

def interactive_view_config(stdscr, selected, models_data, config_file):
    """
    Interactive menu for viewing and editing the current configuration.
    
    Args:
        stdscr (window): Curses window
        selected (set): Set of selected tags
        models_data (dict): Dictionary of models data
        config_file (str): Path to the config file
        
    Returns:
        tuple: (changes_made, selected)
    """
    temp_selected = set(selected)
    tags = sorted(selected)  # Always show all tags from the original config
    changes_made = False

    def get_tag_size(m_data, model, tag_name):
        for size_key, tag_list in m_data[model]["sizes_dict"].items():
            for t in tag_list:
                if t["name"] == tag_name:
                    return t["size"]
        return "N/A"

    idx = 0
    start_idx = 0
    while True:
        stdscr.clear()
        stdscr.addstr(0, 0, "Current Configuration")
        height, width = stdscr.getmaxyx()
        max_viewable = height - 4

        if not tags:
            stdscr.addstr(2, 2, "No tags selected.")
        else:
            visible_tags = tags[start_idx : start_idx + max_viewable]
            for i, tag in enumerate(visible_tags):
                model_tag = tag.split(":", 1)
                if len(model_tag) == 2:
                    the_model, t_name = model_tag
                    if the_model in models_data:
                        tag_size = get_tag_size(models_data, the_model, t_name)
                    else:
                        tag_size = "N/A"
                else:
                    tag_size = "N/A"

                mark = "[X]" if tag in temp_selected else "[ ]"
                prefix = "> " if (start_idx + i) == idx else "  "

                if (start_idx + i) == idx:
                    stdscr.attron(curses.A_REVERSE)
                    stdscr.addstr(i + 2, 2, f"{prefix}{mark} {tag} ({tag_size})"[: width - 3])
                    stdscr.attroff(curses.A_REVERSE)
                else:
                    stdscr.addstr(i + 2, 2, f"{prefix}{mark} {tag} ({tag_size})"[: width - 3])

        # Show instructions
        stdscr.addstr(height-2, 0, "Space: toggle selection | Enter/s: save & exit | q: quit")
        stdscr.refresh()

        key = stdscr.getch()

        if key == curses.KEY_UP and idx > 0 and tags:
            idx -= 1
            if idx < start_idx:
                start_idx -= 1
        elif key == curses.KEY_DOWN and idx < len(tags) - 1 and tags:
            idx += 1
            if idx >= start_idx + max_viewable:
                start_idx += 1
        elif key == ord(' ') and tags:  # Toggle selection
            tag_to_toggle = tags[idx]
            if tag_to_toggle in temp_selected:
                temp_selected.remove(tag_to_toggle)
            else:
                temp_selected.add(tag_to_toggle)
            changes_made = True
            # Do not update tags list here; keep all original tags visible
            if idx >= len(tags):
                idx = max(0, len(tags) - 1)
        elif key in (curses.KEY_ENTER, 10, 13, ord('s')):  # Enter or s key
            if changes_made:
                selected.clear()
                selected.update(temp_selected)
                if save_config(selected, config_file):
                    show_message(stdscr, f"Saved {len(selected)} selected tags to {config_file}")
            return changes_made, selected
        elif key in (27, ord('q')):  # Escape or q
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
                        if save_config(selected, config_file):
                            show_message(stdscr, f"Saved {len(selected)} selected tags to {config_file}")
                        return changes_made, selected
                    elif resp in (ord('n'), ord('N')):
                        return changes_made, selected
            else:
                return changes_made, selected

def run_selector(json_path, config_file):
    """
    Run the tag selector.
    
    Args:
        json_path (str): Path to the models JSON file
        config_file (str): Path to the config file
        
    Returns:
        int: 0 for success, 1 for error
    """
    try:
        # Load models and current selected tags
        models = load_models(json_path)
        selected = load_config(config_file)
        
        # Run the UI
        result = curses.wrapper(_tag_selector_ui, models, selected, config_file)
        return 0 if result else 1
    except Exception as e:
        logger.error(f"Error running tag selector: {e}", exc_info=True)
        return 1

def _tag_selector_ui(stdscr, models, selected, config_file):
    """
    Main UI function for the tag selector.
    
    Args:
        stdscr (window): Curses window
        models (dict): Dictionary of models
        selected (set): Set of selected tags
        config_file (str): Path to the config file
        
    Returns:
        bool: True if successful, False otherwise
    """
    # Setup curses
    curses.curs_set(0)  # Hide cursor
    stdscr.clear()
    
    # Main menu loop
    need_save = False
    while True:
        # Show main menu
        choice, _ = interactive_menu_select(
            stdscr,
            "Main Menu:\nSelect an option (↑/↓, Enter, q=quit)",
            ["Select model", "Edit current config", "Quit"],
        )
        
        if choice == "Select model":
            model_idx = 0
            while True:
                # Create display strings with additional information
                model_display_list = []
                model_name_map = {}  # Map display strings back to model names
                for model_name in sorted(models.keys()):
                    display_str = get_model_info_display(model_name, models[model_name])
                    model_display_list.append(display_str)
                    model_name_map[display_str] = model_name
                model_display, model_idx = interactive_menu_select(
                    stdscr, "\nSelect a model (↑/↓, Enter, q to quit):", model_display_list, model_idx
                )
                if not model_display:
                    break
                model = model_name_map[model_display]
                sizes_dict = models[model]["sizes_dict"]
                def parse_size(size_str):
                    if size_str is None:
                        return 0
                    if isinstance(size_str, str) and size_str.lower().endswith('b'):
                        size_str = size_str[:-1]
                    try:
                        return float(size_str)
                    except (ValueError, TypeError):
                        return 0
                size_list = sorted([s for s in sizes_dict.keys() if s is not None], key=parse_size)
                if not size_list:
                    show_message(stdscr, f"No sizes available for {model}")
                    continue
                size_labels = [f"{size}" if size is not None else "Unknown" for size in size_list]
                size_idx = 0
                while True:
                    title = f"\nSelect parameter size for {model}"
                    size_label, size_idx = interactive_menu_select(stdscr, title, size_labels, size_idx)
                    if size_label is None:
                        break  # Go back to model selection
                    size = size_list[size_idx]
                    tags = models[model]["sizes_dict"][size]
                    if not tags:
                        show_message(stdscr, f"No tags available for {model} at size {size}B")
                        continue
                    changes_made, selected, go_back = interactive_toggle_tags(stdscr, model, f"{size}B", tags, selected)
                    if changes_made:
                        need_save = True
                        save_config(selected, config_file)
                        show_message(stdscr, f"Saved {len(selected)} selected tags to {config_file}")
                    if go_back:
                        continue  # Go back to size selection for this model, restoring size_idx
                    break  # Exit size selection loop
    
        elif choice == "Edit current config":
            changes_made, selected = interactive_view_config(stdscr, selected, models, config_file)
            if changes_made:
                need_save = True
                
        elif choice is None or choice == "Quit":
            break
    
    # Save final selections if needed
    if need_save:
        save_config(selected, config_file)
    
    return True

def get_model_info_display(model_name, model_data):
    """
    Create a display string with model information.
    
    Args:
        model_name (str): Name of the model
        model_data (dict): Model data
        
    Returns:
        str: String with model information
    """
    # Get the sizes and capabilities directly from the model data
    sizes_str = ",".join(map(str, model_data["size_list"])) if model_data["size_list"] else "unknown"
    capabilities_str = ",".join(model_data["capabilities"]) if model_data["capabilities"] else ""
    
    # Format the display string
    if capabilities_str:
        return f"{model_name} ({sizes_str}) ({capabilities_str})"
    else:
        return f"{model_name} ({sizes_str})"
