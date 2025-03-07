#!/usr/bin/env python3

import json
import os
import sys
from typing import Dict, List, Set, Tuple
import inquirer
from collections import defaultdict
import re

# Constants
JSON_FILE = 'ollama_models.json'
SELECTIONS_FILE = 'selected_models.txt'


class ModelSelector:
    def __init__(self):
        self.models = []
        self.previous_selections: Dict[str, Set[str]] = {}
        self.current_selections: Dict[str, List[str]] = {}
        self.load_models()
        self.load_previous_selections()

    def load_models(self):
        """Load models from the JSON file"""
        try:
            with open(JSON_FILE, 'r') as f:
                self.models = json.load(f)
            print(f"Loaded {len(self.models)} models from {JSON_FILE}")
        except FileNotFoundError:
            print(f"Error: {JSON_FILE} not found. Run the scraper first to generate the file.")
            sys.exit(1)
        except json.JSONDecodeError:
            print(f"Error: {JSON_FILE} is not valid JSON.")
            sys.exit(1)

    def load_previous_selections(self):
        """Load previous selections if available"""
        try:
            with open(SELECTIONS_FILE, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    try:
                        model, tag = line.split(':', 1)
                        if model not in self.previous_selections:
                            self.previous_selections[model] = set()
                        self.previous_selections[model].add(tag)
                    except ValueError:
                        # Skip lines that don't have the expected format
                        continue
            print(f"Loaded previous selections from {SELECTIONS_FILE}")
        except FileNotFoundError:
            print(f"No previous selections found. Will create {SELECTIONS_FILE} on save.")

    def get_all_model_tags(self) -> List[Tuple[Dict, Dict, str]]:
        """Get all model:tag combinations from all models"""
        all_model_tags = []
        
        for model in self.models:
            model_name = model.get('name', '')
            tags = model.get('tags', [])
            
            # If no tags or empty list, create a default "latest" tag
            if not tags:
                tags = [{"name": "latest", "updated": model.get("updated", "")}]
            
            for i, tag in enumerate(tags):
                # Extract or create a tag name
                tag_name = tag.get('name', '')
                
                # If the tag doesn't have a name but has an 'updated' field, create a synthetic tag name
                if not tag_name and 'updated' in tag:
                    # Create a tag name based on index for tags without names
                    if i == 0:
                        tag_name = "latest"
                    else:
                        tag_name = f"tag_{i}"
                    
                    # Add the name to the tag object
                    tag['name'] = tag_name
                
                if tag_name:  # Still check that we have a tag name
                    all_model_tags.append((model, tag, tag_name))
        
        return all_model_tags
        
    def browse_models(self) -> List[str]:
        """Browse and select models before viewing their tags"""
        try:
            # Apply capability/size filtering similar to the existing code
            capabilities_set = set()
            for model in self.models:
                for cap in model.get('capabilities', []):
                    if cap:
                        capabilities_set.add(cap)
            
            filter_options = ["all"] + sorted(list(capabilities_set))
            
            size_set = set()
            for model in self.models:
                for size in model.get('sizes', []):
                    if size:
                        size_set.add(size)
            
            filter_choices = filter_options[:]
            
            if size_set:
                filter_choices.append("---------- Filter by Size ----------")
                filter_choices.extend([f"size:{size}" for size in sorted(list(size_set))])
            
            questions = [
                inquirer.List(
                    'filter',
                    message="Filter models (or show all)",
                    choices=filter_choices,
                    default="all",
                )
            ]
            
            filter_answer = inquirer.prompt(questions)
            if not filter_answer:
                return []
                
            filter_choice = filter_answer['filter']
            
            # Skip the separator line if selected
            if filter_choice.startswith("-"):
                print("Please select a valid filter option.")
                return self.browse_models()
                
            # Search for models
            search_options = [
                inquirer.Text('search', message="Filter models by name (leave empty to show all)", default="")
            ]
            
            search_answers = inquirer.prompt(search_options)
            if not search_answers:
                return []
                
            search_term = search_answers['search'].lower().strip()
            
            # Filter and prepare models for display
            filtered_models = []
            
            for model in self.models:
                model_name = model.get('name', '')
                
                # Apply capability/size filter
                if filter_choice != "all":
                    if filter_choice.startswith("size:"):
                        size_filter = filter_choice.split(":", 1)[1]
                        if size_filter not in model.get('sizes', []):
                            continue
                    elif filter_choice not in model.get('capabilities', []):
                        continue
                
                # Apply name search filter
                if search_term and search_term not in model_name.lower():
                    continue
                    
                # Format model for display
                capabilities = model.get('capabilities', [])
                sizes = model.get('sizes', [])
                tag_count = model.get('tag_count', 0)
                pull_count = model.get('pull_count', '')
                description = model.get('description', '')
                
                cap_str = f"[{', '.join(capabilities)}]" if capabilities else ""
                size_str = f"({', '.join(sizes)})" if sizes else ""
                
                # Create display representation
                display = f"{model_name} {cap_str} {size_str}"
                if pull_count:
                    display += f" - {pull_count} pulls"
                if tag_count:
                    display += f" - {tag_count} tags"
                    
                # Add indicator for models that have selected tags
                if model_name in self.current_selections and self.current_selections[model_name]:
                    display = f"[*] {display}"
                    
                # Keep the model info with the display name
                filtered_models.append((display, model))
            
            # Sort filtered models alphabetically by name instead of by popularity
            filtered_models.sort(key=lambda item: item[1].get('name', '').lower())
            
            if not filtered_models:
                print(f"No models match the filter: {filter_choice} and search: {search_term}")
                input("Press Enter to try again...")
                return self.browse_models()
            
            # Add "Done" option at the end to finish selection
            filtered_models.append(("---------- DONE SELECTING MODELS ----------", None))
            
            # Present models for selection
            questions = [
                inquirer.List(
                    'model',
                    message="Select a model to view its tags (or DONE to finish)",
                    choices=[fm[0] for fm in filtered_models],
                )
            ]
            
            answer = inquirer.prompt(questions)
            if not answer:
                return []
            
            selected_display = answer['model']
            
            # Check if DONE was selected
            if selected_display.startswith("---------- DONE"):
                # Return all models that have selected tags
                return [model_name for model_name, tags in self.current_selections.items() if tags]
                
            # Find the selected model
            for display, model in filtered_models:
                if display == selected_display:
                    return self.browse_model_tags(model)
                    
            # Should not reach here
            return self.browse_models()
        
        except KeyboardInterrupt:
            print("\nOperation interrupted. Would you like to save your progress?")
            try:
                confirm = inquirer.prompt([
                    inquirer.Confirm('save_on_exit', 
                                    message="Save all current selections before exiting?", 
                                    default=True)
                ])
                
                if confirm and confirm.get('save_on_exit'):
                    # Convert current selections to the format needed for save_selections
                    selected_tags = []
                    for model_name, tags in self.current_selections.items():
                        for tag in tags:
                            selected_tags.append((model_name, tag))
                    
                    if selected_tags:
                        self.save_selections(selected_tags)
            except:
                pass  # Handle any further interruption by just exiting
            
            sys.exit(0)

    def browse_model_tags(self, model: Dict) -> List[str]:
        """Browse and select tags for a specific model"""
        model_name = model.get('name', '')
        tags = model.get('tags', [])
        
        while True:  # Add loop to allow repeated tag selection and back navigation
            # If no tags or empty list, create a default "latest" tag
            if not tags:
                tags = [{"name": "latest", "updated": model.get("updated", "")}]
            
            # Prepare tags with names
            named_tags = []
            for i, tag in enumerate(tags):
                tag_name = tag.get('name', '')
                
                # If the tag doesn't have a name but has an 'updated' field, create a synthetic tag name
                if not tag_name and 'updated' in tag:
                    if i == 0:
                        tag_name = "latest"
                    else:
                        tag_name = f"tag_{i}"
                    tag['name'] = tag_name
                    
                if tag_name:
                    named_tags.append((tag, tag_name))
                    
            # Sort tags by recency
            named_tags.sort(key=lambda t: (
                0 if t[1] == 'latest' else 1,  # latest tag first
                0 if 'hours ago' in t[0].get('updated', '') else 
                1 if 'days ago' in t[0].get('updated', '') else 
                2 if 'weeks ago' in t[0].get('updated', '') else 
                3 if 'months ago' in t[0].get('updated', '') else 4,  # sort by recency
                t[1]  # alphabetically by tag name
            ))
            
            # Format tags for display
            tag_choices = []
            tag_mapping = {}
            
            for tag, tag_name in named_tags:
                tag_size = tag.get('size', '')
                tag_updated = tag.get('updated', '')
                tag_hash = tag.get('hash', '')
                
                display_name = f"{tag_name}"
                details = []
                if tag_size:
                    details.append(tag_size)
                if tag_updated:
                    details.append(f"updated {tag_updated}")
                if tag_hash:
                    short_hash = tag_hash[:8] if len(tag_hash) > 8 else tag_hash
                    details.append(f"[{short_hash}]")
                    
                if details:
                    display_name += f" ({', '.join(details)})"
                    
                # Mark previously selected tags
                selected = model_name in self.current_selections and tag_name in self.current_selections[model_name]
                if selected:
                    display_name = f"[*] {display_name}"
                    
                tag_choices.append(display_name)
                tag_mapping[display_name] = tag_name
            
            # Display model info first
            capabilities = model.get('capabilities', [])
            sizes = model.get('sizes', [])
            description = model.get('description', '')
            pull_count = model.get('pull_count', '')
            
            print("\n" + "="*60)
            print(f"MODEL: {model_name}")
            if capabilities:
                print(f"Capabilities: {', '.join(capabilities)}")
            if sizes:
                print(f"Sizes: {', '.join(sizes)}")
            if pull_count:
                print(f"Popularity: {pull_count} pulls")
            if description:
                print(f"Description: {description}")
            print("="*60 + "\n")
            
            # First ask if they want to select tags or go back
            # This separates the navigation from the tag selection
            nav_question = [
                inquirer.List(
                    'action',
                    message=f"Select action for model: {model_name}",
                    choices=[
                        "Select tags",
                        "Back to model selection (ESC key also works)"
                    ],
                    default="Select tags"
                )
            ]
            
            try:
                nav_answer = inquirer.prompt(nav_question)
                if not nav_answer:
                    return self.browse_models()  # Handle ESC key
                
                if nav_answer['action'] == "Back to model selection (ESC key also works)":
                    return self.browse_models()
            except KeyboardInterrupt:
                # Handle Ctrl+C as back navigation too
                return self.browse_models()
            
            # If we get here, user chose to select tags
            # Present tags for multi-selection
            if not tag_choices:
                print("No tags available for this model.")
                input("Press Enter to go back...")
                return self.browse_models()

            tag_questions = [
                inquirer.Checkbox(
                    'selected_tags',
                    message=f"Select tags for {model_name} (use spacebar to select/unselect, ESC when done)",
                    choices=tag_choices,
                    default=[c for c in tag_choices if c.startswith('[*]')],
                    carousel=True  # Enable wrapping around the options
                )
            ]
            
            try:
                answers = inquirer.prompt(tag_questions)
                if not answers:  # User pressed ESC
                    # Ask user if they want to save selections before going back
                    confirm = inquirer.prompt([
                        inquirer.Confirm('save_before_back', 
                                        message="Save current tag selections before going back?", 
                                        default=True)
                    ])
                    
                    if confirm and confirm.get('save_before_back'):
                        # Keep existing selections for this model
                        pass
                    else:
                        # Clear selections for this model
                        if model_name in self.current_selections:
                            del self.current_selections[model_name]
                    
                    return self.browse_models()
                
                # Process the selected tags
                selected_displays = answers['selected_tags']
                selected_tags = []
                
                for display in selected_displays:
                    if display in tag_mapping:
                        selected_tags.append(tag_mapping[display])
                        
                # Update current selections
                if selected_tags:
                    self.current_selections[model_name] = selected_tags
                elif model_name in self.current_selections:
                    # Remove the model from selections if no tags selected
                    del self.current_selections[model_name]
                    
                # Ask if user wants to go back or continue selecting tags
                action_q = [
                    inquirer.List(
                        'next_action',
                        message="What would you like to do next?",
                        choices=[
                            "Save these tags and go back to model selection",
                            "Continue selecting tags for this model",
                        ],
                        default="Save these tags and go back to model selection"
                    )
                ]
                
                action_a = inquirer.prompt(action_q)
                if not action_a or action_a['next_action'] == "Save these tags and go back to model selection":
                    return self.browse_models()
                # Otherwise continue loop to select more tags
                    
            except KeyboardInterrupt:
                # Confirm if they want to save or discard changes on Ctrl+C
                print("\nOperation interrupted. Going back to model selection.")
                return self.browse_models()

    def save_selections(self, selections: List[Tuple[str, str]]):
        """Save selections to a file"""
        with open(SELECTIONS_FILE, 'w') as f:
            f.write("# Selected models and tags for Ollama\n")
            f.write("# Format: model:tag\n\n")
            
            # Group by model name for better organization
            selections_by_model = {}
            for model, tag in selections:
                if model not in selections_by_model:
                    selections_by_model[model] = []
                selections_by_model[model].append(tag)
                
            for model, tags in sorted(selections_by_model.items()):
                for tag in sorted(tags):
                    f.write(f"{model}:{tag}\n")
                    
        print(f"\nSaved {len(selections)} selections to {SELECTIONS_FILE}")

    def run(self):
        """Main method to run the model selection process"""
        print("\n=== OLLAMA MODEL SELECTOR ===\n")
        print("Keyboard shortcuts:")
        print("  ESC - Go back to previous menu or cancel")
        print("  Enter - Confirm selection")
        print("  Ctrl+C - Exit the program\n")
        
        # Initialize current selections from previous selections
        for model, tags in self.previous_selections.items():
            self.current_selections[model] = list(tags)
        
        while True:
            selected_model_names = self.browse_models()
            
            if not selected_model_names:
                print("\nNo models selected. Exiting without saving.")
                return
            
            # Convert current selections to the format needed for save_selections
            selected_tags = []
            for model_name, tags in self.current_selections.items():
                if model_name in selected_model_names:
                    for tag in tags:
                        selected_tags.append((model_name, tag))
                
            if not selected_tags:
                print("\nNo model tags selected. Exiting without saving.")
                return
                
            # Show full summary of selections
            print(f"\nSelected {len(selected_tags)} model:tag combinations:")
            selection_groups = defaultdict(list)
            for model, tag in selected_tags:
                selection_groups[model].append(tag)
            
            for model, tags in sorted(selection_groups.items()):
                print(f"  {model}:")
                for tag in sorted(tags):
                    print(f"    - {tag}")
                
            # Confirm and save selections with options to go back
            questions = [
                inquirer.List(
                    'action',
                    message="What would you like to do?",
                    choices=[
                        "Save selections and exit",
                        "Go back to model selection",
                        "Exit without saving"
                    ],
                    default="Save selections and exit"
                )
            ]
            answers = inquirer.prompt(questions)
            
            if not answers:
                return
                
            action = answers['action']
            
            if action == "Save selections and exit":
                self.save_selections(selected_tags)
                return
            elif action == "Go back to model selection":
                continue
            else:  # "Exit without saving"
                print("\nExiting without saving selections.")
                return


if __name__ == "__main__":
    selector = ModelSelector()
    try:
        selector.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting.")
        sys.exit(0)
