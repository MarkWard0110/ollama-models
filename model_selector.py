#!/usr/bin/env python3

import json
import os
import sys
from typing import Dict, List, Set, Tuple
import inquirer
from collections import defaultdict
import re

# Create a custom separator class if inquirer doesn't provide one
class CustomSeparator:
    def __init__(self, message):
        self.message = message

# Constants
JSON_FILE = 'ollama_models.json'
SELECTIONS_FILE = 'selected_models.txt'

# Determine if Separator exists in the inquirer package
try:
    from inquirer import Separator
except ImportError:
    # If not available, use our custom implementation
    Separator = CustomSeparator

class ModelSelector:
    def __init__(self):
        self.models = []
        self.previous_selections: Dict[str, Set[str]] = {}
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
        
    def select_model_tags_directly(self) -> List[Tuple[str, str]]:
        """Present a unified list of model:tag combinations for selection"""
        model_tags = self.get_all_model_tags()
        
        if not model_tags:
            print("No model tags found in the JSON file.")
            return []
        
        # Allow filtering by capabilities
        capabilities_set = set()
        for model in self.models:
            for cap in model.get('capabilities', []):
                if cap:  # Only add non-empty capabilities
                    capabilities_set.add(cap)
        
        filter_options = ["all"] + sorted(list(capabilities_set))
        
        # Add filtering by model size
        size_set = set()
        for model in self.models:
            for size in model.get('sizes', []):
                if size:  # Only add non-empty sizes
                    size_set.add(size)
        
        # Use a text separator instead of Separator class if unavailable
        if size_set:
            filter_options.append("------- Filter by Size -------")
            filter_options.extend([f"size:{size}" for size in sorted(list(size_set))])
        
        questions = [
            inquirer.List(
                'filter',
                message="Filter models (or show all)",
                choices=filter_options,
                default="all",
            )
        ]
        
        filter_answer = inquirer.prompt(questions)
        if not filter_answer:
            return []
            
        filter_choice = filter_answer['filter']
        
        # Group model tags by model name for better organization
        model_tag_groups = defaultdict(list)
        model_info = {}
        
        for model, tag, tag_name in model_tags:
            model_name = model.get('name', '')
            # Store model info for later use
            if model_name not in model_info:
                model_info[model_name] = model
            model_tag_groups[model_name].append((tag, tag_name))
        
        # Filter models based on selected capability or size
        if filter_choice != "all":
            filtered_model_names = []
            
            if filter_choice.startswith("size:"):
                size_filter = filter_choice.split(":", 1)[1]
                for model_name, model in model_info.items():
                    if size_filter in model.get('sizes', []):
                        filtered_model_names.append(model_name)
            else:
                # Filter by capability
                for model_name, model in model_info.items():
                    if filter_choice in model.get('capabilities', []):
                        filtered_model_names.append(model_name)
            
            # Only keep model tags for models that match the filter
            model_tag_groups = {k: v for k, v in model_tag_groups.items() if k in filtered_model_names}
        
        # Now create selection choices with separators for each model
        all_choices = []
        model_tag_mapping = {}
        
        # Sort model names for consistent display
        for model_name in sorted(model_tag_groups.keys()):
            model = model_info[model_name]
            
            # Add model info as a separator
            capabilities = model.get('capabilities', [])
            sizes = model.get('sizes', [])
            description = model.get('description', '')
            pull_count = model.get('pull_count', '')
            
            cap_str = f" [{', '.join(capabilities)}]" if capabilities else ""
            size_str = f" ({', '.join(sizes)})" if sizes else ""
            pull_str = f" - {pull_count} pulls" if pull_count else ""
            desc_str = f": {description[:50]}{'...' if len(description) > 50 else ''}" if description else ""
            
            header = f"{model_name}{cap_str}{size_str}{pull_str}{desc_str}"
            header_text = f"------ {header} ------"
            all_choices.append(Separator(header_text))
            
            # Add tags for this model, sorted by recency
            tags = model_tag_groups[model_name]
            tags.sort(key=lambda t: (
                0 if t[1] == 'latest' else 1,  # latest tag first
                0 if 'hours ago' in t[0].get('updated', '') else 
                1 if 'days ago' in t[0].get('updated', '') else 
                2 if 'weeks ago' in t[0].get('updated', '') else 
                3 if 'months ago' in t[0].get('updated', '') else 4,  # sort by recency
                t[1]  # alphabetically by tag name
            ))
            
            for tag, tag_name in tags:
                # Format tag info
                tag_size = tag.get('size', '')
                tag_updated = tag.get('updated', '')
                tag_hash = tag.get('hash', '')
                
                display_name = f"{model_name}:{tag_name}"
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
                
                # Mark previously selected combinations
                if model_name in self.previous_selections and tag_name in self.previous_selections[model_name]:
                    display_name = f"[*] {display_name}"
                
                all_choices.append(display_name)
                model_tag_mapping[display_name] = (model_name, tag_name)
        
        if not all_choices:
            print(f"No models match the filter: {filter_choice}")
            return []
            
        # Add search/filter feature
        questions = [
            inquirer.Text('search', message="Filter tags (leave empty to show all)", default="")
        ]
        
        search_answers = inquirer.prompt(questions)
        if not search_answers:
            return []
            
        search_term = search_answers['search'].lower().strip()
        
        # Filter choices if search term is provided
        filtered_choices = all_choices
        if search_term:
            filtered_choices = []
            including_separator = False
            current_model = None
            
            for choice in all_choices:
                is_separator = isinstance(choice, Separator) or isinstance(choice, CustomSeparator) or isinstance(choice, str) and choice.startswith("-----")
                
                if is_separator:
                    # Extract model name from separator
                    separator_text = choice.message if hasattr(choice, 'message') else choice
                    model_match = re.search(r"--+ ([^:]+)", separator_text)
                    if model_match:
                        current_model = model_match.group(1).strip()
                    including_separator = False
                    continue
                    
                if search_term in str(choice).lower():
                    if not including_separator and current_model:
                        # Add the separator for this model if we haven't already
                        for sep in all_choices:
                            is_sep = isinstance(sep, Separator) or isinstance(sep, CustomSeparator) or isinstance(sep, str) and sep.startswith("-----")
                            if is_sep and current_model in (sep.message if hasattr(sep, 'message') else sep):
                                filtered_choices.append(sep)
                                break
                        including_separator = True
                    filtered_choices.append(choice)
        
        if not filtered_choices or all(isinstance(c, Separator) or isinstance(c, CustomSeparator) or (isinstance(c, str) and c.startswith("-----")) for c in filtered_choices):
            print(f"No tags match the search term: {search_term}")
            return []
        
        selection_questions = [
            inquirer.Checkbox(
                'selected_model_tags',
                message=f"Select model tags (use spacebar to select/unselect)",
                choices=filtered_choices,
                default=[c for c in filtered_choices if isinstance(c, str) and c.startswith('[*]')]
            )
        ]
        
        answers = inquirer.prompt(selection_questions)
        if not answers:
            return []
        
        # Map selected display names back to model:tag pairs
        selected_display_names = answers['selected_model_tags']
        selected_tags = []
        
        for display_name in selected_display_names:
            if display_name in model_tag_mapping:
                selected_tags.append(model_tag_mapping[display_name])
        
        return selected_tags

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
        
        # Select model tags directly
        selected_tags = self.select_model_tags_directly()
            
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
            
        # Confirm and save selections
        questions = [
            inquirer.Confirm('save', message="Save these selections?", default=True)
        ]
        answers = inquirer.prompt(questions)
        
        if answers and answers['save']:
            self.save_selections(selected_tags)
        else:
            print("\nSelections not saved.")


if __name__ == "__main__":
    selector = ModelSelector()
    try:
        selector.run()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user. Exiting.")
        sys.exit(0)
