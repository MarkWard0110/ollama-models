#!/usr/bin/env python3

import json
import sys

def update_model_sizes(models):
    """
    Update each model's sizes array based on the parameter_sizes in the tags.
    
    For tags with null parameter_size, try to find a matching tag with the same hash
    that has a parameter_size and use that value.
    """
    for model in models:
        # Create a mapping from hash to parameter_size
        hash_to_param_size = {}
        for tag in model.get('tags', []):
            if isinstance(tag, dict) and 'hash' in tag and 'parameter_size' in tag and tag['parameter_size'] is not None:
                hash_to_param_size[tag['hash']] = tag['parameter_size']
        
        # Collect all unique parameter sizes and resolve null values where possible
        unique_sizes = set()
        for tag in model.get('tags', []):
            if isinstance(tag, dict) and 'parameter_size' in tag:
                if tag['parameter_size'] is not None:
                    unique_sizes.add(tag['parameter_size'])
                elif 'hash' in tag and tag['hash'] in hash_to_param_size:
                    # Use parameter_size from another tag with same hash
                    tag['parameter_size'] = hash_to_param_size[tag['hash']]
                    unique_sizes.add(tag['parameter_size'])
        
        # Update the model's sizes array with unique parameter sizes
        if unique_sizes:
            model['sizes'] = sorted(list(unique_sizes))
    
    return models

def main():
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <path_to_models_json>")
        sys.exit(1)
    
    json_path = sys.argv[1]
    
    try:
        with open(json_path, 'r') as f:
            models = json.load(f)
        
        updated_models = update_model_sizes(models)
        
        with open(json_path, 'w') as f:
            json.dump(updated_models, f, indent=2)
        
        print(f"Successfully updated sizes in {json_path}")
    
    except Exception as e:
        print(f"Error processing JSON file: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
