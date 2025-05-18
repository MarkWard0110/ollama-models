#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import re
import sys
from datetime import datetime, timedelta

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OllamaScraper:
    BASE_URL = "https://ollama.com"
    SEARCH_URL = f"{BASE_URL}/search"
    
    def __init__(self, delay=0.001):
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_all_models(self):
        """Scrape all models from the search page"""
        logging.info("Fetching all models from search page...")
        response = self.session.get(self.SEARCH_URL)
        # with open('ollama_search.html', 'w') as f:
        #     f.write(response.text)
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        models = []
        # Select list items with x-test-model attribute
        model_elements = soup.select('li[x-test-model]')
        
        for element in model_elements:
            link = element.select_one('a')
            if not link or not link.get('href'):
                continue
                
            href = link.get('href')
            if '/library/' in href:
                model_name = href.split('/')[-1]
                model_url = self.BASE_URL + href
                
                # Extract description
                description_elem = element.select_one('p.max-w-lg')
                description = description_elem.text.strip() if description_elem else ""
                
                # Extract pull count
                pull_count_elem = element.select_one('span[x-test-pull-count]')
                pull_count = pull_count_elem.text.strip() if pull_count_elem else ""
                
                # Extract tag count
                tag_count_elem = element.select_one('span[x-test-tag-count]')
                tag_count = int(tag_count_elem.text) if tag_count_elem and tag_count_elem.text.isdigit() else 0
                
                # Extract update time
                updated_elem = element.select_one('span[x-test-updated]')
                updated = updated_elem.text.strip() if updated_elem else ""
                updated_timestamp = self.convert_relative_time_to_timestamp(updated)
                
                # Extract capabilities - using find_all instead of select to ensure we get all elements
                capabilities = []
                for span in element.find_all('span'):
                    if span.has_attr('x-test-capability'):
                        cap_text = span.get_text(strip=True)
                        if cap_text:
                            capabilities.append(cap_text)
                
                # Extract sizes - using find_all instead of select to be consistent
                sizes = []
                for span in element.find_all('span'):
                    if span.has_attr('x-test-size'):
                        size_text = span.get_text(strip=True)
                        if size_text:
                            sizes.append(size_text)
                
                models.append({
                    'name': model_name,
                    'url': model_url,
                    'description': description,
                    'pull_count': pull_count,
                    'tag_count': tag_count,
                    'updated_timestamp': updated_timestamp,
                    'capabilities': capabilities,
                    'sizes': sizes
                })
        
        logging.info(f"Found {len(models)} models")
        return models
    
    def convert_relative_time_to_timestamp(self, relative_time):
        """Convert relative time strings to absolute timestamps"""
        if not relative_time:
            return None
        
        now = datetime.now()
        
        if relative_time.lower() == 'yesterday':
            return (now - timedelta(days=1)).isoformat()
        
        if relative_time.lower() == 'today':
            return now.isoformat()
        
        # Parse patterns like "2 weeks ago", "3 months ago", etc.
        match = re.match(r'(\d+)\s+(\w+)\s+ago', relative_time)
        if match:
            value = int(match.group(1))
            unit = match.group(2).lower()
        
            if unit == 'week' or unit == 'weeks':
                return (now - timedelta(weeks=value)).isoformat()
            elif unit == 'day' or unit == 'days':
                return (now - timedelta(days=value)).isoformat()
            elif unit == 'hour' or unit == 'hours':
                return (now - timedelta(hours=value)).isoformat()
            elif unit == 'minute' or unit == 'minutes':
                return (now - timedelta(minutes=value)).isoformat()
            elif unit == 'month' or unit == 'months':
                # Approximate a month as 30 days
                return (now - timedelta(days=30 * value)).isoformat()
            elif unit == 'year' or unit == 'years':
                # Approximate a year as 365 days
                return (now - timedelta(days=365 * value)).isoformat()
        
        # If nothing matches, return the current timestamp
        logging.warning(f"Unknown relative time format: {relative_time}")
        return now.isoformat()
    
    def get_model_tags(self, model_name):
        """Get tags for a specific model"""
        tags_url = f"{self.BASE_URL}/library/{model_name}/tags"
        logging.info(f"Fetching tags for model: {model_name} from {tags_url}")
        
        response = self.session.get(tags_url)
        if response.status_code != 200:
            logging.error(f"Failed to fetch tags for {model_name}: {response.status_code}")
            return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        
        tags = []
        
        # New website structure uses div.group elements for tags instead of li.group
        tag_elements = soup.select('div.group.p-3')
        
        # If no tags found, log it as a potential structure change
        if not tag_elements:
            logging.error(f"No tags found for {model_name}. HTML structure might have changed again.")
            # Save the HTML for debugging if needed
            with open(f"{model_name}_tags_debug.html", "w", encoding="utf-8") as f:
                f.write(response.text)
            return []
            
        logging.info(f"Found {len(tag_elements)} potential tags for {model_name}")
        
        successful_extracts = 0
        
        for element in tag_elements:
            try:
                # The website has different layouts for mobile and desktop
                # Try the mobile layout first (it's usually simpler)
                tag_link = element.select_one('a.md\\:hidden')
                
                if tag_link:
                    # Mobile layout
                    tag_name_elem = tag_link.select_one('span.group-hover\\:underline')
                    if not tag_name_elem:
                        continue
                        
                    tag_name = tag_name_elem.text.strip()
                    
                    # Get metadata from the mobile view
                    metadata_elem = tag_link.select_one('div.flex.flex-col.text-neutral-500.text-\\[13px\\]')
                    if not metadata_elem:
                        metadata_elem = tag_link.select_one('span')
                    
                    # Find the hash value (in a span with font-mono class)
                    hash_elem = metadata_elem.select_one('span.font-mono') if metadata_elem else None
                    hash_value = hash_elem.text.strip() if hash_elem else None
                    
                    # Extract complete metadata text
                    metadata_text = metadata_elem.text.strip() if metadata_elem else ""
                else:
                    # Desktop layout (hidden on mobile, visible on larger screens)
                    desktop_div = element.select_one('div.hidden.md\\:flex')
                    if not desktop_div:
                        continue
                        
                    # Get tag name from desktop layout
                    tag_link = desktop_div.select_one('a.group-hover\\:underline')
                    if not tag_link:
                        continue
                        
                    tag_name = tag_link.text.strip()
                    
                    # Get hash from the font-mono element
                    hash_elem = desktop_div.select_one('span.font-mono')
                    hash_value = hash_elem.text.strip() if hash_elem else None
                    
                    # Get other metadata from paragraphs
                    size_elem = desktop_div.select_one('p.col-span-2:nth-of-type(1)')
                    size = size_elem.text.strip() if size_elem else None
                    
                    context_elem = desktop_div.select_one('p.col-span-2:nth-of-type(2)')
                    context_window = context_elem.text.strip() if context_elem else None
                    
                    # Get the updated time
                    time_elem = desktop_div.select_one('div.flex.text-neutral-500.text-xs')
                    metadata_text = time_elem.text.strip() if time_elem else ""
                
                # Fix tag name to remove model prefix if present
                if ':' in tag_name and tag_name.startswith(f"{model_name}:"):
                    tag_name = tag_name.split(':', 1)[1]
                
                # Parse metadata text for size if not already extracted
                if 'size' not in locals() or not size:
                    size_match = re.search(r'(\d+\.?\d*\s*[KMGT]B)', metadata_text, re.IGNORECASE)
                    size = size_match.group(1) if size_match else None
                
                # Parse context window size if not already extracted
                if 'context_window' not in locals() or not context_window:
                    context_match = re.search(r'(\d+[kK])\s*context', metadata_text, re.IGNORECASE)
                    context_window = context_match.group(1) if context_match else None
                
                # Extract update time
                update_match = re.search(r'((?:\d+\s+\w+\s+ago)|(?:yesterday)|(?:today)|(?:\w+\s+\d+,\s+\d{4}))', metadata_text, re.IGNORECASE)
                updated = update_match.group(1) if update_match else ""
                updated_timestamp = self.convert_relative_time_to_timestamp(updated)
                
                # Extract parameter size from tag name (like 7b, 13b, 32b, etc.)
                param_match = re.search(r'(\d+\.?\d*[bBmMtTkK])', tag_name)
                parameter_size = param_match.group(1).lower() if param_match else None
                
                # Extract model type (Text, Vision, etc.)
                type_match = re.search(r'(Text|Vision|Audio|Multimodal|Video)', metadata_text, re.IGNORECASE)
                model_type = type_match.group(1) if type_match else None
                
                tag_data = {
                    'name': tag_name,
                    'hash': hash_value,
                    'size': size,
                    'context_window': context_window,
                    'model_type': model_type,
                    'updated_timestamp': updated_timestamp,
                    'parameter_size': parameter_size,
                }
                
                # Validate we have at least the crucial information
                if tag_name and hash_value:
                    tags.append(tag_data)
                    successful_extracts += 1
                else:
                    logging.warning(f"Incomplete tag data for {model_name}: {tag_data}")
                    
            except Exception as e:
                logging.error(f"Error parsing tag element for {model_name}: {str(e)}")
        
        logging.info(f"Successfully extracted {successful_extracts} tags for {model_name} out of {len(tag_elements)} elements")
        
        # If no tags were successfully extracted but elements were found, it might indicate a parsing issue
        if successful_extracts == 0 and len(tag_elements) > 0:
            logging.error(f"Failed to extract any tags for {model_name} despite finding {len(tag_elements)} potential elements. HTML structure might have changed.")
            with open(f"{model_name}_tags_debug.html", "w", encoding="utf-8") as f:
                f.write(response.text)
                
        return tags
        
        return tags

    def get_model_details(self, model_url):
        """Get additional details for a specific model"""
        logging.info(f"Fetching details for model: {model_url}")
        response = self.session.get(model_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        model_name = model_url.split('/')[-1]
        
        # Extract any additional information from the detail page
        # Details already extracted from search page don't need to be extracted again
        
        # For example, you might want to extract additional metadata specific to the detail page
        readme_elem = soup.select_one('div.readme-content')
        readme_content = readme_elem.text.strip() if readme_elem else ""
        
        return {
            'name': model_name,
            'url': model_url,
            'readme': readme_content
        }
    
    def update_model_sizes(self, models):
        """
        Update each model's sizes array based on the parameter_sizes in the tags.
        
        For tags with null parameter_size, try to find a matching tag with the same hash
        that has a parameter_size and use that value.
        """
        logging.info("Updating model sizes based on parameter sizes in tags...")
        
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
        
        logging.info(f"Updated sizes for {len(models)} models")
        return models
    
    def scrape_all(self):
        """Main function to scrape all models and their tags"""
        try:
            all_models = self.get_all_models()
            logging.info(f"Found {len(all_models)} models")
            
            if not all_models:
                logging.error("Failed to get any models. The search page structure might have changed.")
                return []
                
            models_with_tags = 0
            total_tags = 0
            
            for i, model in enumerate(all_models):
                try:
                    # Get tags for each model
                    model_name = model['name']
                    logging.info(f"Processing model {i+1}/{len(all_models)}: {model_name}")
                    
                    tags = self.get_model_tags(model_name)
                    all_models[i]['tags'] = tags
                    
                    if tags:
                        models_with_tags += 1
                        total_tags += len(tags)
                        logging.info(f"Found {len(tags)} tags for {model_name}")
                    else:
                        logging.warning(f"No tags found for {model_name}")
                    
                    # Be respectful with requests
                    time.sleep(self.delay)
                except Exception as e:
                    logging.error(f"Error processing model {model['name']}: {str(e)}")
                    # Continue with the next model instead of failing completely
                    all_models[i]['tags'] = []
            
            # Validate we have at least some successful tag extractions
            if models_with_tags == 0:
                logging.error("Failed to get tags for any models. The tags page structure might have changed.")
            else:
                logging.info(f"Successfully extracted tags for {models_with_tags}/{len(all_models)} models (total {total_tags} tags)")
            
            # Process and update model sizes based on tag information
            all_models = self.update_model_sizes(all_models)
            
            return all_models
        except Exception as e:
            logging.error(f"Critical error in scrape_all: {str(e)}")
            import traceback
            logging.error(traceback.format_exc())
            return []

if __name__ == "__main__":
    try:
        # Set up better logging
        log_file = 'ollama_scraper.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logging.getLogger().addHandler(file_handler)
        
        # Allow setting delay via command line args
        delay = 0.001
        if len(sys.argv) > 2:
            try:
                delay = float(sys.argv[2])
                logging.info(f"Using custom delay between requests: {delay}s")
            except ValueError:
                logging.warning(f"Invalid delay value: {sys.argv[2]}, using default: {delay}s")
        
        logging.info("Starting Ollama scraper")
        scraper = OllamaScraper(delay=delay)
        results = scraper.scrape_all()
        
        if not results:
            logging.error("No results obtained. Check error messages above.")
            sys.exit(1)
            
        # Count models and tags for validation
        model_count = len(results)
        tag_count = sum(len(model.get('tags', [])) for model in results)
        
        # Save results to a JSON file
        output_file = 'ollama_models.json'
        
        # Allow specifying a different output file via command line
        if len(sys.argv) > 1:
            output_file = sys.argv[1]
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Validate the results
        if model_count == 0 or tag_count == 0:
            logging.error(f"Warning: Scraped {model_count} models with {tag_count} total tags. This might indicate a parsing failure.")
            print(f"Warning: Scraped {model_count} models with {tag_count} total tags. Results saved to {output_file}, but they may be incomplete.")
        else:
            logging.info(f"Scraped {model_count} models with {tag_count} total tags. Results saved to {output_file}")
            print(f"Scraped {model_count} models with {tag_count} total tags. Results saved to {output_file}")
            
    except Exception as e:
        logging.error(f"Critical error: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        print(f"Error: {str(e)}. See {log_file} for details.")
