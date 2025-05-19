"""
Ollama model scraper for fetching model information from ollama.com
"""
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import re
import sys
from datetime import datetime, timedelta

logger = logging.getLogger("ollama_models.core.scraper")

class OllamaScraper:
    """
    Scraper for Ollama model information
    """
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
        logger.info("Fetching all models from search page...")
        response = self.session.get(self.SEARCH_URL)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        models = []
        # Select list items with x-test-model attribute (new website structure)
        model_elements = soup.select('li[x-test-model]')
        
        if not model_elements:
            logger.warning("No model elements found with x-test-model. Website structure may have changed.")
            # Try generic li elements instead
            model_elements = soup.select('li.flex.flex-col')
        
        for element in model_elements:
            link = element.select_one('a')
            if not link or not link.get('href'):
                continue
                
            href = link.get('href')
            if '/library/' in href:
                # Extract model name using x-test-search-response-title if available
                model_name_elem = element.select_one('span[x-test-search-response-title]')
                if model_name_elem:
                    model_name = model_name_elem.text.strip()
                else:
                    # Fallback to href parsing
                    model_name = href.split('/')[-1]
                
                model_url = self.BASE_URL + href
                
                # Extract description
                description_elem = element.select_one('p.max-w-lg')
                if not description_elem:
                    description_elem = element.select_one('p.break-words')
                if not description_elem:
                    description_elem = element.select_one('p')
                description = description_elem.text.strip() if description_elem else ""
                
                # Extract pull count using x-test-pull-count
                pull_count_elem = element.select_one('span[x-test-pull-count]')
                pull_count = pull_count_elem.text.strip() if pull_count_elem else ""
                
                # Extract tag count using x-test-tag-count
                tag_count_elem = element.select_one('span[x-test-tag-count]')
                tag_count = int(tag_count_elem.text.replace(',', '')) if tag_count_elem and tag_count_elem.text.replace(',', '').isdigit() else 0
                
                # Extract update time using x-test-updated
                updated_elem = element.select_one('span[x-test-updated]')
                updated = updated_elem.text.strip() if updated_elem else ""
                
                # Extract capabilities directly from the model listing
                capabilities = []
                capability_elems = element.select('span[x-test-capability]')
                for cap_elem in capability_elems:
                    cap_text = cap_elem.text.strip().lower()
                    if cap_text and cap_text not in capabilities:
                        capabilities.append(cap_text)
                
                # Extract sizes directly from the model listing
                sizes = []
                size_elems = element.select('span[x-test-size]')
                for size_elem in size_elems:
                    size_text = size_elem.text.strip().lower()
                    # Try to parse the size as a numeric value (remove 'b' from '7b', etc.)
                    size_match = re.search(r'(\d+(?:\.\d+)?)[bB]', size_text)
                    if size_match:
                        try:
                            param_size = float(size_match.group(1))
                            if param_size == int(param_size):
                                param_size = int(param_size)
                            sizes.append(param_size)
                        except ValueError:
                            pass
                  # Add model info
                model_info = {
                    "name": model_name,
                    "url": model_url,
                    "description": description,
                    "pull_count": pull_count,
                    "tag_count": tag_count,
                    "tags": [],
                    "capabilities": capabilities,
                    "sizes": sorted(list(set(sizes))) if sizes else []
                }
                
                # Convert relative date to timestamp
                updated_timestamp = self.convert_relative_date(updated)
                if updated_timestamp:
                    model_info["updated_timestamp"] = updated_timestamp
                models.append(model_info)
                
                # Be nice to the server
                time.sleep(self.delay)
                
        logger.info(f"Found {len(models)} models")
        return models
    
    def extract_param_size(self, tag_name):
        """Extract parameter size from tag name"""
        size_match = re.search(r'(\d+(?:\.\d+)?)[bB]', tag_name)
        if size_match:
            try:
                param_size = size_match.group(1)
                param_size = float(param_size)
                if param_size == int(param_size):
                    param_size = int(param_size)
                return param_size
            except ValueError:
                pass
        
        # Try common parameter sizes
        for param in ['1', '1.6', '3', '4', '7', '8', '10.7', '12', '13', '27', '30', '33', '34', '65', '70', '72']:
            if f"-{param}b" in tag_name.lower() or f":{param}b" in tag_name.lower() or tag_name.lower() == param + "b":
                try:
                    param_size = float(param)
                    if param_size == int(param_size):
                        param_size = int(param_size)
                    return param_size
                except ValueError:
                    pass
        
        return None
    
    def extract_quantization(self, tag_name):
        """Extract quantization from tag name"""
        quant_match = re.search(r'[qQ](\d+[_0-9A-Za-z]*)', tag_name)
        return quant_match.group(0).lower() if quant_match else None
    
    def convert_relative_date(self, date_text):
        """Convert relative date to formatted date string"""
        now = datetime.now()
        
        if not date_text:
            return None
            
        match = re.search(r'(\d+)\s+(day|month|year|hour|minute|second)s?\s+ago', date_text)
        if not match:
            return None
            
        value = int(match.group(1))
        unit = match.group(2).lower()
        
        if unit == 'minute':
            delta = now - timedelta(minutes=value)
        elif unit == 'hour':
            delta = now - timedelta(hours=value)
        elif unit == 'day':
            delta = now - timedelta(days=value)
        elif unit == 'month':
            delta = now - timedelta(days=value*30)  # Approximation
        elif unit == 'year':
            delta = now - timedelta(days=value*365)  # Approximation
        else:
            return None
            
        return delta.strftime("%Y-%m-%dT%H:%M:%S.%f")
        
    def get_model_tags(self, model_name, model_url):
        """Scrape tags for a specific model"""
        logger.info(f"Fetching tags for model: {model_name}")
        
        # Use specific tags URL
        tags_url = f"{self.BASE_URL}/library/{model_name}/tags"
        logger.info(f"Trying tags URL: {tags_url}")
        response = self.session.get(tags_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        tags = []
        
        # Find all tag elements using the current website structure (group.px-4.py-3)
        tag_elements = soup.select('div.group.px-4.py-3')
        if not tag_elements:
            # Fall back to older selectors
            tag_elements = soup.select('div.group.p-3')
        
        if tag_elements:
            logger.info(f"Found {len(tag_elements)} tag elements")
            
            for element in tag_elements:
                try:
                    # Try to get tag name from mobile view
                    mobile_view = element.select_one('a.md\\:hidden')
                    if mobile_view:
                        name_span = mobile_view.select_one('span.group-hover\\:underline')
                        if not name_span:
                            name_span = mobile_view.select_one('span')
                        
                        tag_name = name_span.text.strip() if name_span else ""
                        
                        # Get metadata from the mobile view
                        metadata_div = mobile_view.select_one('div.flex.flex-col.text-neutral-500')
                        if metadata_div:
                            metadata_text = metadata_div.text.strip()
                            
                            # Parse hash/commit
                            hash_match = re.search(r'([a-f0-9]{12})', metadata_text)
                            hash_value = hash_match.group(1) if hash_match else ""
                            
                            # Parse size
                            size_match = re.search(r'(\d+\.?\d*\s*[KMGT]B)', metadata_text, re.IGNORECASE)
                            size = size_match.group(1) if size_match else "Unknown"
                            
                            # Parse context window
                            context_match = re.search(r'(\d+)[Kk]\s*context\s*window', metadata_text)
                            context_window = f"{context_match.group(1)}K" if context_match else "Unknown"
                            
                            # Check if it has image input support
                            model_type = "text"  # Default
                            if "Image" in metadata_text:
                                model_type = "text+vision"
                                
                            # Parse last updated
                            updated_match = re.search(r'(\d+\s+(?:day|month|year|hour|minute)s?\s+ago)', metadata_text)
                            updated = updated_match.group(1) if updated_match else ""
                        else:
                            # If no metadata div found, try older structure
                            metadata_elem = mobile_view.select_one('div.flex.flex-col.text-neutral-500.text-\\[13px\\]')
                            if not metadata_elem:
                                metadata_elem = mobile_view.select_one('span')
                            
                            # Extract complete metadata text
                            metadata_text = metadata_elem.text.strip() if metadata_elem else ""
                            hash_value = ""
                            size_match = re.search(r'(\d+\.?\d*\s*[KMGT]B)', metadata_text, re.IGNORECASE)
                            size = size_match.group(1) if size_match else "Unknown"
                            context_window = "Unknown"
                            model_type = "text"
                            updated = ""
                    else:
                        # Try desktop layout
                        desktop_div = element.select_one('div.hidden.md\\:flex')
                        if not desktop_div:
                            # Older website structure
                            desktop_div = element.select_one('div.md\\:flex')
                            if not desktop_div:
                                continue
                            
                            # Get tag name from older layout
                            tag_link = desktop_div.select_one('a.group-hover\\:underline')
                            if not tag_link:
                                tag_link = desktop_div.select_one('a')
                            
                            if not tag_link:
                                continue
                                
                            tag_name = tag_link.text.strip()
                            hash_value = ""
                            updated = ""
                            
                            # Get other metadata from paragraphs
                            size_elem = desktop_div.select_one('p.col-span-2')
                            size = size_elem.text.strip() if size_elem else "Unknown"
                            context_window = "Unknown"
                            model_type = "text"
                        else:
                            # New desktop layout
                            tag_link = desktop_div.select_one('a')
                            tag_name = tag_link.text.strip() if tag_link else ""
                            
                            # Extract hash
                            hash_elem = desktop_div.select_one('span.font-mono')
                            hash_value = hash_elem.text.strip() if hash_elem else ""
                            
                            # Extract other info from paragraphs
                            paragraphs = desktop_div.select('p')
                            
                            # Initialize defaults
                            size = "Unknown"
                            context_window = "Unknown"
                            model_type = "text"
                            updated = ""
                            
                            # Parse size, context window, and capabilities from paragraphs
                            if len(paragraphs) >= 1:
                                size = paragraphs[0].text.strip()
                            
                            if len(paragraphs) >= 2:
                                context_match = re.search(r'(\d+)[Kk]', paragraphs[1].text)
                                if context_match:
                                    context_window = f"{context_match.group(1)}K"
                            
                            if len(paragraphs) >= 3 and "Image" in paragraphs[2].text:
                                model_type = "text+vision"
                                
                            # Get updated time
                            if len(paragraphs) >= 4:
                                updated = paragraphs[3].text.strip()
                            else:
                                # Try to find updated time from the last info element
                                updated_elem = desktop_div.select_one('div.flex.text-neutral-500.text-xs.items-center')
                                if updated_elem:
                                    updated_text = updated_elem.text.strip()
                                    updated_match = re.search(r'(\d+\s+(?:day|month|year|hour|minute)s?\s+ago)', updated_text)
                                    updated = updated_match.group(1) if updated_match else ""
                    
                    # Convert relative date to timestamp
                    updated_timestamp = self.convert_relative_date(updated)
                    
                    # Fix tag name to remove model prefix if present
                    if ':' in tag_name and tag_name.startswith(f"{model_name}:"):
                        tag_name = tag_name.split(':', 1)[1]
                      # Extract parameter size from tag name
                    param_size = self.extract_param_size(tag_name)
                    
                    # Extract quantization
                    quantization = self.extract_quantization(tag_name)
                    # Create tag info
                    tag_info = {
                        "name": tag_name,
                        "hash": hash_value,
                        "size": size,
                        "context_window": context_window,
                        "model_type": model_type,
                        "updated_timestamp": updated_timestamp,
                        "parameter_size": param_size
                    }
                    # Add quantization only if present
                    if quantization:
                        tag_info["quantization"] = quantization
                    
                    tags.append(tag_info)
                
                except Exception as e:
                    logger.error(f"Error parsing tag element for {model_name}: {str(e)}")
        
        # If no tags found with the div.group selector, try the table-based layout
        if not tags:
            tag_elements = soup.select('table tbody tr')
            logger.info(f"Found {len(tag_elements)} tag elements using table selector")
            
            for element in tag_elements:
                try:
                    # Skip header rows
                    if element.select_one('th'):
                        continue
                        
                    tag_cells = element.select('td')
                    if len(tag_cells) < 2:
                        continue
                        
                    # Extract tag name and size
                    tag_name_elem = tag_cells[0].select_one('code')
                    if not tag_name_elem:
                        tag_name_elem = tag_cells[0]
                    tag_name = tag_name_elem.text.strip() if tag_name_elem else ""
                    
                    size = "Unknown"
                    if len(tag_cells) > 1:
                        size_elem = tag_cells[1]
                        size = size_elem.text.strip() if size_elem else "Unknown"
                      # Extract parameter size from tag name
                    param_size = self.extract_param_size(tag_name)
                    
                    # Extract quantization
                    quantization = self.extract_quantization(tag_name)
                    # Add this tag to our list with additional fields
                    tag_info = {
                        "name": tag_name,
                        "hash": "",  # Not available in this layout
                        "size": size,
                        "context_window": "Unknown",  # Not available in this layout
                        "model_type": "text",  # Default to text
                        "updated_timestamp": None,
                        "parameter_size": param_size
                    }
                    
                    # Add quantization only if present
                    if quantization:
                        tag_info["quantization"] = quantization
                        
                    tags.append(tag_info)
                
                except Exception as e:
                    logger.error(f"Error parsing tag element for {model_name}: {str(e)}")
        
        # If we still don't have tags, try one more approach with general selectors
        if not tags:
            # Try to find any elements that might contain tag information
            logger.info("Trying fallback tag extraction method")
            
            # Look for tag name in span elements
            for span in soup.select('span'):
                if 'latest' in span.text or ':v' in span.text or model_name + ':' in span.text:
                    tag_name = span.text.strip()
                    
                    # Fix tag name to remove model prefix if present
                    if ':' in tag_name and tag_name.startswith(f"{model_name}:"):
                        tag_name = tag_name.split(':', 1)[1]
                      # Extract parameter size from tag name
                    param_size = self.extract_param_size(tag_name)
                    
                    # Extract quantization
                    quantization = self.extract_quantization(tag_name)
                    tag_info = {
                        "name": tag_name,
                        "hash": "",
                        "size": "Unknown",
                        "context_window": "Unknown",
                        "model_type": "text",
                        "updated_timestamp": None,
                        "parameter_size": param_size
                    }
                    
                    # Add quantization only if present
                    if quantization:
                        tag_info["quantization"] = quantization
                        
                    tags.append(tag_info)
          # Make sure we at least have a 'latest' tag if we found no tags
        if not tags:
            logger.warning(f"No tags found for {model_name}, creating a default 'latest' tag")
            tags.append({
                "name": "latest",
                "hash": "",
                "size": "Unknown",
                "context_window": "Unknown",
                "model_type": "text",
                "updated_timestamp": None,
                "parameter_size": None
            })
        
        logger.info(f"Found {len(tags)} tags for model {model_name}")
        return tags
        
    def process_models(self, models):
        """Get tags for all models and process them"""
        for model in models:
            model_name = model["name"]
            model_url = model["url"]
            
            # Get tags for this model
            tags = self.get_model_tags(model_name, model_url)
            model["tags"] = tags
            
            # Extract additional capabilities and sizes from tags
            self._extract_additional_capabilities_and_sizes(model)
            
            # Be nice to the server
            time.sleep(self.delay)
        
        # Sort the model sizes
        for model in models:
            model["sizes"] = sorted(model["sizes"])
        
        return models
    
    def _extract_additional_capabilities_and_sizes(self, model):
        """Extract additional capabilities and sizes from model tags"""
        capabilities = set(model["capabilities"])  # Start with existing capabilities
        sizes = set(model["sizes"])  # Start with existing sizes
        
        for tag in model["tags"]:
            # Add parameter size to sizes if available
            if tag["parameter_size"] is not None:
                sizes.add(tag["parameter_size"])
            
            # Check for capabilities in the tag name
            tag_name = tag["name"].lower()
            
            # Check for vision models
            if any(term in tag_name for term in ["vision", "multimodal", "mm"]) and "vision" not in capabilities:
                capabilities.add("vision")
            
            # Check for embedding models
            if "embed" in tag_name and "embedding" not in capabilities:
                capabilities.add("embedding")
            
            # Check for instruct models
            if "instruct" in tag_name and "instruct" not in capabilities:
                capabilities.add("instruct")
            
            # Check for chat models
            if any(term in tag_name for term in ["chat", "conv"]) and "chat" not in capabilities:
                capabilities.add("chat")
            
            # Check for tool-using models
            if any(term in tag_name for term in ["tool", "function"]) and "tools" not in capabilities:
                capabilities.add("tools")
            
            # Check model type
            if tag["model_type"] == "text+vision" and "vision" not in capabilities:
                capabilities.add("vision")
        
        model["capabilities"] = sorted(list(capabilities))
        model["sizes"] = sorted(list(sizes))
        return model

def scrape_and_save(output_file):
    """Scrape Ollama models and save to JSON file"""
    scraper = OllamaScraper()
    
    # Phase 1: Get all models
    models = scraper.get_all_models()
    
    # Phase 2: Process each model to get tags and additional info
    models = scraper.process_models(models)
    
    # Phase 3: Add timestamps to models
    now = datetime.now()
    for model in models:
        model["updated_timestamp"] = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
        
        # Ensure "updated" field is removed from models
        if "updated" in model:
            del model["updated"]
        
        # Add timestamps to tags and remove "updated" field
        for tag in model["tags"]:
            # If tag doesn't already have a timestamp, add one
            if "updated_timestamp" not in tag or tag["updated_timestamp"] is None:
                tag["updated_timestamp"] = now.strftime("%Y-%m-%dT%H:%M:%S.%f")
            
            # Remove "updated" field from tags
            if "updated" in tag:
                del tag["updated"]
    
    # Phase 4: Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(models, f, indent=2)
    
    logger.info(f"Saved {len(models)} models to {output_file}")
    return len(models)
