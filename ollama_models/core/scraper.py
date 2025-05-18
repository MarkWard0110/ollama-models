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
        # Select list items with x-test-model attribute
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
                model_name = href.split('/')[-1]
                model_url = self.BASE_URL + href
                
                # Extract description
                description_elem = element.select_one('p.max-w-lg')
                if not description_elem:
                    description_elem = element.select_one('p')
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
                
                # Add capabilities and size arrays (to be filled later)
                model_info = {
                    "name": model_name,
                    "url": model_url,
                    "description": description,
                    "pull_count": pull_count,
                    "tag_count": tag_count,
                    "updated": updated,
                    "tags": [],
                    "capabilities": [],
                    "sizes": []
                }
                
                models.append(model_info)
                
                # Be nice to the server
                time.sleep(self.delay)
                
        logger.info(f"Found {len(models)} models")
        return models

    def get_model_tags(self, model_name, model_url):
        """Scrape tags for a specific model"""
        logger.info(f"Fetching tags for model: {model_name}")
        
        # First try the direct model URL
        response = self.session.get(model_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Try specific tags URL if needed
        if not soup.select('div.group.p-3') and not soup.select('table tbody tr'):
            tags_url = f"{self.BASE_URL}/library/{model_name}/tags"
            logger.info(f"Trying alternative tags URL: {tags_url}")
            response = self.session.get(tags_url)
            soup = BeautifulSoup(response.text, 'html.parser')
        
        tags = []
        
        # First try the current website structure (div.group elements)
        tag_elements = soup.select('div.group.p-3')
        
        if tag_elements:
            logger.info(f"Found {len(tag_elements)} tag elements using div.group selector")
            
            for element in tag_elements:
                try:
                    # The website has different layouts for mobile and desktop
                    # Try the mobile layout first (it's usually simpler)
                    tag_link = element.select_one('a.md\\:hidden')
                    
                    if tag_link:
                        # Mobile layout
                        tag_name_elem = tag_link.select_one('span.group-hover\\:underline')
                        if not tag_name_elem:
                            tag_name_elem = tag_link.select_one('span')
                        
                        if not tag_name_elem:
                            continue
                            
                        tag_name = tag_name_elem.text.strip()
                        
                        # Get metadata from the mobile view
                        metadata_elem = tag_link.select_one('div.flex.flex-col.text-neutral-500.text-\\[13px\\]')
                        if not metadata_elem:
                            metadata_elem = tag_link.select_one('span')
                        
                        # Extract complete metadata text
                        metadata_text = metadata_elem.text.strip() if metadata_elem else ""
                        
                        # Parse size from metadata text
                        size_match = re.search(r'(\d+\.?\d*\s*[KMGT]B)', metadata_text, re.IGNORECASE)
                        size = size_match.group(1) if size_match else "Unknown"
                    else:
                        # Desktop layout (hidden on mobile, visible on larger screens)
                        desktop_div = element.select_one('div.hidden.md\\:flex')
                        if not desktop_div:
                            continue
                            
                        # Get tag name from desktop layout
                        tag_link = desktop_div.select_one('a.group-hover\\:underline')
                        if not tag_link:
                            tag_link = desktop_div.select_one('a')
                        
                        if not tag_link:
                            continue
                            
                        tag_name = tag_link.text.strip()
                        
                        # Get other metadata from paragraphs
                        size_elem = desktop_div.select_one('p.col-span-2')
                        size = size_elem.text.strip() if size_elem else "Unknown"
                    
                    # Fix tag name to remove model prefix if present
                    if ':' in tag_name and tag_name.startswith(f"{model_name}:"):
                        tag_name = tag_name.split(':', 1)[1]
                    
                    # Extract parameter size from tag name (like 7b, 13b, 32b, etc.)
                    param_size = None
                    size_match = re.search(r'(\d+(?:\.\d+)?)[bB]', tag_name)
                    if size_match:
                        param_size = size_match.group(1)
                        try:
                            param_size = float(param_size)
                            if param_size == int(param_size):
                                param_size = int(param_size)
                        except ValueError:
                            pass
                    
                    # Extract quantization (e.g., q4_0, q4_K_M, Q8_0)
                    quant_match = re.search(r'[qQ](\d+[_0-9A-Za-z]*)', tag_name)
                    quantization = quant_match.group(0).lower() if quant_match else None
                    
                    # Add this tag to our list
                    tag_info = {
                        "name": tag_name,
                        "size": size,
                        "parameter_size": param_size,
                        "quantization": quantization
                    }
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
                    
                    # Extract parameter size from tag name (like 7b, 13b, 32b, etc.)
                    param_size = None
                    size_match = re.search(r'(\d+(?:\.\d+)?)[bB]', tag_name)
                    if size_match:
                        param_size = size_match.group(1)
                        try:
                            param_size = float(param_size)
                            if param_size == int(param_size):
                                param_size = int(param_size)
                        except ValueError:
                            pass
                    
                    # Extract quantization (e.g., q4_0, q4_K_M, Q8_0)
                    quant_match = re.search(r'[qQ](\d+[_0-9A-Za-z]*)', tag_name)
                    quantization = quant_match.group(0).lower() if quant_match else None
                    
                    # For models like solar-10.7b, llama2-70b etc.
                    if not param_size:
                        for param in ['1.6', '3', '7', '8', '10.7', '13', '30', '33', '34', '65', '70', '72']:
                            if f"-{param}b" in tag_name.lower():
                                try:
                                    param_size = float(param)
                                    if param_size == int(param_size):
                                        param_size = int(param_size)
                                except ValueError:
                                    pass
                                break
                    
                    # Add this tag to our list
                    tag_info = {
                        "name": tag_name,
                        "size": size,
                        "parameter_size": param_size,
                        "quantization": quantization
                    }
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
                    
                    # Extract parameter size from tag name
                    param_size = None
                    size_match = re.search(r'(\d+(?:\.\d+)?)[bB]', tag_name)
                    if size_match:
                        param_size = size_match.group(1)
                        try:
                            param_size = float(param_size)
                            if param_size == int(param_size):
                                param_size = int(param_size)
                        except ValueError:
                            pass
                    
                    # Fix tag name to remove model prefix if present
                    if ':' in tag_name and tag_name.startswith(f"{model_name}:"):
                        tag_name = tag_name.split(':', 1)[1]
                    
                    tag_info = {
                        "name": tag_name,
                        "size": "Unknown",
                        "parameter_size": param_size,
                        "quantization": None
                    }
                    tags.append(tag_info)
        
        # Make sure we at least have a 'latest' tag if we found no tags
        if not tags:
            logger.warning(f"No tags found for {model_name}, creating a default 'latest' tag")
            tags.append({
                "name": "latest",
                "size": "Unknown",
                "parameter_size": None,
                "quantization": None
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
            
            # Extract capabilities and sizes
            self._extract_capabilities_and_sizes(model)
            
            # Be nice to the server
            time.sleep(self.delay)
        
        # Sort the model sizes
        for model in models:
            model["sizes"] = sorted(model["sizes"])
        
        return models
    
    def _extract_capabilities_and_sizes(self, model):
        """Extract capabilities and sizes from model tags"""
        capabilities = set()
        sizes = set()
        
        for tag in model["tags"]:
            # Add parameter size to sizes if available
            if tag["parameter_size"] is not None:
                sizes.add(tag["parameter_size"])
            
            # Check for capabilities in the tag name
            tag_name = tag["name"].lower()
            
            # Check for vision models
            if any(term in tag_name for term in ["vision", "multimodal", "mm"]):
                capabilities.add("vision")
            
            # Check for embedding models
            if "embed" in tag_name:
                capabilities.add("embeddings")
            
            # Check for instruct models
            if "instruct" in tag_name:
                capabilities.add("instruct")
            
            # Check for chat models
            if any(term in tag_name for term in ["chat", "conv"]):
                capabilities.add("chat")
        
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
    
    # Phase 3: Save to JSON file
    with open(output_file, 'w') as f:
        json.dump(models, f, indent=2)
    
    logger.info(f"Saved {len(models)} models to {output_file}")
    return len(models)
