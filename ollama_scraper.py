#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import requests
from bs4 import BeautifulSoup
import json
import time
import logging
import re

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class OllamaScraper:
    BASE_URL = "https://ollama.com"
    SEARCH_URL = f"{BASE_URL}/search"
    
    def __init__(self, delay=1):
        self.delay = delay  # Delay between requests to be respectful
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def get_all_models(self):
        """Scrape all models from the search page"""
        logging.info("Fetching all models from search page...")
        response = self.session.get(self.SEARCH_URL)
        with open('ollama_search.html', 'w') as f:
            f.write(response.text)
        
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
                    'updated': updated,
                    'capabilities': capabilities,
                    'sizes': sizes
                })
        
        logging.info(f"Found {len(models)} models")
        return models
    
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
        tag_elements = soup.select('div.flex.px-4.py-3')
        
        for element in tag_elements:
            tag_link = element.select_one('a')
            if not tag_link:
                continue
                
            tag_name_elem = tag_link.select_one('div.break-all')
            tag_name = tag_name_elem.text.strip() if tag_name_elem else ""
            
            # Extract metadata text which contains hash, size, and update time
            metadata_elem = element.select_one('div.flex.items-baseline')
            metadata_text = metadata_elem.text.strip() if metadata_elem else ""
            
            # Parse metadata text
            hash_match = re.search(r'([a-f0-9]+)', metadata_text)
            hash_value = hash_match.group(1) if hash_match else ""
            
            size_match = re.search(r'(\d+\.?\d*\s*[GM]B)', metadata_text)
            size = size_match.group(1) if size_match else ""
            
            update_match = re.search(r'((?:\d+\s+\w+\s+ago)|(?:\w+\s+\d+,\s+\d{4}))', metadata_text)
            updated = update_match.group(1) if update_match else ""
            
            tags.append({
                'name': tag_name,
                'hash': hash_value,
                'size': size,
                'updated': updated
            })
        
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
    
    def scrape_all(self):
        """Main function to scrape all models and their tags"""
        all_models = self.get_all_models()
        
        for i, model in enumerate(all_models):
            try:
                # Get tags for each model
                model_name = model['name']
                tags = self.get_model_tags(model_name)
                all_models[i]['tags'] = tags
                
                # Be respectful with requests
                time.sleep(self.delay)
            except Exception as e:
                logging.error(f"Error processing model {model['name']}: {str(e)}")
        
        return all_models

if __name__ == "__main__":
    scraper = OllamaScraper()
    results = scraper.scrape_all()
    
    # Save results to a JSON file
    with open('ollama_models.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Scraped {len(results)} models. Results saved to ollama_models.json")
