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
                
                models.append({
                    'name': model_name,
                    'url': model_url,
                    'description': description,
                    'pull_count': pull_count,
                    'tag_count': tag_count,
                    'updated': updated
                })
        
        logging.info(f"Found {len(models)} models")
        return models
    
    def get_model_details(self, model_url):
        """Get details for a specific model including all tags"""
        logging.info(f"Fetching details for model: {model_url}")
        response = self.session.get(model_url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        model_name = model_url.split('/')[-1]
        capabilities = []
        sizes = []
        
        # Look for capability tags
        capability_elements = soup.select('span[x-test-capability]')
        for element in capability_elements:
            capabilities.append(element.text.strip())
        
        # Look for size tags
        size_elements = soup.select('span[x-test-size]')
        for element in size_elements:
            sizes.append(element.text.strip())
        
        # Extract description if available on the detail page
        description_elem = soup.select_one('p.max-w-lg, div.description')
        description = description_elem.text.strip() if description_elem else ""
        
        # Extract other metadata if available
        pull_count_elem = soup.select_one('span[x-test-pull-count]')
        pull_count = pull_count_elem.text.strip() if pull_count_elem else ""
        
        updated_elem = soup.select_one('span[x-test-updated]')
        updated = updated_elem.text.strip() if updated_elem else ""
        
        return {
            'name': model_name,
            'url': model_url,
            'description': description,
            'capabilities': capabilities,
            'sizes': sizes,
            'pull_count': pull_count,
            'updated': updated
        }
    
    def scrape_all(self):
        """Main function to scrape all models and their tags"""
        all_models = self.get_all_models()
        detailed_models = []
        
        for model in all_models:
            try:
                model_details = self.get_model_details(model['url'])
                detailed_models.append(model_details)
                # Be respectful with requests
                time.sleep(self.delay)
            except Exception as e:
                logging.error(f"Error processing model {model['name']}: {str(e)}")
        
        return detailed_models

if __name__ == "__main__":
    scraper = OllamaScraper()
    results = scraper.scrape_all()
    
    # Save results to a JSON file
    with open('ollama_models.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Scraped {len(results)} models. Results saved to ollama_models.json")
