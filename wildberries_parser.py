import requests
import json
import os
import time
import random
from bs4 import BeautifulSoup
import pandas as pd
import cv2
import numpy as np
from PIL import Image
from colorthief import ColorThief
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
import re
import urllib.parse
import urllib.request

# Check if Tesseract OCR is installed
tesseract_available = True
try:
    import pytesseract
    pytesseract.get_tesseract_version()
except Exception:
    tesseract_available = False
    print("Warning: Tesseract OCR is not installed or configured properly.")
    print("Text extraction from images will be disabled.")
    print("Install Tesseract OCR to enable this feature.")
    print("See setup.py for installation instructions.")
    print()

class WildberriesParser:
    def __init__(self, category, min_rating=4.5, min_reviews=50, max_items=100):
        self.category = category
        self.min_rating = min_rating
        self.min_reviews = min_reviews
        self.max_items = max_items
        self.results = []
        self.setup_driver()
        self.create_output_dirs()
        
    def setup_driver(self):
        """Setup Selenium WebDriver"""
        chrome_options = Options()
        chrome_options.add_argument("--headless")  # Run in headless mode
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument("--disable-gpu")
        chrome_options.add_argument("--window-size=1920,1080")
        chrome_options.add_argument("--disable-extensions")
        chrome_options.add_argument("--disable-notifications")
        chrome_options.add_argument("--disable-infobars")
        chrome_options.add_argument("--disable-popup-blocking")
        chrome_options.add_argument("--disable-blink-features=AutomationControlled")
        chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
        chrome_options.add_experimental_option("useAutomationExtension", False)
        
        # Check common Chrome installation paths on Windows
        import os
        chrome_paths = [
            os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Google\\Chrome\\Application\\chrome.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Google\\Chrome\\Application\\chrome.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google\\Chrome\\Application\\chrome.exe')
        ]
        
        chrome_found = False
        for path in chrome_paths:
            print(f"Checking Chrome path: {path}")
            if os.path.exists(path):
                chrome_options.binary_location = path
                print(f"Found Chrome at: {path}")
                chrome_found = True
                break
        
        if not chrome_found:
            print("Warning: Chrome binary not found in common locations. Attempting to proceed anyway.")
        
        try:
            self.driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=chrome_options
            )
            self.driver.execute_cdp_cmd("Network.setUserAgentOverride", {
                "userAgent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            })
            self.driver.execute_script("Object.defineProperty(navigator, 'webdriver', {get: () => undefined})")
        except Exception as e:
            print(f"Error setting up Chrome WebDriver: {e}")
            print("Please make sure Chrome is installed and up to date.")
            raise
        
    def create_output_dirs(self):
        """Create output directories for data and images"""
        # Create main category directory
        self.category_dir = os.path.join(os.getcwd(), self.category.replace(' ', '_'))
        if not os.path.exists(self.category_dir):
            os.makedirs(self.category_dir)
    
    def search_category(self):
        """Search for products in the specified category"""
        print(f"Searching for top products in category: {self.category}")
        
        try:
            # Encode the category for URL
            encoded_category = urllib.parse.quote(self.category)
            search_url = f"https://www.wildberries.ru/catalog/0/search.aspx?sort=popular&search={encoded_category}"
            print(f"Navigating to URL: {search_url}")
            
            # Navigate to the search URL
            try:
                self.driver.get(search_url)
                print("Successfully navigated to search URL")
            except Exception as e:
                print(f"Error navigating to search URL: {str(e)}")
                raise
            
            # Wait for the product cards to load
            print("Waiting for product cards to load...")
            try:
                # Save initial page source for debugging
                print("Saving initial page source...")
                try:
                    with open("initial_page_source.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print("Saved initial page source to initial_page_source.html")
                except Exception as save_err:
                    print(f"Error saving initial page source: {str(save_err)}")
                
                # Print page title for debugging
                print(f"Page title: {self.driver.title}")
                
                # Check if we're facing a CAPTCHA challenge
                if "Почти готово" in self.driver.title or "captcha" in self.driver.page_source.lower():
                    print("\n" + "=" * 80)
                    print("CAPTCHA CHALLENGE DETECTED")
                    print("=" * 80)
                    print("The website is showing a CAPTCHA challenge that requires manual intervention.")
                    print("Please follow these steps:")
                    print("1. The browser window should be open. If not, check your task manager.")
                    print("2. Manually solve the CAPTCHA in the browser window.")
                    print("3. After solving the CAPTCHA, the page should load normally.")
                    print("4. Once the page is loaded, the script will continue automatically.")
                    print("=" * 80)
                    
                    # Wait for the user to solve the CAPTCHA
                    # We'll check every 5 seconds if the CAPTCHA is solved
                    max_wait_time = 300  # 5 minutes
                    wait_interval = 5
                    elapsed_time = 0
                    
                    while elapsed_time < max_wait_time:
                        if not ("Почти готово" in self.driver.title or "captcha" in self.driver.page_source.lower()):
                            print("CAPTCHA appears to be solved. Continuing...")
                            break
                        
                        print(f"Waiting for CAPTCHA to be solved... ({elapsed_time} seconds elapsed)")
                        time.sleep(wait_interval)
                        elapsed_time += wait_interval
                    
                    if elapsed_time >= max_wait_time:
                        print("Timeout waiting for CAPTCHA to be solved. Aborting.")
                        return []
                
                # Define multiple selectors to try for product cards
                card_selectors = [
                    (By.CLASS_NAME, "product-card"),
                    (By.CSS_SELECTOR, ".product-card"),
                    (By.CSS_SELECTOR, ".product-card__wrapper"),
                    (By.CSS_SELECTOR, ".catalog-product-card"),
                    (By.CSS_SELECTOR, ".product-item"),
                    (By.CSS_SELECTOR, ".product"),
                    (By.CSS_SELECTOR, ".card"),
                    (By.CSS_SELECTOR, ".j-card"),
                    (By.CSS_SELECTOR, ".card-item"),
                    (By.CSS_SELECTOR, "[data-card-index]"),
                    (By.CSS_SELECTOR, ".catalog-page__content .product-card")
                ]
                
                # Try each selector until one works
                product_cards = []
                for selector_type, selector in card_selectors:
                    try:
                        print(f"Looking for product cards with selector: {selector}")
                        WebDriverWait(self.driver, 10).until(
                            EC.presence_of_element_located((selector_type, selector))
                        )
                        product_cards = self.driver.find_elements(selector_type, selector)
                        if product_cards:
                            print(f"Found {len(product_cards)} product cards with selector '{selector}'")
                            break
                    except Exception as selector_err:
                        print(f"Selector '{selector}' failed: {str(selector_err)}")
                
                if not product_cards:
                    # Take a screenshot for debugging
                    try:
                        self.driver.save_screenshot("error_screenshot.png")
                        print("Saved screenshot to error_screenshot.png for debugging")
                    except Exception as screenshot_err:
                        print(f"Error saving screenshot: {str(screenshot_err)}")
                    
                    # Save page source for debugging
                    try:
                        with open("page_source.html", "w", encoding="utf-8") as f:
                            f.write(self.driver.page_source)
                        print("Saved page source to page_source.html for debugging")
                    except Exception as save_err:
                        print(f"Error saving page source: {str(save_err)}")
                    
                    raise Exception("Could not find product cards with any selector")
            except Exception as e:
                print(f"Error waiting for product cards: {str(e)}")
                # Save page source for debugging
                try:
                    with open("page_source.html", "w", encoding="utf-8") as f:
                        f.write(self.driver.page_source)
                    print("Saved page source to page_source.html for debugging")
                except Exception as save_err:
                    print(f"Error saving page source: {str(save_err)}")
                
                # Take a screenshot for debugging
                try:
                    self.driver.save_screenshot("error_screenshot.png")
                    print("Saved screenshot to error_screenshot.png for debugging")
                except Exception as screenshot_err:
                    print(f"Error saving screenshot: {str(screenshot_err)}")
                
                raise
            
            # Scroll down to load more products
            print("Scrolling page to load more products...")
            self._scroll_page()
            
            # Get product links
            print("Extracting product links...")
            # Try multiple selectors for finding product cards again after scrolling
            product_cards = []
            for selector_type, selector in card_selectors:
                try:
                    cards = self.driver.find_elements(selector_type, selector)
                    if cards:
                        product_cards = cards
                        print(f"Found {len(product_cards)} product cards with selector '{selector}' after scrolling")
                        break
                except:
                    pass
            
            if not product_cards:
                print("Could not find product cards after scrolling")
                return []
            
            print(f"Found {len(product_cards)} product cards on page")
            
            # Try multiple selectors for finding product links
            link_selectors = [
                "a.product-card__main",
                "a.j-card-link",
                "a.card-link",
                "a.product-card__link",
                "a[href*='/catalog/']",
                "a"
            ]
            
            product_links = []
            
            for i, card in enumerate(product_cards[:min(self.max_items * 2, len(product_cards))]):
                link_found = False
                for link_selector in link_selectors:
                    try:
                        link = card.find_element(By.CSS_SELECTOR, link_selector).get_attribute("href")
                        if link and "/catalog/" in link:
                            product_links.append(link)
                            link_found = True
                            if i < 3:  # Print first few links for debugging
                                print(f"Sample product link {i+1}: {link}")
                            break
                    except:
                        continue
                
                if not link_found:
                    print(f"Could not extract link from card {i+1} with any selector")
            
            print(f"Successfully extracted {len(product_links)} product links")
            return product_links
            
        except Exception as e:
            print(f"Error in search_category: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def _scroll_page(self):
        """Scroll the page to load more products"""
        last_height = self.driver.execute_script("return document.body.scrollHeight")
        
        for _ in range(5):  # Scroll 5 times to load more products
            self.driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)  # Wait for page to load
            
            new_height = self.driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
    
    def parse_product(self, url):
        """Parse a single product page"""
        print(f"Parsing product: {url}")
        self.driver.get(url)
        
        # Wait for the product page to load
        WebDriverWait(self.driver, 10).until(
            EC.presence_of_element_located((By.CLASS_NAME, "product-page"))
        )
        
        try:
            # Get product name
            name = self.driver.find_element(By.CLASS_NAME, "product-page__title").text
            
            # Get product rating
            try:
                rating_text = self.driver.find_element(By.CLASS_NAME, "product-review__rating").text
                rating = float(rating_text.replace(',', '.'))
            except:
                rating = 0
            
            # Get number of reviews
            try:
                reviews_text = self.driver.find_element(By.CLASS_NAME, "product-review__count").text
                reviews = int(re.sub(r'\D', '', reviews_text))
            except:
                reviews = 0
            
            # Check if product meets minimum criteria
            if rating < self.min_rating or reviews < self.min_reviews:
                print(f"Product doesn't meet criteria. Rating: {rating}, Reviews: {reviews}")
                return None
            
            # Get product price
            try:
                price_text = self.driver.find_element(By.CLASS_NAME, "price-block__final-price").text
                price = int(re.sub(r'\D', '', price_text))
            except:
                price = 0
            
            # Get product brand
            try:
                brand = self.driver.find_element(By.CLASS_NAME, "brand-logo__text").text
            except:
                brand = "Unknown"
            
            # Get product category
            try:
                breadcrumbs = self.driver.find_elements(By.CSS_SELECTOR, ".breadcrumbs__item")
                category_path = " > ".join([b.text for b in breadcrumbs])
            except:
                category_path = self.category
            
            # Get product images
            image_urls = []
            try:
                image_elements = self.driver.find_elements(By.CSS_SELECTOR, ".swiper-slide img")
                for img in image_elements:
                    img_url = img.get_attribute("src")
                    if img_url and "https://" in img_url:
                        # Convert to high-res image URL
                        img_url = re.sub(r'/[0-9]+x[0-9]+/', '/big/', img_url)
                        image_urls.append(img_url)
            except Exception as e:
                print(f"Error getting images: {e}")
            
            # Get product characteristics
            characteristics = {}
            try:
                chars_container = self.driver.find_element(By.CLASS_NAME, "product-params")
                char_rows = chars_container.find_elements(By.CSS_SELECTOR, ".product-params__row")
                
                for row in char_rows:
                    try:
                        key = row.find_element(By.CSS_SELECTOR, ".product-params__cell-title").text
                        value = row.find_element(By.CSS_SELECTOR, ".product-params__cell-value").text
                        characteristics[key] = value
                    except:
                        pass
            except:
                pass
            
            # Get product description
            try:
                description = self.driver.find_element(By.CLASS_NAME, "product-detail__description").text
            except:
                description = ""
            
            # Extract key phrases from description
            key_phrases = self._extract_key_phrases(description)
            
            # Create product directory
            product_dir = os.path.join(self.category_dir, self._sanitize_filename(name))
            if not os.path.exists(product_dir):
                os.makedirs(product_dir)
            
            # Download and analyze images
            image_analysis = self._analyze_images(image_urls, product_dir)
            
            # Create product data
            product_data = {
                "name": name,
                "category": category_path,
                "brand": brand,
                "price": price,
                "rating": rating,
                "reviews": reviews,
                "image_count": len(image_urls),
                "image_urls": image_urls,
                "characteristics": characteristics,
                "description": description,
                "key_phrases": key_phrases,
                "image_analysis": image_analysis,
                "product_url": url
            }
            
            return product_data
            
        except Exception as e:
            print(f"Error parsing product: {e}")
            return None
    
    def _sanitize_filename(self, filename):
        """Sanitize filename to be valid for file system"""
        # Replace invalid characters with underscore
        return re.sub(r'[\\/*?:"<>|]', "_", filename)
    
    def _extract_key_phrases(self, text):
        """Extract key phrases from product description"""
        if not text:
            return []
        
        # Split text into sentences
        sentences = re.split(r'[.!?]\s*', text)
        
        # Extract phrases that might be important (containing keywords or bullet points)
        key_phrases = []
        keywords = ["качество", "особенности", "преимущества", "характеристики", "функции", "технология"]
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            # Check if sentence starts with bullet point or contains keywords
            if sentence.startswith("•") or sentence.startswith("-") or any(keyword in sentence.lower() for keyword in keywords):
                key_phrases.append(sentence)
        
        # If no key phrases found, take first 3 sentences as key phrases
        if not key_phrases and len(sentences) > 0:
            key_phrases = [s.strip() for s in sentences[:3] if s.strip()]
        
        return key_phrases
    
    def _analyze_images(self, image_urls, product_dir):
        """Download and analyze product images"""
        image_analysis = {
            "images_with_text": 0,
            "images_without_text": 0,
            "color_palettes": [],
            "text_on_images": [],
            "design_elements": []
        }
        
        for i, url in enumerate(image_urls):
            try:
                # Download image
                image_filename = f"image_{i+1}.jpg"
                image_path = os.path.join(product_dir, image_filename)
                
                urllib.request.urlretrieve(url, image_path)
                
                # Analyze image
                has_text, extracted_text, colors, design_elements = self._analyze_single_image(image_path)
                
                # Update image analysis
                if has_text:
                    image_analysis["images_with_text"] += 1
                    if extracted_text:
                        image_analysis["text_on_images"].append(extracted_text)
                else:
                    image_analysis["images_without_text"] += 1
                
                image_analysis["color_palettes"].append(colors)
                image_analysis["design_elements"].append(design_elements)
                
            except Exception as e:
                print(f"Error analyzing image {url}: {e}")
        
        return image_analysis
    
    def _analyze_single_image(self, image_path):
        """Analyze a single image for text, colors, and design elements"""
        # Initialize results
        has_text = False
        extracted_text = ""
        colors = []
        design_elements = {
            "has_icons": False,
            "font_style": "unknown",
            "composition": "unknown"
        }
        
        try:
            # Load image with OpenCV for text detection
            img = cv2.imread(image_path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Simple text detection using thresholding and contour analysis
            _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Check if contours might be text
            text_like_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                if 0.1 < aspect_ratio < 10 and 10 < w < img.shape[1] // 2 and 10 < h < img.shape[0] // 2:
                    text_like_contours += 1
            
            has_text = text_like_contours > 5
            
            # Extract text using pytesseract if text is detected and Tesseract is available
            if has_text and tesseract_available:
                try:
                    extracted_text = pytesseract.image_to_string(Image.open(image_path), lang='rus')
                    extracted_text = extracted_text.strip()
                except Exception as e:
                    print(f"Error extracting text from image: {e}")
                    extracted_text = ""
            else:
                # If Tesseract is not available, set extracted_text to empty string
                extracted_text = ""
            
            # Extract color palette using ColorThief
            try:
                color_thief = ColorThief(image_path)
                palette = color_thief.get_palette(color_count=3)
                colors = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in palette]
            except:
                colors = []
            
            # Basic design element detection
            # Check for icons (small, isolated contours)
            small_isolated_contours = 0
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                if w < img.shape[1] // 10 and h < img.shape[0] // 10 and w > 10 and h > 10:
                    small_isolated_contours += 1
            
            design_elements["has_icons"] = small_isolated_contours > 3
            
            # Simple composition analysis
            # Check if image has a central focus
            h, w = img.shape[:2]
            center_region = img[h//4:3*h//4, w//4:3*w//4]
            center_std = np.std(center_region)
            overall_std = np.std(img)
            
            if center_std > overall_std * 1.2:
                design_elements["composition"] = "central focus"
            elif np.std(img[:, :w//2]) > np.std(img[:, w//2:]) * 1.2:
                design_elements["composition"] = "left-weighted"
            elif np.std(img[:, w//2:]) > np.std(img[:, :w//2]) * 1.2:
                design_elements["composition"] = "right-weighted"
            else:
                design_elements["composition"] = "balanced"
            
            # Font style estimation (very basic)
            if has_text:
                # Count edges in the image as a very rough proxy for font complexity
                edges = cv2.Canny(gray, 100, 200)
                edge_density = np.sum(edges) / (edges.shape[0] * edges.shape[1])
                
                if edge_density > 0.1:
                    design_elements["font_style"] = "decorative/complex"
                else:
                    design_elements["font_style"] = "simple/sans-serif"
            
        except Exception as e:
            print(f"Error in image analysis: {e}")
        
        return has_text, extracted_text, colors, design_elements
    
    def run(self):
        """Run the parser to collect product data"""
        try:
            print("Starting parser run...")
            # Search for products in the category
            try:
                print(f"Searching for products in category: {self.category}")
                product_links = self.search_category()
                print(f"Found {len(product_links)} product links")
            except Exception as e:
                print(f"Error during category search: {str(e)}")
                import traceback
                traceback.print_exc()
                return []
            
            # Parse each product
            count = 0
            for link in product_links:
                if count >= self.max_items:
                    break
                
                try:    
                    product_data = self.parse_product(link)
                    if product_data:  # Only add if product meets criteria
                        self.results.append(product_data)
                        count += 1
                        print(f"Collected {count}/{self.max_items} products")
                except Exception as e:
                    print(f"Error parsing product {link}: {str(e)}")
                    continue
                
                # Add random delay between requests
                time.sleep(random.uniform(1, 3))
            
            # Save results to JSON and CSV
            if self.results:
                try:
                    self._save_results()
                except Exception as e:
                    print(f"Error saving results: {str(e)}")
            
            print(f"Parsing completed. Collected {len(self.results)} products.")
            return self.results
            
        except Exception as e:
            print(f"Error running parser: {e}")
            import traceback
            traceback.print_exc()
            return []
        finally:
            # Close the browser
            try:
                self.driver.quit()
                print("Browser closed successfully")
            except Exception as e:
                print(f"Error closing browser: {str(e)}")
    
    def _save_results(self):
        """Save results to JSON and CSV files"""
        # Save to JSON
        json_path = os.path.join(self.category_dir, f"{self.category.replace(' ', '_')}_results.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(self.results, f, ensure_ascii=False, indent=4)
        
        # Save to CSV (flattening the nested structure)
        csv_path = os.path.join(self.category_dir, f"{self.category.replace(' ', '_')}_results.csv")
        
        # Prepare data for CSV
        csv_data = []
        for product in self.results:
            row = {
                "name": product["name"],
                "category": product["category"],
                "brand": product["brand"],
                "price": product["price"],
                "rating": product["rating"],
                "reviews": product["reviews"],
                "image_count": product["image_count"],
                "images_with_text": product["image_analysis"]["images_with_text"],
                "images_without_text": product["image_analysis"]["images_without_text"],
                "product_url": product["product_url"]
            }
            
            # Add first 3 color palettes if available
            for i, colors in enumerate(product["image_analysis"]["color_palettes"][:3]):
                for j, color in enumerate(colors[:3]):
                    row[f"image_{i+1}_color_{j+1}"] = color
            
            # Add first 3 key phrases if available
            for i, phrase in enumerate(product["key_phrases"][:3]):
                row[f"key_phrase_{i+1}"] = phrase
            
            csv_data.append(row)
        
        # Convert to DataFrame and save
        df = pd.DataFrame(csv_data)
        df.to_csv(csv_path, index=False, encoding='utf-8')
        
        print(f"Results saved to {json_path} and {csv_path}")

# Example usage
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Parse Wildberries products')
    parser.add_argument('category', type=str, help='Category to search for')
    parser.add_argument('--min-rating', type=float, default=4.5, help='Minimum product rating')
    parser.add_argument('--min-reviews', type=int, default=50, help='Minimum number of reviews')
    parser.add_argument('--max-items', type=int, default=100, help='Maximum number of items to collect')
    
    args = parser.parse_args()
    
    wb_parser = WildberriesParser(
        category=args.category,
        min_rating=args.min_rating,
        min_reviews=args.min_reviews,
        max_items=args.max_items
    )
    
    wb_parser.run()