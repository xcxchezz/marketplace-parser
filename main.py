import argparse
import os
import sys

def check_dependencies():
    """Check if all required dependencies are installed"""
    try:
        import requests
        import bs4
        import PIL
        import selenium
        import pandas
        import cv2
        import pytesseract
        import colorthief
        return True
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Please install all required dependencies using: pip install -r requirements.txt")
        return False

def main():
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Parse products from Wildberries or Ozon')
    parser.add_argument('marketplace', type=str, choices=['wildberries', 'ozon'], 
                        help='Marketplace to parse (wildberries or ozon)')
    parser.add_argument('category', type=str, help='Category to search for (e.g. "сковороды", "женская одежда")')
    parser.add_argument('--min-rating', type=float, default=4.5, help='Minimum product rating (default: 4.5)')
    parser.add_argument('--min-reviews', type=int, default=50, help='Minimum number of reviews (default: 50)')
    parser.add_argument('--max-items', type=int, default=100, help='Maximum number of items to collect (default: 100)')
    
    args = parser.parse_args()
    
    # Run the appropriate parser
    if args.marketplace.lower() == 'wildberries':
        from wildberries_parser import WildberriesParser
        parser = WildberriesParser(
            category=args.category,
            min_rating=args.min_rating,
            min_reviews=args.min_reviews,
            max_items=args.max_items
        )
    else:  # ozon
        from ozon_parser import OzonParser
        parser = OzonParser(
            category=args.category,
            min_rating=args.min_rating,
            min_reviews=args.min_reviews,
            max_items=args.max_items
        )
    
    # Run the parser
    print(f"Starting to parse {args.max_items} products from {args.marketplace} in category '{args.category}'")
    print(f"Filtering products with rating >= {args.min_rating} and reviews >= {args.min_reviews}")
    
    print("\n" + "=" * 80)
    print("IMPORTANT INFORMATION ABOUT CAPTCHA CHALLENGES")
    print("=" * 80)
    print("This parser may encounter CAPTCHA challenges from the website.")
    print("If a CAPTCHA appears, the browser window will open and you'll need to:")
    print("1. Manually solve the CAPTCHA in the browser window")
    print("2. Wait for the script to continue automatically after solving")
    print("The script will wait up to 5 minutes for you to solve the CAPTCHA.")
    print("=" * 80 + "\n")
    
    results = parser.run()
    
    # Print summary
    if results:
        category_dir = os.path.join(os.getcwd(), args.category.replace(' ', '_'))
        print(f"\nParsing completed successfully!")
        print(f"Collected {len(results)} products")
        print(f"Results saved to:")
        print(f"  - JSON: {os.path.join(category_dir, f'{args.category.replace(' ', '_')}_results.json')}")
        print(f"  - CSV: {os.path.join(category_dir, f'{args.category.replace(' ', '_')}_results.csv')}")
        print(f"  - Images: {category_dir}")
    else:
        print("\nNo products were collected. This could be due to:")
        print("1. The CAPTCHA challenge was not solved within the time limit")
        print("2. The search criteria did not match any products")
        print("3. The minimum rating or reviews criteria filtered out all products")
        print("4. The website structure may have changed")
        print("\nPlease check the error messages above for more details.")

if __name__ == "__main__":
    main()