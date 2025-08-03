import subprocess
import sys
import os

def install_requirements():
    """Install required Python packages"""
    print("Installing required Python packages...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("Python packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error installing Python packages: {e}")
        return False

def check_tesseract():
    """Check if Tesseract OCR is installed"""
    print("Checking Tesseract OCR installation...")
    try:
        import pytesseract
        pytesseract.get_tesseract_version()
        print("Tesseract OCR is installed!")
        return True
    except Exception as e:
        print(f"Tesseract OCR is not properly installed or configured: {e}")
        print("\nPlease install Tesseract OCR:")
        print("- Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki")
        print("- macOS: brew install tesseract")
        print("- Linux: sudo apt install tesseract-ocr")
        print("\nAfter installation, make sure to add Tesseract to your PATH.")
        return False

def check_russian_language_data():
    """Check if Russian language data for Tesseract is installed"""
    print("Checking Russian language data for Tesseract...")
    try:
        import pytesseract
        from PIL import Image
        # Create a small test image
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (100, 30), color=(255, 255, 255))
        d = ImageDraw.Draw(img)
        d.text((10, 10), "тест", fill=(0, 0, 0))
        test_img_path = "test_russian.png"
        img.save(test_img_path)
        
        # Try to recognize Russian text
        try:
            text = pytesseract.image_to_string(Image.open(test_img_path), lang='rus')
            os.remove(test_img_path)  # Clean up
            print("Russian language data for Tesseract is installed!")
            return True
        except pytesseract.TesseractError:
            os.remove(test_img_path)  # Clean up
            print("Russian language data for Tesseract is not installed!")
            print("\nPlease install Russian language data for Tesseract:")
            print("- Windows: Download rus.traineddata and place it in the tessdata directory")
            print("- macOS: brew install tesseract-lang")
            print("- Linux: sudo apt install tesseract-ocr-rus")
            return False
    except Exception as e:
        print(f"Error checking Russian language data: {e}")
        return False

def check_chrome_browser():
    """Check if Chrome browser is installed"""
    print("Checking Chrome browser installation...")
    try:
        from selenium import webdriver
        from selenium.webdriver.chrome.service import Service
        from webdriver_manager.chrome import ChromeDriverManager
        import os
        
        # Check common Chrome installation paths on Windows
        chrome_paths = [
            os.path.join(os.environ.get('PROGRAMFILES', 'C:\\Program Files'), 'Google\\Chrome\\Application\\chrome.exe'),
            os.path.join(os.environ.get('PROGRAMFILES(X86)', 'C:\\Program Files (x86)'), 'Google\\Chrome\\Application\\chrome.exe'),
            os.path.join(os.environ.get('LOCALAPPDATA', ''), 'Google\\Chrome\\Application\\chrome.exe')
        ]
        
        chrome_found = False
        for path in chrome_paths:
            if os.path.exists(path):
                chrome_found = True
                print(f"Found Chrome at: {path}")
                break
        
        if chrome_found:
            print("Chrome browser is installed!")
            return True
        
        # If Chrome wasn't found in common paths, try to create a Chrome driver as a fallback
        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        try:
            driver = webdriver.Chrome(
                service=Service(ChromeDriverManager().install()),
                options=options
            )
            driver.quit()
            print("Chrome browser is installed!")
            return True
        except Exception as e:
            if "cannot find Chrome binary" in str(e):
                print("Chrome browser is not installed!")
                print("\nPlease install Chrome browser:")
                print("- Download from https://www.google.com/chrome/")
                return False
            else:
                print(f"Error checking Chrome browser: {e}")
                return False
    except Exception as e:
        print(f"Error checking Chrome browser: {e}")
        return False

def main():
    print("Setting up the parser...\n")
    
    # Install Python packages
    if not install_requirements():
        print("\nFailed to install required Python packages. Setup incomplete.")
        return
    
    # Check Chrome browser
    chrome_installed = check_chrome_browser()
    
    # Check Tesseract OCR
    tesseract_installed = check_tesseract()
    
    # Check Russian language data
    if tesseract_installed:
        russian_data_installed = check_russian_language_data()
    else:
        russian_data_installed = False
    
    # Summary
    print("\nSetup Summary:")
    print(f"- Python packages: {'Installed' if True else 'Not installed'}")
    print(f"- Chrome browser: {'Installed' if chrome_installed else 'Not installed'}")
    print(f"- Tesseract OCR: {'Installed' if tesseract_installed else 'Not installed'}")
    print(f"- Russian language data: {'Installed' if russian_data_installed else 'Not installed'}")
    
    if chrome_installed and tesseract_installed and russian_data_installed:
        print("\nSetup completed successfully! You can now run the parser:")
        print("python main.py [marketplace] [category]")
    else:
        print("\nSetup incomplete. Please resolve the issues above before running the parser.")
        
        if not chrome_installed:
            print("\nNote: Chrome browser is required for the parser to work.")
            print("Download Chrome from: https://www.google.com/chrome/")
            
        if not tesseract_installed:
            print("\nNote: Tesseract OCR is required for text extraction from images.")
            print("The parser will work without it, but text extraction features will be disabled.")
            print("If you want to use text extraction features, please install Tesseract OCR.")
            
        if tesseract_installed and not russian_data_installed:
            print("\nNote: Russian language data for Tesseract OCR is required for Russian text extraction.")
            print("The parser will work without it, but Russian text extraction will be limited.")
            print("If you want to extract Russian text from images, please install Russian language data for Tesseract OCR.")

if __name__ == "__main__":
    main()