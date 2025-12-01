#!/usr/bin/env python3
"""
Convert HTML to PDF using reportlab or other available libraries
"""
import sys
import os

def convert_with_weasyprint(html_file, pdf_file):
    """Try conversion with WeasyPrint"""
    try:
        from weasyprint import HTML
        HTML(html_file).write_pdf(pdf_file)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"WeasyPrint error: {e}")
        return False

def convert_with_pdfkit(html_file, pdf_file):
    """Try conversion with pdfkit/wkhtmltopdf"""
    try:
        import pdfkit
        pdfkit.from_file(html_file, pdf_file)
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"pdfkit error: {e}")
        return False

def convert_with_playwright(html_file, pdf_file):
    """Try conversion with playwright"""
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            page.goto(f'file://{os.path.abspath(html_file)}')
            page.pdf(path=pdf_file, format='A4', margin={
                'top': '1in',
                'right': '1in',
                'bottom': '1in',
                'left': '1in'
            })
            browser.close()
        return True
    except ImportError:
        return False
    except Exception as e:
        print(f"Playwright error: {e}")
        return False

def main():
    if len(sys.argv) != 3:
        print("Usage: python convert_html_to_pdf.py <input.html> <output.pdf>")
        sys.exit(1)
    
    html_file = sys.argv[1]
    pdf_file = sys.argv[2]
    
    if not os.path.exists(html_file):
        print(f"Error: {html_file} not found")
        sys.exit(1)
    
    # Try different conversion methods
    converters = [
        ("WeasyPrint", convert_with_weasyprint),
        ("pdfkit/wkhtmltopdf", convert_with_pdfkit),
        ("Playwright", convert_with_playwright),
    ]
    
    for name, converter in converters:
        print(f"Trying {name}...")
        if converter(html_file, pdf_file):
            print(f"✓ Successfully converted using {name}")
            print(f"Output: {pdf_file}")
            return
    
    print("\n✗ Failed to convert. Please install one of the following:")
    print("  - WeasyPrint: pip install weasyprint")
    print("  - pdfkit: pip install pdfkit (requires wkhtmltopdf)")
    print("  - Playwright: pip install playwright && playwright install chromium")
    sys.exit(1)

if __name__ == "__main__":
    main()
